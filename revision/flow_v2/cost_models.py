"""Training-only transforms and matched supervised/distilled cost-sensitive students."""
import copy
import io
import sys
import weakref
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from neural_revision import MLP

_ROW_WISE=weakref.WeakKeyDictionary()


class RowWiseQuantized(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.features=copy.deepcopy(model.features)
        self.head=copy.deepcopy(model.head)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        output=torch.empty((x.size(0),2),dtype=x.dtype,device=x.device)
        for i in range(x.size(0)):
            output[i:i+1]=self.head(self.features(x[i:i+1]))
        return output


class SourceTransform:
    def fit(self,frame):
        self.columns=list(frame.columns)
        forbidden={'label','attack','src_ip','dst_ip','src_port','dst_port','first_ms','last_ms','capture_id','role'}
        if forbidden.intersection(self.columns):
            raise ValueError('Identifiers and labels must not enter model features.')
        values=frame.replace([np.inf,-np.inf],np.nan)
        self.imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(values)
        self.scaler=StandardScaler().fit(self._log(self.imputer.transform(values)))
        return self

    @staticmethod
    def _log(values):
        return np.sign(values)*np.log1p(np.abs(values))

    def apply(self,frame):
        values=frame[self.columns].replace([np.inf,-np.inf],np.nan)
        return self.scaler.transform(self._log(self.imputer.transform(values))).astype('float32')


def class_costs(y,benign_cost):
    y=np.asarray(y,dtype=int)
    if set(np.unique(y))!={0,1} or benign_cost<1:
        raise ValueError('Both classes and a positive predeclared benign cost are required.')
    counts=np.bincount(y,minlength=2)
    weights=len(y)/(2*counts.astype(float))
    weights[0]*=benign_cost
    return weights/np.mean(weights[y])


def teacher(x,y,seed,cost):
    model=XGBClassifier(n_estimators=300,max_depth=6,learning_rate=.05,subsample=.9,
        colsample_bytree=.9,n_jobs=4,eval_metric='logloss',random_state=seed)
    weights=class_costs(y,cost)
    model.fit(x,y,sample_weight=weights[y])
    return model


def student(x,y,xv,yv,seed,cost,config,teacher_probabilities=None):
    torch.manual_seed(seed);torch.set_num_threads(4)
    model=MLP(x.shape[1])
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=.0001)
    weights=torch.tensor(class_costs(y,cost),dtype=torch.float32)
    arrays=[torch.as_tensor(x,dtype=torch.float32),torch.as_tensor(y,dtype=torch.long)]
    if teacher_probabilities is not None:
        arrays.append(torch.tensor(np.log(np.clip(teacher_probabilities,1e-7,1.)),dtype=torch.float32))
    loader=DataLoader(TensorDataset(*arrays),batch_size=config['batch_size'],shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    xv=torch.as_tensor(xv,dtype=torch.float32);yv=torch.as_tensor(yv,dtype=torch.long)
    best=None;best_loss=np.inf;patience=0;history=[]
    for epoch in range(config['epochs']):
        model.train();training=[]
        for batch in loader:
            xb,yb=batch[:2];optimizer.zero_grad();logits=model(xb)
            row_weights=weights[yb]
            loss_rows=nn.functional.cross_entropy(logits,yb,reduction='none')
            if teacher_probabilities is not None:
                temperature=config['temperature']
                soft=torch.softmax(batch[2]/temperature,dim=1)
                kd=nn.functional.kl_div(torch.log_softmax(logits/temperature,dim=1),soft,
                    reduction='none').sum(1)*temperature**2
                loss_rows=config['alpha_hard']*loss_rows+(1-config['alpha_hard'])*kd
            loss=(row_weights*loss_rows).mean()
            loss.backward();optimizer.step();training.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation=float((weights[yv]*nn.functional.cross_entropy(model(xv),yv,reduction='none')).mean())
        history.append(dict(epoch=epoch+1,training_loss=float(np.mean(training)),source_validation_loss=validation))
        if validation<best_loss-1e-5:
            best=copy.deepcopy(model.state_dict());best_loss=validation;patience=0
        else:
            patience+=1
        if patience>=config['patience']:
            break
    model.load_state_dict(best);model.eval();model.training_history=history
    model.parameter_count=sum(p.numel() for p in model.parameters())
    return model


def probability(model,x):
    if isinstance(model,nn.Module):
        inference=model
        if any(isinstance(m,torch.ao.nn.quantized.dynamic.Linear) for m in model.modules()):
            if model not in _ROW_WISE:
                _ROW_WISE[model]=torch.jit.script(RowWiseQuantized(model).eval())
            inference=_ROW_WISE[model]
        with torch.no_grad():
            return torch.softmax(inference(torch.as_tensor(x,dtype=torch.float32)),dim=1)[:,1].numpy().astype(float)
    return model.predict_proba(x)[:,1].astype(float)


def quantize(model):
    result=torch.ao.quantization.quantize_dynamic(copy.deepcopy(model).eval(),{nn.Linear},dtype=torch.qint8)
    return result


def neural_bytes(model):
    stream=io.BytesIO();torch.save(model.state_dict(),stream)
    return len(stream.getvalue())
