from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from pytorch_tcn import TCN


class MLP(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(n, 64), nn.ReLU(), nn.Dropout(.1),
            nn.Linear(64, 32), nn.ReLU())
        self.head = nn.Linear(32, 2)

    def forward(self, x):
        return self.head(self.features(x))


class TabularTCN(nn.Module):
    """TCN over a declared feature order, NOT over a fabricated traffic chronology."""
    def __init__(self, n):
        super().__init__()
        self.tcn = TCN(num_inputs=1, num_channels=[16, 16], kernel_size=3,
            dilations=[1, 2], dropout=.1, causal=True, use_norm='weight_norm')
        self.head = nn.Linear(16, 2)

    def forward(self, x):
        return self.head(self.tcn(x.unsqueeze(1)).mean(-1))


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.weight*grad, None


def train_neural(kind, x, y, xv, yv, seed, epochs=20, target=None, teacher_proba=None,
    alpha=.25, temperature=3.):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    model = TabularTCN(x.shape[1]) if kind == 'tcn' else MLP(x.shape[1])
    domain = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2)) if kind == 'dann' else None
    params = list(model.parameters())+(list(domain.parameters()) if domain is not None else [])
    optimizer = torch.optim.AdamW(params, lr=.001, weight_decay=.0001)
    x, xv = np.asarray(x, dtype=np.float32), np.asarray(xv, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32) if target is not None else None
    tensors = [torch.tensor(x), torch.tensor(y, dtype=torch.long)]
    if teacher_proba is not None:
        logits = np.log(np.clip(teacher_proba, 1e-7, 1.))
        tensors.append(torch.tensor(logits, dtype=torch.float32))
    loader = DataLoader(TensorDataset(*tensors), batch_size=256, shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    counts = np.bincount(y, minlength=2)
    weight = torch.tensor(len(y)/(2*np.maximum(counts, 1)), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight)
    best, best_loss, patience, used = None, np.inf, 0, 0
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            xb, yb = batch[:2]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if teacher_proba is not None:
                soft = torch.softmax(batch[2]/temperature, dim=1)
                kd = nn.functional.kl_div(torch.log_softmax(logits/temperature, dim=1), soft,
                    reduction='batchmean')*temperature**2
                loss = alpha*loss+(1-alpha)*kd
            if domain is not None:
                idx = rng.integers(0, len(target), len(xb))
                xt = torch.tensor(target[idx])
                f = torch.cat([model.features(xb), model.features(xt)])
                domain_y = torch.cat([torch.zeros(len(xb)), torch.ones(len(xt))]).long()
                reverse_weight = 2/(1+np.exp(-10*(epoch+1)/epochs))-1
                loss = loss+.1*nn.functional.cross_entropy(domain(Reverse.apply(f, reverse_weight)), domain_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = float(criterion(model(torch.tensor(xv)), torch.tensor(yv, dtype=torch.long)))
        used = epoch+1
        if validation < best_loss-1e-5:
            best, best_loss, patience = copy.deepcopy(model.state_dict()), validation, 0
        else:
            patience += 1
        if patience >= 4:
            break
    model.load_state_dict(best)
    model.eval()
    model.trained_epochs = used
    return model


def scores(model, x):
    with torch.no_grad():
        logits = model(torch.tensor(np.asarray(x), dtype=torch.float32)).numpy()
    return (logits[:, 1]-logits[:, 0]).astype(float)
