"""Run on the actual device. Results are never inferred from desktop measurements."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import psutil
import torch
from torch import nn
from torch.nn.utils import prune
from xgboost import XGBClassifier


class Student(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.features=nn.Sequential(nn.Linear(n,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,32),nn.ReLU())
        self.head=nn.Linear(32,2)
    def forward(self,x):
        return self.head(self.features(x))


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except (FileNotFoundError,PermissionError):
        return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--artifacts',type=Path,default=Path(__file__).resolve().parent/'artifacts')
    p.add_argument('--output',type=Path,default=Path('device_results.json'))
    p.add_argument('--device-label',required=True,help='For example Raspberry Pi 4 4GB; do not label a desktop as an edge device')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--warmup',type=int,default=200)
    p.add_argument('--repeats',type=int,default=1000)
    p.add_argument('--trials',type=int,default=5)
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(11)
    state=torch.load(args.artifacts/'student_for_edge.pt',map_location='cpu',weights_only=True)
    model=Student(state['input_features'])
    model.load_state_dict(state['state_dict'])
    model.eval()
    data=np.load(args.artifacts/'edge_input.npz')
    x=np.asarray(data['x'],dtype=np.float32)
    models={'student_float':model}
    pruned=copy.deepcopy(model)
    for layer in pruned.modules():
        if isinstance(layer,nn.Linear):
            prune.l1_unstructured(layer,name='weight',amount=.2)
            prune.remove(layer,'weight')
    models['student_pruned']=pruned
    engines=list(torch.backends.quantized.supported_engines)
    if 'qnnpack' in engines and platform.machine().lower() in ['aarch64','arm64','armv7l']:
        torch.backends.quantized.engine='qnnpack'
    elif 'x86' in engines:
        torch.backends.quantized.engine='x86'
    models['student_pruned_quantized']=torch.ao.quantization.quantize_dynamic(copy.deepcopy(pruned),{nn.Linear},dtype=torch.qint8)
    teacher=XGBClassifier(n_jobs=args.threads)
    teacher.load_model(args.artifacts/'teacher_for_edge.ubj')
    models['teacher_xgb']=teacher
    output=dict(device_label=args.device_label,platform=platform.platform(),machine=platform.machine(),
        cpuinfo=read_optional('/proc/cpuinfo'),device_tree_model=read_optional('/sys/firmware/devicetree/base/model'),
        governor=read_optional('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'),
        physical_cores=psutil.cpu_count(logical=False),logical_cores=psutil.cpu_count(),
        ram_bytes=psutil.virtual_memory().total,python=sys.version,torch=torch.__version__,
        quantized_engine=torch.backends.quantized.engine,threads=args.threads,
        scope='batch one model scoring including input tensor construction; no packet capture or feature extraction',
        artifacts={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in args.artifacts.iterdir() if f.is_file()},
        measurements=[])
    if Path('/usr/sbin/nvpmodel').exists():
        output['nvpmodel_query']=subprocess.run(['/usr/sbin/nvpmodel','-q'],capture_output=True,text=True).stdout
    for name,net in models.items():
        def predict(sample):
            if isinstance(net,nn.Module):
                with torch.inference_mode():
                    logits=net(torch.tensor(sample))
                    return (logits[:,1]-logits[:,0]).numpy()
            return net.predict_proba(sample)[:,1]
        checksum=hashlib.sha256(predict(x).tobytes()).hexdigest()
        for trial in range(args.trials):
            for _ in range(args.warmup):
                predict(x[:1])
            temperature_before=read_optional('/sys/class/thermal/thermal_zone0/temp')
            rss_before=psutil.Process().memory_info().rss
            times=[]
            for i in range(args.repeats):
                sample=x[i%len(x):i%len(x)+1]
                start=time.perf_counter_ns()
                predict(sample)
                times.append((time.perf_counter_ns()-start)/1e6)
            output['measurements'].append(dict(model=name,trial=trial,batch_size=1,warmup=args.warmup,
                repeats=args.repeats,mean_ms=float(np.mean(times)),p50_ms=float(np.quantile(times,.5)),
                p95_ms=float(np.quantile(times,.95)),p99_ms=float(np.quantile(times,.99)),
                throughput_per_second=1000/float(np.mean(times)),rss_before=rss_before,
                rss_after=psutil.Process().memory_info().rss,temperature_before=temperature_before,
                temperature_after=read_optional('/sys/class/thermal/thermal_zone0/temp'),prediction_checksum=checksum))
    args.output.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(args.output.resolve())


if __name__=='__main__':
    main()
