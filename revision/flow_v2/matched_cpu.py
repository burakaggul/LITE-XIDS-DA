"""Isolate INT8 latency/size from TorchScript-versus-eager execution differences."""
import argparse
import io
import json
import time
import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from common import HERE,digest,log,save
from cost_models import RowWiseQuantized,neural_bytes
from experiment import source_data


def run(track='common'):
    if not (HERE/'results'/('result_audit_'+track+'.json')).exists():
        raise RuntimeError('Verified models/results for the requested track are required.')
    torch.set_num_threads(1);config=json.loads((HERE/'config.json').read_text());rows=[]
    for source in config['primary_pair']:
        for seed in config['seeds']:
            directory=HERE/'models'/track/source/str(seed)
            info=json.loads((directory/'frozen.json').read_text())
            chosen=next(c for c in info['selections'] if c['family']=='student_distilled' and c['arm']=='cost_sensitive' and c['fpr_limit']==.05)
            cost=chosen['benign_cost'];transform=joblib.load(directory/'transform.joblib')
            _,_,calibration,_,audit=source_data(track,source,seed)
            if audit!=info['source_audit']:
                raise AssertionError('CPU measurement source cohort differs from its freeze.')
            row=calibration.iloc[:1]
            x=torch.tensor(transform.apply(row[transform.columns]),dtype=torch.float32)
            for family in ['student_distilled','student_distilled_int8']:
                file=directory/(family+'_cost'+str(cost)+'.joblib')
                if digest(file)!=info['artifact_sha256'][file.name]:
                    raise AssertionError('Matched CPU artifact differs from frozen training artifact.')
                model=joblib.load(file);compiled=torch.jit.script(RowWiseQuantized(model).eval())
                with torch.no_grad():
                    for _ in range(100):
                        torch.softmax(compiled(x),dim=1).numpy()
                    elapsed=[]
                    for _ in range(500):
                        start=time.perf_counter_ns();torch.softmax(compiled(x),dim=1).numpy()
                        elapsed.append((time.perf_counter_ns()-start)/1e6)
                    if family=='student_distilled':
                        np.testing.assert_allclose(compiled(x).numpy(),model(x).numpy(),rtol=1e-5,atol=1e-5)
                buffer=io.BytesIO();torch.jit.save(compiled,buffer)
                rows.append(dict(track=track,source=source,seed=seed,family=family,benign_cost=cost,
                    backend='TorchScript row-wise, CPU1, batch1',model_ms_median=float(np.median(elapsed)),
                    model_ms_p95=float(np.quantile(elapsed,.95)),state_dict_bytes=neural_bytes(model),
                    exported_script_bytes=len(buffer.getvalue()),process_rss_bytes=psutil.Process().memory_info().rss,
                    rss_scope='whole framework process, not isolated model-only memory',
                    scope='Completed-flow model inference only; desktop i9-14900HX, not an embedded device'))
            log(f'MATCHED CPU PASS {track}/{source}/seed{seed}: same trained student cost={cost}, same scripted backend and input')
    if len(rows)!=20:
        raise AssertionError('All 20 matched-backend profiles are required.')
    pd.DataFrame(rows).to_csv(HERE/'results'/(track+'_matched_backend_cpu.csv'),index=False)
    save(HERE/'results'/(track+'_matched_backend_cpu_audit.json'),dict(track=track,rows=len(rows),same_backend=True,same_cost_and_float_weights=True,
        warmups=100,timed_calls=500,batch_size=1,cpu_threads=1,
        caution='Primary pipeline timing includes its chosen implementation backend. Only this paired table isolates float versus INT8 under a matched TorchScript row-wise backend. Neither is a real-device measurement.',
        code_sha256=digest(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--track',choices=['common','legacy'],default='common')
    run(parser.parse_args().track)
