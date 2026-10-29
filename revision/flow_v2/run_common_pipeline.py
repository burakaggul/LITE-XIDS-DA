"""Resume-safe gates: no training on incomplete captures or unreliable inferred labels."""
import argparse
import json
import os
import subprocess
import sys
import time
from common import HERE,log


def step(name,script,*arguments):
    command=[sys.executable,'-u',str(HERE/'run_step.py'),'--label',name,'--',sys.executable,'-u',str(HERE/script),*arguments]
    env=os.environ.copy()
    env['PYTHONWARNINGS']='ignore:y_pred contains classes not in y_true:UserWarning,ignore:A single label was found:UserWarning'
    result=subprocess.run(command,env=env)
    if result.returncode:
        raise RuntimeError(name+' failed; downstream stages intentionally not executed.')


def main(official_events=False):
    while True:
        source=HERE/'results/source_integrity.json';extract=HERE/'results/extraction.json'
        s=json.loads(source.read_text()) if source.exists() else {'complete':False,'files':[]}
        e=json.loads(extract.read_text()) if extract.exists() else {'complete':False,'captures':[]}
        n=len(list((HERE/'label_index').glob('*.json')))
        log(f'COMMON PIPELINE GATE: raw objects {len(s["files"])}/67; full PCAPs {len(e["captures"])}/44; original label indexes {n}/23')
        if s['complete'] and e['complete'] and n==23:
            break
        time.sleep(30)
    step('common_label_preparation','label_flows.py','--stage','prepare',*(['--official-events'] if official_events else []))
    step('common_prepared_audit','audit_sources.py','--stage','prepared')
    step('common_data_freeze','freeze_common_data.py','--stage','freeze')
    step('common_cost_training','experiment.py','--track','common','--stage','train')
    step('common_baseline_training','common_baselines.py','--stage','train')
    step('common_final_evaluation','experiment.py','--track','common','--stage','evaluate')
    step('common_result_audit','audit_results.py','--track','common')
    step('common_baseline_evaluation','common_baselines.py','--stage','evaluate')
    step('common_matched_ablation','ablate_common.py')
    step('common_external_xai','explain_common.py')
    step('common_matched_cpu','matched_cpu.py')
    step('common_data_verify','freeze_common_data.py','--stage','verify')
    step('paired_report','summarize.py')
    log('COMMON PIPELINE COMPLETE: data, source-only fitting, frozen final evaluation and independent checks finished')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--official-events',action='store_true')
    main(parser.parse_args().official_events)
