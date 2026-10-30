"""Finish evidence reconstruction, then run a separately registered conditional study."""
import json
import os
import subprocess
import sys
import time
from common import HERE,digest,log


def step(name,script,*args,allow_coverage_failure=False):
    command=[sys.executable,'-u',str(HERE/'run_step.py'),'--label',name,'--',sys.executable,'-u',str(HERE/script),*args]
    result=subprocess.run(command,env=os.environ.copy())
    if result.returncode==3 and allow_coverage_failure:
        manifest=json.loads((HERE/'manifests/common_dataset.json').read_text())
        if manifest['quality_gate_passed'] or manifest.get('label_schema_version')!='v3' or not manifest.get('packet_witness_manifest_sha256'):
            raise AssertionError('Unexpected data preparation failure.')
        log('FULL-TRAFFIC QUALITY FAILURE RETAINED: proceeding only with a distinct conditional labeled-cohort estimand and explicit missing-label bounds')
    elif result.returncode:
        raise RuntimeError(name+' failed; no downstream stage executed.')


def main():
    while True:
        path=HERE/'results/all_packet_witnesses_v3.json'
        if path.exists() and json.loads(path.read_text()).get('complete'):
            break
        log('WAITING FOR COMPLETE PACKET CERTIFICATES: no raw model fitting started')
        time.sleep(30)
    step('packet_labeled_data_v3','label_flows.py','--stage','prepare','--official-events','--label-schema','v3','--packet-witnesses',allow_coverage_failure=True)
    step('packet_labeled_physical_audit_v3','audit_sources.py','--stage','prepared')
    step('labeled_raw_registration','labeled_raw_experiment.py','--stage','register')
    step('labeled_raw_training','labeled_raw_experiment.py','--stage','train')
    step('labeled_raw_evaluation','labeled_raw_experiment.py','--stage','evaluate')
    step('labeled_raw_independent_audit','audit_labeled_raw.py')
    step('labeled_raw_explanations_cpu','post_labeled_raw.py')
    log('RAW CONDITIONAL STUDY FINISHED: independent checks passed; no full-traffic or actual-device success claim')


if __name__=='__main__':
    main()
