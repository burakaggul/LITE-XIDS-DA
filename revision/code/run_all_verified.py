"""Execute every local experiment with a fixed source snapshot and checked exits."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]


def snapshot():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT/'revision'/'code').glob('*.py')) if p.name in
        ['data_revision.py','native_revision.py','protocol.py','neural_revision.py','run_revision.py',
         'run_sensitivity.py','run_mapping_controls.py','run_external_xai.py','postcheck_controls.py',
         'test_protocol.py','test_transform.py','run_all_verified.py']}


def main():
    (ROOT/'revision'/'results').mkdir(exist_ok=True)
    before=snapshot()
    stages=[['-m','unittest','discover','-s','revision/code','-p','test_*.py','-v'],
        ['revision/code/run_revision.py','--stage','all'],
        ['revision/code/run_sensitivity.py'],
        ['revision/code/run_mapping_controls.py'],
        ['revision/code/run_sensitivity.py','--statistics-only'],
        ['revision/code/run_external_xai.py'],
        ['revision/code/postcheck_controls.py','--phase','extras']]
    checks=[]
    for i,args in enumerate(stages,1):
        print(f'CHECKED STAGE {i}/{len(stages)}: '+ ' '.join(args),flush=True)
        start=time.perf_counter()
        path=ROOT/'revision'/'results'/f'verified_stage_{i}.log'
        with path.open('w',encoding='utf-8') as log:
            process=subprocess.Popen([sys.executable,*args],cwd=ROOT,stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in process.stdout:
                log.write(line)
                if line.startswith(('within ','cross ','compression ','split sensitivity ','external SHAP ',
                    'Traceback','FAILED','OK','test_')) or 'Error:' in line:
                    print(line.rstrip(),flush=True)
            code=process.wait()
        checks.append(dict(command=args,returncode=code,seconds=time.perf_counter()-start))
        if code:
            raise RuntimeError(f'Checked stage {i} failed. See {path}')
    after=snapshot()
    if before!=after:
        raise RuntimeError('Computational source changed during the verified run.')
    result=dict(run_id=datetime.now(timezone.utc).isoformat(),stages=checks,
        source_sha256=after,all_local_stages_passed=True,
        external_pending=['actual Raspberry Pi or Jetson measurement'])
    (ROOT/'revision'/'results'/'verified_run.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('ALL LOCAL COMPUTATIONAL STAGES PASSED; actual edge-device measurement remains pending.',flush=True)


if __name__=='__main__':
    main()
