"""Record actual runtimes and desktop hardware without publishing machine/user identifiers."""
import importlib.metadata
import json
import platform
import subprocess
import psutil
from common import HERE,save,log

PACKAGES=['numpy','pandas','scikit-learn','torch','xgboost','lightgbm','psutil','duckdb','pyarrow',
    'nfstream','scapy','shap','pytorch-tcn','scipy','joblib','huggingface_hub','hf-xet']


def record():
    versions={}
    for name in PACKAGES:
        try:
            versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name]=None
    cpu=subprocess.run(['powershell','-NoProfile','-Command',
        'Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json -Compress'],
        capture_output=True,text=True,check=True)
    linux=subprocess.run(['wsl','-d','Ubuntu','--','/home/msi/.venvs/litexids-flow-v2/bin/python','-c',
        "import importlib.metadata,json,platform;print(json.dumps(dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in ['nfstream','numpy','pandas','scapy','psutil','cffi']})))"],
        capture_output=True,text=True,check=True)
    save(HERE/'results/runtime_environment.json',dict(windows=dict(python=platform.python_version(),os=platform.platform(),
        cpu=json.loads(cpu.stdout),ram_bytes=psutil.virtual_memory().total,packages=versions),linux_extraction=json.loads(linux.stdout),
        training_threads=4,inference_threads=1,
        benchmark_context='Desktop CPU measurements during concurrent source download/extraction jobs; not isolated embedded-device performance. Process RSS includes the Python/framework process and cached artifacts, not model-only memory.',
        extractor_workers='4 requested; corrected chronology-safe extraction records effective_n_meters per capture. The earlier archived unsorted exploratory run sometimes reduced to 1.',
        latency_scope='Completed-flow model and preprocessing-plus-model latency only. Packet capture, flow construction, idle timeout (30s) and active timeout (120s) are excluded; no live time-to-detection or full IDS throughput is claimed.',
        real_edge_hardware_used=False))
    log('ENVIRONMENT PASS: actual i9-14900HX desktop and separate Linux extraction runtime recorded; no edge-device result claimed')


if __name__=='__main__':
    record()
