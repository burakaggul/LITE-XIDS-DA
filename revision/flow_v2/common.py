"""Checked local logging and immutable evidence helpers for the new pipeline."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import sys

if hasattr(sys.stdout,'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
for folder in ['logs','results','manifests','cache']:
    (HERE/folder).mkdir(exist_ok=True)


def log(message):
    line=datetime.now(timezone.utc).isoformat(timespec='seconds')+' '+str(message)
    print(line,flush=True)
    with (HERE/'logs/live.log').open('a',encoding='utf-8') as stream:
        stream.write(line+'\n')


def save(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=True),encoding='utf-8')
    temporary.replace(path)


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):
            result.update(chunk)
    return result.hexdigest()
