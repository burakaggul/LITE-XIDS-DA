"""Stream a real subprocess into the visible terminal's shared log."""
import argparse
import subprocess
import time
import re
from common import HERE,log,save

p=argparse.ArgumentParser()
p.add_argument('--label',required=True)
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args()
command=a.command[1:] if a.command[:1]==['--'] else a.command
if not command:
    p.error('A command is required.')
log('RUN '+a.label+': '+subprocess.list2cmdline(command))
start=time.perf_counter()
process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
    encoding='utf-8',errors='replace')
for line in process.stdout:
    if re.match(r'^\d{4}-\d{2}-\d{2}T',line):
        print(line.rstrip(),flush=True)
    else:
        log(line.rstrip())
code=process.wait()
save(HERE/'results'/('step_'+a.label+'.json'),dict(command=command,returncode=code,seconds=time.perf_counter()-start))
log(('PASS ' if code==0 else 'FAIL ')+a.label+' exit='+str(code))
raise SystemExit(code)
