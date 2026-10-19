"""Offline deployment probe; --restart-worker explicitly tests supervised recovery."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--restart-worker', action='store_true')
parser.add_argument('--output', required=True)
args = parser.parse_args()
unit = 'sprout-lm-pd.service'

def run(*argv):
    return subprocess.check_output(argv, text=True).strip()

def health():
    with urllib.request.urlopen('http://127.0.0.1:3000/health', timeout=3) as response:
        return json.load(response)

def ready(previous=None):
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        pid = int(run('systemctl', 'show', unit, '--property=MainPID', '--value'))
        try:
            if pid and pid != previous and health() == {'status': 'ok', 'pendingRequests': 0}:
                return pid
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError('PD service did not become ready')

def state(pid):
    children = [int(v) for v in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
    assert len(children) == 2, children
    gpu_rows = list(csv.DictReader(io.StringIO(run('nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free', '--format=csv'))))
    gpu_rows = [{k.strip(): v.strip() for k, v in row.items()} for row in gpu_rows]
    apps = list(csv.DictReader(io.StringIO(run('nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv'))))
    apps = [{k.strip(): v.strip() for k, v in row.items()} for row in apps]
    assert pid not in [int(a['pid']) for a in apps], 'Coordinator unexpectedly owns a CUDA context'
    assert set(int(a['pid']) for a in apps) == set(children), (apps, children)
    decode_uuid = next(g['uuid'] for g in gpu_rows if g['index'] == '1')
    decode_pid = next(int(a['pid']) for a in apps if a['gpu_uuid'] == decode_uuid)
    return {'coordinatorPid': pid, 'workerPids': children, 'decodePid': decode_pid, 'gpus': gpu_rows, 'computeProcesses': apps}

def generate():
    request = urllib.request.Request('http://127.0.0.1:3000/generate',
        data=json.dumps({'prompt': '1+1=', 'maxNewTokens': 2}).encode(), headers={'content-type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    assert result['output'] == [17, 11], result
    return result

before = state(ready())
report = {'before': before, 'generationBefore': generate()}
if args.restart_worker:
    assert health()['pendingRequests'] == 0
    os.kill(before['decodePid'], signal.SIGKILL)
    after = state(ready(before['coordinatorPid']))
    for pid in [before['coordinatorPid'], *before['workerPids']]:
        assert not Path(f'/proc/{pid}').exists(), f'old process {pid} survived'
    report.update({'after': after, 'generationAfter': generate(), 'restartVerified': True})
report['health'] = health()
Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
