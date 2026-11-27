"""Map NVML host PIDs to container PIDs without inspecting unrelated processes."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--container',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    with (a.out/'host-pid-map.jsonl').open('x') as f:
        while True:
            begin=time.time_ns()
            top=subprocess.check_output(['docker','top',a.container,'-eo','pid'],text=True)
            members={int(s.strip()) for s in top.splitlines()[1:] if s.strip().isdigit()}
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory',
                '--format=csv,noheader,nounits'],text=True)
            rows=[]
            for line in apps.splitlines():
                fields=[s.strip() for s in line.split(',')]
                if len(fields)!=3 or not fields[0].isdigit():continue
                pid=int(fields[0])
                if pid not in members:continue
                try:
                    status=Path(f'/proc/{pid}/status').read_text()
                    nspid=next(s.split(':',1)[1].split() for s in status.splitlines() if s.startswith('NSpid:'))
                    rows.append(dict(host_pid=pid,namespace_pids=list(map(int,nspid)),
                        container_pid=int(nspid[-1]),gpu_uuid=fields[1],used_mib=fields[2]))
                except (FileNotFoundError,ProcessLookupError):pass
            f.write(json.dumps(dict(start_ns=begin,end_ns=time.time_ns(),processes=rows))+'\n');f.flush()
            manifest=a.out/'manifest.json'
            if manifest.exists() and json.loads(manifest.read_text())['status']!='running':break
            time.sleep(5)


if __name__=='__main__':main()
