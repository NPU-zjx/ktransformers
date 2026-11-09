#!/usr/bin/env bash
# Supported CUDA 13 collector; a full native online cycle, independent output.
set -euo pipefail
test "$#" -eq 3
source_dir=$(realpath "$1")
recipe=$(realpath "$2")
output_root=$(realpath "$3")
case "$output_root" in /raid/data/zjx/kt-baseline/results/*) ;; *) exit 2 ;; esac
test ! -e "$output_root/rl-online-profile01"
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
python_bin=/mnt/qjh007/jiahengdai/venvs/kt-py31115/bin/python
nsys_bin=/usr/local/cuda-13.0/bin/nsys
"$python_bin" -B "$script_dir/launch.py" \
  --source "$source_dir" \
  --recipe "$recipe" \
  --root "$output_root" \
  --mode nsys \
  --check
mkdir -p "$output_root/overlay" "$output_root/nsys" "$output_root/smoke"
if [ ! -f "$output_root/overlay/SHA256.txt" ]; then
  cp "$script_dir/launch.py" "$script_dir/sitecustomize.py" "$output_root/overlay/"
  sha256sum "$output_root/overlay/"*.py > "$output_root/overlay/SHA256.txt"
fi
sha256sum -c "$output_root/overlay/SHA256.txt"
test -f "$output_root/nsys/version.txt" || "$nsys_bin" --version > "$output_root/nsys/version.txt"
if ! compgen -G "$output_root/smoke/*.nsys-rep" > /dev/null; then
"$nsys_bin" profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat:8 \
  --kill=none \
  --wait=all \
  --output="$output_root/smoke/smoke" \
  "$python_bin" -B "$script_dir/nsys_smoke.py"
fi
for report in "$output_root/smoke/"*.nsys-rep; do
  test -f "${report%.nsys-rep}.sqlite" || "$nsys_bin" export --type=sqlite --output="${report%.nsys-rep}.sqlite" "$report"
done
/usr/bin/python3 - "$output_root/smoke" <<'PY'
import json,sqlite3,sys
from pathlib import Path
root=Path(sys.argv[1]); kernels=0; labels=set()
for db in root.glob('*.sqlite'):
    con=sqlite3.connect(db)
    kernels+=con.execute('select count(*) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
    labels.update(r[0] for r in con.execute('select text from NVTX_EVENTS where text is not null'))
    labels.update(r[0] for r in con.execute('select s.value from NVTX_EVENTS n join StringIds s on n.textId=s.id'))
    con.close()
assert kernels>=2 and {'rollout-probe','actor-probe'} <= labels,(kernels,labels)
(root/'validated.json').write_text(json.dumps(dict(kernels=kernels,labels=sorted(labels),optimizer_updates=0)))
PY
"$nsys_bin" profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --cuda-graph-trace=node \
  --cuda-flush-interval=1000 \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat:8 \
  --kill=none \
  --wait=all \
  --output="$output_root/nsys/online" \
  "$python_bin" -B -u "$output_root/overlay/launch.py" \
  --source "$source_dir" \
  --recipe "$recipe" \
  --root "$output_root" \
  --mode nsys
/usr/bin/python3 -B "$script_dir/collect_nsys_reports.py" "$output_root" --nsys "$nsys_bin"
/usr/bin/python3 - "$output_root/nsys-analysis/coverage.json" <<'PY'
import json,sys
from pathlib import Path
proof=json.loads(Path(sys.argv[1]).read_text())
assert proof['kernel_api_and_device_coverage_passed'],proof['missing_device_coverage']
print('Eight-device kernel/API coverage verified; raw warnings still require review')
PY
