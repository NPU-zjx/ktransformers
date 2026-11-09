# KT base 单轮实验

在 qjh005 的仓库目录执行。模型路径、容器环境、推理参数和请求采样参数统一在 `profiling/base.sh` 修改。外层 `profiling/run.py` 按顺序调用 Bash。Bash 在嘉珩的容器环境中执行官方 `python -m sglang.launch_server`。

```bash
cd /raid/data/zjx/kt-baseline/ktransformers
python3 -B profiling/run.py run
```

`run` 启动服务，等待就绪，完成最多 64 token 的短热身，验证 8 个 rank 的 decode Graph 回放，再提交一轮 32 条固定输入，最后停止本次容器。要逐步执行，可以依次使用 `serve`、`rollout`、`stop` 代替 `run`。

当前配置使用 Qwen3.6-35B-A3B base、TP8、BF16、32 条 KT CPU 线程和 2 个子池。固定输入位于 `profiling/data/jiaheng-prompts32.json`。结果默认位于 `/raid/data/zjx/kt-baseline/results/base-r01/analysis/`。服务日志在同一实验目录的 `server.log`。阶段解析按当前 8 个 rank 核验。启动要求 GPU 空闲及结果目录尚未存在。

## Base PyTorch Profiler 短诊断

```bash
cd /raid/data/zjx/kt-baseline/ktransformers
python3 -B profiling/run.py torch
```

`base.sh` 沿用同一组模型参数，启动专用容器。客户端 `torch_profile.py` 等服务就绪后，用与 step23 一致的 127-token 输入发一条最多 32-token 的贪心请求。SGLang 原生分阶段采集 CPU/CUDA 活动，要求每个 rank 都有一个 prefill batch 和八个实际 Graph 回放的 decode batch。Python 入口在收尾时停止专用容器。

原始 trace 和 CSV 位于 `/raid/data/zjx/kt-baseline/results/base-torch-r01/torch-diagnostic/`。其中 `analysis/operators-by-rank-phase.csv` 给出各 rank 的 CPU 调用数、CPU/设备总耗时、自耗时和平均耗时（微秒）；`analysis/rank0-prefill-decode-comparison.csv` 将两个阶段并排；`analysis/kernels-by-rank-phase.csv` 给出实际 kernel 次数和执行耗时。Graph 内的算子不逐次重走 Python，CPU 调用数按原始 CPU 事件计数，GPU kernel 数单列。

## Base Nsight Systems 短诊断

```bash
cd /raid/data/zjx/kt-baseline/ktransformers
python3 -B profiling/run.py nsys
```

入口沿用 `base.sh` 的模型参数，在官方服务外运行 `nsys profile`，采集 `cuda,nvtx,osrt` 并启用 CUDA Graph 节点跟踪。服务就绪后，`nsys_profile.py` 通过 CUDA profiler API 开关采集窗口，对固定 127-token 输入生成最多 32 个 token，检查 8 个 rank 的实际 decode 回放。运行产生独立的 `results/base-nsys-r01/rollout.nsys-rep` 和 `rollout.sqlite`，最后停止专用容器。

结果目录 `analysis/` 下，`kernels.csv` 给出各阶段 kernel 调用次数和耗时，`copies.csv` 给出 H2D/D2H 等拷贝的次数、字节和耗时，`cpu-sync.csv` 给出 CPU 线程在显式 CUDA 同步 API 内的时间，`phase-waits.csv` 给出阶段活动跨度内的 GPU 空档与通信类 kernel 占用时间。阶段按提交 API 所在的 NVTX 区间归属，`OUTSIDE_MARKER` 单列。各阶段活动跨度可能重叠，GPU 空档本身不能定位具体等待原因。采集工具的原始诊断信息随结果保存。
