"""Render measured diagnostic tables and an explicit gap list, without estimates."""
import argparse
import json
from pathlib import Path


GIB=2**30


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])+'\n\n'


def f(value):return '未采到' if value is None else f'{value:.3f}'


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    read=lambda rel:json.loads((root/rel).read_text())
    manifest=read('manifest.json');waves=read('analysis/rollout-summary.json')
    stages=read('analysis/detailed/physical-stage-ledger.json');states=read('analysis/detailed/tensor-boundary-ledger.json')
    times=read('analysis/detailed/training-method-times.json');replays=read('analysis/detailed/snapshot-retained-peaks.json')
    limitations=read('analysis/detailed/limitations.json')
    text='# KTRL 30B / 8×RTX 4090：SynSQL 诊断 a02 实测与缺项\n\n'
    text+='本报告只分析已完成的 diagnostic 运行。包含重型 Torch Profiler、内存历史和同步审计，不作为正式性能基线。第一批全零优势跳过；第二批完成1个有效RL轮、2次optimizer更新，发布v2并生成32 tokens。不是两轮有效RL，也没有采到常驻Actor后的下一批完整rollout。\n\n'
    text+=f"- 远程证据根：`{root}`。\n- 源码 HEAD：`{manifest['head']}` + manifest及source.json记录的dirty状态。\n- 配方 SHA256：`{manifest['config_sha256']}`。\n- 8题×4答/批，输入上限6144、输出2048、上下文8192；TP8/FSDP2 8rank，minibatch4题/PPO epoch1/GAS2，CPU专家；save_freq=0。\n- 源码/观察层哈希见manifest；CPU/GPU内存均用GiB，原始bytes保留在JSON。\n\n"
    text+='控制器状态complete、Actor退出0；Rollout服务清理退出-9，不称服务自然退出0。没有把这次结果当成完整多轮/完整profiling验收。\n\n'
    text+='## 1. 两批真实 rollout\n\n'
    rr=[]
    for w in waves:
        phase=read(f"analysis/rollout-{w['wave']:02d}/phase-summary.json")
        rr.append([w['wave'],w['input_tokens'],w['output_tokens'],f(w['client_wall_seconds']),
            f(w['client_output_tps']),f(phase['phases']['EXTEND']['seconds']),f(phase['phases']['EXTEND']['tps']),
            f(phase['phases']['DECODE']['seconds']),f(phase['phases']['DECODE']['tps']),f(w['truncated_fraction']*100)+'%'])
    text+=table(['批次(均v0)','输入token','输出token','客户端整体秒','整体输出TPS','Prefill秒','Prefill TPS','Decode秒','Decode TPS','截断率'],rr)
    text+='整体输出TPS=同批输出token数/客户端首个请求提交至全部请求完成；native控制器的生成阶段还包含start/stop_profile及落盘等待，分别为474.062秒、437.545秒。Prefill/decode分母为全部TP rank对齐CUDA事件区间的并集，不是kernel纯执行时间，也不累加八张卡TPS。decode分子排除每请求第一个token；分别有32个额外调度token单列。输入前缀缓存禁用。时钟锚定误差和重叠见phase-summary.json。\n\n'
    for w in waves:
        text+=f"### 批次{w['wave']}延迟\n\n"
        text+=table(['指标','平均','P50','P95','P99','最大'],[[metric]+[f(w[metric][k]) for k in ['mean','p50','p95','p99','max']] for metric in ['ttft_ms','tpot_ms']])
    text+='单位为ms。来自同一次真实生成的SSE到达事件；TPOT=(末token到达-首token到达)/(输出数-1)。合并chunk记录在逐请求JSON，不伪造每个token的ITL。第1批0/32、第2批4/32通过严格SQL执行结果匹配；14/10/8分别是第1批无最终SQL、执行错误、结果不匹配。\n\n'
    text+='## 2. 阶段耗时\n\n'
    text+=table(['阶段','秒','说明'],[
        ['输入准备','8.483','一次'],['批1 rollout','474.062','全零优势后跳过'],['批1奖励/优势','5.075','包含CPU worker启动与文件读写'],
        ['批2 rollout','437.545','第二批仍为v0，不是新策略rollout'],['批2奖励/优势','5.129','有非零优势'],
        ['Rollout休眠','0.875','采样周期1秒，短阶段峰值覆盖有限'],['Actor阶段','1282.456','含加载、审计、训练、导出与验证'],
        ['rank0模型导出函数','49.831','Actor阶段子项，不能再次加到整轮总耗时'],['发布清单/哈希','65.643','独立阶段'],['恢复/权重发布/Graph及控制验证','90.031','尚未拆清内部子项']])
    tr=[]
    for t in times:
        tr.append([t['label'],f(t['rank_wall_seconds']['0']['sum_seconds']),f(t['cross_rank_union_seconds']),t['rank_wall_seconds']['0']['calls']])
    text+=table(['训练观察函数','rank0累计秒','跨rank区间并集秒','rank0调用数'],tr)
    text+='这些函数存在嵌套与跨rank等待，不能把各行相加当训练总时间。例如training_microbatch包含前向和反向；其他rank进入checkpoint_export后会等待rank0完成CPU优化器，因此其185秒不能称纯磁盘导出。clip_grad_norm观察范围含完整同步裁剪路径，不等于单个GPU kernel时间。尚需用算子trace与原生KT事件进一步拆解。\n\n'
    text+='## 3. 逐卡物理显存：训练阶段采样峰值\n\n'
    actor=next(s for s in stages if s['phase']=='actor_update')
    text+=table(['GPU','整卡峰值GiB','同次采样Actor GiB','同次采样休眠Rollout GiB','整卡-进程差额GiB'],[
        [g['index'],f(g['used_bytes']/GIB),f(sum(p['used_bytes'] for p in g['processes'] if p.get('role')=='actor')/GIB),
         f(sum(p['used_bytes'] for p in g['processes'] if p.get('role')=='rollout')/GIB),f(g['card_minus_processes_bytes']/GIB)] for g in actor['gpu_peaks']])
    text+='每行来自该卡峰值对应的同一次NVML采集；整卡/进程查询先后执行，有约数百毫秒间隔，不保证严格同时。不同卡的峰值可能不同时发生，禁止加总为同步整机峰值；原始start/end保留。\n\n'
    text+='### 同一训练边界的PyTorch组成（第二次optimizer_start）\n\n'
    selected=sorted([r for r in states if r['label']=='optimizer_start' and r['step']==1],key=lambda r:r['rank'])
    data=[]
    for r in selected:
        cap=r['capacity'];d=f"cuda:{r['rank']}/"
        data.append([r['rank'],f(cap.get(d+'parameter',0)/GIB),f(cap.get(d+'gradient',0)/GIB),
            f(cap.get(d+'optimizer.exp_avg',0)/GIB),f(cap.get(d+'optimizer.exp_avg_sq',0)/GIB),
            f(r['unmatched_live_bytes']/GIB),f(r['allocated']/GIB),f(r['reserved_minus_allocated_bytes']/GIB),f(r['reserved']/GIB),f(r['inactive_split_bytes']/GIB)])
    text+=table(['rank','FP32参数','FP32梯度','Adam m','Adam v','其他活跃/待归因','allocated','reserved-allocated','reserved','inactive_split(子项)'],data)
    text+='该表不是上一张NVML峰值时刻的精确拆分；只在同一边界内对账。其他活跃约1.17GiB尚不能全部称激活；其中可能有FSDP计算参数副本，需对象指针补测。reserved-allocated也不是全部不可用碎片，inactive_split仅是其中一个分配器统计子项。NVML进程-reserved的差额不直接标成NCCL。\n\n'
    text+='### Rollout已识别存储\n\n'
    text+=table(['项目','GiB','证据性质'],[
        ['GPU模型参数/TP rank',f(432828416/GIB),'已识别tensor storage；各rank另见边界JSON'],
        ['GPU模型buffer/TP0',f(20971520/GIB),'已识别tensor storage'],
        ['KV cache/TP rank',f(8118487296/GIB),'实际tensor storage，不由0.35比例估算'],
        ['请求token池',f(1049088/GIB),'实际tensor storage'],
        ['Graph池/临时张量/库及上下文','待完整归因','不能用整卡减KV直接全部归为Graph']])
    text+='两批v0 rollout整卡峰值GPU0约9.932GiB，其余卡9.619GiB。恢复/发布阶段Actor与Rollout同时存在，整卡采样峰值约14.604–15.716GiB，但它不是常驻Actor下完整rollout的峰值。\n\n'
    text+='## 4. 容器CPU内存：训练、导出与发布分窗\n\n'
    names=['actor_before_owner_export','model_export_owner','publication_manifest','resume_publish']
    cr=[]
    for name in names:
        s=next(x for x in stages if x['phase']==name);c=s['container_peak']
        cr.append([name,f(c['current_bytes']/GIB),f(c['anon_bytes']/GIB),f(c['file_bytes']/GIB),
            f(c['other_accounting_bytes']/GIB),f(c['shmem_included_in_file_bytes']/GIB)])
    text+=table(['窗口','容器采样峰值','同刻anon','同刻file','内核及其他记账差额','shmem(file子项)'],cr)
    text+=f"起始容器占用{limitations['baseline_container_bytes']/GIB:.3f}GiB，其中file缓存{limitations['baseline_file_bytes']/GIB:.3f}GiB；已有容器保留历史缓存，不能全部归为本次模型新增内存。内核没有提供memory.stat.kernel聚合字段，因此差额不能声称是精确内核内存。训练前段与导出、权重发布的峰值分别报告，memory.peak生命周期值不当本次峰值。\n\n"
    text+='### CPU专家训练张量容量（rank0 owner）\n\n'
    text+=table(['张量','dtype','GiB','驻留规则'],[
        ['计算权重','BF16','54.000','CPU；含192B占位storage差异另见JSON'],['梯度','BF16','54.000','反向/累积时存在，step后.grad清除'],
        ['主权重','FP32','108.000','CPU优化器持久状态'],['Adam一阶矩','FP32','108.000','持久状态'],['Adam二阶矩','FP32','108.000','持久状态'],
        ['含梯度合计','—','432.000','第二次优化器入口已识别容量'],['不含梯度合计','—','378.000','优化器结束/导出边界已识别容量']])
    text+='这是按storage地址去重的容量，不等于进程PSS，也不能从另一时刻的容器anon峰值直接减去432/378来声称精确未知量。推理端CPU专家、CPU备份、原生缓冲和allocator保留页尚须进一步归因。\n\n'
    text+='### 容器峰值附近的进程PSS（非精确同时刻）\n\n'
    pr=[]
    for name in names:
        s=next(x for x in stages if x['phase']==name);v=s['nearest_pss'];t=v['bytes_by_role']
        pr.append([name,f(v['offset_seconds']),f(t.get('actor',0)/GIB),f(t.get('rollout',0)/GIB),f(t.get('controller_or_other_container_process',0)/GIB)])
    text+=table(['窗口','PSS相对峰值偏移秒','Actor全部进程PSS GiB','Rollout全部进程PSS GiB','控制器/其他GiB'],pr)
    text+='PSS采样约10秒，容器/NVML约1秒；PSS按父子树和rank PID分组。共享页按PSS分摊，不对RSS直接求和。PSS与cgroup记账并不等价，表中数值不要求精确加到容器峰值。逐进程PSS/USS/RSS及匿名/文件/共享分量在physical-stage-ledger.json中。\n\n'
    text+='## 5. 已验证与仍缺项\n\n'
    text+='- 80份Actor快照反向回放均未发现账本一致性错误；另有16份Rollout快照，总计96份。\n- 但每进程16384条事件的循环历史不能恢复本轮全部真实峰值：所保留窗口可重建最高约5.211GiB requested，而各rank内部peak_allocated更高。不得把这5.211GiB称整个训练峰值。需要对新高水位及时落盘，并提前记录FSDP根参数计算副本。\n- 两批同轮SSE和全部TP CUDA阶段证据已覆盖；仍需独立审阅计时分母和首次/后续token归属。\n- 原生CPU expert workspace及推理packed权重缺分配调用栈，CUDA分配器外差额缺地址归因。\n- 首微批8rank Torch trace已有，主要算子/CPU专家/通信时间解析仍待完成。\n- 未采到常驻Actor后的完整下一轮rollout；后续正式验收至少3个有效RL轮，跳过批次不计。\n- 长输入乱码/重复、SQL严格列匹配和输出截断需要受控诊断；当前不能冻结正式配方。\n- 未跑轻量正式测量，不能把这次diagnostic耗时和内存直接作为正式性能结论。\n\n'
    text+='原始详细文件：`analysis/detailed/{physical-stage-ledger,tensor-boundary-ledger,training-method-times,snapshot-state-ledger,snapshot-retained-peaks,limitations}.json`；执行/权重发布证据见`rl/accepted.json`与`rl/round-01/`；完整采集覆盖仍为pending，不能称无缺项。\n'
    path=root/'analysis/KTRL诊断a02-已采指标与内存账本.md';path.write_text(text,encoding='utf-8');print(path)


if __name__=='__main__':main()
