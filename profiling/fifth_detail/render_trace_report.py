"""Render actual trace results plus GPU-validation gaps; no new experiment."""
import argparse
import json
from pathlib import Path


def table(headers,data):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(x) for x in row)+' |' for row in data])+'\n\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    data=json.loads((root/'analysis/trace-details/all-ranks.json').read_text())
    text='# KTRL诊断a02：训练细分、采集修正与补测清单\n\n'
    text+='本页来自已完成运行的8份首微批Torch trace与48层KT原生计数器；没有追加GPU实验。该首微批包含重型profiler开销，不代表正式速度。整批时间和内存表见同目录《KTRL诊断a02-已采指标与内存账本.md》。\n\n'
    text+='## 1. 首微批逐rank时间与GPU活动\n\n'
    result=[]
    for r in data['ranks']:
        phase={x['label']:x for x in r['phases']};m=phase['first_microbatch']
        result.append([r['rank'],f"{m['wall_seconds']:.3f}",f"{phase['policy_forward_loss']['wall_seconds']:.3f}",
            f"{phase['backward']['wall_seconds']:.3f}",f"{m['kernel_interval_union_seconds']:.3f}",
            f"{m['kernel_groups'].get('nccl_communication_or_wait',{}).get('interval_union_seconds',0):.3f}",
            f"{m['kernels_linked_by_external_id']}/{m['kernel_calls']}"])
    text+=table(['rank','微批墙钟秒','前向+loss秒','反向秒','kernel区间并集秒','NCCL通信/等待并集秒','kernel可关联CPU producer'],result)
    text+='rank0只有约3.2秒落在已记录GPU kernel区间，其他rank则约90秒有kernel运行，其中绝大部分是NCCL通信或等待。结合CPU owner的KTMoE操作耗时，表明不能将其他卡接近100%的GPU利用率解释为在持续执行模型矩阵计算。NCCL区间仍包含传输、同步、轮询，不能将所有时间机械地标成CPU专家计算。\n\n'
    text+='全部反向窗口内的kernel都可通过记录的External id关联CPU producer；整个微批仍有约144–145个kernel/rank无法这样关联，保留为未关联。没有声称100%全模型事件归因。跨边界事件单独计数；本次CPU嵌套没有发现交叉冲突。\n\n'
    text+='## 2. rank0首微批PCIe传输事件\n\n'
    rank0=next(r for r in data['ranks'] if r['rank']==0)
    transfer_rows=[]
    for phase in rank0['phases']:
        v=phase['transfers']
        htod=sum(n for k,n in v.items() if k.startswith('Memcpy HtoD') and k.endswith('.bytes'))
        dtoh=sum(n for k,n in v.items() if k.startswith('Memcpy DtoH') and k.endswith('.bytes'))
        dtod=sum(n for k,n in v.items() if k.startswith('Memcpy DtoD') and k.endswith('.bytes'))
        transfer_rows.append([phase['label'],f'{htod/2**30:.3f}',f'{dtoh/2**30:.3f}',f'{dtod/2**30:.3f}'])
    text+=table(['窗口','H2D GiB','D2H GiB','D2D GiB'],transfer_rows)
    text+='这是实际memcpy事件的传输字节累计，不是驻留显存。首微批包含前向和反向，不能与子行再相加。rank0为CPU专家owner，处理跨rank数据；当前尚未将每笔传输关联到具体输入/输出/梯度storage，不能仅凭传输大小说是专家权重搬运，也不能把这些流量当所有GPU的总量。\n\n'
    text+='## 3. 原生CPU专家累计计时\n\n'
    selected=['wrapper.tp.forward.total','wrapper.tp.forward.numa_compute','tp.0.forward.recompute_total',
              'wrapper.tp.backward.total','tp.0.backward.base_weight_grad','tp.0.backward.base_weight_grad.matmat',
              'tp.0.backward.base_weight_grad.panel_pack']
    values={r['counter']:r for r in data['native']['counters']}
    text+=table(['计数器','累计秒','调用数','层数'],[[k,f"{values[k]['seconds']:.3f}",values[k]['calls'],values[k]['layers']] for k in selected if k in values])
    text+='这些是owner从启用至收集期间的累计计数，包含old logp、重计算与其他审计前向，不等于仅策略前向的时间。计数器之间存在包含关系，线程累计时间也不能与墙钟相加。完整每层计数器见`trace-details/native-experts-per-layer.csv`。\n\n'
    text+='## 4. 已发现的采集问题与修正\n\n'
    text+=table(['问题','对旧数据的影响','观察层修正','验证状态'],[
        ['后台线程CUDA设备默认0','rank1–7连续allocator曲线无效；主线程边界、NVML和trace另行有效',
         '启动采样线程时显式选择父线程CUDA设备','源码与CPU检查通过；需8卡短跑'],
        ['FSDP内部计算副本未枚举','约1.17GiB其他活跃存储尚未认定用途',
         '只读_get_fsdp_state及已有compute/all_gather存储，按地址去重','CPU隐藏副本/别名检查通过；需真实对象验证'],
        ['16384条循环历史过短','80份Actor快照可回放，但不足以证明整轮峰值组成',
         '262144条历史、前后向和周期新高水位快照、requested峰值记录','仅语法/CPU检查；需GPU峰值回放'],
        ['两批不等于两轮有效RL','a02仅一个有效更新轮，缺常驻后完整rollout',
         '覆盖检查明确要求至少3个有效RL轮；跳过批次不计','正式验证未启动']])
    text+='旧实验和原始证据保持不变，新观察代码用于后续独立目录。以上修正仅在profiling层，没有改变loss、优化器、精度、权重发布或采样奖励。\n\n'
    text+='## 5. 当前门槛与后续顺序\n\n'
    text+='1. 已完成现有阶段/内存结果表和8rank算子、kernel、原生计数器离线解析；GPU峰值和原生分配归因仍有显式缺项。\n2. 有8卡空闲窗口后，先短跑核验采样线程设备、FSDP隐藏存储和峰值快照覆盖，再测常驻Actor完整下一轮rollout；不以CPU测试替代GPU验收。\n3. 用固定异常题对照单请求/批量、prefill分块及2048/4096输出。条件逐项改变；不放宽奖励制造学习信号。\n4. 配方、采集覆盖和资源通过后，才运行至少3轮有效RL的轻量正式profiling；首轮与常驻轮分开比较。\n\n'
    text+='当前7张卡是其他用户的真实训练，GPU1是已核实但按用户指示保留的占卡程序。没有清理它们，也没有启动新GPU任务。GPU补测与正式采集等待资源窗口。未提交、未push。\n'
    output=root/'analysis/KTRL诊断a02-训练细分与补测清单.md';output.write_text(text,encoding='utf-8');print(output)


if __name__=='__main__':main()
