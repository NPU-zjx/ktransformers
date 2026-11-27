"""Client arrival metrics; E2E TPOT remains separate and includes real stalls."""
import statistics

def percentile(values,q):
    values=sorted(values);x=(len(values)-1)*q;lo=int(x);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(x-lo)

def arrival_metrics(events):
    first,last=events[0],events[-1]
    second=next((e for e in events if e['total_tokens']>=2),None)
    remaining=last['total_tokens']-second['total_tokens'] if second else 0
    intervals=[(b['time_ns']-a['time_ns'])/1e6 for a,b in zip(events,events[1:]) if b['delta']==1]
    return dict(first_to_second_token_ms=(second['time_ns']-first['time_ns'])/1e6 if second and first['delta']==1 else None,
        steady_tpot_ms=(last['time_ns']-second['time_ns'])/1e6/remaining if remaining>0 else None,
        arrival_itl_ms=dict(count=len(intervals),mean=statistics.fmean(intervals),p50=percentile(intervals,.5),p95=percentile(intervals,.95),max=max(intervals)) if intervals else None,
        arrival_itl_scope='Intervals ending in a single-token SSE chunk; coalesced-token intervals excluded. Includes real scheduling/network stalls.')
