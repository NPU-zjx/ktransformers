"""Mixed prefill/decode iterations are charged inclusive time, never split by guess."""
import json

def summarize(folder,traces,input_tokens,decode_tokens,scheduled,union_ns):
    totals={}
    for phase,field,tokens in [("EXTEND","context_tokens",input_tokens),("DECODE","generation_tokens",decode_tokens)]:
        intervals=[];mixed=[];bounds={}
        for t in traces:
            selected=[r for r in t["rows"] if r[field]]
            bounds[t["rank"]]={k:max(r[k] for r in selected) for k in
                ("allocated_before_bytes","allocated_after_bytes","reserved_before_bytes","reserved_after_bytes")}
            mid=(t["lower"]+t["upper"])//2
            for r in selected:
                span=(mid+round(r["stream_start_ms"]*1e6),mid+round(r["stream_end_ms"]*1e6))
                intervals.append(span)
                if r["phase"]=="MIXED":mixed.append(span)
        seconds=union_ns(intervals)/1e9
        totals[phase]=dict(tokens=tokens,seconds=seconds,tps=tokens/seconds,mixed_union_seconds=union_ns(mixed)/1e9,boundary_bytes=bounds)
    rows=traces[0]["rows"]
    out=dict(phases=totals,scheduled_decode=scheduled,extra_scheduled_decode=scheduled-decode_tokens,
             batch_counts={s:sum(r["phase"]==s for r in rows) for s in ("EXTEND","DECODE","MIXED","IDLE")},
             decode_actual_replays_by_rank={t["rank"]:sum(r["graph_replayed"] for r in t["rows"] if r["generation_tokens"]) for t in traces},
             definition="Phase denominator includes full intervals with that phase's tokens, through sampling. Mixed overlaps both; do not add phase times.")
    (folder/"summary.json").write_text(json.dumps(out,indent=2))
