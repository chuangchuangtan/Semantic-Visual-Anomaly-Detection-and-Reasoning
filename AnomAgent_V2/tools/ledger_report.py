"""Per-step token / latency / cost report from ledger.jsonl (+ per-image stats).

Usage:
  python tools/ledger_report.py --run-dir outputs/run_..._baseline50 [--image NAME]
"""
import argparse
import json
import os
from collections import defaultdict

def load_run_ledger(run_dir):
    rows = []
    path = os.path.join(run_dir, "ledger.jsonl")
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--image", default=None, help="Filter to one image (name with or without extension).")
    args = ap.parse_args()

    rows = load_run_ledger(args.run_dir)
    img_filter = None
    if args.image:
        img_filter = os.path.splitext(args.image)[0]
        rows = [r for r in rows if os.path.splitext(r.get("image") or "")[0] == img_filter]
    if not rows:
        print(f"no ledger rows found under {args.run_dir} (run may predate the ledger feature)")
        return

    by_step = defaultdict(lambda: defaultdict(float))
    lat = defaultdict(list)
    errors = 0
    truncated = 0
    for r in rows:
        cached = r.get("status") == "cache_hit"
        st = by_step[r.get("step", "unknown")]
        st["calls"] += 1
        if not cached:  # cached replay rows carry the ORIGINAL usage; not consumed now
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if isinstance(r.get(k), (int, float)):
                    st[k] += r[k]
            if isinstance(r.get("latency_ms"), (int, float)):
                lat[r.get("step", "unknown")].append(r["latency_ms"])
        if r.get("status") == "error":
            errors += 1
        if r.get("status") == "cache_hit":
            by_step[r.get("step", "unknown")]["cache_hits"] += 1
        if r.get("finish_reason") == "length" and not cached:
            truncated += 1

    tot_prompt = sum(v["prompt_tokens"] for v in by_step.values())
    tot_comp = sum(v["completion_tokens"] for v in by_step.values())
    tot_all = tot_prompt + tot_comp
    cache_hits_total = 0
    print(f"{'step':<20} {'calls':>6} {'cache':>6} {'prompt':>10} {'completion':>11} {'total':>10} {'avg_ms':>9} {'share%':>7}")
    for step in sorted(by_step, key=lambda k: -by_step[k]["total_tokens"]):
        v = by_step[step]
        lavg = sum(lat[step]) / len(lat[step]) if lat[step] else 0
        share = 100.0 * v["total_tokens"] / tot_all if tot_all else 0
        hits = v.get("cache_hits", 0)
        cache_hits_total += hits
        print(f"{step:<20} {v['calls']:>6.0f} {hits:>6.0f} {v['prompt_tokens']:>10.0f} {v['completion_tokens']:>11.0f} "
              f"{v['total_tokens']:>10.0f} {lavg:>9.0f} {share:>7.1f}")
    print("-" * 84)
    print(f"total tokens: {tot_all:.0f} (prompt {tot_prompt:.0f} / completion {tot_comp:.0f}) | "
          f"error attempts: {errors} | truncated(finish=length): {truncated} | cache hits: {cache_hits_total}")

    # per-image stats from every_steps
    ev = os.path.join(args.run_dir, "every_steps")
    imgs = []
    if os.path.isdir(ev):
        for fn in sorted(os.listdir(ev)):
            if not fn.endswith(".json") or fn.endswith(".ledger.jsonl") or ".viz." in fn:
                continue
            d = json.load(open(os.path.join(ev, fn), encoding="utf-8"))
            if args.image and d.get("image_name") != args.image:
                continue
            u = d.get("usage") or {}
            el = (d.get("final") or {}).get("elapsed_minutes")
            n_ann = len((d.get("final") or {}).get("anomalies") or [])
            imgs.append((fn[:-5], u.get("total_tokens", 0), el, n_ann))
    if imgs:
        tt = sum(x[1] for x in imgs)
        te = [x[2] for x in imgs if x[2] is not None]
        ta = sum(x[3] for x in imgs)
        print(f"\nimages done: {len(imgs)} | total tokens: {tt} | avg tokens/image: {tt/len(imgs):.0f} | "
              f"avg elapsed: {sum(te)/len(te):.1f} min | anomalies: {ta} (avg {ta/len(imgs):.1f}/img)")

if __name__ == "__main__":
    main()
