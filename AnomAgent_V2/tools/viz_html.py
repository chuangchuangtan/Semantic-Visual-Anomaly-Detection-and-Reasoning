#!/usr/bin/env python3
"""Per-sample HTML visualization of an AnomAgent_V2 run.

Renders ONE self-contained HTML page per image: the full step timeline with,
for every LLM call, the input (system + user text + the exact image variant
sent, with object bboxes overlaid on the global view) and the output (raw
text + parsed JSON), plus per-step model / client / tokens / latency /
finish_reason. Crop decisions and the final anomaly list are rendered on top.

Usage:
  python tools/viz_html.py --run-dir outputs/run_... --all
  python tools/viz_html.py --run-dir outputs/run_... --sample 2025_04_18_...jpg
  python tools/viz_html.py --run-dir outputs/run_... --samples a.jpg,b.jpg
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import sys
from typing import Optional

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

CSS = """
:root { --bg:#0f1115; --card:#181b22; --card2:#1f2430; --tx:#e6e9ef; --mut:#8b93a7;
        --acc:#5aa9ff; --ok:#3fb96f; --err:#e05563; --warn:#e0a455; --line:#2a3040; }
* { box-sizing: border-box; }
body { margin:0; font: 14px/1.5 -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'PingFang SC', 'Microsoft YaHei', sans-serif;
       background: var(--bg); color: var(--tx); }
.wrap { max-width: 1280px; margin: 0 auto; padding: 24px 20px 80px; }
h1 { font-size: 22px; margin: 0 0 4px; word-break: break-all; }
h2 { font-size: 17px; margin: 34px 0 10px; color: var(--acc); }
.mut { color: var(--mut); }
.pill { display:inline-block; padding: 1px 9px; border-radius: 10px; font-size: 12px; margin: 0 6px 0 0; border:1px solid var(--line); }
.pill.ok { color: var(--ok); border-color: var(--ok); }
.pill.err { color: var(--err); border-color: var(--err); }
.pill.warn { color: var(--warn); border-color: var(--warn); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; margin: 12px 0; }
.stephead { display:flex; flex-wrap:wrap; align-items:baseline; gap: 8px; margin-bottom: 8px; }
.seq { color: var(--mut); font-variant-numeric: tabular-nums; }
.step { font-weight: 700; font-size: 15px; }
.tool { color: var(--acc); }
.obj { color: var(--warn); }
.grid { display:grid; grid-template-columns: minmax(320px, 420px) 1fr; gap: 16px; align-items:start; }
@media (max-width: 900px){ .grid { grid-template-columns: 1fr; } }
img.view { width:100%; border-radius: 8px; border:1px solid var(--line); background:#000; }
.imglabel { font-size: 12px; color: var(--mut); margin: 6px 0 4px; }
pre { background:#0b0d12; border:1px solid var(--line); border-radius:8px; padding:10px; overflow:auto;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; max-height: 420px; white-space: pre-wrap; word-break: break-word; }
details { margin: 8px 0; }
summary { cursor:pointer; color: var(--acc); font-size: 13px; user-select:none; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border:1px solid var(--line); padding: 6px 9px; text-align:left; vertical-align: top; }
th { background: var(--card2); }
.sev { font-variant-numeric: tabular-nums; }
.toc a { color: var(--acc); text-decoration:none; display:block; padding:4px 0; }
.toc a:hover { text-decoration: underline; }
.overlay { position: relative; display:inline-block; width:100%; }
.overlay .box { position:absolute; border:2px solid #ff5aa9; background: rgba(255,90,169,.10); border-radius:2px; }
.overlay .boxlbl { position:absolute; top:-16px; left:0; font-size:11px; color:#ff5aa9; white-space:nowrap; }
.kv { font-size: 12px; color: var(--mut); }
"""


def _b64_file(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return None


def _global_image_embed(viz: dict, max_side: int = 1600) -> Optional[str]:
    """Embed the ORIGINAL image (resized) as a base64 data URL."""
    path = viz.get("image_path")
    if Image is None or not path or not os.path.exists(path):
        return None
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            if max(im.size) > max_side:
                r = max_side / float(max(im.size))
                im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=86)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _asset_data_url(run_dir: str, assets_dir: str, rel: str) -> Optional[str]:
    b64 = _b64_file(os.path.join(run_dir, "every_steps", assets_dir, rel))
    if not b64:
        return None
    ext = rel.rsplit(".", 1)[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    return f"data:{mime};base64,{b64}"


def render_step_card(idx: int, e: dict, run_dir: str, assets_dir: str, global_b64: Optional[str],
                     global_bboxes: dict, show_global: bool) -> str:
    step = e.get("step", "?")
    tool = e.get("tool", "direct")
    obj = e.get("object") or ""
    status = e.get("status", "?")
    usage = e.get("usage") or {}
    u_p, u_c, u_t = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), usage.get("total_tokens", 0)
    finish = e.get("finish_reason") or "-"
    status_cls = "ok" if status in ("ok", "cache_hit") else "err"
    status_lbl = "cached" if status == "cache_hit" else status
    badges = [
        f'<span class="pill {status_cls}">{html.escape(status_lbl)}</span>',
        f'<span class="pill">finish: {html.escape(str(finish))}</span>',
        f'<span class="pill">tokens {u_t} (p {u_p} / c {u_c})</span>',
        f'<span class="pill">{float(e.get("latency_ms") or 0):.0f} ms</span>',
        f'<span class="pill">T={e.get("temperature", "?")}</span>',
        f'<span class="pill">max_tokens {e.get("max_tokens", "?")}</span>',
        f'<span class="pill">{html.escape(e.get("client","?"))}:{html.escape(str(e.get("model","?"))[:40])}</span>',
    ]
    if e.get("attempts", 1) > 1:
        badges.append(f'<span class="pill warn">attempt {e.get("attempts")}</span>')

    inp = e.get("input") or {}
    user_text = inp.get("user_text") or ""
    system_text = inp.get("system") or ""
    images_html = ""
    for im in inp.get("images") or []:
        kind = im.get("kind", "")
        label = im.get("label", "")
        rel = im.get("file")
        if kind == "global" and global_b64:
            if show_global:
                boxes = {}
                if obj and obj in global_bboxes and global_bboxes[obj]:
                    boxes = {obj: global_bboxes[obj]}
                elif not obj:
                    boxes = {k: v for k, v in global_bboxes.items() if v}
                images_html += (
                    f'<div><div class="imglabel">input image: <b>{html.escape(label or "global (full) tier")}</b>'
                    + (f" · bbox: {obj}" if boxes else "") + "</div>"
                    f'<div class="overlay"><div style="position:relative;width:100%">'
                    f'<img class="view" src="data:image/jpeg;base64,{global_b64}" style="position:relative;z-index:0">'
                    + "".join(
                        f'<span class="box" style="z-index:1;position:absolute;left:{b[0]/10:.2f}%;top:{b[1]/10:.2f}%;'
                        f'width:{(b[2]-b[0])/10:.2f}%;height:{(b[3]-b[1])/10:.2f}%"></span>'
                        for b in boxes.values()
                    )
                    + "</div></div></div>"
                )
            else:
                images_html += (
                    '<div class="imglabel mut">input image: global (full) tier — '
                    'same image, shown at the first global step</div>'
                )
        else:
            data_url = _asset_data_url(run_dir, assets_dir, rel) if rel else None
            if data_url:
                images_html += (
                    f'<div><div class="imglabel">input image: <b>{html.escape(label or kind)}</b></div>'
                    f'<img class="view" src="{data_url}"></div>'
                )
            else:
                images_html += f'<div class="imglabel mut">input image: {html.escape(label or kind)} (asset missing)</div>'
    if not images_html:
        images_html = '<div class="imglabel mut">no image input (text-only step)</div>'

    out = e.get("output") or {}
    raw = out.get("raw")
    parsed = out.get("parsed")
    out_html = ""
    if raw:
        out_html += f'<details{" open" if len(str(raw)) < 1200 else ""}><summary>raw output ({len(str(raw))} chars)</summary><pre>{html.escape(str(raw))}</pre></details>'
    if parsed is not None:
        try:
            pretty = json.dumps(parsed, ensure_ascii=False, indent=1)
        except Exception:
            pretty = str(parsed)
        out_html += f'<details{" open" if len(pretty) < 1600 else ""}><summary>parsed (validated JSON)</summary><pre>{html.escape(pretty)}</pre></details>'
    if e.get("error"):
        out_html += f'<div class="pill err">error: {html.escape(str(e["error"]))}</div>'
    if not out_html:
        out_html = '<div class="mut">no output recorded</div>'

    sys_html = ""
    if system_text:
        sys_html = f'<details><summary>system prompt ({len(system_text)} chars)</summary><pre>{html.escape(system_text[:6000])}</pre></details>'
    user_html = (
        f'<details{" open" if len(user_text) < 1500 else ""}><summary>user prompt ({len(user_text)} chars)</summary>'
        f'<pre>{html.escape(user_text)}</pre></details>'
    )

    return (
        f'<div class="card" id="step{idx}">'
        f'<div class="stephead"><span class="seq">#{idx:02d}</span><span class="step">{html.escape(step)}</span>'
        f'<span class="tool">tool: {html.escape(tool)}</span>'
        + (f'<span class="obj">{html.escape(obj)}</span>' if obj else "")
        + f'<span class="kv">{html.escape(str(e.get("client","")))}</span></div>'
        + "".join(badges)
        + f'<div class="grid"><div>{images_html}</div><div>{sys_html}{user_html}{out_html}</div></div>'
        + "</div>"
    )


def render_sample(run_dir: str, viz: dict) -> str:
    image_name = viz.get("image_name", "?")
    info = viz.get("image_info") or {}
    entries = viz.get("entries") or []
    final = viz.get("final") or {}
    anomalies = final.get("anomalies") or []
    crop_decisions = viz.get("crop_decisions") or {}
    assets_dir = viz.get("assets_dir", "")

    total_tokens = sum((e.get("usage") or {}).get("total_tokens") or 0 for e in entries)
    ok_calls = sum(1 for e in entries if e.get("status") == "ok")
    cached_calls = sum(1 for e in entries if e.get("status") == "cache_hit")
    err_calls = len(entries) - ok_calls - cached_calls
    ts = [e.get("ts") for e in entries if e.get("ts")]
    wall = (max(ts) - min(ts)) if len(ts) >= 2 else 0.0

    global_b64 = _global_image_embed(viz)
    global_bboxes = {k: v.get("bbox") for k, v in crop_decisions.items() if v.get("bbox")}

    # final anomalies table
    if anomalies:
        rows = []
        for a in anomalies:
            rows.append([html.escape(str(a.get(k, ""))) for k in ("Name", "Observed Phenomenon", "Reasoning", "Severity Score")])
        final_table = (
            "<table><tr><th>#</th><th>Name</th><th>Observed Phenomenon</th><th>Reasoning</th><th>Severity</th></tr>"
            + "".join(
                "<tr>" + f"<td class='sev'>{i}</td>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"
                for i, cells in enumerate(rows, 1)
            )
            + "</table>"
        )
    else:
        final_table = "<div class='mut'>no anomalies in final output</div>"

    # crop decisions table
    if crop_decisions:
        def _crows():
            out = []
            for k, v in crop_decisions.items():
                bbox_s = " / ".join(str(x) for x in v["bbox"]) if v.get("bbox") else "-"
                area = v.get("area_frac")
                area_s = f"{area:.1%}" if area else "-"
                iou = v.get("iou")
                iou_s = f"{iou:.2f}" if iou is not None else "-"
                out.append(
                    "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                    % (
                        html.escape(k), html.escape(str(v.get("status", ""))), html.escape(bbox_s),
                        html.escape(area_s), html.escape(iou_s), html.escape(str(v.get("reason", "") or "-")),
                    )
                )
            return "".join(out)

        crows = _crows()
        crop_table = (
            "<table><tr><th>object</th><th>decision</th><th>bbox [x1,y1,x2,y2] (0-1000 units)</th><th>area</th><th>IoU</th><th>reason</th></tr>"
            + crows + "</table>"
        )
    else:
        crop_table = ""

    cards = []
    shown_global = False
    for i, e in enumerate(entries, 1):
        cards.append(render_step_card(i, e, run_dir, assets_dir, global_b64, global_bboxes, show_global=not shown_global))
        if (e.get("input") or {}).get("images") and any(im.get("kind") == "global" for im in (e.get("input") or {}).get("images") or []):
            shown_global = True

    toc = "".join(
        f'<a href="#step{i}">#{i:02d} {html.escape(e.get("step","?"))}'
        + (f" ({html.escape(e.get('object'))})" if e.get("object") else "") + "</a>"
        for i, e in enumerate(entries, 1)
    )

    title = html.escape(image_name)
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>AnomAgent V2 viz - " + title + "</title>"
        f"<style>{CSS}</style></head><body><div class='wrap'>"
        f"<h1>{title}</h1>"
        f"<div class='mut'>{html.escape(str(viz.get('image_path','')))} &nbsp;·&nbsp; "
        f"encoded {info.get('encoded_size')} ({info.get('encoded_format')}, {info.get('encoded_bytes')} bytes) &nbsp;·&nbsp; "
        f"{len(entries)} LLM calls ({ok_calls} live / {cached_calls} cached / {err_calls} err) &nbsp;·&nbsp; "
        f"{total_tokens} tokens{'' if not cached_calls else ' (cached usage from original run)'} &nbsp;·&nbsp; "
        f"wall {wall:.0f}s</div>"
        f"<h2>Final anomalies ({len(anomalies)})</h2>"
        f"<div class='card'>{final_table}</div>"
        + (f"<h2>Crop decisions (accuracy-checked)</h2><div class='card'>{crop_table}</div>" if crop_table else "")
        + f"<h2>Step timeline</h2><div class='card toc'>{toc}</div>"
        + "".join(cards)
        + "</div></body></html>"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="AnomAgent_V2 run directory (contains every_steps/).")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sample", help="visualize one sample (basename without extension, matches every_steps/<name>.viz.json)")
    g.add_argument("--samples", help="comma-separated sample names")
    g.add_argument("--all", action="store_true", help="visualize every sample with a viz.json (default)")
    ap.add_argument("--out-dir", default=None, help="output dir (default: <run-dir>/viz_html)")
    args = ap.parse_args()

    es = os.path.join(args.run_dir, "every_steps")
    if not os.path.isdir(es):
        print(f"error: no every_steps/ under {args.run_dir}", file=sys.stderr)
        return 2
    out_dir = args.out_dir or os.path.join(args.run_dir, "viz_html")
    os.makedirs(out_dir, exist_ok=True)

    if args.sample:
        names = [args.sample]
    elif args.samples:
        names = [s.strip() for s in args.samples.split(",") if s.strip()]
    else:
        names = sorted(fn[:-len(".viz.json")] for fn in os.listdir(es) if fn.endswith(".viz.json"))
    if not names:
        print("error: no samples found (no *.viz.json under every_steps/)", file=sys.stderr)
        return 2

    index_rows = []
    for name in names:
        viz_path = os.path.join(es, f"{name}.viz.json")
        if not os.path.exists(viz_path):
            print(f"skip {name}: no {viz_path}", file=sys.stderr)
            continue
        with open(viz_path, "r", encoding="utf-8") as f:
            viz = json.load(f)
        html_text = render_sample(args.run_dir, viz)
        out_path = os.path.join(out_dir, f"{name}.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_text)
        n_entries = len(viz.get("entries") or [])
        toks = sum((e.get("usage") or {}).get("total_tokens") or 0 for e in viz.get("entries") or [])
        n_ann = len((viz.get("final") or {}).get("anomalies") or [])
        index_rows.append(
            f"<tr><td><a href='{html.escape(name)}.html'>{html.escape(name)}</a></td>"
            f"<td>{n_entries}</td><td>{toks}</td><td>{n_ann}</td><td>{os.path.getsize(out_path)/1048576:.1f} MB</td></tr>"
        )
        print(f"wrote {out_path}")
    if index_rows:
        idx = (
            "<!doctype html><html><head><meta charset='utf-8'><title>AnomAgent V2 viz index</title>"
            f"<style>{CSS}</style></head><body><div class='wrap'><h1>AnomAgent V2 - sample index</h1>"
            "<table><tr><th>sample</th><th>steps</th><th>tokens</th><th>anomalies</th><th>html size</th></tr>"
            + "".join(index_rows)
            + "</table></div></body></html>"
        )
        with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(idx)
        print(f"index: {os.path.join(out_dir, 'index.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
