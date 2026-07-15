"""Build a self-contained static HTML dashboard from an evaluation report."""

# ruff: noqa: E501  (inline HTML/CSS/JavaScript is intentionally kept verbatim)

from __future__ import annotations

import base64
import json
from pathlib import Path

_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Transformer Variant Lab — Evaluation Dashboard</title>
<style>
:root{--bg:#071018;--panel:#0d1a24;--panel2:#112431;--line:#203846;--text:#edf7f7;
--muted:#93abb4;--cyan:#5de4c7;--gold:#ffc857;--red:#ff6b6b;--blue:#6ea8fe;--r:16px}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(
circle at 82% 2%,#123445 0,transparent 32%),var(--bg);color:var(--text);font:15px/1.55
Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
a{color:inherit}.shell{display:grid;grid-template-columns:230px minmax(0,1fr);min-height:100vh}
aside{position:sticky;top:0;height:100vh;padding:28px 20px;border-right:1px solid var(--line);
background:#071018e8;backdrop-filter:blur(18px)}.brand{font-weight:800;letter-spacing:.04em;font-size:18px}
.brand span{display:block;color:var(--cyan);font:12px/1.4 ui-monospace,monospace;margin-top:5px}
nav{display:grid;gap:7px;margin-top:38px}nav a{text-decoration:none;color:var(--muted);padding:9px 11px;
border-radius:10px}nav a:hover{color:var(--text);background:var(--panel2)}.status{position:absolute;bottom:25px;
font-size:12px;color:var(--muted)}.dot{display:inline-block;width:8px;height:8px;background:var(--cyan);
border-radius:50%;box-shadow:0 0 14px var(--cyan);margin-right:7px}main{padding:42px clamp(24px,5vw,72px) 80px;
max-width:1600px;width:100%}.eyebrow{color:var(--cyan);text-transform:uppercase;letter-spacing:.16em;
font:700 12px ui-monospace,monospace}h1{font-size:clamp(34px,5vw,68px);line-height:1.02;margin:12px 0 18px;
max-width:900px;letter-spacing:-.04em}.lede{color:var(--muted);max-width:760px;font-size:18px}.section{padding-top:82px}
h2{font-size:30px;letter-spacing:-.02em;margin:0 0 8px}.section-intro{color:var(--muted);margin:0 0 26px}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}.card{background:linear-gradient(145deg,var(--panel2),
var(--panel));border:1px solid var(--line);border-radius:var(--r);padding:20px;box-shadow:0 18px 55px #0004}
.metric{grid-column:span 3}.metric .label{color:var(--muted);font-size:12px;text-transform:uppercase;
letter-spacing:.1em}.metric .value{font-size:29px;font-weight:760;margin-top:8px}.metric .sub{font-size:12px;
color:var(--muted);margin-top:4px}.wide{grid-column:span 8}.side{grid-column:span 4}.full{grid-column:1/-1}
.toolbar{display:flex;gap:10px;align-items:center;margin-bottom:18px;flex-wrap:wrap}select{background:#07131c;
border:1px solid var(--line);border-radius:10px;color:var(--text);padding:9px 34px 9px 12px}.bars{display:grid;gap:12px}
.bar-row{display:grid;grid-template-columns:145px 1fr 155px;gap:12px;align-items:center}.bar-name{font-weight:650;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.track{height:11px;background:#07131c;border-radius:99px;overflow:hidden}
.fill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--blue),var(--cyan));min-width:3px}
.bar-value{text-align:right;font:12px ui-monospace,monospace;color:var(--muted)}table{width:100%;border-collapse:collapse}
th{text-align:left;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.09em;padding:10px 12px;
border-bottom:1px solid var(--line)}td{padding:12px;border-bottom:1px solid #20384699}tr:last-child td{border:0}
.pill{display:inline-flex;border:1px solid var(--line);border-radius:99px;padding:4px 9px;font-size:11px;color:var(--muted)}
.pill.good{border-color:#2d806e;color:var(--cyan)}.pill.bad{border-color:#8f4545;color:#ff9e9e}.probe-grid{display:grid;
grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.probe{padding:15px;background:#08151e;border:1px solid var(--line);
border-radius:12px}.probe strong{display:block;font-size:18px;margin:5px 0}.spark{width:100%;height:55px;margin-top:10px}
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}.figure{margin:0;overflow:hidden;
background:var(--panel);border:1px solid var(--line);border-radius:var(--r)}.figure img{display:block;width:100%;background:white}
.figure figcaption{padding:13px 16px;color:var(--muted);font-size:13px}.empty{padding:24px;color:var(--muted);text-align:center;
border:1px dashed var(--line);border-radius:12px}.meta{font:12px/1.7 ui-monospace,monospace;color:var(--muted);white-space:pre-wrap}
footer{padding-top:72px;color:var(--muted);font-size:12px}@media(max-width:900px){.shell{display:block}aside{position:relative;
height:auto;border-right:0;border-bottom:1px solid var(--line)}nav{display:flex;overflow:auto;margin-top:18px}.status{display:none}
main{padding-top:30px}.metric{grid-column:span 6}.wide,.side{grid-column:1/-1}}@media(max-width:580px){.metric{grid-column:1/-1}
.bar-row{grid-template-columns:90px 1fr}.bar-value{grid-column:2}.gallery{grid-template-columns:1fr}}
</style>
</head>
<body><div class="shell"><aside><div class="brand">Transformer Variant Lab<span>STATIC REPORT / OFFLINE</span></div>
<nav><a href="#overview">Overview</a><a href="#comparisons">Comparisons</a><a href="#variants">Variants</a>
<a href="#benchmarks">Inference</a><a href="#probes">Probes</a><a href="#artifacts">Artifacts</a><a href="#provenance">Provenance</a></nav>
<div class="status"><span class="dot"></span>Report embedded</div></aside><main>
<header id="overview"><div class="eyebrow">Controlled architecture study</div><h1>Transformer variants,<br>measured honestly.</h1>
<p class="lede">A serverless view of fixed-data, wall-clock, FLOP, representation, and retrieval results. Every value is embedded in this file.</p>
<div class="grid" id="headline"></div></header>
<section class="section" id="comparisons"><h2>Comparison axes</h2><p class="section-intro">Lower validation loss is better. Error ranges are sample standard deviations across seeds.</p>
<section class="section" id="benchmarks"><h2>Inference & long context</h2><p class="section-intro">Generation throughput, persistent cache storage, and held-out context extrapolation. Unsupported paths remain visible.</p>
<div class="card full" style="overflow:auto"><table><thead><tr><th>Variant</th><th>Uncached tok/s</th><th>Cached tok/s</th><th>KV cache</th><th>1K PPL</th><th>2K PPL</th><th>4K PPL</th><th>4K prefill tok/s</th></tr></thead><tbody id="benchmarkRows"></tbody></table></div>
<div class="meta" id="benchmarkLimitations" style="margin-top:14px"></div></section>
<div class="card full"><div class="toolbar"><label for="axis">Axis</label><select id="axis"><option value="fixed_data">Fixed data</option>
<option value="fixed_wallclock">Fixed wall-clock (100%)</option><option value="fixed_flops">Fixed FLOPs</option></select></div><div class="bars" id="bars"></div></div></section>
<section class="section" id="variants"><h2>Variant results</h2><p class="section-intro">Seed-aware final metrics and parameter counts.</p>
<div class="card full" style="overflow:auto"><table><thead><tr><th>Variant</th><th>Val loss</th><th>Perplexity</th><th>ICL α</th><th>Active params</th><th>Total params</th><th>Seeds</th></tr></thead><tbody id="variantRows"></tbody></table></div></section>
<section class="section" id="probes"><h2>Diagnostic probes</h2><p class="section-intro">Aggregated retrieval and representation diagnostics; per-seed provenance remains in the embedded data.</p><div class="probe-grid" id="probeGrid"></div></section>
<section class="section" id="artifacts"><h2>Report artifacts</h2><p class="section-intro">Publication plots embedded as data URIs for true offline use.</p><div class="gallery" id="gallery"></div></section>
<section class="section" id="provenance"><h2>Provenance & limitations</h2><p class="section-intro">Hardware, report time, schema, and experimental validity.</p>
<div class="grid"><div class="card wide"><div class="meta" id="metadata"></div></div><div class="card side" id="parity"></div></div></section>
<footer>Generated by <code>scripts/build_dashboard.py</code>. No server, CDN, tracking, or external runtime.</footer>
</main></div>
<script type="application/json" id="report-data">__REPORT_JSON__</script>
<script type="application/json" id="report-meta">__METADATA_JSON__</script>
<script type="application/json" id="plot-data">__PLOTS_JSON__</script>
<script>
const data=JSON.parse(document.getElementById('report-data').textContent);
const meta=JSON.parse(document.getElementById('report-meta').textContent);
const plots=JSON.parse(document.getElementById('plot-data').textContent);
const cmp=data.comparison||{}, agg=data.aggregated||{}, variants=data.variants||{};
const estimate=v=>typeof v==='number'?{mean:v,std:null,n:1}:(Array.isArray(v)?{mean:v[0],std:v[1],n:null}:v||{});
const fmt=(v,d=4)=>v==null||Number.isNaN(Number(v))?'—':Number(v).toFixed(d);
const metric=(v,d=4)=>{const e=estimate(v);return `${fmt(e.mean,d)}${e.std==null?'':` ± ${fmt(e.std,d)}`}`};
const names=Object.keys(variants).sort();const fixed=cmp.fixed_data||{};
const ranked=Object.entries(fixed).map(([n,v])=>[n,estimate(v).mean]).filter(x=>x[1]!=null).sort((a,b)=>a[1]-b[1]);
const cards=[['Variants',names.length,'architectures'],['Best fixed-data',ranked[0]?fmt(ranked[0][1]):'—',ranked[0]?.[0]||'no data'],
['Pareto front',(cmp.pareto_front||[]).length,(cmp.pareto_front||[]).join(', ')||'none'],['Schema',`v${data.schema_version||1}`,'embedded JSON contract']];
document.getElementById('headline').innerHTML=cards.map(c=>`<div class="card metric"><div class="label">${c[0]}</div><div class="value">${c[1]}</div><div class="sub">${c[2]}</div></div>`).join('');
function axisData(){const key=document.getElementById('axis').value;const raw=cmp[key]||{};if(key==='fixed_wallclock')
return Object.fromEntries(Object.entries(raw).map(([n,v])=>[n,v['1.0']||v[1]||Object.values(v).at(-1)]));return raw}
function renderBars(){const rows=Object.entries(axisData()).map(([n,v])=>[n,estimate(v)]).filter(x=>x[1].mean!=null).sort((a,b)=>a[1].mean-b[1].mean);
const values=rows.map(x=>x[1].mean),lo=Math.min(...values),hi=Math.max(...values),span=hi-lo||1;document.getElementById('bars').innerHTML=rows.length?rows.map(([n,e])=>
`<div class="bar-row"><div class="bar-name">${n}</div><div class="track"><div class="fill" style="width:${10+90*(hi-e.mean)/span}%"></div></div><div class="bar-value">${metric(e)} · n=${e.n??'—'}</div></div>`).join(''):'<div class="empty">No data for this axis.</div>'}
document.getElementById('axis').addEventListener('change',renderBars);renderBars();
document.getElementById('variantRows').innerHTML=names.map(n=>{const a=agg[n]||{},count=(variants[n]||[]).length,p=cmp.parameter_counts?.[n],t=cmp.total_parameter_counts?.[n]??p;return `<tr><td><strong>${n}</strong>${(cmp.pareto_front||[]).includes(n)?' <span class="pill good">Pareto</span>':''}</td><td>${metric(a.val_loss)}</td><td>${metric(a.perplexity,2)}</td><td>${metric(a.icl_alpha,3)}</td><td>${p?Number(p).toLocaleString():'—'}</td><td>${t?Number(t).toLocaleString():'—'}</td><td>${count}</td></tr>`}).join('');
const probeAgg=data.probes?.aggregated||{};function spark(values){if(!values?.length)return'';const w=220,h=50,min=Math.min(...values),max=Math.max(...values),s=max-min||1;
const pts=values.map((v,i)=>`${i*w/Math.max(1,values.length-1)},${h-4-(v-min)*(h-8)/s}`).join(' ');return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><polyline fill="none" stroke="#5de4c7" stroke-width="2" points="${pts}"/></svg>`}
const average=values=>values?.length?values.flat(Infinity).reduce((a,b)=>a+Number(b),0)/values.flat(Infinity).length:null;
const benchmarks=data.benchmarks||{},benchVariants=benchmarks.variants||{};const statusValue=(entry,key,d=1)=>entry?.status==='ok'?fmt(entry[key],d):entry?.status||'—';
document.getElementById('benchmarkRows').innerHTML=Object.keys(benchVariants).length?Object.entries(benchVariants).map(([n,b])=>{const g=b.generation||{},lc=b.long_context||{},cache=g.kv_cache;return `<tr><td><strong>${n}</strong></td><td>${statusValue(g.uncached,'tokens_per_second')}</td><td>${statusValue(g.cached,'tokens_per_second')}</td><td>${cache?.status==='ok'?(cache.bytes/1048576).toFixed(1)+' MiB':cache?.status||'—'}</td><td>${statusValue(lc['1024'],'perplexity',2)}</td><td>${statusValue(lc['2048'],'perplexity',2)}</td><td>${statusValue(lc['4096'],'perplexity',2)}</td><td>${statusValue(lc['4096'],'prefill_tokens_per_second')}</td></tr>`}).join(''):'<tr><td colspan="8"><div class="empty">Run scripts/benchmark_inference.py to populate this section.</div></td></tr>';
document.getElementById('benchmarkLimitations').textContent=(benchmarks.limitations||[]).join('\n');
document.getElementById('probeGrid').innerHTML=Object.keys(probeAgg).length?Object.entries(probeAgg).map(([n,p])=>`<div class="probe"><span class="pill">${n} · n=${p.n}</span><strong>MQAR ${metric({mean:p.mqar?.accuracy,std:p.mqar?.accuracy_std},3)}</strong><span>Stable rank ${metric({mean:p.stable_rank?.mean,std:p.stable_rank?.std},2)}</span><br><span>curve σ: rank ${fmt(average(p.stable_rank?.per_layer_std),2)} · CKA ${fmt(average(p.cka?.adjacent_curve_std),3)} · entropy ${fmt(average(p.attention_entropy?.per_layer_std),3)}</span>${spark(p.stable_rank?.per_layer)}</div>`).join(''):'<div class="empty">Probe data was not serialized for this report.</div>';
document.getElementById('gallery').innerHTML=plots.length?plots.map(p=>`<figure class="figure"><img loading="lazy" src="${p.data_uri}" alt="${p.title}"><figcaption>${p.title}</figcaption></figure>`).join(''):'<div class="empty">No plot files found.</div>';
document.getElementById('metadata').textContent=JSON.stringify({...meta,schema_version:data.schema_version,variant_count:names.length},null,2);
const parity=!!cmp.parameter_parity_valid;document.getElementById('parity').innerHTML=`<span class="pill ${parity?'good':'bad'}">${parity?'PASS':'DOCUMENTED LIMITATION'}</span><h3>Active parameter parity</h3><p style="color:var(--muted)">${parity?'Active parameters per token are within the declared tolerance.':'Active parameters exceed ±5%; total MoE capacity is reported separately and no compensating retrain was performed.'}</p>`;
</script></body></html>"""


def _safe_json(value: object) -> str:
    """Encode JSON safely inside an ``application/json`` script element."""
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")


def _embedded_plots(plots_dir: Path) -> list[dict[str, str]]:
    plots = []
    if not plots_dir.exists():
        return plots
    for path in sorted(plots_dir.glob("*.png")):
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        plots.append(
            {
                "title": path.stem.replace("_", " ").title(),
                "data_uri": f"data:image/png;base64,{encoded}",
            }
        )
    return plots


def build_dashboard(report_dir: str | Path, output_path: str | Path | None = None) -> Path:
    """Build one offline HTML dashboard from a versioned evaluation report."""
    report_dir = Path(report_dir)
    metrics_path = report_dir / "raw" / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Evaluation metrics not found: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    missing = {"variants", "aggregated", "comparison"} - metrics.keys()
    if missing:
        raise ValueError(f"Evaluation report is missing required keys: {sorted(missing)}")

    metadata_path = report_dir / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    benchmarks_path = report_dir / "raw" / "benchmarks.json"
    if benchmarks_path.is_file():
        metrics["benchmarks"] = json.loads(
            benchmarks_path.read_text(encoding="utf-8")
        )
    plots = _embedded_plots(report_dir / "plots")
    html = (
        _HTML.replace("__REPORT_JSON__", _safe_json(metrics))
        .replace("__METADATA_JSON__", _safe_json(metadata))
        .replace("__PLOTS_JSON__", _safe_json(plots))
    )

    output = Path(output_path) if output_path is not None else report_dir / "index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output
