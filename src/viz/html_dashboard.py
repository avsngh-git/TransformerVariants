"""Build a self-contained static HTML dashboard from an evaluation report."""

# ruff: noqa: E501  (inline HTML/CSS/JavaScript is intentionally kept verbatim)

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

from src.evaluation.benchmarks import rank_long_context_variants

_PLOT_SUMMARIES = {
    "cka_adjacent": "Compares representation similarity between neighboring layers. Higher CKA indicates that consecutive layers transform the residual stream less, while lower values indicate larger representational changes.",
    "flop_breakdown": "Breaks each variant's estimated training compute into projections, attention, and feed-forward work so architectural savings and dominant costs are visible.",
    "learning_curves_flops": "Tracks validation loss against cumulative model FLOPs. Curves that reach a lower loss further left are more compute-efficient.",
    "learning_curves_tokens": "Tracks validation loss against the number of training tokens. Because every variant sees the same data budget, lower curves indicate better sample efficiency.",
    "learning_curves_wallclock": "Tracks validation loss against elapsed training time. Lower curves at the same time budget indicate better end-to-end hardware efficiency.",
    "mqar_by_distance": "Shows associative-recall accuracy as the key-to-query distance grows. A flatter, higher curve indicates that retrieval survives over longer spans.",
    "pareto_flops_val_loss": "Places validation loss against cumulative FLOPs. Pareto-front points are not dominated by another variant that is both cheaper and more accurate.",
    "pareto_peak_memory_val_loss": "Places validation loss against peak memory use. Pareto-front points offer the strongest quality-memory trade-offs in this comparison.",
    "pareto_wallclock_val_loss": "Places validation loss against training time. Pareto-front points offer the strongest quality-time trade-offs in this comparison.",
    "per_position_loss": "Shows next-token loss by sequence position. Falling loss indicates useful in-context adaptation; flattening or rising loss reveals positions where additional context stops helping.",
    "roofline": "Compares achieved compute throughput with arithmetic intensity against the hardware roofline. Position relative to the ridge indicates whether a workload is primarily bandwidth- or compute-limited.",
    "stable_rank": "Shows the effective dimensionality of each layer's hidden states. Persistently low or declining stable rank can indicate representational compression or collapse.",
}

_PLOT_GROUPS = {
    "training": (
        0,
        "Training dynamics",
        "Together, these curves distinguish sample efficiency, compute efficiency, and end-to-end time efficiency. A variant can lead on one budget axis without leading on the others.",
    ),
    "efficiency": (
        1,
        "Efficiency and Pareto trade-offs",
        "Together, these plots connect where compute is spent, whether execution is bandwidth- or compute-limited, and which variants remain non-dominated under FLOP, memory, and wall-clock budgets.",
    ),
    "cka": (
        2,
        "Representation similarity (CKA)",
        "The adjacent-layer curve highlights local transitions, while the heatmaps reveal longer-range blocks of representational similarity. Read together, they show both where representations change and where layers may be redundant.",
    ),
    "context": (
        3,
        "Context and representation diagnostics",
        "Together, these diagnostics connect retrieval over distance, position-wise prediction quality, and hidden-state dimensionality. They help separate context-use failures from representational compression.",
    ),
}
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
.plot-stack{display:grid;gap:42px}.plot-group{display:grid;gap:22px}.plot-group-heading h3{font-size:23px;margin:0 0 5px}.plot-group-heading p{color:var(--muted);margin:0}
.plot-panel{margin:0;overflow:hidden;background:var(--panel);border:1px solid var(--line);border-radius:var(--r)}.plot-panel h4{font-size:18px;margin:0;padding:18px 22px;border-bottom:1px solid var(--line)}
.plot-panel img{display:block;width:100%;height:auto;background:white}.visual-summary{padding:16px 20px;color:var(--muted);font-size:14px;line-height:1.65;border-top:1px solid var(--line);background:#091720}
.visual-summary strong{display:block;color:var(--cyan);font-size:11px;text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px}.visual-summary p{margin:0}
.visual-summary.compact{margin-top:12px;padding:10px 0 0;border-top:1px solid var(--line);background:transparent;font-size:12px}.group-summary{border-left:3px solid var(--cyan)}
.visualization{margin:0;padding:0;overflow:hidden}.visualization .lc-chart{padding:20px}.empty{padding:24px;color:var(--muted);text-align:center;
border:1px dashed var(--line);border-radius:12px}.meta{font:12px/1.7 ui-monospace,monospace;color:var(--muted);white-space:pre-wrap}
.lc-heading{display:flex;justify-content:space-between;gap:20px;align-items:end;margin:24px 0 14px}
.lc-heading h3{font-size:22px;margin:0}.lc-heading p{color:var(--muted);max-width:760px;margin:0}
.lc-chart{min-height:430px;overflow:auto}.lc-chart svg{display:block;width:100%;min-width:760px;height:auto}
.lc-grid{stroke:#294352;stroke-width:1}.lc-axis{fill:var(--muted);font:12px ui-monospace,monospace}
.lc-legend{fill:var(--text);font:12px ui-monospace,monospace}.lc-summary{margin-bottom:16px}
.rank-card{grid-column:span 4}.rank-card h4{margin:0 0 4px;font-size:16px}.rank-card ol{margin:14px 0 0;padding-left:24px}
.rank-card li{padding:5px 0;color:var(--muted)}.rank-card li strong{color:var(--text)}.rank-value{float:right;font:12px ui-monospace,monospace}
.provenance-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}.provenance-card{padding:16px;background:#08151e;border:1px solid var(--line);border-radius:12px}
.provenance-card span{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}.provenance-card strong{display:block;margin-top:7px;font-size:17px}
.provenance-details{margin-top:16px}.provenance-details details{border-top:1px solid var(--line);padding:12px 0}.provenance-details summary{cursor:pointer;color:var(--cyan)}.provenance-details ul{color:var(--muted);line-height:1.55}
footer{padding-top:72px;color:var(--muted);font-size:12px}@media(max-width:900px){.shell{display:block}aside{position:relative;
height:auto;border-right:0;border-bottom:1px solid var(--line)}nav{display:flex;overflow:auto;margin-top:18px}.status{display:none}
main{padding-top:30px}.metric{grid-column:span 6}.wide,.side{grid-column:1/-1}}@media(max-width:580px){.metric{grid-column:1/-1}
.bar-row{grid-template-columns:90px 1fr}.bar-value{grid-column:2}}
</style>
</head>
<body><div class="shell"><aside><div class="brand">Transformer Variant Lab<span>STATIC REPORT / OFFLINE</span></div>
<nav><a href="#overview">Overview</a><a href="#comparisons">Comparisons</a><a href="#variants">Variants</a>
<a href="#benchmarks">Inference</a><a href="#probes">Probes</a><a href="#artifacts">Artifacts</a><a href="#provenance">Provenance</a></nav>
<div class="status"><span class="dot"></span>Report embedded</div></aside><main>
<header id="overview"><div class="eyebrow">Controlled architecture study</div><h1>Transformer variants,<br>measured honestly.</h1>
<p class="lede">A serverless view of fixed-data, wall-clock, FLOP, representation, and retrieval results. Every value is embedded in this file.</p>
<div class="grid" id="headline"></div></header>
<section class="section" id="comparisons"><h2>Comparison axes</h2><p class="section-intro">Lower validation loss is better. Error ranges are sample standard deviations across seeds; missing ranges mark incomplete historical diagnostics whose copied logs cannot support independent variability.</p>
<section class="section" id="benchmarks"><h2>Inference & long context</h2><p class="section-intro">Generation throughput, persistent cache storage, and held-out context extrapolation. Unsupported paths remain visible.</p>
<div class="lc-heading"><div><div class="eyebrow">Seed-aware context study</div><h3>Paired tail-token extrapolation</h3></div>
<p>Each curve scores the same final target tokens at every context length. Error bars show sample standard deviation across independent checkpoint seeds.</p></div>
<div class="grid lc-summary" id="longContextSummary"></div>
<div class="grid lc-summary" id="longContextRankings"></div>
<div class="visual-summary group-summary" id="longContextRankingsSummary"></div>
<figure class="card full visualization"><div class="lc-chart" id="longContextChart"></div><figcaption class="visual-summary" id="longContextChartSummary"></figcaption></figure>
<div class="card full" style="overflow:auto"><table><thead><tr><th>Variant</th><th>Uncached tok/s</th><th>Cached tok/s</th><th>KV cache</th><th>1K PPL</th><th>2K PPL</th><th>4K PPL</th><th>4K prefill tok/s</th></tr></thead><tbody id="benchmarkRows"></tbody></table></div>
<div class="meta" id="benchmarkLimitations" style="margin-top:14px"></div></section>
<div class="card full"><div class="toolbar"><label for="axis">Axis</label><select id="axis"><option value="fixed_data">Fixed data</option>
<option value="fixed_wallclock">Fixed wall-clock (100%)</option><option value="fixed_flops">Fixed FLOPs</option></select></div><div class="bars" id="bars"></div><div class="visual-summary" id="axisSummary"></div></div></section>
<section class="section" id="variants"><h2>Variant results</h2><p class="section-intro">Seed-aware final metrics and parameter counts.</p>
<div class="card full" style="overflow:auto"><table><thead><tr><th>Variant</th><th>Val loss</th><th>Perplexity</th><th>ICL α</th><th>Active params</th><th>Total params</th><th>Seeds</th></tr></thead><tbody id="variantRows"></tbody></table></div></section>
<section class="section" id="probes"><h2>Diagnostic probes</h2><p class="section-intro">Aggregated retrieval and representation diagnostics; per-seed provenance remains in the embedded data.</p><div class="probe-grid" id="probeGrid"></div></section>
<section class="section" id="artifacts"><h2>Report artifacts</h2><p class="section-intro">Publication plots are grouped by question, shown separately, and interpreted directly below each figure.</p><div class="plot-stack" id="gallery">__PLOT_HTML__</div></section>
<section class="section" id="provenance"><h2>Provenance & limitations</h2><p class="section-intro">Hardware, report time, schema, and experimental validity.</p>
<div class="grid"><div class="card wide">__PROVENANCE_HTML__</div><div class="card side" id="parity"></div></div></section>
<footer>Generated by <code>scripts/build_dashboard.py</code>. No server, CDN, tracking, or external runtime.</footer>
</main></div>
<script type="application/json" id="report-data">__REPORT_JSON__</script>
<script>
const data=JSON.parse(document.getElementById('report-data').textContent);
const cmp=data.comparison||{}, agg=data.aggregated||{}, variants=data.variants||{};
const estimate=v=>typeof v==='number'?{mean:v,std:null,n:1}:(Array.isArray(v)?{mean:v[0],std:v[1],n:null}:v||{});
const fmt=(v,d=4)=>v==null||Number.isNaN(Number(v))?'—':Number(v).toFixed(d);
const metric=(v,d=4)=>{const e=estimate(v);return `${fmt(e.mean,d)}${e.std==null?'':` ± ${fmt(e.std,d)}`}`};
const names=Object.keys(variants).sort();const fixed=cmp.fixed_data||{};
const ranked=Object.entries(fixed).map(([n,v])=>[n,estimate(v).mean]).filter(x=>x[1]!=null).sort((a,b)=>a[1]-b[1]);
const cards=[['Variants',names.length,'architectures'],['Best fixed-data',ranked[0]?fmt(ranked[0][1]):'—',ranked[0]?.[0]||'no data'],
['Pareto front',(cmp.pareto_front||[]).length,(cmp.pareto_front||[]).join(', ')||'none'],['Schema',`v${data.schema_version||1}`,'embedded JSON contract']];
document.getElementById('headline').innerHTML=cards.map(c=>`<div class="card metric"><div class="label">${c[0]}</div><div class="value">${c[1]}</div><div class="sub">${c[2]}</div></div>`).join('');
const axisLabels={fixed_data:'the shared token budget',fixed_wallclock:'the shared wall-clock budget',fixed_flops:'the shared FLOP budget'};
function axisData(){const key=document.getElementById('axis').value;const raw=cmp[key]||{};if(key==='fixed_wallclock')
return Object.fromEntries(Object.entries(raw).map(([n,v])=>[n,v['1.0']||v[1]||Object.values(v).at(-1)]));return raw}
function renderBars(){const axisKey=document.getElementById('axis').value;
const rows=Object.entries(axisData()).map(([n,v])=>[n,estimate(v)]).filter(x=>x[1].mean!=null).sort((a,b)=>a[1].mean-b[1].mean);
const values=rows.map(x=>x[1].mean),lo=Math.min(...values),hi=Math.max(...values),span=hi-lo||1;document.getElementById('bars').innerHTML=rows.length?rows.map(([n,e])=>
`<div class="bar-row"><div class="bar-name">${n}</div><div class="track"><div class="fill" style="width:${10+90*(hi-e.mean)/span}%"></div></div><div class="bar-value">${metric(e)} · n=${e.n??'—'}</div></div>`).join(''):'<div class="empty">No data for this axis.</div>';
const best=rows[0],worst=rows[rows.length-1];document.getElementById('axisSummary').innerHTML=best?`<strong>What this shows</strong><p>At ${axisLabels[axisKey]}, <b>${best[0]}</b> has the lowest validation loss (${metric(best[1])}). ${rows.length>1?`The displayed spread to ${worst[0]} is ${fmt(worst[1].mean-best[1].mean)} loss.`:''} Lower is better.</p>`:'<strong>What this shows</strong><p>No statistically valid values are available for this comparison axis.</p>'}
document.getElementById('axis').addEventListener('change',renderBars);renderBars();
document.getElementById('variantRows').innerHTML=names.map(n=>{const a=agg[n]||{},count=(variants[n]||[]).length,p=cmp.parameter_counts?.[n],t=cmp.total_parameter_counts?.[n]??p;return `<tr><td><strong>${n}</strong>${(cmp.pareto_front||[]).includes(n)?' <span class="pill good">Pareto</span>':''}</td><td>${metric(a.val_loss)}</td><td>${metric(a.perplexity,2)}</td><td>${metric(a.icl_alpha,3)}</td><td>${p?Number(p).toLocaleString():'—'}</td><td>${t?Number(t).toLocaleString():'—'}</td><td>${count}</td></tr>`}).join('');
const probeAgg=data.probes?.aggregated||{};function spark(values){if(!values?.length)return'';const w=220,h=50,min=Math.min(...values),max=Math.max(...values),s=max-min||1;
const pts=values.map((v,i)=>`${i*w/Math.max(1,values.length-1)},${h-4-(v-min)*(h-8)/s}`).join(' ');return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><polyline fill="none" stroke="#5de4c7" stroke-width="2" points="${pts}"/></svg>`}
const average=values=>values?.length?values.flat(Infinity).reduce((a,b)=>a+Number(b),0)/values.flat(Infinity).length:null;
const benchmarks=data.benchmarks||{},benchVariants=benchmarks.variants||{};const statusValue=(entry,key,d=1)=>entry?.status==='ok'?fmt(entry[key],d):entry?.status||'—';
const isContextMeasurement=entry=>entry&&['ok','partial'].includes(entry.status);
const contextLengths=(benchmarks.settings?.context_lengths||[1024,2048,4096]).map(String);
const longestContext=contextLengths.at(-1);
const variantBenchmarks=Object.entries(benchVariants);
const contextEstimate=(measurement,metricKey)=>isContextMeasurement(measurement)?estimate(measurement[metricKey]):{};
const rankings=benchmarks.long_context_rankings||{};
const qualityRanking=rankings.quality||[],retentionRanking=rankings.retention||[],throughputRanking=rankings.throughput||[];
const benchmarkSettings=benchmarks.settings||{},bestQuality=qualityRanking[0],bestRetention=retentionRanking[0];
const longContextCards=[
['Best '+Number(longestContext).toLocaleString()+'-token quality',bestQuality?metric(bestQuality.estimate,2):'—',bestQuality?.variant||'no supported result'],
['Best context retention',bestRetention?metric(bestRetention.estimate,3)+'x':'—',bestRetention?.variant||'no paired result'],
['Evidence unit',(bestQuality?.estimate.n??0)+' seeds',(benchmarkSettings.long_context_windows_per_checkpoint??'—')+' windows per checkpoint']
];
document.getElementById('longContextSummary').innerHTML=longContextCards.map(card=>`<div class="card metric"><div class="label">${card[0]}</div><div class="value">${card[1]}</div><div class="sub">${card[2]}</div></div>`).join('');
const rankingDefinitions=[
['4K quality','quality','lower tail perplexity',2,''],
['Context retention','retention','ratio closest to 1.0',3,'x'],
['4K prefill','throughput','higher tokens/second',0,' tok/s']
];
document.getElementById('longContextRankings').innerHTML=rankingDefinitions.map(([title,rankingKey,subtitle,digits,suffix])=>{const entries=rankings[rankingKey]||[];return `<div class="card rank-card"><h4>${title}</h4><span class="pill">${subtitle}</span><ol>${entries.map(entry=>`<li><strong>${entry.variant}</strong><span class="rank-value">${metric(entry.estimate,digits)}${suffix}</span></li>`).join('')||'<li>No supported results</li>'}</ol></div>`}).join('');
const bestThroughput=throughputRanking[0];
document.getElementById('longContextRankingsSummary').innerHTML=bestQuality?`<strong>Combined reading</strong><p><b>${bestQuality.variant}</b> leads absolute ${Number(longestContext).toLocaleString()}-token quality, <b>${bestRetention?.variant||'—'}</b> preserves native-context perplexity most closely, and <b>${bestThroughput?.variant||'—'}</b> has the highest long-context prefill throughput. These are separate rankings because quality, stability, and speed answer different questions.</p>`:'<strong>Combined reading</strong><p>No supported long-context rankings are available.</p>';
document.getElementById('longContextChartSummary').innerHTML=bestQuality?`<strong>What this shows</strong><p>The curves show mean tail-token validation loss as available context grows to ${Number(longestContext).toLocaleString()} tokens; vertical bars are sample standard deviations across checkpoint seeds. <b>${bestQuality.variant}</b> reaches the lowest final tail perplexity (${metric(bestQuality.estimate,2)}), while flatter curves indicate stronger extrapolation stability.</p>`:'<strong>What this shows</strong><p>No supported multi-seed long-context curves are available.</p>';

function renderLongContextChart(){
const colors=['#5de4c7','#ffc857','#6ea8fe','#ff6b6b','#c792ea','#82aaff','#f78c6c','#89ddff','#c3e88d','#f07178'];
const chartSeries=variantBenchmarks.map(([variantName,variantBenchmark],colorIndex)=>({variantName,color:colors[colorIndex%colors.length],points:contextLengths.map((contextLength,contextIndex)=>({contextIndex,contextLength,estimate:contextEstimate(variantBenchmark.long_context?.[contextLength],'val_loss')})).filter(point=>point.estimate.mean!=null)})).filter(seriesEntry=>seriesEntry.points.length);
const errorBounds=chartSeries.flatMap(seriesEntry=>seriesEntry.points.flatMap(point=>[point.estimate.mean-(point.estimate.std||0),point.estimate.mean+(point.estimate.std||0)]));
if(!errorBounds.length){document.getElementById('longContextChart').innerHTML='<div class="empty">Run the multi-seed long-context benchmark to populate this chart.</div>';return}
const width=1050,height=450,left=62,right=220,top=28,bottom=58,plotWidth=width-left-right,plotHeight=height-top-bottom;
const minLoss=Math.min(...errorBounds),maxLoss=Math.max(...errorBounds),padding=Math.max(.05,(maxLoss-minLoss)*.08),lowerLoss=minLoss-padding,upperLoss=maxLoss+padding,lossSpan=upperLoss-lowerLoss||1;
const xForContext=contextIndex=>left+(contextLengths.length===1?plotWidth/2:contextIndex*plotWidth/(contextLengths.length-1));
const yForLoss=loss=>top+(upperLoss-loss)*plotHeight/lossSpan;
let chartSvg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Mean tail-token validation loss by context length with seed standard-deviation error bars">`;
for(let gridIndex=0;gridIndex<=5;gridIndex++){const loss=lowerLoss+gridIndex*lossSpan/5,gridY=yForLoss(loss);chartSvg+=`<line class="lc-grid" x1="${left}" x2="${left+plotWidth}" y1="${gridY}" y2="${gridY}"/><text class="lc-axis" x="${left-10}" y="${gridY+4}" text-anchor="end">${fmt(loss,2)}</text>`}
contextLengths.forEach((contextLength,contextIndex)=>{chartSvg+=`<text class="lc-axis" x="${xForContext(contextIndex)}" y="${height-24}" text-anchor="middle">${Number(contextLength).toLocaleString()}</text>`});
chartSvg+=`<text class="lc-axis" x="${left+plotWidth/2}" y="${height-4}" text-anchor="middle">available context tokens</text><text class="lc-axis" transform="translate(15 ${top+plotHeight/2}) rotate(-90)" text-anchor="middle">tail validation loss (lower is better)</text>`;
chartSeries.forEach((seriesEntry,legendIndex)=>{if(seriesEntry.points.length>1)chartSvg+=`<polyline fill="none" stroke="${seriesEntry.color}" stroke-width="2.5" points="${seriesEntry.points.map(point=>xForContext(point.contextIndex)+','+yForLoss(point.estimate.mean)).join(' ')}"/>`;seriesEntry.points.forEach(point=>{const pointX=xForContext(point.contextIndex),pointY=yForLoss(point.estimate.mean),error=point.estimate.std||0,errorTop=yForLoss(point.estimate.mean+error),errorBottom=yForLoss(point.estimate.mean-error);chartSvg+=`<line x1="${pointX}" x2="${pointX}" y1="${errorTop}" y2="${errorBottom}" stroke="${seriesEntry.color}"/><line x1="${pointX-5}" x2="${pointX+5}" y1="${errorTop}" y2="${errorTop}" stroke="${seriesEntry.color}"/><line x1="${pointX-5}" x2="${pointX+5}" y1="${errorBottom}" y2="${errorBottom}" stroke="${seriesEntry.color}"/><circle cx="${pointX}" cy="${pointY}" r="4.5" fill="${seriesEntry.color}"><title>${seriesEntry.variantName}: ${point.contextLength} tokens, loss ${metric(point.estimate,3)}</title></circle>`});const legendY=top+legendIndex*30;chartSvg+=`<line x1="${left+plotWidth+26}" x2="${left+plotWidth+48}" y1="${legendY}" y2="${legendY}" stroke="${seriesEntry.color}" stroke-width="3"/><text class="lc-legend" x="${left+plotWidth+56}" y="${legendY+4}">${seriesEntry.variantName}</text>`});
document.getElementById('longContextChart').innerHTML=chartSvg+'</svg>'}
renderLongContextChart();
const metricStatus=(measurement,metricKey,digits=2)=>isContextMeasurement(measurement)?metric(measurement[metricKey],digits):measurement?.status||'—';
const qualityPosition=new Map(qualityRanking.map(entry=>[entry.variant,entry.rank]));
const sortedBenchmarks=variantBenchmarks.sort(([leftName],[rightName])=>(qualityPosition.get(leftName)??Infinity)-(qualityPosition.get(rightName)??Infinity)||leftName.localeCompare(rightName));
document.querySelector('#benchmarks table thead').innerHTML='<tr><th>Rank</th><th>Variant</th><th>Uncached tok/s</th><th>Cached tok/s</th><th>KV cache</th><th>1K tail PPL</th><th>2K tail PPL</th><th>4K tail PPL</th><th>4K PPL ratio</th><th>4K prefill tok/s</th><th>Seeds</th></tr>';
document.getElementById('benchmarkRows').innerHTML=sortedBenchmarks.length?sortedBenchmarks.map(([variantName,variantBenchmark])=>{const generation=variantBenchmark.generation||{},longContext=variantBenchmark.long_context||{},cache=generation.kv_cache,longestMeasurement=longContext[longestContext];return `<tr><td>${qualityPosition.get(variantName)??'—'}</td><td><strong>${variantName}</strong></td><td>${statusValue(generation.uncached,'tokens_per_second')}</td><td>${statusValue(generation.cached,'tokens_per_second')}</td><td>${cache?.status==='ok'?(cache.bytes/1048576).toFixed(1)+' MiB':cache?.status||'—'}</td><td>${metricStatus(longContext['1024'],'perplexity')}</td><td>${metricStatus(longContext['2048'],'perplexity')}</td><td>${metricStatus(longContext['4096'],'perplexity')}</td><td>${metricStatus(longestMeasurement,'perplexity_ratio')}</td><td>${metricStatus(longestMeasurement,'prefill_tokens_per_second',0)}</td><td>${contextEstimate(longestMeasurement,'val_loss').n??'—'}</td></tr>`}).join(''):'<tr><td colspan="11"><div class="empty">No benchmark data.</div></td></tr>';
const method=benchmarks.long_context_method||{};
document.getElementById('benchmarkLimitations').textContent=[...(benchmarks.limitations||[]),'','Method: '+Object.values(method).join('; ')].join('\n');
document.getElementById('probeGrid').innerHTML=Object.keys(probeAgg).length?Object.entries(probeAgg).map(([n,p])=>`<div class="probe"><span class="pill">${n} · n=${p.n}</span><strong>MQAR ${metric({mean:p.mqar?.accuracy,std:p.mqar?.accuracy_std},3)}</strong><span>Stable rank ${metric({mean:p.stable_rank?.mean,std:p.stable_rank?.std},2)}</span><br><span>curve σ: rank ${fmt(average(p.stable_rank?.per_layer_std),2)} · CKA ${fmt(average(p.cka?.adjacent_curve_std),3)} · entropy ${fmt(average(p.attention_entropy?.per_layer_std),3)}</span>${spark(p.stable_rank?.per_layer)}${p.stable_rank?.per_layer?.length?`<div class="visual-summary compact"><strong>What this shows</strong><p>The sparkline traces hidden-state stable rank through the layers; its shape shows where representation dimensionality expands or contracts.</p></div>`:''}</div>`).join(''):'<div class="empty">Probe data was not serialized for this report.</div>';
const parity=!!cmp.parameter_parity_valid;document.getElementById('parity').innerHTML=`<span class="pill ${parity?'good':'bad'}">${parity?'PASS':'DOCUMENTED LIMITATION'}</span><h3>Active parameter parity</h3><p style="color:var(--muted)">${parity?'Active parameters per token are within the declared tolerance.':'Active parameters exceed ±5%; total MoE capacity is reported separately and no compensating retrain was performed.'}</p>`;
</script></body></html>"""


def _safe_json(value: object) -> str:
    """Encode JSON safely inside an ``application/json`` script element."""
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")


def _estimate_mean(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict) and isinstance(value.get("mean"), (int, float)):
        return float(value["mean"])
    return None


def _leader(values: dict[str, float], *, highest: bool = False) -> tuple[str, float] | None:
    if not values:
        return None
    selector = max if highest else min
    return selector(values.items(), key=lambda item: item[1])


def _fixed_axis_leader(metrics: dict, axis: str) -> tuple[str, float] | None:
    estimates = metrics.get("comparison", {}).get(axis, {})
    values = {}
    for variant, estimate in estimates.items():
        if axis == "fixed_wallclock" and isinstance(estimate, dict):
            estimate = estimate.get("1.0") or next(reversed(estimate.values()), None)
        mean = _estimate_mean(estimate)
        if mean is not None:
            values[variant] = mean
    return _leader(values)


def _plot_finding(stem: str, metrics: dict) -> str:
    axis_by_plot = {
        "learning_curves_tokens": ("fixed_data", "fixed-data quality"),
        "learning_curves_flops": ("fixed_flops", "fixed-FLOP quality"),
        "learning_curves_wallclock": ("fixed_wallclock", "fixed-wall-clock quality"),
    }
    if stem in axis_by_plot:
        axis, label = axis_by_plot[stem]
        result = _fixed_axis_leader(metrics, axis)
        return (
            f"{result[0]} leads {label} with mean validation loss {result[1]:.4f}."
            if result
            else ""
        )

    comparison = metrics.get("comparison", {})
    if stem == "pareto_flops_val_loss":
        front = comparison.get("pareto_front", [])
        return f"The reported Pareto front contains {', '.join(front)}." if front else ""

    probes = metrics.get("probes", {}).get("aggregated", {})
    if stem == "cka_adjacent":
        values = {
            variant: sum(curve) / len(curve)
            for variant, probe in probes.items()
            if (curve := probe.get("cka", {}).get("adjacent_curve"))
        }
        highest, lowest = _leader(values, highest=True), _leader(values)
        return (
            f"{highest[0]} has the highest mean adjacent-layer CKA ({highest[1]:.3f}), "
            f"while {lowest[0]} has the lowest ({lowest[1]:.3f})."
            if highest and lowest
            else ""
        )
    if stem.startswith("cka_heatmap_"):
        variant = stem.removeprefix("cka_heatmap_")
        cka = probes.get(variant, {}).get("cka", {})
        curve, matrix = cka.get("adjacent_curve", []), cka.get("full_matrix", [])
        if curve:
            adjacent_mean = sum(curve) / len(curve)
            first_last = matrix[0][-1] if matrix and matrix[0] else None
            suffix = (
                f"; first-to-last-layer similarity is {first_last:.3f}"
                if first_last is not None
                else ""
            )
            return f"{variant} averages {adjacent_mean:.3f} adjacent-layer CKA{suffix}."
        return ""
    if stem == "stable_rank":
        values = {
            variant: float(stable_rank["mean"])
            for variant, probe in probes.items()
            if isinstance((stable_rank := probe.get("stable_rank")), dict)
            and isinstance(stable_rank.get("mean"), (int, float))
        }
        highest, lowest = _leader(values, highest=True), _leader(values)
        return (
            f"{highest[0]} has the highest mean stable rank ({highest[1]:.2f}); "
            f"{lowest[0]} has the lowest ({lowest[1]:.2f})."
            if highest and lowest
            else ""
        )
    if stem == "mqar_by_distance":
        values = {
            variant: float(accuracy)
            for variant, probe in probes.items()
            if isinstance((accuracy := probe.get("mqar", {}).get("accuracy")), (int, float))
        }
        best = _leader(values, highest=True)
        return f"{best[0]} records the highest MQAR accuracy ({best[1]:.3f})." if best else ""
    if stem == "per_position_loss":
        values = {
            variant: mean
            for variant, aggregate in metrics.get("aggregated", {}).items()
            if (mean := _estimate_mean(aggregate.get("icl_alpha"))) is not None
        }
        best = _leader(values, highest=True)
        return (
            f"{best[0]} has the steepest fitted loss-decay exponent ({best[1]:.3f})."
            if best
            else ""
        )
    if stem == "flop_breakdown":
        values = {
            variant: mean
            for variant, aggregate in metrics.get("aggregated", {}).items()
            if (mean := _estimate_mean(aggregate.get("step_flops"))) is not None
        }
        lowest, highest = _leader(values), _leader(values, highest=True)
        return (
            f"Estimated step compute ranges from {lowest[0]} at {lowest[1] / 1e9:.1f} GFLOPs "
            f"to {highest[0]} at {highest[1] / 1e9:.1f} GFLOPs."
            if lowest and highest
            else ""
        )
    return ""


def _cka_heatmap_group_finding(metrics: dict) -> str:
    values = {}
    for variant, probe in metrics.get("probes", {}).get("aggregated", {}).items():
        matrix = probe.get("cka", {}).get("full_matrix", [])
        if matrix and matrix[0] and isinstance(matrix[0][-1], (int, float)):
            values[variant] = float(matrix[0][-1])
    if len(values) == 1:
        variant, similarity = next(iter(values.items()))
        return f"{variant}'s first-to-last-layer CKA is {similarity:.3f}."
    strongest, weakest = _leader(values, highest=True), _leader(values)
    return (
        f"Across the heatmaps, {strongest[0]} has the strongest first-to-last-layer CKA "
        f"({strongest[1]:.3f}), while {weakest[0]} has the weakest ({weakest[1]:.3f})."
        if strongest and weakest
        else ""
    )


def _group_summary(group_key: str, metrics: dict) -> str:
    _, _, explanation = _PLOT_GROUPS[group_key]
    group_plots = {
        "training": [
            "learning_curves_tokens",
            "learning_curves_flops",
            "learning_curves_wallclock",
        ],
        "efficiency": [
            "flop_breakdown",
            "pareto_flops_val_loss",
            "pareto_wallclock_val_loss",
            "pareto_peak_memory_val_loss",
            "roofline",
        ],
        "cka": ["cka_adjacent"],
        "context": ["mqar_by_distance", "stable_rank", "per_position_loss"],
    }
    findings = [_plot_finding(stem, metrics) for stem in group_plots[group_key]]
    if group_key == "cka":
        findings.append(_cka_heatmap_group_finding(metrics))
    observed = " ".join(finding for finding in findings if finding)
    result = f"{explanation} In these results, {observed}" if observed else explanation
    if group_key == "efficiency":
        result += (
            " The wall-clock and memory plots mark their own axis-specific fronts, "
            "and the roofline relates arithmetic intensity to achieved throughput."
        )
    return result


def _plot_summary(stem: str, metrics: dict) -> str:
    """Return interpretation and observed findings below a publication plot."""
    if stem.startswith("cka_heatmap_"):
        variant = stem.removeprefix("cka_heatmap_").replace("_", " ")
        explanation = (
            f"Maps pairwise layer-representation similarity for {variant}. "
            "Brighter off-diagonal regions indicate layers that encode similar information."
        )
    else:
        explanation = _PLOT_SUMMARIES.get(
            stem,
            "Shows a publication-ready diagnostic generated from the embedded evaluation results.",
        )
    finding = _plot_finding(stem, metrics)
    return f"{explanation} Observed result: {finding}" if finding else explanation


def _plot_group_key(stem: str) -> str:
    """Map a publication plot to its reader-facing question group."""
    if stem.startswith("learning_curves_"):
        return "training"
    if stem.startswith("cka_"):
        return "cka"
    if stem.startswith("pareto_") or stem in {"flop_breakdown", "roofline"}:
        return "efficiency"
    return "context"


def _embedded_plots(plots_dir: Path, metrics: dict) -> list[dict[str, str]]:
    plots = []
    if not plots_dir.exists():
        return plots
    paths = sorted(
        plots_dir.glob("*.png"),
        key=lambda path: (_PLOT_GROUPS[_plot_group_key(path.stem)][0], path.stem),
    )
    for path in paths:
        group_key = _plot_group_key(path.stem)
        _, group_title, _ = _PLOT_GROUPS[group_key]
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        plots.append(
            {
                "title": path.stem.replace("_", " ").title(),
                "summary": _plot_summary(path.stem, metrics),
                "group": group_title,
                "group_summary": _group_summary(group_key, metrics),
                "data_uri": f"data:image/png;base64,{encoded}",
            }
        )
    return plots


def _render_plot_groups(plots: list[dict[str, str]]) -> str:
    """Render independent figures grouped by the scientific question they answer."""
    if not plots:
        return '<div class="empty">No plot files found.</div>'

    grouped: dict[str, tuple[str, list[dict[str, str]]]] = {}
    for plot in plots:
        _, group_plots = grouped.setdefault(plot["group"], (plot["group_summary"], []))
        group_plots.append(plot)

    sections = []
    for group_title, (group_summary, group_plots) in grouped.items():
        figures = "".join(
            (
                '<figure class="plot-panel">'
                f"<h4>{html.escape(plot['title'])}</h4>"
                f'<img loading="lazy" src="{plot["data_uri"]}" alt="{html.escape(plot["title"])}">'
                '<figcaption class="visual-summary"><strong>What this shows</strong>'
                f"<p>{html.escape(plot['summary'])}</p></figcaption></figure>"
            )
            for plot in group_plots
        )
        figure_label = "figure" if len(group_plots) == 1 else "figures"
        sections.append(
            '<section class="plot-group">'
            '<header class="plot-group-heading"><div class="eyebrow">Visualization group</div>'
            f"<h3>{html.escape(group_title)}</h3>"
            f"<p>{len(group_plots)} {figure_label}, each shown separately.</p></header>"
            f"{figures}"
            '<div class="visual-summary group-summary"><strong>Combined reading</strong>'
            f"<p>{html.escape(group_summary)}</p></div></section>"
        )
    return "".join(sections)


def _render_provenance(metadata: dict, schema_version: object, benchmarks: dict) -> str:
    """Render provenance as labeled fields and readable expandable records."""
    checkpoints = metadata.get("evaluated_checkpoints", [])
    software_versions = metadata.get("software_versions", {})
    warnings = metadata.get("warnings", [])
    cards = [
        (
            "Evaluation hardware",
            metadata.get("hardware") or benchmarks.get("hardware") or "Not recorded",
        ),
        (
            "Report generated",
            metadata.get("timestamp") or benchmarks.get("generated_at") or "Not recorded",
        ),
        ("Schema version", f"v{schema_version or 1}"),
        ("Evaluated checkpoints", len(checkpoints)),
    ]
    card_html = "".join(
        '<div class="provenance-card">'
        f"<span>{html.escape(str(label))}</span><strong>{html.escape(str(value))}</strong></div>"
        for label, value in cards
    )

    details = []
    if software_versions:
        items = "".join(
            f"<li><b>{html.escape(str(name))}</b>: {html.escape(str(version))}</li>"
            for name, version in software_versions.items()
        )
        details.append(
            f"<details><summary>Software versions ({len(software_versions)})</summary><ul>{items}</ul></details>"
        )
    if checkpoints:
        items = "".join(f"<li>{html.escape(str(checkpoint))}</li>" for checkpoint in checkpoints)
        details.append(
            f"<details><summary>Evaluated checkpoints ({len(checkpoints)})</summary><ul>{items}</ul></details>"
        )
    if warnings:
        items = "".join(f"<li>{html.escape(str(warning))}</li>" for warning in warnings)
        details.append(
            f"<details><summary>Methodology warnings ({len(warnings)})</summary><ul>{items}</ul></details>"
        )

    return (
        f'<div class="provenance-grid" id="provenanceCards">{card_html}</div>'
        '<div class="provenance-details" id="provenanceDetails">'
        "<h3>Evaluation record</h3>"
        '<p style="color:var(--muted)">These fields identify the environment and exact checkpoint '
        "set behind this report. Expand the lists for the complete record.</p>"
        f"{''.join(details)}</div>"
    )


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
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    )
    benchmarks_path = report_dir / "raw" / "benchmarks.json"
    if benchmarks_path.is_file():
        benchmarks = json.loads(benchmarks_path.read_text(encoding="utf-8"))
        context_lengths = benchmarks.get("settings", {}).get("context_lengths", [])
        if context_lengths and "long_context_rankings" not in benchmarks:
            benchmarks["long_context_rankings"] = rank_long_context_variants(
                benchmarks.get("variants", {}),
                context_length=max(context_lengths),
            )
        metrics["benchmarks"] = benchmarks
    plots = _embedded_plots(report_dir / "plots", metrics)
    html = (
        _HTML.replace("__REPORT_JSON__", _safe_json(metrics))
        .replace(
            "__PROVENANCE_HTML__",
            _render_provenance(
                metadata,
                metrics.get("schema_version"),
                metrics.get("benchmarks", {}),
            ),
        )
        .replace("__PLOT_HTML__", _render_plot_groups(plots))
    )

    output = Path(output_path) if output_path is not None else report_dir / "index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output
