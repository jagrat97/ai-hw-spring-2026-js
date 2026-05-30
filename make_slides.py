"""Build a self-contained slides.html for the MNIST adversarial-attack assignment (#7.1).

Charts are inline SVG; the original-vs-adversarial sample images are base64-embedded,
so slides.html is a single portable file that runs offline in any browser.

    python make_slides.py
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

PALETTE = {"FGSM": "#fbbf24", "I-FGSM": "#4f9dff", "MI-FGSM": "#f472b6"}


def b64_img(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def parse_results():
    """Parse results/results.md -> (clean_acc, {attack: {eps: (adv_acc, asr)}})."""
    text = (RESULTS / "results.md").read_text()
    clean = float(re.search(r"recognition rate.*?(\d\.\d+)", text).group(1))
    rows = {}
    for m in re.finditer(r"\|\s*(FGSM|I-FGSM|MI-FGSM)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|", text):
        atk, eps, adv, asr = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
        rows.setdefault(atk, {})[eps] = (adv, asr)
    return clean, rows


def asr_chart(rows, width=820, height=400):
    """Line chart: ASR (y) vs epsilon (x), one line per attack."""
    eps_vals = sorted({e for atk in rows.values() for e in atk})
    pad_l, pad_r, pad_t, pad_b = 70, 150, 30, 60
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n = len(eps_vals)

    def x(i):
        return pad_l + (plot_w * i / (n - 1) if n > 1 else plot_w / 2)

    def y(v):
        return pad_t + plot_h * (1 - v)  # ASR in [0,1]

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    for i in range(6):
        v = i / 5
        yy = y(v)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{pad_l+plot_w}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{pad_l-10}" y="{yy+4:.1f}" class="ytick">{v*100:.0f}%</text>')
    for i, e in enumerate(eps_vals):
        parts.append(f'<text x="{x(i):.1f}" y="{pad_t+plot_h+30:.1f}" class="xtick">&#949; = {e}</text>')
    parts.append(f'<text x="{pad_l-46}" y="{pad_t+plot_h/2:.1f}" class="axislabel" transform="rotate(-90 {pad_l-46} {pad_t+plot_h/2:.1f})">Attack success rate</text>')
    for si, (atk, color) in enumerate(PALETTE.items()):
        vals = [rows[atk][e][1] for e in eps_vals]
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>')
        for i, v in enumerate(vals):
            parts.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="5" fill="{color}"/>')
        ly = pad_t + 28 * si + 16
        parts.append(f'<rect x="{pad_l+plot_w+20}" y="{ly-12}" width="18" height="6" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{pad_l+plot_w+44}" y="{ly-3}" class="legend">{atk}</text>')
    parts.append("</svg>")
    return "".join(parts)


SLIDE_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{background:#0d1117;color:#e6edf3;font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;overflow:hidden}
#deck{height:100vh;width:100vw;position:relative}
.slide{position:absolute;inset:0;display:none;flex-direction:column;justify-content:center;
  padding:4vh 7vw;animation:fade .35s ease}
.slide.active{display:flex}
@keyframes fade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
h1{font-size:3.4rem;line-height:1.1;margin-bottom:1rem}
h2{font-size:2.3rem;color:#f472b6;margin-bottom:1.6rem;border-bottom:1px solid #21262d;padding-bottom:.5rem}
.sub{font-size:1.4rem;color:#8b949e}
.meta{margin-top:2.5rem;font-size:1.2rem;color:#8b949e}
ul{list-style:none;font-size:1.5rem;line-height:2.1}
ul li{padding-left:1.6rem;position:relative}
ul li::before{content:"\\25B8";color:#f472b6;position:absolute;left:0}
.two{display:grid;grid-template-columns:1.05fr .95fr;gap:3vw;align-items:center}
.chart{width:100%;height:auto;background:#0d1117}
.chart .grid{stroke:#21262d;stroke-width:1}
.chart .ytick{fill:#8b949e;font-size:15px;text-anchor:end}
.chart .xtick{fill:#c9d1d9;font-size:17px;text-anchor:middle}
.chart .legend{fill:#c9d1d9;font-size:16px}
.chart .axislabel{fill:#8b949e;font-size:15px;text-anchor:middle}
table{border-collapse:collapse;font-size:1.3rem;width:100%}
th,td{padding:.5rem 1rem;text-align:left;border-bottom:1px solid #21262d}
th{color:#f472b6} td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
img.grid{width:100%;border-radius:10px;border:1px solid #21262d;background:#000}
.cap{font-size:1.1rem;color:#8b949e;margin-top:.8rem}
code{background:#161b22;padding:.15em .45em;border-radius:5px;font-size:.92em;color:#79c0ff}
.formula{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:1rem 1.4rem;
  font-size:1.4rem;color:#e6edf3;margin:.5rem 0;font-family:ui-monospace,Menlo,Consolas,monospace}
.big{font-size:5rem;color:#34d399;font-weight:800}
.tag{display:inline-block;background:#161b22;border:1px solid #30363d;border-radius:999px;
  padding:.3rem 1rem;font-size:1.15rem;color:#8b949e;margin:.3rem .4rem .3rem 0}
#nav{position:fixed;bottom:18px;right:26px;font-size:1rem;color:#586069;z-index:10}
#bar{position:fixed;bottom:0;left:0;height:4px;background:#f472b6;transition:width .3s}
.kbd{color:#586069;font-size:.95rem;position:fixed;bottom:18px;left:26px}
"""

SLIDE_JS = """
const slides=[...document.querySelectorAll('.slide')];let i=0;
function show(n){slides[i].classList.remove('active');i=Math.max(0,Math.min(slides.length-1,n));
  slides[i].classList.add('active');
  document.getElementById('nav').textContent=(i+1)+' / '+slides.length;
  document.getElementById('bar').style.width=((i+1)/slides.length*100)+'%';}
document.addEventListener('keydown',e=>{
  if(['ArrowRight','ArrowDown',' ','PageDown'].includes(e.key)){show(i+1);e.preventDefault();}
  else if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key)){show(i-1);e.preventDefault();}
  else if(e.key==='Home'){show(0);} else if(e.key==='End'){show(slides.length-1);}
  else if(e.key==='f'){if(!document.fullscreenElement)document.documentElement.requestFullscreen();else document.exitFullscreen();}});
document.addEventListener('click',e=>{if(e.clientX>window.innerWidth*0.5)show(i+1);else show(i-1);});
show(0);
"""


def build():
    clean, rows = parse_results()
    chart = asr_chart(rows)
    ifgsm_img = b64_img(RESULTS / "samples_ifgsm.png")
    fgsm_img = b64_img(RESULTS / "samples_fgsm.png")

    eps_vals = sorted({e for atk in rows.values() for e in atk})
    trows = ""
    for atk in PALETTE:
        for e in eps_vals:
            adv, asr = rows[atk][e]
            trows += (f"<tr><td>{atk}</td><td class='num'>{e}</td>"
                      f"<td class='num'>{adv*100:.1f}%</td><td class='num'>{asr*100:.1f}%</td></tr>")

    slides = []
    slides.append("""
<section class="slide active">
  <h1>Attacking an MNIST<br>Classifier</h1>
  <p class="sub">Assignment #7.1 — gradient-based adversarial examples vs a CNN</p>
  <p class="meta">Jagrat Shrivastav · AI HW, Spring 2026</p>
</section>""")

    slides.append(f"""
<section class="slide">
  <h2>The Idea</h2>
  <ul>
    <li>A CNN classifies MNIST digits at <b>{clean*100:.2f}%</b> accuracy.</li>
    <li>Add a tiny, <b>human-invisible</b> perturbation to the pixels &hellip;</li>
    <li>&hellip; and the same model confidently predicts the <b>wrong</b> digit.</li>
    <li>Threat model: <b>untargeted</b>, <b>L&#8734;-bounded</b> (each pixel moves &le; &epsilon;), inputs kept in [0,1].</li>
  </ul>
</section>""")

    slides.append("""
<section class="slide">
  <h2>Three Attacks</h2>
  <p style="font-size:1.4rem;margin-bottom:.6rem"><b>FGSM</b> — one gradient step (Goodfellow 2014):</p>
  <div class="formula">x_adv = x + &epsilon; &middot; sign(&nabla;<sub>x</sub> L)</div>
  <p style="font-size:1.4rem;margin:.6rem 0 .6rem"><b>I-FGSM / PGD</b> — 10 small steps, projected back into the &epsilon;-ball:</p>
  <div class="formula">x<sub>t+1</sub> = clip<sub>&epsilon;</sub>( x<sub>t</sub> + &alpha; &middot; sign(&nabla;<sub>x</sub> L) )</div>
  <p style="font-size:1.4rem;margin:.6rem 0 .6rem"><b>MI-FGSM</b> — I-FGSM with momentum (Dong 2018, &mu;=1.0):</p>
  <div class="formula">g<sub>t+1</sub> = &mu;&middot;g<sub>t</sub> + &nabla;L / &#8214;&nabla;L&#8214;<sub>1</sub> &nbsp;;&nbsp; x<sub>t+1</sub> = clip<sub>&epsilon;</sub>( x<sub>t</sub> + &alpha;&middot;sign(g<sub>t+1</sub>) )</div>
</section>""")

    slides.append(f"""
<section class="slide">
  <h2>Clean Baseline</h2>
  <p class="big">{clean*100:.2f}%</p>
  <p class="sub">recognition accuracy on the 10,000-image MNIST test set, before any attack.</p>
  <p class="meta">Attack success rate (ASR) = fraction of originally-correct images flipped to a wrong label.</p>
</section>""")

    slides.append(f"""
<section class="slide">
  <h2>Attack Success vs &epsilon;</h2>
  {chart}
  <p class="cap">ASR rises monotonically with the perturbation budget. Iterative attacks dominate single-step FGSM.</p>
</section>""")

    slides.append(f"""
<section class="slide">
  <h2>Full Results</h2>
  <table>
    <tr><th>Attack</th><th class='num'>&epsilon;</th><th class='num'>Acc. on adv. examples</th><th class='num'>ASR</th></tr>
    {trows}
  </table>
  <p class="cap">At &epsilon;=0.3, I-FGSM drives model accuracy to 0.05% (ASR 99.95%).</p>
</section>""")

    slides.append(f"""
<section class="slide">
  <h2>What the Attack Looks Like</h2>
  <div class="two">
    <div><img class="grid" src="{ifgsm_img}" alt="I-FGSM original vs adversarial"/></div>
    <div>
      <ul>
        <li>I-FGSM at &epsilon;=0.2, one pair per digit class.</li>
        <li>Top row: original. Bottom row: adversarial.</li>
        <li>Still obvious to a human &mdash; the model misreads them.</li>
      </ul>
    </div>
  </div>
</section>""")

    slides.append("""
<section class="slide">
  <h2>Takeaways</h2>
  <ul>
    <li>A 98.99%-accurate CNN is <b>not</b> robust — tiny L&#8734; noise breaks it.</li>
    <li>ASR grows with &epsilon;; <b>iterative</b> attacks (I-FGSM, MI-FGSM) beat single-step FGSM.</li>
    <li>At &epsilon;=0.3 the model is essentially destroyed (&gt;99% ASR).</li>
    <li>MI-FGSM matches I-FGSM here; its real edge is <b>transferability</b> across models.</li>
    <li>Reproducible: <code>python train.py</code> then <code>python test.py</code>.</li>
  </ul>
  <p class="meta">Thank you — questions?</p>
</section>""")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Attacking MNIST — Assignment #7.1</title>
<style>{SLIDE_CSS}</style></head>
<body>
<div id="deck">{''.join(slides)}</div>
<div id="bar"></div>
<div class="kbd">&larr; &rarr; navigate &middot; F fullscreen</div>
<div id="nav"></div>
<script>{SLIDE_JS}</script>
</body></html>"""

    out = ROOT / "slides.html"
    out.write_text(html)
    print(f"wrote {out}  ({len(html)//1024} KB, {len(slides)} slides)")


if __name__ == "__main__":
    build()
