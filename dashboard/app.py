"""
Sell Now or Store?  A mandi-style decision helper for Karnataka onion and maize farmers.

Run from the fds folder (Anaconda Prompt):
    python -m streamlit run dashboard/app.py

Reads the files made by Notebooks 2-6 in ../data. Nothing is downloaded except the Google fonts
(if there is no internet, the page still works with normal fonts). No paid hosting is needed.

Simple version: 4 inputs, 4 tabs. Tab 1 forecasts from any date and price a farmer types in.
Design idea: the page looks like a real APMC mandi - a painted signboard, an LED price board,
a chalkboard rate card, a tarazu (balance scale), jute sacks, a trader's patti (receipt)
and a khata (ledger book). The numbers behind them are exactly the ones from Notebooks 5 and 6.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import HistGradientBoostingRegressor

DATA = Path(__file__).resolve().parent.parent / "data"
H = np.array([4, 8, 12])
MONTHS_STORED = H / 4.345
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# mandi palette: jute sacks, soil, leaves, maize gold, red onion, chalkboard, receipt paper
JUTE, SOIL, LEAF, GOLD, ONION_RED = "#C8A96B", "#4A2E1E", "#3E7C3A", "#E0A526", "#A8325A"
BOARD, PAPER, BRICK, CREAM, MUTED = "#1F3B2D", "#FFFBEF", "#B23A2E", "#F5ECD7", "#8A7A62"
CROP_COLOR = {"Onion": ONION_RED, "Maize": GOLD}
KANNADA = {"Onion": "ಈರುಳ್ಳಿ", "Maize": "ಮೆಕ್ಕೆಜೋಳ"}

SCENARIOS = {  # same presets as Notebook 6
    "Onion": {
        "Farm ventilated storage": dict(loss=10.0, rent=5.0, rate=7.0),
        "Well-managed storage": dict(loss=5.0, rent=15.0, rate=7.0),
        "Cold storage": dict(loss=1.25, rent=50.0, rate=7.0),
        "No costs (price effect only)": dict(loss=0.0, rent=0.0, rate=0.0),
    },
    "Maize": {
        "Home storage in bags": dict(loss=1.5, rent=0.0, rate=7.0),
        "Warehouse": dict(loss=0.5, rent=10.0, rate=7.0),
        "Poor home storage": dict(loss=3.0, rent=0.0, rate=7.0),
        "No costs (price effect only)": dict(loss=0.0, rent=0.0, rate=0.0),
    },
}

st.set_page_config(page_title="Sell Now or Store? | Karnataka Mandi Desk", page_icon="🌾", layout="wide")

# ------------------------------------------------------------------ look and feel
FONTS = ("https://fonts.googleapis.com/css2?family=Rozha+One&family=Mukta:wght@400;600;800"
         "&family=VT323&family=Kalam:wght@400;700&family=Courier+Prime:wght@400;700"
         "&family=Baloo+Tamma+2:wght@600;800&display=swap")

CSS = f"""
@import url('{FONTS}');
:root {{ --jute:{JUTE}; --soil:{SOIL}; --leaf:{LEAF}; --gold:{GOLD}; --board:{BOARD}; --brick:{BRICK}; }}
.stApp {{
  background-color:#EFE2C4;
  background-image:
    repeating-linear-gradient(45deg, rgba(110,80,30,.045) 0 2px, transparent 2px 7px),
    repeating-linear-gradient(-45deg, rgba(110,80,30,.045) 0 2px, transparent 2px 7px);
}}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp [data-testid="stMarkdownContainer"] {{ font-family:'Mukta', 'Segoe UI', sans-serif; }}
[data-testid="stIconMaterial"], [class*="material-symbols"], [data-testid="stExpanderToggleIcon"] {{ font-family:'Material Symbols Rounded' !important; }}
.stApp p, .stApp li {{ color:{SOIL}; font-size:1.02rem; }}
[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ padding-top:1.4rem; max-width:1380px; }}

/* ---------- sidebar = chalkboard on a wooden frame ---------- */
[data-testid="stSidebar"] {{
  background-color:{BOARD};
  background-image: radial-gradient(rgba(255,255,255,.05) 1px, transparent 1px),
                    linear-gradient(160deg, rgba(255,255,255,.04), rgba(0,0,0,.18));
  background-size: 5px 5px, 100% 100%;
  border-right: 12px solid #7A4E2D;
  box-shadow: inset -3px 0 0 #A06A3C, 4px 0 14px rgba(0,0,0,.25);
}}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {{ color:#F4EFDF; }}
[data-testid="stSidebar"] label p {{ font-family:'Kalam', cursive !important; font-size:1.05rem; color:#F4EFDF; }}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [data-testid="stNumberInput"] button {{
  background:#2C4E3C !important; color:#F4EFDF !important; border-color:rgba(244,239,223,.35) !important;
}}
[data-testid="stSidebar"] svg {{ fill:#F4EFDF; }}
.chalk-h {{ font-family:'Kalam', cursive; font-size:1.55rem; color:#FFF8DC; margin:.6rem 0 .1rem;
           border-bottom:2px dashed rgba(255,248,220,.45); padding-bottom:.15rem; }}
.chalk-h small {{ font-family:'Baloo Tamma 2', sans-serif; font-size:1rem; color:#F2D16B; margin-left:.4rem; }}
.chalk-note {{ font-family:'Kalam', cursive; font-size:.92rem; color:#D9E3D2; line-height:1.35; }}

/* ---------- tabs = wooden price tags hanging under the board ---------- */
.stTabs [data-baseweb="tab-list"], .stTabs [role="tablist"] {{ gap:12px; border-bottom:5px solid #7A4E2D; padding-left:4px; }}
.stTabs [role="tab"] {{
  background-color:{JUTE} !important; padding:10px 20px !important;
  background-image: repeating-linear-gradient(90deg, rgba(90,60,20,.12) 0 1px, transparent 1px 4px),
                    repeating-linear-gradient(0deg, rgba(90,60,20,.12) 0 1px, transparent 1px 4px);
  border-radius:12px 12px 0 0; height:auto; cursor:pointer;
  box-shadow: inset 0 -3px 0 rgba(0,0,0,.12);
}}
.stTabs [role="tab"], .stTabs [role="tab"] * {{ font-family:'Rozha One', serif !important; font-size:1.12rem; color:{SOIL}; }}
.stTabs [aria-selected="true"] {{ background:{LEAF} !important; }}
.stTabs [aria-selected="true"], .stTabs [aria-selected="true"] * {{ color:#FFF6D6 !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none; }}

/* ---------- signboard ---------- */
.board {{ position:relative; margin:22px 0 0; border:12px solid #7A4E2D; border-radius:10px;
  box-shadow: 0 0 0 3px #A06A3C, 0 10px 24px rgba(60,35,10,.35);
  background:#1E5A3A;
  background-image: linear-gradient(170deg, rgba(255,255,255,.07), rgba(0,0,0,.15)),
                    repeating-linear-gradient(0deg, rgba(0,0,0,.05) 0 2px, transparent 2px 9px); }}
.board:before, .board:after {{ content:""; position:absolute; top:-34px; width:4px; height:24px;
  background:repeating-linear-gradient(0deg,#5a4a3a 0 4px,#8d7b66 4px 8px); }}
.board:before {{ left:14%; }} .board:after {{ right:14%; }}
.board-in {{ display:flex; align-items:center; justify-content:space-between; gap:18px;
  padding:14px 26px 16px; border:2px dashed rgba(247,208,70,.55); margin:8px; border-radius:4px; }}
.board-txt {{ text-align:center; flex:1; }}
.board-top {{ font-family:'Baloo Tamma 2', sans-serif; color:#F7D046; letter-spacing:1px; font-size:1.05rem; }}
.board-top b {{ font-family:'Mukta'; letter-spacing:4px; color:#FDF1C4; font-weight:800; font-size:.85rem; }}
.board-title {{ font-family:'Rozha One', serif; color:#F7D046; font-size:3.4rem; line-height:1.05;
  text-shadow: 3px 3px 0 #6B4A00, 0 0 18px rgba(247,208,70,.25); }}
.board-title span {{ color:#FFFFFF; font-size:2.3rem; text-shadow: 2px 2px 0 #3a3a3a; }}
.board-kn {{ font-family:'Baloo Tamma 2', sans-serif; color:#FFFFFF; font-size:1.6rem; font-weight:800;
  text-shadow: 2px 2px 0 #0d2e1c; }}
.board-sub {{ color:#DCEBD8 !important; font-size:.98rem !important; margin-top:2px; }}

/* ---------- LED price board ---------- */
.led {{ background:#0B0B08; border:6px solid #3a3a36; border-radius:6px; overflow:hidden; white-space:nowrap;
  box-shadow: inset 0 0 18px rgba(0,0,0,.9), 0 6px 14px rgba(0,0,0,.25); margin:12px 0 18px; position:relative; }}
.led-tag {{ position:absolute; left:0; top:0; bottom:0; z-index:2; background:#B23A2E; color:#FFE9A8;
  font-family:'VT323', monospace; font-size:1.5rem; padding:4px 14px; display:flex; align-items:center;
  box-shadow: 6px 0 10px #0B0B08; }}
.led-run {{ display:inline-block; padding:4px 0; animation: ledrun 70s linear infinite;
  font-family:'VT323', monospace; font-size:1.6rem; color:#FFB627; text-shadow:0 0 6px rgba(255,182,39,.8);
  background-image: radial-gradient(rgba(255,255,255,.07) 1px, transparent 1px); background-size:3px 3px; }}
.led:hover .led-run {{ animation-play-state:paused; }}
.led-run span {{ margin-right:46px; }}
.led-run .up {{ color:#7CFF6B; text-shadow:0 0 6px rgba(124,255,107,.7); }}
.led-run .dn {{ color:#FF5B4D; text-shadow:0 0 6px rgba(255,91,77,.7); }}
@keyframes ledrun {{ from {{ transform:translateX(0); }} to {{ transform:translateX(-50%); }} }}

/* ---------- section headings painted on a plank ---------- */
.plank {{ display:flex; flex-wrap:wrap; align-items:baseline; column-gap:12px; row-gap:0; margin:22px 0 8px; padding:6px 16px;
  background:linear-gradient(90deg, rgba(122,78,45,.16), rgba(122,78,45,0)); border-left:7px solid {LEAF};
  border-radius:4px; }}
.plank .en {{ font-family:'Rozha One', serif; font-size:1.55rem; color:{SOIL}; }}
.plank .kn {{ font-family:'Baloo Tamma 2', sans-serif; font-size:1.05rem; color:{LEAF}; font-weight:800; }}
.plank .kn {{ white-space:nowrap; }}
.plank .sub {{ font-size:.95rem; color:{MUTED}; }}

/* ---------- chalkboard rate card ---------- */
.rate {{ background:{BOARD}; border:12px solid #7A4E2D; border-radius:8px; padding:16px 22px 14px; color:#F4EFDF;
  box-shadow: 0 0 0 3px #A06A3C, 0 10px 20px rgba(60,35,10,.3);
  background-image: radial-gradient(rgba(255,255,255,.05) 1px, transparent 1px); background-size:5px 5px; }}
.rate .k {{ font-family:'Baloo Tamma 2'; color:#F2D16B; font-size:1.15rem; font-weight:800; }}
.rate .k span {{ font-family:'Kalam'; color:#F4EFDF; font-weight:400; }}
.rate .m {{ font-family:'Kalam', cursive; font-size:1.25rem; color:#FFFFFF; }}
.rate .p {{ font-family:'Kalam', cursive; font-weight:700; font-size:3.4rem; line-height:1.05; color:#FFFFFF;
  text-shadow: 0 0 2px rgba(255,255,255,.6); }}
.rate .p small {{ font-size:1.15rem; font-weight:400; color:#D9E3D2; }}
.rate .r {{ font-family:'Kalam', cursive; font-size:1.05rem; color:#D9E3D2; border-top:2px dashed rgba(255,255,255,.3);
  margin-top:8px; padding-top:6px; }}
.rate .r b {{ color:#FFFFFF; }}
.rate .chip {{ display:inline-block; font-family:'Mukta'; font-size:.8rem; padding:1px 8px; border-radius:20px;
  background:rgba(242,209,107,.18); color:#F2D16B; border:1px solid rgba(242,209,107,.5); }}

/* ---------- patti (trader's receipt) ---------- */
.patti {{ background:{PAPER}; position:relative; padding:22px 26px 20px; font-family:'Courier Prime', monospace;
  color:#2C2418; box-shadow: 0 8px 18px rgba(60,35,10,.25); border-radius:2px;
  background-image: linear-gradient(rgba(0,0,0,.025) 1px, transparent 1px); background-size:100% 26px;
  -webkit-mask: radial-gradient(circle 7px at 50% 0, transparent 98%, #000) 0 0/22px 100%;
          mask: radial-gradient(circle 7px at 50% 0, transparent 98%, #000) 0 0/22px 100%; }}
.patti * {{ font-family:'Courier Prime', monospace !important; color:#2C2418; }}
.patti .hd {{ text-align:center; border-bottom:2px solid #2C2418; padding-bottom:6px; margin-bottom:8px; }}
.patti .hd .kn {{ font-family:'Baloo Tamma 2', sans-serif !important; font-size:1.1rem; font-weight:800; }}
.patti .hd .t {{ font-size:1.3rem; font-weight:700; letter-spacing:3px; }}
.patti .hd .s {{ font-size:.85rem; }}
.patti .ln {{ display:flex; gap:6px; font-size:.98rem; line-height:1.65; }}
.patti .ln .d {{ flex:1; border-bottom:2px dotted #9b8f78; transform:translateY(-6px); }}
.patti .ln.neg b {{ color:{BRICK}; }}
.patti .ln.tot {{ border-top:2px solid #2C2418; margin-top:4px; padding-top:2px; font-weight:700; }}
.patti .sec {{ margin-top:8px; font-weight:700; text-decoration:underline; font-size:.95rem; }}
.patti .foot {{ display:flex; align-items:center; justify-content:space-between; margin-top:12px; gap:10px; }}
.patti .foot {{ flex-wrap:wrap; }} .patti .sig {{ white-space:nowrap; font-family:'Kalam', cursive !important; font-size:1.05rem; color:#26408B !important; }}
.stamp {{ display:inline-block; transform:rotate(-8deg); border:5px double currentColor; border-radius:8px;
  padding:4px 14px; font-family:'Rozha One', serif !important; font-size:1.6rem; white-space:nowrap; letter-spacing:2px;
  opacity:.88; animation: stampin .55s cubic-bezier(.2,1.6,.4,1) both; text-align:center; line-height:1.1; }}
.stamp small {{ display:block; font-family:'Courier Prime', monospace !important; font-size:.75rem; letter-spacing:1px; }}
.stamp.go {{ color:#1F7A3A !important; }} .stamp.go * {{ color:#1F7A3A !important; }}
.stamp.sell {{ color:{BRICK} !important; }} .stamp.sell * {{ color:{BRICK} !important; }}
@keyframes stampin {{ 0% {{ transform:rotate(-8deg) scale(2.4); opacity:0; }} 100% {{ transform:rotate(-8deg) scale(1); opacity:.88; }} }}
.caution {{ margin-top:10px; padding:8px 12px; background:#FFF1C9; border-left:5px solid {GOLD}; font-size:.92rem; }}

/* ---------- khata (ledger book) ---------- */
.khata {{ background:#FFFDF3; border-radius:4px 10px 10px 4px; box-shadow: -10px 0 0 #8E2A23, 0 10px 22px rgba(60,35,10,.28);
  padding:14px 18px 12px 46px; position:relative; overflow-x:auto; }}
.khata:before {{ content:""; position:absolute; left:34px; top:0; bottom:0; border-left:2px solid #E3A3A3; }}
.khata .kt {{ font-family:'Rozha One', serif; font-size:1.35rem; color:#8E2A23; }}
.khata .kt span {{ font-family:'Baloo Tamma 2'; font-size:1rem; color:{SOIL}; margin-left:8px; }}
.khata table {{ width:100%; border-collapse:collapse; font-size:.95rem; }}
.khata td, .khata th {{ font-family:'Mukta', sans-serif; }}
.khata td.num {{ font-family:'Courier Prime', monospace; white-space:nowrap; }}
.khata th {{ text-align:left; color:#8E2A23; font-weight:700; border-bottom:2px solid #8E2A23; padding:6px 8px; font-size:.85rem; }}
.khata td {{ border-bottom:1px solid #BFD4EA; padding:6px 8px; color:#2C2418; }}
.khata td.num, .khata th.num {{ text-align:right; }}
.khata .jama {{ color:#1F7A3A; font-weight:700; }}
.khata .naam {{ color:{BRICK}; font-weight:700; }}
.khata tr.hl td {{ background:rgba(224,165,38,.16); font-weight:700; }}
.khata .note {{ font-family:'Kalam', cursive; color:#26408B; font-size:.95rem; margin-top:6px; }}


"""

FRAME_CSS = f"""
@import url('{FONTS}');
body {{ margin:0; font-family:'Mukta', 'Segoe UI', sans-serif; color:{SOIL}; background:transparent; }}
.sacks {{ display:flex; gap:16px; }}
.sack-card {{ flex:1; background:rgba(255,251,239,.75); border:2px solid #D8C49A; border-radius:14px; padding:10px 12px 12px;
  text-align:center; position:relative; }}
.sack-card.best {{ border:3px solid {LEAF}; background:rgba(234,244,226,.9); box-shadow:0 8px 18px rgba(62,124,58,.22); }}
.tag {{ position:absolute; top:-12px; right:14px; background:{LEAF}; color:#FFF6D6; font-family:'Rozha One', serif;
  padding:2px 12px; border-radius:4px; font-size:.95rem; transform:rotate(4deg); box-shadow:0 3px 6px rgba(0,0,0,.2); }}
.big {{ font-family:'Rozha One', serif; font-size:2rem; line-height:1; }}
.rng {{ font-size:.9rem; color:{MUTED}; }}
.money {{ font-size:.98rem; margin-top:4px; }}
.money b {{ font-family:'Rozha One', serif; font-size:1.15rem; }}
.odds {{ margin-top:6px; font-size:.86rem; color:{SOIL}; }}
.icons {{ display:flex; justify-content:center; gap:2px; margin-top:3px; }}
.pos {{ color:#1F7A3A; }} .neg {{ color:{BRICK}; }}
.scale-wrap {{ text-align:center; }}
.verdict {{ font-family:'Rozha One', serif; font-size:1.3rem; margin-top:-6px; }}
"""


def md(html):
    # one line, so Streamlit's markdown treats it as raw HTML
    st.markdown(" ".join(l.strip() for l in html.splitlines()), unsafe_allow_html=True)


def frame(html, height):
    page = f"<!doctype html><html><head><meta charset='utf-8'><style>{FRAME_CSS}</style></head><body>{html}</body></html>"
    if hasattr(st, "iframe"):          # newer Streamlit
        st.iframe(page, height=height)
    else:                              # older Streamlit
        components.html(page, height=height)


def plank(en, kn="", sub=""):
    md(f'<div class="plank"><span class="en">{en}</span><span class="kn">{kn}</span><span class="sub">{sub}</span></div>')


def inr(v, sign=False):
    # Indian digit grouping: 6,20,000
    neg = v < 0
    s = str(int(round(abs(v))))
    if len(s) > 3:
        s = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", s[:-3]) + "," + s[-3:]
    pre = "−" if neg else ("+" if sign else "")
    return f"{pre}₹{s}"


def show_chart(fig):
    # works on both newer and older Streamlit versions
    try:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except Exception:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def style(fig, height=380, title=None):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=56 if title else 20, b=10),
                      title=dict(text=title, font=dict(family="Rozha One, serif", size=19, color=SOIL)) if title else None,
                      plot_bgcolor="rgba(255,251,239,.85)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Mukta, sans-serif", size=13, color=SOIL),
                      legend=dict(orientation="h", yanchor="top", y=-0.14, x=0, bgcolor="rgba(0,0,0,0)"),
                      hoverlabel=dict(font_family="Mukta"))
    fig.update_xaxes(showgrid=True, gridcolor="#E6D7B5", linecolor="#C9B48A", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E6D7B5", linecolor="#C9B48A", zeroline=False)
    return fig


# ------------------------------------------------------------------ little drawings (SVG)
def onion_svg(size=56, fill=ONION_RED):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 64 64">'
            f'<path d="M32 3 C33 8 31 12 32 14" stroke="#5E8F35" stroke-width="3" fill="none" stroke-linecap="round"/>'
            f'<path d="M32 12 C29 18 20 21 15 29 C8 41 13 55 32 59 C51 55 56 41 49 29 C44 21 35 18 32 12Z" fill="{fill}"/>'
            f'<path d="M32 15 C27 26 24 40 29 57 M32 15 C37 26 40 40 35 57 M32 15 C21 28 18 43 24 55 M32 15 C43 28 46 43 40 55" '
            f'stroke="rgba(0,0,0,.18)" stroke-width="1.4" fill="none"/>'
            f'<path d="M24 26 C21 31 20 36 21 41" stroke="rgba(255,255,255,.35)" stroke-width="2.5" fill="none" stroke-linecap="round"/>'
            f'<path d="M28 59 l-3 4 M32 59 v4 M36 59 l3 4" stroke="#C9B48A" stroke-width="1.5"/></svg>')


def maize_svg(size=56, fill="#F2C230"):
    dots = "".join(f'<circle cx="{x}" cy="{y}" r="1.7"/>' for y in range(10, 49, 4)
                   for x in range(24, 41, 4) if ((x - 32) / 10) ** 2 + ((y - 28) / 21) ** 2 < 0.9)
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 64 64">'
            f'<ellipse cx="32" cy="28" rx="11" ry="22" fill="{fill}"/>'
            f'<g fill="rgba(160,105,10,.55)">{dots}</g>'
            f'<path d="M32 62 C19 53 15 37 21 17 C24 36 28 48 32 62Z" fill="#5E9E3A"/>'
            f'<path d="M32 62 C45 53 49 37 43 17 C40 36 36 48 32 62Z" fill="#477F2C"/>'
            f'<path d="M32 6 C30 2 27 1 25 2 M32 6 C34 2 37 1 39 2" stroke="#B8935A" stroke-width="1.5" fill="none"/></svg>')


def crop_icon(crop, size=56, faded=False):
    if crop == "Onion":
        return onion_svg(size, "#D9CDB8" if faded else ONION_RED)
    return maize_svg(size, "#E2D9C3" if faded else "#F2C230")


def sack_svg(label, keep_frac, color):
    # jute sack; the darker grain level shows how much crop is left after storage loss
    body = "M28 52 Q14 120 24 176 Q80 192 136 176 Q146 120 132 52 Q80 62 28 52 Z"
    top = 186 - 128 * keep_frac
    return f"""
    <svg width="150" height="180" viewBox="0 0 160 196">
      <defs>
        <pattern id="weave{label}" width="6" height="6" patternUnits="userSpaceOnUse">
          <rect width="6" height="6" fill="{JUTE}"/>
          <path d="M0 3 H6 M3 0 V6" stroke="rgba(90,60,20,.22)" stroke-width="1.2"/>
        </pattern>
        <clipPath id="clip{label}"><path d="{body}"/></clipPath>
      </defs>
      <path d="{body}" fill="url(#weave{label})" stroke="#8C6B3A" stroke-width="2"/>
      <rect x="0" y="{top:.1f}" width="160" height="200" fill="rgba(110,70,25,.30)" clip-path="url(#clip{label})">
        <animate attributeName="y" from="186" to="{top:.1f}" dur="1.1s" fill="freeze"/>
      </rect>
      <path d="M50 52 Q80 34 110 52" fill="none" stroke="#8C6B3A" stroke-width="2"/>
      <path d="M56 46 L46 16 Q80 28 114 16 L104 46 Z" fill="url(#weave{label})" stroke="#8C6B3A" stroke-width="2"/>
      <ellipse cx="80" cy="47" rx="27" ry="5" fill="none" stroke="#6B4A22" stroke-width="4"/>
      <text x="80" y="104" text-anchor="middle" font-family="Rozha One, serif" font-size="25" fill="{color}"
            opacity=".9">{label}</text>
      <text x="80" y="126" text-anchor="middle" font-family="Courier Prime, monospace" font-size="13" fill="#4A2E1E">WEEKS</text>
      <text x="80" y="160" text-anchor="middle" font-family="Courier Prime, monospace" font-size="11.5"
            fill="#4A2E1E">{100 * keep_frac:.0f}% crop left</text>
    </svg>"""


def tarazu_svg(left_val, right_val, right_label, crop):
    # balance scale: the heavier pan (more money) goes down
    diff = (right_val - left_val) / left_val * 100
    ang = float(np.clip(diff * 1.2, -13, 13))
    a = np.radians(ang)
    L, cx, cy = 120, 180, 62
    dxl, dyl = L - L * np.cos(a), -L * np.sin(a)          # how far the left end moves
    dxr, dyr = -(L - L * np.cos(a)), L * np.sin(a)        # how far the right end moves
    coins = "".join(f'<ellipse cx="60" cy="{148 - 5 * i}" rx="17" ry="5" fill="#E8B83A" stroke="#9C7414" stroke-width="1.2"/>'
                    for i in range(5)) + '<text x="60" y="131" text-anchor="middle" font-family="Rozha One" font-size="15" fill="#7a5a10">₹</text>'
    sack = ('<path d="M282 152 Q276 132 284 116 Q300 120 316 116 Q324 132 318 152 Q300 157 282 152Z" fill="#C8A96B" stroke="#8C6B3A" stroke-width="1.5"/>'
            '<path d="M290 116 L286 106 Q300 110 314 106 L310 116" fill="#C8A96B" stroke="#8C6B3A" stroke-width="1.5"/>'
            '<ellipse cx="300" cy="115" rx="11" ry="2.5" fill="none" stroke="#6B4A22" stroke-width="2"/>')

    def pan(x, contents, dx, dy):
        anim = (f'<animateTransform attributeName="transform" type="translate" from="0 0" to="{dx:.2f} {dy:.2f}" '
                f'dur="1.6s" begin="0.2s" fill="freeze" calcMode="spline" keySplines=".25 .1 .25 1" keyTimes="0;1"/>')
        return (f'<g>{anim}'
                f'<path d="M{x} {cy} L{x - 40} 156 M{x} {cy} L{x + 40} 156" stroke="#6B5434" stroke-width="1.6"/>'
                f'{contents}'
                f'<path d="M{x - 48} 154 Q{x} 186 {x + 48} 154 Z" fill="#B7873A" stroke="#7A5420" stroke-width="2"/>'
                f'</g>')

    beam_anim = (f'<animateTransform attributeName="transform" type="rotate" from="0 {cx} {cy}" to="{ang:.2f} {cx} {cy}" '
                 f'dur="1.6s" begin="0.2s" fill="freeze" calcMode="spline" keySplines=".25 .1 .25 1" keyTimes="0;1"/>')
    return f"""
    <svg width="100%" height="250" viewBox="0 0 360 250">
      <path d="M130 240 L230 240 L210 222 L150 222 Z" fill="#7A4E2D"/>
      <rect x="175" y="58" width="10" height="166" fill="#8C5F35"/>
      <path d="M180 30 L170 52 L190 52 Z" fill="#B7873A"/>
      {pan(60, coins, dxl, dyl)}
      {pan(300, sack, dxr, dyr)}
      <g>{beam_anim}
        <rect x="{cx - L - 6}" y="{cy - 5}" width="{2 * L + 12}" height="10" rx="5" fill="#9C6B3A" stroke="#6B4A22" stroke-width="1.5"/>
        <circle cx="{cx}" cy="{cy}" r="9" fill="#E0A526" stroke="#6B4A22" stroke-width="2"/>
        <path d="M{cx} {cy - 6} L{cx} {cy + 26}" stroke="#4A2E1E" stroke-width="3"/>
      </g>
      <text x="60" y="210" text-anchor="middle" font-family="Rozha One, serif" font-size="15" fill="#4A2E1E">SELL NOW</text>
      <text x="60" y="228" text-anchor="middle" font-family="Mukta, sans-serif" font-size="14" fill="#4A2E1E">{inr(left_val)}</text>
      <text x="300" y="210" text-anchor="middle" font-family="Rozha One, serif" font-size="15" fill="#4A2E1E">{right_label.upper()}</text>
      <text x="300" y="228" text-anchor="middle" font-family="Mukta, sans-serif" font-size="14" fill="#4A2E1E">{inr(right_val)}</text>
    </svg>"""


# ------------------------------------------------------------------ data
@st.cache_data
def load():
    needed = ["predictions.csv", "weekly_panel.csv", "market_month.csv"]
    missing = [f for f in needed if not (DATA / f).exists()]
    if missing:
        return None, missing
    d = {
        "pred": pd.read_csv(DATA / "predictions.csv", parse_dates=["week"]),
        "panel": pd.read_csv(DATA / "weekly_panel.csv", parse_dates=["week"]),
        "mm": pd.read_csv(DATA / "market_month.csv"),
    }

    # LED board: latest price in the busiest mandis, and the change over the last 4 weeks
    p = d["panel"]
    last = p["week"].max()
    items = []
    for crop in ["Onion", "Maize"]:
        pc = p[p["commodity"] == crop]
        busiest = pc.groupby("market")["n_reports"].sum().nlargest(9).index
        for mkt in busiest:
            s = pc[pc["market"] == mkt].set_index("week")["price_filled"]
            now, before = s.get(last), s.get(last - pd.Timedelta(weeks=4))
            if pd.notna(now) and pd.notna(before):
                items.append((crop, mkt, now, 100 * (now / before - 1)))
    d["ticker"], d["last_week"] = items, last
    return d, []


# ------------------------------------------------------------------ economics (same as Notebook 6)
def net_gain(price_now, log_change, loss, rent, rate):
    keep = (1 - loss) ** MONTHS_STORED
    return (price_now[:, None] * np.exp(log_change) * keep - rent * MONTHS_STORED
            - price_now[:, None] * rate * H / 52 - price_now[:, None])


def with_sell(g):
    return np.column_stack([np.zeros(len(g)), g])


def simulate(e, loss, rent, rate):
    P = e["price_filled"].to_numpy()
    actual = with_sell(net_gain(P, e[[f"target_{h}w" for h in H]].to_numpy(), loss, rent, rate))
    seasonal = with_sell(net_gain(P, e[[f"base_seasonal_{h}w" for h in H]].to_numpy(), loss, rent, rate))
    forecast = with_sell(net_gain(P, e[[f"pred_{h}w_best" for h in H]].to_numpy(), loss, rent, rate))
    pessimistic = with_sell(net_gain(P, e[[f"q10_{h}w" for h in H]].to_numpy(), loss, rent, rate))
    ok = pessimistic >= -0.10 * P[:, None]
    ok[:, 0] = True
    n = len(e)
    choices = {
        "Sell at harvest": np.zeros(n, dtype=int), "Always store 4 weeks": np.full(n, 1),
        "Always store 8 weeks": np.full(n, 2), "Always store 12 weeks": np.full(n, 3),
        "Seasonal rule": seasonal.argmax(1), "Forecast rule": forecast.argmax(1),
        "Risk-aware forecast": np.where(ok, forecast, -np.inf).argmax(1), "Perfect foresight": actual.argmax(1),
    }
    return {r: (c, actual[np.arange(n), c]) for r, c in choices.items()}


def sample_log_change(q10, q50, q90, n, rng, centre=None):
    # piecewise-linear quantile function through the 10th, 50th and 90th percentiles;
    # if centre is given, the distribution is shifted so its median equals the main model's forecast
    q10, q50, q90 = np.sort([q10, q50, q90])
    if centre is not None:
        q10, q90, q50 = q10 - q50 + centre, q90 - q50 + centre, centre
    u = rng.uniform(0.005, 0.995, n)
    return np.where(u < 0.5, q50 - (0.5 - u) / 0.4 * (q50 - q10), q50 + (u - 0.5) / 0.4 * (q90 - q50))


def harvest_months(mm, crop):
    s = mm[(mm["commodity"] == crop) & mm["full_year"] & (mm["year"] <= 2015)]
    return sorted(s.groupby("month")["seasonal_rel"].median().nsmallest(3).index.tolist())


def khata(df, title, kn, money_cols=(), pct_cols=(), highlight=None, note=""):
    # renders a table as a page of a ledger book; positive money in green (jama), negative in red (naam)
    head = "".join(f'<th class="{"num" if c in money_cols or c in pct_cols else ""}">{c}</th>' for c in df.columns)
    body = ""
    for _, r in df.iterrows():
        cells = ""
        for c in df.columns:
            v = r[c]
            if c in money_cols and pd.notna(v):
                cells += f'<td class="num {"jama" if v >= 0 else "naam"}">{inr(v, sign=True)}</td>'
            elif c in pct_cols:
                cells += f'<td class="num">{"-" if pd.isna(v) else f"{100 * v:.0f}%"}</td>'
            else:
                cells += f"<td>{v}</td>"
        body += f'<tr class="{"hl" if highlight is not None and r.iloc[0] == highlight else ""}">{cells}</tr>'
    md(f'<div class="khata"><div class="kt">{title}<span>{kn}</span></div>'
       f'<table><tr>{head}</tr>{body}</table>'
       f'{f"<div class=note>{note}</div>" if note else ""}</div>')


# ------------------------------------------------------------------ farmer forecast (any date, any district)
# The Notebook 5 models need the last 52 weeks of prices, which a farmer does not have for dates after
# April 2026. This smaller model needs only what a farmer knows: district, date and today's price.
# It learns from 2010-2026 how prices moved 4, 8 and 12 weeks after a week when the price was
# X% above or below its normal level, at that time of year, in that district.
FARMER_X = ["dev", "sin", "cos", "dcode"]
CAL_START = pd.Timestamp("2022-01-01")     # 2022-2026 is used to widen the ranges (conformal step)


def district_table(crop):
    panel = load()[0]["panel"]
    return (panel[panel["commodity"] == crop].groupby(["week", "district"])["price_filled"]
            .median().unstack().sort_index())


def farmer_frame(d):
    parts = []
    for dist in d.columns:
        s = d[dist]
        ref = s.shift(1).rolling(52, min_periods=26).median()        # normal level: median of the last 52 weeks
        f = pd.DataFrame({"district": dist, "week": s.index, "P": s.values, "dev": np.log(s / ref).values})
        for h in H:
            f[f"t{h}"] = np.log(s.shift(-h) / s).values             # what happened h weeks later
        parts.append(f)
    f = pd.concat(parts).dropna(subset=["P", "dev"])
    woy = f["week"].dt.isocalendar().week.astype(int).clip(1, 52)
    f["woy"] = woy
    f["sin"], f["cos"] = np.sin(2 * np.pi * woy / 52), np.cos(2 * np.pi * woy / 52)
    f["dcode"] = f["district"].map({c: i for i, c in enumerate(d.columns)})
    return f


@st.cache_resource(show_spinner="Training the farmer forecast (first time only, about half a minute)...")
def farmer_models(crop):
    d = district_table(crop)
    f = farmer_frame(d)

    def fit(rows, h, q):
        return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=200, learning_rate=0.03,
                                             max_leaf_nodes=7, min_samples_leaf=200, categorical_features=[3],
                                             random_state=0).fit(rows[FARMER_X], rows[f"t{h}"])

    models, adj = {}, {}
    for h in H:
        rows = f.dropna(subset=[f"t{h}"])
        early, cal = rows[rows["week"] < CAL_START], rows[rows["week"] >= CAL_START]
        lo, hi = fit(early, h, 0.1).predict(cal[FARMER_X]), fit(early, h, 0.9).predict(cal[FARMER_X])
        y = cal[f"t{h}"].to_numpy()
        adj[h] = float(np.quantile(np.maximum(lo - y, y - hi), 0.8))   # how much to widen the 80% range
        models[h] = {q: fit(rows, h, q) for q in (0.1, 0.5, 0.9)}     # final models use all years
    yearly = np.log(d.median(axis=1)).groupby(d.index.year).median()
    growth = float(np.polyfit(yearly.index, yearly.values, 1)[0])      # long-run price rise per year (log)
    season = f.groupby(["district", "woy"])["dev"].median()            # usual level at each time of year
    return dict(models=models, adj=adj, d=d, growth=growth, season=season, codes=list(d.columns))


def farmer_forecast(fm, district, when, price):
    when = pd.Timestamp(when)
    s = fm["d"][district].dropna()
    past = s[s.index < when]
    ref = past.iloc[-52:].median()
    gap = max((when - past.index[-1]).days, 0) / 365.25
    ref *= np.exp(fm["growth"] * gap)                                  # carry the level forward past April 2026
    woy = min(int(when.isocalendar()[1]), 52)
    normal = ref * np.exp(fm["season"].get((district, woy), 0.0))
    x = pd.DataFrame([{"dev": np.log(price / ref), "sin": np.sin(2 * np.pi * woy / 52),
                       "cos": np.cos(2 * np.pi * woy / 52), "dcode": fm["codes"].index(district)}])
    fc = {}
    for h in H:
        q = {k: float(m.predict(x)[0]) for k, m in fm["models"][h].items()}
        q10, q50, q90 = np.sort([q[0.1], q[0.5], q[0.9]])
        fc[h] = dict(mid=q50, q10=q10 - fm["adj"][h], q50=q50, q90=q90 + fm["adj"][h])
    return fc, normal, gap


# ------------------------------------------------------------------ the decision page (used by two tabs)
def decision_view(P, fc, place, when, crop, quintals, params, hist=None, after=None):
    # fc = {h: {'mid','q10','q50','q90'}} forecast log price changes for 4, 8 and 12 weeks
    sell_value = P * quintals
    # expected gain of each option, and 5,000 possible futures to get the odds
    point = with_sell(net_gain(np.array([P]), np.array([[fc[h]['mid'] for h in H]]), **params))[0]
    rng = np.random.default_rng(42)
    chance = [np.nan]
    for k, h in enumerate(H):
        x = sample_log_change(fc[h]['q10'], fc[h]['q50'], fc[h]['q90'], 5000, rng,
                              centre=fc[h]['mid'])
        g = net_gain(np.array([P]), np.tile(x[:, None], (1, 3)), **params)[:, k]
        chance.append(np.mean(g > 0))
    options = ["Sell now"] + [f"Store {h} weeks" for h in H]
    best = int(np.argmax(point))
    top_store = 1 + int(np.argmax(point[1:]))     # best storage option, even if selling wins

    # rate card + balance scale
    c1, c2 = st.columns([1, 1.25])
    with c1:
        md(f"""
        <div class="rate">
          <div class="k">ಇಂದಿನ ಧಾರಣೆ <span>· today's rate</span></div>
          <div class="m">{place} · {crop} ({KANNADA[crop]})</div>
          <div class="p">{inr(P)}<small> / quintal</small></div>
          <span class="chip">{when:%d %b %Y}</span>
          <div class="r">Your lot: <b>{quintals} quintals</b> = <b>{inr(sell_value)}</b> if sold today</div>
        </div>""")
    with c2:
        lean = "store" if best > 0 else "sell"
        frame(f"""<div class="scale-wrap">{tarazu_svg(sell_value, sell_value + point[top_store] * quintals, options[top_store], crop)}
              <div class="verdict" style="color:{'#1F7A3A' if lean == 'store' else BRICK}">
              {'The scale tips towards storing' if lean == 'store' else 'The scale tips towards selling now'}</div></div>""", 290)

    # three sacks
    plank("Three ways to store", "ಸಂಗ್ರಹಣೆ", "forecast price change and what is left in the sack")
    cards = ""
    for k, h in enumerate(H):
        mid = 100 * np.expm1(fc[h]['mid'])
        lo, hi = 100 * np.expm1(fc[h]['q10']), 100 * np.expm1(fc[h]['q90'])
        keep = (1 - params["loss"]) ** MONTHS_STORED[k]
        gain = point[k + 1] * quintals
        wins = int(round(10 * chance[k + 1]))
        icons = "".join(crop_icon(crop, 22, faded=i >= wins) for i in range(10))
        is_best = best == k + 1
        cards += f"""
        <div class="sack-card {'best' if is_best else ''}">
          {'<div class="tag">BEST BET</div>' if is_best else ''}
          {sack_svg(str(h), keep, CROP_COLOR[crop] if crop == 'Onion' else '#8A5A00')}
          <div class="big {'pos' if mid >= 0 else 'neg'}">{mid:+.0f}%</div>
          <div class="rng">likely {lo:+.0f}% to {hi:+.0f}%</div>
          <div class="money">vs selling now: <b class="{'pos' if gain >= 0 else 'neg'}">{inr(gain, sign=True)}</b></div>
          <div class="odds">wins in <b>{wins} of 10</b> futures</div>
          <div class="icons">{icons}</div>
        </div>"""
    frame(f'<div class="sacks" style="padding-top:14px">{cards}</div>', 400)

    # receipt with the verdict + forecast chart
    a, b = st.columns([1, 1.35])
    with a:
        h = H[top_store - 1]
        m = MONTHS_STORED[top_store - 1]
        fut_rate = P * np.exp(fc[h]['mid'])
        left_q = quintals * (1 - params["loss"]) ** m
        costs = params["rent"] * m * quintals + P * params["rate"] * h / 52 * quintals
        net = fut_rate * left_q - costs
        stamp = ('<div class="stamp sell">SELL NOW<small>ಈಗಲೇ ಮಾರಿ</small></div>' if best == 0 else
                 f'<div class="stamp go">STORE {H[best - 1]} WEEKS<small>ಸಂಗ್ರಹಿಸಿ</small></div>')
        md(f"""
        <div class="patti">
          <div class="hd"><div class="kn">ಮಾರಾಟ ಪಟ್ಟಿ</div><div class="t">PATTI</div>
            <div class="s">{place.upper()} · {when:%d-%m-%Y} · {crop.upper()} {quintals} Q</div></div>
          <div class="ln tot"><span>A. Sell now</span><span class="d"></span><b>{inr(sell_value)}</b></div>
          <div class="sec">B. Store {h} weeks</div>
          <div class="ln"><span>Expected rate</span><span class="d"></span><b>{inr(fut_rate)}/q</b></div>
          <div class="ln neg"><span>Crop left after loss</span><span class="d"></span><b>{left_q:,.1f} q</b></div>
          <div class="ln neg"><span>Rent + interest</span><span class="d"></span><b>{inr(-costs)}</b></div>
          <div class="ln tot"><span>B. Expected net</span><span class="d"></span><b>{inr(net)}</b></div>
          <div class="ln tot {'neg' if net < sell_value else ''}"><span>B minus A</span><span class="d"></span><b>{inr(net - sell_value, sign=True)}</b></div>
          <div class="foot"><span></span>{stamp}</div>
        </div>""")
    with b:
        w = pd.Timestamp(when)
        xs = [w] + [w + pd.Timedelta(weeks=int(x)) for x in H]
        mid_p = [P] + [P * np.exp(fc[x]['mid']) for x in H]
        lo_p = [P] + [P * np.exp(fc[x]['q10']) for x in H]
        hi_p = [P] + [P * np.exp(fc[x]['q90']) for x in H]
        keep = (1 - params["loss"]) ** MONTHS_STORED
        be = [P] + list((P + params["rent"] * MONTHS_STORED + P * params["rate"] * H / 52) / keep)
        col = CROP_COLOR[crop]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi_p + lo_p[::-1], fill="toself", mode="lines", line=dict(width=0),
                                 fillcolor="rgba(168,50,90,.16)" if crop == "Onion" else "rgba(224,165,38,.22)",
                                 name="Likely range", hoverinfo="skip"))
        if hist is not None and len(hist):
            fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Price so far", line=dict(color=SOIL, width=2.4)))
        fig.add_trace(go.Scatter(x=xs, y=mid_p, name="Forecast", mode="lines+markers",
                                 line=dict(color=col, width=3, dash="dash"), marker=dict(size=9, color=col)))
        fig.add_trace(go.Scatter(x=xs, y=be, name="Break-even price", mode="lines+markers",
                                 line=dict(color="#26408B", width=2.2, dash="dot"), marker=dict(size=7, symbol="x")))
        if after is not None and len(after):
            fig.add_trace(go.Scatter(x=after.index, y=after.values, mode="markers", name="What actually happened",
                                     marker=dict(color="#1F3B2D", size=7, symbol="diamond")))
        fig.update_yaxes(title="Rs per quintal")
        show_chart(style(fig, 440, "Forecast vs break-even"))
        md(f'<p style="font-size:.9rem;color:{MUTED};margin-top:-8px">Storing pays only when the forecast is above '
           f'the break-even line.</p>')


# ------------------------------------------------------------------ page
md(f"<style>{CSS}</style>")
data, missing = load()

md(f"""
<div class="board"><div class="board-in">
  <div>{onion_svg(92)}</div>
  <div class="board-txt">
    <div class="board-top">ಕೃಷಿ ಉತ್ಪನ್ನ ಮಾರುಕಟ್ಟೆ &nbsp;·&nbsp; <b>KARNATAKA MANDI DESK</b></div>
    <div class="board-title">Sell Now <span>or</span> Store?</div>
    <div class="board-kn">ಈಗ ಮಾರಬೇಕೆ, ಸಂಗ್ರಹಿಸಬೇಕೆ?</div>
    <p class="board-sub">Onion and maize prices from Karnataka's APMC mandis &nbsp;·&nbsp; forecasts for 4, 8 and 12 weeks</p>
  </div>
  <div>{maize_svg(92)}</div>
</div></div>
""")

if data is None:
    st.error(f"Missing files in {DATA}: {', '.join(missing)}. Run Notebooks 2-6 first.")
    st.stop()

pred, panel, mm = data["pred"], data["panel"], data["mm"]

# LED price board (Mandi bhav)
run = ""
for c, mkt, now, ch in data["ticker"]:
    cls, arrow = ("up", "▲") if ch >= 0 else ("dn", "▼")
    run += f'<span>{c.upper()} · {mkt.upper()} {inr(now)} <b class="{cls}">{arrow} {abs(ch):.1f}%</b></span>'
md(f'<div class="led"><div class="led-tag">MANDI BHAV {data["last_week"]:%d-%m-%Y}</div>'
   f'<div class="led-run" style="padding-left:260px">{run}{run}</div></div>')

# ------------------------------------------------------------------ sidebar: only 4 inputs
with st.sidebar:
    md('<div class="chalk-h">Your crop <small>ನಿಮ್ಮ ಬೆಳೆ</small></div>')
    crop = st.radio("Crop", ["Onion", "Maize"], horizontal=True,
                    format_func=lambda c: f"{'🧅' if c == 'Onion' else '🌽'} {c}")
    markets = pred[pred["commodity"] == crop]["market"].value_counts().index.tolist()
    market = st.selectbox("Your mandi", sorted(markets), index=sorted(markets).index(markets[0]))
    quintals = st.number_input("Quantity (quintals)", min_value=1, max_value=10000, value=50, step=5)
    preset = st.selectbox("Storage type", [s for s in SCENARIOS[crop] if not s.startswith("No costs")])
    p = SCENARIOS[crop][preset]
    params = dict(loss=p["loss"] / 100, rent=p["rent"], rate=p["rate"] / 100)
    md(f'<div class="chalk-note" style="margin-top:10px">This storage loses <b>{p["loss"]:g}%</b> of the crop a month, '
       f'costs <b>Rs {p["rent"]:g}</b> per quintal a month, plus <b>{p["rate"]:g}%</b> interest a year.</div>')
    md('<div class="chalk-note" style="margin-top:18px;opacity:.85">FDS project · Amrita, Bengaluru<br>'
       'Janani Sree · Shreyash Shanmukh Piriya</div>')

tab_farmer, tab_decide, tab_prices, tab_backtest = st.tabs(
    ["🧑‍🌾  My crop today", "⚖️  Past weeks", "📈  Prices", "📒  Khata: back-test"])

# ------------------------------------------------------------------ tab 0: farmer enters today's details
with tab_farmer:
    fm = farmer_models(crop)
    md('<p>Tell the mandi desk about your crop. The forecast says where the price may be in 4, 8 and 12 weeks '
       'and whether storing beats selling today, with your storage type from the left.</p>')
    dists = fm["codes"]
    home = panel.loc[(panel["commodity"] == crop) & (panel["market"] == market), "district"]
    c1, c2, c3 = st.columns(3)
    with c1:
        district = st.selectbox("Your district", dists,
                                index=dists.index(home.iloc[0]) if len(home) and home.iloc[0] in dists else 0)
    with c2:
        when = st.date_input("Date", value=pd.Timestamp.today().date(),
                             min_value=pd.Timestamp("2011-01-03").date(), max_value=pd.Timestamp("2027-12-31").date())
    _, normal, gap = farmer_forecast(fm, district, when, 1000.0)
    with c3:
        price = st.number_input("Today's mandi price (Rs per quintal)", min_value=100, max_value=20000,
                                value=int(round(normal, -1)), step=50, key=f"price-{crop}-{district}-{when}",
                                help="The modal price you were offered at the mandi today.")
    fc, normal, gap = farmer_forecast(fm, district, when, float(price))
    diff = 100 * (price / normal - 1)
    md(f'<div class="chalk-note" style="background:{BOARD};padding:10px 16px;border-radius:8px;border:6px solid #7A4E2D;'
       f'font-size:1.02rem">A normal {crop.lower()} price in {district} at this time of year is about '
       f'<b style="color:#F7D046">{inr(normal)}</b>. '
       + ("Your price is about normal." if abs(diff) < 1 else
          f'Your price is <b style="color:#F7D046">{abs(diff):.0f}% {"above" if diff > 0 else "below"}</b> normal.')
       + (f' (Our data ends in April 2026, so the normal level is carried forward {gap:.1f} years with the '
          f'long-run price trend.)' if gap > 0.1 else '') + '</div>')
    dser = fm["d"][district].dropna()
    w = pd.Timestamp(when)
    decision_view(float(price), fc, f"{district} district", w, crop, quintals, params,
                  hist=dser.loc[w - pd.Timedelta(weeks=30):w] if w <= dser.index[-1] + pd.Timedelta(weeks=1) else None,
                  after=dser.loc[w + pd.Timedelta(days=1):w + pd.Timedelta(weeks=12)])
    md(f'<p style="font-size:.88rem;color:{MUTED}">How this forecast works: a model trained on 2010-2026 Karnataka '
       f'prices learns how prices moved after weeks when they were above or below normal, at each time of year, in '
       f'each district. Tested on 2019-2026 it was more accurate than the seasonal average for onion '
       f'(12-week error 0.34 vs 0.38) and about equal for maize; its ranges held the real price 80-86% of the time. '
       f'A forecast gives odds, not certainty.</p>')

# ------------------------------------------------------------------ tab 1: decision helper
with tab_decide:
    rows = pred[(pred["commodity"] == crop) & (pred["market"] == market)].sort_values("week")
    rows = rows.dropna(subset=["price_filled", "pred_12w_best", "q10_12w"])
    if rows.empty:
        st.warning("No forecasts for this market.")
    else:
        weeks = rows["week"].dt.date.tolist()
        week = st.select_slider("Decision week (test period, Nov 2019 to Apr 2026)", options=weeks, value=weeks[-1])
        row = rows[rows["week"].dt.date == week].iloc[0]
        pm = panel[(panel["commodity"] == crop) & (panel["market"] == market)].set_index("week").sort_index()
        w = row["week"]
        fc = {h: dict(mid=row[f"pred_{h}w_best"], q10=row[f"q10_{h}w"], q50=row[f"q50_{h}w"], q90=row[f"q90_{h}w"]) for h in H}
        decision_view(row["price_filled"], fc, f"{market} APMC", w, crop, quintals, params,
                      hist=pm.loc[w - pd.Timedelta(weeks=30):w, "price_filled"],
                      after=pm.loc[w + pd.Timedelta(days=1):w + pd.Timedelta(weeks=12), "price"].dropna())

# ------------------------------------------------------------------ tab 2: prices
with tab_prices:
    pmk = panel[(panel["commodity"] == crop) & (panel["market"] == market)].sort_values("week")
    plank(f"{crop} price at {market}", f"{KANNADA[crop]} ಬೆಲೆ", "weekly modal price")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pmk["week"], y=pmk["state_median"], name="Karnataka median", line=dict(color=MUTED, width=1)))
    fig.add_trace(go.Scatter(x=pmk["week"], y=pmk["price_filled"], name=market,
                             line=dict(color=CROP_COLOR[crop] if crop == "Onion" else "#B07C00", width=1.8)))
    fig.update_yaxes(title="Rs per quintal")
    show_chart(style(fig, 400))

    plank("Crop calendar", "ಬೆಳೆ ಕ್ಯಾಲೆಂಡರ್", "price in each month vs the yearly average: green = cheap (harvest)")
    s = mm[(mm["commodity"] == crop) & mm["full_year"] & (mm["year"] <= 2025)]
    vals = (s[s["market"] == market].groupby("month")["seasonal_rel"].median().reindex(range(1, 13))
            .fillna(s.groupby("month")["seasonal_rel"].median()) * 100)
    colors = [LEAF if v < 100 else (ONION_RED if crop == "Onion" else "#C98A00") for v in vals]
    fig = go.Figure(go.Barpolar(r=vals, theta=MONTH_NAMES, marker_color=colors, marker_line_color="#FFFBEF",
                                marker_line_width=2, hovertemplate="%{theta}: %{r:.0f}% of yearly average<extra></extra>"))
    fig.update_layout(polar=dict(hole=0.3, bgcolor="rgba(255,251,239,.85)",
                                 radialaxis=dict(range=[vals.min() - 8, vals.max() + 4], ticksuffix="%", tickfont=dict(size=10)),
                                 angularaxis=dict(direction="clockwise", rotation=90)), showlegend=False)
    show_chart(style(fig, 440))

# ------------------------------------------------------------------ tab 3: back-test as a khata
with tab_backtest:
    months = harvest_months(mm, crop)
    md(f'<p>Every harvest-time decision ({", ".join(MONTH_NAMES[m - 1] for m in months)}) from 2020 to 2026, replayed '
       f'with your storage type. Gain over selling at harvest, in Rs per quintal: '
       f'<span style="color:#1F7A3A;font-weight:700">green = profit</span>, '
       f'<span style="color:{BRICK};font-weight:700">red = loss</span>.</p>')
    e = pred[(pred["commodity"] == crop) & pred["week"].dt.month.isin(months) & pred["price"].notna()]
    e = e.dropna(subset=[f"target_{h}w" for h in H] + [f"pred_{h}w_best" for h in H] + [f"q10_{h}w" for h in H])
    if len(e) < 5:
        st.warning("Too few harvest-time decisions to replay.")
        st.stop()
    res = simulate(e, **params)
    summ = pd.DataFrame([{"Rule": r, "Mean gain (Rs/q)": g.mean(), "Worse than selling": (g < 0).mean(),
                          "Stored in": (c > 0).mean()} for r, (c, g) in res.items()])
    fr = summ.set_index("Rule").loc["Forecast rule", "Mean gain (Rs/q)"]
    khata(summ, f"{crop} khata · {len(e):,} decisions", "ಲೆಕ್ಕದ ಪುಸ್ತಕ",
          money_cols=["Mean gain (Rs/q)"], pct_cols=["Worse than selling", "Stored in"], highlight="Forecast rule",
          note=f"For your {quintals} quintals, the forecast rule averages {inr(fr * quintals, sign=True)} "
               f"per decision compared with selling at harvest.")
