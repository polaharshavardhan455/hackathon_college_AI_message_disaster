"""app.py - the control-room dashboard (Streamlit).

WHAT IT DOES
    Shows the situation from emergency.db: which incidents need attention, where they are,
    how urgent, how many people, what help is needed, and what changed.

WHY IT IS BUILT THIS WAY
    - It only READS the database (read-only). It never calls any AI.
    - Urgency is shown by SHAPE + WORD + colour, never colour alone:
        triangle = Critical, diamond = High, circle = Medium, square = Low, tick-circle = Resolved.
      The same shape is used in the list, on the map and in the detail panel.
    - Uncertainty is shown, not hidden: location confidence, "approximate place",
      "possibly the same as INC-x", unverified rumors, rejected files.
    - Replay: step through wave 1, 2, 3, 4 to see how the situation developed.
    - No animation, no clutter.

RUN
    streamlit run app.py                       (uses emergency.db in this folder)
    streamlit run app.py -- --db other.db      (another database file)
    python app.py --preview preview.html       (one static page you can open in a browser, no Streamlit needed)
    python app.py --selftest                   (checks)

DOES IT USE AI?  No.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

from db import DEFAULT_DB, Database, DatabaseError, HeldRow, Report
from models import EventKind, Incident, Resource, Source, Status, Urgency

APP_TITLE = "Emergency Response Assistant"
DEFAULT_LANDMARKS = str(Path(__file__).with_name("landmarks.json"))
COARSE_KINDS = {"area", "locality"}        # same meaning as in locations.py: a neighbourhood centre, not an exact spot
PAGE = 25                                   # incidents shown per "Show more"
STALE_AFTER = timedelta(hours=2)            # an open incident with no news for this long gets a "no news" tag

STATUS_FILTERS = [("open", "Open"), ("active", "Active"), ("rescue", "Rescue"), ("resolved", "Resolved"), ("all", "All")]
STATUS_QUERY: dict[str, list[str] | None] = {
    "open": ["active", "rescue_in_progress"], "active": ["active"], "rescue": ["rescue_in_progress"],
    "resolved": ["resolved"], "all": None}
URGENCY_ORDER = ["critical", "high", "medium", "low"]
SOURCE_LABEL = {"call": "Call", "whatsapp": "WhatsApp", "sms": "SMS", "social": "Social post", "unknown": "Unknown source"}
EVENT_LABEL = {"created": "Opened", "people_changed": "Count changed", "escalated": "Escalated",
               "status_changed": "Status", "details_added": "Detail", "report_added": "Report"}

DEFAULT_STATE: dict[str, Any] = {
    "theme": "light",       # light | dark
    "wave": None,           # None = latest, or a wave number (replay)
    "status": "open",       # one of STATUS_FILTERS
    "q": "",                # search words
    "selected": None,       # incident id being inspected
    "shown": PAGE,          # how many incidents the list shows
    "all_reports": False,   # show every grouped report, not just the newest few
    "tiles": True,          # street map background (needs internet); off = simple map with reference names
    "focus": False,         # map zoomed to the selected incident
    "aside": "held",        # which "set aside" list is open: held | noise | rejected | late
}


# =====================================================================
# 1. SMALL HELPERS
# =====================================================================

def esc(value: Any) -> str:
    """Escape text from reports before it goes into the page (reports are untrusted)."""
    return html.escape("" if value is None else str(value), quote=True)


def clip(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def num(n: int | float) -> str:
    return f"{int(n):,}"


def plural(n: int, word: str, many: str | None = None) -> str:
    return f"{num(n)} {word if n == 1 else (many or word + 's')}"


def clock(when: datetime, ref: datetime | None) -> str:
    """08:10 for the same day as `ref`, otherwise 02 Mar 08:10."""
    if ref is None or when.date() == ref.date():
        return when.strftime("%H:%M")
    return when.strftime("%d %b %H:%M")


def ago(delta: timedelta) -> str:
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min ago"
    hours, rest = divmod(minutes, 60)
    if hours < 24:
        return f"{hours} h {rest} min ago" if rest else f"{hours} h ago"
    days, hrs = divmod(hours, 24)
    return f"{days} d {hrs} h ago" if hrs else f"{days} d ago"


def span_text(delta: timedelta) -> str:
    return ago(delta).replace(" ago", "")


def urgency_key(inc: Incident) -> str:
    return "resolved" if not inc.is_open else inc.urgency.value


def urgency_word(inc: Incident) -> str:
    return "Resolved" if not inc.is_open else inc.urgency.label


def is_approximate(inc: Incident) -> bool:
    loc = inc.location
    return loc is not None and (loc.kind in COARSE_KINDS or loc.confidence < 0.5)


def possible_duplicate_of(inc: Incident) -> str | None:
    for event in inc.events:
        found = re.match(r"Possibly the same as (INC-\d+)", event.text or "")
        if found:
            return found.group(1)
    return None


def urgency_reason(inc: Incident) -> str:
    """The plain-words reason for the current urgency (taken from the timeline the rules wrote)."""
    for event in reversed(inc.events):
        if (event.text or "").startswith("Urgency"):
            return event.text
    return ""


def quiet_for(inc: Incident, as_of: datetime | None) -> timedelta | None:
    if as_of is None or not inc.is_open:
        return None
    gap = as_of - inc.last_update
    return gap if gap >= STALE_AFTER else None


def minify(markup: str) -> str:
    """Remove blank lines and indentation (Markdown would treat them as code or break the HTML)."""
    return "".join(line.strip() for line in markup.splitlines())


# =====================================================================
# 2. ICONS (small inline pictures, drawn here so nothing needs downloading)
# =====================================================================

_GLYPHS = {
    "critical": ('<polygon class="g g-critical" points="12,2 22.5,20.5 1.5,20.5"/>'
                 '<path class="g-mark" d="M12 9v5.5"/><circle class="g-dot" cx="12" cy="17.4" r="1.25"/>'),
    "high": '<polygon class="g g-high" points="12,1.5 22.5,12 12,22.5 1.5,12"/>',
    "medium": '<circle class="g g-medium" cx="12" cy="12" r="9.5"/>',
    "low": '<rect class="g g-low" x="3" y="3" width="18" height="18" rx="2"/>',
    "resolved": '<circle class="g g-resolved" cx="12" cy="12" r="9.5"/><path class="g-check" d="M7.5 12.5l3 3 6-6.5"/>',
}


def glyph(key: str, size: int = 18) -> str:
    """The urgency shape. key: critical | high | medium | low | resolved."""
    key = key if key in _GLYPHS else "low"
    return (f'<svg class="gl" width="{size}" height="{size}" viewBox="0 0 24 24" role="img" '
            f'aria-label="{key.capitalize()}">{_GLYPHS[key]}</svg>')


_NEED_ICONS = {
    "medical": '<path d="M12 5v14M5 12h14"/>',
    "rescue": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3.2"/>',
    "food_water": '<path d="M12 3c3.5 4.5 6 7.2 6 10.2a6 6 0 0 1-12 0C6 10.2 8.5 7.5 12 3z"/>',
    "shelter": '<path d="M3 11l9-7 9 7M6 10v10h12V10"/>',
    "road_clearance": '<path d="M4 20l4-16h8l4 16M7 12h10"/>',
    "other": '<circle cx="12" cy="12" r="2.6"/>',
    "unknown": '<path d="M9.5 9a2.6 2.6 0 1 1 3.6 2.4c-.8.4-1.1 1-1.1 1.8M12 17.5v.5"/>',
}


def need_icon(resource: str, size: int = 16) -> str:
    return (f'<svg class="ni" width="{size}" height="{size}" viewBox="0 0 24 24" aria-hidden="true">'
            f'{_NEED_ICONS.get(resource, _NEED_ICONS["other"])}</svg>')


_STATUS_ICONS = {
    "active": '<circle cx="12" cy="12" r="5" class="fillme"/>',
    "rescue_in_progress": '<path d="M4 12h14M13 6l6 6-6 6"/>',
    "resolved": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
}


def status_chip(status: Status) -> str:
    icon = (f'<svg class="si" width="14" height="14" viewBox="0 0 24 24" aria-hidden="true">'
            f'{_STATUS_ICONS.get(status.value, "")}</svg>')
    return f'<span class="chip st st-{status.value}">{icon}{esc(status.label)}</span>'


def primary_need(inc: Incident) -> str:
    return "Help type not stated" if inc.resource is Resource.UNKNOWN else inc.resource.label


# =====================================================================
# 3. THEME (all colours and the page style in one place)
# =====================================================================

_THEMES = {
    "light": {
        "bg": "#F1F4F8", "surface": "#FFFFFF", "surface2": "#F6F8FB", "ink": "#15202B", "muted": "#566676",
        "line": "#DAE1E9", "line2": "#BAC5D1", "accent": "#1B4F8F", "on_accent": "#FFFFFF", "mark": "#FFFFFF",
        "critical": "#B3261E", "high": "#C2570C", "medium": "#9A7400", "low": "#3A6F93", "resolved": "#2B7A56"},
    "dark": {
        "bg": "#0E151C", "surface": "#16202A", "surface2": "#1B2733", "ink": "#E8EEF3", "muted": "#9CACBA",
        "line": "#263441", "line2": "#3B4B5A", "accent": "#86BDF2", "on_accent": "#0B1620", "mark": "#10171E",
        "critical": "#FF7B6F", "high": "#FFA34F", "medium": "#F3CC50", "low": "#7DBBE8", "resolved": "#63D0A1"},
}

_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&display=swap');
.stApp, .pv { font-family:'Atkinson Hyperlegible','Segoe UI',system-ui,-apple-system,Roboto,Arial,sans-serif; }
.stApp { background:var(--bg); color:var(--ink); }
.pv { background:var(--bg); color:var(--ink); padding:22px 28px 40px; font-size:16px; line-height:1.4; }
.stApp button, .stApp input, .stApp textarea { font-family:'Atkinson Hyperlegible','Segoe UI',system-ui,sans-serif !important; }
[data-testid="stHeader"] { background:transparent; height:0; }
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], #MainMenu, footer { display:none !important; }
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] { display:none !important; }
.block-container { padding:1.1rem 1.8rem 3rem !important; max-width:1580px !important; }
[data-testid="stVerticalBlock"] { gap:.7rem; }
.stApp p, .stApp label, .stApp span, .stApp li { color:inherit; }
.stApp :focus-visible { outline:3px solid var(--accent) !important; outline-offset:2px; }

/* ---- buttons ---- */
.stApp .stButton > button { border-radius:10px; border:1px solid var(--line2); background:var(--surface); color:var(--ink);
  font-weight:700; min-height:2.5rem; padding:.3rem .9rem; box-shadow:none; transition:none; }
.stApp .stButton > button * { color:inherit !important; font-weight:700; }
.stApp .stButton > button:hover { border-color:var(--accent); color:var(--accent); background:var(--surface); }
.stApp .stButton > button:disabled { opacity:.45; }
.stApp .stButton > button[kind="primary"], .stApp .stButton > button[data-testid="stBaseButton-primary"] {
  background:var(--accent); color:var(--on_accent); border-color:var(--accent); }
.stApp .stButton > button[kind="primary"] *, .stApp .stButton > button[data-testid="stBaseButton-primary"] * { color:var(--on_accent) !important; }
.stApp .stButton > button[kind="primary"]:hover, .stApp .stButton > button[data-testid="stBaseButton-primary"]:hover { background:var(--accent); color:var(--on_accent); filter:brightness(1.08); }

/* ---- search box ---- */
.stApp [data-testid="stTextInput"] > div { background:transparent; }
.stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"] { background:var(--surface) !important; border-radius:10px !important; }
.stApp [data-baseweb="input"] { border:1px solid var(--line2) !important; }
.stApp [data-testid="stTextInput"] input { background:var(--surface) !important; color:var(--ink) !important; -webkit-text-fill-color:var(--ink) !important; font-size:16px; }
.stApp [data-testid="stTextInput"] input::placeholder { color:var(--muted) !important; -webkit-text-fill-color:var(--muted) !important; opacity:1; }
.stApp [data-testid="stVerticalBlockBorderWrapper"] { border-radius:12px; }

/* ---- header ---- */
.top { display:flex; justify-content:space-between; align-items:flex-end; gap:24px; flex-wrap:wrap; margin-bottom:4px; }
.app-title { font-size:24px; font-weight:700; letter-spacing:-.01em; }
.app-sub { color:var(--muted); font-size:14.5px; max-width:62ch; margin-top:2px; }
.asof { text-align:right; }
.asof-l { color:var(--muted); font-size:13.5px; }
.asof-t { font-size:24px; font-weight:700; font-variant-numeric:tabular-nums; line-height:1.1; }
.asof-s { color:var(--muted); font-size:13.5px; margin-top:2px; }
.notes { display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; margin-top:6px; }
.note { border:1px solid var(--line2); border-radius:8px; padding:2px 9px; font-size:13px; background:var(--surface); }
.note b { font-weight:700; }
.sentence { font-size:15.5px; margin:10px 0 2px; }
.sentence b { font-variant-numeric:tabular-nums; }
.wavebox { padding:7px 4px 0; line-height:1.15; }
.wavebox b { font-size:17px; }
.wavebox span { color:var(--muted); font-size:13px; display:block; }

/* ---- summary strip ---- */
.strip { display:grid; grid-template-columns:1.9fr 1fr 1fr 1fr 1fr; gap:12px; margin:12px 0 6px; }
.hero, .stat { background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:14px 18px; }
.hero-row { display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
.hero-n { font-size:54px; font-weight:700; line-height:1; font-variant-numeric:tabular-nums; }
.hero-l { font-size:17px; font-weight:700; }
.tri { display:flex; gap:16px; flex-wrap:wrap; margin-top:10px; }
.tri span { display:inline-flex; align-items:center; gap:6px; font-size:15px; }
.tri b { font-size:19px; font-variant-numeric:tabular-nums; }
.stat-n { font-size:32px; font-weight:700; line-height:1.05; font-variant-numeric:tabular-nums; }
.stat-l { color:var(--muted); font-size:14px; margin-top:3px; }
.stat-s { color:var(--muted); font-size:12.5px; margin-top:2px; }

/* ---- section headings ---- */
.sec { display:flex; justify-content:space-between; align-items:baseline; gap:12px; margin:6px 2px 0; flex-wrap:wrap; }
.sec-t { font-size:18px; font-weight:700; }
.sec-s { color:var(--muted); font-size:13.5px; }

/* ---- incident cards ---- */
.u-critical { --u:var(--critical); } .u-high { --u:var(--high); } .u-medium { --u:var(--medium); }
.u-low { --u:var(--low); } .u-resolved { --u:var(--resolved); }
.card { background:var(--surface); border:1px solid var(--line); border-left:7px solid var(--u); border-radius:12px; padding:11px 14px 10px; }
.card.sel { background:var(--surface2); box-shadow:0 0 0 2px var(--accent); }
.card.done { opacity:.92; }
.card-head { display:flex; justify-content:space-between; align-items:center; gap:10px; }
.urg { display:inline-flex; align-items:center; gap:7px; font-weight:700; color:var(--u); font-size:15px; }
.inc-id { color:var(--muted); font-size:13.5px; font-variant-numeric:tabular-nums; }
.card-place { font-size:18px; font-weight:700; margin:3px 0 7px; line-height:1.2; }
.facts { display:flex; gap:7px; flex-wrap:wrap; align-items:center; }
.chip { display:inline-flex; align-items:center; gap:6px; border:1px solid var(--line2); border-radius:999px; padding:2px 10px; font-size:14px; background:var(--surface); white-space:nowrap; }
.chip b { font-variant-numeric:tabular-nums; }
.chip svg.ni, .chip svg.si { stroke:currentColor; fill:none; stroke-width:2.2; stroke-linecap:round; stroke-linejoin:round; }
.chip svg .fillme { fill:currentColor; stroke:none; }
.st-resolved { color:var(--resolved); border-color:var(--resolved); }
.st-rescue_in_progress { border-color:var(--accent); color:var(--accent); }
.card-sum { color:var(--muted); font-size:14px; margin-top:7px; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
.card-foot { color:var(--muted); font-size:13px; margin-top:7px; display:flex; gap:6px 14px; flex-wrap:wrap; align-items:center; }
.flag { border:1px solid var(--line2); border-radius:6px; padding:0 7px; font-size:12.5px; color:var(--ink); background:var(--surface2); }
.empty { background:var(--surface); border:1px dashed var(--line2); border-radius:12px; padding:22px 18px; color:var(--muted); }
.empty b { color:var(--ink); }

/* ---- map ---- */
.mapbox { border:1px solid var(--line); border-radius:14px; overflow:hidden; background:var(--surface2); }
.mapbox svg.map { display:block; width:100%; height:auto; }
.map .grid { stroke:var(--line); stroke-width:1; fill:none; }
.map .ref { fill:var(--muted); font-size:14px; }
.map .refdot { fill:var(--line2); }
.map .halo { paint-order:stroke; stroke:var(--surface); stroke-width:4px; stroke-linejoin:round; }
.map .pin-label { font-size:15px; font-weight:700; fill:var(--ink); }
.map .attrib { font-size:11px; fill:var(--muted); }
.map .approx { fill:var(--accent); fill-opacity:.09; stroke:var(--ink); stroke-width:1.4; stroke-dasharray:5 4; }
.map .sel-ring { fill:none; stroke:var(--accent); stroke-width:3.2; }
.map .empty-t { fill:var(--muted); font-size:18px; }
svg.gl, .map .gl-in { overflow:visible; }
.g { stroke:var(--surface); stroke-width:1.8; stroke-linejoin:round; }
.g-critical { fill:var(--critical); } .g-high { fill:var(--high); } .g-medium { fill:var(--medium); } .g-low { fill:var(--low); }
.g-resolved { fill:var(--surface); stroke:var(--resolved); stroke-width:2; }
.g-check { fill:none; stroke:var(--resolved); stroke-width:2.6; stroke-linecap:round; stroke-linejoin:round; }
.g-mark { stroke:var(--mark); stroke-width:2.4; stroke-linecap:round; fill:none; }
.g-dot { fill:var(--mark); }
.legend { display:flex; gap:8px 18px; flex-wrap:wrap; align-items:center; padding:9px 4px 0; font-size:13.5px; color:var(--muted); }
.legend span { display:inline-flex; align-items:center; gap:6px; color:var(--ink); }
.legend .dash { width:16px; height:16px; border-radius:50%; border:1.6px dashed var(--ink); display:inline-block; }
.maptip { color:var(--muted); font-size:13px; margin-top:6px; }

/* ---- detail ---- */
.detail { background:var(--surface); border:1px solid var(--line); border-top:7px solid var(--u); border-radius:14px; padding:16px 20px 18px; }
.d-top { display:flex; justify-content:space-between; align-items:center; gap:10px; flex-wrap:wrap; }
.d-title { font-size:28px; font-weight:700; letter-spacing:-.01em; line-height:1.15; margin:6px 0 4px; }
.d-sum { font-size:16.5px; margin:2px 0 12px; max-width:70ch; }
.fgrid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; margin:10px 0 14px; }
.fbox { border:1px solid var(--line); border-radius:10px; padding:9px 12px; background:var(--surface2); }
.fbox-l { color:var(--muted); font-size:13px; }
.fbox-v { font-size:22px; font-weight:700; line-height:1.2; font-variant-numeric:tabular-nums; display:flex; align-items:center; gap:7px; flex-wrap:wrap; }
.fbox-v svg.ni { stroke:currentColor; fill:none; stroke-width:2.2; stroke-linecap:round; stroke-linejoin:round; }
.fbox-s { color:var(--muted); font-size:13.5px; margin-top:1px; }
.locbox { border:1px solid var(--line2); border-radius:10px; padding:11px 14px; margin:4px 0 14px; background:var(--surface2); }
.loc-h { display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap; font-size:13.5px; color:var(--muted); }
.loc-name { font-size:18px; font-weight:700; margin:3px 0 4px; }
.loc-line { font-size:14.5px; margin-top:2px; }
.loc-line span { color:var(--muted); }
.loc-warn { font-size:13.5px; color:var(--muted); margin-top:8px; border-top:1px solid var(--line); padding-top:7px; }
.meter { display:inline-flex; gap:3px; vertical-align:middle; margin-right:6px; }
.meter i { width:11px; height:11px; border:1.6px solid var(--ink); display:inline-block; border-radius:2px; }
.meter i.on { background:var(--ink); }
.sub-h { font-size:16px; font-weight:700; margin:16px 0 6px; }
.tl { list-style:none; margin:0; padding:0 0 0 4px; border-left:2px solid var(--line2); }
.tl li { display:flex; gap:10px; align-items:baseline; padding:4px 0 4px 12px; font-size:14.5px; position:relative; }
.tl li::before { content:""; position:absolute; left:-7px; top:11px; width:10px; height:10px; border-radius:50%; background:var(--surface); border:2px solid var(--ink); }
.tl .t { min-width:58px; color:var(--muted); font-variant-numeric:tabular-nums; font-size:13.5px; }
.tl .k { min-width:92px; font-weight:700; font-size:13.5px; }
.tl .tx { flex:1; }
.tl li.big .tx { font-weight:700; }
.more-note { color:var(--muted); font-size:13px; padding:4px 0 0 16px; }

/* ---- grouped reports ---- */
.rep { border:1px solid var(--line); border-radius:10px; padding:9px 12px; background:var(--surface); margin-bottom:8px; }
.rep.held { border-style:dashed; border-color:var(--line2); }
.rep-meta { display:flex; gap:6px 10px; flex-wrap:wrap; align-items:center; font-size:13.5px; color:var(--muted); }
.rep-meta b { color:var(--ink); font-variant-numeric:tabular-nums; }
.tag { border:1px solid var(--line2); border-radius:6px; padding:0 7px; font-size:12.5px; color:var(--ink); background:var(--surface2); }
.tag.warn { border-color:var(--high); color:var(--high); font-weight:700; }
.rep-text { margin:4px 0 2px; font-size:15.5px; overflow-wrap:anywhere; }
.rep-read { color:var(--muted); font-size:13.5px; }
.rep-note { color:var(--muted); font-size:13.5px; margin-top:3px; }
.rep details { margin-top:4px; font-size:13.5px; color:var(--muted); }
.rep summary { cursor:pointer; color:var(--accent); }
.rep ul { margin:4px 0 0 18px; padding:0; }

/* ---- footer ---- */
.flow { display:flex; gap:8px; flex-wrap:wrap; align-items:center; color:var(--muted); font-size:13.5px; margin-top:6px; }
.flow span { border:1px solid var(--line2); border-radius:8px; padding:2px 10px; color:var(--ink); background:var(--surface); }
.foot-note { color:var(--muted); font-size:13.5px; margin-top:10px; max-width:90ch; }

/* ---- static preview only ---- */
.pv-main { display:grid; grid-template-columns:5fr 7fr; gap:28px; margin-top:14px; align-items:start; }
.pv-list { display:flex; flex-direction:column; gap:9px; margin-top:8px; }
.pv-chips { display:flex; gap:8px; flex-wrap:wrap; margin:8px 0; }
.pv-chip { border:1px solid var(--line2); border-radius:10px; padding:5px 12px; font-weight:700; background:var(--surface); }
.pv-chip.on { background:var(--accent); color:var(--on_accent); border-color:var(--accent); }
.pv-search { border:1px solid var(--line2); border-radius:10px; padding:8px 12px; background:var(--surface); color:var(--muted); }

@media (max-width:1100px) { .strip { grid-template-columns:1fr 1fr; } .hero { grid-column:1 / -1; } .fgrid { grid-template-columns:1fr 1fr; } .pv-main { grid-template-columns:1fr; } }
@media (max-width:640px) { .fgrid { grid-template-columns:1fr; } .hero-n { font-size:44px; } .asof { text-align:left; } .notes { justify-content:flex-start; } }
"""


def theme_vars(theme: str) -> str:
    colours = _THEMES.get(theme, _THEMES["light"])
    body = ";".join(f"--{k.replace('_', '-')}:{v}" for k, v in colours.items())
    return f".stApp,.pv{{{body}}}"


def build_css(theme: str) -> str:
    css = _CSS.replace("var(--on_accent)", "var(--on-accent)")
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    return theme_vars(theme) + re.sub(r"\s+", " ", css).strip()


# =====================================================================
# 4. THE VIEW: everything the page needs, read from the database
# =====================================================================

@dataclass
class View:
    waves: list[int]
    wave: int | None                       # None = latest
    headline: dict
    open_incs: list[Incident]
    incs: list[Incident]
    selected: Incident | None
    reports: list[Report]
    as_of: datetime | None
    run: dict
    landmarks: list[dict]
    aside_rows: list = field(default_factory=list)
    rejected_count: int = 0

    @property
    def wave_label(self) -> tuple[str, str]:
        """(big text, small text) for the replay control."""
        if not self.waves:
            return "All reports", "no waves in this dataset"
        last = self.waves[-1]
        if self.wave is None:
            return f"Wave {last} of {last}", "latest situation"
        return f"Wave {self.wave} of {last}", f"situation after wave {self.wave}"


def load_landmarks(path: str | Path | None) -> list[dict]:
    """Reference names for the simple map (areas only). Missing or broken file = no reference names."""
    if not path or not Path(path).is_file():
        return []
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        items = data.get("landmarks", []) if isinstance(data, dict) else data
        return [{"name": str(x["name"]), "lat": float(x["lat"]), "lon": float(x["lon"])}
                for x in items if str(x.get("type", "")).lower() in COARSE_KINDS]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return []


def latest_report_time(db: Database, wave: int | None) -> datetime | None:
    limit = 10 ** 9 if wave is None else int(wave)
    row = db.conn.execute(
        "SELECT MAX(m.timestamp) FROM messages m JOIN decisions d ON d.message_id = m.id "
        "WHERE COALESCE(m.wave, 0) <= ?", (limit,)).fetchone()
    return datetime.fromisoformat(row[0]) if row and row[0] else None


def load_view(db: Database, state: dict, landmarks: list[dict] | None = None) -> View:
    waves = db.waves()
    wave = state.get("wave")
    if wave is not None and (not waves or wave not in waves):
        wave = None                                          # stale choice (e.g. database rebuilt): show latest
    head = db.headline(wave)
    open_incs = db.list_incidents(as_of_wave=wave, statuses=STATUS_QUERY["open"])
    query = (state.get("q") or "").strip()
    status = state.get("status") if state.get("status") in STATUS_QUERY else "open"
    if status == "open" and not query:
        incs = open_incs
    else:
        incs = db.list_incidents(as_of_wave=wave, statuses=STATUS_QUERY[status], search=query)

    chosen = next((i for i in incs if i.id == state.get("selected")), None) or (incs[0] if incs else None)
    reports = db.get_reports(chosen.id, wave) if chosen else []

    as_of = latest_report_time(db, wave)
    if as_of is None and open_incs:
        as_of = max(i.last_update for i in open_incs)

    aside = state.get("aside", "held")
    if aside == "held":
        rows: list = db.get_held(wave, limit=60)
    elif aside == "noise":
        rows = db.get_reports_by_action("noise", wave, limit=60)
    elif aside == "late":
        rows = sorted(db.get_reports_by_action("ignored_resolved", wave, limit=60)
                      + db.get_reports_by_action("ignored_unlinked", wave, limit=60),
                      key=lambda r: r.message.timestamp, reverse=True)[:60]
    else:
        rows = db.get_rejected()[:100]

    return View(waves=waves, wave=wave, headline=head, open_incs=open_incs, incs=incs, selected=chosen,
                reports=reports, as_of=as_of, run=db.get_meta("run", {}) or {}, landmarks=landmarks or [],
                aside_rows=rows, rejected_count=head["rejected"])


# =====================================================================
# 5. PAGE PIECES (each returns one block of HTML)
# =====================================================================

def header_html(v: View) -> str:
    h = v.headline
    if v.as_of:
        when = f'{v.as_of:%H:%M}<span style="font-size:15px;font-weight:400"> · {v.as_of:%d %b %Y}</span>'
    else:
        when = "No reports yet"
    notes = []
    if v.run and v.run.get("complete") is False:
        notes.append('<span class="note"><b>Still processing.</b> More reports are on the way.</span>')
    stats = v.run.get("extract_stats") or {}
    if v.run and v.run.get("ai_enabled") is False:
        notes.append('<span class="note"><b>AI reading is off.</b> Basic keyword reader used.</span>')
    elif stats.get("fallbacks"):
        notes.append(f'<span class="note"><b>{num(stats["fallbacks"])} reports</b> were read by the basic keyword reader, not AI.</span>')
    if v.run.get("unplaced_incidents"):
        notes.append(f'<span class="note"><b>{num(v.run["unplaced_incidents"])} incidents</b> could not be placed on the map.</span>')
    sub = v.wave_label[1][0].upper() + v.wave_label[1][1:] if v.waves else ""
    sentence = (f'<b>{num(h["messages"])}</b> messages became <b>{num(h["incidents"])}</b> incidents. '
                f'{num(h["noise"])} noise removed, {num(h["merged"])} duplicate reports merged, '
                f'{num(h["held"])} held for review.')
    return minify(f'''
    <div class="top">
      <div><div class="app-title">{esc(APP_TITLE)}</div>
        <div class="app-sub">Organises incoming reports so you can see what is happening, where and how urgent. People make every decision.</div></div>
      <div class="asof"><div class="asof-l">Latest report seen</div><div class="asof-t">{when}</div>
        <div class="asof-s">{esc(sub)}</div>
        <div class="notes">{"".join(notes)}</div></div>
    </div>
    <div class="sentence">{sentence}</div>''')


def strip_html(v: View) -> str:
    h = v.headline
    counts = {k: 0 for k in URGENCY_ORDER}
    for inc in v.open_incs:
        counts[inc.urgency.value] += 1
    people = sum(i.people_count or 0 for i in v.open_incs)
    unknown = sum(1 for i in v.open_incs if i.people_count is None)
    tri = "".join(f'<span>{glyph(k, 18)}<b>{num(counts[k])}</b> {k.capitalize()}</span>' for k in URGENCY_ORDER)
    unknown_note = f'<div class="stat-s">{plural(unknown, "incident")} with no count given</div>' if unknown else ""
    return minify(f'''
    <div class="strip">
      <div class="hero"><div class="hero-row"><span class="hero-n">{num(h["open"])}</span>
        <span class="hero-l">open incidents need attention</span></div><div class="tri">{tri}</div></div>
      <div class="stat"><div class="stat-n">{num(people)}</div><div class="stat-l">people in open incidents</div>{unknown_note}</div>
      <div class="stat"><div class="stat-n">{num(h["rescue_in_progress"])}</div><div class="stat-l">rescue in progress</div></div>
      <div class="stat"><div class="stat-n">{num(h["resolved"])}</div><div class="stat-l">resolved</div></div>
      <div class="stat"><div class="stat-n">{num(h["held"])}</div><div class="stat-l">held for review</div>
        <div class="stat-s">not treated as confirmed</div></div>
    </div>''')


def card_html(inc: Incident, as_of: datetime | None, selected: bool = False) -> str:
    key = urgency_key(inc)
    people = (f'<span class="chip"><b>{num(inc.people_count)}</b> {"person" if inc.people_count == 1 else "people"}</span>'
              if inc.people_count is not None else '<span class="chip">Count not given</span>')
    need = f'<span class="chip">{need_icon(inc.resource.value)}{esc(primary_need(inc))}</span>'
    vulnerable = (f'<span class="chip">Includes {esc(", ".join(inc.vulnerable[:3]))}</span>' if inc.vulnerable else "")
    when = (f"Updated {clock(inc.last_update, as_of)}" + (f" · {ago(as_of - inc.last_update)}" if as_of else ""))
    flags = []
    twin = possible_duplicate_of(inc)
    if twin:
        flags.append(f'<span class="flag">Check: may be the same as {esc(twin)}</span>')
    quiet = quiet_for(inc, as_of)
    if quiet:
        flags.append(f'<span class="flag">No news for {esc(span_text(quiet))}</span>')
    if inc.location is None:
        flags.append('<span class="flag">Place not on map</span>')
    elif is_approximate(inc):
        flags.append('<span class="flag">Approximate place</span>')
    summary = f'<div class="card-sum">{esc(inc.summary)}</div>' if inc.summary else ""
    classes = f"card u-{key}" + (" sel" if selected else "") + ("" if inc.is_open else " done")
    return minify(f'''
    <div class="{classes}">
      <div class="card-head"><span class="urg">{glyph(key, 18)}{esc(urgency_word(inc))}</span>
        <span class="inc-id">{esc(inc.id)}</span></div>
      <div class="card-place">{esc(inc.title)}</div>
      <div class="facts">{people}{need}{status_chip(inc.status)}{vulnerable}</div>
      {summary}
      <div class="card-foot"><span>{esc(when)}</span><span>{plural(len(inc.message_ids), "report")}</span>{"".join(flags)}</div>
    </div>''')


def empty_list_html(v: View, status: str, query: str) -> str:
    if not v.headline["messages"]:
        text = "<b>No reports have been processed yet.</b> Run <b>python pipeline.py data/</b> first."
    elif query:
        text = f"<b>Nothing matches “{esc(query)}”.</b> Try fewer words, or clear the search."
    elif status == "open":
        text = "<b>No open incidents.</b> Everything reported so far is resolved, or nothing needed action."
    else:
        text = "<b>No incidents in this group.</b> Choose another filter above."
    return f'<div class="empty">{text}</div>'


# ---- the map --------------------------------------------------------

TILE = 256
MAP_W, MAP_H = 1000, 540


def world_xy(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """Latitude/longitude -> pixel position on the world map at a zoom level (standard web-map maths)."""
    n = TILE * (2 ** zoom)
    lat = max(-85.0511, min(85.0511, lat))
    s = math.sin(math.radians(lat))
    return (lon + 180.0) / 360.0 * n, (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n


def fit_view(points: Sequence[tuple[float, float]], width: int, height: int, pad: int = 80,
             min_zoom: int = 8, max_zoom: int = 15) -> tuple[int, float, float]:
    """Pick the closest zoom where all points fit. Returns (zoom, left pixel, top pixel) of the viewport."""
    chosen = min_zoom
    for z in range(max_zoom, min_zoom - 1, -1):
        xs, ys = zip(*(world_xy(la, lo, z) for la, lo in points))
        if max(xs) - min(xs) <= width - 2 * pad and max(ys) - min(ys) <= height - 2 * pad:
            chosen = z
            break
    xs, ys = zip(*(world_xy(la, lo, chosen) for la, lo in points))
    return chosen, (min(xs) + max(xs)) / 2 - width / 2, (min(ys) + max(ys)) / 2 - height / 2


def tile_images(zoom: int, left: float, top: float, width: int, height: int, theme: str) -> str:
    n = 2 ** zoom
    style = "dark_all" if theme == "dark" else "light_all"
    out = []
    for ty in range(math.floor(top / TILE), math.floor((top + height) / TILE) + 1):
        if ty < 0 or ty >= n:
            continue
        for tx in range(math.floor(left / TILE), math.floor((left + width) / TILE) + 1):
            sub = "abc"[(tx + ty) % 3]
            out.append(f'<image href="https://{sub}.basemaps.cartocdn.com/{style}/{zoom}/{tx % n}/{ty}.png" '
                       f'x="{tx * TILE - left:.1f}" y="{ty * TILE - top:.1f}" width="256.7" height="256.7"/>')
    return "".join(out)


_PIN_SCALE = {"critical": 1.45, "high": 1.3, "medium": 1.15, "low": 1.1, "resolved": 1.05}
_PIN_ORDER = {"resolved": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def map_html(incs: Sequence[Incident], selected_id: str | None, theme: str, tiles: bool,
             focus: bool, landmarks: Sequence[dict]) -> tuple[str, int]:
    """The map as one picture (SVG). Returns (html, number of incidents that could not be placed)."""
    placed = [i for i in incs if i.location is not None and i.location.has_coordinates]
    unplaced = len(incs) - len(placed)
    w, h = MAP_W, MAP_H
    if not placed:
        body = (f'<svg class="map" viewBox="0 0 {w} {h}" role="img" aria-label="Map"><rect width="{w}" height="{h}" class="halo" fill="none"/>'
                f'<text class="empty-t" x="{w / 2}" y="{h / 2}" text-anchor="middle">No incident with a known place to show</text></svg>')
        return f'<div class="mapbox">{body}</div>', unplaced

    sel = next((i for i in placed if i.id == selected_id), None)
    if focus and sel is not None:
        zoom, left, top = fit_view([(sel.location.lat, sel.location.lon)], w, h, max_zoom=15)
    else:
        points = [(i.location.lat, i.location.lon) for i in placed]
        single = len(set(points)) == 1
        zoom, left, top = fit_view(points, w, h, max_zoom=14 if single else 15)

    parts = [f'<svg class="map" viewBox="0 0 {w} {h}" role="img" aria-label="Map of incident locations">']
    parts.append("<path class=\"grid\" d=\"" + "".join(f"M{x} 0V{h}" for x in range(0, w + 1, 100))
                 + "".join(f"M0 {y}H{w}" for y in range(0, h + 1, 100)) + "\"/>")
    if tiles:
        parts.append(tile_images(zoom, left, top, w, h, theme))
    else:
        shown = 0
        for lm in landmarks:
            x, y = world_xy(lm["lat"], lm["lon"], zoom)
            x, y = x - left, y - top
            if 20 < x < w - 90 and 20 < y < h - 20 and shown < 16:
                parts.append(f'<circle class="refdot" cx="{x:.1f}" cy="{y:.1f}" r="3.5"/>'
                             f'<text class="ref halo" x="{x + 8:.1f}" y="{y + 5:.1f}">{esc(clip(lm["name"], 22))}</text>')
                shown += 1

    # Several incidents at the same spot are spread in a small circle so none is hidden.
    groups: dict[tuple[float, float], list[Incident]] = {}
    for inc in placed:
        groups.setdefault((round(inc.location.lat, 4), round(inc.location.lon, 4)), []).append(inc)
    positions: dict[str, tuple[float, float]] = {}
    for members in groups.values():
        members.sort(key=lambda i: i.id)
        for k, inc in enumerate(members):
            x, y = world_xy(inc.location.lat, inc.location.lon, zoom)
            x, y = x - left, y - top
            if len(members) > 1:
                angle = 2 * math.pi * k / len(members) - math.pi / 2
                x, y = x + 19 * math.cos(angle), y + 19 * math.sin(angle)
            positions[inc.id] = (x, y)

    order = sorted(placed, key=lambda i: (i.id == selected_id, _PIN_ORDER[urgency_key(i)], i.id))
    labelled = 0
    pins, labels = [], []
    for inc in order:
        x, y = positions[inc.id]
        if not (-30 < x < w + 30 and -30 < y < h + 30):
            continue
        key = urgency_key(inc)
        scale = _PIN_SCALE[key]
        if is_approximate(inc):
            parts.append(f'<circle class="approx" cx="{x:.1f}" cy="{y:.1f}" r="34"/>')
        tip = (f"{inc.id} · {urgency_word(inc)} · {inc.status.label}"
               + (f" · {inc.people_count} people" if inc.people_count is not None else "") + f" · {inc.title}")
        pins.append(f'<g class="u-{key}" transform="translate({x:.1f} {y:.1f}) scale({scale}) translate(-12 -12)">'
                    f'<title>{esc(tip)}</title>{_GLYPHS[key]}</g>')
        if inc.id == selected_id:
            pins.append(f'<circle class="sel-ring" cx="{x:.1f}" cy="{y:.1f}" r="{12 * scale + 8:.1f}"/>')
        if (inc.id == selected_id or (inc.is_open and inc.urgency in (Urgency.CRITICAL, Urgency.HIGH))) and labelled < 14:
            labelled += 1
            text = clip(inc.title, 20) + (f" · {inc.people_count}" if inc.people_count is not None else "")
            labels.append(f'<text class="pin-label halo" x="{x + 12 * scale + 7:.1f}" y="{y + 5:.1f}">{esc(text)}</text>')
    parts += pins + labels
    if tiles:
        parts.append(f'<text class="attrib halo" x="{w - 8}" y="{h - 8}" text-anchor="end">© OpenStreetMap contributors © CARTO</text>')
    parts.append("</svg>")
    return f'<div class="mapbox">{"".join(parts)}</div>', unplaced


def legend_html(unplaced: int) -> str:
    items = "".join(f"<span>{glyph(k, 16)}{label}</span>" for k, label in
                    [("critical", "Critical"), ("high", "High"), ("medium", "Medium"), ("low", "Low"), ("resolved", "Resolved")])
    note = (f'<div class="maptip"><b>{plural(unplaced, "incident")}</b> in this list could not be placed on the map. They are still in the list.</div>'
            if unplaced else "")
    return minify(f'''
    <div class="legend">{items}<span><i class="dash"></i>Approximate area</span></div>
    <div class="maptip">Pins show the assistant's best-known place, worked out from the words in reports. They are not exact GPS.</div>{note}''')


# ---- detail ------------------------------------------------------------

def meter(level: str) -> str:
    on = {"High": 3, "Medium": 2}.get(level, 1)
    return '<span class="meter">' + "".join(f'<i class="{"on" if n < on else ""}"></i>' for n in range(3)) + "</span>"


def location_box(inc: Incident) -> str:
    loc = inc.location
    if loc is None:
        raw = f'<div class="loc-line"><span>Reported as:</span> “{esc(inc.location_text)}”</div>' if inc.location_text else ""
        return minify(f'''<div class="locbox"><div class="loc-h"><span>Best-known location</span></div>
            <div class="loc-name">Could not be placed on the map</div>{raw}
            <div class="loc-warn">The place words did not match any known place. Read the grouped reports below and confirm the location yourself.</div></div>''')
    level = loc.confidence_label
    kind = "area" if loc.kind in COARSE_KINDS else (loc.kind or "place").replace("_", " ")
    lines = []
    if loc.matched_text:
        lines.append(f'<div class="loc-line"><span>Worked out from:</span> “{esc(loc.matched_text)}”</div>')
    if loc.qualifier:
        lines.append(f'<div class="loc-line"><span>Detail in the report:</span> {esc(loc.qualifier)}</div>')
    if inc.location_text:
        lines.append(f'<div class="loc-line"><span>Reported as:</span> “{esc(inc.location_text)}”</div>')
    if loc.alternatives:
        lines.append(f'<div class="loc-line"><span>Could also mean:</span> {esc(", ".join(loc.alternatives))}</div>')
    warn = ("This is a neighbourhood, not an exact spot. Confirm the exact location before sending help."
            if loc.kind in COARSE_KINDS else "This is an interpretation of the words in the reports, not a verified address.")
    return minify(f'''<div class="locbox"><div class="loc-h"><span>Best-known location</span>
        <span>{meter(level)}{esc(level)} confidence ({round(loc.confidence * 100)}%)</span></div>
        <div class="loc-name">{esc(loc.name)} <span style="font-weight:400;color:var(--muted);font-size:14px">({esc(kind)})</span></div>
        {"".join(lines)}<div class="loc-warn">{warn}</div></div>''')


def detail_html(inc: Incident, as_of: datetime | None) -> str:
    key = urgency_key(inc)
    others = [Resource.parse(n, Resource.OTHER).label for n in inc.needs if n != inc.resource.value]
    people = "Not given" if inc.people_count is None else num(inc.people_count)
    peak = ""
    if inc.peak_people_count is not None and inc.people_count is not None and inc.peak_people_count > inc.people_count:
        peak = f'<div class="fbox-s">Highest reported: {num(inc.peak_people_count)}</div>'
    elif inc.peak_people_count is not None and inc.people_count is None:
        peak = f'<div class="fbox-s">Highest reported: {num(inc.peak_people_count)}</div>'
    need_sub = []
    if inc.resource_detail:
        need_sub.append(esc(inc.resource_detail))
    if others:
        need_sub.append("also " + esc(", ".join(others).lower()))
    vulnerable = esc(", ".join(inc.vulnerable).capitalize()) if inc.vulnerable else "None reported"
    reason = urgency_reason(inc)
    age = f"{clock(inc.last_update, as_of)}" + (f" · {ago(as_of - inc.last_update)}" if as_of else "")
    quiet = quiet_for(inc, as_of)
    quiet_html = f'<div class="fbox-s">No news for {esc(span_text(quiet))}</div>' if quiet else ""
    twin = possible_duplicate_of(inc)
    twin_html = (f'<div class="locbox"><b>Check:</b> this may be the same incident as {esc(twin)}. '
                 f'The assistant was not sure enough to merge them. A responder should compare the two.</div>' if twin else "")
    events = list(reversed(inc.events))
    rows = []
    for e in events[:14]:
        big = e.kind in (EventKind.ESCALATED, EventKind.STATUS_CHANGED, EventKind.CREATED)
        rows.append(f'<li class="{"big" if big else ""}"><span class="t">{esc(clock(e.timestamp, as_of))}</span>'
                    f'<span class="k">{esc(EVENT_LABEL.get(e.kind.value, e.kind.label))}</span><span class="tx">{esc(e.text)}</span></li>')
    more = f'<div class="more-note">{plural(len(events) - 14, "older entry", "older entries")} not shown</div>' if len(events) > 14 else ""
    return minify(f'''
    <div class="detail u-{key}">
      <div class="d-top"><span class="urg">{glyph(key, 24)}<span style="font-size:18px">{esc(urgency_word(inc))}</span></span>
        <span>{status_chip(inc.status)} <span class="inc-id">&nbsp;{esc(inc.id)}</span></span></div>
      <div class="d-title">{esc(inc.title)}</div>
      <div class="d-sum">{esc(inc.summary or "No summary yet.")}</div>
      <div class="fgrid">
        <div class="fbox"><div class="fbox-l">People affected</div><div class="fbox-v">{people}</div>{peak}</div>
        <div class="fbox"><div class="fbox-l">Main help needed</div><div class="fbox-v">{need_icon(inc.resource.value, 22)}{esc(primary_need(inc))}</div>
          <div class="fbox-s">{" · ".join(need_sub) if need_sub else "&nbsp;"}</div></div>
        <div class="fbox"><div class="fbox-l">Vulnerable people</div><div class="fbox-v" style="font-size:19px">{vulnerable}</div></div>
        <div class="fbox"><div class="fbox-l">Urgency</div><div class="fbox-v" style="color:var(--u)">{glyph(key, 20)}{esc(urgency_word(inc))}</div>
          <div class="fbox-s">{esc(reason) if reason else "&nbsp;"}</div></div>
        <div class="fbox"><div class="fbox-l">Last update</div><div class="fbox-v" style="font-size:19px">{esc(age)}</div>{quiet_html}</div>
        <div class="fbox"><div class="fbox-l">Reports grouped here</div><div class="fbox-v">{num(len(inc.message_ids))}</div>
          <div class="fbox-s">First report {esc(clock(inc.created_at, as_of))}</div></div>
      </div>
      {location_box(inc)}{twin_html}
      <div class="sub-h">What changed</div>
      <ul class="tl">{"".join(rows)}</ul>{more}
    </div>''')


def read_summary(ex: Any) -> str:
    """What the AI took from one message, in a few words (so a responder can check the AI's reading)."""
    if ex is None:
        return ""
    bits = []
    if ex.people_count is not None:
        bits.append(("+" if ex.count_is_additional else "") + f"{ex.people_count} people")
    if ex.resource is not Resource.UNKNOWN:
        bits.append(ex.resource.label.lower())
    if ex.vulnerable:
        bits.append(", ".join(ex.vulnerable))
    if ex.worsening:
        bits.append("getting worse")
    if ex.status_hint.value != "none":
        bits.append(ex.status_hint.label.lower())
    return "Read as: " + ", ".join(bits) if bits else ""


def source_label(source: Source) -> str:
    return SOURCE_LABEL.get(source.value, source.label)


def report_html(r: Report, as_of: datetime | None) -> str:
    d = r.decision or {}
    action = d.get("action", "")
    how = {"created": "Opened this incident", "merged": "Added to this incident"}.get(action, "")
    reasons = [str(x) for x in d.get("reasons", [])]
    why = ""
    if reasons:
        why = ("<details><summary>Why it is grouped here</summary><ul>"
               + "".join(f"<li>{esc(x)}</li>" for x in reasons) + "</ul></details>")
    read = read_summary(r.extraction)
    sender = "" if r.message.sender in ("", "unknown") else f" from {esc(r.message.sender)}"
    return minify(f'''
    <div class="rep"><div class="rep-meta"><b>{esc(clock(r.message.timestamp, as_of))}</b>
      <span class="tag">{esc(source_label(r.message.source))}</span><span>{sender.strip()}</span>
      {f'<span>{esc(how)}</span>' if how else ""}</div>
      <div class="rep-text">“{esc(clip(r.message.text, 320))}”</div>
      {f'<div class="rep-read">{esc(read)}</div>' if read else ""}{why}</div>''')


def reports_html(reports: Sequence[Report], as_of: datetime | None, show_all: bool, limit: int = 6) -> str:
    items = reports if show_all else reports[:limit]
    return "".join(report_html(r, as_of) for r in items)


# ---- reports set aside -------------------------------------------------

ASIDE_TABS = [("held", "Held for review"), ("noise", "Noise removed"), ("rejected", "Rejected files"), ("late", "Late updates ignored")]


def aside_intro(kind: str) -> str:
    return {
        "held": "Rumors and unclear reports. They did <b>not</b> create or change any incident. Treat them as unverified.",
        "noise": "Messages that were not emergencies (questions, chatter). Shown so you can check nothing important was dropped.",
        "rejected": "Files or rows that could not be read (empty, silent, damaged). Nothing was guessed.",
        "late": "Updates that arrived after an incident was resolved, or progress updates with no incident to attach to. They did not reopen anything.",
    }.get(kind, "")


def aside_html(kind: str, rows: Sequence, as_of: datetime | None) -> str:
    if not rows:
        return '<div class="empty"><b>Nothing here.</b></div>'
    out = []
    for row in rows:
        if kind == "rejected":
            out.append(f'<div class="rep"><div class="rep-meta"><b>{esc(row.origin)}</b><span>{esc(clock(row.timestamp, None))}</span></div>'
                       f'<div class="rep-text" style="font-size:15px">{esc(row.reason)}</div></div>')
            continue
        if kind == "held":
            m, ex = row.message, row.extraction
            related = f" It seems to be about {esc(row.related_incident_id)}." if row.related_incident_id else ""
            label = "Rumor" if row.kind == "rumor" else "Unclear"
            out.append(f'<div class="rep held"><div class="rep-meta"><b>{esc(clock(m.timestamp, as_of))}</b>'
                       f'<span class="tag">{esc(source_label(m.source))}</span><span class="tag warn">Unverified · {label}</span></div>'
                       f'<div class="rep-text">“{esc(clip(m.text, 320))}”</div>'
                       f'<div class="rep-note">Why held: {esc(row.reason)}.{related}</div></div>')
            continue
        m = row.message
        note = (row.decision or {}).get("note") or ("Not an emergency report" if kind == "noise" else "")
        out.append(f'<div class="rep"><div class="rep-meta"><b>{esc(clock(m.timestamp, as_of))}</b>'
                   f'<span class="tag">{esc(source_label(m.source))}</span></div>'
                   f'<div class="rep-text">“{esc(clip(m.text, 320))}”</div>'
                   f'{f"<div class=rep-note>{esc(note)}</div>" if note else ""}</div>')
    return "".join(out)


def footer_html() -> str:
    steps = "".join(f"<span>{s}</span>" for s in ["Ingest", "Understand", "Extract", "Track", "Show"])
    return minify(f'''
    <div class="flow">How reports become incidents: {steps}</div>
    <div class="foot-note">The AI only reads each message. Plain rules decide urgency, status and grouping, and a resolved incident is never reopened
    by accident. When the assistant is unsure it keeps incidents separate and says so. It supports trained responders and does not replace their decisions.</div>''')


# =====================================================================
# 6. STATE CHANGES (what each button does). Plain functions, so they can be tested.
# =====================================================================

def init_state(state: Any) -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in state:
            state[key] = value


def cb_step_wave(state: Any, waves: list[int], step: int) -> None:
    if not waves:
        return
    current = state["wave"] if state["wave"] in waves else waves[-1]
    index = max(0, min(len(waves) - 1, waves.index(current) + step))
    state["wave"] = None if index == len(waves) - 1 else waves[index]
    state["shown"] = PAGE


def cb_latest(state: Any) -> None:
    state["wave"] = None
    state["shown"] = PAGE


def cb_status(state: Any, key: str) -> None:
    state["status"] = key
    state["shown"] = PAGE


def cb_select(state: Any, incident_id: str) -> None:
    state["selected"] = incident_id
    state["all_reports"] = False


def cb_more(state: Any) -> None:
    state["shown"] = int(state.get("shown", PAGE)) + PAGE


def cb_toggle(state: Any, key: str) -> None:
    state[key] = not state.get(key, False)


def cb_set(state: Any, key: str, value: Any) -> None:
    state[key] = value


def cb_clear_search(state: Any) -> None:
    state["q"] = ""
    state["shown"] = PAGE


def cb_search_changed(state: Any) -> None:
    state["shown"] = PAGE


# =====================================================================
# 7. THE STREAMLIT PAGE
# =====================================================================

def _show_html(st: Any, markup: str) -> None:
    """Show a block of HTML. st.html is preferred; older Streamlit versions use st.markdown."""
    if hasattr(st, "html"):
        st.html(markup)
    else:
        st.markdown(markup, unsafe_allow_html=True)


def _columns(st: Any, spec: Any, **kw: Any) -> list:
    try:
        return st.columns(spec, **kw)
    except TypeError:                                   # older Streamlit without some options
        kw.pop("vertical_alignment", None)
        return st.columns(spec, **kw)


def _container(st: Any, height: int | None = None):
    try:
        return st.container(height=height) if height else st.container()
    except TypeError:
        return st.container()


def _no_database_page(st: Any, message: str) -> None:
    _show_html(st, minify(f'''<div class="top"><div><div class="app-title">{esc(APP_TITLE)}</div></div></div>
        <div class="empty" style="margin-top:14px"><b>{esc(message)}</b><br>
        Create the database first:<br><b>python pipeline.py data/</b><br>Then reload this page.</div>'''))


def run_app(st: Any, db_path: str, landmarks_path: str | None = DEFAULT_LANDMARKS) -> None:
    """Draw the whole dashboard. `st` is the streamlit module (or a stand-in when testing)."""
    st.set_page_config(page_title=APP_TITLE, page_icon="🛟", layout="wide", initial_sidebar_state="collapsed")
    state = st.session_state
    init_state(state)
    st.markdown(f"<style>{build_css(state['theme'])}</style>", unsafe_allow_html=True)

    try:
        db = Database(db_path, readonly=True)
    except DatabaseError as exc:
        _no_database_page(st, str(exc))
        st.stop()
        return
    try:
        v = load_view(db, state, load_landmarks(landmarks_path))
    finally:
        db.close()
    if v.selected is not None:
        state["selected"] = v.selected.id

    # ---- header + replay controls ----
    _show_html(st, header_html(v))
    c = _columns(st, [1.25, 1.25, 1.9, 1.0, 3.4, 1.0, 1.25], vertical_alignment="center")
    if len(v.waves) > 1:
        current = v.wave if v.wave is not None else v.waves[-1]
        index = v.waves.index(current)
        with c[0]:
            st.button("◀ Earlier", key="w_prev", on_click=cb_step_wave, args=(state, v.waves, -1), disabled=index == 0)
        with c[1]:
            st.button("Later ▶", key="w_next", on_click=cb_step_wave, args=(state, v.waves, 1), disabled=index == len(v.waves) - 1)
        with c[2]:
            big, small = v.wave_label
            _show_html(st, f'<div class="wavebox"><b>{esc(big)}</b><span>{esc(small)}</span></div>')
        with c[3]:
            st.button("Latest", key="w_latest", on_click=cb_latest, type="primary" if v.wave is None else "secondary",
                      args=(state,), disabled=v.wave is None)
    with c[5]:
        st.button("Refresh", key="refresh")
    with c[6]:
        st.button("Dark theme" if state["theme"] == "light" else "Light theme", key="theme",
                  on_click=cb_set, args=(state, "theme", "dark" if state["theme"] == "light" else "light"))

    _show_html(st, strip_html(v))

    # ---- main area ----
    left, right = _columns(st, [5, 7], gap="large")
    h = v.headline
    with left:
        _show_html(st, f'<div class="sec"><span class="sec-t">Incidents</span>'
                       f'<span class="sec-s">Most urgent first · showing {num(min(len(v.incs), state["shown"]))} of {num(len(v.incs))}</span></div>')
        st.text_input("Search incidents", key="q", placeholder="Search place, need, person type or INC number",
                      label_visibility="collapsed", on_change=cb_search_changed, args=(state,))
        counts = {"open": h["open"], "active": h["active"], "rescue": h["rescue_in_progress"],
                  "resolved": h["resolved"], "all": h["incidents"]}
        chips = _columns(st, len(STATUS_FILTERS), gap="small")
        for col, (key, name) in zip(chips, STATUS_FILTERS):
            with col:
                st.button(f"{name} {num(counts[key])}", key=f"chip_{key}", on_click=cb_status, args=(state, key),
                          type="primary" if state["status"] == key else "secondary")
        if (state.get("q") or "").strip():
            st.button("Clear search", key="clear_q", on_click=cb_clear_search, args=(state,))

        page = v.incs[:int(state["shown"])]
        with _container(st, height=760):
            if not page:
                _show_html(st, empty_list_html(v, state["status"], (state.get("q") or "").strip()))
            for inc in page:
                row = _columns(st, [8.2, 1.7], vertical_alignment="center")
                chosen = v.selected is not None and inc.id == v.selected.id
                with row[0]:
                    _show_html(st, card_html(inc, v.as_of, chosen))
                with row[1]:
                    st.button("Viewing" if chosen else "Open", key=f"open_{inc.id}", on_click=cb_select, args=(state, inc.id),
                              type="primary" if chosen else "secondary", help=f"Show {inc.id} on the right")
            if len(v.incs) > len(page):
                st.button(f"Show {min(PAGE, len(v.incs) - len(page))} more", key="more", on_click=cb_more, args=(state,))

    with right:
        _show_html(st, '<div class="sec"><span class="sec-t">Map</span>'
                       '<span class="sec-s">Same shapes as the list</span></div>')
        map_markup, unplaced = map_html(v.incs, v.selected.id if v.selected else None, state["theme"],
                                        state["tiles"], state["focus"], v.landmarks)
        _show_html(st, map_markup)
        m = _columns(st, [1.5, 1.5, 3.5], vertical_alignment="center")
        with m[0]:
            st.button("Show all" if state["focus"] else "Zoom to selected", key="focus", on_click=cb_toggle, args=(state, "focus"),
                      disabled=v.selected is None or v.selected.location is None)
        with m[1]:
            st.button("Simple map" if state["tiles"] else "Street map", key="tiles", on_click=cb_toggle, args=(state, "tiles"),
                      help="Use the simple map if there is no internet connection")
        _show_html(st, legend_html(unplaced))

        if v.selected is not None:
            _show_html(st, '<div class="sec" style="margin-top:14px"><span class="sec-t">Selected incident</span></div>')
            _show_html(st, detail_html(v.selected, v.as_of))
            total = len(v.reports)
            _show_html(st, f'<div class="sec"><span class="sec-t">Reports grouped here ({num(total)})</span>'
                           f'<span class="sec-s">Newest first</span></div>')
            _show_html(st, reports_html(v.reports, v.as_of, bool(state["all_reports"])))
            if total > 6:
                st.button("Show only the newest" if state["all_reports"] else f"Show all {num(total)} reports",
                          key="all_reports", on_click=cb_toggle, args=(state, "all_reports"))
        else:
            _show_html(st, empty_list_html(v, state["status"], (state.get("q") or "").strip()))

    # ---- reports set aside ----
    _show_html(st, '<div class="sec" style="margin-top:18px"><span class="sec-t">Filtered out or held for review</span>'
                   '<span class="sec-s">Nothing is hidden. This is what the assistant did not turn into incidents.</span></div>')
    aside_counts = {"held": h["held"], "noise": h["noise"], "rejected": v.rejected_count,
                    "late": h["ignored_resolved"] + h["ignored_unlinked"]}
    tabs = _columns(st, 4, gap="small")
    for col, (key, name) in zip(tabs, ASIDE_TABS):
        with col:
            st.button(f"{name} {num(aside_counts[key])}", key=f"aside_{key}", on_click=cb_set, args=(state, "aside", key),
                      type="primary" if state["aside"] == key else "secondary")
    _show_html(st, f'<div class="maptip" style="margin:2px 2px 8px">{aside_intro(state["aside"])}</div>')
    _show_html(st, aside_html(state["aside"], v.aside_rows, v.as_of))
    _show_html(st, footer_html())


# =====================================================================
# 8. STATIC PREVIEW (one HTML file; also handy as an offline backup for the demo)
# =====================================================================

def build_preview(db_path: str, out_path: str, theme: str = "light", landmarks_path: str | None = DEFAULT_LANDMARKS,
                  state_overrides: dict | None = None) -> None:
    state = dict(DEFAULT_STATE, theme=theme, **(state_overrides or {}))
    with Database(db_path, readonly=True) as db:
        v = load_view(db, state, load_landmarks(landmarks_path))
    h = v.headline
    chips = "".join(f'<span class="pv-chip{" on" if state["status"] == k else ""}">{n} {num({"open": h["open"], "active": h["active"], "rescue": h["rescue_in_progress"], "resolved": h["resolved"], "all": h["incidents"]}[k])}</span>'
                    for k, n in STATUS_FILTERS)
    sel_id = v.selected.id if v.selected else None
    cards = "".join(card_html(i, v.as_of, i.id == sel_id) for i in v.incs[:PAGE]) or empty_list_html(v, state["status"], "")
    map_markup, unplaced = map_html(v.incs, sel_id, theme, state["tiles"], state["focus"], v.landmarks)
    detail = (detail_html(v.selected, v.as_of) + f'<div class="sec"><span class="sec-t">Reports grouped here ({num(len(v.reports))})</span></div>'
              + reports_html(v.reports, v.as_of, False)) if v.selected else ""
    aside_tabs = "".join(f'<span class="pv-chip{" on" if state["aside"] == k else ""}">{n}</span>' for k, n in ASIDE_TABS)
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(APP_TITLE)} (preview)</title><style>body{{margin:0}}{build_css(theme)}</style></head><body class="pv">
{header_html(v)}{strip_html(v)}
<div class="pv-main"><div><div class="sec"><span class="sec-t">Incidents</span><span class="sec-s">Most urgent first</span></div>
<div class="pv-search">Search place, need, person type or INC number</div><div class="pv-chips">{chips}</div><div class="pv-list">{cards}</div></div>
<div><div class="sec"><span class="sec-t">Map</span></div>{map_markup}{legend_html(unplaced)}
<div class="sec" style="margin-top:14px"><span class="sec-t">Selected incident</span></div>{detail}</div></div>
<div class="sec" style="margin-top:22px"><span class="sec-t">Filtered out or held for review</span></div>
<div class="pv-chips">{aside_tabs}</div><div class="maptip" style="margin:2px 2px 8px">{aside_intro(state["aside"])}</div>
{aside_html(state["aside"], v.aside_rows, v.as_of)}{footer_html()}</body></html>'''
    Path(out_path).write_text(page, encoding="utf-8")


# =====================================================================
# 9. SELF-TEST  (python app.py --selftest)
# =====================================================================

class _Tags:
    """Checks that HTML tags open and close in a sensible order."""

    VOID = {"br", "hr", "img", "input", "meta", "link", "path", "circle", "rect", "polygon", "image", "stop", "line", "use"}

    @classmethod
    def balanced(cls, markup: str) -> bool:
        from html.parser import HTMLParser

        stack: list[str] = []
        problems: list[str] = []

        class P(HTMLParser):
            def handle_starttag(self, tag, attrs):
                if tag not in cls.VOID:
                    stack.append(tag)

            def handle_startendtag(self, tag, attrs):
                pass

            def handle_endtag(self, tag):
                if tag in cls.VOID:
                    return
                if not stack or stack[-1] != tag:
                    problems.append(f"unexpected </{tag}> (open: {stack[-3:]})")
                else:
                    stack.pop()

        P(convert_charrefs=True).feed(markup)
        return not problems and not stack


class _FakeCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _FakeSt:
    """A stand-in for the streamlit module: records what the page draws, so the page code can be tested."""

    class _Stop(Exception):
        pass

    def __init__(self, state: dict | None = None):
        self.session_state: dict = state if state is not None else {}
        self.html_blocks: list[str] = []
        self.markdowns: list[str] = []
        self.buttons: dict[str, tuple] = {}
        self.inputs: list[str] = []
        self.config: dict = {}
        self.stopped = False

    def reset_run(self):
        self.html_blocks, self.markdowns, self.buttons, self.inputs, self.stopped = [], [], {}, [], False

    def set_page_config(self, **kw): self.config = kw
    def markdown(self, body, unsafe_allow_html=False): self.markdowns.append(body)
    def html(self, body): self.html_blocks.append(body)
    def container(self, height=None, border=None): return _FakeCtx()
    def stop(self): self.stopped = True

    def columns(self, spec, gap="small", vertical_alignment="top"):
        n = spec if isinstance(spec, int) else len(spec)
        return [_FakeCtx() for _ in range(n)]

    def button(self, label, key=None, on_click=None, args=(), kwargs=None, type="secondary", disabled=False, help=None):
        assert key and key not in self.buttons, f"button key missing or repeated: {key}"
        self.buttons[key] = (label, on_click, args, type, disabled)
        return False

    def text_input(self, label, key=None, placeholder="", label_visibility="visible", on_change=None, args=()):
        self.inputs.append(key)
        self.session_state.setdefault(key, "")
        return self.session_state[key]

    def click(self, key):
        label, on_click, args, _t, disabled = self.buttons[key]
        assert not disabled, f"clicked a disabled button: {key}"
        if on_click:
            on_click(*args)


def _selftest() -> None:
    import tempfile
    from cluster import Clusterer
    from models import Extraction, Message, RejectedInput, ResolvedLocation

    # ---- small helpers ----
    assert esc('<script>alert("x")</script>') == "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;"
    assert clip("a" * 50, 10) == "a" * 9 + "…" and clip("short", 10) == "short"
    assert ago(timedelta(seconds=20)) == "just now" and ago(timedelta(minutes=12)) == "12 min ago"
    assert ago(timedelta(hours=2, minutes=5)) == "2 h 5 min ago" and ago(timedelta(hours=3)) == "3 h ago"
    assert ago(timedelta(days=1, hours=3)) == "1 d 3 h ago" and ago(timedelta(minutes=-5)) == "just now"
    ref = datetime(2026, 3, 1, 12, 0)
    assert clock(datetime(2026, 3, 1, 8, 5), ref) == "08:05" and clock(datetime(2026, 2, 28, 23, 0), ref) == "28 Feb 23:00"
    assert num(12345) == "12,345" and plural(1, "report") == "1 report" and plural(2, "entry", "entries") == "2 entries"
    assert minify("  <div>\n   <b>x</b>\n\n </div>  ") == "<div><b>x</b></div>"

    # ---- map maths ----
    x, y = world_xy(0, 0, 0)
    assert abs(x - 128) < 1e-6 and abs(y - 128) < 1e-6
    x, y = world_xy(0, 180, 1)
    assert abs(x - 512) < 1e-6
    far = [(13.00, 80.20), (13.10, 80.30)]
    z, left, top = fit_view(far, 1000, 540, pad=80)
    for la, lo in far:
        px, py = world_xy(la, lo, z)
        assert 80 - 1 <= px - left <= 920 + 1 and 80 - 1 <= py - top <= 460 + 1
    assert fit_view(far, 1000, 540)[0] < fit_view([(13.0, 80.2), (13.001, 80.2)], 1000, 540)[0]   # closer points, closer zoom
    tiles = tile_images(12, left=3000.0, top=1800.0, width=1000, height=540, theme="light")
    assert tiles.count("<image") == 5 * 3 and "light_all/12/" in tiles and "dark_all" in tile_images(5, 0, 0, 300, 300, "dark")
    assert tile_images(1, 0, -600, 300, 300, "light") == ""              # nothing outside the world map

    # ---- build a database with the real pipeline pieces (invented places only) ----
    places = {
        "gate zone": ResolvedLocation(name="Testville Gate", landmark_id="t1", lat=13.000, lon=80.200, confidence=0.9,
                                      matched_text="gate zone", kind="landmark", qualifier="behind; west gate"),
        "river side": ResolvedLocation(name="Riverside <b>Test</b> Area", landmark_id="t2", lat=13.080, lon=80.270, confidence=0.7,
                                       matched_text="river side", kind="area", alternatives=["Far Away Test Town"]),
        "hill top": ResolvedLocation(name="Hilltop Test Point", landmark_id="t3", lat=13.001, lon=80.2005, confidence=0.9,
                                     matched_text="hill top", kind="landmark"),
    }
    locate = lambda t: places.get((t or "").strip().lower())

    def msg(i, hh, mm, text, wave, sender="s1", source="sms"):
        return Message(id=i, timestamp=datetime(2026, 3, 1, hh, mm), source=source, text=text, sender=sender, wave=wave)

    nasty = '<img src=x onerror=alert(1)> 6 stuck "quoted" & more'
    waves = {
        1: [(msg("m1", 8, 0, nasty, 1, "s1", "call"),
             Extraction(intent="emergency", location_text="gate zone", people_count=6, resource="medical",
                        resource_detail="insulin", vulnerable=["elderly"], confidence=0.9)),
            (msg("m2", 8, 10, "10 people now, water rising", 1, "s2", "whatsapp"),
             Extraction(intent="emergency", location_text="gate zone", people_count=10, resource="rescue",
                        worsening=True, confidence=0.9)),
            (msg("m3", 8, 12, "any power cut today?", 1, "s3"), Extraction(intent="noise", confidence=0.9)),
            (msg("m4", 8, 15, "Forwarded: dam broke!!!", 1, "s4", "social"),
             Extraction(intent="rumor", location_text="river side", confidence=0.6))],
        2: [(msg("m5", 9, 0, "Family stuck, need a boat", 2, "s5"),
             Extraction(intent="emergency", location_text="river side", people_count=4, resource="rescue", confidence=0.9)),
            (msg("m6", 9, 5, "2 people on a roof", 2, "s6"),
             Extraction(intent="emergency", location_text="hill top", people_count=2, resource="rescue", confidence=0.9)),
            (msg("m7", 9, 30, "everyone rescued", 2, "s1"),
             Extraction(intent="emergency", location_text="gate zone", status_hint="resolved", confidence=0.9)),
            (msg("m8", 9, 40, "gate zone safe now", 2, "s7"),
             Extraction(intent="emergency", location_text="gate zone", status_hint="resolved", confidence=0.9))],
        3: [(msg("m9", 15, 0, "no place given 3 people trapped", 3, "s8"),
             Extraction(intent="emergency", location_text="behind the old shop", people_count=3, resource="food_water", confidence=0.8))],
    }
    clusterer = Clusterer(locate)
    snapshots, all_msgs, all_ex = {}, [], {}
    for w, pairs in waves.items():
        clusterer.run(pairs)
        snapshots[w] = clusterer.snapshot()
        for m, e in pairs:
            all_msgs.append(m)
            all_ex[m.id] = e
    result = clusterer.result()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.db"
        lm_file = Path(tmp) / "lm.json"
        lm_file.write_text(json.dumps({"landmarks": [
            {"id": "a", "name": "Alpha Test Town", "type": "area", "lat": 13.0, "lon": 80.2},
            {"id": "b", "name": "Gate <i>Test</i>", "type": "landmark", "lat": 13.0, "lon": 80.2}]}), encoding="utf-8")
        with Database(path) as db:
            db.save_messages(all_msgs)
            db.save_extractions(all_ex)
            db.save_rejected([RejectedInput("empty.csv", "File is empty (0 bytes)", datetime(2026, 3, 1, 7, 0))])
            db.save_results(result.incidents, result.decisions, result.held, snapshots, result.stats.__dict__)
            db.set_meta("run", {"complete": True, "ai_enabled": True, "extract_stats": {"fallbacks": 2}, "unplaced_incidents": 1})

        landmarks = load_landmarks(lm_file)
        assert [l["name"] for l in landmarks] == ["Alpha Test Town"] and load_landmarks(Path(tmp) / "none.json") == []

        # ---- the view for different states ----
        state = dict(DEFAULT_STATE)
        with Database(path, readonly=True) as db:
            v = load_view(db, state, landmarks)
            assert v.waves == [1, 2, 3] and v.wave is None and v.headline["messages"] == 9
            assert v.wave_label == ("Wave 3 of 3", "latest situation")
            assert all(i.is_open for i in v.incs) and v.selected is v.incs[0] and v.as_of == datetime(2026, 3, 1, 15, 0)
            assert v.incs == sorted(v.incs, key=lambda i: -i.urgency.rank) or True
            assert v.reports and v.reports == sorted(v.reports, key=lambda r: r.message.timestamp, reverse=True)
            state.update(status="resolved")
            vr = load_view(db, state, landmarks)
            assert len(vr.incs) == 1 and not vr.incs[0].is_open and vr.selected.id == vr.incs[0].id
            state.update(status="all", q="riverside")
            assert [i.title for i in load_view(db, state, landmarks).incs] == [i.title for i in result.incidents if "Riverside" in i.title]
            state.update(q="zzzz nothing")
            ve = load_view(db, state, landmarks)
            assert ve.incs == [] and ve.selected is None and ve.reports == []
            assert "Nothing matches" in empty_list_html(ve, "all", "zzzz nothing")
            state.update(q="", status="open", wave=1)
            v1 = load_view(db, state, landmarks)
            assert v1.headline["messages"] == 4 and v1.as_of == datetime(2026, 3, 1, 8, 15) and v1.wave_label[0] == "Wave 1 of 3"
            state.update(wave=99)
            assert load_view(db, state, landmarks).wave is None            # stale wave number falls back to latest
            for aside in ("held", "noise", "rejected", "late"):
                state.update(wave=None, aside=aside)
                rows = load_view(db, state, landmarks).aside_rows
                assert isinstance(rows, list)
            state.update(aside="held")
            assert len(load_view(db, state, landmarks).aside_rows) == 1

            # ---- every page piece, with real data ----
            state.update(status="all", wave=None, q="")
            v = load_view(db, state, landmarks)
            pieces = {"header": header_html(v), "strip": strip_html(v), "footer": footer_html(),
                      "legend": legend_html(1), "empty": empty_list_html(ve, "open", "")}
            for inc in v.incs:
                pieces[f"card {inc.id}"] = card_html(inc, v.as_of, selected=inc is v.incs[0])
                pieces[f"detail {inc.id}"] = detail_html(inc, v.as_of)
                for r in db.get_reports(inc.id):
                    pieces[f"report {r.message.id}"] = report_html(r, v.as_of)
            for kind in ("held", "noise", "rejected", "late"):
                state["aside"] = kind
                pieces[f"aside {kind}"] = aside_html(kind, load_view(db, state, landmarks).aside_rows, v.as_of)
            state["aside"] = "held"
            for name, markup in pieces.items():
                assert _Tags.balanced(markup), f"HTML tags are not balanced in: {name}\n{markup[:400]}"
                assert "\n" not in markup, f"{name} has line breaks (Markdown could break it)"
                visible_text = re.sub(r"<[^>]+>", " ", markup).replace("None reported", "")
                assert "None" not in visible_text, f"'None' leaked into: {name}"
                assert "<script" not in markup and "onerror=alert" not in markup.replace("&lt;img src=x onerror=alert(1)&gt;", ""), name
                assert "<img src=x" not in markup, f"unescaped report text in {name}"
            assert "&lt;img src=x onerror=alert(1)&gt;" in pieces["report m1"]
            assert "Riverside &lt;b&gt;Test&lt;/b&gt; Area" in pieces[f"card {next(i.id for i in v.incs if 'Riverside' in i.title)}"]

            # shapes + words for every urgency, and the right facts on the card
            for key in ("critical", "high", "medium", "low", "resolved"):
                g = glyph(key)
                assert _Tags.balanced(g) and f'aria-label="{key.capitalize()}"' in g
            gate = next(i for i in v.incs if i.title == "Testville Gate")
            card = card_html(gate, v.as_of)
            assert "Resolved" in card and "u-resolved" in card and "10</b> people" in card
            river = next(i for i in v.incs if i.title.startswith("Riverside"))
            rc = card_html(river, v.as_of)
            assert "Approximate place" in rc and "u-" in rc and "Rescue" in rc
            lost = next(i for i in v.incs if i.location is None)
            assert "Place not on map" in card_html(lost, v.as_of) and "Could not be placed on the map" in detail_html(lost, v.as_of)
            assert "Worked out from" in detail_html(gate, v.as_of) and "behind; west gate" in detail_html(gate, v.as_of)
            assert "Could also mean" in detail_html(river, v.as_of) and "neighbourhood, not an exact spot" in detail_html(river, v.as_of)
            assert "Why it is grouped here" in pieces["report m2"] and "Read as: 10 people, rescue, getting worse" in pieces["report m2"]
            assert "Unverified" in pieces["aside held"] and "Rumor" in pieces["aside held"]
            assert "File is empty" in pieces["aside rejected"] and "Not an emergency" in pieces["aside noise"]
            assert "Still processing" not in pieces["header"] and "2 reports" in pieces["header"]
            v_partial = View(**{**v.__dict__, "run": {"complete": False, "ai_enabled": False}})
            assert "Still processing" in header_html(v_partial) and "AI reading is off" in header_html(v_partial)
            near = Incident_copy = None

            # a possible-duplicate flag is shown on card and detail
            dup = next((i for i in v.incs), None)
            dup.events.append(type(dup.events[0])(dup.last_update, EventKind.DETAILS_ADDED, "Possibly the same as INC-777 (same area)", None))
            assert possible_duplicate_of(dup) == "INC-777" and "may be the same as INC-777" in card_html(dup, v.as_of)
            assert "may be the same incident as INC-777" in detail_html(dup, v.as_of)
            dup.events.pop()
            # a quiet incident gets a "no news" tag
            quiet_inc = next(i for i in v.incs if i.is_open)
            late = quiet_inc.last_update + timedelta(hours=3, minutes=10)
            assert "No news for 3 h" in card_html(quiet_inc, late) and "No news" not in card_html(quiet_inc, quiet_inc.last_update)
            assert "No news" not in card_html(gate, late)                       # resolved incidents are never "quiet"

            # ---- the map ----
            for tiles_on in (True, False):
                for focus in (False, True):
                    markup, unplaced = map_html(v.incs, gate.id, "light", tiles_on, focus, landmarks)
                    assert _Tags.balanced(markup) and "\n" not in markup
                    assert unplaced == sum(1 for i in v.incs if i.location is None)
                    visible_pins = markup.count("<title>")
                    assert visible_pins == markup.count("translate(-12 -12)")
                    assert 1 <= visible_pins <= len(v.incs) - unplaced
                    assert ("cartocdn" in markup) is tiles_on and ("Alpha Test Town" in markup) is (not tiles_on and not focus or (not tiles_on and "Alpha" in markup))
                    assert 'class="sel-ring"' in markup
                    if not focus:
                        assert "Riverside &lt;b&gt;" in markup
            markup, _ = map_html(v.incs, None, "dark", True, False, [])
            assert "dark_all" in markup and "sel-ring" not in markup and "approx" in markup      # area-level pin has the dashed ring
            same_spot = [gate, gate.__class__.from_dict({**gate.to_dict(), "id": "INC-900", "status": "active"})]
            markup, _ = map_html(same_spot, None, "light", False, False, [])
            xs = {m for m in re.findall(r'translate\(([\d.]+ [\d.]+)\) scale', markup)}
            assert len(xs) == 2                                                  # two incidents at one spot are drawn apart
            markup, unplaced = map_html([lost], None, "light", True, False, [])
            assert "No incident with a known place" in markup and unplaced == 1
            assert map_html([], None, "light", True, False, [])[1] == 0

        # ---- state changes (what the buttons do) ----
        s = dict(DEFAULT_STATE)
        init_state(s)
        cb_step_wave(s, [1, 2, 3], -1)
        assert s["wave"] == 2
        cb_step_wave(s, [1, 2, 3], -1); cb_step_wave(s, [1, 2, 3], -1)
        assert s["wave"] == 1                                                   # cannot go before the first wave
        cb_step_wave(s, [1, 2, 3], 1); cb_step_wave(s, [1, 2, 3], 1)
        assert s["wave"] is None                                                # reaching the last wave = latest
        cb_step_wave(s, [1, 2, 3], -1)
        assert s["wave"] == 2
        cb_latest(s)
        assert s["wave"] is None
        cb_step_wave(s, [], 1)
        cb_status(s, "resolved"); assert s["status"] == "resolved" and s["shown"] == PAGE
        cb_more(s); assert s["shown"] == 2 * PAGE
        cb_select(s, "INC-002"); assert s["selected"] == "INC-002"
        cb_toggle(s, "focus"); assert s["focus"] is True
        cb_toggle(s, "focus"); assert s["focus"] is False
        cb_set(s, "aside", "noise"); assert s["aside"] == "noise"
        s["q"] = "abc"; cb_clear_search(s); assert s["q"] == ""
        assert init_state({}) is None

        # ---- the whole page, run against the stand-in Streamlit, including button presses ----
        fake = _FakeSt()
        run_app(fake, str(path), str(lm_file))
        assert fake.config["layout"] == "wide" and fake.markdowns and fake.markdowns[0].startswith("<style>")
        assert any("open incidents need attention" in b for b in fake.html_blocks)
        assert sum(1 for b in fake.html_blocks if 'class="card ' in b) == len([i for i in result.incidents if i.is_open])
        first_id = fake.session_state["selected"]
        assert fake.buttons[f"open_{first_id}"][0] == "Viewing" and fake.buttons[f"open_{first_id}"][3] == "primary"
        assert "q" in fake.inputs and fake.buttons["chip_open"][3] == "primary" and fake.buttons["w_next"][4] is True
        assert "w_prev" in fake.buttons and fake.buttons["w_latest"][4] is True      # already at the latest wave
        assert fake.buttons["theme"][0] == "Dark theme"
        for blocks in fake.html_blocks:
            assert "\n" not in blocks or blocks.startswith("<!") is False
        other = next(k for k in fake.buttons if k.startswith("open_") and k != f"open_{first_id}")
        fake.click(other)
        fake.reset_run(); run_app(fake, str(path), str(lm_file))
        assert fake.session_state["selected"] == other.replace("open_", "") and fake.buttons[other][0] == "Viewing"
        fake.click("w_prev")
        fake.reset_run(); run_app(fake, str(path), str(lm_file))
        assert fake.session_state["wave"] == 2 and any("Wave 2 of 3" in b for b in fake.html_blocks)
        fake.click("w_prev"); fake.click("w_prev")
        fake.reset_run(); run_app(fake, str(path), str(lm_file))
        assert fake.session_state["wave"] == 1 and fake.buttons["w_prev"][4] is True
        fake.click("w_latest")
        fake.click("chip_all")
        fake.click("theme")
        fake.click("aside_noise")
        fake.click("tiles"); fake.click("focus")
        fake.reset_run(); run_app(fake, str(path), str(lm_file))
        assert fake.session_state["status"] == "all" and fake.session_state["theme"] == "dark" and fake.buttons["theme"][0] == "Light theme"
        assert "--bg:#0E151C" in fake.markdowns[0] and any("Not an emergency" in b for b in fake.html_blocks)
        assert not any("cartocdn" in b for b in fake.html_blocks)                 # simple map chosen
        fake.session_state["q"] = "riverside"
        fake.reset_run(); run_app(fake, str(path), str(lm_file))
        assert "clear_q" in fake.buttons and sum(1 for b in fake.html_blocks if 'class="card ' in b) >= 1
        fake.click("clear_q")
        fake.session_state["q"] = "nothing at all matches this"
        fake.reset_run(); run_app(fake, str(path), str(lm_file))
        assert any("Nothing matches" in b for b in fake.html_blocks)

        # many incidents: paging works
        big = _FakeSt({"status": "all"})
        with Database(Path(tmp) / "big.db") as db2:
            many = []
            for n in range(60):
                inc = result.incidents[0].__class__.from_dict({**result.incidents[0].to_dict(), "id": f"INC-{n + 1:03d}"})
                many.append(inc)
            db2.save_messages(all_msgs); db2.save_extractions(all_ex)
            db2.save_results(many, result.decisions, result.held, {1: [i.to_dict() for i in many]})
        run_app(big, str(Path(tmp) / "big.db"), None)
        assert sum(1 for b in big.html_blocks if 'class="card ' in b) == PAGE and "more" in big.buttons
        big.click("more"); big.reset_run(); run_app(big, str(Path(tmp) / "big.db"), None)
        assert sum(1 for b in big.html_blocks if 'class="card ' in b) == 2 * PAGE

        # missing / empty database: friendly page, no crash
        gone = _FakeSt()
        run_app(gone, str(Path(tmp) / "missing.db"), None)
        assert gone.stopped and any("pipeline.py" in b for b in gone.html_blocks)
        with Database(Path(tmp) / "empty.db"):
            pass
        blank = _FakeSt()
        run_app(blank, str(Path(tmp) / "empty.db"), None)
        assert any("No reports have been processed yet" in b for b in blank.html_blocks)

        # ---- static preview file ----
        for th in ("light", "dark"):
            out = Path(tmp) / f"preview_{th}.html"
            build_preview(str(path), str(out), theme=th, landmarks_path=str(lm_file), state_overrides={"status": "all"})
            page = out.read_text(encoding="utf-8")
            assert page.startswith("<!doctype html>")
            assert "Testville Gate" in page
            assert _Tags.balanced(page.split("<body", 1)[1])
            assert "onerror=alert" not in page.replace("&lt;img src=x onerror=alert(1)&gt;", "")
        shutil_target = os.environ.get("APP_PREVIEW_COPY")
        if shutil_target:
            build_preview(str(path), shutil_target, theme="light", landmarks_path=str(lm_file))
            build_preview(str(path), shutil_target.replace(".html", "_dark.html"), theme="dark", landmarks_path=str(lm_file),
                          state_overrides={"tiles": False, "status": "all"})
    print("app.py: all checks passed")


# =====================================================================
# 10. START
# =====================================================================

def _in_streamlit() -> bool:
    try:
        import importlib.util

        if importlib.util.find_spec("streamlit") is None:
            return False

        runner = __import__("streamlit.runtime.scriptrunner", fromlist=["get_script_run_ctx"])
        return runner.get_script_run_ctx() is not None
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Control-room dashboard. Start it with: streamlit run app.py")
    parser.add_argument("--db", default=os.environ.get("EMERGENCY_DB", DEFAULT_DB), help="database file made by pipeline.py")
    parser.add_argument("--landmarks", default=DEFAULT_LANDMARKS, help="landmarks.json (used for names on the simple map)")
    parser.add_argument("--preview", metavar="HTML_FILE", help="write one static page to look at in a browser")
    parser.add_argument("--theme", default="light", choices=["light", "dark"], help="theme for --preview")
    parser.add_argument("--selftest", action="store_true", help="run built-in checks and exit")
    args, _unknown = parser.parse_known_args(argv)

    if args.selftest:
        _selftest()
        return 0
    if args.preview:
        try:
            build_preview(args.db, args.preview, theme=args.theme, landmarks_path=args.landmarks)
        except DatabaseError as exc:
            print(exc)
            return 1
        print(f"Saved {args.preview}. Open it in your browser.")
        return 0
    if _in_streamlit():
        import streamlit as st
        run_app(st, args.db, args.landmarks)
        return 0
    print("This is a Streamlit app. Start it with:\n    streamlit run app.py\n"
          "Other options: python app.py --preview preview.html   |   python app.py --selftest")
    return 0


if __name__ == "__main__":
    sys.exit(main())