import dash
from dash import dcc, html, Input, Output, State, callback, ctx, no_update
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timezone, timedelta
import math

# =============================================================================
# PROJECT IMPORTS
# =============================================================================
try:
    try:
        from src.deployer import OrbitDeployer
    except ModuleNotFoundError:
        from deployer import OrbitDeployer
    deployer = OrbitDeployer()
    MODEL_LOADED = True
    print("[+] PINN Model loaded successfully (v3.3).")
except Exception as e:
    print(f"[!] Warning: Model not loaded — {e}")
    MODEL_LOADED = False

    class MockDeployer:
        """Generates visually distinct mock orbits per object using TLE hash as seed."""
        def predict(self, line1, line2, target_str):
            seed = abs(hash(line1)) % 100000
            rng = np.random.RandomState(seed)
            a = 6700 + rng.uniform(0, 600)
            inc = np.radians(rng.uniform(20, 85))
            raan = rng.uniform(0, 2 * np.pi)
            anom = rng.uniform(0, 2 * np.pi)
            r_orbit = a * np.array([np.cos(anom), np.sin(anom) * np.cos(inc), np.sin(anom) * np.sin(inc)])
            c, s = np.cos(raan), np.sin(raan)
            r0 = np.array([c * r_orbit[0] - s * r_orbit[1], s * r_orbit[0] + c * r_orbit[1], r_orbit[2]])
            v_mag = np.sqrt(398600.4 / a)
            v0 = np.cross([0, 0, 1], r0 / (np.linalg.norm(r0) + 1e-9)) * v_mag * 0.001
            nr, nv = rng.randn(3) * 30, rng.randn(3) * 0.005
            return (r0, v0), (r0 + rng.randn(3) * 50, v0 + rng.randn(3) * 0.01), (r0 + nr, v0 + nv)

        def get_trajectory(self, l1, l2, target, steps=200, window_minutes=50):
            seed = abs(hash(l1)) % 100000
            rng = np.random.RandomState(seed)
            a = 6700 + rng.uniform(0, 600)
            inc = np.radians(rng.uniform(20, 85))
            raan = rng.uniform(0, 2 * np.pi)
            t = np.linspace(0, 2 * np.pi, steps)
            x = a * np.cos(t)
            y = a * np.sin(t) * np.cos(inc)
            z = a * np.sin(t) * np.sin(inc)
            c, s = np.cos(raan), np.sin(raan)
            xr = c * x - s * y
            yr = s * x + c * y
            return np.column_stack([xr, yr, z])

    deployer = MockDeployer()

# TLE Fetcher
try:
    try:
        from src.tle_fetcher import fetch_tle, fetch_batch, is_tle_fresh
    except ModuleNotFoundError:
        from tle_fetcher import fetch_tle, fetch_batch, is_tle_fresh
    TLE_FETCHER_AVAILABLE = True
    print("[+] TLE Fetcher loaded (CelesTrak GP API).")
except Exception as e:
    TLE_FETCHER_AVAILABLE = False
    print(f"[!] TLE Fetcher not available — {e}")


# =============================================================================
# APP CONFIG
# =============================================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap"
    ],
    suppress_callback_exceptions=True
)
app.title = "EPOCH ZERO | Fleet Surveillance System"
server = app.server  # Expose Flask server for gunicorn/production deployment


# =============================================================================
# SATELLITE & DEBRIS DATABASE
# All NORAD IDs are REAL and trackable via CelesTrak GP API.
# Fallback TLEs (Jan 2022) are provided for offline mode only.
# For live ops, click "FETCH LIVE TLEs" to get today's data.
# =============================================================================
SAT_DATABASE = {
    # ---- PAYLOADS ----
    "25544": {
        "name": "ISS (ZARYA)", "short": "ISS", "norad": "25544",
        "type": "PAYLOAD", "country": "ISS", "launch": "1998-11-20",
        "tle1": "1 25544U 98067A   22011.50000000  .00006730  00000-0  12500-3 0  9990",
        "tle2": "2 25544  51.6435 200.1234 0006828 300.1234  59.8765 15.48919755370000",
    },
    "48274": {
        "name": "CSS (TIANHE)", "short": "CSS", "norad": "48274",
        "type": "PAYLOAD", "country": "PRC", "launch": "2021-04-29",
        "tle1": "1 48274U 21024A   22011.50000000  .00012000  00000-0  80000-4 0  9990",
        "tle2": "2 48274  53.0540 170.2345 0001234  85.6789 274.4321 15.06400000 40000",
    },
    "46984": {
        "name": "SENTINEL-6A", "short": "SENT-6A", "norad": "46984",
        "type": "PAYLOAD", "country": "EU", "launch": "2020-11-21",
        "tle1": "1 46984U 20084A   22011.50000000  .00000100  00000-0  20000-4 0  9990",
        "tle2": "2 46984  66.0400 120.0000 0008100 230.0000 130.0000 14.27000000 50000",
    },
    "43013": {
        "name": "NOAA 20 (JPSS-1)", "short": "NOAA-20", "norad": "43013",
        "type": "PAYLOAD", "country": "US", "launch": "2017-11-18",
        "tle1": "1 43013U 17073A   22011.50000000  .00000020  00000-0  18000-4 0  9990",
        "tle2": "2 43013  98.7100  30.5000 0001500  90.0000 270.0000 14.19500000 21000",
    },
    "41866": {
        "name": "GOES 16", "short": "GOES-16", "norad": "41866",
        "type": "PAYLOAD", "country": "US", "launch": "2016-11-19",
        "tle1": "1 41866U 16071A   22011.50000000  .00000010  00000-0  10000-4 0  9990",
        "tle2": "2 41866   0.0500 270.0000 0002000 260.0000 100.0000  1.00270000 30000",
    },
    "56700": {
        "name": "STARLINK-30053", "short": "STL-300", "norad": "56700",
        "type": "PAYLOAD", "country": "US", "launch": "2023-05-19",
        "tle1": "1 56700U 23071A   22011.50000000  .00010000  00000-0  60000-4 0  9990",
        "tle2": "2 56700  43.0000 150.0000 0001500  90.0000 270.0000 15.05000000 10000",
    },
    # ---- DEBRIS / ROCKET BODIES (Real NORAD IDs) ----
    "49271": {
        "name": "FREGAT DEB", "short": "FRG-DEB", "norad": "49271",
        "type": "DEBRIS", "country": "CIS", "launch": "N/A",
        "tle1": "1 49271U 82092AKM 22011.50000000  .00025000  00000-0  30000-3 0  9990",
        "tle2": "2 49271  82.5600  45.0000 0050000 200.0000 160.0000 14.85000000 10000",
    },
    "22285": {
        "name": "SL-16 R/B", "short": "SL16-RB", "norad": "22285",
        "type": "DEBRIS", "country": "CIS", "launch": "N/A",
        "tle1": "1 22285U 93009B   22011.50000000  .00001000  00000-0  25000-4 0  9990",
        "tle2": "2 22285  71.0000 150.0000 0005000 100.0000 260.0000 14.15000000 95000",
    },
    "54600": {
        "name": "CZ-6A DEB", "short": "CZ6-DEB", "norad": "54600",
        "type": "DEBRIS", "country": "PRC", "launch": "N/A",
        "tle1": "1 54600U 22142A   22011.50000000  .00005000  00000-0  50000-4 0  9990",
        "tle2": "2 54600  98.5000  30.0000 0010000  80.0000 280.0000 14.30000000 20000",
    },
    "25730": {
        "name": "FENGYUN 1C", "short": "FY-1C", "norad": "25730",
        "type": "DEBRIS", "country": "PRC", "launch": "N/A",
        "tle1": "1 25730U 99025A   22011.50000000  .00003000  00000-0  40000-4 0  9990",
        "tle2": "2 25730  99.2500  45.0000 0015000 150.0000 210.0000 14.35000000 80000",
    },
    "24946": {
        "name": "IRIDIUM 33", "short": "IR-33", "norad": "24946",
        "type": "DEBRIS", "country": "US", "launch": "N/A",
        "tle1": "1 24946U 97051C   22011.50000000  .00002000  00000-0  30000-4 0  9990",
        "tle2": "2 24946  86.4000  80.0000 0015000 200.0000 160.0000 14.40000000 60000",
    },
    "27602": {
        "name": "ARIANE 2 DEB", "short": "ARI-DEB", "norad": "27602",
        "type": "DEBRIS", "country": "FR", "launch": "N/A",
        "tle1": "1 27602U 86019QK  22011.50000000  .00000500  00000-0  15000-4 0  9990",
        "tle2": "2 27602  10.5000 300.0000 7100000  50.0000 310.0000  4.72000000 70000",
    },
}

# Store for live TLE data (updated at runtime via FETCH button)
LIVE_TLES: dict = {}  # { norad_id: {"tle1": ..., "tle2": ..., "epoch_str": ...} }

# --- Color Palettes (cool = satellites, warm = debris) ---
SAT_PALETTE = ['#00d4ff', '#4488ff', '#00ffcc', '#a855f7', '#66aaff', '#00ff88']
DEB_PALETTE = ['#ff6b35', '#ff4488', '#ff4444', '#ffaa00', '#ff6699', '#ffcc00']

OBJECT_COLORS = {}
_si, _di = 0, 0
for _nid, _info in SAT_DATABASE.items():
    if _info["type"] == "PAYLOAD":
        OBJECT_COLORS[_nid] = SAT_PALETTE[_si % len(SAT_PALETTE)]
        _si += 1
    else:
        OBJECT_COLORS[_nid] = DEB_PALETTE[_di % len(DEB_PALETTE)]
        _di += 1

# --- Dropdown Options ---
FLEET_OPTIONS = []
for _nid, _info in SAT_DATABASE.items():
    tag = "SAT" if _info["type"] == "PAYLOAD" else "DEB"
    FLEET_OPTIONS.append({"label": f"[{tag}] {_info['name']} ({_nid})", "value": _nid})

DEFAULT_FLEET = ["25544", "48274", "46984", "49271", "22285", "54600"]
PAYLOAD_IDS = [k for k, v in SAT_DATABASE.items() if v["type"] == "PAYLOAD"]
DEBRIS_IDS  = [k for k, v in SAT_DATABASE.items() if v["type"] == "DEBRIS"]
ALL_IDS     = list(SAT_DATABASE.keys())


# =============================================================================
# PRECOMPUTED ASSETS  (Stars · Earth · Atmosphere · Grid)
# =============================================================================
np.random.seed(42)
NUM_STARS = 800;  STAR_R = 35000
s_theta = np.random.uniform(0, 2 * np.pi, NUM_STARS)
s_phi   = np.random.uniform(0, np.pi, NUM_STARS)
s_sizes = np.random.choice([1, 1.5, 2, 2.5, 3], NUM_STARS, p=[0.4, 0.25, 0.2, 0.1, 0.05])
stars_x = STAR_R * np.sin(s_phi) * np.cos(s_theta)
stars_y = STAR_R * np.sin(s_phi) * np.sin(s_theta)
stars_z = STAR_R * np.cos(s_phi)

STAR_TRACE = go.Scatter3d(
    x=stars_x, y=stars_y, z=stars_z,
    mode='markers', marker=dict(size=s_sizes, color='white', opacity=0.7),
    hoverinfo='skip', name='Stars', showlegend=False,
)

N_EARTH = 50
u_e = np.linspace(0, 2 * np.pi, N_EARTH)
v_e = np.linspace(0, np.pi, N_EARTH)
R_EARTH = 6371
EX = R_EARTH * np.outer(np.cos(u_e), np.sin(v_e))
EY = R_EARTH * np.outer(np.sin(u_e), np.sin(v_e))
EZ = R_EARTH * np.outer(np.ones(np.size(u_e)), np.cos(v_e))
earth_colors = [[0,'#0a1628'],[0.2,'#0d2137'],[0.4,'#0f3460'],[0.6,'#16537e'],[0.8,'#1a759f'],[1.0,'#34a0a4']]

EARTH_TRACE = go.Surface(
    x=EX, y=EY, z=EZ, colorscale=earth_colors, showscale=False,
    lighting=dict(ambient=0.3, diffuse=0.6, specular=0.2, roughness=0.8, fresnel=0.2),
    lightposition=dict(x=10000, y=5000, z=8000),
    hoverinfo='skip', opacity=0.95, name='Earth',
)

ATM_SCALE = 1.015
ATMO_TRACE = go.Surface(
    x=EX*ATM_SCALE, y=EY*ATM_SCALE, z=EZ*ATM_SCALE,
    colorscale=[[0,'rgba(0,170,255,0.05)'],[1,'rgba(0,100,255,0.12)']],
    showscale=False, opacity=0.15, hoverinfo='skip', name='Atmosphere',
    lighting=dict(ambient=0.8, diffuse=0.2, specular=0.0),
)

def make_grid_lines():
    traces = []
    for lat in range(-60, 90, 30):
        theta = np.linspace(0, 2*np.pi, 80)
        r = R_EARTH * np.cos(np.radians(lat)) * 1.002
        z = R_EARTH * np.sin(np.radians(lat)) * 1.002
        traces.append(go.Scatter3d(
            x=r*np.cos(theta), y=r*np.sin(theta), z=np.full(80, z),
            mode='lines', line=dict(color='rgba(0,200,255,0.15)', width=1),
            hoverinfo='skip', showlegend=False))
    for lon in range(0, 360, 30):
        phi = np.linspace(0, np.pi, 80)
        traces.append(go.Scatter3d(
            x=R_EARTH*1.002*np.sin(phi)*np.cos(np.radians(lon)),
            y=R_EARTH*1.002*np.sin(phi)*np.sin(np.radians(lon)),
            z=R_EARTH*1.002*np.cos(phi),
            mode='lines', line=dict(color='rgba(0,200,255,0.1)', width=1),
            hoverinfo='skip', showlegend=False))
    return traces

GRID_TRACES = make_grid_lines()


# =============================================================================
# COMPONENT BUILDERS
# =============================================================================

def make_icon(icon, size=18, color="#00d4ff"):
    return DashIconify(icon=icon, width=size, color=color, style={"marginRight": "8px"})

def make_sidebar_section(title, icon, children):
    return html.Div([
        html.Div([make_icon(icon, 16, "#00d4ff"), html.Span(title, className="section-title")], className="section-header"),
        html.Div(children, className="section-body"),
    ], className="sidebar-section")

def make_batch_stat(icon, label, value_id, color="#00d4ff"):
    return html.Div([
        DashIconify(icon=icon, width=14, color=color, style={"marginRight":"4px"}),
        html.Span(label+" ", className="batch-stat-label"),
        html.Span("---", id=value_id, className="batch-stat-value", style={"color": color}),
    ], className="batch-stat-item")


# =============================================================================
# HEADER
# =============================================================================
header = html.Div([
    html.Div([
        html.Div([
            DashIconify(icon="mdi:satellite-variant", width=28, color="#00d4ff"),
            html.Span("EPOCH", style={"color":"#00d4ff","fontWeight":"800","fontSize":"20px","fontFamily":"'Orbitron', sans-serif"}),
            html.Span("ZERO", style={"color":"#ff6b35","fontWeight":"800","fontSize":"20px","fontFamily":"'Orbitron', sans-serif","marginLeft":"4px"}),
        ], className="header-brand"),
        html.Div([
            html.Div([html.Div(className="status-dot status-dot-green"), html.Span("SYS NOMINAL", className="status-text")], className="status-badge"),
            html.Div([html.Div(className="status-dot status-dot-blue"),
                       html.Span("PINN ONLINE" if MODEL_LOADED else "PINN OFFLINE", className="status-text",
                                 style={"color":"#00d4ff"} if MODEL_LOADED else {"color":"#ff4444"})], className="status-badge"),
            html.Div([DashIconify(icon="mdi:radar", width=14, color="#a855f7"),
                       html.Span("FLEET MODE", className="status-text", style={"color":"#a855f7"})], className="status-badge"),
            html.Div([DashIconify(icon="mdi:clock-outline", width=14, color="#888"),
                       html.Span(id="header-clock", className="status-text")], className="status-badge"),
        ], className="header-status"),
        html.Div([
            html.Span("FLEET SURVEILLANCE SYSTEM", className="header-subtitle"),
            html.Span("v4.0 | LIVE TLE + PINN OPS", className="header-version"),
        ], className="header-right"),
    ], className="header-inner"),
], className="app-header")


# =============================================================================
# LEFT SIDEBAR — FLEET CONTROL
# =============================================================================
left_sidebar = html.Div([
    make_sidebar_section("FLEET SELECTION", "mdi:radar", [
        html.Label("SELECT OBJECTS TO MONITOR", className="input-label"),
        dcc.Dropdown(
            id="fleet-select", options=FLEET_OPTIONS, value=DEFAULT_FLEET,
            multi=True, className="dark-dropdown",
            placeholder="Select satellites & debris...",
            style={"marginBottom":"8px"},
        ),
        html.Div([
            dbc.Button("ALL",    id="btn-select-all",    className="quick-btn", n_clicks=0, size="sm"),
            dbc.Button("SATS",   id="btn-select-sats",   className="quick-btn quick-btn-sat", n_clicks=0, size="sm"),
            dbc.Button("DEBRIS", id="btn-select-debris",  className="quick-btn quick-btn-deb", n_clicks=0, size="sm"),
            dbc.Button("CLEAR",  id="btn-clear",          className="quick-btn quick-btn-clear", n_clicks=0, size="sm"),
        ], className="quick-select-group"),
    ]),

    make_sidebar_section("LIVE TLE DATA", "mdi:cloud-download-outline", [
        html.Div([
            dbc.Button([
                DashIconify(icon="mdi:satellite-uplink", width=16, color="#000"),
                html.Span(" FETCH LIVE TLEs", style={"marginLeft":"6px"}),
            ], id="btn-fetch-tles", className="btn-action btn-primary-action", n_clicks=0,
               style={"width":"100%","fontSize":"11px","padding":"6px 10px"}),
        ]),
        html.Div(id="tle-status", children=[
            html.Span("TLE Source: ", style={"color":"#555","fontSize":"10px"}),
            html.Span("OFFLINE (fallback 2022)", id="tle-source-label",
                      style={"color":"#ff6b35","fontSize":"10px","fontFamily":"'Share Tech Mono', monospace"}),
        ], style={"marginTop":"6px"}),
        html.Div(id="tle-age-info", style={"fontSize":"9px","color":"#445566","marginTop":"3px"}),
    ]),

    make_sidebar_section("PREDICTION WINDOW", "mdi:clock-fast", [
        html.Label("TARGET DATE", className="input-label"),
        dcc.DatePickerSingle(id="target-date",
                             date=(datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d"),
                             display_format="YYYY-MM-DD",
                             className="dark-datepicker", style={"width":"100%"}),
        html.Label("TARGET TIME (UTC)", className="input-label", style={"marginTop":"8px"}),
        dcc.Input(id="target-time", value="12:00:00", type="text", className="dark-input", placeholder="HH:MM:SS"),
        html.Label("PROPAGATION WINDOW", className="input-label", style={"marginTop":"8px"}),
        dcc.Slider(id="prop-window", min=30, max=180, step=10, value=100,
                   marks={30:"30m",60:"1h",120:"2h",180:"3h"}, className="dark-slider",
                   tooltip={"placement":"bottom","always_visible":False}),
    ]),

    make_sidebar_section("ANIMATION", "mdi:motion-play-outline", [
        html.Label("PLAYBACK SPEED", className="input-label"),
        dcc.Slider(id="anim-speed", min=50, max=500, step=50, value=200,
                   marks={50:"0.5x",100:"1x",200:"2x",500:"5x"}, className="dark-slider"),
        html.Div([
            dbc.Button([make_icon("mdi:play",16,"#000"),"RUN"],  id="btn-run",   className="btn-action btn-primary-action",   n_clicks=0),
            dbc.Button([make_icon("mdi:pause",16,"#fff"),""],    id="btn-pause", className="btn-action btn-secondary-action", n_clicks=0),
            dbc.Button([make_icon("mdi:skip-forward",16,"#fff"),""], id="btn-step",  className="btn-action btn-secondary-action", n_clicks=0),
        ], className="btn-group-controls"),
    ]),

    make_sidebar_section("VISUAL LAYERS", "mdi:layers-triple-outline", [
        dbc.Checklist(id="visual-toggles", options=[
            {"label":" Orbit Trajectories","value":"rails"},
            {"label":" All Collision Lines","value":"killzone"},
            {"label":" Atmosphere","value":"atmo"},
            {"label":" Star Field","value":"stars"},
            {"label":" Grid Lines","value":"grid"},
            {"label":" Velocity Vectors","value":"vectors"},
            {"label":" Debris Cloud Halos","value":"halos"},
        ], value=["rails","killzone","atmo","stars"], className="visual-checklist", switch=True),
    ]),

    make_sidebar_section("CAMERA CONTROL", "mdi:video-3d-variant", [
        dbc.Checklist(id="camera-auto-orbit", options=[
            {"label": " Auto-Orbit Camera", "value": "auto"},
        ], value=["auto"], className="visual-checklist", switch=True),
        html.Div([
            dbc.Button([make_icon("mdi:camera-flip-outline",14,"#000"), "RESET VIEW"],
                       id="btn-reset-cam", className="btn-action btn-primary-action", n_clicks=0,
                       style={"fontSize":"11px","padding":"4px 10px","marginTop":"6px","width":"100%"}),
        ]),
        html.Div([
            html.Span("Scroll to zoom · Drag to rotate", className="input-label",
                       style={"marginTop":"6px","color":"#334455","fontSize":"9px"}),
        ]),
    ]),

    html.Div([
        dbc.Button([
            DashIconify(icon="mdi:radar", width=20, color="#000"),
            html.Span(" BATCH CONJUNCTION SCAN", style={"marginLeft":"8px"}),
        ], id="btn-scan", className="btn-scan", n_clicks=0),
    ], className="scan-btn-wrap"),
], className="left-sidebar", id="left-sidebar")


# =============================================================================
# RIGHT SIDEBAR — FLEET TELEMETRY & COLLISION MATRIX
# =============================================================================
right_sidebar = html.Div([
    # -- Fleet Status --
    make_sidebar_section("FLEET STATUS", "mdi:shield-check-outline", [
        html.Div([
            make_batch_stat("mdi:satellite-variant","OBJECTS:","batch-stat-objects","#00d4ff"),
            make_batch_stat("mdi:vector-line","PAIRS:","batch-stat-pairs","#a855f7"),
            make_batch_stat("mdi:alert-circle","ALERTS:","batch-stat-critical","#ff0040"),
            make_batch_stat("mdi:ruler","MIN DIST:","batch-stat-mindist","#ffaa00"),
        ], className="batch-summary"),
        html.Div([
            html.Div("FLEET THREAT", className="threat-label"),
            html.Div("STANDBY", id="fleet-threat-level", className="threat-badge threat-standby"),
        ], className="threat-block"),
        dbc.Progress(value=0, id="fleet-threat-bar", className="threat-progress", style={"height":"4px"}),
    ]),

    # -- Collision Risk Matrix --
    make_sidebar_section("COLLISION RISK MATRIX", "mdi:grid", [
        html.Div("Select 2+ objects and scan to generate matrix.", id="matrix-placeholder", className="placeholder-text"),
        dcc.Graph(id="collision-matrix-graph", style={"height":"280px","display":"none"},
                  config={'displayModeBar': False}),
    ]),

    # -- Top Threats --
    make_sidebar_section("TOP THREATS", "mdi:alert-octagon-outline", [
        html.Div(id="top-threats-container", children=[
            html.Div("No data — run a batch scan.", className="placeholder-text"),
        ]),
    ]),

    # -- Fleet Telemetry --
    make_sidebar_section("FLEET TELEMETRY", "mdi:satellite-uplink", [
        html.Div(id="fleet-telem-container", children=[
            html.Div("No data — run a batch scan.", className="placeholder-text"),
        ]),
    ]),

    # -- Event Log --
    make_sidebar_section("EVENT LOG", "mdi:format-list-bulleted", [
        html.Div(id="event-log", className="event-log", children=[
            html.Div([
                html.Span("SYS", className="log-tag log-tag-info"),
                html.Span(" Fleet Surveillance initialized. Select objects and scan.", className="log-msg"),
            ], className="log-entry"),
        ]),
    ]),
], className="right-sidebar", id="right-sidebar")


# =============================================================================
# CENTER — 3D VIEWPORT
# =============================================================================
center_viewport = html.Div([
    html.Div([
        html.Div([
            html.Span("3D ORBITAL VIEWPORT", className="viewport-title"),
            html.Span(" | FLEET SURVEILLANCE MODE", className="viewport-subtitle"),
        ]),
        html.Div([
            html.Span("OBJECTS: ", style={"color":"#666","fontSize":"11px"}),
            html.Span("0", id="obj-counter", style={"color":"#a855f7","fontSize":"11px","fontFamily":"'Share Tech Mono', monospace","marginRight":"12px"}),
            html.Span("FRAME: ", style={"color":"#666","fontSize":"11px"}),
            html.Span("0/0", id="frame-counter", style={"color":"#00d4ff","fontSize":"11px","fontFamily":"'Share Tech Mono', monospace"}),
        ]),
    ], className="viewport-header"),
    dcc.Graph(id="globe-graph", style={"height":"100%","width":"100%"},
              config={'displayModeBar': False, 'scrollZoom': True}),
    dcc.Interval(id="anim-interval", interval=200, n_intervals=0, disabled=True),
    dcc.Store(id="sim-store", data=None),
    dcc.Store(id="anim-frame", data=0),
    dcc.Store(id="anim-playing", data=False),
    dcc.Store(id="selected-pair-store", data=None),
    dcc.Store(id="user-camera-store", data=None),
    dcc.Store(id="live-tle-store", data=None),
], className="center-viewport")


# =============================================================================
# BOTTOM BAR
# =============================================================================
bottom_bar = html.Div([
    html.Div([
        DashIconify(icon="mdi:information-outline", width=14, color="#555"),
        html.Span(" EPOCH ZERO v4.0 | ", style={"color":"#555"}),
        html.Span("PINN-Enhanced Fleet Conjunction Analysis + Live TLE", style={"color":"#666"}),
    ], className="bottom-left"),
    html.Div([
        html.Span("MODEL: ", style={"color":"#555"}),
        html.Span("ResidualPINN v3.3", style={"color":"#00d4ff" if MODEL_LOADED else "#ff4444"}),
        html.Span(" | ", style={"color":"#333"}),
        html.Span("MODE: ", style={"color":"#555"}),
        html.Span("BATCH FLEET", style={"color":"#a855f7"}),
        html.Span(" | ", style={"color":"#333"}),
        html.Span("ENGINE: ", style={"color":"#555"}),
        html.Span("SGP4 + J2/Drag", style={"color":"#888"}),
        html.Span(" | ", style={"color":"#333"}),
        html.Span("PROPAGATOR: ", style={"color":"#555"}),
        html.Span("ACTIVE", style={"color":"#00ff88"}),
    ], className="bottom-right"),
], className="bottom-bar")


# =============================================================================
# MAIN LAYOUT
# =============================================================================
app.layout = html.Div([
    header,
    html.Div([left_sidebar, center_viewport, right_sidebar], className="main-body"),
    bottom_bar,
    dcc.Interval(id="clock-interval", interval=1000, n_intervals=0),
], className="app-root")


# #############################################################################
#                              CALLBACKS
# #############################################################################

# ---- 1. Clock ----
@callback(Output("header-clock","children"), Input("clock-interval","n_intervals"))
def update_clock(_):
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ---- 2. Quick-select buttons ----
@callback(
    Output("fleet-select","value"),
    [Input("btn-select-all","n_clicks"), Input("btn-select-sats","n_clicks"),
     Input("btn-select-debris","n_clicks"), Input("btn-clear","n_clicks")],
    prevent_initial_call=True,
)
def quick_select(*_):
    t = ctx.triggered_id
    if t == "btn-select-all":    return ALL_IDS
    if t == "btn-select-sats":   return PAYLOAD_IDS
    if t == "btn-select-debris": return DEBRIS_IDS
    if t == "btn-clear":         return []
    return no_update


# ---- 2b. FETCH LIVE TLEs from CelesTrak ----
@callback(
    [Output("live-tle-store","data"),
     Output("tle-source-label","children"),
     Output("tle-source-label","style"),
     Output("tle-age-info","children"),
     Output("event-log","children",allow_duplicate=True)],
    Input("btn-fetch-tles","n_clicks"),
    State("fleet-select","value"),
    State("event-log","children"),
    prevent_initial_call=True,
)
def fetch_live_tles(n_clicks, selected_ids, current_log):
    if not n_clicks:
        return no_update, no_update, no_update, no_update, no_update

    log_entries = list(current_log) if current_log else []

    if not TLE_FETCHER_AVAILABLE:
        log_entries.insert(0, html.Div([
            html.Span("ERR", className="log-tag log-tag-err"),
            html.Span(" TLE Fetcher module not available", className="log-msg"),
        ], className="log-entry"))
        return (None,
                "OFFLINE (module error)",
                {"color":"#ff4444","fontSize":"10px","fontFamily":"'Share Tech Mono', monospace"},
                "", log_entries[:30])

    ids_to_fetch = selected_ids if selected_ids else ALL_IDS
    log_entries.insert(0, html.Div([
        html.Span("NET", className="log-tag log-tag-cmd"),
        html.Span(f" Fetching {len(ids_to_fetch)} TLEs from CelesTrak...", className="log-msg"),
    ], className="log-entry"))

    live_data = {}
    n_ok, n_fail = 0, 0
    epochs = []

    for nid in ids_to_fetch:
        result = fetch_tle(nid)
        if result:
            live_data[str(nid)] = {
                "tle1": result["tle1"],
                "tle2": result["tle2"],
                "epoch_str": result.get("epoch_str", ""),
                "name": result.get("name", ""),
            }
            # Also update SAT_DATABASE in-place for the scan
            if str(nid) in SAT_DATABASE:
                SAT_DATABASE[str(nid)]["tle1"] = result["tle1"]
                SAT_DATABASE[str(nid)]["tle2"] = result["tle2"]
                if result.get("name") and result["name"] != f"NORAD-{nid}":
                    SAT_DATABASE[str(nid)]["name"] = result["name"]
            epochs.append(result.get("epoch_str", ""))
            n_ok += 1
        else:
            n_fail += 1
            log_entries.insert(0, html.Div([
                html.Span("SKIP", className="log-tag log-tag-err"),
                html.Span(f" NORAD {nid}: not found on CelesTrak", className="log-msg"),
            ], className="log-entry"))

    if n_ok > 0:
        log_entries.insert(0, html.Div([
            html.Span("OK", className="log-tag log-tag-ok"),
            html.Span(f" Fetched {n_ok} live TLEs ({n_fail} failed)", className="log-msg"),
        ], className="log-entry"))
        latest_epoch = max(epochs) if epochs else ""
        src_label = f"LIVE ({n_ok}/{n_ok+n_fail})"
        src_style = {"color":"#00ff88","fontSize":"10px","fontFamily":"'Share Tech Mono', monospace"}
        age_info = f"Latest epoch: {latest_epoch}"
    else:
        src_label = "FETCH FAILED"
        src_style = {"color":"#ff4444","fontSize":"10px","fontFamily":"'Share Tech Mono', monospace"}
        age_info = "All fetches failed. Check internet connection."

    return live_data, src_label, src_style, age_info, log_entries[:30]


# ---- 3. BATCH CONJUNCTION SCAN ----
@callback(
    [Output("sim-store","data"),
     Output("collision-matrix-graph","figure"),
     Output("collision-matrix-graph","style"),
     Output("matrix-placeholder","style"),
     Output("top-threats-container","children"),
     Output("fleet-telem-container","children"),
     Output("batch-stat-objects","children"),
     Output("batch-stat-pairs","children"),
     Output("batch-stat-critical","children"),
     Output("batch-stat-mindist","children"),
     Output("fleet-threat-level","children"),
     Output("fleet-threat-level","className"),
     Output("fleet-threat-bar","value"),
     Output("fleet-threat-bar","color"),
     Output("anim-interval","disabled"),
     Output("anim-frame","data"),
     Output("anim-playing","data"),
     Output("event-log","children"),
     Output("obj-counter","children")],
    [Input("btn-scan","n_clicks"), Input("btn-run","n_clicks")],
    [State("fleet-select","value"), State("target-date","date"),
     State("target-time","value"), State("prop-window","value"),
     State("event-log","children"), State("live-tle-store","data")],
    prevent_initial_call=True,
)
def run_batch_scan(scan_clicks, run_clicks, selected_ids, date_val, time_val, prop_window, current_log, live_tles):
    if not scan_clicks and not run_clicks:
        return (no_update,) * 19

    log_entries = list(current_log) if current_log else []
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    def _err(msg):
        log_entries.insert(0, html.Div([
            html.Span("ERR", className="log-tag log-tag-err"),
            html.Span(f" {msg}", className="log-msg"),
        ], className="log-entry"))
        return (None, empty_fig, {"display":"none"}, {"display":"block"},
                [html.Div(msg, className="placeholder-text")],
                [html.Div(msg, className="placeholder-text")],
                "---","---","---","---",
                "STANDBY","threat-badge threat-standby",0,"info",
                True,0,False,log_entries[:30],"0")

    if not selected_ids or len(selected_ids) < 2:
        return _err("Select at least 2 objects for batch scan.")

    target_str = f"{date_val} {time_val}"
    n_selected = len(selected_ids)
    using_live = bool(live_tles and any(str(sid) in live_tles for sid in selected_ids))

    log_entries.insert(0, html.Div([
        html.Span("CMD", className="log-tag log-tag-cmd"),
        html.Span(f" Batch scan: {n_selected} objects, {n_selected*(n_selected-1)//2} pairs"
                  + (" [LIVE TLEs]" if using_live else " [offline TLEs]"), className="log-msg"),
    ], className="log-entry"))

    # ===== BATCH PROCESS ALL OBJECTS =====
    objects = {}
    n_tle_stale = 0
    for obj_id in selected_ids:
        sat = SAT_DATABASE.get(obj_id)
        if not sat:
            continue

        # Use live TLE if available, else fallback to database
        if live_tles and str(obj_id) in live_tles:
            tle1 = live_tles[str(obj_id)]["tle1"]
            tle2 = live_tles[str(obj_id)]["tle2"]
        else:
            tle1 = sat["tle1"]
            tle2 = sat["tle2"]
        try:
            result = deployer.predict(tle1, tle2, target_str)
            traj   = deployer.get_trajectory(tle1, tle2, target_str,
                                                steps=200, window_minutes=prop_window // 2)
            if result[0] is None:
                n_tle_stale += 1
                log_entries.insert(0, html.Div([
                    html.Span("SKIP", className="log-tag log-tag-err"),
                    html.Span(f" {sat['name']}: TLE too stale for target date (SGP4 diverged)", className="log-msg"),
                ], className="log-entry"))
                continue
            if traj.ndim != 2 or traj.shape[0] < 2:
                n_tle_stale += 1
                log_entries.insert(0, html.Div([
                    html.Span("SKIP", className="log-tag log-tag-err"),
                    html.Span(f" {sat['name']}: trajectory empty — TLE epoch too far from target", className="log-msg"),
                ], className="log-entry"))
                continue
            (r0, v0), (r_sgp4, v_sgp4), (r_pinn, v_pinn) = result
            objects[obj_id] = {
                "name":     sat["name"],
                "short":    sat.get("short", sat["name"][:10]),
                "type":     sat["type"],
                "color":    OBJECT_COLORS.get(obj_id, "#ffffff"),
                "trajectory": traj.tolist(),
                "pos_pinn": r_pinn.tolist(),
                "vel_pinn": v_pinn.tolist(),
                "pos_sgp4": list(r_sgp4),
                "vel_sgp4": list(v_sgp4),
                "dr":       float(np.linalg.norm(r_pinn - np.array(r_sgp4))),
                "altitude": float(np.linalg.norm(r_pinn) - R_EARTH),
                "speed":    float(np.linalg.norm(v_pinn)),
            }
        except Exception as ex:
            log_entries.insert(0, html.Div([
                html.Span("ERR", className="log-tag log-tag-err"),
                html.Span(f" {sat['name']}: {str(ex)[:50]}", className="log-msg"),
            ], className="log-entry"))

    if n_tle_stale > 0 and len(objects) >= 2:
        log_entries.insert(0, html.Div([
            html.Span("WARN", className="log-tag log-tag-err"),
            html.Span(f" {n_tle_stale} object(s) skipped — TLE epoch too far from target date. Use a date closer to Jan 2022.", className="log-msg"),
        ], className="log-entry"))

    if len(objects) < 2:
        msg = "Less than 2 objects propagated successfully."
        if n_tle_stale > 0:
            msg += f" {n_tle_stale} object(s) skipped — target date is too far from TLE epoch (Jan 2022). Use a closer date."
        return _err(msg)

    # ===== COLLISION MATRIX =====
    ids = list(objects.keys())
    n   = len(ids)
    matrix = np.full((n, n), np.nan)
    pairs  = []

    for i in range(n):
        for j in range(i+1, n):
            dist = float(np.linalg.norm(
                np.array(objects[ids[i]]["pos_pinn"]) - np.array(objects[ids[j]]["pos_pinn"])))
            matrix[i][j] = dist
            matrix[j][i] = dist
            threat = "LOW"
            if   dist < 100:  threat = "CRITICAL"
            elif dist < 500:  threat = "HIGH"
            elif dist < 2000: threat = "WARNING"
            pairs.append({"a": ids[i], "b": ids[j],
                          "name_a": objects[ids[i]]["short"],
                          "name_b": objects[ids[j]]["short"],
                          "miss_dist": dist, "threat": threat})

    pairs.sort(key=lambda p: p["miss_dist"])
    n_critical = sum(1 for p in pairs if p["threat"] in ("CRITICAL","HIGH"))
    n_warning  = sum(1 for p in pairs if p["threat"] == "WARNING")
    min_miss   = pairs[0]["miss_dist"] if pairs else 9999
    closest_str = f"{pairs[0]['name_a']} ↔ {pairs[0]['name_b']}" if pairs else "---"

    # ===== BUILD MATRIX HEATMAP =====
    short_names = [objects[oid]["short"] for oid in ids]
    hover = [[f"{short_names[i]} ↔ {short_names[j]}<br>{matrix[i][j]:.1f} km"
              if i != j else f"{short_names[i]}" for j in range(n)] for i in range(n)]

    fig_matrix = go.Figure(data=go.Heatmap(
        z=matrix.tolist(), x=short_names, y=short_names,
        text=hover, hovertemplate='%{text}<extra></extra>',
        colorscale=[[0,'#ff0040'],[0.08,'#ff6b35'],[0.2,'#ffaa00'],[0.5,'#00ff88'],[1.0,'#0d2137']],
        zmin=0, zmax=max(5000, float(np.nanmax(matrix)) if not np.all(np.isnan(matrix)) else 5000),
        showscale=False, xgap=2, ygap=2,
    ))
    fig_matrix.update_layout(
        template="plotly_dark", margin=dict(l=5,r=5,b=5,t=5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(size=8, color="#8899aa", family="Share Tech Mono"), tickangle=-45, side="bottom"),
        yaxis=dict(tickfont=dict(size=8, color="#8899aa", family="Share Tech Mono"), autorange="reversed"),
        height=max(200, 40*n+60),
    )
    matrix_h = max(200, 40*n+60)

    # ===== TOP THREATS LIST =====
    color_map = {"CRITICAL":"#ff0040","HIGH":"#ff4444","WARNING":"#ffaa00","LOW":"#00ff88"}
    cls_map   = {"CRITICAL":"threat-critical","HIGH":"threat-high","WARNING":"threat-warn","LOW":"threat-low"}
    top_n = min(7, len(pairs))
    threat_children = []
    for idx, p in enumerate(pairs[:top_n]):
        t = p["threat"]
        threat_children.append(html.Div([
            html.Span(f"#{idx+1}", className="threat-rank", style={"color":color_map.get(t,"#888")}),
            html.Div([
                html.Span(f"{p['name_a']} ", style={"color":"#00d4ff","fontSize":"11px"}),
                html.Span("↔ ", style={"color":"#555","fontSize":"11px"}),
                html.Span(p['name_b'], style={"color":"#ff6b35","fontSize":"11px"}),
            ], className="threat-pair-names"),
            html.Span(f"{p['miss_dist']:.1f} km", className="threat-dist", style={"color":color_map.get(t,"#888")}),
            html.Span(t, className=f"threat-tag {cls_map.get(t,'threat-low')}"),
        ], className="threat-item"))
    if not threat_children:
        threat_children = [html.Div("No pairs.", className="placeholder-text")]

    # ===== FLEET TELEMETRY CARDS =====
    telem_children = []
    for oid in ids:
        obj = objects[oid]
        c   = obj["color"]
        tl  = "SAT" if obj["type"] == "PAYLOAD" else "DEB"
        tcls = "type-badge-sat" if obj["type"] == "PAYLOAD" else "type-badge-deb"
        telem_children.append(html.Div([
            html.Div([
                html.Div(className="telem-color-dot", style={"background":c}),
                html.Span(obj["short"], className="telem-obj-name", style={"color":c}),
                html.Span(tl, className=f"type-badge {tcls}"),
            ], className="telem-card-header"),
            html.Div([
                html.Div([html.Span("ALT ", className="telem-mini-label"),
                           html.Span(f"{obj['altitude']:.0f} km", className="telem-mini-value")], className="telem-mini"),
                html.Div([html.Span("SPD ", className="telem-mini-label"),
                           html.Span(f"{obj['speed']:.3f} km/s", className="telem-mini-value")], className="telem-mini"),
                html.Div([html.Span("Δr ", className="telem-mini-label"),
                           html.Span(f"{obj['dr']:.2f} km", className="telem-mini-value")], className="telem-mini"),
            ], className="telem-card-body"),
        ], className="telem-compact-card"))

    # ===== FLEET THREAT LEVEL =====
    if n_critical > 0:
        ft_txt, ft_cls, ft_val, ft_col = "CRITICAL", "threat-badge threat-critical", 100, "danger"
    elif n_warning > 0:
        ft_txt, ft_cls, ft_val, ft_col = "WARNING",  "threat-badge threat-warn",     40,  "warning"
    else:
        ft_txt, ft_cls, ft_val, ft_col = "LOW",      "threat-badge threat-low",      10,  "success"

    # ===== SIM DATA STORE =====
    sim_data = {
        "objects": objects,
        "collision_matrix": {"ids": ids, "short_names": short_names, "distances": matrix.tolist()},
        "pairs": pairs[:20],
        "total_frames": 200,
        "target": target_str,
        "n_objects": n, "n_pairs": len(pairs),
        "n_critical": n_critical, "n_warning": n_warning,
        "min_miss": min_miss, "closest_pair": closest_str,
    }

    # ===== LOG =====
    log_entries.insert(0, html.Div([
        html.Span("OK", className="log-tag log-tag-ok"),
        html.Span(f" Batch complete: {n} objects, {len(pairs)} pairs analyzed", className="log-msg"),
    ], className="log-entry"))
    if n_critical > 0:
        log_entries.insert(0, html.Div([
            html.Span("WARN", className="log-tag log-tag-err"),
            html.Span(f" {n_critical} CRITICAL/HIGH conjunction(s) detected!", className="log-msg"),
        ], className="log-entry"))
    log_entries.insert(0, html.Div([
        html.Span("TCA", className="log-tag log-tag-cmd"),
        html.Span(f" Closest: {closest_str} @ {min_miss:.1f} km", className="log-msg"),
    ], className="log-entry"))

    return (
        sim_data,
        fig_matrix,
        {"height":f"{matrix_h}px","display":"block"},
        {"display":"none"},
        threat_children,
        telem_children,
        str(n),
        str(len(pairs)),
        str(n_critical),
        f"{min_miss:.1f} km",
        ft_txt, ft_cls, ft_val, ft_col,
        False,  # enable animation
        0, True,
        log_entries[:30],
        str(n),
    )


# ---- 4. Pause ----
@callback(
    [Output("anim-interval","disabled",allow_duplicate=True),
     Output("anim-playing","data",allow_duplicate=True)],
    Input("btn-pause","n_clicks"),
    State("anim-playing","data"),
    prevent_initial_call=True,
)
def toggle_pause(n, playing):
    if n:
        return playing, not playing
    return no_update, no_update


# ---- 5. Speed ----
@callback(Output("anim-interval","interval"), Input("anim-speed","value"))
def update_speed(speed):
    return max(50, 600 - speed)


# ---- 5b. Step forward one frame ----
@callback(
    Output("anim-frame","data",allow_duplicate=True),
    Input("btn-step","n_clicks"),
    [State("anim-frame","data"), State("sim-store","data")],
    prevent_initial_call=True,
)
def step_frame(n, current_frame, sim_data):
    if n and sim_data:
        total = max(sim_data.get("total_frames", 1), 1)
        frame = current_frame if current_frame else 0
        return (frame + 1) % total
    return no_update


# ---- 6a. Capture user camera from graph relayout (zoom/pan/rotate) ----
@callback(
    Output("user-camera-store","data"),
    [Input("globe-graph","relayoutData"),
     Input("btn-reset-cam","n_clicks")],
    State("user-camera-store","data"),
    prevent_initial_call=True,
)
def capture_user_camera(relayout, reset_clicks, current_cam):
    triggered = ctx.triggered_id
    # Reset button → clear stored camera so auto-orbit resumes
    if triggered == "btn-reset-cam":
        return None
    # Capture camera changes from user interaction (zoom, rotate, pan)
    if relayout:
        cam = {}
        for key in ("scene.camera.eye.x","scene.camera.eye.y","scene.camera.eye.z",
                    "scene.camera.center.x","scene.camera.center.y","scene.camera.center.z",
                    "scene.camera.up.x","scene.camera.up.y","scene.camera.up.z"):
            if key in relayout:
                cam[key] = relayout[key]
        if cam:
            # Merge with existing
            merged = current_cam or {}
            merged.update(cam)
            return merged
    return no_update


# ---- 6b. Matrix click → select pair ----
@callback(
    Output("selected-pair-store","data"),
    Input("collision-matrix-graph","clickData"),
    State("sim-store","data"),
    prevent_initial_call=True,
)
def select_pair_from_matrix(clickData, sim_data):
    if clickData and sim_data:
        pt = clickData["points"][0]
        xl, yl = pt.get("x"), pt.get("y")
        if xl and yl and xl != yl:
            short_to_id = {obj["short"]: oid for oid, obj in sim_data["objects"].items()}
            a, b = short_to_id.get(yl), short_to_id.get(xl)
            if a and b:
                return {"a": a, "b": b}
    return None


# ---- 7. RENDER FRAME  (Multi-object simultaneous visualization) ----
@callback(
    [Output("globe-graph","figure"),
     Output("anim-frame","data",allow_duplicate=True),
     Output("frame-counter","children")],
    [Input("anim-interval","n_intervals"),
     Input("sim-store","data"),
     Input("visual-toggles","value"),
     Input("selected-pair-store","data")],
    [State("anim-frame","data"),
     State("anim-playing","data"),
     State("user-camera-store","data"),
     State("camera-auto-orbit","value")],
    prevent_initial_call='initial_duplicate',
)
def render_frame(n_intervals, sim_data, toggles, selected_pair, current_frame, playing, user_cam, auto_orbit_toggle):
    fig = go.Figure()

    # Base layers
    if toggles and "stars" in toggles:
        fig.add_trace(STAR_TRACE)
    fig.add_trace(EARTH_TRACE)
    if toggles and "atmo" in toggles:
        fig.add_trace(ATMO_TRACE)
    if toggles and "grid" in toggles:
        for gt in GRID_TRACES:
            fig.add_trace(gt)

    frame_text = "0/0"
    frame = current_frame if current_frame else 0
    use_auto = False     # default: no auto-orbit camera
    camera_dict = None   # None → let uirevision preserve user camera

    if sim_data and sim_data.get("objects"):
        objects = sim_data["objects"]
        total   = max(sim_data.get("total_frames", 1), 1)

        if playing:
            frame = (frame + 1) % total
        frame = min(max(frame, 0), total - 1)
        frame_text = f"{frame+1}/{total}"

        obj_ids       = list(objects.keys())
        positions_now = {}

        # ---- Draw every object ----
        for oid in obj_ids:
            obj   = objects[oid]
            traj  = np.array(obj.get("trajectory", []))
            color = obj.get("color", "#ffffff")

            if traj.ndim != 2 or traj.shape[0] == 0 or traj.shape[1] < 3:
                continue   # skip objects with bad/empty trajectories

            f   = min(frame, len(traj) - 1)
            pos = traj[f]
            positions_now[oid] = pos

            # Orbit trail (past + future)
            if toggles and "rails" in toggles:
                fig.add_trace(go.Scatter3d(
                    x=traj[:f+1,0], y=traj[:f+1,1], z=traj[:f+1,2],
                    mode='lines', line=dict(color=color, width=3),
                    hoverinfo='skip', showlegend=False))
                fig.add_trace(go.Scatter3d(
                    x=traj[f:,0], y=traj[f:,1], z=traj[f:,2],
                    mode='lines', line=dict(color=color, width=1, dash='dot'),
                    hoverinfo='skip', showlegend=False, opacity=0.2))

            # Object marker
            sym  = 'diamond' if obj["type"] == "PAYLOAD" else 'circle'
            sz   = 7 if obj["type"] == "PAYLOAD" else 6
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers+text',
                marker=dict(size=sz, color=color, symbol=sym, line=dict(width=1, color='white')),
                text=[obj["short"]], textposition="top center",
                textfont=dict(color=color, size=9, family="Rajdhani"),
                showlegend=False,
                hovertext=f"{obj['name']}<br>Type: {obj['type']}<br>Alt: {obj['altitude']:.0f} km<br>Spd: {obj['speed']:.3f} km/s<br>PINN Δr: {obj['dr']:.2f} km",
                hoverinfo='text'))

            # Debris cloud halo
            if toggles and "halos" in toggles and obj["type"] == "DEBRIS":
                u_h = np.linspace(0, 2*np.pi, 10)
                v_h = np.linspace(0, np.pi, 10)
                hr  = 80
                hx = pos[0] + hr * np.outer(np.cos(u_h), np.sin(v_h))
                hy = pos[1] + hr * np.outer(np.sin(u_h), np.sin(v_h))
                hz = pos[2] + hr * np.outer(np.ones(len(u_h)), np.cos(v_h))
                fig.add_trace(go.Surface(
                    x=hx, y=hy, z=hz,
                    colorscale=[[0,color],[1,color]],
                    showscale=False, opacity=0.08, hoverinfo='skip',
                    lighting=dict(ambient=1, diffuse=0, specular=0)))

            # Velocity vectors
            if toggles and "vectors" in toggles:
                vel = np.array(obj["vel_pinn"])
                vc  = '#00ff88' if obj["type"] == "PAYLOAD" else '#ff4444'
                fig.add_trace(go.Scatter3d(
                    x=[pos[0], pos[0]+vel[0]*500],
                    y=[pos[1], pos[1]+vel[1]*500],
                    z=[pos[2], pos[2]+vel[2]*500],
                    mode='lines', line=dict(color=vc, width=2),
                    hoverinfo='skip', showlegend=False))

        # ---- Collision / LOS lines between all close pairs ----
        if toggles and "killzone" in toggles:
            oid_list = [o for o in obj_ids if o in positions_now]
            for i in range(len(oid_list)):
                for j in range(i+1, len(oid_list)):
                    pa = positions_now[oid_list[i]]
                    pb = positions_now[oid_list[j]]
                    d  = np.linalg.norm(pa - pb)
                    if d < 5000:
                        if   d < 100:  lc, lw = '#ff0040', 3
                        elif d < 500:  lc, lw = '#ff4444', 2
                        elif d < 2000: lc, lw = '#ffaa00', 1.5
                        else:          lc, lw = 'rgba(255,255,255,0.06)', 1
                        fig.add_trace(go.Scatter3d(
                            x=[pa[0],pb[0]], y=[pa[1],pb[1]], z=[pa[2],pb[2]],
                            mode='lines', line=dict(color=lc, width=lw, dash='dot'),
                            hoverinfo='skip', showlegend=False))

        # ---- Highlight selected pair ----
        if selected_pair:
            ia, ib = selected_pair.get("a"), selected_pair.get("b")
            if ia in positions_now and ib in positions_now:
                pa, pb = positions_now[ia], positions_now[ib]
                fig.add_trace(go.Scatter3d(
                    x=[pa[0],pb[0]], y=[pa[1],pb[1]], z=[pa[2],pb[2]],
                    mode='lines', line=dict(color='#ffffff', width=4),
                    hoverinfo='skip', showlegend=False))
                for pp in (pa, pb):
                    fig.add_trace(go.Scatter3d(
                        x=[pp[0]], y=[pp[1]], z=[pp[2]],
                        mode='markers',
                        marker=dict(size=14, color='rgba(255,255,255,0)', symbol='circle',
                                    line=dict(width=2, color='#ffffff')),
                        hoverinfo='skip', showlegend=False))

        # ---- Camera Logic ----
        # Auto-orbit ONLY when toggle is on AND user hasn't interacted
        use_auto = bool(auto_orbit_toggle and "auto" in auto_orbit_toggle and not user_cam)
        if use_auto:
            angle = (frame / total) * 2 * math.pi * 0.3
            camera_dict = dict(eye=dict(x=2.2*math.cos(angle), y=2.2*math.sin(angle), z=0.5),
                               center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1))
        # else: camera_dict stays None → Plotly preserves user camera via uirevision

    # ---- Build scene dict ----
    scene_cfg = dict(
        xaxis=dict(visible=False, showgrid=False, zeroline=False, backgroundcolor='rgba(0,0,0,0)'),
        yaxis=dict(visible=False, showgrid=False, zeroline=False, backgroundcolor='rgba(0,0,0,0)'),
        zaxis=dict(visible=False, showgrid=False, zeroline=False, backgroundcolor='rgba(0,0,0,0)'),
        dragmode='turntable', aspectmode='data',
    )
    if camera_dict:
        scene_cfg['camera'] = camera_dict

    # uirevision: change every frame when auto-orbiting (forces camera update);
    # keep constant when user-controlled (preserves zoom/pan/rotate).
    ui_rev = f"auto-{frame}" if use_auto else "user-stable"

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=0),
        scene=scene_cfg,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        uirevision=ui_rev,
    )

    return fig, frame, frame_text


if __name__ == "__main__":
    app.run(debug=True, port=8051, host='0.0.0.0', use_reloader=False)
