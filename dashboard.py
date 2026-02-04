"""
Bot Dashboard — Single-file Flask web dashboard for managing the bot.
Usage: python3 dashboard.py
"""

import json
import os
import sqlite3
import subprocess
import time
from datetime import datetime
from functools import wraps
from pathlib import Path

import httpx
from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template_string,
    request,
    session,
    url_for,
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(32)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "bot.db"
BOT_LOG = BASE_DIR / "bot.log"
ENV_FILE = BASE_DIR / ".env"

DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "admin")
PORT = int(os.getenv("DASHBOARD_PORT", "5555"))

BOT_SERVICE = os.getenv("BOT_SERVICE_LABEL", "com.telegrambot.bot")

EDITABLE_KEYS = [
    "TEXT_BACKEND", "VISION_BACKEND", "OLLAMA_MODEL", "OLLAMA_VISION_MODEL",
    "VENICE_MODEL", "VENICE_VISION_MODEL",
    "BUFFER_SIZE", "MIN_MSG_GAP", "RANDOM_CHIME_CHANCE",
]


def fetch_venice_models():
    """Fetch text and vision model lists from Venice API, sorted by input price."""
    env = read_env()
    key = env.get("VENICE_API_KEY", "")
    if not key:
        return [], []
    try:
        resp = httpx.get(
            "https://api.venice.ai/api/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return [], []

    text_models = []
    vision_models = []

    for m in data.get("data", []):
        if m.get("type") != "text":
            continue
        spec = m.get("model_spec", {})
        if spec.get("offline"):
            continue
        mid = m["id"]
        name = spec.get("name", mid)
        pricing = spec.get("pricing", {})
        pin = pricing.get("input", {}).get("usd", 0)
        pout = pricing.get("output", {}).get("usd", 0)
        label = f"{name} — ${pin}/{pout}"
        caps = spec.get("capabilities", {})

        text_models.append((mid, label, pin))
        if caps.get("supportsVision"):
            vision_models.append((mid, label, pin))

    text_models.sort(key=lambda x: x[2])
    vision_models.sort(key=lambda x: x[2])

    return [(mid, label) for mid, label, _ in text_models], [(mid, label) for mid, label, _ in vision_models]


# ── DB helpers ─────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Auth ───────────────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authed"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ── Helpers ────────────────────────────────────────────────────────────────────

def is_process_running(name):
    try:
        result = subprocess.run(["pgrep", "-f", name], capture_output=True, text=True)
        pids = [p for p in result.stdout.strip().split("\n") if p]
        return pids if pids else None
    except Exception:
        return None


def get_launchd_status(label):
    try:
        result = subprocess.run(
            ["launchctl", "list", label], capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if '"PID"' in line or "PID" in line:
                    parts = line.strip().rstrip(";").split("=")
                    if len(parts) == 2:
                        return {"running": True, "pid": parts[1].strip().rstrip(";")}
            return {"running": True, "pid": "unknown"}
        return {"running": False, "pid": None}
    except Exception:
        return {"running": False, "pid": None}


def tail_file(path, lines=50):
    try:
        result = subprocess.run(
            ["tail", "-n", str(lines), str(path)],
            capture_output=True, text=True
        )
        return result.stdout
    except Exception:
        return f"Could not read {path}"


def read_env():
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                env[key.strip()] = val.strip()
    return env


def write_env(updates: dict):
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    updated_keys = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                updated_keys.add(key)
                continue
        new_lines.append(line)
    for key, val in updates.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={val}")
    ENV_FILE.write_text("\n".join(new_lines) + "\n")


def fetch_ollama_models():
    ollama_url = read_env().get("OLLAMA_URL", "http://localhost:11434")
    try:
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return sorted(m["name"] for m in data.get("models", []))
    except Exception:
        return []


def ts_format(ts):
    if not ts:
        return "N/A"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def render_page(content_template, active="", **kwargs):
    flash_msg = session.pop("flash_msg", None)
    flash_type = session.pop("flash_type", "success")
    full = BASE_HTML.replace("{% block content %}{% endblock %}", content_template)
    full = full.replace("{% block title_extra %}{% endblock %}", kwargs.pop("title_extra", ""))
    full = full.replace("{% block scripts %}{% endblock %}", kwargs.pop("extra_scripts", ""))
    return render_template_string(
        full, active=active, flash_msg=flash_msg, flash_type=flash_type, **kwargs
    )


def flash(msg, type_="success"):
    session["flash_msg"] = msg
    session["flash_type"] = type_


# ── Base template ──────────────────────────────────────────────────────────────

BASE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bot Dashboard{% block title_extra %}{% endblock %}</title>
<style>
:root {
    --bg: #0d0d0d;
    --surface: #1a1a1a;
    --surface2: #242424;
    --border: #333;
    --text: #e0e0e0;
    --text2: #888;
    --accent: #7c5cbf;
    --accent2: #5b9bd5;
    --danger: #c0392b;
    --success: #27ae60;
    --warn: #f39c12;
    --heated: #e74c3c;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
}
a { color: var(--accent2); text-decoration: none; }
a:hover { text-decoration: underline; }
.shell { display: flex; min-height: 100vh; }
.sidebar {
    width: 220px;
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 20px 0;
    position: fixed;
    top: 0; bottom: 0;
    overflow-y: auto;
}
.sidebar h1 { font-size: 18px; padding: 0 20px 16px; border-bottom: 1px solid var(--border); margin-bottom: 8px; color: var(--accent); letter-spacing: 1px; }
.sidebar a { display: block; padding: 10px 20px; color: var(--text2); font-size: 14px; border-left: 3px solid transparent; transition: all 0.15s; }
.sidebar a:hover, .sidebar a.active { background: var(--surface2); color: var(--text); border-left-color: var(--accent); text-decoration: none; }
.main { margin-left: 220px; flex: 1; padding: 24px 32px; min-width: 0; }
.page-title { font-size: 22px; font-weight: 600; margin-bottom: 20px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 16px; }
.card h3 { font-size: 14px; color: var(--text2); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
.stat { text-align: center; padding: 16px; }
.stat .val { font-size: 28px; font-weight: 700; color: var(--accent); }
.stat .label { font-size: 12px; color: var(--text2); margin-top: 4px; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
.badge-ok { background: rgba(39,174,96,0.2); color: var(--success); }
.badge-err { background: rgba(192,57,43,0.2); color: var(--danger); }
.badge-warn { background: rgba(243,156,18,0.2); color: var(--warn); }
.badge-heated { background: rgba(231,76,60,0.15); color: var(--heated); }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th { text-align: left; padding: 10px 12px; border-bottom: 2px solid var(--border); color: var(--text2); font-size: 12px; text-transform: uppercase; }
td { padding: 10px 12px; border-bottom: 1px solid var(--border); }
tr:hover { background: var(--surface2); }
input[type="text"], input[type="password"], input[type="search"], textarea, select {
    background: var(--surface2); border: 1px solid var(--border); color: var(--text);
    padding: 8px 12px; border-radius: 6px; font-size: 14px; width: 100%;
}
input:focus, textarea:focus, select:focus { outline: none; border-color: var(--accent); }
.btn { display: inline-block; padding: 8px 16px; border-radius: 6px; border: none; font-size: 13px; font-weight: 600; cursor: pointer; transition: opacity 0.15s; }
.btn:hover { opacity: 0.85; }
.btn-primary { background: var(--accent); color: #fff; }
.btn-danger { background: var(--danger); color: #fff; }
.btn-warn { background: var(--warn); color: #000; }
.btn-sm { padding: 4px 10px; font-size: 12px; }
.form-row { margin-bottom: 12px; }
.form-row label { display: block; font-size: 13px; color: var(--text2); margin-bottom: 4px; }
.log-viewer { background: #111; border: 1px solid var(--border); border-radius: 6px; padding: 12px; font-family: 'SF Mono','Menlo',monospace; font-size: 12px; line-height: 1.6; white-space: pre-wrap; word-break: break-all; max-height: 500px; overflow-y: auto; color: #aaa; }
.alert { padding: 10px 16px; border-radius: 6px; margin-bottom: 16px; font-size: 14px; }
.alert-success { background: rgba(39,174,96,0.15); color: var(--success); border: 1px solid rgba(39,174,96,0.3); }
.alert-error { background: rgba(192,57,43,0.15); color: var(--danger); border: 1px solid rgba(192,57,43,0.3); }
.flex { display: flex; gap: 8px; align-items: center; }
.flex-between { display: flex; justify-content: space-between; align-items: center; }
.mb-8 { margin-bottom: 8px; }
.mb-16 { margin-bottom: 16px; }
.mt-8 { margin-top: 8px; }
.mt-16 { margin-top: 16px; }
.section + .section { margin-top: 24px; }
@media (max-width: 768px) { .sidebar { display: none; } .main { margin-left: 0; padding: 16px; } .stats-grid { grid-template-columns: repeat(2, 1fr); } }
</style>
</head>
<body>
<div class="shell">
    <nav class="sidebar">
        <h1>BOT</h1>
        <a href="/" class="{{ 'active' if active == 'overview' }}">Overview</a>
        <a href="/personality" class="{{ 'active' if active == 'personality' }}">Personality</a>
        <a href="/messages" class="{{ 'active' if active == 'messages' }}">Messages</a>
        <a href="/grudges" class="{{ 'active' if active == 'grudges' }}">Grudges</a>
        <a href="/config" class="{{ 'active' if active == 'config' }}">Configuration</a>
    </nav>
    <div class="main">
        {% if flash_msg %}
        <div class="alert alert-{{ flash_type|default('success') }}">{{ flash_msg }}</div>
        {% endif %}
        {% block content %}{% endblock %}
    </div>
</div>
{% block scripts %}{% endblock %}
</body>
</html>"""


# ── Login ──────────────────────────────────────────────────────────────────────

LOGIN_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bot Dashboard - Login</title>
<style>
body { background: #0d0d0d; color: #e0e0e0; font-family: -apple-system, system-ui, sans-serif;
       display: flex; align-items: center; justify-content: center; min-height: 100vh; }
.login-box { background: #1a1a1a; border: 1px solid #333; border-radius: 12px; padding: 40px;
             width: 320px; text-align: center; }
.login-box h1 { color: #7c5cbf; margin-bottom: 24px; font-size: 20px; letter-spacing: 1px; }
.login-box input { background: #242424; border: 1px solid #333; color: #e0e0e0; padding: 10px 14px;
                   border-radius: 6px; width: 100%; font-size: 14px; margin-bottom: 16px; }
.login-box input:focus { outline: none; border-color: #7c5cbf; }
.login-box button { background: #7c5cbf; color: #fff; border: none; padding: 10px 24px;
                    border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer; width: 100%; }
.login-box button:hover { opacity: 0.85; }
.err { color: #c0392b; font-size: 13px; margin-bottom: 12px; }
</style></head><body>
<div class="login-box">
    <h1>BOT DASHBOARD</h1>
    {% if error %}<div class="err">{{ error }}</div>{% endif %}
    <form method="POST">
        <input type="password" name="password" placeholder="Password" autofocus>
        <button type="submit">Enter</button>
    </form>
</div>
</body></html>"""


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("password") == DASHBOARD_PASSWORD:
            session["authed"] = True
            return redirect("/")
        error = "Wrong password"
    return render_template_string(LOGIN_HTML, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# ── Overview ───────────────────────────────────────────────────────────────────

@app.route("/")
@login_required
def overview():
    db = get_db()
    now = time.time()
    today_start = now - (now % 86400)
    week_start = now - 7 * 86400

    total_messages = db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    today_messages = db.execute(
        "SELECT COUNT(*) FROM messages WHERE timestamp >= ?", (today_start,)
    ).fetchone()[0]
    week_messages = db.execute(
        "SELECT COUNT(*) FROM messages WHERE timestamp >= ?", (week_start,)
    ).fetchone()[0]
    grudge_count = db.execute("SELECT COUNT(*) FROM grudges").fetchone()[0]
    override_count = db.execute("SELECT COUNT(*) FROM personality_overrides").fetchone()[0]
    db.close()

    bot_status = get_launchd_status(BOT_SERVICE)
    bot_log = tail_file(BOT_LOG, 30)

    content = """
<div class="page-title">Overview</div>
<div class="stats-grid">
    <div class="card stat"><div class="val">{{ total_messages }}</div><div class="label">Total Messages</div></div>
    <div class="card stat"><div class="val">{{ today_messages }}</div><div class="label">Today</div></div>
    <div class="card stat"><div class="val">{{ week_messages }}</div><div class="label">This Week</div></div>
    <div class="card stat"><div class="val">{{ grudge_count }}</div><div class="label">Grudges</div></div>
    <div class="card stat"><div class="val">{{ override_count }}</div><div class="label">Personality Overrides</div></div>
</div>
<div class="card section">
    <h3>Service Status</h3>
    <table>
        <tr>
            <td style="width:200px">Bot</td>
            <td>{% if bot_status.running %}<span class="badge badge-ok">Running</span> PID {{ bot_status.pid }}{% else %}<span class="badge badge-err">Stopped</span>{% endif %}</td>
            <td style="width:100px"><form method="POST" action="/bot/restart" style="display:inline"><button class="btn btn-sm btn-warn" type="submit">Restart</button></form></td>
        </tr>
    </table>
</div>
<div class="card section">
    <h3>Recent Bot Log</h3>
    <div class="log-viewer">{{ bot_log }}</div>
</div>
"""
    return render_page(
        content, active="overview",
        total_messages=total_messages, today_messages=today_messages,
        week_messages=week_messages, grudge_count=grudge_count,
        override_count=override_count,
        bot_status=bot_status, bot_log=bot_log,
    )


# ── Personality ────────────────────────────────────────────────────────────────

@app.route("/personality")
@login_required
def personality():
    db = get_db()
    overrides = db.execute(
        "SELECT id, instruction, added_by, timestamp FROM personality_overrides ORDER BY id"
    ).fetchall()
    db.close()

    content = """
<div class="page-title">Personality</div>

<div class="card section">
    <h3>Add Override</h3>
    <form method="POST" action="/personality/add">
        <div class="form-row">
            <label>New personality instruction</label>
            <textarea name="instruction" placeholder="e.g. You are sarcastic and dry"></textarea>
        </div>
        <div class="form-row">
            <label>Added by</label>
            <input type="text" name="added_by" value="dashboard" style="width:200px">
        </div>
        <button class="btn btn-primary" type="submit">Add Override</button>
    </form>
</div>

<div class="card section">
    <div class="flex-between mb-8">
        <h3 style="margin:0">Active Overrides ({{ overrides|length }})</h3>
        {% if overrides %}
        <form method="POST" action="/personality/clear">
            <button class="btn btn-sm btn-danger" type="submit" onclick="return confirm('Clear ALL overrides?')">Clear All</button>
        </form>
        {% endif %}
    </div>
    {% if overrides %}
    <table>
        <thead><tr><th>Instruction</th><th>Added By</th><th>When</th><th></th></tr></thead>
        <tbody>
        {% for o in overrides %}
        <tr>
            <td>{{ o.instruction }}</td>
            <td>{{ o.added_by }}</td>
            <td style="white-space:nowrap">{{ ts_format(o.timestamp) }}</td>
            <td><form method="POST" action="/personality/delete/{{ o.id }}"><button class="btn btn-sm btn-danger" type="submit">Delete</button></form></td>
        </tr>
        {% endfor %}
        </tbody>
    </table>
    {% else %}
    <p style="color:var(--text2)">No overrides active. Base personality only.</p>
    {% endif %}
</div>
"""
    return render_page(content, active="personality", overrides=overrides, ts_format=ts_format)


@app.route("/personality/add", methods=["POST"])
@login_required
def personality_add():
    instruction = request.form.get("instruction", "").strip()
    added_by = request.form.get("added_by", "dashboard").strip()
    if instruction:
        db = get_db()
        db.execute(
            "INSERT INTO personality_overrides (instruction, added_by, timestamp) VALUES (?, ?, ?)",
            (instruction, added_by, time.time()),
        )
        db.commit()
        db.close()
        flash("Personality override added")
    return redirect("/personality")


@app.route("/personality/delete/<int:oid>", methods=["POST"])
@login_required
def personality_delete(oid):
    db = get_db()
    db.execute("DELETE FROM personality_overrides WHERE id = ?", (oid,))
    db.commit()
    db.close()
    flash("Override deleted")
    return redirect("/personality")


@app.route("/personality/clear", methods=["POST"])
@login_required
def personality_clear():
    db = get_db()
    db.execute("DELETE FROM personality_overrides")
    db.commit()
    db.close()
    flash("All overrides cleared")
    return redirect("/personality")


# ── Messages ──────────────────────────────────────────────────────────────────

@app.route("/messages")
@login_required
def messages():
    db = get_db()
    q = request.args.get("q", "").strip()
    user_filter = request.args.get("user", "").strip()
    flag = request.args.get("flag", "")
    page_num = max(1, int(request.args.get("page", 1)))
    per_page = 50

    where = []
    params = []
    if q:
        where.append("text LIKE ?")
        params.append(f"%{q}%")
    if user_filter:
        where.append("user_name = ?")
        params.append(user_filter)
    if flag == "heated":
        where.append("is_heated = 1")
    elif flag == "bot":
        where.append("is_bot = 1")

    where_clause = " AND ".join(where) if where else "1=1"

    total = db.execute(f"SELECT COUNT(*) FROM messages WHERE {where_clause}", params).fetchone()[0]
    rows = db.execute(
        f"SELECT * FROM messages WHERE {where_clause} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        params + [per_page, (page_num - 1) * per_page],
    ).fetchall()

    users = db.execute("SELECT DISTINCT user_name FROM messages ORDER BY user_name").fetchall()
    db.close()

    total_pages = max(1, (total + per_page - 1) // per_page)

    content = """
<div class="page-title">Messages</div>

<div class="card section">
    <form method="GET" action="/messages">
        <div class="flex">
            <input type="search" name="q" value="{{ q }}" placeholder="Search messages..." style="flex:2">
            <select name="user" style="flex:1">
                <option value="">All users</option>
                {% for u in users %}<option value="{{ u.user_name }}" {{ 'selected' if u.user_name == user_filter }}>{{ u.user_name }}</option>{% endfor %}
            </select>
            <select name="flag" style="width:120px">
                <option value="">All</option>
                <option value="heated" {{ 'selected' if flag == 'heated' }}>Heated</option>
                <option value="bot" {{ 'selected' if flag == 'bot' }}>Bot</option>
            </select>
            <button class="btn btn-primary" type="submit">Filter</button>
        </div>
    </form>
</div>

<div class="card section">
    <div class="flex-between mb-8">
        <h3 style="margin:0">{{ total }} messages{% if q %} matching "{{ q }}"{% endif %}</h3>
        <div class="flex" style="font-size:13px;color:var(--text2)">
            Page {{ page_num }} of {{ total_pages }}
            {% if page_num > 1 %}<a href="?q={{ q }}&user={{ user_filter }}&flag={{ flag }}&page={{ page_num-1 }}">&laquo; Prev</a>{% endif %}
            {% if page_num < total_pages %}<a href="?q={{ q }}&user={{ user_filter }}&flag={{ flag }}&page={{ page_num+1 }}">Next &raquo;</a>{% endif %}
        </div>
    </div>
    <table>
        <thead><tr><th>User</th><th>Message</th><th>Flags</th><th>Time</th></tr></thead>
        <tbody>
        {% for m in rows %}
        <tr>
            <td>{{ m.user_name }}</td>
            <td style="max-width:500px;overflow:hidden;text-overflow:ellipsis">{{ m.text[:200] if m.text else '[media]' }}{% if m.image_desc %} <span style="color:var(--text2)">[img: {{ m.image_desc[:50] }}]</span>{% endif %}</td>
            <td style="white-space:nowrap">
                {% if m.is_heated %}<span class="badge badge-heated">heated</span> {% endif %}
                {% if m.is_bot %}<span class="badge badge-ok">bot</span> {% endif %}
            </td>
            <td style="white-space:nowrap">{{ ts_format(m.timestamp) }}</td>
        </tr>
        {% endfor %}
        </tbody>
    </table>
</div>
"""
    return render_page(
        content, active="messages",
        rows=rows, users=users, q=q, user_filter=user_filter,
        flag=flag, total=total, page_num=page_num, total_pages=total_pages,
        ts_format=ts_format,
    )


# ── Grudges ───────────────────────────────────────────────────────────────────

@app.route("/grudges")
@login_required
def grudges():
    db = get_db()
    rows = db.execute(
        "SELECT id, chat_id, user_name, reason, timestamp FROM grudges ORDER BY timestamp DESC"
    ).fetchall()
    db.close()

    content = """
<div class="page-title">Grudges</div>

<div class="card section">
    <h3>Add Grudge</h3>
    <form method="POST" action="/grudges/add">
        <div class="flex">
            <input type="text" name="user_name" placeholder="Username" style="flex:1" required>
            <input type="text" name="reason" placeholder="Reason" style="flex:3" required>
            <input type="text" name="chat_id" placeholder="Chat ID" value="0" style="width:200px">
            <button class="btn btn-primary" type="submit">Add</button>
        </div>
    </form>
</div>

<div class="card section">
    <h3>All Grudges ({{ rows|length }})</h3>
    {% if rows %}
    <table>
        <thead><tr><th>User</th><th>Reason</th><th>Chat</th><th>When</th><th></th></tr></thead>
        <tbody>
        {% for g in rows %}
        <tr>
            <td>{{ g.user_name }}</td>
            <td>{{ g.reason }}</td>
            <td style="font-size:12px;color:var(--text2)">{{ g.chat_id }}</td>
            <td style="white-space:nowrap">{{ ts_format(g.timestamp) }}</td>
            <td><form method="POST" action="/grudges/delete/{{ g.id }}"><button class="btn btn-sm btn-danger" type="submit">Delete</button></form></td>
        </tr>
        {% endfor %}
        </tbody>
    </table>
    {% else %}
    <p style="color:var(--text2)">No grudges recorded.</p>
    {% endif %}
</div>
"""
    return render_page(content, active="grudges", rows=rows, ts_format=ts_format)


@app.route("/grudges/add", methods=["POST"])
@login_required
def grudges_add():
    user_name = request.form.get("user_name", "").strip()
    reason = request.form.get("reason", "").strip()
    chat_id = request.form.get("chat_id", "0").strip()
    if user_name and reason:
        db = get_db()
        db.execute(
            "INSERT INTO grudges (chat_id, user_name, reason, timestamp) VALUES (?, ?, ?, ?)",
            (int(chat_id), user_name, reason[:500], time.time()),
        )
        db.commit()
        db.close()
        flash(f"Grudge added against {user_name}")
    return redirect("/grudges")


@app.route("/grudges/delete/<int:gid>", methods=["POST"])
@login_required
def grudges_delete(gid):
    db = get_db()
    db.execute("DELETE FROM grudges WHERE id = ?", (gid,))
    db.commit()
    db.close()
    flash("Grudge deleted")
    return redirect("/grudges")


# ── Configuration ─────────────────────────────────────────────────────────────

@app.route("/config")
@login_required
def config():
    env = read_env()
    bot_log = tail_file(BOT_LOG, 100)
    ollama_models = fetch_ollama_models()
    venice_text_models, venice_vision_models = fetch_venice_models()

    content = """
<div class="page-title">Configuration</div>

<div class="card section">
    <h3>Text</h3>
    <form method="POST" action="/config/save">

        <div class="form-row">
            <label>TEXT_BACKEND</label>
            <select name="TEXT_BACKEND" style="width:500px">
                <option value="venice" {{ 'selected' if env.get('TEXT_BACKEND','venice') == 'venice' }}>venice</option>
                <option value="ollama" {{ 'selected' if env.get('TEXT_BACKEND') == 'ollama' }}>ollama</option>
            </select>
        </div>

        <div class="form-row">
            <label>VENICE_MODEL <span style="color:var(--text2);font-weight:normal">(input/output per 1M tokens)</span></label>
            <select name="VENICE_MODEL" style="width:500px">
                {% for mid, mlabel in venice_text_models %}
                <option value="{{ mid }}" {{ 'selected' if mid == env.get('VENICE_MODEL','llama-3.3-70b') }}>{{ mlabel }}</option>
                {% endfor %}
                {% if not venice_text_models %}
                <option value="{{ env.get('VENICE_MODEL','llama-3.3-70b') }}" selected>{{ env.get('VENICE_MODEL','llama-3.3-70b') }} (API unavailable)</option>
                {% endif %}
            </select>
        </div>

        <div class="form-row">
            <label>OLLAMA_MODEL</label>
            <select name="OLLAMA_MODEL" style="width:500px">
                {% for m in ollama_models %}
                <option value="{{ m }}" {{ 'selected' if m == env.get('OLLAMA_MODEL','qwen3:14b') }}>{{ m }}</option>
                {% endfor %}
                {% if not ollama_models %}
                <option value="{{ env.get('OLLAMA_MODEL','qwen3:14b') }}" selected>{{ env.get('OLLAMA_MODEL','qwen3:14b') }} (ollama offline)</option>
                {% endif %}
            </select>
        </div>
</div>

<div class="card section">
    <h3>Vision</h3>

        <div class="form-row">
            <label>VISION_BACKEND</label>
            <select name="VISION_BACKEND" style="width:500px">
                <option value="" {{ 'selected' if not env.get('VISION_BACKEND') }}>same as text backend</option>
                <option value="venice" {{ 'selected' if env.get('VISION_BACKEND') == 'venice' }}>venice</option>
                <option value="ollama" {{ 'selected' if env.get('VISION_BACKEND') == 'ollama' }}>ollama</option>
            </select>
        </div>

        <div class="form-row">
            <label>VENICE_VISION_MODEL <span style="color:var(--text2);font-weight:normal">(vision-capable models only, input/output per 1M tokens)</span></label>
            <select name="VENICE_VISION_MODEL" style="width:500px">
                {% for mid, mlabel in venice_vision_models %}
                <option value="{{ mid }}" {{ 'selected' if mid == env.get('VENICE_VISION_MODEL','mistral-31-24b') }}>{{ mlabel }}</option>
                {% endfor %}
                {% if not venice_vision_models %}
                <option value="{{ env.get('VENICE_VISION_MODEL','mistral-31-24b') }}" selected>{{ env.get('VENICE_VISION_MODEL','mistral-31-24b') }} (API unavailable)</option>
                {% endif %}
            </select>
        </div>

        <div class="form-row">
            <label>OLLAMA_VISION_MODEL</label>
            <select name="OLLAMA_VISION_MODEL" style="width:500px">
                {% for m in ollama_models %}
                <option value="{{ m }}" {{ 'selected' if m == env.get('OLLAMA_VISION_MODEL','llava:7b') }}>{{ m }}</option>
                {% endfor %}
                {% if not ollama_models %}
                <option value="{{ env.get('OLLAMA_VISION_MODEL','llava:7b') }}" selected>{{ env.get('OLLAMA_VISION_MODEL','llava:7b') }} (ollama offline)</option>
                {% endif %}
            </select>
        </div>
</div>

<div class="card section">
    <h3>Tuning</h3>

        <div class="form-row">
            <label>BUFFER_SIZE</label>
            <input type="text" name="BUFFER_SIZE" value="{{ env.get('BUFFER_SIZE', '50') }}" style="width:500px">
        </div>
        <div class="form-row">
            <label>MIN_MSG_GAP</label>
            <input type="text" name="MIN_MSG_GAP" value="{{ env.get('MIN_MSG_GAP', '15') }}" style="width:500px">
        </div>
        <div class="form-row">
            <label>RANDOM_CHIME_CHANCE</label>
            <input type="text" name="RANDOM_CHIME_CHANCE" value="{{ env.get('RANDOM_CHIME_CHANCE', '0.08') }}" style="width:500px">
        </div>

        <button class="btn btn-primary mt-8" type="submit">Save &amp; Restart Bot</button>
    </form>
</div>

<div class="card section">
    <h3>Services</h3>
    <div class="flex">
        <form method="POST" action="/bot/restart"><button class="btn btn-warn" type="submit">Restart Bot</button></form>
        <form method="POST" action="/bot/stop"><button class="btn btn-danger" type="submit">Stop Bot</button></form>
        <form method="POST" action="/bot/start"><button class="btn btn-primary" type="submit">Start Bot</button></form>
    </div>
</div>

<div class="card section">
    <h3>Bot Log (last 100 lines)</h3>
    <div class="log-viewer">{{ bot_log }}</div>
</div>
"""
    return render_page(
        content, active="config",
        env=env, bot_log=bot_log,
        ollama_models=ollama_models,
        venice_text_models=venice_text_models,
        venice_vision_models=venice_vision_models,
    )


@app.route("/config/save", methods=["POST"])
@login_required
def config_save():
    updates = {}
    for key in EDITABLE_KEYS:
        val = request.form.get(key)
        if val is not None:
            updates[key] = val
    write_env(updates)
    try:
        subprocess.run(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{BOT_SERVICE}"],
                        capture_output=True, timeout=10)
        flash("Configuration saved. Bot restarting...")
    except Exception:
        try:
            subprocess.run(["launchctl", "stop", BOT_SERVICE], capture_output=True, timeout=5)
            time.sleep(1)
            subprocess.run(["launchctl", "start", BOT_SERVICE], capture_output=True, timeout=5)
            flash("Configuration saved. Bot restarting...")
        except Exception:
            flash("Configuration saved. Restart bot manually to apply.")
    return redirect("/config")


# ── Bot management ────────────────────────────────────────────────────────────

@app.route("/bot/restart", methods=["POST"])
@login_required
def bot_restart():
    try:
        subprocess.run(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{BOT_SERVICE}"],
                        capture_output=True, timeout=10)
        flash("Bot restart triggered")
    except Exception:
        try:
            subprocess.run(["launchctl", "stop", BOT_SERVICE], capture_output=True, timeout=5)
            time.sleep(1)
            subprocess.run(["launchctl", "start", BOT_SERVICE], capture_output=True, timeout=5)
            flash("Bot restarted (stop/start)")
        except Exception as e2:
            flash(f"Error restarting bot: {e2}", "error")
    return redirect(request.referrer or "/")


@app.route("/bot/stop", methods=["POST"])
@login_required
def bot_stop():
    try:
        subprocess.run(["launchctl", "stop", BOT_SERVICE], capture_output=True, timeout=5)
        flash("Bot stopped")
    except Exception as e:
        flash(f"Error: {e}", "error")
    return redirect("/config")


@app.route("/bot/start", methods=["POST"])
@login_required
def bot_start():
    try:
        subprocess.run(["launchctl", "start", BOT_SERVICE], capture_output=True, timeout=5)
        flash("Bot started")
    except Exception as e:
        flash(f"Error: {e}", "error")
    return redirect("/config")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Bot Dashboard running on http://localhost:{PORT}")
    print(f"Password: {'(set via DASHBOARD_PASSWORD)' if os.getenv('DASHBOARD_PASSWORD') else DASHBOARD_PASSWORD}")
    app.run(host="0.0.0.0", port=PORT, debug=True)
