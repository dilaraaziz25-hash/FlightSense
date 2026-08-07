"""
app_nicegui.py — FlightSense UI (NiceGUI version)
Same data layer as Streamlit version (flights.json, pnr_database.json, disruption_log.json).
Adds: mic input for passenger turns via browser Web Speech API.

Install: pip install nicegui airportsdata python-dotenv boto3
Run:     python app_nicegui.py
"""

import asyncio
import base64
import datetime
import io
import json
import math
import os
import subprocess
import sys

import boto3
import airportsdata
from dotenv import load_dotenv
from nicegui import app, ui

load_dotenv()

# ── AWS clients ───────────────────────────────────────────────────────────────

bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
polly   = boto3.client("polly",           region_name=os.getenv("AWS_REGION", "us-east-1"))

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# ── Airport helpers ───────────────────────────────────────────────────────────

airports_db = airportsdata.load("IATA")
city_cache  = {}

def get_city(iata):
    if iata in city_cache:
        return city_cache[iata]
    try:
        a = airports_db.get(iata)
        c = a["city"].title() if a else iata
        city_cache[iata] = c
        return c
    except:
        return iata

def get_coords(iata):
    a = airports_db.get(iata)
    return (a["lat"], a["lon"]) if a else None

# ── Distance ──────────────────────────────────────────────────────────────────

LONG_HAUL_KM   = 3000
HIGH_VALUE_USD  = 1000
DISRUPTED       = {"cancelled", "delayed", "diverted", "incident"}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def check_long_haul(origin, dest):
    o, d = airports_db.get(origin), airports_db.get(dest)
    if not o or not d:
        return False, 0
    dist = haversine(o["lat"], o["lon"], d["lat"], d["lon"])
    return dist > LONG_HAUL_KM, round(dist)

# ── Data helpers ──────────────────────────────────────────────────────────────

def load_flights():
    try:
        with open("data/flights.json") as f:
            data = json.load(f)
        for fl in data:
            s = fl.get("status", "").lower()
            fl["status"] = "cancelled" if s == "canceled" else s
        return data
    except:
        return []

def load_passengers():
    try:
        with open("data/pnr_database.json") as f:
            return json.load(f)
    except:
        return {}

def load_logs():
    try:
        with open("data/disruption_log.json") as f:
            return json.load(f)
    except:
        return []

def mark_processed(flight_iata):
    try:
        with open("data/flights.json") as f:
            data = json.load(f)
        for fl in data:
            if fl["flight_iata"] == flight_iata:
                fl["processed"] = True
                break
        with open("data/flights.json", "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        ui.notify(f"Could not mark processed: {e}", type="negative")

def write_log(flight_iata, status, airline, destination, resolution):
    entry = {
        "flight_iata":    flight_iata,
        "airline":        airline,
        "destination":    destination,
        "status":         status,
        "affected_count": 1,
        "resolution":     resolution,
        "timestamp":      str(datetime.datetime.now()),
    }
    logs = load_logs()
    logs.append(entry)
    with open("data/disruption_log.json", "w") as f:
        json.dump(logs, f, indent=2)

# ── Polly ─────────────────────────────────────────────────────────────────────

def synth(text):
    r = polly.synthesize_speech(Text=text, OutputFormat="mp3", VoiceId="Amy", Engine="neural")
    return r["AudioStream"].read()

def audio_b64(audio_bytes):
    return base64.b64encode(audio_bytes).decode()

# ── Emma (Bedrock) ────────────────────────────────────────────────────────────

CS_SYSTEM_TEMPLATE = """You are Emma, a customer service agent at {airline} handling a flight disruption.
Tone: professional, warm, efficient — the way an experienced airline agent speaks, not a script.
Use the passenger's first name naturally, not in every line. Use city names, not airport codes.
No markdown formatting. Keep responses concise and focused.
Do not invent flight times or numbers not already provided.
Never claim to be human or an AI. If asked, say warmly: "I'm Emma, and I'm here to help you."
If a passenger asks to speak with a human agent or a supervisor, acknowledge the request
professionally and let them know you're connecting them now — do not argue or redirect them
back to yourself."""

def emma_call(prompt, history, airline="the airline"):
    system_prompt = CS_SYSTEM_TEMPLATE.format(airline=airline)
    msgs = history + [{"role": "user", "content": [{"text": prompt}]}]
    r = bedrock.converse(
        modelId=MODEL_ID,
        system=[{"text": system_prompt}],
        messages=msgs,
        inferenceConfig={"maxTokens": 300},
    )
    return r["output"]["message"]["content"][0]["text"]

def classify(reply, history):
    lines = []
    for m in history[-6:]:
        who = "Emma" if m["role"] == "assistant" else "Passenger"
        lines.append(f"{who}: {m['content'][0]['text']}")
    ctx = "\n".join(lines)
    prompt = f"""Conversation:\n{ctx}\nPassenger's latest: "{reply}"
Reply with exactly one word: refund, rebook, voucher, or unclear."""
    r = bedrock.converse(
        modelId=MODEL_ID,
        system=[{"text": "Intent classifier. One word only: refund, rebook, voucher, or unclear."}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 10},
    )
    w = r["output"]["message"]["content"][0]["text"].strip().lower()
    return w if w in {"refund", "rebook", "voucher"} else "unknown"

def wants_human(text):
    t = text.lower()
    return any(w in t for w in [
        "human", "real person", "real agent", "speak to someone",
        "supervisor", "manager", "not a bot", "not a robot", "talk to a person"
    ])

# ── App state (single-user POC) ───────────────────────────────────────────────

state = {
    "data_loaded":    False,
    "cs_phase":       "idle",   # idle | turn1 | turn2 | approval | approved | turn3 | complete
    "cs_passenger":   None,
    "cs_messages":    [],
    "cs_choice":      "",
    "cs_handoff":     "",
    "cs_distance_km": 0,
}

# ── NiceGUI page ──────────────────────────────────────────────────────────────

@ui.page("/")
def main_page():

    # ── Dark theme + global styles ────────────────────────────────────────────
    ui.dark_mode().enable()
    ui.add_head_html("""
    <style>
        body { background: #0a0a0a; font-family: monospace; font-size: 16px; }
        .nicegui-content { background: #0a0a0a !important; }
        .emma-bubble  { background:#1a1a2e; border-left:4px solid #FFD700;
                        padding:14px; margin:8px 0; border-radius:6px; color:#fff; font-size:16px; }
        .pax-bubble   { background:#2a2a2a; border-left:4px solid #888;
                        padding:14px; margin:8px 0; border-radius:6px; color:#fff; font-size:16px; }
        .flight-row   { padding:8px 0; border-bottom:1px solid #222; }
        .badge-high   { color:#00CC66; font-size:13px; }
        .badge-conf   { color:#FF8800; font-size:13px; }
        .badge-single { color:#888; font-size:13px; }
    </style>
    """)

    # ── Header ────────────────────────────────────────────────────────────────
    with ui.column().classes("w-full q-pa-md").style("gap:2px;"):
        ui.label("✈️ FLIGHTSENSE").style(
            "color:#FFD700; font-size:3rem; font-weight:bold; letter-spacing:4px; font-family:monospace;"
        )
        ui.label("FLIGHT DISRUPTION INTELLIGENCE").style(
            "color:#666; font-size:1.1rem; font-family:monospace; letter-spacing:2px;"
        )

    ui.separator().style("border-color:#333;")

    # ── RUN AGENT / REFRESH DATA ──────────────────────────────────────────────

    flights_exist = os.path.exists("data/flights.json") and os.path.getsize("data/flights.json") > 10

    with ui.column().classes("q-pa-md w-full").style("gap:8px;"):

        with ui.row().classes("items-center gap-4"):

            async def on_run():
                is_first = not os.path.exists("data/flights.json")
                script   = "pnr_generator_demo_bedrock.py" if is_first else "fetch_live_status.py"
                run_btn.disable()
                run_btn.set_text("⏳ Working...")

                # ── Live step-by-step status ──────────────────────────────
                status_label.set_text("Fetching AviationStack...")
                aviationstack_badge.set_text("AviationStack ...")
                aviationstack_badge.style("color:#FFD700; font-family:monospace; font-size:13px; background:#1a1a0a; border:1px solid #FFD700; padding:3px 10px; border-radius:12px;")
                opensky_badge.set_text("AeroDataBox ...")
                opensky_badge.style("color:#FFD700; font-family:monospace; font-size:13px; background:#1a1a0a; border:1px solid #FFD700; padding:3px 10px; border-radius:12px;")
                opensky_tb_badge.set_text("OpenSky (conflict tiebreaker)")
                opensky_tb_badge.style(BADGE_DIM)

                result = await asyncio.to_thread(
                    subprocess.run,
                    [sys.executable, script],
                    capture_output=True, text=True,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )

                # Ping AeroDataBox to confirm the second source is reachable
                def check_aerodatabox():
                    try:
                        import requests as _req
                        adb_key = os.getenv("AERODATABOX_KEY", "")
                        if not adb_key:
                            return False
                        # Use a lightweight airport lookup to confirm the key works
                        r = _req.get(
                            "https://prod.api.market/api/v1/aedbx/aerodatabox/airports/iata/LHR",
                            headers={"x-api-market-key": adb_key},
                            timeout=8
                        )
                        return r.status_code == 200
                    except Exception:
                        return False

                adb_ok = await asyncio.to_thread(check_aerodatabox)

                # Always reset the disruption log on a new fetch — old processed entries
                # are stale once fresh flight data comes in
                with open("data/disruption_log.json", "w") as f:
                    json.dump([], f)

                state["data_loaded"] = True
                state["cs_phase"]    = "idle"
                state["cs_messages"] = []
                state["cs_passenger"] = None
                run_btn.enable()
                run_btn.set_text("🔄 REFRESH DATA" if os.path.exists("data/flights.json") else "▶ RUN AGENT")

                # ── Update badges permanently ─────────────────────────────
                # Check if any conflicts exist (OpenSky was called as tiebreaker)
                try:
                    _flights = json.load(open("data/flights.json"))
                    _has_conflict = any(f.get("confidence") in ("conflict", "conflict_operating", "confirmed") for f in _flights)
                    _opensky_used = any("OpenSky" in f.get("sources", []) for f in _flights)
                except Exception:
                    _has_conflict = False
                    _opensky_used = False

                n_sources = "3" if _opensky_used else ("2" if adb_ok else "1")
                status_label.set_text(f"✅ Data loaded — {n_sources} source{'s' if int(n_sources) > 1 else ''} active")
                aviationstack_badge.set_text("AviationStack ✓")
                aviationstack_badge.style(BADGE_GREEN)
                if adb_ok:
                    opensky_badge.set_text("AeroDataBox ✓")
                    opensky_badge.style(BADGE_GREEN)
                else:
                    opensky_badge.set_text("AeroDataBox — unavailable")
                    opensky_badge.style(BADGE_ORANGE)
                if _opensky_used:
                    opensky_tb_badge.set_text("OpenSky ✓ (tiebreaker)")
                    opensky_tb_badge.style(BADGE_GREEN)
                elif _has_conflict:
                    opensky_tb_badge.set_text("OpenSky — unavailable (conflict unresolved)")
                    opensky_tb_badge.style(BADGE_ORANGE)
                else:
                    opensky_tb_badge.set_text("OpenSky (conflict tiebreaker)")
                    opensky_tb_badge.style(BADGE_DIM)

                refresh_all()

            btn_text = "🔄 REFRESH DATA" if flights_exist else "▶ RUN AGENT"
            run_btn = ui.button(btn_text, on_click=on_run).style(
                "background:#1a1a2e; color:#FFD700; border:1px solid #FFD700; font-family:monospace;"
            )
            status_label = ui.label("Awaiting agent start..." if not flights_exist else "Data available — click REFRESH to update").style(
                "color:#888; font-family:monospace; font-size:0.85rem;"
            )
            if flights_exist:
                state["data_loaded"] = True

        # ── Persistent source badges (shown after first run) ──────────────
        BADGE_GREEN  = "color:#00CC66; font-family:monospace; font-size:13px; background:#0a1a0a; border:1px solid #00CC66; padding:3px 10px; border-radius:12px;"
        BADGE_ORANGE = "color:#FF8800; font-family:monospace; font-size:13px; background:#1a1000; border:1px solid #FF8800; padding:3px 10px; border-radius:12px;"
        BADGE_DIM    = "color:#555; font-family:monospace; font-size:13px; background:#111; border:1px solid #333; padding:3px 10px; border-radius:12px;"

        with ui.row().classes("items-center gap-3").style("margin-top:2px;"):
            av_text  = "AviationStack ✓" if flights_exist else "AviationStack"
            adb_text = "AeroDataBox ✓"   if flights_exist else "AeroDataBox"
            os_text  = "OpenSky (conflict tiebreaker)"
            av_style  = BADGE_GREEN if flights_exist else BADGE_DIM
            adb_style = BADGE_GREEN if flights_exist else BADGE_DIM
            aviationstack_badge = ui.label(av_text).style(av_style)
            opensky_badge       = ui.label(adb_text).style(adb_style)
            opensky_tb_badge    = ui.label(os_text).style(BADGE_DIM)

    ui.separator().style("border-color:#333;")

    # ── Tabs ──────────────────────────────────────────────────────────────────

    with ui.tabs().classes("w-full") as tabs:
        t1 = ui.tab("✈️  FLIGHT MONITOR")
        t2 = ui.tab("🚨  AFFECTED PASSENGERS")
        t3 = ui.tab("📞  CUSTOMER SERVICE")
        t4 = ui.tab("📊  ANALYTICS")
        t5 = ui.tab("📋  DISRUPTION LOG")

    with ui.tab_panels(tabs, value=t1).classes("w-full"):

        # ── Tab 1: Flight Monitor ─────────────────────────────────────────────
        with ui.tab_panel(t1):
            t1_container = ui.column().classes("w-full q-pa-md")

        # ── Tab 2: Affected Passengers ────────────────────────────────────────
        with ui.tab_panel(t2):
            t2_container = ui.column().classes("w-full q-pa-md")

        # ── Tab 3: Customer Service ───────────────────────────────────────────
        with ui.tab_panel(t3):
            t3_container = ui.column().classes("w-full q-pa-md")

        # ── Tab 4: Analytics ─────────────────────────────────────────────────
        with ui.tab_panel(t4):
            t4_container = ui.column().classes("w-full q-pa-md")

        # ── Tab 5: Disruption Log ─────────────────────────────────────────────
        with ui.tab_panel(t5):
            t5_container = ui.column().classes("w-full q-pa-md")

    # ── Render helpers ────────────────────────────────────────────────────────

    WAITING = "<p style='color:#444;font-family:monospace;'>⏳ Awaiting data — click RUN AGENT to fetch live flights.</p>"

    def render_tab1():
        t1_container.clear()
        with t1_container:
            if not state["data_loaded"]:
                ui.html(WAITING); return
            flights = load_flights()
            if not flights:
                ui.html(WAITING); return

            disrupted_n  = sum(1 for f in flights if f["status"] in DISRUPTED)
            processed_n  = sum(1 for f in flights if f.get("processed"))
            scheduled_n  = sum(1 for f in flights if f["status"] == "scheduled")
            conflict_n   = sum(1 for f in flights if f.get("confidence") == "conflict")

            with ui.row().classes("gap-8 q-mb-md"):
                for label, val, color in [
                    ("Total Flights", len(flights),  "#FFD700"),
                    ("Scheduled",     scheduled_n,   "#4488FF"),
                    ("Disrupted",     disrupted_n,   "#FF4444"),
                    ("Conflict",      conflict_n,    "#FF8800"),
                    ("CS Processed",  processed_n,   "#00CC66"),
                ]:
                    with ui.card().style("background:#1a1a2e; min-width:120px;"):
                        ui.label(str(val)).style(f"color:{color}; font-size:1.8rem; font-family:monospace; font-weight:bold;")
                        ui.label(label).style("color:#888; font-size:0.75rem; font-family:monospace;")

            # Header row
            with ui.row().classes("w-full q-mb-sm").style("flex-wrap:nowrap; gap:0;"):
                for h, w in [("FLIGHT","10%"),("AIRLINE","20%"),("DESTINATION","18%"),
                              ("TIME","10%"),("STATUS","12%"),("NOTE","30%")]:
                    ui.label(h).style(f"color:#FFD700; font-family:monospace; font-size:16px; font-weight:bold; width:{w}; flex-shrink:0;")

            for fl in flights:
                status = fl["status"]
                proc   = fl.get("processed", False)
                conf   = fl.get("confidence", "")

                status_color = "#FF4444" if status in DISRUPTED else "#888888" if status == "departed" else "#4488FF" if status == "active" else "#00CC66"

                effective_conf = conf if conf else "unconfirmed"
                if status in DISRUPTED:
                    if proc:
                        note_html = '<span style="color:#888;font-size:14px;">✓ CS processed</span>'
                    else:
                        note_html = '<span style="color:#FF4444;font-size:14px;">⚠ needs CS</span>'
                    if effective_conf == "high":
                        note_html += ' <span class="badge-high">✅ 2-source</span>'
                    elif effective_conf == "confirmed":
                        note_html += ' <span class="badge-high">✅ 3-source</span>'
                    elif effective_conf == "conflict":
                        note_html += ' <span class="badge-conf">⚡ conflict</span>'
                    elif effective_conf == "conflict_operating":
                        note_html += ' <span class="badge-conf">✈️ likely operating</span>'
                    elif effective_conf == "unconfirmed":
                        note_html += ' <span class="badge-single">⚠ 1-source</span>'
                elif status == "departed":
                    note_html = '<span style="color:#888;font-size:12px;">🛫 departed</span>'
                else:
                    note_html = ""

                sched = fl.get("scheduled", "")
                sched = sched[11:16] if len(sched) >= 16 else sched
                is_conflict = status in DISRUPTED and effective_conf in ("conflict", "conflict_operating", "confirmed")

                # Build conflict detail block and embed toggle link into note_html
                detail_html = ""
                if is_conflict:
                    av_says = status.upper()
                    if effective_conf == "conflict_operating":
                        header_color = "#FF8800"
                        header_text  = "⚡ Conflict — OpenSky confirms flight is OPERATING"
                        opensky_line = f'&nbsp;&nbsp;OpenSky &nbsp;&nbsp;&nbsp;&nbsp;→ <span style="color:#00CC66;">AIRBORNE ✈️ — cancellation likely incorrect</span><br>'
                        footer       = "Passenger contact BLOCKED. Verify with airline directly."
                    elif effective_conf == "confirmed":
                        header_color = "#FFD700"
                        header_text  = "✅ Conflict resolved — OpenSky supports cancellation"
                        opensky_line = f'&nbsp;&nbsp;OpenSky &nbsp;&nbsp;&nbsp;&nbsp;→ <span style="color:#888;">NOT DETECTED 🚫 — transponder silent</span><br>'
                        footer       = "Cancellation confirmed by 3 sources. Safe to contact passengers."
                    else:
                        header_color = "#FF8800"
                        header_text  = "⚡ Data conflict — OpenSky tiebreaker unavailable"
                        opensky_line = f'&nbsp;&nbsp;OpenSky &nbsp;&nbsp;&nbsp;&nbsp;→ <span style="color:#555;">unavailable</span><br>'
                        footer       = "Passenger contact on hold. Refresh to retry OpenSky check."

                    # Show conflict details inline — always visible, no toggle needed
                    note_html += (
                        f' <span style="font-family:monospace;font-size:11px;color:#888;">'
                        f'AV:<span style="color:#FF4444;">{av_says}</span> '
                        f'ADB:<span style="color:#00CC66;">EXPECTED</span> '
                        f'{opensky_line.replace("<br>","").strip()}'
                        f'</span>'
                    )

                with ui.column().classes("w-full").style("gap:0;"):
                    with ui.row().classes("w-full flight-row items-center").style("flex-wrap:nowrap; gap:0;"):
                        ui.label(fl["flight_iata"]).style("color:#fff;font-family:monospace;font-size:16px;width:10%;flex-shrink:0;")
                        ui.label(fl["airline"]).style("color:#ccc;font-family:monospace;font-size:15px;width:20%;flex-shrink:0;")
                        ui.label(get_city(fl["destination"])).style("color:#fff;font-family:monospace;font-size:16px;width:18%;flex-shrink:0;")
                        ui.label(sched).style("color:#ccc;font-family:monospace;font-size:16px;width:10%;flex-shrink:0;")
                        ui.label(status.upper()).style(f"color:{status_color};font-family:monospace;font-size:16px;width:12%;flex-shrink:0;")
                        ui.html(note_html).style("width:30%;flex-shrink:0;font-size:15px;")

    def render_tab2():
        t2_container.clear()
        with t2_container:
            if not state["data_loaded"]:
                ui.html(WAITING); return
            flights    = load_flights()
            passengers = load_passengers()
            affected   = []
            for pnr, p in passengers.items():
                fi = next((f for f in flights if f["flight_iata"] == p["flight_iata"]), None)
                if fi and fi["status"] in DISRUPTED:
                    affected.append({"pnr": pnr, "name": p["passenger_name"],
                                     "flight": p["flight_iata"], "dest": p["destination"],
                                     "status": fi["status"], "processed": fi.get("processed", False),
                                     "airline": fi.get("airline", ""),
                                     "cabin": p.get("cabin_class", "economy"),
                                     "fare": p.get("fare_amount", 0)})
            if not affected:
                ui.html("<p style='color:#00FF00;font-family:monospace;'>✅ No affected passengers.</p>")
                return
            ui.label(f"{len(affected)} passenger(s) on disrupted flights").style("color:#fff;font-family:monospace;")
            ui.separator().style("border-color:#333; margin:8px 0;")
            for p in affected:
                border = "#888" if p["processed"] else "#FF4444"
                proc_t = " &nbsp;<span style='color:#888;font-size:11px;'>✓ CS done</span>" if p["processed"] else ""
                ui.html(f"""
                <div style='background:#1a1a2e;border-left:4px solid {border};padding:12px;margin:6px 0;border-radius:5px;'>
                  <span style='color:#FFD700;font-family:monospace;font-weight:bold;'>{p['pnr']}</span>
                  <span style='color:#fff;font-family:monospace;'> — {p['name']}</span>
                  <span style='color:#aaa;font-family:monospace;font-size:13px;'> | {p['flight']} → {get_city(p['dest'])} | {p['airline']}</span>
                  <span style='color:#FF4444;font-family:monospace;font-size:12px;'> [{p['status'].upper()}]</span>{proc_t}<br>
                  <span style='color:#aaa;font-family:monospace;font-size:12px;'>{p['cabin'].replace('_',' ').title()} — ${p['fare']}</span>
                </div>""")

    def render_tab4():
        t4_container.clear()
        with t4_container:
            if not state["data_loaded"]:
                ui.html(WAITING); return

            flights = load_flights()

            # ── Count by status ───────────────────────────────────────────────
            counts = {}
            for fl in flights:
                s = fl["status"]
                counts[s] = counts.get(s, 0) + 1

            scheduled  = counts.get("scheduled", 0)
            cancelled  = counts.get("cancelled", 0)
            delayed    = counts.get("delayed", 0)
            departed   = counts.get("departed", 0)
            diverted   = counts.get("diverted", 0)
            other      = sum(v for k, v in counts.items()
                             if k not in {"scheduled","cancelled","delayed","departed","diverted"})

            # Conflict = disrupted flights where sources disagree (any conflict variant)
            conflicts  = sum(1 for fl in flights if fl.get("confidence") in ("conflict", "conflict_operating"))

            # ── Count departures by hour ──────────────────────────────────────
            hour_counts = [0] * 24
            for fl in flights:
                sched = fl.get("scheduled", "")
                try:
                    # ISO format: 2026-08-07T12:35:00+00:00
                    h = int(sched[11:13])
                    hour_counts[h] += 1
                except Exception:
                    pass

            # ── Affected passengers ───────────────────────────────────────────
            pnr_data = load_passengers()  # dict: {pnr: passenger_record}
            flight_map = {f["flight_iata"]: f for f in flights}
            pax_pending   = 0
            pax_resolved  = 0
            for pnr, p in pnr_data.items():
                fi = flight_map.get(p.get("flight_iata", ""))
                if fi and fi["status"] in DISRUPTED:
                    if fi.get("processed"):
                        pax_resolved += 1
                    else:
                        pax_pending += 1

            # ── Row 1: two charts side by side ───────────────────────────────
            with ui.row().classes("w-full gap-4"):

                # Donut — status breakdown
                donut_data = []
                if scheduled:  donut_data.append({"value": scheduled,  "name": "Scheduled",  "itemStyle": {"color": "#4488FF"}})
                if cancelled:  donut_data.append({"value": cancelled,  "name": "Cancelled",  "itemStyle": {"color": "#FF4444"}})
                if delayed:    donut_data.append({"value": delayed,    "name": "Delayed",    "itemStyle": {"color": "#FF8800"}})
                if departed:   donut_data.append({"value": departed,   "name": "Departed",   "itemStyle": {"color": "#888888"}})
                if diverted:   donut_data.append({"value": diverted,   "name": "Diverted",   "itemStyle": {"color": "#CC44FF"}})
                if conflicts:  donut_data.append({"value": conflicts,  "name": "Conflict",   "itemStyle": {"color": "#FF8800"}})
                if other:      donut_data.append({"value": other,      "name": "Other",      "itemStyle": {"color": "#44CCAA"}})

                with ui.card().style("background:#1a1a2e; flex:1; min-width:300px;"):
                    ui.label("Flight Status Breakdown").style(
                        "color:#FFD700; font-family:monospace; font-size:16px; font-weight:bold; margin-bottom:8px;"
                    )
                    ui.echart({
                        "backgroundColor": "transparent",
                        "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
                        "legend": {
                            "orient": "vertical", "right": "5%", "top": "center",
                            "textStyle": {"color": "#ccc", "fontFamily": "monospace", "fontSize": 13}
                        },
                        "series": [{
                            "type": "pie",
                            "radius": ["45%", "70%"],
                            "center": ["40%", "50%"],
                            "data": donut_data,
                            "label": {"show": False},
                            "emphasis": {"label": {"show": True, "fontSize": 14, "fontWeight": "bold", "color": "#FFD700"}}
                        }]
                    }).style("height:280px;")

                # Bar — departures by hour
                active_hours   = [h for h in range(24) if hour_counts[h] > 0]
                hour_labels    = [f"{h:02d}:00" for h in active_hours]
                hour_vals      = [hour_counts[h] for h in active_hours]

                with ui.card().style("background:#1a1a2e; flex:1; min-width:300px;"):
                    ui.label("Departures by Hour (UTC)").style(
                        "color:#FFD700; font-family:monospace; font-size:16px; font-weight:bold; margin-bottom:8px;"
                    )
                    ui.echart({
                        "backgroundColor": "transparent",
                        "tooltip": {"trigger": "axis"},
                        "grid": {"left": "8%", "right": "4%", "bottom": "15%", "top": "5%"},
                        "xAxis": {
                            "type": "category",
                            "data": hour_labels,
                            "axisLabel": {"color": "#888", "fontFamily": "monospace", "fontSize": 11, "rotate": 45},
                            "axisLine": {"lineStyle": {"color": "#333"}}
                        },
                        "yAxis": {
                            "type": "value",
                            "axisLabel": {"color": "#888", "fontFamily": "monospace"},
                            "splitLine": {"lineStyle": {"color": "#222"}}
                        },
                        "series": [{
                            "type": "bar",
                            "data": hour_vals,
                            "itemStyle": {"color": "#4488FF", "borderRadius": [3, 3, 0, 0]},
                            "emphasis": {"itemStyle": {"color": "#FFD700"}}
                        }]
                    }).style("height:280px;")

            # ── Row 2: affected passengers ────────────────────────────────────
            ui.separator().style("border-color:#333; margin:16px 0 12px;")
            ui.label("Affected Passengers").style(
                "color:#FFD700; font-family:monospace; font-size:16px; font-weight:bold; margin-bottom:8px;"
            )
            with ui.card().style("background:#1a1a2e; width:100%;"):
                ui.echart({
                    "backgroundColor": "transparent",
                    "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                    "grid": {"left": "4%", "right": "8%", "bottom": "5%", "top": "5%", "containLabel": True},
                    "xAxis": {
                        "type": "value",
                        "minInterval": 1,
                        "axisLabel": {"color": "#888", "fontFamily": "monospace"},
                        "splitLine": {"lineStyle": {"color": "#222"}}
                    },
                    "yAxis": {
                        "type": "category",
                        "data": ["Pending CS", "CS Resolved"],
                        "axisLabel": {"color": "#ccc", "fontFamily": "monospace", "fontSize": 13}
                    },
                    "series": [{
                        "type": "bar",
                        "data": [
                            {"value": pax_pending,  "itemStyle": {"color": "#FF4444"}},
                            {"value": pax_resolved, "itemStyle": {"color": "#00CC66"}}
                        ],
                        "label": {
                            "show": True, "position": "right",
                            "formatter": "{c} pax",
                            "color": "#FFD700", "fontFamily": "monospace", "fontSize": 13
                        },
                        "barMaxWidth": 50
                    }]
                }).style("height:140px;")

    def render_tab5():
        t5_container.clear()
        with t5_container:
            logs = load_logs()

            def clear_log():
                with open("data/disruption_log.json", "w") as f:
                    json.dump([], f)
                render_tab5()

            ui.button("🗑️ CLEAR LOG", on_click=clear_log).style(
                "background:transparent;color:#888;border:1px solid #444;font-family:monospace;font-size:11px;"
            )
            if not logs:
                ui.html("<p style='color:#00FF00;font-family:monospace;margin-top:12px;'>✅ No disruptions logged yet.</p>")
                return
            ui.label(f"{len(logs)} disruption(s) handled").style("color:#fff;font-family:monospace;margin:8px 0;")
            for log in logs:
                res_color = "#00CC66" if log.get("resolution") else "#888"
                res_label = (log.get("resolution") or "pending").upper()
                ui.html(f"""
                <div style='background:#1a1a2e;border-left:4px solid #FF4444;padding:15px;margin:8px 0;border-radius:5px;'>
                  <span style='color:#FF4444;font-family:monospace;font-size:16px;font-weight:bold;'>{log['flight_iata']}</span>
                  <span style='color:#FFD700;font-family:monospace;'> &nbsp;{log.get('airline','')}</span>
                  <span style='color:{res_color};font-family:monospace;font-size:12px;float:right;'>{res_label}</span><br>
                  <span style='color:#fff;font-family:monospace;'>→ {get_city(log.get('destination',''))} &nbsp;|&nbsp; {log['status'].upper()} &nbsp;|&nbsp; {log['affected_count']} passenger(s)</span><br>
                  <span style='color:#888;font-family:monospace;font-size:12px;'>{log['timestamp']}</span>
                </div>""")

    # ── CS Tab (most complex) ─────────────────────────────────────────────────

    def render_tab3():
        t3_container.clear()
        with t3_container:
            if not state["data_loaded"]:
                ui.html(WAITING); return

            flights    = load_flights()
            passengers = load_passengers()

            candidates = []
            for pnr, p in passengers.items():
                fi = next((f for f in flights if f["flight_iata"] == p["flight_iata"]), None)
                if fi and fi["status"] in DISRUPTED:
                    candidates.append({
                        "pnr":        pnr,
                        "name":       p["passenger_name"],
                        "flight":     p["flight_iata"],
                        "origin":     fi.get("origin", "LHR"),
                        "destination": get_city(p["destination"]),
                        "dest_iata":  p["destination"],
                        "status":     fi["status"],
                        "airline":    fi.get("airline", ""),
                        "processed":  fi.get("processed", False),
                        "cabin":      p.get("cabin_class", "economy"),
                        "fare":       p.get("fare_amount", 0),
                    })

            if not candidates:
                ui.html("<p style='color:#00FF00;font-family:monospace;'>✅ No disrupted passengers.</p>")
                return

            names = [f"{'✓ ' if c['processed'] else ''}{c['pnr']} — {c['name']} ({c['flight']} → {c['destination']})"
                     for c in candidates]

            # Controls row
            with ui.row().classes("items-center gap-4 q-mb-md"):
                sel = ui.select(names, value=names[0]).style(
                    "min-width:380px; font-family:monospace; background:#1a1a2e; color:#fff;"
                )

                async def start_call():
                    idx = names.index(sel.value)
                    p   = candidates[idx]

                    # Block CS if flight is in conflict — sources disagree, do not contact
                    flights    = load_flights()
                    flight_map = {f["flight_iata"]: f for f in flights}
                    fl         = flight_map.get(p["flight"], {})
                    confidence = fl.get("confidence", "")
                    if confidence in ("conflict", "conflict_operating"):
                        reason = "sources disagree (AviationStack vs AeroDataBox)" if confidence == "conflict" else "OpenSky confirms flight is AIRBORNE"
                        ui.notify(
                            f"⚡ BLOCKED — {p['flight']} is in conflict: {reason}. Do not contact passengers until resolved.",
                            type="negative", timeout=6000
                        )
                        return

                    state["cs_passenger"] = p
                    state["cs_messages"]  = []
                    state["cs_choice"]    = ""
                    state["cs_handoff"]   = ""

                    prompt = (f"You are calling {p['name']} (PNR: {p['pnr']}). "
                              f"Their flight {p['flight']} to {p['destination']} with {p['airline']} is {p['status']}. "
                              f"Greet them warmly by first name. Inform them of the disruption with genuine empathy. "
                              f"Ask if it is a good time to talk. Under 80 words.")

                    text = await asyncio.to_thread(emma_call, prompt, [], p['airline'])
                    state["cs_messages"].append({"role": "assistant", "content": [{"text": text}]})
                    try:
                        audio = await asyncio.to_thread(synth, text)
                    except:
                        audio = None
                    state["cs_phase"] = "turn1"
                    render_cs_conversation(text, audio)

                def new_call():
                    state["cs_phase"]     = "idle"
                    state["cs_passenger"] = None
                    state["cs_messages"]  = []
                    state["cs_choice"]    = ""
                    render_tab3()

                start_btn = ui.button("📞 Start Call", on_click=start_call).style(
                    "background:#1a1a4e; color:#FFD700; border:1px solid #FFD700; font-family:monospace;"
                )
                if state["cs_phase"] != "idle":
                    start_btn.disable()

                ui.button("🔄 New Call", on_click=new_call).style(
                    "background:transparent; color:#888; border:1px solid #444; font-family:monospace;"
                )

            # Conversation area
            conv_area = ui.column().classes("w-full")

            def render_cs_conversation(new_emma_text=None, new_audio=None):
                conv_area.clear()
                with conv_area:
                    # Replay all bubbles from messages
                    for m in state["cs_messages"]:
                        role = m["role"]
                        txt  = m["content"][0]["text"]
                        cls  = "emma-bubble" if role == "assistant" else "pax-bubble"
                        tag  = "🎧 EMMA" if role == "assistant" else "🧑 PASSENGER"
                        color = "#FFD700" if role == "assistant" else "#888"
                        ui.html(f"""<div class='{cls}'>
                            <span style='color:{color};font-family:monospace;font-size:12px;'>{tag}</span><br><br>
                            <span style='font-size:15px;'>{txt}</span></div>""")

                    # Play audio for the latest Emma turn
                    if new_audio:
                        b64 = audio_b64(new_audio)
                        ui.run_javascript(f"""
                            const a = new Audio('data:audio/mp3;base64,{b64}');
                            a.play().catch(()=>{{}});
                        """)

                    phase = state["cs_phase"]

                    # ── Input area ────────────────────────────────────────────
                    if phase in ("turn1", "turn2", "turn3"):
                        placeholder = {
                            "turn1": "e.g. Yes, what happened to my flight?",
                            "turn2": "e.g. I'd like a refund please",
                            "turn3": "e.g. Thank you, that's all",
                        }[phase]

                        with ui.row().classes("w-full items-center gap-2 q-mt-md"):
                            reply_input = ui.input(placeholder=placeholder).style(
                                "flex:1; font-family:monospace; background:#1a1a2e; color:#fff;"
                            )

                            async def send_reply():
                                reply = reply_input.value.strip()
                                if not reply:
                                    return
                                # Keep text visible while Emma is thinking — clear only once reply arrives
                                state["cs_messages"].append({"role": "user", "content": [{"text": reply}]})
                                await process_reply(reply)
                                try:
                                    reply_input.set_value("")
                                except Exception:
                                    pass  # element may have been removed if tab re-rendered

                            async def use_mic():
                                try:
                                    transcript = await ui.run_javascript("""
                                        return new Promise((resolve) => {
                                            const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
                                            if (!SR) { resolve(''); return; }
                                            const r = new SR();
                                            r.lang = 'en-US';
                                            r.continuous = false;
                                            r.interimResults = false;
                                            r.onresult = (e) => resolve(e.results[0][0].transcript);
                                            r.onerror  = () => resolve('');
                                            r.onend    = () => resolve('');
                                            r.start();
                                        });
                                    """, timeout=20.0)
                                except TimeoutError:
                                    transcript = ''
                                if transcript:
                                    try:
                                        reply_input.set_value(transcript)
                                    except Exception:
                                        pass

                            ui.button("📤 Send", on_click=send_reply).style(
                                "background:#1a1a4e; color:#FFD700; border:1px solid #FFD700; font-family:monospace;"
                            )
                            ui.button("🎙️ Mic", on_click=use_mic).style(
                                "background:#2a1a2e; color:#FF88FF; border:1px solid #884488; font-family:monospace;"
                            ).tooltip("Click, then speak — transcript appears in the text box")

                    elif phase == "approval":
                        with ui.card().style("background:#2a1a1a; border:1px solid #FF4444; padding:16px; margin-top:12px;"):
                            p = state["cs_passenger"]
                            dist = state["cs_distance_km"]
                            ui.html(f"""<b style='color:#FF4444;'>ESCALATION REQUIRED</b><br><br>
                                <span style='color:#fff;'>Passenger: <b>{p['name']}</b></span><br>
                                <span style='color:#fff;'>Flight: <b>{p['flight']}</b> → {p['destination']}</span><br>
                                <span style='color:#fff;'>Route: <b>{dist} km</b></span><br><br>
                                <span style='color:#aaa;font-size:13px;'>{state['cs_handoff']}</span>""")
                            with ui.row().classes("gap-4 q-mt-md"):
                                async def approve_refund():
                                    state["cs_choice"] = "refund"
                                    state["cs_phase"]  = "approved"
                                    await process_approved()
                                async def approve_rebook():
                                    state["cs_choice"] = "rebook"
                                    state["cs_phase"]  = "approved"
                                    await process_approved()
                                async def approve_voucher():
                                    state["cs_choice"] = "voucher"
                                    state["cs_phase"]  = "approved"
                                    await process_approved()
                                ui.button("✅ Approve Refund",  on_click=approve_refund).style("background:#1a2e1a; color:#00CC66; border:1px solid #00CC66;")
                                ui.button("✈️ Rebook",          on_click=approve_rebook).style("background:#1a1a2e; color:#FFD700; border:1px solid #FFD700;")
                                ui.button("🏨 Hotel Voucher",   on_click=approve_voucher).style("background:#2e1a2e; color:#FF88FF; border:1px solid #884488;")

                    elif phase == "human_handoff":
                        with ui.card().style("background:#2a1a1a; border:1px solid #FF4444; padding:16px; margin-top:12px;"):
                            p = state["cs_passenger"]
                            ui.html(f"""<b style='color:#FF4444;'>ESCALATION — HUMAN AGENT REQUESTED</b><br><br>
                                <span style='color:#fff;'>Passenger: <b>{p['name']}</b></span><br>
                                <span style='color:#fff;'>Flight: <b>{p['flight']}</b> → {p['destination']}</span><br>
                                <span style='color:#fff;'>Requested: <b>{state['cs_choice'] or 'not yet specified'}</b></span><br><br>
                                <span style='color:#aaa;font-size:13px;'>{state['cs_handoff']}</span>""")
                            with ui.row().classes("gap-4 q-mt-md"):
                                async def human_refund():
                                    state["cs_choice"] = "refund"
                                    state["cs_phase"]  = "approved"
                                    await process_approved()
                                async def human_rebook():
                                    state["cs_choice"] = "rebook"
                                    state["cs_phase"]  = "approved"
                                    await process_approved()
                                async def human_voucher():
                                    state["cs_choice"] = "voucher"
                                    state["cs_phase"]  = "approved"
                                    await process_approved()
                                ui.button("✅ Process Refund",  on_click=human_refund).style("background:#1a2e1a; color:#00CC66; border:1px solid #00CC66;")
                                ui.button("✈️ Process Rebook",  on_click=human_rebook).style("background:#1a1a2e; color:#FFD700; border:1px solid #FFD700;")
                                ui.button("🏨 Process Voucher", on_click=human_voucher).style("background:#2e1a2e; color:#FF88FF; border:1px solid #884488;")

                    elif phase == "complete":
                        ui.html("<p style='color:#888;font-family:monospace;margin-top:15px;'>— Call concluded. Flight marked as processed. —</p>")

            async def process_reply(reply):
                p     = state["cs_passenger"]
                phase = state["cs_phase"]
                hist  = state["cs_messages"]

                if wants_human(reply):
                    state["cs_handoff"] = "Passenger explicitly requested a human agent."
                    prompt = (f"The passenger asked to speak with a human agent. Acknowledge this "
                              f"professionally and let {p['name'].split()[0]} know you're connecting "
                              f"them now. Under 40 words.")
                    text  = await asyncio.to_thread(emma_call, prompt, hist, p['airline'])
                    try:   audio = await asyncio.to_thread(synth, text)
                    except: audio = None
                    state["cs_messages"].append({"role": "assistant", "content": [{"text": text}]})
                    state["cs_phase"] = "human_handoff"
                    render_cs_conversation(text, audio)
                    return

                # ── Turn 1: first response ─────────────────────────────────
                if phase == "turn1":
                    prompt = (f'The passenger said: "{reply}"\n'
                              "Offer 3 clear numbered options (no markdown, no bold):\n"
                              "1. Rebook on the next available flight at no extra cost\n"
                              "2. Full refund to original payment method within 5-7 business days\n"
                              "3. Hotel accommodation tonight and rebook on tomorrow's flight, all covered\n"
                              "Ask which option they prefer. Under 100 words.")
                    text  = await asyncio.to_thread(emma_call, prompt, hist, p['airline'])
                    try:   audio = await asyncio.to_thread(synth, text)
                    except: audio = None
                    state["cs_messages"].append({"role": "assistant", "content": [{"text": text}]})
                    state["cs_phase"] = "turn2"
                    render_cs_conversation(text, audio)

                # ── Turn 2: classify choice ────────────────────────────────
                elif phase == "turn2":
                    choice = await asyncio.to_thread(classify, reply, hist)
                    state["cs_choice"] = choice

                    if choice == "refund":
                        long_haul, dist = check_long_haul(p["origin"], p["dest_iata"])
                        high_val        = p.get("fare", 0) >= HIGH_VALUE_USD
                        state["cs_distance_km"] = dist

                        if long_haul or high_val:
                            reasons = []
                            if long_haul: reasons.append(f"long-haul route ({dist} km)")
                            if high_val:  reasons.append(f"high-value fare (${p.get('fare',0)})")
                            state["cs_handoff"] = "Requires approval: " + " and ".join(reasons) + "."
                            prompt = (f"The passenger requested a refund. Requires supervisor approval ({' and '.join(reasons)}). "
                                      f"Tell {p['name'].split()[0]} warmly you are escalating — standard procedure. Under 60 words.")
                            text  = await asyncio.to_thread(emma_call, prompt, hist, p['airline'])
                            try:   audio = await asyncio.to_thread(synth, text)
                            except: audio = None
                            state["cs_messages"].append({"role": "assistant", "content": [{"text": text}]})
                            state["cs_phase"] = "approval"
                            render_cs_conversation(text, audio)
                        else:
                            prompt = (f"Confirm the refund for {p['name'].split()[0]}. "
                                      "5-7 business days to original payment method. Warm and brief. Do NOT say goodbye yet. Under 60 words.")
                            text  = await asyncio.to_thread(emma_call, prompt, hist, p['airline'])
                            try:   audio = await asyncio.to_thread(synth, text)
                            except: audio = None
                            state["cs_messages"].append({"role": "assistant", "content": [{"text": text}]})
                            state["cs_phase"] = "turn3"
                            render_cs_conversation(text, audio)

                    elif choice == "rebook":
                        prompt = (f"Confirm rebooking to {p['destination']} for {p['name'].split()[0]}. "
                                  "Next available flight, confirmation email shortly. Do NOT say goodbye yet. Under 60 words.")
                        text  = await asyncio.to_thread(emma_call, prompt, hist, p['airline'])
                        try:   audio = await asyncio.to_thread(synth, text)
                        except: audio = None
                        state["cs_messages"].append({"role": "assistant", "content": [{"text": text}]})
                        state["cs_phase"] = "turn3"
                        render_cs_conversation(text, audio)

                    elif choice == "voucher":
                        prompt = (f"Confirm hotel voucher tonight and rebooking tomorrow to {p['destination']} for {p['name'].split()[0]}. "
                                  "All costs covered by the airline. Do NOT say goodbye yet. Under 60 words.")
                        text  = await asyncio.to_thread(emma_call, prompt, hist, p['airline'])
                        try:   audio = await asyncio.to_thread(synth, text)
                        except: audio = None
                        state["cs_messages"].append({"role": "assistant", "content": [{"text": text}]})
                        state["cs_phase"] = "turn3"
                        render_cs_conversation(text, audio)

                    else:
                        text  = "Sorry — could you clarify? Would you like a refund, to be rebooked on the next flight, or a hotel voucher tonight with rebooking tomorrow?"
                        try:   audio = await asyncio.to_thread(synth, text)
                        except: audio = None
                        state["cs_messages"].append({"role": "assistant", "content": [{"text": text}]})
                        render_cs_conversation(text, audio)

                # ── Turn 3: closing ────────────────────────────────────────
                elif phase == "turn3":
                    prompt = (f"Close the call warmly with {p['name'].split()[0]}. "
                              "Apologise once more for the inconvenience, wish them well, say a warm goodbye. Under 50 words.")
                    text  = await asyncio.to_thread(emma_call, prompt, hist, p['airline'])
                    try:   audio = await asyncio.to_thread(synth, text)
                    except: audio = None
                    state["cs_messages"].append({"role": "assistant", "content": [{"text": text}]})

                    mark_processed(p["flight"])
                    write_log(p["flight"], p["status"], p["airline"], p["dest_iata"], state["cs_choice"])
                    state["cs_phase"] = "complete"
                    render_cs_conversation(text, audio)
                    refresh_tab1_tab2()
                    render_tab4()
                    render_tab5()

            async def process_approved():
                p      = state["cs_passenger"]
                choice = state["cs_choice"]
                hist   = state["cs_messages"]

                if choice == "refund":
                    prompt = f"The supervisor approved the refund for {p['name'].split()[0]}. Confirm warmly — refund approved, 5-7 business days. Do NOT say goodbye yet. Under 60 words."
                elif choice == "rebook":
                    prompt = f"The supervisor rebooked {p['name'].split()[0]} on the next flight to {p['destination']}. Confirm warmly, mention confirmation email. Do NOT say goodbye yet. Under 60 words."
                else:
                    prompt = f"Hotel voucher and rebooking tomorrow to {p['destination']} for {p['name'].split()[0]} all arranged. Confirm warmly, all costs covered. Do NOT say goodbye yet. Under 60 words."

                text  = await asyncio.to_thread(emma_call, prompt, hist, p['airline'])
                try:   audio = await asyncio.to_thread(synth, text)
                except: audio = None
                state["cs_messages"].append({"role": "assistant", "content": [{"text": text}]})
                state["cs_phase"] = "turn3"
                render_cs_conversation(text, audio)

            # Initial render if already in a call
            if state["cs_phase"] != "idle" and state["cs_messages"]:
                render_cs_conversation()

    def refresh_all():
        render_tab1()
        render_tab2()
        render_tab3()
        render_tab4()
        render_tab5()

    def refresh_tab1_tab2():
        render_tab1()
        render_tab2()

    # Initial render
    if state["data_loaded"]:
        refresh_all()
    else:
        render_tab1()
        render_tab2()
        render_tab3()
        render_tab4()
        render_tab5()


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ui.run(
        title="FlightSense",
        port=8080,
        dark=True,
        reload=False,
        favicon="✈️",
    )
