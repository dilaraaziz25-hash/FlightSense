import streamlit as st
import boto3
import json
import os
import io
import math
import subprocess
import sys
from dotenv import load_dotenv
import airportsdata

load_dotenv()

# ── AWS clients ───────────────────────────────────────────────────────────────

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv("AWS_REGION", "us-east-1")
)
polly = boto3.client(
    service_name='polly',
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# ── Airport helpers ───────────────────────────────────────────────────────────

airports   = airportsdata.load('IATA')
city_cache = {}

def get_city_name(iata_code):
    if iata_code in city_cache:
        return city_cache[iata_code]
    try:
        airport = airports.get(iata_code)
        city = airport['city'].title() if airport else iata_code
        city_cache[iata_code] = city
        return city
    except:
        return iata_code

def get_airport_coords(iata_code):
    airport = airports.get(iata_code)
    return [airport['lat'], airport['lon']] if airport else None

# ── Distance helpers ──────────────────────────────────────────────────────────

LONG_HAUL_KM = 3000

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def check_long_haul(origin_iata, dest_iata):
    o = airports.get(origin_iata)
    d = airports.get(dest_iata)
    if not o or not d:
        return False, 0
    dist = haversine_km(o['lat'], o['lon'], d['lat'], d['lon'])
    return dist > LONG_HAUL_KM, round(dist)

# ── Polly ─────────────────────────────────────────────────────────────────────

def text_to_speech_polly(text):
    response = polly.synthesize_speech(
        Text=text, OutputFormat='mp3', VoiceId='Amy', Engine='neural'
    )
    return response['AudioStream'].read()

# ── Data helpers ──────────────────────────────────────────────────────────────

def load_data():
    try:
        with open('data/flights.json', 'r') as f:
            flights = json.load(f)
        for f_ in flights:
            s = f_.get('status', '').lower()
            f_['status'] = 'cancelled' if s == 'canceled' else s
    except:
        flights = []
    try:
        with open('data/pnr_database.json', 'r') as f:
            passengers = json.load(f)
    except:
        passengers = {}
    try:
        with open('data/disruption_log.json', 'r') as f:
            logs = json.load(f)
    except:
        logs = []
    return flights, passengers, logs

def mark_flight_processed(flight_iata):
    try:
        with open('data/flights.json', 'r') as f:
            flights = json.load(f)
        for flight in flights:
            if flight['flight_iata'] == flight_iata:
                flight['processed'] = True
                break
        with open('data/flights.json', 'w') as f:
            json.dump(flights, f, indent=2)
    except Exception as e:
        st.error(f"Could not mark flight processed: {e}")

def log_disruption_event(flight_iata, status, affected_count, airline, destination, resolution):
    import datetime
    entry = {
        'flight_iata':    flight_iata,
        'airline':        airline,
        'destination':    destination,
        'status':         status,
        'affected_count': affected_count,
        'resolution':     resolution,
        'timestamp':      str(datetime.datetime.now())
    }
    try:
        with open('data/disruption_log.json', 'r') as f:
            logs = json.load(f)
    except:
        logs = []
    logs.append(entry)
    with open('data/disruption_log.json', 'w') as f:
        json.dump(logs, f, indent=2)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="FlightSense", page_icon="✈️", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #000000; }
[data-testid="stMarkdownContainer"] p { color: #FFFFFF; font-family: monospace; font-size: 16px; }
.stButton button {
    background-color: transparent; color: #CCCCCC;
    border: 1px solid #444444; font-family: monospace; font-size: 10px; cursor: pointer;
}
.stButton button:hover { background-color: #2a2a4e; color: #FFFFFF; border: 1px solid #666666; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align:left; color:#FFD700; background-color:#000000;
    padding:20px; font-family:monospace; letter-spacing:3px;'>
    ✈️ FLIGHTSENSE<br>
    <span style='font-size:0.6em; color:#FFD700;'>FLIGHT DISRUPTION INTELLIGENCE</span>
    </h1>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# ── RUN AGENT button ──────────────────────────────────────────────────────────

col_btn1, col_btn2 = st.columns([1, 4])

with col_btn1:
    # First run: full fetch + generate PNRs. Subsequent: compare & merge only.
    flights_exist = os.path.exists('data/flights.json') and os.path.getsize('data/flights.json') > 10

    if not flights_exist:
        btn_label   = "▶ RUN AGENT"
        btn_script  = "pnr_generator_demo_bedrock.py"
        btn_spinner = "Fetching live flights and generating passenger data..."
    else:
        btn_label   = "🔄 REFRESH DATA"
        btn_script  = "fetch_live_status.py"
        btn_spinner = "Refreshing flight statuses and detecting new disruptions..."

    if st.button(btn_label):
        with st.spinner(btn_spinner):
            result = subprocess.run(
                [sys.executable, btn_script],
                capture_output=True, text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            if result.returncode == 0:
                if not flights_exist:
                    # First run — clear log for fresh session
                    with open('data/disruption_log.json', 'w') as f:
                        json.dump([], f)
                st.session_state.data_loaded = True
                # Reset CS if a new call cycle is starting
                for key in ['cs_phase','cs_bubbles','cs_messages','cs_passenger','cs_choice','cs_approval','cs_distance_km','cs_handoff_reason']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
            else:
                st.error(f"Error:\n{result.stderr}")

# ── Load data ─────────────────────────────────────────────────────────────────

flights, passengers, logs = load_data()

DISRUPTED_STATUSES = ['cancelled', 'delayed', 'diverted', 'incident']

WAITING_MSG = "<p style='color:#444444; font-family:monospace; font-size:14px;'>⏳ Awaiting data — click RUN AGENT to fetch live flights.</p>"

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "✈️ FLIGHT MONITOR",
    "🚨 AFFECTED PASSENGERS",
    "📞 CUSTOMER SERVICE",
    "🗺️ LIVE MAP",
    "📋 DISRUPTION LOG"
])

# ── Tab 1: Flight Monitor ─────────────────────────────────────────────────────

with tab1:
    st.markdown("<h4 style='color:#FFD700; font-family:monospace;'>✈️ FLIGHT MONITOR</h4>", unsafe_allow_html=True)

    if not st.session_state.data_loaded or not flights:
        st.markdown(WAITING_MSG, unsafe_allow_html=True)
    else:
        total       = len(flights)
        disrupted_n = sum(1 for f in flights if f['status'] in DISRUPTED_STATUSES)
        processed_n = sum(1 for f in flights if f.get('processed', False))
        scheduled_n = sum(1 for f in flights if f['status'] == 'scheduled')

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Flights",  total)
        c2.metric("Scheduled",      scheduled_n)
        c3.metric("Disrupted",      disrupted_n)
        c4.metric("Processed by CS", processed_n)

        st.markdown("<br>", unsafe_allow_html=True)

        for col, label in zip(
            st.columns([1.5, 2, 2, 2, 1.5, 1.5]),
            ["FLIGHT", "AIRLINE", "DESTINATION", "SCHEDULED", "STATUS", "NOTE"]
        ):
            col.markdown(f"<span style='color:#FFD700;font-family:monospace;font-size:13px;'>{label}</span>", unsafe_allow_html=True)

        for flight in flights:
            status = flight['status'].lower()
            proc   = flight.get('processed', False)

            if status in DISRUPTED_STATUSES:
                status_color = '#FF4444'
                confidence = flight.get('confidence', '')
                if confidence == 'high':
                    conf_badge = ' <span style="color:#00CC66;font-size:11px;">✅ 2-source</span>'
                elif confidence == 'conflict':
                    conf_badge = ' <span style="color:#FF8800;font-size:11px;">⚡ conflict</span>'
                elif confidence == 'unconfirmed':
                    conf_badge = ' <span style="color:#888888;font-size:11px;">⚠ 1-source</span>'
                else:
                    conf_badge = ''
                note = f'<span style="color:#888888;font-size:12px;">✓ CS processed</span>{conf_badge}' if proc else f'<span style="color:#FF4444;font-size:12px;">⚠ needs CS</span>{conf_badge}'
            elif status == 'departed':
                status_color = '#888888'
                note = '<span style="color:#888888;font-size:12px;">🛫 departed</span>'
            else:
                status_color = '#00CC66'
                note = ''

            sched = flight.get('scheduled', '')
            sched = sched[11:16] if len(sched) >= 16 else sched

            cols = st.columns([1.5, 2, 2, 2, 1.5, 1.5])
            cols[0].markdown(f"<span style='color:#FFFFFF;font-family:monospace;'>{flight['flight_iata']}</span>", unsafe_allow_html=True)
            cols[1].markdown(f"<span style='color:#CCCCCC;font-family:monospace;font-size:13px;'>{flight['airline']}</span>", unsafe_allow_html=True)
            cols[2].markdown(f"<span style='color:#FFFFFF;font-family:monospace;'>{get_city_name(flight['destination'])}</span>", unsafe_allow_html=True)
            cols[3].markdown(f"<span style='color:#CCCCCC;font-family:monospace;'>{sched}</span>", unsafe_allow_html=True)
            cols[4].markdown(f"<span style='color:{status_color};font-family:monospace;'>{status.upper()}</span>", unsafe_allow_html=True)
            cols[5].markdown(f"<span style='font-family:monospace;'>{note}</span>", unsafe_allow_html=True)

# ── Tab 2: Affected Passengers ────────────────────────────────────────────────

with tab2:
    st.markdown("<h4 style='color:#FFD700; font-family:monospace;'>🚨 AFFECTED PASSENGERS</h4>", unsafe_allow_html=True)

    if not st.session_state.data_loaded or not flights:
        st.markdown(WAITING_MSG, unsafe_allow_html=True)
    else:
        affected = []
        for pnr, p in passengers.items():
            fi = next((f for f in flights if f['flight_iata'] == p['flight_iata']), None)
            if fi and fi['status'] in DISRUPTED_STATUSES:
                affected.append({
                    'pnr':       pnr,
                    'name':      p['passenger_name'],
                    'flight':    p['flight_iata'],
                    'dest':      p['destination'],
                    'email':     p['contact_email'],
                    'status':    fi['status'],
                    'processed': fi.get('processed', False),
                    'airline':   fi.get('airline', ''),
                    'cabin':     p.get('cabin_class', 'economy'),
                    'fare':      p.get('fare_amount', 0)
                })

        if not affected:
            st.markdown("<p style='color:#00FF00;font-family:monospace;'>✅ No affected passengers.</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='color:#FFFFFF;font-family:monospace;'>{len(affected)} passenger(s) on disrupted flights</p>", unsafe_allow_html=True)
            for p in affected:
                border     = '#888888' if p['processed'] else '#FF4444'
                proc_label = ' &nbsp;<span style="color:#888888;font-size:11px;">✓ CS done</span>' if p['processed'] else ''
                st.markdown(f"""
                <div style='background-color:#1a1a2e;border-left:4px solid {border};
                padding:12px;margin:6px 0;border-radius:5px;'>
                <span style='color:#FFD700;font-family:monospace;font-weight:bold;'>{p['pnr']}</span>
                <span style='color:#FFFFFF;font-family:monospace;'> — {p['name']}</span>
                <span style='color:#AAAAAA;font-family:monospace;font-size:13px;'> | {p['flight']} → {get_city_name(p['dest'])} | {p['airline']}</span>
                <span style='color:#FF4444;font-family:monospace;font-size:12px;'> [{p['status'].upper()}]</span>{proc_label}<br>
                <span style='color:#AAAAAA;font-family:monospace;font-size:12px;'>{p['cabin'].replace('_',' ').title()} — ${p['fare']}</span>
                </div>
                """, unsafe_allow_html=True)

# ── Tab 3: Customer Service ───────────────────────────────────────────────────

CS_SYSTEM = """You are Emma, a customer service agent at Turkish Airlines handling flight disruptions.
Tone: professional, warm, efficient — the way an experienced airline agent speaks, not a script.
Use the passenger's first name naturally, not in every line. Use city names, not airport codes.
No markdown formatting. Keep responses concise and focused.
Do not invent specific flight times, flight numbers, or other concrete details that were not
already provided to you in this conversation or in your instructions — if you don't have a
specific detail, speak in general terms (e.g. "the next available flight") rather than making one up.
If a passenger asks to speak with a human agent or a supervisor, acknowledge the request professionally
and let them know you're connecting them — do not argue, deny, or try to redirect them back to yourself."""

HIGH_VALUE_USD = 1000   # refunds at or above this need approval regardless of distance

def wants_human(text):
    t = text.lower()
    return any(w in t for w in [
        'human', 'real person', 'real agent', 'speak to someone',
        'supervisor', 'manager', 'not a bot', 'not a robot', 'talk to a person'
    ])

def emma_respond(prompt, history):
    msgs = history + [{"role": "user", "content": [{"text": prompt}]}]
    r = bedrock.converse(
        modelId=MODEL_ID,
        system=[{"text": CS_SYSTEM}],
        messages=msgs,
        inferenceConfig={"maxTokens": 300}
    )
    text = r['output']['message']['content'][0]['text']
    try:
        audio = text_to_speech_polly(text)
    except:
        audio = None
    return text, audio

def classify_choice(reply, history):
    """Classify passenger intent using the LLM with conversation context — not just keyword matching,
    so replies like 'I need to be in Moscow asap' or 'yes, that's fine' resolve correctly."""
    context_lines = []
    for m in history[-6:]:
        speaker = "Emma" if m.get("role") == "assistant" else "Passenger"
        txt = m.get("content", [{}])[0].get("text", "")
        context_lines.append(f"{speaker}: {txt}")
    context_text = "\n".join(context_lines)

    prompt = f"""Conversation so far:
{context_text}
Passenger's latest reply: "{reply}"

Classify what the passenger wants, based on the full conversation, not just the latest reply alone.
Reply with exactly one word:
refund - wants their money back
rebook - wants to be rebooked on another flight (including agreeing to a rebooking Emma already proposed)
voucher - wants the hotel-tonight-plus-rebook-tomorrow option
unclear - their intent genuinely cannot be determined yet"""

    r = bedrock.converse(
        modelId=MODEL_ID,
        system=[{"text": "You are an intent classifier. Reply with exactly one word: refund, rebook, voucher, or unclear. No punctuation, no explanation, nothing else."}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 10}
    )
    result = r['output']['message']['content'][0]['text'].strip().lower()
    return result if result in ['refund', 'rebook', 'voucher'] else 'unknown'

def display_bubble(role, text, audio=None):
    if role == "emma":
        st.markdown(f"""
        <div style='background-color:#1a1a2e;border-left:4px solid #FFD700;
        padding:15px;margin:8px 0;border-radius:5px;'>
        <span style='color:#FFD700;font-family:monospace;font-size:12px;'>🎧 EMMA</span><br><br>
        <span style='color:#FFFFFF;font-size:15px;'>{text}</span>
        </div>""", unsafe_allow_html=True)
        if audio:
            st.audio(io.BytesIO(audio), format='audio/mp3')
    else:
        st.markdown(f"""
        <div style='background-color:#2a2a2a;border-left:4px solid #AAAAAA;
        padding:15px;margin:8px 0;border-radius:5px;'>
        <span style='color:#AAAAAA;font-family:monospace;font-size:12px;'>🧑 PASSENGER</span><br><br>
        <span style='color:#FFFFFF;font-size:15px;'>{text}</span>
        </div>""", unsafe_allow_html=True)

@st.dialog("🧑‍💼 Escalation — Human Agent Required")
def handoff_dialog():
    p      = st.session_state.cs_passenger
    reason = st.session_state.cs_handoff_reason or "Passenger explicitly requested a human agent."
    st.markdown(f"""
    <div style='background-color:#2a1a1a;border:1px solid #FF4444;padding:15px;border-radius:5px;'>
    <b style='color:#FF4444;'>ESCALATION</b><br><br>
    <span style='color:#FFFFFF;'>Passenger: <b>{p['name']}</b></span><br>
    <span style='color:#FFFFFF;'>Flight: <b>{p['flight']}</b> → {p['destination']}</span><br>
    <span style='color:#FFFFFF;'>Requested: <b>{st.session_state.cs_choice or 'not yet specified'}</b></span><br><br>
    <span style='color:#AAAAAA;font-size:13px;'>{reason}</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Process Refund", use_container_width=True):
            st.session_state.cs_choice = 'refund'
            st.session_state.cs_phase  = 'approved'
            st.rerun()
    with col2:
        if st.button("✈️ Process Rebook", use_container_width=True):
            st.session_state.cs_choice = 'rebook'
            st.session_state.cs_phase  = 'approved'
            st.rerun()
    with col3:
        if st.button("🏨 Process Voucher", use_container_width=True):
            st.session_state.cs_choice = 'voucher'
            st.session_state.cs_phase  = 'approved'
            st.rerun()

with tab3:
    st.markdown("<h4 style='color:#FFD700;font-family:monospace;'>📞 CUSTOMER SERVICE</h4>", unsafe_allow_html=True)

    if not st.session_state.data_loaded or not flights:
        st.markdown(WAITING_MSG, unsafe_allow_html=True)
    else:
        # Init session state
        for key, default in [
            ('cs_phase',       'idle'),
            ('cs_bubbles',     []),
            ('cs_messages',    []),
            ('cs_passenger',   None),
            ('cs_choice',      ''),
            ('cs_approval',    None),
            ('cs_distance_km', 0),
            ('cs_handoff_reason', ''),
        ]:
            if key not in st.session_state:
                st.session_state[key] = default

        # Disrupted passengers for dropdown
        cs_candidates = []
        for pnr, p in passengers.items():
            fi = next((f for f in flights if f['flight_iata'] == p['flight_iata']), None)
            if fi and fi['status'] in DISRUPTED_STATUSES:
                cs_candidates.append({
                    'pnr':        pnr,
                    'name':       p['passenger_name'],
                    'flight':     p['flight_iata'],
                    'origin':     fi.get('origin', 'IST'),
                    'destination': get_city_name(p['destination']),
                    'dest_iata':  p['destination'],
                    'status':     fi['status'],
                    'airline':    fi.get('airline', ''),
                    'processed':  fi.get('processed', False),
                    'cabin_class': p.get('cabin_class', 'economy'),
                    'fare_amount': p.get('fare_amount', 0)
                })

        if not cs_candidates:
            st.markdown("<p style='color:#00FF00;font-family:monospace;'>✅ No disrupted passengers at this time.</p>", unsafe_allow_html=True)
        else:
            names    = [f"{'✓ ' if p['processed'] else ''}{p['pnr']} — {p['name']} ({p['flight']} → {p['destination']})" for p in cs_candidates]
            selected = st.selectbox("Select passenger:", names, key="cs_select")

            col1, col2 = st.columns([1, 1])
            with col2:
                if st.button("🔄 New Call"):
                    for key in ['cs_phase','cs_bubbles','cs_messages','cs_passenger','cs_choice','cs_approval','cs_distance_km','cs_handoff_reason']:
                        st.session_state[key] = [] if key in ['cs_bubbles','cs_messages'] else None if key in ['cs_passenger','cs_approval'] else 0 if key == 'cs_distance_km' else 'idle' if key == 'cs_phase' else ''
                    st.rerun()

            with col1:
                start = st.button("📞 Start Call", disabled=(st.session_state.cs_phase != 'idle'))

            # ── PHASE: START CALL ──────────────────────────────────────────────
            if start and st.session_state.cs_phase == 'idle':
                p = cs_candidates[names.index(selected)]
                st.session_state.cs_passenger = p

                prompt = f"""You are calling {p['name']} (PNR: {p['pnr']}).
Their flight {p['flight']} to {p['destination']} with {p['airline']} is {p['status']}.
Greet them warmly by first name. Inform them of the disruption with genuine empathy.
Tell them you are here to help. Ask if it is a good time to talk. Under 80 words."""

                text, audio = emma_respond(prompt, [])
                st.session_state.cs_bubbles  = [("emma", text, audio)]
                st.session_state.cs_messages = [{"role": "assistant", "content": [{"text": text}]}]
                st.session_state.cs_phase    = 'passenger_turn_1'
                st.rerun()

            # ── Display conversation ───────────────────────────────────────────
            for role, text, audio in st.session_state.cs_bubbles:
                display_bubble(role, text, audio)

            phase = st.session_state.cs_phase

            # ── PHASE: PASSENGER TURN 1 ────────────────────────────────────────
            if phase == 'passenger_turn_1':
                reply = st.text_input("✍️ Your reply:", key="p_reply_1",
                                      placeholder="e.g. Yes, what happened to my flight?")
                if st.button("📤 Send", key="send_1") and reply.strip():
                    st.session_state.cs_bubbles.append(("passenger", reply, None))
                    history = st.session_state.cs_messages + [{"role": "user", "content": [{"text": reply}]}]
                    p = st.session_state.cs_passenger

                    if wants_human(reply):
                        prompt = f"""The passenger asked to speak with a human agent.
Acknowledge this professionally and let {p['name'].split()[0]} know you're connecting them now. Under 40 words."""
                        text, audio = emma_respond(prompt, history)
                        history.append({"role": "assistant", "content": [{"text": text}]})
                        st.session_state.cs_bubbles.append(("emma", text, audio))
                        st.session_state.cs_messages = history
                        st.session_state.cs_handoff_reason = "Passenger explicitly requested a human agent."
                        st.session_state.cs_phase = 'awaiting_approval'
                        st.rerun()
                    else:
                        prompt = f"""The passenger said: "{reply}"
Offer 3 clear numbered options (no markdown, no bold):
1. Rebook on the next available flight at no extra cost
2. Full refund to original payment method within 5-7 business days
3. Hotel accommodation tonight and rebook on tomorrow's flight, all covered
Ask which option they prefer. Under 100 words."""

                        text, audio = emma_respond(prompt, history)
                        history.append({"role": "assistant", "content": [{"text": text}]})
                        st.session_state.cs_bubbles.append(("emma", text, audio))
                        st.session_state.cs_messages = history
                        st.session_state.cs_phase    = 'passenger_turn_2'
                        st.rerun()

            # ── PHASE: PASSENGER TURN 2 — choose option ────────────────────────
            elif phase == 'passenger_turn_2':
                reply = st.text_input("✍️ Your choice:", key="p_reply_2",
                                      placeholder="e.g. I would like a refund please")
                if st.button("📤 Send", key="send_2") and reply.strip():
                    st.session_state.cs_bubbles.append(("passenger", reply, None))
                    history = st.session_state.cs_messages + [{"role": "user", "content": [{"text": reply}]}]
                    p = st.session_state.cs_passenger

                    if wants_human(reply):
                        prompt = f"""The passenger asked to speak with a human agent.
Acknowledge this professionally and let {p['name'].split()[0]} know you're connecting them now. Under 40 words."""
                        text, audio = emma_respond(prompt, history)
                        history.append({"role": "assistant", "content": [{"text": text}]})
                        st.session_state.cs_bubbles.append(("emma", text, audio))
                        st.session_state.cs_messages = history
                        st.session_state.cs_handoff_reason = "Passenger explicitly requested a human agent."
                        st.session_state.cs_phase = 'awaiting_approval'
                        st.rerun()

                    choice = classify_choice(reply, history)
                    st.session_state.cs_choice = choice

                    if choice == 'refund':
                        p = st.session_state.cs_passenger
                        long_haul, dist = check_long_haul(p['origin'], p['dest_iata'])
                        fare = p.get('fare_amount', 0)
                        high_value = fare >= HIGH_VALUE_USD
                        st.session_state.cs_distance_km = dist
                        if long_haul or high_value:
                            reasons = []
                            if long_haul:  reasons.append(f"long-haul route ({dist} km)")
                            if high_value: reasons.append(f"high-value fare (${fare})")
                            reason_text = f"Requires approval: {' and '.join(reasons)}."
                            st.session_state.cs_handoff_reason = reason_text

                            prompt = f"""The passenger requested a refund. This requires supervisor approval per policy ({' and '.join(reasons)}).
Tell {p['name'].split()[0]} warmly that you need to get approval — standard procedure.
Assure them you are escalating now. Under 60 words."""
                            text, audio = emma_respond(prompt, history)
                            history.append({"role": "assistant", "content": [{"text": text}]})
                            st.session_state.cs_bubbles.append(("emma", text, audio))
                            st.session_state.cs_messages = history
                            st.session_state.cs_phase    = 'awaiting_approval'
                            st.rerun()
                        else:
                            prompt = f"""Confirm the refund for {st.session_state.cs_passenger['name'].split()[0]}.
5-7 business days to original payment method. Warm and brief. Do NOT say goodbye yet. Under 60 words."""
                            text, audio = emma_respond(prompt, history)
                            history.append({"role": "assistant", "content": [{"text": text}]})
                            st.session_state.cs_bubbles.append(("emma", text, audio))
                            st.session_state.cs_messages = history
                            st.session_state.cs_phase    = 'passenger_turn_3'
                            st.rerun()

                    elif choice == 'rebook':
                        p = st.session_state.cs_passenger
                        prompt = f"""Confirm rebooking to {p['destination']} for {p['name'].split()[0]}.
Next available flight, confirmation email shortly. Do NOT say goodbye yet. Under 60 words."""
                        text, audio = emma_respond(prompt, history)
                        history.append({"role": "assistant", "content": [{"text": text}]})
                        st.session_state.cs_bubbles.append(("emma", text, audio))
                        st.session_state.cs_messages = history
                        st.session_state.cs_phase    = 'passenger_turn_3'
                        st.rerun()

                    elif choice == 'voucher':
                        p = st.session_state.cs_passenger
                        prompt = f"""Confirm hotel voucher tonight and rebooking tomorrow to {p['destination']} for {p['name'].split()[0]}.
All costs covered by the airline. Do NOT say goodbye yet. Under 60 words."""
                        text, audio = emma_respond(prompt, history)
                        history.append({"role": "assistant", "content": [{"text": text}]})
                        st.session_state.cs_bubbles.append(("emma", text, audio))
                        st.session_state.cs_messages = history
                        st.session_state.cs_phase    = 'passenger_turn_3'
                        st.rerun()

                    else:
                        text = "Sorry, just to make sure I get this right — would you like a refund, to be rebooked on the next flight, or a hotel voucher tonight with rebooking tomorrow?"
                        try:
                            audio = text_to_speech_polly(text)
                        except:
                            audio = None
                        history.append({"role": "assistant", "content": [{"text": text}]})
                        st.session_state.cs_bubbles.append(("emma", text, audio))
                        st.session_state.cs_messages = history
                        st.rerun()

            # ── PHASE: AWAITING HUMAN AGENT ──────────────────────────────────────
            elif phase == 'awaiting_approval':
                st.warning(f"⚠️ {st.session_state.cs_handoff_reason or 'Escalation required.'}")
                if st.button("🔐 Open Escalation Panel"):
                    handoff_dialog()

            # ── PHASE: APPROVED / PROCESSED BY HUMAN AGENT ───────────────────────
            elif phase == 'approved':
                p      = st.session_state.cs_passenger
                choice = st.session_state.cs_choice

                if choice == 'refund':
                    prompt = f"""The human agent has approved and processed the refund for {p['name'].split()[0]}.
Confirm warmly — refund approved and will be processed within 5-7 business days. Do NOT say goodbye yet. Under 60 words."""
                elif choice == 'rebook':
                    prompt = f"""The human agent has rebooked {p['name'].split()[0]} on the next available flight to {p['destination']}.
Confirm warmly, mention a confirmation email shortly. Do NOT say goodbye yet. Under 60 words."""
                else:
                    prompt = f"""The human agent has arranged a hotel voucher tonight and rebooking tomorrow to {p['destination']} for {p['name'].split()[0]}.
Confirm warmly, all costs covered by the airline. Do NOT say goodbye yet. Under 60 words."""

                text, audio = emma_respond(prompt, st.session_state.cs_messages)
                st.session_state.cs_messages.append({"role": "assistant", "content": [{"text": text}]})
                st.session_state.cs_bubbles.append(("emma", text, audio))
                st.session_state.cs_phase = 'passenger_turn_3'
                st.rerun()

            # ── PHASE: PASSENGER TURN 3 ────────────────────────────────────────
            elif phase == 'passenger_turn_3':
                reply = st.text_input("✍️ Anything else?", key="p_reply_3",
                                      placeholder="e.g. Thank you, that's all")
                if st.button("📤 Send", key="send_3") and reply.strip():
                    p = st.session_state.cs_passenger
                    st.session_state.cs_bubbles.append(("passenger", reply, None))
                    history = st.session_state.cs_messages + [{"role": "user", "content": [{"text": reply}]}]

                    if wants_human(reply):
                        prompt = f"""The passenger asked to speak with a human agent.
Acknowledge this professionally and let {p['name'].split()[0]} know you're connecting them now. Under 40 words."""
                        text, audio = emma_respond(prompt, history)
                        st.session_state.cs_bubbles.append(("emma", text, audio))
                        st.session_state.cs_messages = history
                        st.session_state.cs_handoff_reason = "Passenger explicitly requested a human agent."
                        st.session_state.cs_phase = 'awaiting_approval'
                        st.rerun()

                    prompt = f"""Close the call warmly with {p['name'].split()[0]}.
Apologise once more for the inconvenience, wish them well, say a warm goodbye. Under 50 words."""
                    text, audio = emma_respond(prompt, history)
                    st.session_state.cs_bubbles.append(("emma", text, audio))

                    # Mark flight as processed and log the event
                    mark_flight_processed(p['flight'])
                    log_disruption_event(
                        flight_iata    = p['flight'],
                        status         = p['status'],
                        affected_count = 1,
                        airline        = p['airline'],
                        destination    = p['dest_iata'],
                        resolution     = st.session_state.cs_choice
                    )
                    st.session_state.cs_phase = 'complete'
                    st.rerun()

            # ── PHASE: COMPLETE ────────────────────────────────────────────────
            elif phase == 'complete':
                st.markdown("<p style='color:#888888;font-family:monospace;margin-top:15px;'>— Call concluded. Flight marked as processed. —</p>", unsafe_allow_html=True)

# ── Tab 4: Live Map ───────────────────────────────────────────────────────────

with tab4:
    st.markdown("<h4 style='color:#FFD700;font-family:monospace;'>🗺️ LIVE MAP</h4>", unsafe_allow_html=True)

    if not st.session_state.data_loaded or not flights:
        st.markdown(WAITING_MSG, unsafe_allow_html=True)
    else:
        import folium
        from streamlit_folium import st_folium

        m = folium.Map()
        all_coords = []

        for flight in flights:
            if flight['status'] == 'departed':
                continue
            origin = get_airport_coords(flight['origin'])
            dest   = get_airport_coords(flight['destination'])
            is_dis = flight['status'] in DISRUPTED_STATUSES
            line_color = '#FF4444' if is_dis else '#00CC66'

            if origin and dest:
                all_coords.extend([origin, dest])
                folium.PolyLine(
                    locations=[origin, dest], color=line_color, weight=2,
                    tooltip=f"{flight['flight_iata']} — {flight['status'].upper()}"
                ).add_to(m)
                folium.CircleMarker(
                    location=dest, radius=5,
                    color=line_color, fill=True, fill_color=line_color,
                    tooltip=f"{get_city_name(flight['destination'])} [{flight['status'].upper()}]"
                ).add_to(m)

        if all_coords:
            m.fit_bounds(all_coords)

        st_folium(m, width=1200, height=500)

# ── Tab 5: Disruption Log ─────────────────────────────────────────────────────

with tab5:
    st.markdown("<h4 style='color:#FFD700;font-family:monospace;'>📋 DISRUPTION LOG</h4>", unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.markdown(WAITING_MSG, unsafe_allow_html=True)
    else:
        if st.button("🗑️ CLEAR LOG"):
            with open('data/disruption_log.json', 'w') as f:
                json.dump([], f)
            st.rerun()

        if not logs:
            st.markdown("<p style='color:#00FF00;font-family:monospace;'>✅ No disruptions logged yet.</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='color:#FFFFFF;font-family:monospace;'>{len(logs)} disruption(s) handled</p>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            for log in logs:
                resolution_color = '#00CC66' if log.get('resolution') else '#888888'
                resolution_label = log.get('resolution', 'pending').upper()
                st.markdown(f"""
                <div style='background-color:#1a1a2e;border-left:4px solid #FF4444;
                padding:15px;margin:8px 0;border-radius:5px;'>
                <span style='color:#FF4444;font-family:monospace;font-size:16px;font-weight:bold;'>{log['flight_iata']}</span>
                <span style='color:#FFD700;font-family:monospace;'> &nbsp;{log.get('airline','')}</span>
                <span style='color:{resolution_color};font-family:monospace;font-size:12px;float:right;'>{resolution_label}</span><br>
                <span style='color:#FFFFFF;font-family:monospace;'>→ {get_city_name(log.get('destination',''))} &nbsp;|&nbsp; {log['status'].upper()} &nbsp;|&nbsp; {log['affected_count']} passenger(s)</span><br>
                <span style='color:#888888;font-family:monospace;font-size:12px;'>{log['timestamp']}</span>
                </div>
                """, unsafe_allow_html=True)
