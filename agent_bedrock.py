"""
agent_bedrock.py
FlightSense disruption detection agent — AWS Bedrock version.

Workflow (Python-orchestrated):
1. Load disrupted + unprocessed flights from flights.json
2. For each: find affected passengers → notify via Bedrock → log → mark processed
3. flights.json is NEVER deleted from — processed=True marks a handled disruption
"""
import boto3
import json
import os
from dotenv import load_dotenv

load_dotenv()

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DISRUPTED_STATUSES = ['cancelled', 'delayed', 'diverted', 'incident']

# ── Tool definitions (kept for architecture demo / future agentic use) ────────

tools = [
    {
        "toolSpec": {
            "name": "fetch_daily_flights",
            "description": "Fetches unprocessed disrupted flights from the monitoring list",
            "inputSchema": {"json": {"type": "object", "properties": {"airport": {"type": "string"}}, "required": ["airport"]}}
        }
    },
    {
        "toolSpec": {
            "name": "fetch_all_pnrs",
            "description": "Loads all passenger PNR records for a flight",
            "inputSchema": {"json": {"type": "object", "properties": {"flight_iata": {"type": "string"}}, "required": ["flight_iata"]}}
        }
    },
    {
        "toolSpec": {
            "name": "find_affected_pnrs",
            "description": "Finds all passengers affected by a disrupted flight",
            "inputSchema": {"json": {"type": "object", "properties": {"flight_iata": {"type": "string"}}, "required": ["flight_iata"]}}
        }
    },
    {
        "toolSpec": {
            "name": "notify_passenger",
            "description": "Generates customer service response for affected passenger",
            "inputSchema": {"json": {"type": "object", "properties": {
                "pnr":            {"type": "string"},
                "passenger_name": {"type": "string"},
                "flight_iata":    {"type": "string"},
                "destination":    {"type": "string"},
                "status":         {"type": "string"}
            }, "required": ["pnr", "passenger_name", "flight_iata", "destination", "status"]}}
        }
    },
    {
        "toolSpec": {
            "name": "log_disruption_event",
            "description": "Logs disruption event for audit trail",
            "inputSchema": {"json": {"type": "object", "properties": {
                "flight_iata":    {"type": "string"},
                "status":         {"type": "string"},
                "affected_count": {"type": "integer"}
            }, "required": ["flight_iata", "status", "affected_count"]}}
        }
    }
]

# ── Tool implementations ──────────────────────────────────────────────────────

def fetch_daily_flights(airport):
    """Return disrupted flights that have not yet been processed"""
    with open('data/flights.json', 'r') as f:
        flights = json.load(f)
    for f_ in flights:
        s = f_.get('status', '').lower()
        f_['status'] = 'cancelled' if s == 'canceled' else s
    unprocessed = [
        f for f in flights
        if f['status'] in DISRUPTED_STATUSES and not f.get('processed', False)
    ]
    print(f"  → {len(unprocessed)} unprocessed disruption(s) out of {len(flights)} total flights")
    return unprocessed

def find_affected_pnrs(flight_iata):
    with open('data/pnr_database.json', 'r') as f:
        pnrs = json.load(f)
    return [{'pnr': k, **v} for k, v in pnrs.items() if v['flight_iata'] == flight_iata]

def notify_passenger(pnr, passenger_name, flight_iata, destination, status):
    """Generate warm customer service notification via Bedrock"""
    prompt = f"""Passenger: {passenger_name} | PNR: {pnr} | Flight: {flight_iata} to {destination} | Status: {status}

Call this passenger now. Use their first name. Inform them of the disruption with genuine empathy.
Briefly offer 3 options: rebook, refund, or hotel voucher + rebook tomorrow.
Under 120 words. No markdown."""

    response = bedrock.converse(
        modelId=MODEL_ID,
        system=[{"text": """You are Emma, a customer service agent at Turkish Airlines handling flight disruptions.
Tone: professional, warm, efficient — the way an experienced airline agent speaks, not a script.
Use city names, not airport codes. If a passenger asks to speak with a human agent, acknowledge
the request professionally rather than arguing or redirecting them back to yourself."""}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 300}
    )
    return response['output']['message']['content'][0]['text']

def log_disruption_event(flight_iata, status, affected_count, airline=""):
    """Append disruption event to audit log"""
    import datetime

    # Look up airline name from flights.json
    if not airline:
        try:
            with open('data/flights.json', 'r') as f:
                flights = json.load(f)
            airline = next((f['airline'] for f in flights if f['flight_iata'] == flight_iata), "Unknown")
            destination = next((f['destination'] for f in flights if f['flight_iata'] == flight_iata), "")
        except:
            airline = "Unknown"
            destination = ""
    else:
        destination = ""

    log_entry = {
        'flight_iata':    flight_iata,
        'airline':        airline,
        'destination':    destination,
        'status':         status,
        'affected_count': affected_count,
        'timestamp':      str(datetime.datetime.now())
    }

    log_file = 'data/disruption_log.json'
    try:
        with open(log_file, 'r') as f:
            logs = json.load(f)
    except:
        logs = []

    logs.append(log_entry)
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)

    print(f"  📋 Logged: {flight_iata} | {status} | {affected_count} passenger(s)")
    return log_entry

def mark_flight_processed(flight_iata):
    """Mark a flight as processed so it won't be re-handled on next run"""
    with open('data/flights.json', 'r') as f:
        flights = json.load(f)
    for flight in flights:
        if flight['flight_iata'] == flight_iata:
            flight['processed'] = True
            break
    with open('data/flights.json', 'w') as f:
        json.dump(flights, f, indent=2)

# ── Main orchestration loop ───────────────────────────────────────────────────

def run_disruption_agent():
    print("🚨 FlightSense Agent Starting (Bedrock)...")
    print("=" * 55)

    # Step 1: Find unprocessed disrupted flights
    disrupted = fetch_daily_flights("IST")

    if not disrupted:
        print("✅ No new disruptions to process.")
        return

    print(f"🔴 Processing {len(disrupted)} disrupted flight(s)...\n")

    # Step 2: For each disrupted flight, run the full workflow
    for flight in disrupted:
        flight_iata  = flight['flight_iata']
        destination  = flight['destination']
        status       = flight['status']
        airline      = flight['airline']

        print(f"── {flight_iata} → {destination} [{status.upper()}] ({airline})")

        # Find affected passengers
        affected = find_affected_pnrs(flight_iata)

        if not affected:
            print(f"   ⚠️  No passengers found for {flight_iata}")
            log_disruption_event(flight_iata, status, 0, airline)
            mark_flight_processed(flight_iata)
            continue

        print(f"   {len(affected)} passenger(s) affected")

        # Notify each passenger
        for passenger in affected:
            print(f"   🔧 Notifying {passenger['passenger_name']} ({passenger['pnr']})")
            message = notify_passenger(
                passenger['pnr'],
                passenger['passenger_name'],
                flight_iata,
                passenger['destination'],
                status
            )
            print(f"   📢 {message[:100]}...")

        # Log and mark as processed
        log_disruption_event(flight_iata, status, len(affected), airline)
        mark_flight_processed(flight_iata)
        print(f"   ✅ Done\n")

    print("=" * 55)
    print("🏁 Agent run complete.")

if __name__ == "__main__":
    run_disruption_agent()
