"""
pnr_generator_demo_bedrock.py
Fetches live LHR flights from AviationStack (2 calls: general + cancelled feed).
Cross-validates disruptions against AeroDataBox for 2-source confidence.
Skips: active (in-air), landed (arrived) — not relevant for disruption POC.
Keeps: scheduled, cancelled, diverted, incident, delayed.
Generates synthetic PNR database via AWS Bedrock.
"""
import boto3
import json
import re
import os
import requests
from dotenv import load_dotenv

load_dotenv()

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
AVIATIONSTACK_KEY = os.getenv("AVIATIONSTACK_API_KEY")
AERODATABOX_KEY   = os.getenv("AERODATABOX_KEY")
AERODATABOX_BASE  = "https://prod.api.market/api/v1/aedbx/aerodatabox"

SKIP_STATUSES      = ['landed']   # keep active — real in-flight departures shown on board
DISRUPTED_STATUSES = ['cancelled', 'delayed', 'diverted', 'incident']


def confirm_with_aerodatabox(flight_iata, scheduled_date):
    """Query AeroDataBox for a flight's current status. Returns normalised status string."""
    if not AERODATABOX_KEY:
        return 'unknown'
    try:
        url = f"{AERODATABOX_BASE}/flights/Number/{flight_iata}/{scheduled_date}"
        resp = requests.get(url, headers={"x-api-market-key": AERODATABOX_KEY}, timeout=10)
        if resp.status_code != 200:
            return 'unknown'
        data = resp.json()
        if isinstance(data, list):
            today_flights = [
                f for f in data
                if scheduled_date in (f.get('departure', {}).get('scheduledTime', {}).get('utc', ''))
            ]
            data = today_flights[0] if today_flights else (data[-1] if data else {})
        raw = (data.get('status') or '').lower()
        if 'cancel' in raw:   return 'cancelled'
        if 'delay'  in raw:   return 'delayed'
        if 'divert' in raw:   return 'diverted'
        if raw in ('active', 'en-route', 'airborne', 'enroute'): return 'active'
        if raw in ('scheduled', 'expected'):  return 'scheduled'
        if raw == 'arrived':  return 'arrived'
        return raw or 'unknown'
    except Exception as e:
        print(f"  ⚠️  AeroDataBox error for {flight_iata}: {e}")
        return 'unknown'


def cross_validate(flight_iata, av_status, scheduled_date):
    """Returns (confidence, sources) tuple."""
    if av_status not in DISRUPTED_STATUSES:
        return 'ok', ['AviationStack']

    adb_status = confirm_with_aerodatabox(flight_iata, scheduled_date)
    print(f"  🔍 {flight_iata}: AviationStack={av_status.upper()} | AeroDataBox={adb_status.upper()}")

    if adb_status in ('unknown', 'arrived'):
        return 'unconfirmed', ['AviationStack']

    if av_status in DISRUPTED_STATUSES and adb_status in DISRUPTED_STATUSES:
        return 'high', ['AviationStack', 'AeroDataBox']
    elif av_status in DISRUPTED_STATUSES and adb_status in ('active', 'scheduled'):
        print(f"  ⚡ CONFLICT: {flight_iata} — AviationStack={av_status.upper()} but AeroDataBox={adb_status.upper()}")
        return 'conflict', ['AviationStack', 'AeroDataBox']
    return 'unconfirmed', ['AviationStack']

def fetch_real_flights():
    """
    Two API calls:
    1. Scheduled flights from LHR (the bulk)
    2. Cancelled/disrupted flights from LHR (explicit filter — AviationStack
       often omits cancellations from the general feed on free plans)
    Results are merged and deduplicated by flight IATA.
    """
    url = "http://api.aviationstack.com/v1/flights"
    base_params = {"access_key": AVIATIONSTACK_KEY, "dep_iata": "LHR"}

    # Call 1 — general feed (scheduled + whatever else comes back)
    r1 = requests.get(url, params={**base_params, "limit": 80})
    data1 = r1.json()

    # Call 2 — explicitly request cancelled flights
    r2 = requests.get(url, params={**base_params, "flight_status": "cancelled", "limit": 20})
    data2 = r2.json()

    print(f"  AviationStack general feed: {len(data1.get('data', []))} flights")
    print(f"  AviationStack cancelled feed: {len(data2.get('data', []))} flights")

    # Merge — use dict keyed on flight IATA to deduplicate
    merged = {}
    for flight in data1.get('data', []) + data2.get('data', []):
        iata = flight.get('flight', {}).get('iata')
        if iata and iata not in merged:
            merged[iata] = flight

    return {"data": list(merged.values())}

def parse_flights(data):
    flights = []
    skipped = 0
    no_iata = 0
    for flight in data.get('data', []):
        status = flight['flight_status'].lower()
        status = 'cancelled' if status == 'canceled' else status
        if status in SKIP_STATUSES:
            skipped += 1
            continue
        if not flight.get('flight', {}).get('iata'):
            no_iata += 1
            continue
        flights.append({
            'flight_iata': flight['flight']['iata'],
            'airline':     flight['airline']['name'],
            'origin':      flight['departure']['iata'],
            'destination': flight['arrival']['iata'],
            'scheduled':   flight['departure']['scheduled'],
            'status':      status,
            'delay':       flight['departure']['delay'],
            'processed':   False          # agent marks True after handling
        })
    print(f"  Skipped {skipped} active/landed flights")
    print(f"  Skipped {no_iata} flight(s) with no IATA code")

    # Deduplicate codeshares — same destination + time = one physical flight
    seen = set()
    unique = []
    for f in flights:
        key = (f['destination'], f['scheduled'])
        if key not in seen:
            seen.add(key)
            unique.append(f)

    dupes = len(flights) - len(unique)
    if dupes:
        print(f"  Removed {dupes} codeshare duplicate(s)")

    return unique

def generate_pnrs_for_batch(batch, existing_pnrs):
    """Generate PNRs for a batch of flights, avoiding duplicate PNR codes"""
    flight_list = "\n".join([
        f"{f['flight_iata']} from {f['origin']} to {f['destination']} [{f['status']}]"
        for f in batch
    ])
    used_pnrs = list(existing_pnrs.keys())

    prompt = f"""Create a passenger PNR database for these flights.
1 passenger per flight. Use ONLY flight codes from this list — including cancelled ones
(passengers booked BEFORE the cancellation, so every flight has exactly 1 passenger):
{flight_list}

For each passenger, assign a realistic cabin_class ("economy", "premium_economy", or "business")
and a fare_amount in USD consistent with that class (economy: 150-600, premium_economy: 400-1200,
business: 1500-6000).

Do NOT use these PNR codes (already taken): {', '.join(used_pnrs) if used_pnrs else 'none'}

Return pure JSON only — no markdown, no explanation:
{{"PNR_CODE": {{"passenger_name": "Full Name", "flight_iata": "code", "destination": "IATA", "contact_email": "email", "cabin_class": "economy|premium_economy|business", "fare_amount": 000}}}}"""

    response = bedrock.converse(
        modelId=MODEL_ID,
        system=[{"text": "You are a JSON generator. Return pure valid JSON only. No markdown. No code fences. No explanation."}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 4000}
    )

    raw = response['output']['message']['content'][0]['text']
    cleaned = re.sub(r'```json\n?|```', '', raw).strip()
    return json.loads(cleaned)

def generate_pnr_database(flights):
    """Generate PNRs in batches of 20 to avoid token limits"""
    print("Generating PNR database via Bedrock...")
    all_pnrs = {}
    batch_size = 20

    for i in range(0, len(flights), batch_size):
        batch = flights[i:i + batch_size]
        print(f"  Batch {i//batch_size + 1}: {len(batch)} flights...")
        try:
            batch_pnrs = generate_pnrs_for_batch(batch, all_pnrs)
            all_pnrs.update(batch_pnrs)
        except Exception as e:
            print(f"  ⚠️  Batch failed: {e}")

    return all_pnrs


if __name__ == "__main__":
    from collections import Counter
    from datetime import datetime

    print("Fetching LHR flights from Aviationstack...")
    raw_data = fetch_real_flights()
    flights = parse_flights(raw_data)

    status_counts = Counter(f['status'] for f in flights)
    print(f"Parsed {len(flights)} flights: {dict(status_counts)}")

    # Cross-validate disrupted flights against AeroDataBox
    today = datetime.now().strftime('%Y-%m-%d')
    if AERODATABOX_KEY:
        print(f"\n🔍 Cross-validating disruptions with AeroDataBox...")
        for flight in flights:
            if flight['status'] in DISRUPTED_STATUSES:
                confidence, sources = cross_validate(flight['flight_iata'], flight['status'], today)
                flight['confidence'] = confidence
                flight['sources']    = sources
            else:
                flight['confidence'] = 'ok'
                flight['sources']    = ['AviationStack']
    else:
        print("⚠️  AERODATABOX_KEY not set — skipping cross-validation")
        for flight in flights:
            flight['confidence'] = 'unconfirmed' if flight['status'] in DISRUPTED_STATUSES else 'ok'
            flight['sources']    = ['AviationStack']

    with open('data/flights.json', 'w') as f:
        json.dump(flights, f, indent=2)

    pnr_dict = generate_pnr_database(flights)

    with open('data/pnr_database.json', 'w') as f:
        json.dump(pnr_dict, f, indent=2)

    print(f"PNR database saved — {len(pnr_dict)} passengers")
    for pnr, p in pnr_dict.items():
        marker = "🔴" if any(f['flight_iata'] == p['flight_iata'] and f['status'] != 'scheduled' for f in flights) else "🟢"
        print(f"  {marker} {pnr} — {p['passenger_name']} — {p['flight_iata']}")
