"""
pnr_generator_demo_bedrock.py
Fetches live IST flights from Aviationstack (1 API call).
Skips: active (in-air), landed (arrived) — not relevant for disruption POC.
Keeps: scheduled, cancelled, diverted, incident, delayed.
Adds processed=False field to all flights.
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

SKIP_STATUSES = ['active', 'landed']   # not useful for disruption POC

def fetch_real_flights():
    url = "http://api.aviationstack.com/v1/flights"
    params = {
        "access_key": AVIATIONSTACK_KEY,
        "dep_iata": "IST",
        "limit": 100
    }
    response = requests.get(url, params=params)
    return response.json()

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
1 passenger per flight. Use ONLY flight codes from this list:
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
    print("Fetching IST flights from Aviationstack...")
    raw_data = fetch_real_flights()
    flights = parse_flights(raw_data)

    from collections import Counter
    status_counts = Counter(f['status'] for f in flights)
    print(f"Saved {len(flights)} flights: {dict(status_counts)}")

    with open('data/flights.json', 'w') as f:
        json.dump(flights, f, indent=2)

    pnr_dict = generate_pnr_database(flights)

    with open('data/pnr_database.json', 'w') as f:
        json.dump(pnr_dict, f, indent=2)

    print(f"PNR database saved — {len(pnr_dict)} passengers")
    for pnr, p in pnr_dict.items():
        marker = "🔴" if any(f['flight_iata'] == p['flight_iata'] and f['status'] != 'scheduled' for f in flights) else "🟢"
        print(f"  {marker} {pnr} — {p['passenger_name']} — {p['flight_iata']}")
