"""
fetch_live_status.py

Production-grade monitoring — run periodically after initial baseline fetch.

Logic:
- Fetch fresh IST snapshot from Aviationstack
- Cross-validate disruptions against OpenSky Network (free, no key required)
- For flights in watchlist: update status if changed
- Add confidence field: 'high' (both sources agree) / 'unconfirmed' (single source only)
- Remove landed/active flights from watchlist (no longer relevant)
- Add new flights not yet in watchlist (scheduled/cancelled only)
- Preserve processed=True flag — never reset a processed flight
"""
import requests
import json
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

API_KEY          = os.getenv("AVIATIONSTACK_API_KEY")
DISRUPTED_STATUSES = ['cancelled', 'delayed', 'diverted', 'incident']
SKIP_STATUSES      = ['active', 'landed']

# IST (LTFM) bounding box for OpenSky
IST_LAT, IST_LON = 40.976, 28.814
OPENSKY_BBOX = {
    'lamin': IST_LAT - 2.5,
    'lomin': IST_LON - 2.5,
    'lamax': IST_LAT + 2.5,
    'lomax': IST_LON + 2.5,
}


def fetch_opensky_callsigns():
    """
    Fetch callsigns of aircraft currently airborne near IST from OpenSky Network.
    Returns a set of normalised callsigns (uppercase, stripped).
    Returns empty set on failure — gracefully degrades if OpenSky is unavailable.
    """
    try:
        url = "https://opensky-network.org/api/states/all"
        resp = requests.get(url, params=OPENSKY_BBOX, timeout=10)
        if resp.status_code != 200:
            print(f"  ⚠️  OpenSky returned {resp.status_code} — skipping cross-validation")
            return set()
        data = resp.json()
        states = data.get('states', []) or []
        # state[1] is callsign (may be None or padded with spaces)
        callsigns = set()
        for state in states:
            cs = state[1]
            if cs:
                callsigns.add(cs.strip().upper())
        print(f"  🛰️  OpenSky: {len(callsigns)} aircraft airborne near IST")
        return callsigns
    except Exception as e:
        print(f"  ⚠️  OpenSky unavailable ({e}) — skipping cross-validation")
        return set()


def cross_validate(flight_iata, status, opensky_callsigns):
    """
    Cross-validate a disruption against OpenSky.

    Logic:
    - If OpenSky data is unavailable (empty set): confidence = 'unconfirmed'
    - If flight is disrupted AND not airborne in OpenSky: confidence = 'high'
    - If flight is disrupted BUT airborne in OpenSky: confidence = 'conflict'
      (AviationStack says cancelled but plane is in the air — possible data lag)
    - Scheduled flights: no cross-validation needed
    """
    if not opensky_callsigns:
        return 'unconfirmed'

    if status not in DISRUPTED_STATUSES:
        return 'ok'

    iata_upper = flight_iata.strip().upper()
    airborne = iata_upper in opensky_callsigns

    if airborne:
        print(f"  ⚡ CONFLICT: {flight_iata} is {status.upper()} per AviationStack but AIRBORNE per OpenSky")
        return 'conflict'
    else:
        return 'high'


def fetch_live_status():
    # ── Load current watchlist ────────────────────────────────────────────────
    with open('data/flights.json', 'r') as f:
        watchlist = json.load(f)
    for f_ in watchlist:
        s = f_.get('status', '').lower()
        f_['status'] = 'cancelled' if s == 'canceled' else s

    watchlist_by_iata = {f['flight_iata']: f for f in watchlist}
    print(f"📋 Watchlist: {len(watchlist)} flight(s)")

    # ── Fetch OpenSky first (cross-validation layer) ──────────────────────────
    print("\n🛰️  Fetching OpenSky cross-validation data...")
    opensky_callsigns = fetch_opensky_callsigns()
    opensky_available = bool(opensky_callsigns)

    # ── Fetch fresh AviationStack snapshot ───────────────────────────────────
    print("\n🌐 Fetching AviationStack snapshot...")
    url = "http://api.aviationstack.com/v1/flights"
    params = {"access_key": API_KEY, "dep_iata": "IST", "limit": 100}
    response = requests.get(url, params=params)
    live_data = response.json()

    # Parse into dict, skip active/landed, deduplicate codeshares
    live_snapshot = {}
    seen_routes = set()
    for flight in live_data.get('data', []):
        status = flight['flight_status'].lower()
        status = 'cancelled' if status == 'canceled' else status
        if status in SKIP_STATUSES:
            continue
        route_key = (flight['arrival']['iata'], flight['departure']['scheduled'])
        if route_key in seen_routes:
            continue
        iata = flight.get('flight', {}).get('iata')
        if not iata:
            continue
        seen_routes.add(route_key)

        confidence = cross_validate(iata, status, opensky_callsigns)

        live_snapshot[iata] = {
            'flight_iata': iata,
            'airline':     flight['airline']['name'],
            'origin':      flight['departure']['iata'],
            'destination': flight['arrival']['iata'],
            'scheduled':   flight['departure']['scheduled'],
            'status':      status,
            'delay':       flight['departure']['delay'],
            'processed':   False,
            'confidence':  confidence,
            'sources':     ['AviationStack', 'OpenSky'] if opensky_available else ['AviationStack']
        }

    print(f"   AviationStack: {len(live_snapshot)} relevant flight(s)\n")

    # ── Compare: detect status changes on watched flights ─────────────────────
    changes = 0
    departed = 0
    updated_watchlist = []

    for iata, flight in watchlist_by_iata.items():
        live = live_snapshot.get(iata)

        if not live:
            if flight['status'] == 'scheduled':
                print(f"  🛫 {iata} not in live data — marking as departed")
                flight['status'] = 'departed'
                departed += 1
            updated_watchlist.append(flight)
            continue

        old_status = flight['status']
        new_status = live['status']

        if new_status != old_status and not flight.get('processed', False):
            print(f"  🚨 STATUS CHANGE: {iata} {old_status.upper()} → {new_status.upper()} [{live['confidence'].upper()}]")
            flight['status']     = new_status
            flight['delay']      = live['delay']
            flight['confidence'] = live['confidence']
            flight['sources']    = live['sources']
            changes += 1
        else:
            flight['delay']      = live['delay']
            # Refresh confidence on existing disruptions too
            if new_status in DISRUPTED_STATUSES:
                flight['confidence'] = live['confidence']
                flight['sources']    = live['sources']

        updated_watchlist.append(flight)

    # ── Merge: add new flights not in watchlist ───────────────────────────────
    new_added = 0
    for iata, live_flight in live_snapshot.items():
        if iata not in watchlist_by_iata:
            updated_watchlist.append(live_flight)
            new_added += 1
            print(f"  ➕ New flight added: {iata} → {live_flight['destination']} [{live_flight['status']}]")

    # ── Save ──────────────────────────────────────────────────────────────────
    with open('data/flights.json', 'w') as f:
        json.dump(updated_watchlist, f, indent=2)

    print(f"\n✅ Sync complete at {datetime.now().strftime('%H:%M:%S')}")
    print(f"   {changes} status change(s) | {departed} departed | {new_added} new flight(s)")
    print(f"   {len(updated_watchlist)} total flight(s) in watchlist")
    print(f"   OpenSky cross-validation: {'active' if opensky_available else 'unavailable (graceful degradation)'}")

    # Confidence summary
    disrupted = [f for f in updated_watchlist if f.get('status') in DISRUPTED_STATUSES]
    if disrupted:
        high       = sum(1 for f in disrupted if f.get('confidence') == 'high')
        conflict   = sum(1 for f in disrupted if f.get('confidence') == 'conflict')
        unconfirmed = sum(1 for f in disrupted if f.get('confidence') == 'unconfirmed')
        print(f"\n🔴 {len(disrupted)} disrupted flight(s):")
        print(f"   ✅ High confidence (both sources): {high}")
        print(f"   ⚡ Conflicts (data mismatch):      {conflict}")
        print(f"   ⚠️  Unconfirmed (single source):   {unconfirmed}")

    return changes


if __name__ == "__main__":
    fetch_live_status()
