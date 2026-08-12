"""
fetch_live_status.py

Production-grade monitoring — run periodically after initial baseline fetch.

Three-source validation logic:
  1. AviationStack  — primary flight status feed
  2. AeroDataBox    — independent status cross-check
  3. OpenSky        — transponder tiebreaker (only called on conflict)

Confidence levels:
  'high'             — AviationStack + AeroDataBox both confirm disruption
  'conflict'         — sources disagree; OpenSky called as tiebreaker:
                         airborne  → 'conflict_operating' (flight is flying, do NOT contact)
                         silent    → 'confirmed' (transponder silent, treat as cancelled)
  'unconfirmed'      — AeroDataBox has no data; single source only
  'ok'               — scheduled/active flight, AeroDataBox confirms operating
  'ok_unconfirmed'   — scheduled/active, departing within 2h but AeroDataBox has no data
  'ok_not_checked'   — scheduled/active, departure > 2h away (not worth checking yet)
"""
import requests
import json
import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

API_KEY            = os.getenv("AVIATIONSTACK_API_KEY")
AERODATABOX_KEY    = os.getenv("AERODATABOX_KEY")
DISRUPTED_STATUSES = ['cancelled', 'delayed', 'diverted', 'incident']
SKIP_STATUSES      = ['landed']   # keep active — real in-flight departures shown on board

AERODATABOX_BASE   = "https://prod.api.market/api/v1/aedbx/aerodatabox"

# LHR bounding box for OpenSky (±2.5° around 51.477°N, -0.461°W)
OPENSKY_BBOX = {"lamin": 48.977, "lomin": -2.961, "lamax": 53.977, "lomax": 2.039}


def check_opensky_airborne(flight_iata):
    """
    Tiebreaker: check if a flight's callsign is currently airborne near LHR.
    Returns True (airborne) / False (not detected) / None (OpenSky unavailable).
    """
    try:
        resp = requests.get(
            "https://opensky-network.org/api/states/all",
            params=OPENSKY_BBOX, timeout=10
        )
        if resp.status_code != 200:
            print(f"  ⚠️  OpenSky returned {resp.status_code}")
            return None
        states = resp.json().get('states', []) or []
        callsigns = {(s[1] or '').strip().upper() for s in states}
        airborne = flight_iata.strip().upper() in callsigns
        print(f"  🛰️  OpenSky tiebreaker for {flight_iata}: {'AIRBORNE ✈️' if airborne else 'NOT DETECTED 🚫'}")
        return airborne
    except Exception as e:
        print(f"  ⚠️  OpenSky unavailable: {e}")
        return None


def confirm_with_aerodatabox(flight_iata, scheduled_date):
    """
    Query AeroDataBox for the current status of a specific flight.
    Returns: 'cancelled' | 'delayed' | 'scheduled' | 'active' | 'unknown'
    Degrades gracefully to 'unknown' on any error.
    """
    if not AERODATABOX_KEY:
        return 'unknown'
    try:
        url = f"{AERODATABOX_BASE}/flights/Number/{flight_iata}/{scheduled_date}"
        headers = {"x-api-market-key": AERODATABOX_KEY}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 404:
            return 'unknown'   # flight not in AeroDataBox
        if resp.status_code != 200:
            print(f"  ⚠️  AeroDataBox {resp.status_code} for {flight_iata}")
            return 'unknown'
        data = resp.json()
        # AeroDataBox returns a list of flights (can include yesterday + today)
        # Pick the one departing on scheduled_date
        if isinstance(data, list):
            today_flights = [
                f for f in data
                if scheduled_date in (f.get('departure', {}).get('scheduledTime', {}).get('utc', ''))
            ]
            data = today_flights[0] if today_flights else (data[-1] if data else {})
        raw_status = (data.get('status') or '').lower()
        # Normalise to our status vocabulary
        if 'cancel' in raw_status:
            return 'cancelled'
        if 'delay' in raw_status:
            return 'delayed'
        if 'divert' in raw_status:
            return 'diverted'
        if raw_status in ('active', 'en-route', 'airborne', 'enroute'):
            return 'active'
        if raw_status in ('scheduled', 'expected', 'unknown'):
            return 'scheduled'
        if raw_status == 'arrived':
            return 'arrived'   # yesterday's flight — caller treats as 'unknown'
        return raw_status or 'unknown'
    except Exception as e:
        print(f"  ⚠️  AeroDataBox error for {flight_iata}: {e}")
        return 'unknown'


def cross_validate(flight_iata, av_status, scheduled_date, scheduled_time=None):
    """
    Three-source validation pipeline:
      Step 1 — AeroDataBox cross-check
      Step 2 — OpenSky tiebreaker (only on conflict)

    For non-disrupted flights, only validates if departure is within 2 hours
    (to conserve AeroDataBox API quota).

    Returns (confidence, sources) tuple.
    """
    if av_status not in DISRUPTED_STATUSES:
        # Only cross-check non-disrupted flights departing within 2 hours
        if scheduled_time:
            try:
                dep = datetime.fromisoformat(scheduled_time.replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                mins_until = (dep - now).total_seconds() / 60
                if mins_until > 120 or mins_until < -30:
                    return 'ok_not_checked', ['AviationStack']
            except Exception:
                return 'ok_not_checked', ['AviationStack']
        else:
            return 'ok_not_checked', ['AviationStack']

        # Within 2h window — check AeroDataBox
        adb_status = confirm_with_aerodatabox(flight_iata, scheduled_date)
        print(f"  🔍 {flight_iata} (non-disrupted): AviationStack={av_status.upper()} | AeroDataBox={adb_status.upper()}")
        if adb_status in ('unknown', 'arrived'):
            return 'ok_unconfirmed', ['AviationStack']
        if adb_status in DISRUPTED_STATUSES:
            # AeroDataBox sees a disruption that AviationStack missed!
            print(f"  🚨 {flight_iata}: AeroDataBox reports {adb_status.upper()} — AviationStack missed it!")
            return 'unconfirmed', ['AeroDataBox']
        return 'ok', ['AviationStack', 'AeroDataBox']

    adb_status = confirm_with_aerodatabox(flight_iata, scheduled_date)
    print(f"  🔍 {flight_iata}: AviationStack={av_status.upper()} | AeroDataBox={adb_status.upper()}")

    if adb_status in ('unknown', 'arrived'):
        return 'unconfirmed', ['AviationStack']

    if adb_status in DISRUPTED_STATUSES:
        return 'high', ['AviationStack', 'AeroDataBox']

    if adb_status in ('active', 'scheduled'):
        # Conflict — call OpenSky as tiebreaker
        print(f"  ⚡ CONFLICT: {flight_iata} — AviationStack={av_status.upper()} | AeroDataBox={adb_status.upper()} → calling OpenSky...")
        airborne = check_opensky_airborne(flight_iata)

        if airborne is True:
            # All three sources give a verdict: flight is operating
            print(f"  ✈️  OpenSky confirms {flight_iata} is AIRBORNE — overriding cancellation")
            return 'conflict_operating', ['AviationStack', 'AeroDataBox', 'OpenSky']
        elif airborne is False:
            # AeroDataBox disagrees but OpenSky also sees nothing → support cancellation
            print(f"  ✅ OpenSky silent for {flight_iata} — cancellation likely correct")
            return 'confirmed', ['AviationStack', 'AeroDataBox', 'OpenSky']
        else:
            # OpenSky unavailable — stay as conflict
            return 'conflict', ['AviationStack', 'AeroDataBox']

    return 'unconfirmed', ['AviationStack']


def fetch_live_status():
    # ── Load current watchlist ────────────────────────────────────────────────
    with open('data/flights.json', 'r') as f:
        watchlist = json.load(f)
    for f_ in watchlist:
        s = f_.get('status', '').lower()
        f_['status'] = 'cancelled' if s == 'canceled' else s

    watchlist_by_iata = {f['flight_iata']: f for f in watchlist}
    print(f"📋 Watchlist: {len(watchlist)} flight(s)")

    today = datetime.now().strftime('%Y-%m-%d')
    adb_available = bool(AERODATABOX_KEY)
    print(f"\n{'🔍' if adb_available else '⚠️ '} AeroDataBox cross-validation: {'enabled' if adb_available else 'disabled (no key)'}")

    # ── Fetch fresh AviationStack snapshot (general + cancelled) ─────────────
    print("\n🌐 Fetching AviationStack snapshot...")
    url = "http://api.aviationstack.com/v1/flights"
    base_params = {"access_key": API_KEY, "dep_iata": "LHR"}
    r1 = requests.get(url, params={**base_params, "limit": 80})
    r2 = requests.get(url, params={**base_params, "flight_status": "cancelled", "limit": 20})
    all_flights = r1.json().get('data', []) + r2.json().get('data', [])
    live_data = {"data": all_flights}
    print(f"   General feed: {len(r1.json().get('data', []))} | Cancelled feed: {len(r2.json().get('data', []))}")

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

        # Cross-validate all flights — disrupted fully, non-disrupted within 2h window
        scheduled_time = flight['departure']['scheduled']
        confidence, sources = cross_validate(iata, status, today, scheduled_time)

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
            'sources':     sources,
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
            # Refresh confidence for all flights (disrupted and non-disrupted)
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
    print(f"   AeroDataBox cross-validation: {'active' if adb_available else 'unavailable (no key)'}")

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
