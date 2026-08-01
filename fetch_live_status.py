"""
fetch_live_status.py

Production-grade monitoring — run periodically after initial baseline fetch.

Logic:
- Fetch fresh IST snapshot from Aviationstack
- For flights in watchlist: update status if changed
- Remove landed/active flights from watchlist (no longer relevant)
- Add new flights not yet in watchlist (scheduled/cancelled only)
- Preserve processed=True flag — never reset a processed flight
"""
import requests
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
DISRUPTED_STATUSES  = ['cancelled', 'delayed', 'diverted', 'incident']
SKIP_STATUSES       = ['active', 'landed']


def fetch_live_status():
    # ── Load current watchlist ────────────────────────────────────────────────
    with open('data/flights.json', 'r') as f:
        watchlist = json.load(f)
    for f_ in watchlist:
        s = f_.get('status', '').lower()
        f_['status'] = 'cancelled' if s == 'canceled' else s

    watchlist_by_iata = {f['flight_iata']: f for f in watchlist}
    print(f"📋 Watchlist: {len(watchlist)} flight(s)")

    # ── Fetch fresh snapshot ──────────────────────────────────────────────────
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
        live_snapshot[iata] = {
            'flight_iata': iata,
            'airline':     flight['airline']['name'],
            'origin':      flight['departure']['iata'],
            'destination': flight['arrival']['iata'],
            'scheduled':   flight['departure']['scheduled'],
            'status':      status,
            'delay':       flight['departure']['delay'],
            'processed':   False
        }

    print(f"🌐 Live snapshot: {len(live_snapshot)} relevant flight(s)\n")

    # ── Compare: detect status changes on watched flights ─────────────────────
    changes = 0
    departed = 0
    updated_watchlist = []

    for iata, flight in watchlist_by_iata.items():
        live = live_snapshot.get(iata)

        if not live:
            # Not in live data — flight may have departed or data gap
            # If it was scheduled and now missing, mark as departed
            if flight['status'] == 'scheduled':
                print(f"  🛫 {iata} not in live data — marking as departed")
                flight['status'] = 'departed'
                departed += 1
            # If already disrupted/processed, keep it
            updated_watchlist.append(flight)
            continue

        old_status = flight['status']
        new_status = live['status']

        if new_status != old_status and not flight.get('processed', False):
            print(f"  🚨 STATUS CHANGE: {iata} {old_status.upper()} → {new_status.upper()}")
            flight['status'] = new_status
            flight['delay']  = live['delay']
            changes += 1
        else:
            flight['delay'] = live['delay']  # always refresh delay

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

    if changes == 0:
        print("\n🟢 No new disruptions detected")
    else:
        print(f"\n🔴 {changes} new disruption(s) — run agent_bedrock.py to process")

    return changes


if __name__ == "__main__":
    fetch_live_status()
