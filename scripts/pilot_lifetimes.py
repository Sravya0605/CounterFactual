import os
from src.ingestion.parser import parse_cape_json
from scripts.match_resource_lifetimes import match_resource_lifetimes

pilot_dir = 'data/pilot_emotet'
files = [f for f in os.listdir(pilot_dir) if f.endswith('.json')]

for fname in files:
    path = os.path.join(pilot_dir, fname)
    events = parse_cape_json(path)
    result = match_resource_lifetimes(events)
    acquisitions = len(result['acquisitions'])
    releases = len(result['releases'])
    lifetimes = len(result['lifetimes'])
    orphans = len(result['orphan_releases'])
    still_active = len(result['still_active'])
    print(f'{fname}: acquisitions={acquisitions} releases={releases} lifetimes={lifetimes} orphans={orphans} still_active={still_active}')