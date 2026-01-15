#!/usr/bin/env sage
"""
Expand sha data for 46a1 from max_d=50000 to max_d=100000.
"""

import sys
import json
import signal
import time

sys.path.insert(0, 'ants_xvii/rs_conjecture')

from sage.all import EllipticCurve, is_fundamental_discriminant, gcd, kronecker

# Import sha_an from RS_visualization
from RS_visualization import sha_an

# Flush stdout for real-time output
import functools
print = functools.partial(print, flush=True)

# Timeout handler
class ShaTimeout(Exception):
    pass

def timeout_handler(signum, frame):
    raise ShaTimeout("sha_an computation timed out")

# Load existing data
input_file = 'ants_xvii/rs_conjecture/output/sha_data/46a1_maxd50000_sha_data.json'
output_file = 'ants_xvii/rs_conjecture/output/sha_data/46a1_maxd100000_sha_data.json'

print(f"Loading existing data from {input_file}...")
with open(input_file, 'r') as f:
    existing_data = json.load(f)

curve_label = existing_data['curve']
old_max_d = existing_data['max_d']
new_max_d = 100000

print(f"Curve: {curve_label}")
print(f"Expanding from max_d={old_max_d} to max_d={new_max_d}")

# Setup curve
E = EllipticCurve(curve_label)
N = int(E.conductor())
epsE = int(E.root_number())

print(f"Conductor N={N}, root_number={epsE}")

# Copy existing data
sha_data = dict(existing_data['data'])
print(f"Existing data points: {len(sha_data)}")

# Iterate over new d values
new_count = 0
timeout_count = 0
skipped_count = 0

# We need to check d in range (old_max_d, new_max_d] for both positive and negative
d_values_to_check = []
for abs_d in range(old_max_d + 1, new_max_d + 1):
    d_values_to_check.append(abs_d)
    d_values_to_check.append(-abs_d)

print(f"Checking {len(d_values_to_check)} new d values...")
print()

start_time = time.time()
last_progress_time = start_time

for i, d in enumerate(d_values_to_check):
    # Check if d is a fundamental discriminant
    if not is_fundamental_discriminant(d):
        continue

    # Check gcd condition
    if gcd(d, 2*N) != 1:
        continue

    # Check kronecker condition for root number +1
    kron_val = kronecker(d, -N)
    if epsE * kron_val != 1:
        continue

    # This d qualifies - compute sha for the twist
    Ed = E.quadratic_twist(d)

    # Set up timeout (10 seconds)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)

    compute_start = time.time()
    try:
        sha_val = sha_an(Ed, 1)
        sha_data[str(d)] = float(sha_val)
        new_count += 1
        compute_time = time.time() - compute_start
        print(f"d={d:>7}: sha={sha_val:<6} ({compute_time:.2f}s) [+{new_count}, T{timeout_count}]")
    except ShaTimeout:
        sha_data[str(d)] = 0.0
        timeout_count += 1
        print(f"d={d:>7}: TIMEOUT [+{new_count}, T{timeout_count}]")
    except Exception as e:
        sha_data[str(d)] = 0.0
        timeout_count += 1
        print(f"d={d:>7}: ERROR - {e} [+{new_count}, T{timeout_count}]")
    finally:
        signal.alarm(0)

    # Periodic summary every 100 qualifying d values
    if (new_count + timeout_count) % 100 == 0:
        elapsed = time.time() - start_time
        rate = (new_count + timeout_count) / elapsed if elapsed > 0 else 0
        print(f"--- Summary: {new_count + timeout_count} computed in {elapsed:.1f}s ({rate:.2f}/s) ---")

total_elapsed = time.time() - start_time
print(f"\n{'='*60}")
print(f"FINISHED in {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
print(f"Added {new_count} new data points, {timeout_count} timeouts")
print(f"Total data points: {len(sha_data)}")
print(f"{'='*60}")

# Write output
output_data = {
    "curve": curve_label,
    "max_d": new_max_d,
    "restricted": False,
    "data": sha_data
}

print(f"Writing to {output_file}...")
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print("Done!")
