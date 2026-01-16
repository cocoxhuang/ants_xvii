"""RS_visualization.py

Generates an HTML animation and PNG frames visualizing the Radziwiłł-Soundararajan
conjecture for quadratic twists of elliptic curves.

Usage:
    # PDF (histogram) visualization
    sage -python RS_visualization.py --max_d 10000 --num_frames 10 --pdf
    sage -python RS_visualization.py --curve 11a1 --max_d 5000 --num_frames 5 --pdf

    # CDF (cumulative distribution) visualization
    sage -python RS_visualization.py --max_d 10000 --num_frames 10 --cdf

    # Load from previously saved data file (fast, skips computation)
    sage -python RS_visualization.py --data_file output/46a1_maxd10000_nf10_..._data.json --cdf
    sage -python RS_visualization.py --data_file output/46a1_maxd10000_nf10_..._data.json --max_d 5000 --pdf

    # Restricted mode (use specific d values from a JSON file)
    sage -python RS_visualization.py --restrict_file ../infinite_bsd/output/res.json --num_frames 8 --pdf

Each run creates uniquely named outputs to preserve previous runs:
    - PNG frames: output/frames/<run_id>/frame_*.png
    - HTML file: output/<run_id>.html
    - Data file: output/<run_id>_data.json (for reuse)

Where <run_id> is:
    - Unrestricted: <curve>_maxd<max_d>_nf<num_frames>_<timestamp>_<mode>
    - Restricted:   <curve>_restricted_nf<num_frames>_<timestamp>_<mode>

Arguments:
    --curve         Elliptic curve label (default: 46a1; ignored if --data_file provided)
    --max_d         Maximum absolute value of discriminant d (default: 1000; also filters --data_file)
    --num_frames    Number of frames/checkpoints for the animation (default: 5)
    --output_dir    Directory to save output files (default: output)
    --restrict_file Path to JSON file containing restricted d values (mutually exclusive with --data_file)
    --data_file     Path to previously saved SHA data JSON file (skips computation)
    --pdf           Use PDF (histogram/density) visualization (required, mutually exclusive with --cdf)
    --cdf           Use CDF (cumulative distribution) visualization (required, mutually exclusive with --pdf)

"""

import argparse
import json
import os
import signal
import sys
from datetime import datetime


class TeeStream:
    """Stream that writes to both stdout and a log file."""

    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'a', buffering=1)  # Line buffered for real-time

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure real-time writing

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


class ShaTimeout(Exception):
    """Raised when SHA computation times out."""
    pass


def _sha_timeout_handler(signum, frame):
    raise ShaTimeout()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.stats import norm, gaussian_kde
from sage.all import EllipticCurve, is_fundamental_discriminant, gcd, kronecker


def generate_run_id(curve_label, max_d, num_frames, restricted=False):
    """Generate a unique run identifier based on parameters and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize curve label for use in filename (replace colons, etc.)
    safe_label = curve_label.replace(":", "_").replace("/", "_")
    if restricted:
        return f"{safe_label}_restricted_nf{num_frames}_{timestamp}"
    else:
        return f"{safe_label}_maxd{max_d}_nf{num_frames}_{timestamp}"


def sha_an(E, root_number):

    if root_number == -1:
        raise ValueError("we are not dealing with this case")
    
    # Henceforth the rank is even

    Lvalue_at_1, err = E.lseries().at1()

    # if zero is in the interval (L_value_at_1 - err, L_value_at_1 + err
    # then the rank might be 2, 4, ...; in this case we want to do the 
    # same computation as in `generate_z_scores` below

    if (Lvalue_at_1 - err < 0.0) and (Lvalue_at_1 + err > 0.0):
        signal.signal(signal.SIGALRM, _sha_timeout_handler)
        signal.alarm(10)
        try:
            sha_an_float = E.sha().an_numerical(prec=20, proof=False)
        finally:
            signal.alarm(0)  # Cancel the alarm

        sha_size = int(sha_an_float.round())
    else:
        # in this case we are sure that the rank is 0
        # so we can compute sha directly from L(1)
        E = E.minimal_model()
        c_infty = E.period_lattice().omega(bsd_normalise=True)
        tamagawa_product = E.tamagawa_product()
        sha_size_float = (Lvalue_at_1 * E.torsion_order()**2) / (c_infty * tamagawa_product)
        sha_size = int(sha_size_float.round())

    return sha_size



def generate_z_scores(curve_label, max_d=None, restricted_ds=None, debug=True):
    """
    Generate z-scores for quadratic twists of the given curve.

    If restricted_ds is provided, iterate over exactly those d values.
    Otherwise, iterate over d in range(-max_d, max_d+1) testing both signs.

    Returns:
        tuple: (data_list, sha_dict) where:
            - data_list: list of (abs(d), z_score) tuples, sorted by abs(d)
            - sha_dict: dict mapping d -> sha_analytic value (for persistence)
    """
    E = EllipticCurve(curve_label)
    N = int(E.conductor())
    epsE = int(E.root_number())

    if restricted_ds is not None:
        print(f"Generating data for curve {curve_label} with {len(restricted_ds)} restricted d values...")
    else:
        print(f"Generating data for curve {curve_label} with |d| < {max_d}...")

    if debug:
        print(f"  Curve {curve_label}: conductor N={N}, root_number={epsE}")

    mu_E = -0.5 - 1.5 * np.log(2)
    sigma_sq_E = 1.0 + 2.5 * (np.log(2)**2)

    data = []
    sha_dict = {}  # Store d -> sha_analytic for persistence

    # Debug counters
    count_total = 0
    count_fund = 0
    count_gcd = 0
    count_kron = 0
    count_sha_ok = 0
    count_loglog = 0

    # Determine which d values to iterate over
    if restricted_ds is not None:
        d_values = restricted_ds
    else:
        # Original behavior: iterate d_val from 1 to max_d, testing both +d and -d
        d_values = []
        for d_val in range(1, max_d + 1):
            d_values.append(d_val)
            d_values.append(-d_val)

    for idx, d in enumerate(d_values):
        if debug and idx % 50 == 0:
            print(f"  Processing d #{idx+1}/{len(d_values)} (d={d})...")
            sys.stdout.flush()
        count_total += 1

        if not is_fundamental_discriminant(d):
            continue
        count_fund += 1

        if gcd(d, 2*N) != 1:
            continue
        count_gcd += 1

        kron_val = kronecker(d, -N)
        if epsE * kron_val != 1:
            continue
        count_kron += 1

        try:
            E_d = E.quadratic_twist(d)
            if debug:
                print(f"    Computing SHA for d={d}...", flush=True)

            sha_int = sha_an(E_d, 1)

            if sha_int < 1:
                continue
            count_sha_ok += 1

            # Store the SHA value for persistence
            sha_dict[d] = sha_int

            val = np.log(sha_int / np.sqrt(abs(d)))
            log_log_d = np.log(np.log(abs(d)))
            if log_log_d <= 0:
                continue
            count_loglog += 1

            z = (val - (mu_E * log_log_d)) / np.sqrt(sigma_sq_E * log_log_d)
            data.append((abs(d), z))

        except ShaTimeout:
            if debug:
                print("TIMEOUT")
            continue

        except Exception as e:
            if debug and count_kron < 10:
                print(f"  Exception for d={d}: {e}")
            continue

    if debug:
        print(f"  Filter stats:")
        print(f"    Total candidates: {count_total}")
        print(f"    After fundamental_discriminant: {count_fund}")
        print(f"    After gcd(d,2N)==1: {count_gcd}")
        print(f"    After kronecker condition: {count_kron}")
        print(f"    After sha >= 1: {count_sha_ok}")
        print(f"    After log(log(d)) > 0: {count_loglog}")

    data.sort(key=lambda x: x[0])
    return data, sha_dict


def save_sha_data(sha_dict, curve_label, max_d, restricted, output_path):
    """
    Save SHA data to a JSON file for later reuse.

    Args:
        sha_dict: dict mapping d -> sha_analytic value
        curve_label: elliptic curve label
        max_d: maximum |d| used in computation (None if restricted mode)
        restricted: whether restricted mode was used
        output_path: path to save the JSON file
    """
    # Convert int keys to strings for JSON compatibility
    data_for_json = {str(k): v for k, v in sha_dict.items()}

    output_data = {
        "curve": curve_label,
        "max_d": max_d,
        "restricted": restricted,
        "data": data_for_json
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved SHA data ({len(sha_dict)} entries) to: {output_path}")


def load_and_compute_z_scores(data_file_path, max_d=None):
    """
    Load SHA data from a previously saved file and compute z-scores.

    Args:
        data_file_path: path to the JSON data file
        max_d: optional maximum |d| to filter data (if None, use all data)

    Returns:
        tuple: (data_list, curve_label, file_max_d) where:
            - data_list: list of (abs(d), z_score) tuples, sorted by abs(d)
            - curve_label: the curve label from the file
            - file_max_d: the max_d stored in the file (for reference)
    """
    print(f"Loading SHA data from: {data_file_path}")

    with open(data_file_path, 'r') as f:
        file_data = json.load(f)

    curve_label = file_data["curve"]
    file_max_d = file_data.get("max_d")
    sha_data = file_data["data"]

    print(f"  Curve: {curve_label}")
    print(f"  File max_d: {file_max_d}")
    print(f"  Total entries in file: {len(sha_data)}")

    if max_d is not None:
        print(f"  Filtering to |d| <= {max_d}")

    # RS conjecture constants
    mu_E = -0.5 - 1.5 * np.log(2)
    sigma_sq_E = 1.0 + 2.5 * (np.log(2)**2)

    data = []
    count_filtered = 0
    count_bad = 0
    count_loglog = 0

    for d_str, sha_int in sha_data.items():
        d = int(d_str)

        # Apply max_d filter if specified
        if max_d is not None and abs(d) > max_d:
            count_filtered += 1
            continue

        # Skip bad entries (e.g., 0.0 values from failed/incomplete computations)
        if sha_int <= 0:
            count_bad += 1
            continue

        if sha_int < 1:
            continue

        val = np.log(sha_int / np.sqrt(abs(d)))
        log_log_d = np.log(np.log(abs(d)))
        if log_log_d <= 0:
            continue
        count_loglog += 1

        z = (val - (mu_E * log_log_d)) / np.sqrt(sigma_sq_E * log_log_d)
        data.append((abs(d), z))

    if max_d is not None:
        print(f"  Filtered out {count_filtered} entries with |d| > {max_d}")
    if count_bad > 0:
        print(f"  Skipped {count_bad} bad entries (sha <= 0)")
    print(f"  Generated {len(data)} z-scores (after log(log(d)) > 0 filter)")

    data.sort(key=lambda x: x[0])
    return data, curve_label, file_max_d


def create_frame(all_data, current_max_d, curve_label, output_filename=None, restricted=False, mode='pdf'):
    """
    Create a single frame showing the distribution for |d| <= current_max_d.
    Saves to output_filename if provided, otherwise displays.

    Args:
        all_data: list of (abs(d), z_score) tuples
        current_max_d: maximum |d| to include in this frame
        curve_label: elliptic curve label
        output_filename: path to save the figure (if None, displays)
        restricted: whether restricted mode was used
        mode: 'pdf' for histogram/density, 'cdf' for cumulative distribution
    """
    subset_z = np.array([z for (d_abs, z) in all_data if d_abs <= current_max_d])

    fig, ax = plt.subplots(figsize=(10, 7))

    x_axis = np.linspace(-4, 4, 200 if mode == 'pdf' else 500)

    if mode == 'cdf':
        # CDF visualization
        if len(subset_z) > 0:
            z_sorted = np.sort(subset_z)
            n = len(z_sorted)
            y_values = np.arange(1, n + 1) / n

            # Theoretical standard normal CDF (black dashed)
            ax.plot(x_axis, norm.cdf(x_axis), 'k--', linewidth=2.5,
                    label='Theoretical Prediction (CDF)')

            # Empirical CDF (blue step)
            ax.step(z_sorted, y_values, color='cornflowerblue', where='post',
                    linewidth=2, label='Observed CDF')
        else:
            print(f"  Warning: No data points for |d| <= {current_max_d}")

        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 1.05)
        if restricted:
            ax.set_title(f"Cumulative Distribution of Normalized Sha\n(Good twists of {curve_label}, |d| < {current_max_d}, N={len(subset_z)})")
        else:
            ax.set_title(f"Cumulative Distribution of Normalized Sha\n(Twists of {curve_label}, |d| < {current_max_d}, N={len(subset_z)})")
        ax.set_xlabel("Normalized Discrepancy (V)")
        ax.set_ylabel("Probability P(Z < V)")
        ax.legend(loc='upper left')

    else:
        # PDF visualization (original behavior)
        # Draw histogram FIRST (lower zorder, appears behind)
        if len(subset_z) > 10:
            fixed_bins = np.linspace(-4, 4, 40)
            ax.hist(subset_z, bins=fixed_bins, density=True,
                    color='cornflowerblue', alpha=0.6, label='Observed Histogram',
                    zorder=1, edgecolor='steelblue', linewidth=0.5)

            try:
                kde = gaussian_kde(subset_z)
                ax.plot(x_axis, kde(x_axis), 'b-', linewidth=2.5,
                        label='Observed Density (KDE)', zorder=3)
            except Exception:
                pass
        else:
            print(f"  Warning: Only {len(subset_z)} data points for |d| <= {current_max_d}")

        # Draw theoretical normal AFTER (higher zorder, appears on top)
        ax.plot(x_axis, norm.pdf(x_axis), 'k--', linewidth=2.5,
                label='Theoretical N(0,1)', zorder=2)

        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 0.75)
        if restricted:
            ax.set_title(f"Distribution of Normalized Sha\n(Good twists of {curve_label}, |d| < {current_max_d}, N={len(subset_z)})")
        else:
            ax.set_title(f"Distribution of Normalized Sha\n(Twists of {curve_label}, |d| < {current_max_d}, N={len(subset_z)})")
        ax.set_xlabel("Standard Deviations from Mean (Z-score)")
        ax.set_ylabel("Probability Density")
        ax.legend(loc='upper right')

    ax.grid(True, alpha=0.3, zorder=0)

    if output_filename:
        fig.savefig(output_filename, dpi=150)
        plt.close(fig)
        print(f"Saved {output_filename} (N={len(subset_z)})")
    else:
        plt.show()


def create_html_animation(all_data, checkpoints, curve_label, output_filename, restricted=False, mode='pdf'):
    """
    Create an HTML animation from the data and checkpoints.

    Args:
        all_data: list of (abs(d), z_score) tuples
        checkpoints: list of |d| values for each frame
        curve_label: elliptic curve label
        output_filename: path to save the HTML file
        restricted: whether restricted mode was used
        mode: 'pdf' for histogram/density, 'cdf' for cumulative distribution
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    def update(frame_idx):
        ax.clear()

        current_X = checkpoints[frame_idx]
        subset_z = np.array([z for (d_abs, z) in all_data if d_abs <= current_X])
        N = len(subset_z)

        x_axis = np.linspace(-4, 4, 200 if mode == 'pdf' else 500)

        if mode == 'cdf':
            # CDF visualization
            if N > 0:
                z_sorted = np.sort(subset_z)
                y_values = np.arange(1, N + 1) / N

                # Theoretical standard normal CDF (black dashed)
                ax.plot(x_axis, norm.cdf(x_axis), 'k--', linewidth=2.5,
                        label='Theoretical Prediction (CDF)')

                # Empirical CDF (blue step)
                ax.step(z_sorted, y_values, color='cornflowerblue', where='post',
                        linewidth=2, label='Observed CDF')

            ax.set_xlim(-4, 4)
            ax.set_ylim(0, 1.05)
            ax.legend(loc='upper left', framealpha=0.9)
            ax.set_xlabel("Normalized Discrepancy (V)")
            ax.set_ylabel("Probability P(Z < V)")

            if restricted:
                title_str = (
                    r"$\bf{Radziwiłł-Soundararajan\ Conjecture\ (CDF)}$" + f" ({curve_label}, good twists)\n" +
                    f"Discriminant Bound: $|d| < {current_X}$   |   Sample Size: $N = {N}$"
                )
            else:
                title_str = (
                    r"$\bf{Radziwiłł-Soundararajan\ Conjecture\ (CDF)}$" + f" ({curve_label})\n" +
                    f"Discriminant Bound: $|d| < {current_X}$   |   Sample Size: $N = {N}$"
                )

        else:
            # PDF visualization (original behavior)
            # Draw histogram FIRST (lower zorder)
            if N > 10:
                fixed_bins = np.linspace(-4, 4, 40)
                ax.hist(subset_z, bins=fixed_bins, density=True,
                        color='cornflowerblue', alpha=0.6, label='Observed Histogram',
                        zorder=1, edgecolor='steelblue', linewidth=0.5)

                try:
                    kde = gaussian_kde(subset_z)
                    ax.plot(x_axis, kde(x_axis), 'b-', linewidth=2.5,
                            label='Observed Density (KDE)', zorder=3)
                except Exception:
                    pass

            # Draw theoretical normal AFTER (higher zorder)
            ax.plot(x_axis, norm.pdf(x_axis), 'k--', linewidth=2.5,
                    label='Conjecture (Standard Normal)', zorder=2)

            ax.set_xlim(-4, 4)
            ax.set_ylim(0, 0.75)
            ax.legend(loc='upper right', framealpha=0.9)
            ax.set_xlabel("Normalized Discrepancy (Z-score)")
            ax.set_ylabel("Probability Density")

            if restricted:
                title_str = (
                    r"$\bf{Radziwiłł-Soundararajan\ Conjecture\ Verification}$" + f" ({curve_label}, good twists)\n" +
                    f"Discriminant Bound: $|d| < {current_X}$   |   Sample Size: $N = {N}$"
                )
            else:
                title_str = (
                    r"$\bf{Radziwiłł-Soundararajan\ Conjecture\ Verification}$" + f" ({curve_label})\n" +
                    f"Discriminant Bound: $|d| < {current_X}$   |   Sample Size: $N = {N}$"
                )

        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_title(title_str, fontsize=14)

    ani = animation.FuncAnimation(fig, update, frames=len(checkpoints), interval=1000)
    plt.close()

    html_content = ani.to_jshtml()
    with open(output_filename, "w") as f:
        f.write(html_content)

    print(f"Saved HTML animation to '{output_filename}'")


def compute_checkpoints(max_d, num_frames):
    """Compute evenly-spaced checkpoints from max_d/num_frames to max_d."""
    step = max_d // num_frames
    checkpoints = [step * (i + 1) for i in range(num_frames)]
    if checkpoints[-1] != max_d:
        checkpoints[-1] = max_d
    return checkpoints


def compute_checkpoints_from_data(data, num_frames):
    """Compute evenly-spaced checkpoints based on the actual data range."""
    if not data:
        return []
    max_d = max(abs(d) for d, _ in data)
    return compute_checkpoints(max_d, num_frames)


def main():
    parser = argparse.ArgumentParser(
        description="Generate RS conjecture visualization (HTML animation + PNG frames)"
    )
    parser.add_argument(
        "--curve",
        type=str,
        default="46a1",
        help="Elliptic curve label (default: 46a1; ignored if --data_file is provided)"
    )
    parser.add_argument(
        "--max_d",
        type=int,
        default=None,
        help="Maximum absolute value of discriminant d (default: 1000 for unrestricted mode; can also be used to filter data when using --data_file)"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=5,
        help="Number of frames/checkpoints for the animation (default: 5)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save output files (default: output)"
    )

    # Mutually exclusive group for data source
    data_source_group = parser.add_mutually_exclusive_group()
    data_source_group.add_argument(
        "--restrict_file",
        type=str,
        default=None,
        help="Path to JSON file containing restricted d values (keyed by curve label)"
    )
    data_source_group.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Path to previously saved SHA data JSON file (skips computation)"
    )

    # Required mutually exclusive group for visualization mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--pdf",
        action="store_true",
        help="Use PDF (histogram/density) visualization"
    )
    mode_group.add_argument(
        "--cdf",
        action="store_true",
        help="Use CDF (cumulative distribution) visualization"
    )

    args = parser.parse_args()

    curve_label = args.curve
    max_d = args.max_d
    num_frames = args.num_frames
    output_dir = args.output_dir
    restrict_file = args.restrict_file
    data_file = args.data_file
    viz_mode = 'cdf' if args.cdf else 'pdf'

    # Determine data source and load/generate data
    restricted_ds = None
    sha_dict = None
    is_restricted = False
    from_data_file = False

    if data_file:
        # Load from previously saved data file
        from_data_file = True
        full_data, curve_label, file_max_d = load_and_compute_z_scores(data_file, max_d=max_d)
        # Use the effective max_d for run_id generation
        effective_max_d = max_d if max_d is not None else file_max_d
        print(f"Loaded data for curve {curve_label}")

    elif restrict_file:
        # Load restricted d values
        with open(restrict_file, 'r') as f:
            restrict_data = json.load(f)

        if curve_label not in restrict_data:
            available_keys = list(restrict_data.keys())
            raise ValueError(
                f"Curve '{curve_label}' not found in restrict file. "
                f"Available curves: {available_keys}"
            )

        restricted_ds = restrict_data[curve_label]
        print(f"Loaded {len(restricted_ds)} restricted d values for curve {curve_label}")

        # Filter restricted_ds by max_d if both are provided
        if max_d is not None:
            restricted_ds = [d for d in restricted_ds if abs(d) <= max_d]
            print(f"Filtered to {len(restricted_ds)} values with |d| <= {max_d}")

        is_restricted = True
        effective_max_d = max_d

    else:
        # Default unrestricted mode
        if max_d is None:
            max_d = 1000
        effective_max_d = max_d

    # Generate unique run identifier
    run_id = generate_run_id(curve_label, effective_max_d, num_frames, restricted=is_restricted)
    # Add mode suffix to run_id
    run_id = f"{run_id}_{viz_mode}"
    print(f"Run ID: {run_id}")

    # Create output directories
    frames_dir = os.path.join(output_dir, "frames", run_id)
    os.makedirs(frames_dir, exist_ok=True)

    # Set up logging to file for real-time monitoring
    log_file_path = os.path.join(output_dir, f"{run_id}.log")
    tee = TeeStream(log_file_path)
    sys.stdout = tee
    print(f"Logging to: {log_file_path}")

    print(f"Curve: {curve_label}")
    print(f"Visualization mode: {viz_mode.upper()}")
    if from_data_file:
        print(f"Data source: {data_file}")
    elif is_restricted:
        print(f"Mode: RESTRICTED ({len(restricted_ds)} d values)")
    else:
        print(f"Mode: unrestricted, max_d={max_d}")
    print(f"num_frames: {num_frames}")

    # Generate or use loaded data
    if not from_data_file:
        print("\nStep 1: Generating z-scores...")
        full_data, sha_dict = generate_z_scores(curve_label, max_d=max_d, restricted_ds=restricted_ds)
        print(f"Generated {len(full_data)} data points.\n")

        # Save SHA data for future reuse
        data_file_path = os.path.join(output_dir, f"{run_id}_data.json")
        save_sha_data(sha_dict, curve_label, max_d, is_restricted, data_file_path)
    else:
        print(f"\nStep 1: Using loaded data ({len(full_data)} data points)\n")

    # Compute checkpoints based on actual data if restricted or from file, otherwise use max_d
    if is_restricted or from_data_file:
        checkpoints = compute_checkpoints_from_data(full_data, num_frames)
    else:
        checkpoints = compute_checkpoints(max_d, num_frames)
    print(f"checkpoints: {checkpoints}")

    print("\nStep 2: Creating PNG frames...")
    for val in checkpoints:
        output_path = os.path.join(frames_dir, f"frame_{val}.png")
        create_frame(full_data, val, curve_label, output_filename=output_path,
                     restricted=is_restricted, mode=viz_mode)

    print("\nStep 3: Creating HTML animation...")
    html_path = os.path.join(output_dir, f"{run_id}.html")
    create_html_animation(full_data, checkpoints, curve_label, html_path,
                          restricted=is_restricted, mode=viz_mode)

    print(f"\nDone!")
    print(f"  Frames saved to: {frames_dir}/")
    print(f"  Animation saved to: {html_path}")
    print(f"  Log saved to: {log_file_path}")
    if sha_dict is not None:
        print(f"  Data saved to: {data_file_path}")

    # Restore stdout and close log file
    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()
