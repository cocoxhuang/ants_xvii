"""RS_visualization.py

Generates an HTML animation and PNG frames visualizing the Radziwiłł-Soundararajan
conjecture for quadratic twists of the elliptic curve 46a1.

Usage:
    sage -python RS_visualization.py --max_d 10000 --num_frames 10

This will generate:
    - PNG frames: output/frames/frame_1000.png, frame_2000.png, ..., frame_10000.png
    - HTML file: output/rs_conjecture_convergence.html

Arguments:
    --max_d       Maximum absolute value of discriminant d (default: 1000)
    --num_frames  Number of frames/checkpoints for the animation (default: 5)
    --output_dir  Directory to save output files (default: output)

"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.stats import norm, gaussian_kde
from sage.all import EllipticCurve, is_fundamental_discriminant, gcd, kronecker


def generate_z_scores(max_d, debug=True):
    """Generate z-scores for quadratic twists of 46a1 with |d| < max_d."""
    print(f"Generating data for |d| < {max_d}...")
    E = EllipticCurve('46a1')
    N = int(E.conductor())
    epsE = int(E.root_number())

    if debug:
        print(f"  Curve 46a1: conductor N={N}, root_number={epsE}")

    mu_E = -0.5 - 1.5 * np.log(2)
    sigma_sq_E = 1.0 + 2.5 * (np.log(2)**2)

    data = []

    # Debug counters
    count_total = 0
    count_fund = 0
    count_gcd = 0
    count_kron = 0
    count_sha_ok = 0
    count_loglog = 0

    for d_val in range(1, max_d + 1):
        for sign in [1, -1]:
            d = sign * d_val
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
                sha_an_float = float(E_d.sha().an(descent_second_limit=5))

                sha_int = int(round(sha_an_float))
                if sha_int < 1:
                    continue
                count_sha_ok += 1

                val = np.log(sha_int / np.sqrt(abs(d)))
                log_log_d = np.log(np.log(abs(d)))
                if log_log_d <= 0:
                    continue
                count_loglog += 1

                z = (val - (mu_E * log_log_d)) / np.sqrt(sigma_sq_E * log_log_d)
                data.append((abs(d), z))

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
    return data


def create_frame(all_data, current_max_d, output_filename=None):
    """
    Create a single frame showing the distribution for |d| <= current_max_d.
    Saves to output_filename if provided, otherwise displays.
    """
    subset_z = np.array([z for (d_abs, z) in all_data if d_abs <= current_max_d])

    fig, ax = plt.subplots(figsize=(10, 7))

    x_axis = np.linspace(-4, 4, 200)

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
    ax.set_title(f"Distribution of Normalized Sha\n(Twists of 46a1, |d| < {current_max_d}, N={len(subset_z)})")
    ax.set_xlabel("Standard Deviations from Mean (Z-score)")
    ax.set_ylabel("Probability Density")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2, zorder=0)

    if output_filename:
        fig.savefig(output_filename, dpi=150)
        plt.close(fig)
        print(f"Saved {output_filename} (N={len(subset_z)})")
    else:
        plt.show()


def create_html_animation(all_data, checkpoints, output_filename):
    """Create an HTML animation from the data and checkpoints."""
    fig, ax = plt.subplots(figsize=(10, 7))

    def update(frame_idx):
        ax.clear()

        current_X = checkpoints[frame_idx]
        subset_z = np.array([z for (d_abs, z) in all_data if d_abs <= current_X])
        N = len(subset_z)

        x_axis = np.linspace(-4, 4, 200)

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
        ax.grid(True, alpha=0.2, zorder=0)
        ax.legend(loc='upper right', framealpha=0.9)

        ax.set_xlabel("Normalized Discrepancy (Z-score)")
        ax.set_ylabel("Probability Density")

        title_str = (
            r"$\bf{Radziwiłł-Soundararajan\ Conjecture\ Verification\ (46a1)}$" + "\n" +
            f"Discriminant Bound: $|d| < {current_X}$   |   Sample Size: $N = {N}$"
        )
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate RS conjecture visualization (HTML animation + PNG frames)"
    )
    parser.add_argument(
        "--max_d",
        type=int,
        default=1000,
        help="Maximum absolute value of discriminant d (default: 1000)"
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
    args = parser.parse_args()

    max_d = args.max_d
    num_frames = args.num_frames
    output_dir = args.output_dir

    # Create output directories
    frames_dir = os.path.join(output_dir, "frames")
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    checkpoints = compute_checkpoints(max_d, num_frames)
    print(f"max_d: {max_d}")
    print(f"num_frames: {num_frames}")
    print(f"checkpoints: {checkpoints}")

    print("\nStep 1: Generating z-scores...")
    full_data = generate_z_scores(max_d)
    print(f"Generated {len(full_data)} data points.\n")

    print("Step 2: Creating PNG frames...")
    for val in checkpoints:
        output_path = os.path.join(frames_dir, f"frame_{val}.png")
        create_frame(full_data, val, output_filename=output_path)

    print("\nStep 3: Creating HTML animation...")
    html_path = os.path.join(output_dir, "rs_conjecture_convergence.html")
    create_html_animation(full_data, checkpoints, html_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
