'''
Normality test and estimations for Radziwill-Soundararajan conjecture data:
- Shapiro-Wilk test for normality
- Kolmogorov-Smirnov distance to standard normal
- Wasserstein distance to standard normal

To run:
    python3 normal_test.py --label <CREMONA_LABEL>
'''

import json
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import argparse

def read_sha_data(label: str = None, data_path: str = None) -> tuple[np.ndarray, np.ndarray]:
    ''' Read sha data from JSON file given by label or data_path.

    Args:
        label: CREMONA Label for the data set (e.g., "69a1"). CAUTION: this uses precomputed data
            with max |d| = 10000.
        data_path: Path to the JSON file containing the SHA data.
    '''
    if label is not None:
        file = f'output/sha_data/{label}_maxd10000_sha_data.json'
    elif data_path is not None:
        file = f'{data_path}'
    else:
        raise ValueError("Either label or data_path must be provided.")
    
    with open(file, 'r') as f:
        data = json.load(f)
    sha_data = data['data']
    ds = np.array(list(sha_data.keys())).astype(int)
    sha_values = np.array(list(sha_data.values()))
    return ds, sha_values

def compute_Z_values(ds: np.ndarray, sha_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ''' Compute the "Z" values as per Radziwill-Soundararajan conjecture'''

    # From Conjecture 1 in Radziwill-Soundararajan
    mu_E = -0.5 - 1.5 * np.log(2)
    sigma_sq_E = 1.0 + 2.5 * (np.log(2)**2)

    # Compute the log normalized Sha
    normalized_sha_values = (sha_values/np.sqrt(np.abs(ds)))
    log_normalized_sha_values = np.log(normalized_sha_values)

    # Compute the "Z" values
    ob_data = log_normalized_sha_values - mu_E * np.log(np.log(np.abs(ds)))
    ob_data /= np.sqrt(sigma_sq_E * np.log(np.log(np.abs(ds))))
    return ob_data, ds

def SW_by_bound(label: str, ob_data: np.ndarray, ds: np.ndarray) -> dict[int, float]:
    '''Compute Shapiro-Wilk for normality test as bound on |d| increases by the increment of 1000.
    Save the statistics and p-values in a dictionary to output/SW_test/{label}_shapiro_wilk_{bound}.json
    '''
    # Run a loop to perform Shapiro-Wilk test for normality as bound increases
    SW_res = {}
    max_bound = np.max(np.abs(ds))
    max_bound = ((max_bound // 1000) + 1) * 1000 # round up to nearest 1000
    
    for bound in range(1000, max_bound + 1000, 1000):
        val_ind = np.where((np.abs(ds) > 20) & (np.abs(ds) < bound))
        input = ob_data[val_ind]
        statistic, p_value = stats.shapiro(input)
        SW_res[bound] = (statistic, p_value)

    with open(f'output/SW_test/{label}_shapiro_wilk_{max_bound}.json', 'w') as f:
        json.dump(SW_res, f)
    return SW_res

def plot_p_values(SW_res: dict[int, float], label: str) -> None:
    '''
    Plot the Shapiro-Wilk p-values as a function of bound on |d|.
    Save the plot as a png file to output/p_values/plots/{label}_shapiro_wilk_log_pvalues_{max(bounds)}.png
    '''
    bounds = list(SW_res.keys())
    pvals = [SW_res[b][1] for b in bounds]
    plt.plot(bounds, np.log(pvals), marker='o')
    plt.xlabel('Bound on |d|')
    plt.ylabel('Shapiro-Wilk log p-value')
    plt.title(f'Shapiro-Wilk log p-values for curve {label}')
    plt.legend(['log p-value'])
    plt.savefig(f'output/SW_test/plots/{label}_shapiro_wilk_log_pvalues_{max(bounds)}.png')
    plt.close()

def plot_SW_statistics(SW_res: dict[int, float], label: str) -> None:
    '''
    Plot the Shapiro-Wilk statistics as a function of bound on |d|.
    Save the plot as a png file to output/normality/plots/{label}_shapiro_wilk_statistics_{max(bounds)}.png
    '''
    bounds = list(SW_res.keys())
    stats = [SW_res[b][0] for b in bounds]
    plt.plot(bounds, stats, marker='o')
    plt.xlabel('Bound on |d|')
    plt.ylabel('Shapiro-Wilk statistic')
    plt.title(f'Shapiro-Wilk statistics for curve {label}')
    plt.legend(['statistic'])
    plt.savefig(f'output/SW_test/plots/{label}_shapiro_wilk_statistics_{max(bounds)}.png')
    plt.close()

def compute_KS_distance(ob_data: np.ndarray, ds: np.ndarray) -> float:
    '''
    Compute the Kolmogorov-Smirnov distance between ob_data and a standard normal distribution.
    Returns: KS distance (supremum of absolute difference between CDFs)
    '''
    # Filter data where |d| > 20
    val_ind = np.where(np.abs(ds) > 20)
    filtered_data = ob_data[val_ind]
    
    # Compute KS distance (the statistic is the distance)
    distance, _ = stats.kstest(filtered_data, 'norm', args=(0, 1))
    return distance

def compute_Wasserstein_distance(ob_data: np.ndarray, ds: np.ndarray) -> float:
    '''
    Compute the Wasserstein distance between ob_data and a standard normal distribution.
    Returns: Wasserstein distance
    '''
    # Filter data where |d| > 20
    val_ind = np.where(np.abs(ds) > 20)
    filtered_data = ob_data[val_ind]
    
    # Generate standard normal samples with same size
    normal_samples = np.random.normal(0, 1, len(filtered_data))
    
    # Compute Wasserstein distance
    distance = stats.wasserstein_distance(filtered_data, normal_samples)
    return distance

def KS_by_bound(label: str, ob_data: np.ndarray, ds: np.ndarray) -> dict[int, float]:
    '''
    Compute KS distance to standard normal as bound on |d| increases by the increment of 1000.
    Save the distances in a dictionary to output/KS_dist/{label}_ks_{bound}.json
    '''
    KS_res = {}
    max_bound = np.max(np.abs(ds))
    max_bound = ((max_bound // 1000) + 1) * 1000  # round up to nearest 1000
    
    for bound in range(1000, max_bound + 1000, 1000):
        val_ind = np.where((np.abs(ds) > 20) & (np.abs(ds) < bound))
        input_data = ob_data[val_ind]
        distance, _ = stats.kstest(input_data, 'norm', args=(0, 1))
        KS_res[bound] = distance
    
    with open(f'output/KS_dist/{label}_ks_{max_bound}.json', 'w') as f:
        json.dump(KS_res, f)
    return KS_res

def Wasserstein_by_bound(label: str, ob_data: np.ndarray, ds: np.ndarray) -> dict[int, float]:
    '''
    Compute Wasserstein distance to standard normal as bound on |d| increases by the increment of 1000.
    Save the distances in a dictionary to output/Wasserstein_dist/{label}_wasserstein_{bound}.json
    '''
    Wasserstein_res = {}
    max_bound = np.max(np.abs(ds))
    max_bound = ((max_bound // 1000) + 1) * 1000  # round up to nearest 1000
    
    for bound in range(1000, max_bound + 1000, 1000):
        val_ind = np.where((np.abs(ds) > 20) & (np.abs(ds) < bound))
        input_data = ob_data[val_ind]
        # Generate standard normal samples with same size
        normal_samples = np.random.normal(0, 1, len(input_data))
        distance = stats.wasserstein_distance(input_data, normal_samples)
        Wasserstein_res[bound] = distance
    
    with open(f'output/Wasserstein_dist/{label}_wasserstein_{max_bound}.json', 'w') as f:
        json.dump(Wasserstein_res, f)
    return Wasserstein_res

def plot_KS_distances(KS_res: dict[int, float], label: str) -> None:
    '''
    Plot the KS distances as a function of bound on |d|.
    Save the plot as a png file to output/KS_dist/plots/{label}_ks_distances_{max(bounds)}.png
    '''
    bounds = list(KS_res.keys())
    distances = [KS_res[b] for b in bounds]
    plt.plot(bounds, distances, marker='o')
    plt.xlabel('Bound on |d|')
    plt.ylabel('KS distance')
    plt.title(f'KS distances for curve {label}')
    plt.legend(['distance'])
    plt.savefig(f'output/KS_dist/plots/{label}_ks_distances_{max(bounds)}.png')
    plt.close()

def plot_Wasserstein_distances(Wasserstein_res: dict[int, float], label: str) -> None:
    '''
    Plot the Wasserstein distances as a function of bound on |d|.
    Save the plot as a png file to output/Wasserstein_dist/plots/{label}_wasserstein_{max(bounds)}.png
    '''
    bounds = list(Wasserstein_res.keys())
    distances = [Wasserstein_res[b] for b in bounds]
    plt.plot(bounds, distances, marker='o')
    plt.xlabel('Bound on |d|')
    plt.ylabel('Wasserstein distance')
    plt.title(f'Wasserstein distances for curve {label}')
    plt.legend(['distance'])
    plt.savefig(f'output/Wasserstein_dist/plots/{label}_wasserstein_{max(bounds)}.png')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Normality test for Radziwill-Soundararajan conjecture data.')
    parser.add_argument('--label', type=str, default=None, help='CREMONA Label for the data set (e.g., "69a1") using precomputed data with max |d| = 10000.')
    parser.add_argument('--data-path', type=str, default=None, help='Path to the directory containing the SHA data JSON files.')
    args = parser.parse_args()

    label = args.label
    data_path = args.data_path
    
    assert label is not None or data_path is not None, "Either --label or --data-path must be provided."

    # Read data and compute Z values
    ds, sha_values = read_sha_data(label, data_path)
    ob_data, ds = compute_Z_values(ds, sha_values)

    # Shapiro-Wilk test
    SW_res = SW_by_bound(label, ob_data, ds)
    plot_SW_statistics(SW_res, label)
    plot_p_values(SW_res, label)
    
    # Kolmogorov-Smirnov distance
    KS_res = KS_by_bound(label, ob_data, ds)
    plot_KS_distances(KS_res, label)
    
    # Wasserstein distance
    Wasserstein_res = Wasserstein_by_bound(label, ob_data, ds)
    plot_Wasserstein_distances(Wasserstein_res, label)

