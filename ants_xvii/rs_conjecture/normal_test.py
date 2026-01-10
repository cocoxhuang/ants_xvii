import json
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import argparse

def read_sha_data(label: str) -> tuple[np.ndarray, np.ndarray]:
    file = f'output/{label}_sha_data.json'
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

def compute_p_values_by_bound(label: str, ob_data: np.ndarray, ds: np.ndarray) -> dict[int, float]:
    '''Compute Shapiro-Wilk p-values for normality test as bound on |d| increases by the increment of 1000.
    Save the p-values in a dictionary to output/p_values/{label}_shapiro_wilk_p_values_{bound}.json
    '''
    # Run a loop to perform Shapiro-Wilk test for normality as bound increases
    p_values = {}
    max_bound = np.max(np.abs(ds))
    max_bound = ((max_bound // 1000) + 1) * 1000 # round up to nearest 1000
    
    for bound in range(1000, max_bound + 1000, 1000):
        val_ind = np.where((np.abs(ds) > 20) & (np.abs(ds) < bound))
        input = ob_data[val_ind]
        statistic, p_value = stats.shapiro(input)
        p_values[bound] = (p_value)

    # dump p-values to a json file
    with open(f'output/p_values/{label}_shapiro_wilk_p_values_{max_bound}.json', 'w') as f:
        json.dump(p_values, f)
    return p_values

def plot_p_values(p_values: dict[int, float], label: str) -> None:
    '''
    Plot the Shapiro-Wilk p-values as a function of bound on |d|.
    Save the plot as a png file to output/p_values/plots/{label}_shapiro_wilk_log_pvalues.png
    '''
    bounds = list(p_values.keys())
    pvals = [p_values[b] for b in bounds]
    plt.plot(bounds, np.log(pvals), marker='o')
    plt.xlabel('Bound on |d|')
    plt.ylabel('Shapiro-Wilk log p-value')
    plt.title(f'Shapiro-Wilk log p-values for curve {label}')
    plt.legend(['log p-value'])
    plt.grid()
    plt.savefig(f'output/p_values/plots/{label}_shapiro_wilk_log_pvalues_{max(bounds)}.png')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Normality test for Radziwill-Soundararajan conjecture data.')
    parser.add_argument('--label', type=str, required=True, help='CREMONA Label for the data set (e.g., "69a1").')
    args = parser.parse_args()

    label = args.label
    ds, sha_values = read_sha_data(label)
    ob_data, ds = compute_Z_values(ds, sha_values)
    p_values = compute_p_values_by_bound(label, ob_data, ds)
    plot_p_values(p_values, label)