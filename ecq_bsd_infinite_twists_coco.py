"""ecq_bsd_infinite_twists.py

This script generates data files for elliptic curves with conductor less than 1024.
The data files contain the normalized a_p values for the first 50 primes,
as well as the conductor, rank, torsion, adelic level, adelic index, adelic genus,
and sha of the curve. The data files are saved in the parquet format.

To run this, execute the following command in the terminal:

    sage -python ecq_bsd_infinite_twists.py

"""


from lmfdb import db
import pandas as pd
import numpy as np

from sage.all import EllipticCurve, primes_first_n, round, Integer
from sage.schemes.elliptic_curves.isogeny_small_degree import isogenies_2
import time

OUTPUT_FILE = 'data/output.txt'

# Load the elliptic curve table from the LMFDB
ecq = db.ec_curvedata
mwbsd = db.ec_mwbsd
ec_localdata = db.ec_localdata

# Run a search query to get only the rank 0 or 1 curves.
# For now we set a limit to make things faster

NUM_AP_VALS = 100  # Number of primes to use for the a_p values
ADDITIONAL_COLS = ['conductor', 'rank', 'torsion', 'absD', 'bad_primes', 'sha', 'manin_constant', 'regulator']
SEARCH_COLS = ['ainvs', 'lmfdb_label'] + ADDITIONAL_COLS
OUTPUT_COLS = ['lmfdb_label'] + [str(p) for p in primes_first_n(NUM_AP_VALS)] + ADDITIONAL_COLS

MWBSD_COLS = ['lmfdb_label', 'real_period', 'special_value', 'tamagawa_product']

def ap_normalized(E, p):
    ap = E.ap(p)
    normalization_quotient = 2 * p.sqrt()
    return np.float32(round(ap / normalization_quotient, NUM_DECIMAL_PLACES))

# Function to get the curves data from ecq
def get_curves_data(tab):
    data = []
    for curve in tab:
        ainvs = curve['ainvs']
        cond = Integer(curve['conductor'])
        minimal_disc = Integer(curve['absD'])
        bad_primes = curve['bad_primes']
        manin_constant = curve['manin_constant']
        torsion = curve['torsion']
        regulator = curve['regulator']
        
        E = EllipticCurve(ainvs)

        # condition 1b: a3 = 0
        a3 = E.ap(3)  
        condition_1b = (a3 == 0)

        # condition 1c: irreducibility of residual representation at ood primes
        isogeny_degrees = [phi.degree() for phi in E.isogenies_prime_degree()]
        condition_1c = (isogeny_degrees == []) or (isogeny_degrees == [2])

        # condition 1d: ramification at all bad primes 
        condition_1d = True  # initialize to True
        for p in bad_primes:
            ord_p_of_min_disc = minimal_disc.valuation(p)
            order_of_order = ord_p_of_min_disc.valuation(p)
            if order_of_order > 0:
                condition_1d = False
                break

        # condition 1f: odd manin_constant 
        condition_1f = (manin_constant % 2 != 0)

        # condition 1g: 2 torsion size is 2
        # so the torsion size ise either 2 or 6 or 10
        condition_1g = (torsion in [2, 6, 10])
        
        # filter all conditions
        conditions = [condition_1b, condition_1c, condition_1d, condition_1f, condition_1g]
        if all(conditions):
            # now we know the curve has a 2_isogenous_curve
            # condition 1i: sha(2_isogenous_curve) = 0
            E_iso = [phi.codomain().ainvs() for phi in isogenies_2(E)]
            if len(E_iso) > 1:
                raise Exception("There are multiple 2-isogenous curves. Check the code.")
            # E_iso = EllipticCurve(E_iso[0])._lmfdb_label
            E_iso = (float(a_i) for a_i in E_iso[0])
            data.append([curve['lmfdb_label'], curve['regulator'], E_iso])

    return data

def get_mwbsd_data(tab):
    data = []
    for curve in tab:
        label = curve['lmfdb_label']
        special_value = curve['special_value']
        real_period = curve['real_period']

        data.append([label, special_value, real_period])

    return data

def foo(cond_bound=20):

    print(f"Generating data file for curves of conductor up to {cond_bound}...")
    output_file = OUTPUT_FILE  # .format(NUM_AP_VALS, 1, MY_LOCAL_LIM-1)

    # fetch curve data
    curves_query = {'conductor': {'$lt': cond_bound}, 'semistable' : True}

    # it does not seem like Burungale et al. used the optimality condition
    # my_query = {'conductor': {'$lt': cond_bound}, 'semistable' : True, 'optimality': 1}
    the_curves = ecq.search(curves_query, projection=SEARCH_COLS, one_per=['lmfdb_iso'])
    data = get_curves_data(the_curves)
    # data = list(the_curves)
    # turn it into a pandas dataframe
    data = pd.DataFrame(data, columns=['lmfdb_label', 'regulator', '2_iso_ainvs'])
    data.set_index('lmfdb_label', inplace=True)

    # getting the bsd data
    mwbsd_query = {'lmfdb_label': {'$in': list(data.index)}}
    the_bsd = mwbsd.search(mwbsd_query, projection=MWBSD_COLS)
    bsd_data = get_mwbsd_data(the_bsd)
    # turn it into a pandas dataframe
    bsd_data = pd.DataFrame(bsd_data, columns=['lmfdb_label', 'special_value', 'real_period'])
    bsd_data.set_index('lmfdb_label', inplace=True)

    # merge the data
    data = pd.merge(data, bsd_data, left_index=True, right_index=True, how='inner')

    # further filtering
    # condition 1h: ord_2(special_value/(real_period * regulator)) = -1
    data['bsd_lhs'] = np.log2(data['special_value']/(data['real_period'] * data['regulator'])).astype(np.float32)
    data['bsd_lhs'] = np.allclose(data['bsd_lhs'].values, np.ones(len(data)) * -1, atol=1e-5)   # allow a numerical tolerance

    # condition 1i: sha(2_isogenous_curve) = 0
    # first get the sha of the 2-isogenous curve
    # two_iso_ainvs = data['2_iso_ainvs'][0]
    # print(two_iso_ainvs)
    # curves_query = {'ainvs': two_iso_ainvs, 'semistable' : True}
    # two_iso_tab = ecq.search(curves_query, projection=['sha'], one_per=['lmfdb_iso'])
    # two_iso_sha = [curve['sha'] for curve in two_iso_tab]
    # data['2_iso_sha'] = two_iso_sha
    
    # print(data)

    # Export labels to a text file
    with open(output_file, 'w') as f:
        for label in data.index:
            f.write(f"{label}\n")

    print(f"SUCCESS!!! Data file saved to {output_file}")

print("Working...")

start_time = time.time()

foo(50)

end_time = time.time()
elapsed_time = end_time - start_time

minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print(f"Elapsed time: {minutes} minutes {seconds} seconds")