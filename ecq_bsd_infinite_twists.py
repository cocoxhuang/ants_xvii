"""ecq_bsd_infinite_twists.py

This script generates a text file which includes elliptic curves with
a family of quadratic twists that satisfy the full BSD conjecture formula.

To run this, execute the following command in the terminal:

    sage -python ecq_bsd_infinite_twists.py

"""


from lmfdb import db
import pandas as pd
import numpy as np

from sage.all import EllipticCurve, primes_first_n, round, Integer, QQ, RR
import time

OUTPUT_FILE = 'data/output.txt'

# Load the elliptic curve table from the LMFDB
ecq = db.ec_curvedata
ec_classdata = db.ec_classdata
ec_mwbsd = db.ec_mwbsd

# Run a search query to get only the rank 0 or 1 curves.
# For now we set a limit to make things faster

NUM_AP_VALS = 100  # Number of primes to use for the a_p values
ECQ_COLS = ['ainvs', 'lmfdb_label', 'conductor', 'rank', 'torsion', 'absD', 'bad_primes', 'manin_constant', 'regulator', 'sha', 'lmfdb_iso', 'torsion_structure']
CLASSDATA_COLS = ['lmfdb_iso', 'aplist']
MWBSD_COLS = ['lmfdb_label', 'tamagawa_product'] + ['real_period', 'special_value',]

# def ap_normalized(E, p):
#     ap = E.ap(p)
#     normalization_quotient = 2 * p.sqrt()
#     return np.float32(round(ap / normalization_quotient, NUM_DECIMAL_PLACES))

# Function to get the data and labels
def filter_conditions_c_d_i(df):
    data = []
    for index,curve in df.iterrows():
        ainvs = curve['ainvs']
        minimal_disc = Integer(curve['absD'])
        bad_primes = curve['bad_primes']

        E = EllipticCurve(ainvs)
        isogeny_primes = [phi.degree() for phi in E.isogenies_prime_degree()]
        assert 2 in isogeny_primes, "Isogeny degree 2 not found"  # this is because the curves have a rational 2 torsion point
        condition_1c = (len(isogeny_primes) == 1)  # condition 1c: 2 is the only isogeny prime

        condition_1d = True  # condition 1d: bad primes do not dividie the order of minimal discriminant; initialize to True
        for p in bad_primes:
            ord_p_of_min_disc = minimal_disc.valuation(p)
            order_of_order = ord_p_of_min_disc.valuation(p)
            if order_of_order > 0:
                condition_1d = False
                break

        E_torsion_gens = E.torsion_subgroup().gens()
        assert len(E_torsion_gens) == 1
        P = E_torsion_gens[0]
        if P.order() == 2:
            E_two_torsion_gen = P
        elif P.order() == 4:
            E_two_torsion_gen = 2 * P
        elif P.order() == 6:
            E_two_torsion_gen = 3 * P
        elif P.order() == 8:
            E_two_torsion_gen = 4 * P
        elif P.order() == 10:
            E_two_torsion_gen = 5 * P
        else:
            raise ValueError("Unexpected torsion order")

        assert E_two_torsion_gen.order() == 2   # condition 1g: E(Q)[2] = Z/2Z (part 2)

        C = E(E_two_torsion_gen)
        E_prime = E.isogeny_codomain(C)
        E_prime_sha_order = E_prime.sha().an().round()  # the "round" is for the case where analytic rank is > 1
        condition_1i = (E_prime_sha_order == 1)

        conditions = [condition_1c, condition_1d, condition_1i]

        if all(conditions):
            data.append(curve['lmfdb_label'])

    return data

def foo(cond_bound=20):

    print(f"Generating data file for curves of conductor up to {cond_bound}...")
    output_file = OUTPUT_FILE  # .format(NUM_AP_VALS, 1, MY_LOCAL_LIM-1)
    ecq_query = {'conductor': {'$lt': cond_bound},
                'semistable' : True,  # condition 1a: semistable
                'optimality' : 1,  # condition 1e: E is optimal
                'manin_constant' : {'$mod': [2, 1]} # condition 1f: manin constant is odd
                # 'torsion_structure': {'$in': [[2], [4], [6], [8], [10], [12]]}  # condition 1g: E(Q)[2] = Z/2Z (part 1)
                }
    ecq_payload = ecq.search(ecq_query, projection=ECQ_COLS)
    df = pd.DataFrame(list(ecq_payload))
    assert df['lmfdb_iso'].nunique() == len(df), "Values in the 'lmfdb_iso' column are not unique!"

    allowed_torsion_structures = [[2], [4], [6], [8], [10], [12]]
    df = df[df['torsion_structure'].isin(allowed_torsion_structures)] # condition 1g: E(Q)[2] = Z/2Z (part 1)

    classdata_query = {'lmfdb_iso': {'$in': df['lmfdb_iso'].tolist()}}
    classdata_payload = ec_classdata.search(classdata_query, projection=CLASSDATA_COLS)
    classdata_df = pd.DataFrame(list(classdata_payload))
    classdata_df['a3'] = classdata_df['aplist'].apply(lambda x: x[1])
    classdata_df.drop(columns=['aplist'], inplace=True)

    lmfdb_iso_labels_zero_a3 = classdata_df[classdata_df['a3'] == 0]  # condition 1b: a3 = 0

    df = pd.merge(df, lmfdb_iso_labels_zero_a3, on='lmfdb_iso', how='inner')

    mwbsd_query = {'lmfdb_label': {'$in': df['lmfdb_label'].tolist()}}
    mwbsd_payload = ec_mwbsd.search(mwbsd_query, projection=MWBSD_COLS)
    mwbsd_df = pd.DataFrame(list(mwbsd_payload))

    df = pd.merge(df, mwbsd_df, on='lmfdb_label', how='inner')

    df['my_condition_1h_quantity'] = (df['tamagawa_product'] * df['sha'].round()) / (df['torsion']**2)
    df['my_condition_1h_quantity'] = df['my_condition_1h_quantity'].apply(lambda x : QQ(x).valuation(2))
    # df = df[df['my_condition_1h_quantity'] == -1]  # condition 1h: 2-ord of special_value/(real_period*regulator) = -1
    df['condition_1h_quantity'] = (df['special_value']/(df['real_period'] * df['regulator']))
    df['condition_1h_quantity'] = df['condition_1h_quantity'].apply(lambda x: QQ(RR(x)).valuation(2))
    # df = df[df['condition_1h_quantity'] == -1] # condition 1h: 2-ord of special_value/(real_period*regulator) = -1

    assert df['my_condition_1h_quantity'].equals(df['condition_1h_quantity']), \
    "The values in 'my_condition_1h_quantity' and 'condition_1h_quantity' are not identical!"

    df = df[df['condition_1h_quantity'] == -1]  # condition 1h: 2-ord of special_value/(real_period*regulator) = -1

    labels = filter_conditions_c_d_i(df)
    # final_labels = []
    # for label in labels:
    #     print(f"Checking curve {label}...")
    #     tamagawa_product = ec_mwbsd.lookup(label, projection='tamagawa_product')
    #     sha = ecq.lookup(label, projection='sha')
    #     torsion_order = ecq.lookup(label, projection='torsion')
    #     the_quantity = QQ((tamagawa_product * sha) / (torsion_order)**2).valuation(2)
    #     if the_quantity == -1:
    #         final_labels.append(label)

    # Export labels to a text file
    with open(output_file, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")

    print(f"SUCCESS!!! Data file saved to {output_file}.")

    final_cremona_labels = []
    for label in labels:
        c_label = ecq.lookup(label, projection='Clabel')
        final_cremona_labels.append(c_label)
    print(f"The labels are {final_cremona_labels}.")

print("Working...")

start_time = time.time()

foo(1000)

end_time = time.time()
elapsed_time = end_time - start_time

minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print(f"Elapsed time: {minutes} minutes {seconds} seconds")
