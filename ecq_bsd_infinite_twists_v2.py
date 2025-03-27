"""ecq_bsd_infinite_twists.py

This script generates a text file which includes elliptic curves with
a family of quadratic twists that satisfy the full BSD conjecture formula.

To run this, execute the following command in the terminal:

    sage -python ecq_bsd_infinite_twists_v2.py

"""


from lmfdb import db
import pandas as pd
import numpy as np
import pytz
from sage.all import EllipticCurve, primes_first_n, round, Integer, QQ, RR
import time
from datetime import datetime

OUTPUT_FILE = 'data/output_v2.txt'

# Load the elliptic curve table from the LMFDB
ecq = db.ec_curvedata
ec_classdata = db.ec_classdata
ec_mwbsd = db.ec_mwbsd

# Run a search query to get only the rank 0 or 1 curves.
# For now we set a limit to make things faster

NUM_AP_VALS = 100  # Number of primes to use for the a_p values
ECQ_COLS = ['ainvs', 'lmfdb_label', 'conductor', 'rank', 'torsion','torsion_primes', 'absD', 'bad_primes', 'manin_constant', 'regulator', 'sha', 'lmfdb_iso']
CLASSDATA_COLS = ['lmfdb_iso', 'aplist']
MWBSD_COLS = ['lmfdb_label', 'tamagawa_product'] + ['real_period', 'special_value']


def get_current_time_str():
    # Get the current time in UTC
    utc_now = datetime.now(pytz.utc)

    # Convert UTC time to US Eastern Time
    eastern = pytz.timezone('US/Eastern')
    eastern_now = utc_now.astimezone(eastern)

    # Format the time in a human-readable format
    return eastern_now.strftime("Generated at %H:%M (eastern) on %A %d %B %Y")

# Function to get the data and labels
def filter_conditions_c_d(df):
    data = []
    for index,curve in df.iterrows():
        ainvs = curve['ainvs']
        minimal_disc = Integer(curve['absD'])
        bad_primes = curve['bad_primes']

        condition_1c = True
        E = EllipticCurve(ainvs)
        isogeny_primes = [phi.degree() for phi in E.isogenies_prime_degree()]
        condition_1c = (isogeny_primes == [2] or isogeny_primes == [])  # condition 1c: 2 is the only isogeny prime

        condition_1d = True  # condition 1d: bad primes do not dividie the order of minimal discriminant; initialize to True
        for p in bad_primes:
            ord_p_of_min_disc = minimal_disc.valuation(p)
            order_of_order = ord_p_of_min_disc.valuation(p)
            if order_of_order > 0:
                condition_1d = False
                break

        conditions = [condition_1c, condition_1d]

        if all(conditions):
            data.append(curve['lmfdb_label'])

    return data



def foo(cond_upper_bound=None, cond_lower_bound=None):

    start_time = time.time()
    
    run_summary = f"Params: cond_upper_bound={cond_upper_bound}, cond_lower_bound={cond_lower_bound}"
    print(run_summary)

    output_file = OUTPUT_FILE  # .format(NUM_AP_VALS, 1, MY_LOCAL_LIM-1)
    ecq_query = {
                'semistable' : True,  # condition 1a: semistable
                'optimality' : 1,  # condition 1e: E is optimal
                'manin_constant' : {'$mod': [2, 1]}, # condition 1f: manin constant is odd
                'signD' : {'$lt': 0},  # condition 1g: sign of the discriminant is negative
                }
    if cond_upper_bound is not None:
        if cond_lower_bound is not None:
            ecq_query['conductor'] = {'$gte' : cond_lower_bound,  '$lt': cond_upper_bound}
        else:
            ecq_query['conductor'] = {'$lt': cond_upper_bound}
    else:
        if cond_lower_bound is not None:
            ecq_query['conductor'] = {'$gte' : cond_lower_bound}

    print(f"ecq_query={ecq_query}")
    ecq_payload = ecq.search(ecq_query, projection=ECQ_COLS)
    df = pd.DataFrame(list(ecq_payload))
    assert df['lmfdb_iso'].nunique() == len(df), "Values in the 'lmfdb_iso' column are not unique!"

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

    df = df[~df['torsion_primes'].apply(lambda x: 2 in x)]  # condition 1h: E[2](Q) = 0

    df['real_components'] = df['ainvs'].apply(lambda x: EllipticCurve(x).real_components())  
    df['my_condition_1i_quantity'] = (df['tamagawa_product'] * df['sha'].round()) / (df['torsion']**2 * df['real_components'])
    # df['my_condition_1i_quantity'] = df['my_condition_1i_quantity'] * df['regulator']
    df['my_condition_1i_quantity'] = df['my_condition_1i_quantity'].apply(lambda x : QQ(RR(x)).valuation(2))
    # df = df[df['my_condition_1i_quantity'] == -1]  # condition 1i: 2-ord of special_value/(real_period+ * regulator) = 0
    df['condition_1i_quantity'] = (df['special_value']/(df['real_period'] * df['regulator'] * df['real_components']))
    # df['condition_1i_quantity'] = df['condition_1i_quantity'] * df['regulator']
    df['condition_1i_quantity'] = df['condition_1i_quantity'].apply(lambda x: QQ(RR(x)).valuation(2))
    # df = df[df['condition_1i_quantity'] == -1] # condition 1i: 2-ord of special_value/(real_period+ * regulator) = 0

    assert df['my_condition_1i_quantity'].equals(df['condition_1i_quantity']), \
    "The values in 'my_condition_1i_quantity' and 'condition_1i_quantity' are not identical!"

    df = df[df['condition_1i_quantity'] == 0]  # condition 1i: 2-ord of special_value/(real_period+ *regulator) = 0

    labels = filter_conditions_c_d(df)
    
    current_time = get_current_time_str()
    end_time = time.time()
    elapsed_time = end_time - start_time

    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    with open(output_file, 'w') as f:
        # Write the timestamp at the top of the file
        f.write(f"{current_time}\n")
        f.write(f"{run_summary}\n")
        f.write(f"Run took: {minutes} minutes {seconds} seconds\n\n")
        for label in labels:
            f.write(f"{label}\n")
    
    print(f"SUCCESS!!! Data file saved to {output_file}.")

foo()

