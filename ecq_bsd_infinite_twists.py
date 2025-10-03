"""ecq_bsd_infinite_twists.py

This script generates a text file which includes elliptic curves with
a family of quadratic twists that satisfy the full BSD conjecture formula.

To run this, execute the following command in the terminal:

    sage -python ecq_bsd_infinite_twists.py

"""


from lmfdb import db
import pandas as pd
import numpy as np
import pytz
from sage.all import EllipticCurve, primes_first_n, round, Integer, QQ, RR, ZZ
import time
from datetime import datetime

LMFDB_FILE = 'data/lmfdb_labels.txt'
CREMONA_FILE = 'data/cremona_labels.txt'

# Load the elliptic curve table from the LMFDB
ecq = db.ec_curvedata
ec_classdata = db.ec_classdata
ec_mwbsd = db.ec_mwbsd

# Run a search query to get only the rank 0 or 1 curves.
# For now we set a limit to make things faster

ECQ_COLS = ['ainvs', 'lmfdb_label', 'conductor', 'rank', 'torsion', 'absD', 'bad_primes', 'manin_constant', 
'regulator', 'sha', 'lmfdb_iso', 'torsion_structure', 'torsion_primes', 'signD']
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


def is_ramified(bad_primes, minimal_disc):
    return all(
        any(minimal_disc.valuation(p) % ell != 0 for p in bad_primes if p != ell)
        for ell in bad_primes
    )

def ordinary_357(a_ps):
    '''
    returns the ordinary primes among 3,5,7
    '''
    primes = [3,5,7]
    return [primes[i] for i in range(len(primes)) if a_ps[i] % primes[i] != 0]

def get_a3_a5_a7(df):
    classdata_query = {'lmfdb_iso': {'$in': df['lmfdb_iso'].tolist()}}
    classdata_payload = ec_classdata.search(classdata_query, projection=CLASSDATA_COLS)
    classdata_df = pd.DataFrame(list(classdata_payload))
    classdata_df['a3'] = classdata_df['aplist'].apply(lambda x: x[1])
    classdata_df['a5'] = classdata_df['aplist'].apply(lambda x: x[2])
    classdata_df['a7'] = classdata_df['aplist'].apply(lambda x: x[3])
    classdata_df.drop(columns=['aplist'], inplace=True)
    return classdata_df

def merge_mwbsd(df):
    mwbsd_query = {'lmfdb_label': {'$in': df['lmfdb_label'].tolist()}}
    mwbsd_payload = ec_mwbsd.search(mwbsd_query, projection=MWBSD_COLS)
    mwbsd_df = pd.DataFrame(list(mwbsd_payload))
    df = pd.merge(df, mwbsd_df, on='lmfdb_label', how='inner')
    return df

# Function to get the data and labels
def filter_CONDITIONs_c_d(df):
    data_idx = []
    for index,curve in df.iterrows():
        ainvs = curve['ainvs']
        minimal_disc = Integer(curve['absD'])
        bad_primes = curve['bad_primes']

        # CONDITION 1c: irreducibility CONDITION at certain odd primes
        CONDITION_1c = True
        a_3, a_5, a_7 = curve['a3'], curve['a5'], curve['a7']
        if 2 in bad_primes:
            non_isogeny_primes = bad_primes.copy()
            non_isogeny_primes.remove(2)
            non_isogeny_primes = set(non_isogeny_primes)
        else:
            non_isogeny_primes = set(bad_primes)
        non_isogeny_primes.update(ordinary_357([a_3, a_5, a_7]))
    
        E = EllipticCurve(ainvs)
        isogeny_primes = [phi.degree() for phi in E.isogenies_prime_degree()]
        CONDITION_1c = (all(p not in isogeny_primes for p in non_isogeny_primes))

        # condititon 1d: ramification CONDITION at bad primes
        CONDITION_1d = is_ramified(bad_primes, minimal_disc)

        CONDITIONs = [CONDITION_1c, CONDITION_1d]

        if all(CONDITIONs):
            data_idx.append(index)

    return df.loc[data_idx]

# CONDITION 1g: E(Q)[2] = Z/2Z
def filter_CONDITIONs_g(df):
    allowed_torsion_structures = [[2], [4], [6], [8], [10], [12]]
    df = df[df['torsion_structure'].isin(allowed_torsion_structures)] 
    data_idx = []
    for index,curve in df.iterrows():
        ainvs = curve['ainvs']
        E = EllipticCurve(ainvs)
        E_torsion_gens = E.torsion_subgroup().gens()
        if len(E_torsion_gens) == 1:
            data_idx.append(index)
    return df.loc[data_idx]

def filter_CONDITIONs_i(df):
    data_idx = []
    for index,curve in df.iterrows():
        ainvs = curve['ainvs']
        E = EllipticCurve(ainvs)
        # find the generator of the 2-torsion subgroup
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
        assert E_two_torsion_gen.order() == 2   
        # construct E' and retrieve the order of sha(E')
        C = E(E_two_torsion_gen)
        E_prime = E.isogeny_codomain(C)
        try:
            E_prime_sha_order = E_prime.sha().an().round()  # the "round" is for the case where analytic rank is > 1
        except Exception as e:
            print(f"Error calculating sha for curve {curve['lmfdb_label']}: {e}")
            print("Getting sha value from db")
            try:
                E_prime_sha_order = ecq.lucky({'ainvs':[int(x) for x in E_prime.ainvs()]}, projection='sha')
            except:
                import pdb; pdb.set_trace()
        '''
        revision: now require E_prime[2] is also Z/2
        if E_prime_sha_order == 1:
            data_idx.append(index)
        '''
        if E_prime_sha_order == 1:
            # extra requirement of E'(Q)[2] being isomorphic to Z/2Z
            n_gen = E_prime.torsion_subgroup().ngens()
            E_prime_tors_order = E_prime.torsion_order()
            if n_gen == 1 and E_prime_tors_order % 2 == 0:
                data_idx.append(index)
    return df.loc[data_idx]

def ord_2_two_torsion_order(torsion_structure, torsion):
    '''returns ord_2( the order of the 2-torsion subgroup)'''
    if len(torsion_structure) == 1:
        two_torsion_val_2 = min(ZZ(torsion).valuation(2), 1)
    else:
        two_torsion_val_2 = ZZ(torsion_structure[0]).valuation(2) + ZZ(torsion_structure[1]).valuation(2)
    return two_torsion_val_2

def foo(cond_upper_bound=None, cond_lower_bound=None):

    start_time = time.time()
    run_summary = f"Params: cond_upper_bound={cond_upper_bound}, cond_lower_bound={cond_lower_bound}"
    lmfdb_file = LMFDB_FILE  # .format(NUM_AP_VALS, 1, MY_LOCAL_LIM-1)
    cremona_file = CREMONA_FILE
    
    '''
    2-part of the BSD verification references
        [CLZ20] : L. Cai, C. Li and S. Zhai, On the 2-part of the Birch and Swinnerton-Dyer conjecture for quadratic twists of elliptic
    curves, J. Lond. Math. Soc. (2) 101 (2020), no. 2, 714–734.
        [Zha16] :  S. Zhai, Non-vanishing theorems for quadratic twists of elliptic curves, Asian J. Math. 20 (2016), no. 3, 475–502
    '''

    # get curve data from the LMFDB
    # checking CONDITIONs 1a, 1d, 1e, 1f
    ecq_query = {
                'semistable' : True,  # CONDITION 1a: semistable
                'num_bad_primes' : {'$gte' : 2},  # CONDITION 1d: at least two bad prime
                'optimality' : 1,  # CONDITION 1e: E is optimal
                'manin_constant' : {'$mod': [1, 2]} # CONDITION 1f: manin constant is odd, using psycodict's ordering on '$mod
                }
    if cond_upper_bound is not None:
        if cond_lower_bound is not None:
            ecq_query['conductor'] = {'$gte' : cond_lower_bound,  '$lt': cond_upper_bound}
        else:
            ecq_query['conductor'] = {'$lt': cond_upper_bound}
    else:
        if cond_lower_bound is not None:
            ecq_query['conductor'] = {'$gte' : cond_lower_bound}
    ecq_payload = ecq.search(ecq_query, projection=ECQ_COLS)
    df = pd.DataFrame(list(ecq_payload))
    assert df['lmfdb_iso'].nunique() == len(df), "Values in the 'lmfdb_iso' column are not unique!"

    # CONDITION 1b: a3 in {-2, -1, 0, 1, 2}
    classdata_df = get_a3_a5_a7(df)
    lmfdb_iso_labels_a3_cond = classdata_df[classdata_df['a3'].isin({-2, -1, 0, 1, 2})]
    df = pd.merge(df, lmfdb_iso_labels_a3_cond, on='lmfdb_iso', how='inner')

    # CONDITION 1c: 2 is the only isogeny prime
    # CONDITION 1d: ramification CONDITION at bad primes
    df = filter_CONDITIONs_c_d(df)

    # checking the 2-part of BSD
    # --------------
    # using criteria from [CLZ20, Theorem 1.5]

    # CONDITION 1g: E(Q)[2] = Z/2Z
    df_CLZ20 = filter_CONDITIONs_g(df)
    
    # CONDITION 1h: 2-ord of special_value/(real_period*regulator) = -1
    df_CLZ20 = merge_mwbsd(df_CLZ20)
    df_CLZ20['L_alg'] = (df_CLZ20['special_value']/(df_CLZ20['real_period'] * df_CLZ20['regulator']))
    df_CLZ20['L_alg'] = df_CLZ20['L_alg'].apply(lambda x: QQ(RR(x)).valuation(2))    # now L_alg is 2-valuation of the quantity
    df_CLZ20 = df_CLZ20[df_CLZ20['L_alg'] == -1]  

    # condtion 1i: sha(E') = 1
    df_CLZ20 = filter_CONDITIONs_i(df_CLZ20)

    '''
    revision:
        extra conditions for CLZ20: rank = 0
    '''
    df_CLZ20 = df_CLZ20[df_CLZ20['rank'] == 0]

    df_CLZ20['source'] = 'CLZ20'

    # --------------
    # using criteria from [Zha16, Theorem 1.1 - 1.4]
    # --------------
    # CONDITION 1g: 
    df_Zha16 = df
    df_Zha16 = merge_mwbsd(df_Zha16)
    df_Zha16['real_components'] = df_Zha16['ainvs'].apply(lambda x: EllipticCurve(x).real_components())  
    df_Zha16['L_alg'] = (df_Zha16['special_value'] * df_Zha16['real_components']/(df_Zha16['real_period']))
    df_Zha16['L_alg_ord_2'] = df_Zha16['L_alg'].apply(lambda x: QQ(RR(x)).valuation(2))
    
    # part 1: no 2 torsion & ( CONDITION on L_alg based on signD )
    df_Zha16_no_2_tors = df_Zha16[~df_Zha16['torsion_primes'].apply(lambda x: 2 in x)]   
    df_Zha16_no_2_tors = df_Zha16_no_2_tors.loc[( (df_Zha16_no_2_tors['L_alg_ord_2'] == 0) & (df_Zha16_no_2_tors['signD'] < 0) )| ( (df_Zha16_no_2_tors['L_alg_ord_2'] == 1) & (df_Zha16_no_2_tors['signD'] > 0) )]

    # part 2: 2 torsion & L value conditions
    df_Zha16_2_tors = df_Zha16[df_Zha16['torsion_primes'].apply(lambda x: 2 in x)]
    
    df_Zha16_2_tors = df_Zha16_2_tors[df_Zha16_2_tors['rank'] == 0]      # rank = 0
    # df_Zha16_2_tors = df_Zha16_2_tors[df_Zha16_2_tors['sha'] % 2 != 0]   # sha is odd -> Sel_2(E) = 1

    # L value conditions
    df_Zha16_2_tors = df_Zha16_2_tors.loc[( (df_Zha16_2_tors['L_alg_ord_2'] != 0) & (df_Zha16_2_tors['signD'] < 0) )| ( (df_Zha16_2_tors['L_alg_ord_2'] != 1) & (df_Zha16_2_tors['signD'] > 0) )]
    
    # speculations
    df_Zha16_2_tors = df_Zha16_2_tors[df_Zha16_2_tors['torsion'] > 2] 

    # combine the two
    df_Zha16 = pd.concat([df_Zha16_no_2_tors, df_Zha16_2_tors])

    # give the source of the data
    df_Zha16['source'] = 'Zha16'
    # --------------

    # combine the results from CLZ20 and Zha16
    df = pd.concat([df_CLZ20, df_Zha16]).sort_values(by=['source','conductor']).reset_index(drop=True)
    df = df[['source','conductor','lmfdb_label']].drop_duplicates(subset=['lmfdb_label'])
    labels = df['lmfdb_label'].tolist()
    sources = df['source'].tolist()
    res = dict(zip(labels, sources))

    # calculate the run time
    current_time = get_current_time_str()
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    # save lmfdb labels
    with open(lmfdb_file, 'w') as f:
        # Write the timestamp at the top of the file
        f.write(f"{current_time}\n")
        f.write(f"{run_summary}\n")
        f.write(f"Run took: {minutes} minutes {seconds} seconds\n\n")
        for label, source in res.items():
            f.write(f"{label}, {source}\n")

    # save cremona labels
    final_cremona_labels = []
    for label in labels:
        c_label = ecq.lookup(label, projection='Clabel')
        final_cremona_labels.append(c_label)
    cremona_res = dict(zip(final_cremona_labels, sources))
    with open(cremona_file, 'w') as f:
        # Write the timestamp at the top of the file
        f.write(f"{current_time}\n")
        f.write(f"{run_summary}\n")
        f.write(f"Run took: {minutes} minutes {seconds} seconds\n\n")
        for label, source in cremona_res.items():
            f.write(f"{label}, {source}\n")
    print(f"SUCCESS!!! Data files saved to {lmfdb_file} and {cremona_file}.")

foo(cond_upper_bound=150)