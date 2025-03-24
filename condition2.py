from sage.all import EllipticCurve, primes, round, Integer, QQ, RR, kronecker
import copy
from lmfdb import db
import pandas as pd
import json


def squarefree_integers(M_upper_bound):
    '''
    Returns a dictionary of squarefree integers up to M_upper_bound.
    The keys are the squarefree integers and the values are lists of their prime factors: e.g.
        {M (squarefree): [list of prime factors]}
    '''
    squarefree = {}
    for M in range(2, M_upper_bound + 1):
        factors = Integer(M).factor()
        if all(exp == 1 for _, exp in factors):
            squarefree[M] = [factor for factor, _ in factors]
            squarefree[-1*M] = [factor for factor, _ in factors]
    return squarefree

def get_E_data(labels):
    '''
    returns a dictionary of elliptic curves of the format: e.g.
        {'lmfdb_label': {p: a_p, 'conductor': conductor of E}}
    '''
    res = {}

    # Load the elliptic curve table from the LMFDB
    ecq = db.ec_curvedata
    ec_classdata = db.ec_classdata
    ec_mwbsd = db.ec_mwbsd

     # create a dictionary of {'lmfdb_label' : {p : a_p, 'conductor': conductor}}}
    ecq_payload = ecq.search({"lmfdb_label": {"$in": labels}}, projection=['lmfdb_label','conductor','lmfdb_iso'])
    df_ecq = pd.DataFrame(list(ecq_payload))
    
    classdata_payload = ec_classdata.search({'lmfdb_iso': {'$in': df_ecq['lmfdb_iso'].tolist()}}, projection=['lmfdb_iso','aplist'])
    df_class = pd.DataFrame(list(classdata_payload))
    # seperate the aplist into individual columns
    first_100_primes = list(primes(100))
    df_class = df_class.join(pd.DataFrame(df_class['aplist'].tolist(), columns=first_100_primes))
    df_class = df_class.drop(columns=['aplist'])

    # merge the two dataframes on 'lmfdb_iso'
    merged_df = pd.merge(df_ecq, df_class, on='lmfdb_iso', how='inner')

    # create the final dictionary
    for index, row in merged_df.iterrows():
        lmfdb_label = row['lmfdb_label']
        conductor = row['conductor']
        a_p_dict = {p: row[p] for p in first_100_primes}
        res[lmfdb_label] = {**a_p_dict, 'conductor': conductor}
    return res

def generate_Ms(E_data, Ms):
    '''
    Input: 
    E_data: a dictionary of data of elliptic curves of the format: e.g.
        {p: a_p, 'conductor': conductor of E}
    Ms: a dictionary of squarefree integers and their prime factors: e.g.
        {M: [list of prime factors of M]}
    Output: a list of quadratic twists of the elliptic curves in E_list
    '''
    res = []

    bad_primes = set([factor for factor, _  in Integer(E_data['conductor']).factor()])
    bad_primes_2 = copy.deepcopy(bad_primes)
    bad_primes_2.add(2)

    for M in Ms:
        M_div = Ms[M]   # list of prime factors of M

        # condition 2(b): all M divisors do not divide ap
        condition2b = True
        for p in M_div:
            if p in bad_primes:
                condition2b = False # (M,N) = 1
                break
            if E_data[p] % p == 0:  # p does not divide a_p
                condition2b = False
                break
        if not condition2b:
            # print(M, "condition 2(b) not satisfied")
            continue

        # condition 2(c): each prime factor q of M is in the set S:
        # q  = 1 mod 4, and q does not divide conductor(E), and ord_2(N_q) = 1
        condition2c = True
        for q in M_div:
            N_q = 1 + q - E_data[q]
            if q % 4 != 1 or q in bad_primes or Integer(N_q).valuation(2) != 1:
            # if Integer(N_q).valuation(2) != 1:
                condition2c = False
                break
        if not condition2c:
            # print(M, "condition 2(c) not satisfied")
            continue
        
        # condition 2(d): bad primes and 2 all split in Q(sqrt(M))
        condition2d = True
        for p in bad_primes_2:
            a = M if M % 4 == 1 else 4*M
            if kronecker(a,p) != 1:
                condition2d = False
                break
        if not condition2d:
            # print(M, "condition 2(d) not satisfied")
            continue

        res.append(M)   # if all are satisfied, add M to the list

    return res

def generate_BSD_quadratic_twists(E_list, M_upper_bound = 100):
    '''
    Input: a dictionary of elliptic curves E_list and an upper bound M_upper_bound
    E_list: a dictionary of elliptic curves of the format 
        {label: {p: a_p, 'conductor': conductor of E} }
    Output: a list of quadratic twists of the elliptic curves in E_list
    '''
    res = {}

    Ms = squarefree_integers(M_upper_bound)  # condition 2(i): square free

    for E, E_data in E_list.items():
        res[E] = generate_Ms(E_data, Ms)
    return res

labels_path = 'data/output.txt'
with open(labels_path, 'r') as file:
    labels = [line.strip() for line in file.readlines()]

E_list = get_E_data(labels)
res = generate_BSD_quadratic_twists(E_list, M_upper_bound = 100)
# dump the res to a json file
with open('data/res.json', 'w') as f:
    json.dump(res, f)