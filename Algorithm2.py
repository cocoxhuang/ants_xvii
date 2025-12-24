from sage.all import EllipticCurve, primes, round, Integer, QQ, RR, kronecker, ZZ, gcd, prime_range, fundamental_discriminant, kronecker_symbol, NumberField
import copy
from lmfdb import db
import pandas as pd
import json

ecq = db.ec_curvedata

'''
Legacy implementation

def check_condition_2d(M, bad_primes_2):
    # we want to check that both 2 and 73 split in Q(sqrt(M))
    delta = M if M % 4 == 1 else 4*M
    return all([kronecker(delta,p) == 1 for p in bad_primes_2])

def get_admissible_twists_CLZ_v2(E,B):
    cond = E.conductor()
    bad_primes_2 = (2*cond).prime_divisors()
    possible_twists = [M for M in range(2,B) if Integer(M).is_squarefree() and M%3 != 0 and M%cond != 0]
    admissible_twists = []
    for M in possible_twists:
        condition_2d = check_condition_2d(M, bad_primes_2)
        if not condition_2d:
            continue
        prime_divs = ZZ(M).prime_divisors()
        condition_2b = True
        condition_2c = True
        for p in prime_divs:
            if p % 4 != 1:
                condition_2c = False
                break
            ap_val = E.ap(p)
            if ap_val % p == 0:
                condition_2b = False
                break
            N_p = p + 1 - ap_val
            if N_p.valuation(2) != 1:
                condition_2c = False
                break

        if all([condition_2b, condition_2c, condition_2d]):
            admissible_twists.append(M)
    return admissible_twists
'''

def get_admissible_twists_CLZ(E: EllipticCurve, B: int = 150) -> list:
    '''Given an elliptic curve E over Q, return the list of admissible BSD twists M up to bound B'''
    conductor = E.conductor()
    conductor_primes = ZZ(conductor).prime_divisors()
    a_p_dict = {}
    for p in prime_range(B):
        a_p = E.ap(p)
        a_p_dict[p] = a_p

    admissible_twists = []

    for M in range(2, B + 1):
        if ZZ(M).is_squarefree() and M != 0:  # Condition: squarefree
            if gcd(M, conductor) == 1: # Condition: bad primes do not divide ap
                primes_to_check = ZZ(M).prime_divisors()
                condition_b = all(a_p_dict[p] % p != 0 for p in primes_to_check)
                if condition_b:
                    # Condition: congruent to 1 mod 4 and N_p has 2-valuation 1
                    condition_c = all(((p % 4 == 1) and (a_p_dict[p].valuation(2) == 1)) for p in primes_to_check)
                    if condition_c:
                        # Condition: Kronecker symbol condition
                        if M % 8 == 1:
                            if all(kronecker_symbol(fundamental_discriminant(M),p) == 1 for p in conductor_primes if p != 2):
                                admissible_twists.append(M)
    
    return admissible_twists

def p_inert_in_F(p: int, F: NumberField) -> bool:
    '''Check if prime p is inert in number field F'''
    I = F.ideal(p)
    return I.is_prime()

def get_admissible_twists_Zhai(E: EllipticCurve, B: int = 150) -> list:
    '''Given an elliptic curve E over Q, return the list of admissible BSD twists M up to bound B'''
    conductor = E.conductor()
    conductor_primes = ZZ(conductor).prime_divisors()
    a_p_dict = {}
    for p in prime_range(B):
        a_p = E.ap(p)
        a_p_dict[p] = a_p

    admissible_twists = []

    for M in range(-B, B + 1):
        if E.discriminant() > 0 and M < 0:
            continue
        if ZZ(M).is_squarefree() and M != 0:  # Condition: squarefree
            if gcd(M, conductor) == 1: # Condition: bad primes do not divide ap
                primes_to_check = ZZ(M).prime_divisors()
                condition_b = all(a_p_dict[p] % p != 0 for p in primes_to_check)
                if condition_b:
                    condition_c = (M % 4 == 1)
                    if condition_c:
                        # Condition d: check that all primes dividing M are inert in Q(E[2])
                        f = E.two_division_polynomial()
                        F = NumberField(f, 'a')
                        condition_d = all(p_inert_in_F(p, F) for p in primes_to_check)
                        if condition_d:
                            # condition e: Kronecker symbol condition
                            if M % 8 == 1:
                                if all(kronecker_symbol(fundamental_discriminant(M),p) == 1 for p in conductor_primes if p != 2):
                                    admissible_twists.append(M)
        
    return admissible_twists

if __name__ == '__main__':
    B = 10000 # bound for admissible twists

    res = {}
    labels_path = 'data/lmfdb_labels.txt'
    with open(labels_path, 'r') as file:
        labels = [line.strip() for line in file.readlines()]

    for label in labels[5:]:
        label = label.split(', ')
        label, source = label[0], label[1]      
        ainvs = ecq.lookup(label, projection='ainvs')
        E = EllipticCurve(ainvs)
        if source == 'Zha16_no_2_tors':
            res[label] = get_admissible_twists_Zhai(E,B)
        elif source == 'CLZ20':
            res[label] = get_admissible_twists_CLZ(E,B)
        else:
            raise NotImplementedError(f"Source {source} not implemented yet.")

    with open('data/res.json', 'w') as f:
        json.dump(res, f, indent=4)
