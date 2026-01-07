#!/usr/bin/env python3
import sys
import os
# Add repo root to path (go up 2 levels: infinite_bsd -> ants_xvii -> repo_root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
from lmfdb import db
from sage.all import ZZ, EllipticCurve, QQ, RR

def check_condition_1a(curve):
    N = ZZ(curve['conductor'])
    if N.is_squarefree() and not N.is_prime():
        return True
    else:
        return False

def check_condition_1b(E):
    return E.ap(3) in {-2, -1, 0, 1, 2}

def check_condition_1c(E):
    isogeny_primes = [phi.degree() for phi in E.isogenies_prime_degree()]
    assert 2 in isogeny_primes, "Isogeny degree 2 not found"  # this is because the curves have a rational 2 torsion point
    return (len(isogeny_primes) == 1)

def check_condition_1d(bad_primes, minimal_disc):
    return all(
        any(minimal_disc.valuation(p) % ell != 0 for p in bad_primes if p != ell)
        for ell in bad_primes
    )

def check_condition_1e(curve):
    return curve['optimality'] == 1

def check_condition_1f(curve):
    return curve['manin_constant'] % 2 == 1

def check_condition_1g(E):
    return E.two_torsion_rank() == 1

def check_condition_1h(curve):
    bsd_info = db.ec_mwbsd.lucky({'lmfdb_label' : curve['lmfdb_label']})
    val = bsd_info['special_value']/(bsd_info['real_period'] * curve['regulator'])
    val = QQ(RR(val))
    return val.valuation(2) == -1

def check_condition_1i(E):
    two_tors_pts = [x for x in E.torsion_points() if x.order() == 2]
    assert len(two_tors_pts) == 1
    E_two_torsion_gen = two_tors_pts[0]
    C = E(E_two_torsion_gen)
    E_prime = E.isogeny_codomain(C)
    try:
        E_prime_sha_order = E_prime.sha().an().round()  # the "round" is for the case where analytic rank is > 1
    except Exception as e:
        print(f"Error calculating sha for curve {curve['lmfdb_label']}: {e}")
        print("Getting sha value from db")
        try:
            E_prime_sha_order = db.ec_curvedata.lucky({'ainvs':[int(x) for x in E_prime.ainvs()]}, projection='sha')
        except:
            import pdb; pdb.set_trace()
    return (E_prime_sha_order == 1)


def main(Clabel: str):
    curve = db.ec_curvedata.lucky({'Clabel' : Clabel})
    E = EllipticCurve(curve['ainvs'])

    condition_1a = check_condition_1a(curve)
    condition_1b = check_condition_1b(E)
    condition_1c = check_condition_1c(E)
    condition_1d = check_condition_1d(curve['bad_primes'], curve['absD'])
    condition_1e = check_condition_1e(curve)
    condition_1f = check_condition_1f(curve)
    condition_1g = check_condition_1g(E)
    condition_1h = check_condition_1h(curve)
    condition_1i = check_condition_1i(E)

    print(f"Cremona label: {Clabel}")
    print(f"Condition 1a: {condition_1a}")
    print(f"Condition 1b: {condition_1b}")
    print(f"Condition 1c: {condition_1c}")
    print(f"Condition 1d: {condition_1d}")
    print(f"Condition 1e: {condition_1e}")
    print(f"Condition 1f: {condition_1f}")
    print(f"Condition 1g: {condition_1g}")
    print(f"Condition 1h: {condition_1h}")
    print(f"Condition 1i: {condition_1i}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("Clabel", help="Input string (no spaces)")
    args = parser.parse_args()
    main(args.Clabel)