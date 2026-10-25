#!/usr/bin/env python

from typing import (
    Dict,
    List,
    Tuple,
    Set,
    Union,
    Generator,
    Callable,
    Optional,
    Any,
    Hashable,
    Iterable,
)

import bisect
import functools
import heapq
import itertools
import math
import numpy as np
import os
import random
import sys
import time

from collections import deque, defaultdict
from sortedcontainers import SortedDict, SortedList, SortedSet
from gmpy2 import mpfr

from data_structures.fractions import CustomFraction
from data_structures.prime_sieves import PrimeSPFsieve, SimplePrimeSieve
from data_structures.fenwick_tree import FenwickTree

from algorithms.number_theory_algorithms import gcd, lcm, isqrt, integerNthRoot, solveLinearCongruence, extendedEuclideanAlgorithm, solveLinearNonHomogeneousDiophantineEquation, floorHarmonicSeries
from algorithms.pseudorandom_number_generators import blumBlumShubPseudoRandomGenerator
from algorithms.continued_fractions_and_Pell_equations import pellSolutionGenerator, generalisedPellSolutionGenerator, pellFundamentalSolution
from algorithms.Pythagorean_triple_generators import pythagoreanTripleGeneratorByHypotenuse
from algorithms.string_searching_algorithms import KnuthMorrisPratt, rollingHashWithValue


def calculatePrimeFactorisation(
    num: int,
    ps: Optional[PrimeSPFsieve]=None,
) -> Dict[int, int]:
    """
    For a strictly positive integer, calculates its prime
    factorisation.

    This is performed using direct division.

    Args:
        Required positional:
        num (int): The strictly positive integer whose prime
                factorisation is to be calculated.

        Optional named:
        ps (PrimeSPFsieve object or None): If given, a smallest
                prime factor prime sieve object used to calculate
                the prime factorisation. If given as None, the
                factorisation is performed through direct division.
    
    Returns:
    Dictionary (dict) giving the prime factorisation of num, whose
    keys are strictly positive integers (int) giving the prime
    numbers that appear in the prime factorisation of num, with the
    corresponding value being a strictly positive integer (int)
    giving the number of times that prime appears in the
    factorisation (i.e. the power of that prime in the prime
    factorisation of the factor num). An empty dictionary is
    returned if and only if num is the multiplicative identity
    (i.e. 1).
    """
    if ps is not None:
        return ps.primeFactorisation(num)
    exp = 0
    while not num & 1:
        num >>= 1
        exp += 1
    res = {2: exp} if exp else {}
    for p in range(3, num, 2):
        if p ** 2 > num: break
        exp = 0
        while not num % p:
            num //= p
            exp += 1
        if exp: res[p] = exp
    if num > 1:
        res[num] = 1
    return res

# Problem 351
def eulerTotientSum(m: int, ps: Optional[PrimeSPFsieve]=None) -> int:
    # Using the identity:
    # phi_cumu(n) = n * (n + 1) / 2 - sum(d = 2 to n) phi_cumu(floor(n / d))
    m_twothirds_floor = integerNthRoot(m * m, 3)
    phi = [0, 1]
    phi_cumu = [0, 1]
    for num in range(2, m_twothirds_floor + 1):
        pf = calculatePrimeFactorisation(num, ps=ps)
        totient = 1
        for p, f in pf.items():
            totient *= (p - 1) * p ** (f - 1)
        phi.append(totient)
        phi_cumu.append(phi_cumu[-1] + phi[-1])
    memo = {}
    def recur(num: int) -> int:
        if num <= m_twothirds_floor:
            return phi_cumu[num]
        elif num in memo.keys():
            return memo[num]
        res = (num * (num + 1)) >> 1
        i = 2
        while i < num:
            q = num // i
            r = num // q
            i2 = min(r, num)
            res -= (i2 - i + 1) * recur(q)
            i = i2 + 1
        memo[num] = res
        return res
    res = recur(m)
    #print(memo)
    return res

def hiddenPointsInTriangularLatticeHexagon(
    hexagon_side_length: int=10 ** 8,
    ps: Optional[PrimeSPFsieve]=None,
) -> int:
    """
    Solution to Project Euler #351
    """
    n = hexagon_side_length
    res = 3 * (n) * (n + 1)
    if ps is not None: ps.extendSieve(n)
    #print("hello", res)
    #for num in range(1, n + 1):
    #    #pf = calculatePrimeFactorisation(num, ps=ps)
    #    #totient = 1
    #    #for p, f in pf.items():
    #    #    totient *= (p - 1) * p ** (f - 1)
    #    res -= 6 * totient
    #    #print(num, res)
    ets = eulerTotientSum(n, ps=ps)
    #print(n, ets)
    return res - 6 * ets

# Problem 352
def bloodTestOptimalStrategyMeanTestCountTopDown(n_subjects: int, p_infected: int) -> float:

    memo = {}
    def recur(n_subjects: int, known_contains_infected: bool) -> float:
        if not n_subjects: return 0.
        if n_subjects == 1:
            return float(not known_contains_infected)
        args = (n_subjects, known_contains_infected)
        if args in memo.keys():
            return memo[args]
        #p2 = p_infected / (1 - (1 - p_infected) ** n_subjects) if known_contains_infected else p_infected
        res = float("inf")
        for n_select in range(1, n_subjects + (not known_contains_infected)):
            p_pos = (1 - (1 - p_infected) ** n_select) / (1 - (1 - p_infected) ** n_subjects) if known_contains_infected else 1 - (1 - p_infected) ** n_select
            ans = 1 + p_pos * (recur(n_select, True) + recur(n_subjects - n_select, False)) + (1 - p_pos) * recur(n_subjects - n_select, known_contains_infected)
            res = min(res, ans)
        memo[args] = res
        return res
    
    res = recur(n_subjects, False)
    #print(memo)
    return res

def bloodTestOptimalStrategyMeanTestCountBottomUp(
    n_subjects: int,
    p_infected: int,
) -> float:
    # TODO- write version of this using fractions rather than floats
    dp = [[float("inf"), float("inf")] for _ in range(n_subjects + 1)]
    
    dp[0] = [0., 0.]
    if n_subjects >= 1:
        dp[1] = [1., 0.]

    for num in range(2, n_subjects + 1):
        #print(num)
        #dp.append([float("inf"), float("inf")])
        for n_select in range(1, num):
            #print(f"n_select = {n_select}")
            p_pos = (1 - (1 - p_infected) ** n_select) / (1 - (1 - p_infected) ** num)
            #print(p_pos)
            #print((dp[n_select][1] + dp[num - n_select][0]), (1 - p_pos) * dp[num - n_select][1])
            dp[num][1] = min(dp[num][1], 1 + p_pos * (dp[n_select][1] + dp[num - n_select][0]) + (1 - p_pos) * dp[num - n_select][1])
        for n_select in range(1, num + 1):
            p_pos = 1 - (1 - p_infected) ** n_select
            dp[num][0] = min(dp[num][0], 1 + p_pos * (dp[n_select][1] + dp[num - n_select][0]) + (1 - p_pos) * dp[num - n_select][0])
    #print(dp)
    return dp[n_subjects][0]

def bloodTestOptimalStrategyMeanTestCountSum(
    n_subjects: int=10 ** 4,
    p_infected_vals: Iterable[int]=(.01 * x for x in range(1, 51)),
) -> float:
    """
    Solution to Project Euler #352
    """
    res = 0.
    for p in p_infected_vals:
        ans = bloodTestOptimalStrategyMeanTestCountBottomUp(n_subjects, p)
        print(n_subjects, p, ans)
        res += ans
    return res

# Problem 353
def calculateMoonPathMinimumRisk(r: int) -> float:
    r_sq = r * r
    r_dbl_sq = r_sq << 2

    def arcNormalisedDistance(start: tuple[int, int, int], end: tuple[int, int, int]) -> float:
        # Note this assumes that start and end are both exactly
        # r away from the origin
        #
        dot_prod = sum(x * y for x, y in zip(start, end))
        if abs(dot_prod) > r_sq:
            raise ValueError("The points are too far apart to both be on the surface of the sphere")
        return math.acos(dot_prod / r_sq) / math.pi
        """
        d_sq = sum((x - y) ** 2 for x, y in zip(start, end))
        if d_sq > r_dbl_sq:
            raise ValueError("The points are too far apart to both be on the surface of the sphere")
        return (2 * math.asin(math.sqrt(d_sq / r_dbl_sq)) / math.pi)
        """

    def arcRisk(start: tuple[int, int, int], end: tuple[int, int, int]) -> float:
        # Note this assumes that start and end are both exactly
        # r away from the origin
        arc_dist = arcNormalisedDistance(start, end)
        return arc_dist * arc_dist

    arc_dist_lb = (2 * math.asin(math.sqrt(2 / r_dbl_sq)) / math.pi)
    arc_dist_lb_risk = arc_dist_lb * arc_dist_lb

    def heuristic(pt: tuple[int, int, int]) -> float:
        d = arcNormalisedDistance(pt, (0, 0, -r))
        m = math.ceil(d / arc_dist_lb)
        return m * arc_dist_lb_risk

    pts = [(0, 0, r)]

    sq_lst = [x * x for x in range(r + 1)]

    for z in reversed(range(r)):
        z_sq = z * z
        rem = r_sq - z_sq
        x_max = bisect.bisect_right(sq_lst, rem >> 1) - 1
        for x in range(x_max + 1):
            #print(f"z = {z}, x = {x}")
            y_sq = r_sq - z_sq - x * x
            y = bisect.bisect_right(sq_lst, y_sq) - 1
            if sq_lst[y] != y_sq: continue
            pts.append((x, y, z))
    print(f"number of points = {len(pts)}")
    #heuristics = [heuristic(pt) for pt in pts]
    #heuristics, pts = zip(*sorted(zip(heuristics, pts)))
    #print("pts:", pts)
    #print(heuristics)

    h = [(0., 0)]
    dists = [float("inf") for _ in pts]
    dists[0] = 0.
    remain = set(range(len(pts)))

    # Dijkstra algorithm
    while h:
        d0, idx0 = heapq.heappop(h)
        if idx0 not in remain: continue
        remain.remove(idx0)
        dists[idx0] = d0
        #d0 = -d0_neg
        pt0 = pts[idx0]
        pt0_lst = list({pt0, (pt0[1], pt0[0], pt0[2]), (-pt0[0], pt0[1], pt0[2])})#, (pt0[1], -pt0[0], pt0[2])})
        for idx in remain:
            d = float("inf")
            pt = pts[idx]
            for pt0 in pt0_lst:
                d = min(d, arcRisk(pt0, pt))
            d += d0
            if d >= dists[idx]: continue
            dists[idx] = d
            heapq.heappush(h, (d, idx))
    #print(pts)
    #print(dists)
    res = float("inf")
    for i1, (pt1, d1) in enumerate(zip(pts, dists)):
        #res = min(res, 2 * d1 + arcRisk(pt1, (pt1[0], pt1[1], -pt1[2])))
        for i2 in range(i1 + 1):
            pt2_0 = pts[i2]
            d2 = dists[i2]
            d = d1 + d2
            if d >= res: continue
            pt2_lst = list({(pt2_0[0], pt2_0[1], -pt2_0[2]), (pt2_0[1], pt2_0[0], -pt2_0[2]), (-pt2_0[0], pt2_0[1], -pt2_0[2])})
            for pt2 in pt2_lst:
                #print(pt1, pt2, d)
                res = min(res, d + arcRisk(pt1, pt2))
    return res
    
    """
    cnt = 1
    #print((0, 0, r))
    dists = SortedList()
    dists_dict = {}
    pt = (0, 0, r)
    dists.add((0., pt))
    #dists_dict[pt] = 0.
    
    for z in reversed(range(r)):
        z_sq = z * z
        rem = r_sq - z_sq
        x_max = bisect.bisect_right(sq_lst, rem >> 1) - 1
        for x in range(x_max + 1):
            #print(f"z = {z}, x = {x}")
            y_sq = r_sq - z_sq - x * x
            y = bisect.bisect_right(sq_lst, y_sq) - 1
            if sq_lst[y] != y_sq: continue
            pt = (x, y, z)
            d = float("inf")
            #print((x, y, z))
            #cnt += 1
            for d0, pt0 in dists:
                if d0 >= d: break
                pt0_lst = [pt0, (pt0[1], pt0[0], pt0[2]), (-pt0[0], pt0[1], pt0[2])]
                for pt0_0 in pt0_lst:
                    d = min(d, d0 + arcRisk(pt0_0, pt))
            dists.add((d, pt))
    #print(dists)
    res = float("inf")
    for i1, (d1, pt1) in enumerate(dists):
        for i2 in range(i1 + 1):
            d2, pt2_0 = dists[i2]
            d = d1 + d2
            if d >= res: continue
            pt2_lst = [(pt2_0[0], pt2_0[1], -pt2_0[2]), (pt2_0[1], pt2_0[0], -pt2_0[2]), (-pt2_0[0], pt2_0[1], -pt2_0[2])]
            for pt2 in pt2_lst:
                res = min(res, d + arcRisk(pt1, pt2))
    return res
    """
    """
    for z in reversed(range(r)):
        z_sq = z * z
        rem = r_sq - z_sq
        for x in range((isqrt(rem >> 1)) + 1):
            #print(f"z = {z}, x = {x}")
            y_sq = r_sq - z_sq - x * x
            y = isqrt(y_sq)
            if y_sq != y * y: continue
            print((x, y, z))
            cnt += 1
    """
    #print(f"total number of integer points = {cnt}")
    #return 0.

def calculateMoonPathMinimumRiskMersenneNumberRadiiSum(mersenne_max: int=15) -> float:
    """
    Solution to Project Euler #353
    """
    res = 0.
    since0 = time.time()
    for n in range(1, mersenne_max + 1):
        since = time.time()
        ans = calculateMoonPathMinimumRisk((1 << n) - 1)
        res += ans
        t = time.time()
        print(n, (1 << n) - 1, ans, res, f"time for this case = {t - since:.4f} seconds, total time so far = {t - since0:.4f} seconds")
    return res

# Problem 354
def honeycombDistanceCount(
    dist_sq: int,
    ps: Optional[PrimeSPFsieve]=None,
) -> int:
    q, r = divmod(dist_sq, 3)
    if r: return 0
    pf = calculatePrimeFactorisation(q, ps=ps)
    res = 1
    for p, f in pf.items():
        r = p % 3
        if not r: continue
        elif r == 2:
            if f & 1: return 0
            continue
        res *= f + 1
    return 6 * res

def distancesWithExactHoneycombNumberCountBruteForce(
    honeycomb_number: int,
    dist_max: int,
    ps: Optional[PrimeSPFsieve]=None,
) -> int:
    if honeycomb_number % 6: return 0
    res = 0
    for dist_sq in range(3, dist_max * dist_max + 1, 3):
        if honeycombDistanceCount(dist_sq, ps=ps) != honeycomb_number:
            continue
        print(dist_sq)
        res += 1
    return res

# Problem 356
def specialCubicLargestRoot(n: int) -> float:
    """
    Calculates the largest root of the cubic polynomial:
     g(x) = x ** 3 - 2 ** 2 * x ** 2 + n
    """
    return (1 << n) * (1 + 2 * math.cos(math.acos(1 - 27 * n / (2 * (1 << (3 * n)))) / 3)) / 3

def specialCubicLargestRootToPowerSeriesTermsGenerator(
    n: int,
    exp: int,
    #res_md: Optional[int]=None,
) -> Generator[CustomFraction, None, None]:
    #addMod = (lambda x, y: x + y) if res_md is None else (lambda x, y: (x + y) % res_md)
    #mulMod = (lambda x, y: x * y) if res_md is None else (lambda x, y: (x * y) % res_md)
    def binom(n: int, k: int) -> int:
        if n >= 0:
            res = math.comb(n, k) if k >= 0 else 0
            return res# if res_md is None else res % res_md
        k = max(k, n - k)
        if k < 0: return 0
        res = 1
        for k_ in range(k):
            res *= n - k_
        res //= math.factorial(k)
        return res# if res_md is None else res % res_md

    #yield CustomFraction(pow(2, n, res_md), 1)

    for m in itertools.count(1):
        print(3 * m - exp - 1, m - 1, binom(3 * m - exp - 1, m - 1))
        term = CustomFraction(-exp * binom(3 * m - exp - 1, m - 1) * (n ** m) * (1 << (exp * n)), m * n * (1 << (3 * m * n)))
        #if pow2 > 0: term /= (1 << pow2)
        #elif pow2 < 0: term *= (1 << (-pow2))
        yield term

def specialCubicLargestRootToPowerFloor(
    n: int,
    exp: int,
    res_md: Optional[int]=None,
) -> int:
    # Using identities that for exp >= 3 * m (where m is an integer):
    #  -exp * ((3 * m - exp - 1) choose (m - 1)) = (-1) ** (m - 1) * ((exp - 2 * m - 1) choose (m - 1))
    # and:
    #  exp * ((exp - 2 * m - 1) choose (m - 1)) = m * (((exp - 2 * m) choose m) + 2 * ((exp - 2 * m - 1) choose (m - 1)))

    #if n == 1:
    #    return 

    addMod = (lambda x, y: x + y) if res_md is None else (lambda x, y: (x + y) % res_md)
    mulMod = (lambda x, y: x * y) if res_md is None else (lambda x, y: (x * y) % res_md)

    res = addMod(pow(2, n * exp, res_md), -1)
    #print(res)
    binom1 = exp - 2
    binom2 = 1
    m_max = (exp // 3)
    for m in range(1, m_max + 1):
        if not m % (10 ** 4): print(f"m = {m} (of {m_max})")
        #term = mulMod(mulMod(addMod(math.comb(exp - 2 * m, m), 2 * math.comb(exp - 2 * m - 1, m - 1)), pow(n, m - 1)), pow(2, (exp - 3 * m) * n, res_md))
        term = mulMod(mulMod(addMod(binom1, 2 * binom2), pow(n, m - 1)), pow(2, (exp - 3 * m) * n, res_md))
        binom1 *= ((exp - 3 * m - 1) * (exp - 3 * m - 2) * (exp - 3 * m - 3))
        binom1 //= m + 1
        binom1 //= exp - 2 * m
        binom1 //= exp - 2 * m - 1
        binom2 *= ((exp - 3 * m) * (exp - 3 * m - 1) * (exp - 3 * m - 2))
        binom2 //= m
        binom2 //= exp - 2 * m - 1
        binom2 //= exp - 2 * m - 2
        res = addMod(res, -term) if m & 1 else addMod(res, term)
        #print(term, res)
    return res

def specialCubicLargestRootsToPowerFloorSum(
    n_min: int=1,
    n_max: int=30,
    exp: int=987654321,
    res_md: Optional[int]=10 ** 8,
) -> int:
    """
    Solution to Project Euler #356
    """
    addMod = (lambda x, y: x + y) if res_md is None else (lambda x, y: (x + y) % res_md)
    res = 0
    for n in range(n_min, n_max + 1):
        term = specialCubicLargestRootToPowerFloor(n, exp, res_md)
        print(f"n = {n}, term = {term}")
        res = addMod(res, term)
    return res


# Problem 357
def allFactorPairSumsPrimeSum(n_max: int=10 ** 8) -> int:
    """
    Solution to Project Euler #357
    """
    ps = SimplePrimeSieve(n_max + 1)

    def primeTest(num: int) -> int:
        return ps.millerRabinPrimalityTestWithKnownBounds(num, max_n_additional_trials_if_above_max=10)[0]

    p_i_mx = len(ps.p_lst) if not ps.p_lst or ps.p_lst[-1] <= n_max + 1 else bisect.bisect_right(ps.p_lst)
    res = 0
    for p_i in range(p_i_mx):
        p = ps.p_lst[p_i]
        num = p - 1
        #print(f"num = {num}")
        for fact1 in range(2, isqrt(num) + 1):
            fact2, r = divmod(num, fact1)
            if r: continue
            #print(fact1, fact2, fact1 + fact2)
            if not primeTest(fact1 + fact2):
                break
        else:
            res += num
            #print(num)
            continue
    return res

# Problem 358
def cyclicNumbersWithGivenPrefixAndSuffixCount(
    pref_val: int,
    pref_n_dig: int,
    suff_val: int,
    suff_n_dig: int,
    base: int=10,
) -> list[int]:
    if not pref_val:
        raise ValueError("pref_val must be a strictly positive integer")
    pref_nines = (base ** pref_n_dig) - 1
    if pref_val > pref_nines:
        raise ValueError("pref_val can contain at most pref_n_dig digits when represented in the chosen base")

    mult_rng = [max(((pref_nines - 1) // (pref_val + 1)) + 1, pref_n_dig + suff_n_dig + 1), ((pref_nines - 1) // pref_val) + 1]
    it = range(mult_rng[0] + (not mult_rng[0] & 1), mult_rng[1], 2)
    if mult_rng[0] <= 2:
        it = itertools.chain([2], it)
    #mult_rng[0] += not mult_rng[0] & 1
    print(f"multiple range = [{mult_rng[0]}, {mult_rng[1] - 1}]")
    #res = []
    suff_digs = []
    num = suff_val
    while num:
        num, d = divmod(num, base)
        suff_digs.append(d)
    cnt = 0
    ps = SimplePrimeSieve()
    def primeTest(num: int) -> int:
        return ps.millerRabinPrimalityTestWithKnownBounds(num, max_n_additional_trials_if_above_max=10)[0]
    for mult in it:
        #print(f"mult = {mult}")
        if not primeTest(mult) or not base % mult: continue
        prod = mult * suff_val
        if (prod + 1) % (base ** suff_n_dig):
            continue
        print(f"mult value {mult} passed initial screen")
        n_dig = mult - 1
        """
        nines = (base ** n_dig) - 1
        num, r = divmod(nines, mult)
        print(f"n_dig = {n_dig}, mult = {mult}, num = {num}, r = {r}")
        #print(num % (base ** suff_n_dig))
        if r or num % (base ** suff_n_dig) != suff_val: continue
        print(num)
        res.append(num)
        """
        curr = 0
        #seen = {0}
        rpt_seen = False
        print("prefix:")
        for i in range(pref_n_dig):
            d, curr = divmod(base * (curr + 1) - 1, mult)
            #print(i, curr)
            if not curr: rpt_seen = True
            print(d)
            #if curr in seen:
            #    rpt_seen = True
            #seen.add(curr)
        if rpt_seen:
            print("recurrence found early")
            continue
        for i in range(pref_n_dig, n_dig - suff_n_dig):
            #if i and not i % (10 ** 7): print(f"processed first {i} digits (of {n_dig})")
            d, curr = divmod(base * (curr + 1) - 1, mult)
            #print(d, curr)
            if not curr:
                rpt_seen = True
                break
        if rpt_seen:
            print("recurrence found early")
            continue
        print("suffix:")
        for i in range(suff_n_dig - 1):
            d, curr = divmod(base * (curr + 1) - 1, mult)
            #print(d, curr)
            print(d)
            if d != suff_digs[~i]:
                print(f"discrepency with expected digit value ({suff_digs[~i]})")
                break
            elif not curr:
                print("recurrence found early")
                break
        else:
            d, curr = divmod(base * (curr + 1) - 1, mult)
            print(d)
            if d != suff_digs[0]:
                print(f"discrepency with expected digit value ({suff_digs[0]})")
                break
            if curr:
                print(f"division has non-zero remainder")
                continue
            print(f"solution found for mult = {mult}")
            cnt += 1
    print(f"solution count = {cnt}")
    """
    for n_dig in range(pref_n_dig + suff_n_dig, max_n_dig):
        nines = (base ** n_dig) - 1
        print(f"n_dig = {n_dig}")
        for mult in range(*mult_rng):
            
            num, r = divmod(nines, mult)
            #print(f"n_dig = {n_dig}, mult = {mult} num = {num}, r = {r}")
            #print(num % (base ** suff_n_dig))
            if r or num % (base ** suff_n_dig) != suff_val: continue
            print(num)
            res.append(num)
    """ 
    return cnt

def cyclicNumbersWithGivenPrefixAndSuffixDigitSums(
    pref_val: int,
    pref_n_dig: int,
    suff_val: int,
    suff_n_dig: int,
    base: int=10,
) -> list[int]:
    """
    Solution to Project Euler #358
    """
    # TODO- look into the solutions on the Project Euler forum, specifically
    # those that use that the sum of digits is (base - 1) * (p - 1) / 2
    if not pref_val:
        raise ValueError("pref_val must be a strictly positive integer")
    pref_nines = (base ** pref_n_dig) - 1
    if pref_val > pref_nines:
        raise ValueError("pref_val can contain at most pref_n_dig digits when represented in the chosen base")

    mult_rng = [max(((pref_nines - 1) // (pref_val + 1)) + 1, pref_n_dig + suff_n_dig + 1), ((pref_nines - 1) // pref_val) + 1]
    it = range(mult_rng[0] + (not mult_rng[0] & 1), mult_rng[1], 2)
    if mult_rng[0] <= 2:
        it = itertools.chain([[2], it])
    #mult_rng[0] += not mult_rng[0] & 1
    print(f"multiple range = [{mult_rng[0]}, {mult_rng[1] - 1}]")
    res = []
    suff_digs = []
    num = suff_val
    while num:
        num, d = divmod(num, base)
        suff_digs.append(d)
    #cnt = 0
    ps = SimplePrimeSieve()
    def primeTest(num: int) -> int:
        return ps.millerRabinPrimalityTestWithKnownBounds(num, max_n_additional_trials_if_above_max=10)[0]
    for mult in it:
        if not primeTest(mult) or not base % mult: continue
        prod = mult * suff_val
        if (prod + 1) % (base ** suff_n_dig):
            continue
        print(f"multiplier value {mult} passed initial screen")
        n_dig = mult - 1
        curr = 0
        #seen = {0}
        rpt_seen = False
        ans = 0
        for i in range(n_dig - suff_n_dig):
            #if i and not i % (10 ** 7): print(f"processed first {i} digits (of {n_dig})")
            d, curr = divmod(base * (curr + 1) - 1, mult)
            ans += d
            #print(d, curr)
            if not curr:
                rpt_seen = True
                break
        if rpt_seen:
            print("recurrence found early")
            continue
        print("suffix:")
        for i in range(suff_n_dig - 1):
            d, curr = divmod(base * (curr + 1) - 1, mult)
            #print(d, curr)
            print(d)
            if d != suff_digs[~i]:
                print(f"discrepency with expected digit value ({suff_digs[~i]})")
                break
            elif not curr:
                print("recurrence found early")
                break
            ans += d
        else:
            d, curr = divmod(base * (curr + 1) - 1, mult)
            print(d)
            if d != suff_digs[0]:
                print(f"discrepency with expected digit value ({suff_digs[0]})")
                break
            if curr:
                print(f"division has non-zero remainder")
                continue
            ans += d
            print(f"solution found for mult = {mult}")
            res.append(ans)
    #print(f"solution count = {cnt}")
    """
    for n_dig in range(pref_n_dig + suff_n_dig, max_n_dig):
        nines = (base ** n_dig) - 1
        print(f"n_dig = {n_dig}")
        for mult in range(*mult_rng):
            
            num, r = divmod(nines, mult)
            #print(f"n_dig = {n_dig}, mult = {mult} num = {num}, r = {r}")
            #print(num % (base ** suff_n_dig))
            if r or num % (base ** suff_n_dig) != suff_val: continue
            print(num)
            res.append(num)
    """ 
    return res

# Problem 359
def hilbertNewHotelBruteForce(f: int, r: int) -> int:
    res = [(0, -1) for _ in range(f)]
    sqs = [0]
    for num in itertools.count(1):
        for floor in range(1, f + 1):
            if not res[floor - 1][0]:
                res[floor - 1] = (1, num)
                print(floor, num)
                break
            num2 = num + res[floor - 1][1]
            while sqs[-1] < num2:
                sqs.append(len(sqs) ** 2)
            j = bisect.bisect_left(sqs, num2)
            if sqs[j] == num2:
                res[floor - 1] = (res[floor - 1][0] + 1, num)
                #if res[floor - 1][0] == 2:
                #    print(floor, *res[floor - 1])
                break
        else: continue
        if res[f - 1][0] == r:
            break#return res[floor - 1][-1]
    #print(sqs)
    print(res)
    return res[floor - 1][-1]

def hilbertNewHotel(f: int, r: int) -> int:
    # TODO- prove these formulae
    if f == 1:
        return (r * (r + 1)) >> 1
    res = (f + r) ** 2
    if f & 1:
        res -= (2 * f + 3 * r - 1) if r & 1 else (4 * f + 3 * r - 3)
    else:
        res -= (2 * f + r) if r & 1 else (r)

    return res >> 1

def hilbertNewHotelFactorisationSum(
    f_r_prod: int=71328803586048,
    ps: Optional[PrimeSPFsieve]=None,
    res_md: Optional[int]=10 ** 8,
) -> int:
    addMod = (lambda x, y: x + y) if res_md is None else (lambda x, y: (x + y) % res_md)
    pf = calculatePrimeFactorisation(f_r_prod, ps=ps)
    def factorGenerator(pf: dict[int, int]) -> Generator[int, None, None]:
        p_lst = sorted(pf.keys())
        n_p = len(p_lst)
        def recur(idx: int=0, curr: int=1) -> Generator[int, None, None]:
            if idx == n_p:
                yield curr
                return
            p = p_lst[idx]
            for _ in range(pf[p] + 1):
                yield from recur(idx=idx + 1, curr=curr)
                curr *= p
            return
        yield from recur(idx=0, curr=1)

    res = 0
    for f in factorGenerator(pf):
        r = f_r_prod // f
        res = addMod(res, hilbertNewHotel(f, r))

    return res

# Problem 360
def sphereIntegerPointsManhattanDistanceToOriginSumBruteForce(rad: int=10 ** 10) -> int:
    r = rad
    if rad <= 0: return 0
    r_sq = r * r
    res = 6 * r
    print((0, 0, r), 6)
    z_min = isqrt(max(r_sq // 3 - 1, 0)) + 1
    z_max = r - 1
    for z in range(z_min, z_max + 1):
        z_sq = z * z
        rem = r_sq - z_sq
        y_min = isqrt((rem - 1) >> 1) + 1
        y_max = isqrt(rem)
        if not rem & 1:
            y_sq = y_min * y_min
            x_sq = rem - y_sq
            if x_sq == y_sq:
                print((y_min, y_min, z), 8 * (1 + 2 * (y_min != z)))
                res += 8 * (1 + 2 * (y_min != z)) * (z + 2 * y_min)
                y_min += 1
        for y in range(y_min, min(y_max + 1, z)):
            #print(f"z = {z}, x = {x}")
            x_sq = rem - y * y
            x = isqrt(x_sq)
            if x * x != x_sq: continue
            print((x, y, z), 24 * (1 + bool(x)))
            res += 24 * (1 + bool(x)) * (x + y + z)
        if y_min > y_max or y_max < z: continue
        x_sq = rem - z_sq
        x = isqrt(x_sq)
        if x * x != x_sq: continue
        print((x, z, z), 12 * (1 + bool(x)))
        res += 12 * (1 + bool(x)) * (x + 2 * z)
    return res

"""
def sumOfTwoSquaresSolutionsCount(
    target: int,
    ps: Optional[PrimeSPFsieve]=None,
) -> int:
    ""
    Calculates the number of ordered integer pairs (a, b) for which:
        a ** 2 + b ** 2 = target
    Note that this includes solutions for which a and b are equal and
    for which a and/or b is/are negative.
    ""
    if not target: return 1
    elif target < 0: return 0
    res = 1
    if ps is not None:
        pf = calculatePrimeFactorisation(target, ps=ps)
        #gaussian_pf = {}
        #mult = 1
        if 2 in pf.keys():
            pf.pop(2)
        for p, f in pf.items():
            residue = p % 4
            if residue == 3:
                if f & 1:
                    #print(p, f)
                    return 0
                continue
            res *= f + 1
    else:
        ps2 = SimplePrimeSieve()
        def primeTest(num: int) -> bool:
            return ps2.millerRabinPrimalityTestWithKnownBounds(num, max_n_additional_trials_if_above_max=10)[0]
        num = target
        if not primeTest(num):
            f = 0
            while not num & 1:
                num >>= 1
                f += 1
            #if f: print(2, f)
            if not ps2.millerRabinPrimalityTestWithKnownBounds(num)[0]:
                for p in range(3, isqrt(num) + 1, 2):
                    if p * p > num: break
                    if p % 10 ** 7 == 1:
                        print(f"num = {num}, p = {p} (of {isqrt(num)})")
                    f = 0
                    num2, r = divmod(num, p)
                    while not r:
                        num = num2
                        f += 1
                        num2, r = divmod(num, p)
                    if not f: continue
                    #print(p, f)
                    if p & 3 == 3:
                        if f & 1: return 0
                        continue
                    res *= f + 1
                    if not ps2.millerRabinPrimalityTestWithKnownBounds(num)[0]:
                        break
        if num > 1:
            #print(num, 1)
            if num & 3 == 3: return 0
            res <<= 1
    return res << 2
"""

def sumOfTwoSquaresSolutionsCountFromPrimeFactorisationBreakdown(
    target_pf: dict[int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """
    For the integer target with prime factorisation given by target_pf,
    calculates the number of non-negative ordered integer pairs (a, b)
    for which a <= b, and:
        a ** 2 + b ** 2 = target
    separating the counts into those where a and b are strictly positive, those
    those where exactly one of a and b is zero and those where both of a and b
    are zero, and further subdividing these counts into those where a and b are
    equal and those where a and b are unequal
    """
    
    res = 1
    f2 = target_pf.pop(2, 0)
    sq_status = 1 << (f2 & 1)
    for p, f in target_pf.items():
        if f & 1: sq_status = 0
        residue = p % 4
        if residue == 3:
            if f & 1:
                #print(p, f)
                res = 0
                break
            continue
        res *= f + 1
    #print(sq_status, res)
    if f2: target_pf[2] = f2 # Adding 2 back into the prime factorisation
                             # to eliminate that potential side effect
    if sq_status == 1:
        return ((0, (res - 1) >> 1), (0, 1), (0, 0))
    elif sq_status == 2:
        return ((1, (res - 1) >> 1), (0, 0), (0, 0))
    return ((0, res >> 1), (0, 0), (0, 0))

def sumOfTwoSquaresSolutionsCountBreakdown(
    target: int,
    ps: Optional[PrimeSPFsieve]=None,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """
    Calculates the number of non-negative ordered integer pairs (a, b) for which
    a <= b and:
        a ** 2 + b ** 2 = target
    separating the counts into those where a and b are strictly positive, those
    those where exactly one of a and b is zero and those where both of a and b
    are zero, and further subdividing these counts into those where a and b are
    equal and those where a and b are unequal
    """
    if target < 0: return ((0, 0), (0, 0), (0, 0))
    if not target: return ((0, 0), (0, 0), (1, 0))
    
    res = 1
    sq_status = 0
    if ps is not None:
        return sumOfTwoSquaresSolutionsCountFromPrimeFactorisationBreakdown(
            calculatePrimeFactorisation(target, ps=ps),
        )
    else:
        num = target
        f_mod2 = False
        while not num & 1:
            num >>= 1
            f_mod2 = not f_mod2
        sq_status = 1 << (f_mod2)
        for p in range(3, isqrt(num) + 1, 2):
            if p * p > num: break
            f = 0
            num2, r = divmod(num, p)
            while not r:
                num = num2
                f += 1
                num2, r = divmod(num, p)
            if not f: continue
            elif f & 1: sq_status = 0
            if p & 3 == 3:
                if f & 1: return ((0, 0), (0, 0), (0, 0))
                continue
            res *= f + 1
        if num > 1:
            sq_status = 0
            if num & 3 == 3: return ((0, 0), (0, 0), (0, 0))
            res <<= 1
    #print(sq_status, res)
    if sq_status == 1:
        return ((0, (res - 1) >> 1), (0, 1), (0, 0))
    elif sq_status == 2:
        return ((1, (res - 1) >> 1), (0, 0), (0, 0))
    return ((0, res >> 1), (0, 0), (0, 0))

def sumOfTwoSquaresSolutionsCountBreakdownCoprimeProduct(
    breakdown1: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    breakdown2: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:

    if not sum(sum(x) for x in breakdown1) or not sum(sum(x) for x in breakdown2):
        return ((0, 0), (0, 0), (0, 0))
    if breakdown1[2][0] or breakdown2[2][0]:
        return ((0, 0), (0, 0), (1, 0))
    tot1 = ((breakdown1[0][1]) << 1) + breakdown1[0][0] + breakdown1[1][1]
    tot2 = ((breakdown2[0][1]) << 1) + breakdown2[0][0] + breakdown2[1][1]
    res = tot1 * tot2
    if breakdown1[1][1] == 1 and breakdown2[1][1] == 1:
        return ((0, (res - 1) >> 1), (0, 1), (0, 0))
    if (breakdown1[0][0] == 1 and breakdown2[1][1] == 1) or (breakdown1[1][1] == 1 and breakdown2[0][0] == 1):
        return ((1, (res - 1) >> 1), (0, 0), (0, 0))
    return ((0, res >> 1), (0, 0), (0, 0))

def sumOfTwoSquaresSolutionsCountFromBreakdown(
    target_breakdown: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> int:
    return (target_breakdown[0][0] << 2) +\
        (target_breakdown[0][1] << 3) +\
        (target_breakdown[1][1] << 2) +\
        (target_breakdown[2][0])


def sumOfTwoSquaresSolutionsCount(
    target: int,
    ps: Optional[PrimeSPFsieve]=None,
) -> int:
    target_breakdown = sumOfTwoSquaresSolutionsCountBreakdown(target, ps=ps)
    return sumOfTwoSquaresSolutionsCountFromBreakdown(target_breakdown)
    #print(cnt_breakdown)
    

def sphereIntegerPointsManhattanDistanceToOriginSum(
    rad: int=10 ** 10,
    ps: Optional[PrimeSPFsieve]=None,
) -> int:
    # Relies on there being no integer n for which 3 * n ** 2
    # is a square (follows from the prime factorisation having
    # an odd power of 3)
    #r_sq = r * r
    res = rad
    for z in range(rad):
        #z_sq = z * z
        #rem = r_sq - z_sq
        rem_facts = [rad - z, rad + z]
        cnt = 1
        g = gcd(*rem_facts)
        
        g_pf = calculatePrimeFactorisation(g, ps=ps)
        #print(f"rad = {rad}, z = {z}, rem_facts = {rem_facts}, g = {g}, g_pf = {g_pf}")
        common_pf = {}
        for p in g_pf.keys():
            #print(f"p = {p}")
            common_pf[p] = 0
            for j in range(2):
                num2, r = divmod(rem_facts[j], p)
                while not r:
                    rem_facts[j] = num2
                    common_pf[p] += 1
                    num2, r = divmod(rem_facts[j], p)
        
        bd = sumOfTwoSquaresSolutionsCountFromPrimeFactorisationBreakdown(common_pf)
        #print(f"common_pf = {common_pf}, bd = {bd}, post division rem_facts = {rem_facts}")
        for rem in rem_facts:
            
            bd2 = sumOfTwoSquaresSolutionsCountBreakdown(rem, ps=ps)
            
            bd = sumOfTwoSquaresSolutionsCountBreakdownCoprimeProduct(
                bd,
                bd2,
            )
            #print(rem, bd2, bd)
        
        cnt = sumOfTwoSquaresSolutionsCountFromBreakdown(bd)
        #print(f"rad = {rad}, z = {z}, overall breakdown = {bd}, count = {cnt}")
        term = z * cnt
        res += term
        if not z % 10 ** 5:
            print(f"z = {z} (of {rad}), res = {res}")
        if term:
            print(f"Non-zero term for z = {z}: {term}")
        #print(f"rem = {rem}, cnt = {cnt}, term = {term}")
    return res * 6

##############
project_euler_num_range = (351, 400)

def evaluateProjectEulerSolutions351to400(eval_nums: Optional[Set[int]]=None) -> None:
    if not eval_nums:
        eval_nums = set(range(project_euler_num_range[0], project_euler_num_range[1] + 1))
    since0 = time.time()

    if 351 in eval_nums:
        since = time.time()
        res = hiddenPointsInTriangularLatticeHexagon(
            hexagon_side_length=10 ** 8,
            ps=None,
        )
        print(f"Solution to Project Euler #351 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 352 in eval_nums:
        since = time.time()
        res = bloodTestOptimalStrategyMeanTestCountSum(
            n_subjects=10 ** 4,
            p_infected_vals=(.01 * x for x in range(1, 51)),
        )
        print(f"Solution to Project Euler #352 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 353 in eval_nums:
        since = time.time()
        res = calculateMoonPathMinimumRiskMersenneNumberRadiiSum(mersenne_max=15)
        print(f"Solution to Project Euler #353 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 356 in eval_nums:
        since = time.time()
        res = specialCubicLargestRootsToPowerFloorSum(
            n_min=1,
            n_max=30,
            exp=987654321,
            res_md=10 ** 8,
        )
        print(f"Solution to Project Euler #356 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 357 in eval_nums:
        since = time.time()
        res = allFactorPairSumsPrimeSum(n_max=10 ** 2)
        print(f"Solution to Project Euler #357 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 358 in eval_nums:
        since = time.time()
        res = cyclicNumbersWithGivenPrefixAndSuffixDigitSums(
            pref_val=137,
            pref_n_dig=11,
            suff_val=56789,
            suff_n_dig=5,
            base=10,
        )
        print(f"Solution to Project Euler #358 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 359 in eval_nums:
        since = time.time()
        res = hilbertNewHotelFactorisationSum(
            f_r_prod=71328803586048,
            ps=None,
            res_md=10 ** 8,
        )
        print(f"Solution to Project Euler #359 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 360 in eval_nums:
        since = time.time()
        res = sphereIntegerPointsManhattanDistanceToOriginSum(
            rad=10 ** 8,
            ps=None,
        )
        print(f"Solution to Project Euler #360 = {res}, calculated in {time.time() - since:.4f} seconds")

    print(f"Total time taken = {time.time() - since0:.4f} seconds")

if __name__ == "__main__":
    eval_nums = {360}
    evaluateProjectEulerSolutions351to400(eval_nums)

#num = (1 << 15) - 1
#print(calculateMoonPathMinimumRisk(num))
"""
for dist_sq in [3, 21, 111_111_111 ** 2]:
    print(dist_sq, honeycombDistanceCount(dist_sq, ps=None))

print(distancesWithExactHoneycombNumberCountBruteForce(
    honeycomb_number=12,
    dist_max=5,
    ps=None,
))
"""


"""
print(cyclicNumbersWithGivenPrefixAndSuffixCount(
    pref_val=137,
    pref_n_dig=11,
    suff_val=56789,
    suff_n_dig=5,
    base=10,
))
"""
"""
print(findCyclicNumbersWithGivenPrefixAndSuffix(
    pref_val=1,
    pref_n_dig=1,
    suff_val=7,
    suff_n_dig=1,
    base=10,
))
"""

#for n in range(1, 21):
#    print(n, specialCubicLargestRoot(n))
"""
n = 1
exp = 987654321
res_md = 10 ** 8
print(specialCubicLargestRootToPowerFloor(
    n=n,
    exp=exp,
    res_md=res_md,
))
"""
"""
res = 1 << (n * exp)
for i, term in zip(range(1, 101), specialCubicLargestRootToPowerSeriesTermsGenerator(n, exp)):
    res += term.numerator / term.denominator
    print(i, term.numerator / term.denominator, term, res)
"""

#for f, r in [(1, 1), (1, 2), (2, 1), (10, 20), (25, 75), (99, 100)]:
#    print(f"P({f}, {r}) = {hilbertNewHotelBruteForce(f, r)}")
#f, r = 10, 1
#print(f"P({f}, {r}) = {hilbertNewHotelBruteForce(f, r)}")
#for f in range(1, 10):
#    print([hilbertNewHotel(f, r) for r in range(1, 10)])
"""
use_ps = True
ps = PrimeSPFsieve() if use_ps else None
for target in range(51):
    print(target, sumOfTwoSquaresSolutionsCountBreakdown(target, ps=ps), sumOfTwoSquaresSolutionsCount(target, ps=ps))
"""