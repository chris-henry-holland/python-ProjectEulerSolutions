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
def findCyclicNumbersWithGivenPrefixAndSuffix(
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
    print(f"multiple range = [{mult_rng[0]}, {mult_rng[1] - 1}]")
    res = []
    suff_digs = []
    num = suff_val
    while num:
        num, d = divmod(num, base)
        suff_digs.append(d)
    cnt = 0
    for mult in range(*mult_rng):
        #print(f"mult = {mult}")
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
        for i in range(n_dig - suff_n_dig):
            #if i and not i % (10 ** 7): print(f"processed first {i} digits (of {n_dig})")
            d, curr = divmod(base * (curr + 1) - 1, mult)
            #print(d, curr)
        for i in range(suff_n_dig):
            d, curr = divmod(base * (curr + 1) - 1, mult)
            #print(d, curr)
            if d != suff_digs[~i]: break
        else:
            if curr: continue
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
    return res

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

    if 357 in eval_nums:
        since = time.time()
        res = allFactorPairSumsPrimeSum(n_max=10 ** 2)
        print(f"Solution to Project Euler #357 = {res}, calculated in {time.time() - since:.4f} seconds")

    print(f"Total time taken = {time.time() - since0:.4f} seconds")

if __name__ == "__main__":
    eval_nums = {3580}
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
print(findCyclicNumbersWithGivenPrefixAndSuffix(
    pref_val=137,
    pref_n_dig=11,
    suff_val=56789,
    suff_n_dig=5,
    base=10,
))
"""
print(findCyclicNumbersWithGivenPrefixAndSuffix(
    pref_val=137,
    pref_n_dig=11,
    suff_val=56789,
    suff_n_dig=5,
    base=10,
))
