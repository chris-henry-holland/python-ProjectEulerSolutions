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

    if 357 in eval_nums:
        since = time.time()
        res = allFactorPairSumsPrimeSum(n_max=10 ** 8)
        print(f"Solution to Project Euler #357 = {res}, calculated in {time.time() - since:.4f} seconds")

    print(f"Total time taken = {time.time() - since0:.4f} seconds")

if __name__ == "__main__":
    eval_nums = {357}
    evaluateProjectEulerSolutions351to400(eval_nums)