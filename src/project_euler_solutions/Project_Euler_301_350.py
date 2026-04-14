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

from algorithms.number_theory_algorithms import gcd, lcm, isqrt, integerNthRoot, solveLinearCongruence, extendedEuclideanAlgorithm
from algorithms.pseudorandom_number_generators import blumBlumShubPseudoRandomGenerator
from algorithms.continued_fractions_and_Pell_equations import pellSolutionGenerator, generalisedPellSolutionGenerator, pellFundamentalSolution
from algorithms.Pythagorean_triple_generators import pythagoreanTripleGeneratorByHypotenuse


# Problem 301
def nimVariantPlayer2WinsWithPerfectPlayConfigurationsCount(pow2: int=30) -> int:
    """
    Solution to Project Euler #301
    """
    curr = [1, 1]
    for _ in range(pow2):
        curr = [curr[1], sum(curr)]
    return curr[1]

# Problem 302
def strongAchillesNumberCount(n_max: int=10 ** 18 - 1) -> int:
    """
    Solution to Project Euler #302
    """

    p_max = integerNthRoot(n_max, 3)
    ps = PrimeSPFsieve(p_max)

    def primeFactorisationIsStrongAchilles(pf: Dict[int, int]) -> bool:
        for p in list(pf.keys()):
            if pf[p] <= 0: pf.pop(p)
        if min(pf.values()) < 2:
            return False
        if pf[max(pf.keys())] < 3: return False
        g1 = 0
        for f in pf.values():
            g1 = gcd(g1, f)
            if g1 == 1: break
        else: return False
        pf2 = {}
        for p, f in pf.items():
            if f > 1: pf2[p] = pf2.get(p, 0) + (f - 1)
            pf3 = ps.primeFactorisation(p - 1)
            for p2, f2 in pf3.items():
                if f2 > 1: pf2[p2] = pf2.get(p2, 0) + f2
        if min(pf2.values()) < 2: return False
        g2 = 0
        for f in pf2.values():
            g2 = gcd(g2, f)
            if g2 == 1: break
        else: return False
        return True
    
    p_lst = ps.p_lst
    print(f"p_max = {p_max}, p_list length = {len(p_lst)}")
    p_dict = {x: i for i, x in enumerate(p_lst)}
    p_sq_lst = [p * p for p in p_lst]
    m = len(p_lst)
    
    res = [0]
    #curr_pf = [0] * m
    curr_totient_pf = [0] * m
    curr_incomplete_totient = SortedSet()

    def recur(idx: int=m - 1, remain_mx: int=n_max, g1: int=0, g2: int=0) -> None:
        if idx < 0 or remain_mx < 2:
            res[0] += (g1 == 1 and g2 == 1 and not curr_incomplete_totient)
            return
        p_mx = max(remain_mx, p_lst[idx])
        if curr_incomplete_totient:
            if idx < 0 or p_lst[curr_incomplete_totient[-1]] > p_mx: return
        
        #if curr_pf[idx] != 1:
        recur(idx=idx - 1, remain_mx=remain_mx, g1=g1, g2=g2)

        if p_lst[idx] > p_mx:
            return

        exp_mn = (2 + (not curr_totient_pf[idx]))
        exp_mx = integerNthRoot(remain_mx, 3)
        remain_mx2 = remain_mx
        for _ in range(exp_mn - 1):
            remain_mx2 //= p_lst[idx]
        p_minus_one_pf = ps.primeFactorisation(p_lst[idx] - 1)
        for p, f in p_minus_one_pf.items():
            if not f: continue
            p_idx = p_dict[p]
            if curr_totient_pf[p_idx] == 1:
                curr_incomplete_totient.remove(p_idx)
            elif f == 1 and not curr_totient_pf[p_idx]:
                curr_incomplete_totient.add(p_idx)
            curr_totient_pf[p_idx] += f
        curr_totient_pf[idx] += exp_mn - 2
        idx2 = idx
        g2_2 = g2
        for exp in range(exp_mn, exp_mx + 1):
            curr_totient_pf[idx] += 1
            remain_mx2 //= p_lst[idx]
            # Review- may be able to restrict further, as for any prime greater
            # than 3, p - 1 is not prime as 2 is a factor
            if curr_incomplete_totient and curr_incomplete_totient[-1] >= remain_mx2 - 1:
                curr_totient_pf[idx] -= exp - 1
                break
            idx2_prev = idx2
            idx2 = min(idx, bisect.bisect_right(p_sq_lst, remain_mx2)) - 1
            for i in reversed(range(idx2, idx2_prev)):
                if g2_2 == 1: break
                g2_2 = gcd(g2_2, curr_totient_pf[i])
            g1_2 = gcd(g1, exp)
            recur(idx=idx2, remain_mx=remain_mx2, g1=g1_2, g2=g2_2)
        else:
            curr_totient_pf[idx] -= exp_mx - 1
        for p, f in p_minus_one_pf.items():
            if not f: continue
            p_idx = p_dict[p]
            if curr_totient_pf[p_idx] == 1:
                curr_incomplete_totient.remove(p_idx)
            elif curr_totient_pf[p_idx] - f == 1:
                curr_incomplete_totient.add(p_idx)
            curr_totient_pf[p_idx] -= f

    recur(idx=m - 1, remain_mx=n_max)
    return res[0]

# Problem 304
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

def primonacciSum(
    n_max: int=10 ** 5,
    prime_start: int=10 ** 14,
    res_md: Optional[int]=1234567891011,
) -> int:
    """
    Solution to Project Euler #304
    """
    # Note: res_md must be odd

    # Review- look into solutions utilising Pisano periods

    ps = SimplePrimeSieve()
    def primeCheck(num: int) -> int:
        return ps.millerRabinPrimalityTestWithKnownBounds(num, max_n_additional_trials_if_above_max=10)[0]
    
    def nextPrime(num: int) -> int:
        if num < 2: return 2
        start = num + 1 + (num & 1)
        for num2 in itertools.count(start, step=2):
            if primeCheck(num2): return num2
        return -1
    
    addMod = (lambda a, b: a + b) if res_md is None else (lambda a, b: (a + b) % res_md)
    multMod = (lambda a, b: a * b) if res_md is None else (lambda a, b: (a * b) % res_md)

    res_md_totient = 1
    if res_md is not None:
        pf = calculatePrimeFactorisation(res_md)
        for p, f in pf.items():
            res_md_totient *= p ** (f - 1) * (p - 1)
    
    def calculateFibonacci(n: int) -> int:
        # Using Binet's formula
        #print(n)
        curr1 = [1, 0]
        curr2 = [1, 0]
        exp1 = [1, 1]
        exp2 = [1, -1] if res_md is None else [1, res_md - 1]
        n2 = n
        while n2:
            #print(n2)
            if n2 & 1:
                #print("hi")
                curr1 = [addMod(multMod(curr1[0], exp1[0]), multMod(5, multMod(curr1[1], exp1[1]))), addMod(multMod(curr1[0], exp1[1]), multMod(curr1[1], exp1[0]))]
                curr2 = [addMod(multMod(curr2[0], exp2[0]), multMod(5, multMod(curr2[1], exp2[1]))), addMod(multMod(curr2[0], exp2[1]), multMod(curr2[1], exp2[0]))]
            n2 >>= 1
            exp1 = [addMod(multMod(exp1[0], exp1[0]), multMod(5, multMod(exp1[1], exp1[1]))), multMod(multMod(exp1[0], exp1[1]), 2)]
            exp2 = [addMod(multMod(exp2[0], exp2[0]), multMod(5, multMod(exp2[1], exp2[1]))), multMod(multMod(exp2[0], exp2[1]), 2)]
        #print(curr1, curr2)
        #print(exp1, exp2)
        res = addMod(curr1[1], -curr2[1])
        if res_md is None:
            return res >> n
        inv2 = pow(2, res_md_totient - 1, res_md)
        #print(f"inv2 = {inv2}, (inv2 * 2) % res_md = {(inv2 * 2) % res_md}")
        return multMod(res, pow(inv2, n, res_md))
    
    curr = prime_start
    res = 0
    for _ in range(n_max):
        curr = nextPrime(curr)
        #print(f"curr = {curr}")
        fib = calculateFibonacci(curr)
        res = addMod(res, fib)
        #print(curr, fib)
        #print(f"res = {res}")
    return res


# Problem 306
def paperStripGamePlayer1WinsWithPerfectPlayCountInitialSolution(n_square_pick: int=2, n_max: int=10 ** 6) -> int:
    """
    Solution to Project Euler #306
    """
    # Using Sprague-Grundy theorem
    g_arr = [0] * n_square_pick
    res = 0
    for i in range(n_square_pick, n_max + 1):
        if not i % 1000: print(f"i = {i} of {n_max}")
        seen = set()
        for i2 in range(0, ((i - n_square_pick) >> 1) + 1):
            g = g_arr[i2] ^ g_arr[i - i2 - n_square_pick]
            seen.add(g)
        for g in range(len(seen) + 1):
            if g in seen: continue
            g_arr.append(g)
            break
        res += bool(g_arr[-1])
    return res

def paperStripGamePlayer1WinsWithPerfectPlayCount(n_max: int=10 ** 6) -> int:
    """
    Solution to Project Euler #306
    """
    # Review- prove that this works

    # Using Sprague-Grundy theorem and OEIS A215721
    n_square_pick = 2
    g_arr = [0] * n_square_pick
    z_lst = [1]
    
    for i in range(n_square_pick, n_max + 1):
        if not i % 1000: print(f"i = {i} of {n_max}")
        seen = set()
        for i2 in range(0, ((i - n_square_pick) >> 1) + 1):
            g = g_arr[i2] ^ g_arr[i - i2 - n_square_pick]
            seen.add(g)
        for g in range(len(seen) + 1):
            if g in seen: continue
            g_arr.append(g)
            break
        if not g_arr[-1]:
            z_lst.append(i)
            if len(z_lst) == 13: break
    if len(z_lst) < 13:
        return n_max - len(z_lst)
    res = n_max - 8
    print(z_lst)
    print(res)
    for i in range(8, 13):
        print(z_lst[i], 1 + ((n_max - z_lst[i]) // 34))
        res -= 1 + ((n_max - z_lst[i]) // 34)
    return res


##############
project_euler_num_range = (301, 350)

def evaluateProjectEulerSolutions251to300(eval_nums: Optional[Set[int]]=None) -> None:
    if not eval_nums:
        eval_nums = set(range(project_euler_num_range[0], project_euler_num_range[1] + 1))

    since0 = time.time()

    if 301 in eval_nums:
        since = time.time()
        res = nimVariantPlayer2WinsWithPerfectPlayConfigurationsCount(pow2=30)
        print(f"Solution to Project Euler #301 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 302 in eval_nums:
        since = time.time()
        res = strongAchillesNumberCount(n_max=10 ** 18 - 1)
        print(f"Solution to Project Euler #302 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 304 in eval_nums:
        since = time.time()
        res = primonacciSum(
            n_max=10 ** 5,
            prime_start=10 ** 14,
            res_md=1234567891011,
        )
        print(f"Solution to Project Euler #304 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 306 in eval_nums:
        since = time.time()
        res = paperStripGamePlayer1WinsWithPerfectPlayCount(n_max=10 ** 6)
        print(f"Solution to Project Euler #301 = {res}, calculated in {time.time() - since:.4f} seconds")

    print(f"Total time taken = {time.time() - since0:.4f} seconds")

if __name__ == "__main__":
    eval_nums = {302}
    evaluateProjectEulerSolutions251to300(eval_nums)
