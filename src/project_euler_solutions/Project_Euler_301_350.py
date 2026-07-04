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

from algorithms.number_theory_algorithms import gcd, lcm, isqrt, integerNthRoot, solveLinearCongruence, extendedEuclideanAlgorithm, solveLinearNonHomogeneousDiophantineEquation
from algorithms.pseudorandom_number_generators import blumBlumShubPseudoRandomGenerator
from algorithms.continued_fractions_and_Pell_equations import pellSolutionGenerator, generalisedPellSolutionGenerator, pellFundamentalSolution
from algorithms.Pythagorean_triple_generators import pythagoreanTripleGeneratorByHypotenuse
from algorithms.string_searching_algorithms import KnuthMorrisPratt


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

# Problem 301
def nimVariantPlayer2WinsWithPerfectPlayConfigurationsCount(pow2: int=30) -> int:
    """
    Solution to Project Euler #301

    For three-heap normal-play games of Nim, calculates the
    number of non-negative integers n no greater than 2 ** pow2
    for which the game where the piles initially contain n,
    2 * n and 3 * n stones respectively, perfect play
    from both players results in the player to move first losing.

    In an m-heap normal-play game of Nim (where m is a strictly
    positive integer), there are two players and are m piles of stones,
    each containing a specified number of stones (where the number in
    each pile may differ from one another). The two players alternate
    turns, with a turn consisting of removing a strictly positive
    number of stones from a non-empty pile. A player loses (and so
    the other player wins) if they are unable to make a legal move
    (due to all piles being empty).

    Args:
        Optional named:
        pow2 (int): Non-negative integer giving the power of 2 corresponding
                to the inclusive upper bound on the values of n considered.
            Default: 30
    
    Returns:
    Integer (int) giving the number of non-negative integers n no
    greater than 2 ** pow2 for which the three-heap normal-play game
    of Nim where the piles initially contain n, 2 * n and 3 * n stones 
    respectively, perfect play from both players results in the
    player to move first losing.

    Outline of rationale:
    TODO
    """
    # Using Sprague Grundy
    curr = [1, 1]
    for _ in range(pow2):
        curr = [curr[1], sum(curr)]
    return curr[1]

# Problem 302
def strongAchillesNumberCountBruteForce(
    n_max: int,
    ps: Optional[PrimeSPFsieve]=None,
) -> int:
    """
    Counts the number of strictly positive integers no greater
    than n_max that are strong Achilles.

    An Achilles number is a strictly positive integer that is
    divisible by the square of each of its prime factors and
    is not a perfect power (i.e. it cannot be expressed as
    an integer power greater than 1 of any integer). A strong
    Achilles number is a strictly positive integer that is
    an Achilles number and its Euler totient function value
    (i.e. the number of strictly positive integers less than
    and coprime with the given integer) is also an Achilles
    number.

    Args:
        Requires positional:
        n_max (int): Strictly positive integer giving the
                inclusive upper bound on the integers considered
                for inclusion in the count.
        
        Optional named:
        ps (PrimeSPFsieve object or None): If given, a prime
                sieve object for calculating prime factorisations.
                If not given or given as None, the prime
                factorisations are calculated by direct
                division.
            Default: None
    
    Returns:
    Integer (int) giving the number of strictly positive integers
    no greater than n_max that are strong Achilles.

    Outline of rationale:
    TODO
    """
    #ps = PrimeSPFsieve(n_max)
    ps = None

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
        #print(f"pf = {pf}")
        pf2 = {}
        for p, f in pf.items():
            if f > 1: pf2[p] = pf2.get(p, 0) + (f - 1)
            pf3 = calculatePrimeFactorisation(p - 1, ps=ps)
            #print(pf3)
            for p2, f2 in pf3.items():
                if f2: pf2[p2] = pf2.get(p2, 0) + f2
            #print(p, f, pf2)
        if min(pf2.values()) < 2: return False
        g2 = 0
        for f in pf2.values():
            g2 = gcd(g2, f)
            if g2 == 1: break
        else: return False
        print(pf, pf2)
        return True
    
    res = 0
    solutions = []
    for num in range(2, n_max + 1):
        if primeFactorisationIsStrongAchilles(calculatePrimeFactorisation(num, ps=ps)):
            res += 1
            solutions.append(num)
            print(f"solution found: {num}, current total = {res}")
    print(solutions)
    return res

def strongAchillesNumberCount(n_max: int=10 ** 18 - 1) -> int:
    """
    Solution to Project Euler #302

    Counts the number of strictly positive integers no greater
    than n_max that are strong Achilles.

    An Achilles number is a strictly positive integer that is
    divisible by the square of each of its prime factors and
    is not a perfect power (i.e. it cannot be expressed as
    an integer power greater than 1 of any integer). A strong
    Achilles number is a strictly positive integer that is
    an Achilles number and its Euler totient function value
    (i.e. the number of strictly positive integers less than
    and coprime with the given integer) is also an Achilles
    number.

    Args:
        Optional named:
        n_max (int): Strictly positive integer giving the
                inclusive upper bound on the integers considered
                for inclusion in the count.
            Default: 10 ** 18 - 1
        
        Optional named:
        ps (PrimeSPFsieve object or None): If given, a prime
                sieve object for calculating prime factorisations.
                If not given or given as None, the prime
                factorisations are calculated by direct
                division.
            Default: None
    
    Returns:
    Integer (int) giving the number of strictly positive integers
    no greater than n_max that are strong Achilles.

    Outline of rationale:
    TODO
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
        #print(f"pf = {pf}")
        pf2 = {}
        for p, f in pf.items():
            if f > 1: pf2[p] = pf2.get(p, 0) + (f - 1)
            pf3 = ps.primeFactorisation(p - 1)
            #print(pf3)
            for p2, f2 in pf3.items():
                if f2: pf2[p2] = pf2.get(p2, 0) + f2
            #print(p, f, pf2)
        if min(pf2.values()) < 2: return False
        g2 = 0
        for f in pf2.values():
            g2 = gcd(g2, f)
            if g2 == 1: break
        else: return False
        #print(pf, pf2)
        return True
    
    def iLog(a: int, b: int) -> int:
        # Using binary search
        if a <= 0: raise ValueError("a must be strictly positive")
        elif b <= 1: raise ValueError("b must be strictly greater than 1")
        elif a < b: return 0
        lo, hi = 1, 1
        while b ** hi <= a:
            lo, hi = hi, hi << 1
        while lo < hi:
            mid = hi - ((hi - lo) >> 1)
            if b ** mid <= a:
                lo = mid
            else: hi = mid - 1
        return lo


    p_lst = ps.p_lst
    #print(f"p_max = {p_max}, p_list length = {len(p_lst)}")
    p_dict = {x: i for i, x in enumerate(p_lst)}
    p_sq_lst = [p * p for p in p_lst]
    m = len(p_lst)
    
    res = [0]
    #curr_pf = [0] * m
    curr_totient_pf = [0] * m
    curr_incomplete_totient = SortedSet()

    #solutions = []

    def recur(idx: int=m - 1, remain_mx: int=n_max, g1: int=0, g2: int=0, num: int=1, tot: int=1) -> None:
        #if num in {20000}:
        #    print(idx, num, tot, remain_mx, g1, g2, curr_incomplete_totient, curr_totient_pf)
        if idx < 0 or remain_mx < 2:
            
            g2_2 = g2
            for i in reversed(range(idx + 1)):
                if g2_2 == 1: break
                g2_2 = gcd(g2_2, curr_totient_pf[i])
            #if (g1 == 1 and g2 == 1 and not curr_incomplete_totient):
            #    #print(f"solution found: {num} (totient = {tot})")
            #    solutions.append(num)
            res[0] += (g1 == 1 and g2 == 1 and not curr_incomplete_totient)
            return
        p_mx = max(remain_mx, p_lst[idx])
        if curr_incomplete_totient:
            if idx < 0 or p_lst[curr_incomplete_totient[-1]] > p_mx: return
        
        #if curr_pf[idx] != 1:
        #recur(idx=idx - 1, remain_mx=remain_mx, g1=g1, g2=gcd(g2, curr_totient_pf[idx]), num=num, tot=tot)

        if p_lst[idx] > p_mx:
            return
        
        curr_incomplete_totient.discard(idx)
        exp_mn = (2 + (not curr_totient_pf[idx]))
        exp_mx = iLog(remain_mx, p_lst[idx])
        #if num == 30375:
        #    print(f"p = {p_lst[idx]}, remain_mx = {remain_mx}, exp range = [{exp_mn}, {exp_mx}]")
        remain_mx2 = remain_mx
        for _ in range(exp_mn - 1):
            remain_mx2 //= p_lst[idx]
        p_minus_one_pf = ps.primeFactorisation(p_lst[idx] - 1)
        num2 = num * p_lst[idx] ** (exp_mn - 1)
        tot2 = tot * p_lst[idx] ** (exp_mn - 2) * (p_lst[idx] - 1)
        for p, f in p_minus_one_pf.items():
            if not f: continue
            p_idx = p_dict[p]
            #print(idx, p_idx)
            if curr_totient_pf[p_idx] == 1:
                curr_incomplete_totient.remove(p_idx)
            elif f == 1 and not curr_totient_pf[p_idx]:
                curr_incomplete_totient.add(p_idx)
            curr_totient_pf[p_idx] += f
        curr_totient_pf[idx] += exp_mn - 2
        idx2_mx = idx - 1
        g2_2 = g2
        #print(f"exp range = [{exp_mn}, {exp_mx}]")
        for exp in range(exp_mn, exp_mx + 1):
            curr_totient_pf[idx] += 1
            num2 *= p_lst[idx]
            tot2 *= p_lst[idx]
            remain_mx2 //= p_lst[idx]
            # Review- may be able to restrict further, as for any prime greater
            # than 3, p - 1 is not prime as 2 is a factor
            if curr_incomplete_totient and curr_incomplete_totient[-1] >= remain_mx2 - 1:
                curr_totient_pf[idx] -= exp - 1
                break
            idx2_mx_prev = idx2_mx
            idx2_mx = min(idx, bisect.bisect_right(p_sq_lst, remain_mx2)) - 1
            for i in reversed(range(idx2_mx + 1, idx2_mx_prev + 1)):
                #print(f"i = {i}")
                if g2_2 == 1: break
                g2_2 = gcd(g2_2, curr_totient_pf[i])
                #if idx == 2 and num2 == 625: print(f"num2 = {num2}, g2 = {g2}, g2_2 = {g2_2}, i = {i}, curr_totient_pf[i] = {curr_totient_pf[i]}")
            g1_2 = gcd(g1, exp)
            g2_3 = g2_2
            for idx2 in reversed(range(idx2_mx + 1)):
                #if idx == 2 and num2 == 625: print(f"num2 = {num2}, g2 = {g2}, g2_2 = {g2_3}")
                recur(idx=idx2, remain_mx=remain_mx2, g1=g1_2, g2=gcd(g2_3, curr_totient_pf[idx]), num=num2, tot=tot2)
                if curr_totient_pf[idx2] == 1: break
                g2_3 = gcd(g2_3, curr_totient_pf[idx2])
            else:
                #if idx == 2 and num2 == 625: print(f"num2 = {num2}, g2 = {g2}, g2_3 = {g2_3}")
                recur(idx=-1, remain_mx=remain_mx2, g1=g1_2, g2=gcd(g2_3, curr_totient_pf[idx]), num=num2, tot=tot2)
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
        if curr_totient_pf[idx] == 1:
            curr_incomplete_totient.add(idx)

    for m_ in range(m):
        if not m_ % 1000:
            print(f"computing for largest prime index {m_} (p = {p_lst[m_]}), max prime index = {m - 1}")
        recur(idx=m_, remain_mx=n_max)
    #print(sorted(solutions))
    return res[0]

# Problem 303
def smallestMultiplierGivingMultipleWithDigitValueUpperBound(
    num: int,
    max_dig_val: int,
    base: int=10,
) -> int:
    """
    For strictly positive integer num, calculates the smallest strictly
    positive integer such that the product of that integer with num
    has a representation in the chosen base containing only digits
    with value no greater than max_dig_val if such a value exists,
    otherwise -1.

    Args:
        Required positional:
        num (int): Strictly positive integer giving the number for
                which the product with the returned value (if any)
                should when represented in the chosen base contain no
                digit with value greater than max_dig_val.
        max_dig_val (int): Strictly positive integer less than base
                giving the largest digit value allowed in the
                representation in the chosen base of the product of num
                and the returned value.

        Optional named:
        base (int): Integer strictly greater than 1 giving the base in
                which the product of num and the returned value (if any)
                should be represented when assessing whether it
                contains any digit with value exceeding max_dig_val.
            Default: 10

    Returns:
    Integer (int) giving the smallest strictly positive integer such
    that the product of that integer with num as a representation in
    the chosen base containing only digits with value no greater than
    max_dig_val if such a value exists, otherwise -1.

    Outline of rationale:
    TODO
    """
    digs = []
    num2 = num
    while num2:
        num2, d = divmod(num2, base)
        digs.append(d)
    #n_dig = len(digs)

    q = deque([0])
    res = float("inf")
    for idx in itertools.count(0):
        if not q or isinstance(res, int): break
        mul = base ** idx
        for _ in range(len(q)):
            suff = q.popleft()
            for d in range(not suff, base):
                suff2 = suff + mul * d
                if suff2 >= res: continue
                num2 = suff2 * num
                num3 = num2 // mul
                d2 = num3 % base
                if d2 > max_dig_val: continue
                #if num == 103:
                #    print(suff2)
                num3 //= base
                while num3:
                    num3, d2 = divmod(num3, base)
                    if d2 > max_dig_val: break
                else:
                    res = min(res, suff2)
                    continue
                q.append(suff2)

    return res if isinstance(res, int) else -1


def smallestMultiplierGivingMultipleWithDigitValueUpperBoundSum(
    num_max: int=10 ** 4,
    max_dig_val: int=2,
    base: int=10,
) -> int:
    """
    Solution to Project Euler #303

    For each integer between 1 and num_max inclusive, calculates
    the smallest strictly positive integer whose product with the
    given integer has a representation in the chosen base containing
    only digits with value no greater than max_dig_val and returns
    the sum of these integers if such a value exists for all such
    integers, otherwise returns -1.

    Args:
        Optional named:
        num_max (int): Strictly positive integer giving the inclusive
                upper bound on the numbers whose integer multipliers
                are to be included in the returned sum.
            Default: 10 ** 4
        max_dig_val (int): Strictly positive integer less than base
                giving the largest digit value allowed in the
                product of each integer with the multiplier it
                contributes to the returned sum when represented in
                the chosen base.
            Default: 2
        base (int): Integer strictly greater than 1 giving the base in
                integers should be represented when assessing whether it
                contains any digit with value exceeding max_dig_val.
            Default: 10

    Returns:
    Integer (int) giving the sum of each of the smallest strictly positive
    integer whose product with the integers between 1 and num_max inclusive
    has a representation in the chosen base containing only digits with
    value no greater than max_dig_val if such a value exists for all such
    integers, otherwise -1.

    Outline of rationale:
    See outline of rationale section in the documentation of the function
    smallestMultiplierGivingMultipleWithDigitValueUpperBound().
    """
    res = 0
    for num in range(1, num_max + 1):
        mul = smallestMultiplierGivingMultipleWithDigitValueUpperBound(
            num,
            max_dig_val,
            base=base,
        )
        print(num, mul, num * mul)
        if mul < 0: return -1
        res += mul
    return res

# Problem 304
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

# Problem 305
def consecutiveNumbersDigitsGenerator(
    base: int=10,
) -> Generator[int, None, None]:
    digs = [1]
    while True:
        #print(digs)
        for d in reversed(digs):
            yield d
        carry = 1
        for i in range(len(digs)):
            carry, digs[i] = divmod(digs[i] + carry, base)
            if not carry: break
        else: digs.append(1)
        
    return

def startingPositionsInNumberConcatenator(
    num: int,
    base: int=10,
) -> Generator[int, None, None]:
    if num < 0: return
    num_digs = []
    num2 = num
    while num2:
        num2, d = divmod(num2, base)
        num_digs.append(d)
    num_digs = num_digs[::-1]
    if not num_digs: num_digs = [0]



    kmp = KnuthMorrisPratt(num_digs)
    num_concat_gen = consecutiveNumbersDigitsGenerator(base)
    yield from kmp.matchStartGenerator(num_concat_gen)
    return

def reflexivePositionsInNumberConcatenatorPowersSumBruteForce(
    a: int=3,
    k_min: int=1,
    k_max: int=13,
    base: int=10,
) -> int:
    res = 0
    for k in range(k_min, k_max + 1):
        since = time.time()
        num = a ** k
        for _, term in zip(range(num), startingPositionsInNumberConcatenator(
            num,
            base=base,
        )):
            pass
        term += 1
        print(f"term for k = {k} is {term} found in {time.time() - since:.4f} seconds")
        res += term
    return res

def nthStartingPositionOfTargetInNumberConcatenator(
    n: int,
    target: int,
    base: int=10,
) -> int:
    
    target_digs = []
    num2 = target
    while num2:
        num2, d = divmod(num2, base)
        target_digs.append(d)
    target_digs = target_digs[::-1]
    if not target_digs: target_digs = [0]
    target_n_dig = len(target_digs)

    def nDigitsLECount(
        n_dig: int,
        idx_mx: int,
    ) -> int:
        #if idx_mx >= n_d * (base - 1) * base ** (n_d - 1)
        
        int_mx0, r = divmod(idx_mx, n_dig)
        
        int_mx = int_mx0 + base ** (n_dig - 1)
        
        int_mx1 = int_mx - 1
        int_mx2 = int_mx
        int_mx1_digs = [0] * n_dig
        int_mx2_digs = [0] * n_dig
        int_mx1_2 = int_mx1
        int_mx2_2 = int_mx2
        for j in reversed(range(n_dig)):
            int_mx1_2, d1 = divmod(int_mx1_2, base)
            int_mx1_digs[j] = d1
            int_mx2_2, d2 = divmod(int_mx2_2, base)
            int_mx2_digs[j] = d2
        
        def countWhenTargetInsideInteger(i0: int) -> int:
            #int_mx, r = divmod(idx_mx, n_dig)
            ##print(f"n_dig = {n_dig}, idx_mx = {idx_mx}, int_mx0 = {int_mx}")
            #int_mx += base ** (n_dig - 1) - (r < i0)# - 1 - (r < i0)
            #
            #int_mx_digs = [0] * n_dig
            #int_mx2 = int_mx
            #for j in reversed(range(n_dig)):
            #    int_mx2, d = divmod(int_mx2, base)
            #    int_mx_digs[j] = d

            int_mx, int_mx_digs = [int_mx1, int_mx1_digs] if r < i0 else [int_mx2, int_mx2_digs]

            def recur(idx: int, tight: bool=True) -> int:
                if idx == n_dig:
                    return 1
                if not tight:
                    res = 1 if idx else (base - 1)
                    idx2 = max(idx, 1)
                    res *= base ** (max(0, n_dig - max(idx2, i0 + target_n_dig)) + max(0, i0 - idx2))
                    #print(idx, i0 + target_n_dig, n_dig - max(1, idx, i0 + target_n_dig), i0 - idx)
                    #print(f"idx = {idx}, tight = {tight}, res = {res}")
                    return res
                if idx >= i0 and idx < i0 + target_n_dig:
                    mx = int_mx_digs[idx]
                    tight2 = target_digs[idx - i0] == mx
                    return 0 if target_digs[idx - i0] > mx else recur(idx + 1, tight=tight2)# + tight2 * (int_mx % base ** (n_dig - idx - 1))
                return (int_mx_digs[idx] - (not idx)) * recur(idx + 1, tight=False) + recur(idx + 1, tight=True)
            
            res = recur(0, tight=True)
            #print(f"countWhenTargetIsInsideInteger({i0}) when n_dig = {n_dig}, idx_mx = {idx_mx} (int_mx = {int_mx}, dig_idx = {r}) is {res}")
            return res

        def countWhenTargetStraddlesIntegers(i0: int) -> int:
            int_mx, int_mx_digs = [int_mx1, int_mx1_digs] if r < i0 else [int_mx2, int_mx2_digs]
            #print(int_mx_digs)
            
            tail_len = n_dig - i0
            head_len = target_n_dig - tail_len
            if not target_digs[tail_len]: return 0 # The integers cannot start with the digit 0
            #tight = False
            #for i in range(head_len):
            #    diff = target_digs[target_n_dig - head_len + i] - int_mx_digs[i]
            #    if diff > 0: return 0
            #    elif diff < 0: break
            #else: tight = True
            #head_tail_delta = n_dig - target_n_dig
            memo = {}
            def recur(idx: int, carry: bool, tight: bool=True) -> int:
                #print(f"calling recur() with idx = {idx}, carry = {carry}, tight = {tight}")
                if idx == n_dig:
                    #if carry: print("solution found")
                    return int(carry)
                args = (idx, carry, tight)
                if args in memo.keys():
                    return memo[args]
                res = 0
                head_idx = idx + tail_len
                tail_idx = head_idx - n_dig
                if idx < head_len:
                    #print("in head")
                    if carry:
                        if idx and not target_digs[head_idx] and (tail_idx < 0 or target_digs[tail_idx] == base - 1) and (not tight or int_mx_digs[idx] == base - 1):
                            #print("hi1")
                            res += recur(idx + 1, True, tight=tight)
                    #elif tight and target_digs[head_idx] > int_mx_digs[idx]:
                    #    res = 0
                    elif tail_idx >= 0:
                        #print("in tail")
                        if target_digs[head_idx] == target_digs[tail_idx] and (idx or target_digs[tail_idx]) and (not tight or target_digs[head_idx] <= int_mx_digs[idx]):
                            #print("hi2")
                            res += recur(idx + 1, False, tight=(tight and target_digs[head_idx] == int_mx_digs[idx]))
                        elif target_digs[head_idx] - 1 == target_digs[tail_idx] and (idx or target_digs[tail_idx]) and (not tight or target_digs[head_idx] - 1 <= int_mx_digs[idx]):
                            #print("hi3")
                            res += recur(idx + 1, True, tight=(tight and target_digs[head_idx] - 1 == int_mx_digs[idx]))
                    elif tight:
                        if target_digs[head_idx] - 1 <= int_mx_digs[idx]:
                            
                            if target_digs[head_idx] and (idx or target_digs[head_idx] - 1 > 0):
                                #print("hi4")
                                #print(f"head_idx = {head_idx}")
                                res += recur(idx + 1, True, tight=(target_digs[head_idx] - 1 == int_mx_digs[idx]))
                            if target_digs[head_idx] <= int_mx_digs[idx] and (idx or target_digs[head_idx] > 0):
                                #print("hi5", target_digs[head_idx], int_mx_digs[idx])
                                res += recur(idx + 1, False, tight=(target_digs[head_idx] == int_mx_digs[idx]))
                    elif idx or target_digs[head_idx] > 0:
                        #print("hi6", target_digs[head_idx], int_mx_digs[idx])
                        if target_digs[head_idx] > 0:
                            res += recur(idx + 1, True, tight=False)
                        if idx or target_digs[head_idx] - 1 > 0:
                            #print("hi7")
                            res += recur(idx + 1, False, tight=False)
                        
                            
                elif tail_idx >= 0:
                    #print("in tail")
                    if carry:
                        if target_digs[tail_idx] == base - 1 and (not tight or int_mx_digs[idx] == base - 1):
                            #print("hi8")
                            res += recur(idx + 1, True, tight=tight)
                    #elif target_digs[tail_idx] == base - 1: pass # No carry when it is needed
                    elif tight:
                        if target_digs[tail_idx] <= int_mx_digs[idx] and (idx or target_digs[tail_idx]):
                            #print("hi9")
                            res += recur(idx + 1, False, tight=(target_digs[tail_idx] == int_mx_digs[idx]))
                            if target_digs[tail_idx] < base - 1:
                                res += recur(idx + 1, True, tight=(target_digs[tail_idx] == int_mx_digs[idx]))
                    elif idx or target_digs[tail_idx]:
                        #print("hi10")
                        res += recur(idx + 1, False, tight=False)
                        if target_digs[tail_idx] < base - 1:
                            res += recur(idx + 1, True, tight=False)
                elif carry:
                    if (not tight or int_mx_digs[idx] == base - 1):
                        #print("hi11")
                        res += recur(idx + 1, True, tight=tight)
                elif tight:
                    #print("hi12")
                    res += max(0, int_mx_digs[idx] - (not idx)) * (recur(idx + 1, False, tight=False) + recur(idx + 1, True, tight=False))
                    res += recur(idx + 1, False, tight=True)
                    if int_mx_digs[idx] < base - 1:
                        #print("hi13")
                        res += recur(idx + 1, True, tight=True)
                else:
                    #print("hi14")
                    res += (base - (not idx)) * recur(idx + 1, False, tight=False) + (base - 1 - (not idx)) * recur(idx + 1, True, tight=False)

                memo[args] = res
                return res

            res = recur(0, carry=False, tight=True)# + recur(0, carry=True, tight=True)
            #print(f"memo for i0 = {i0}: {memo}")
            #print(f"countWhenTargetStraddlesIntegers({i0}) = {res}")

            
            # Check for transition to the next digit count (e.g. in base 10
            # 999...99 to 1000...00)
            nxt_ndig_idx = n_dig * ((base - 1) * base ** (n_dig - 1) - 1) + i0
            if nxt_ndig_idx > idx_mx: return res
            for idx in range(tail_len):
                if target_digs[idx] != base - 1: return res
            if head_len >= 1 and target_digs[tail_len] != 1: return res
            for idx in range(tail_len + 1, target_n_dig):
                if target_digs[idx]: return res
            
            return res + 1

            """
            tail_one_less_base_pow = True
            #print(f"i0 = {i0}, head_len = {head_len}, tail_len = {tail_len}, target_digs = {target_digs}, tight = {tight}")
            # Review
            for i in range(tail_len):
                if target_digs[i] != base - 1:
                    tail_one_less_base_pow = False
            if not tail_one_less_base_pow:
                if not target_digs[-head_len]:
                    #print("hi1")
                    return 0
                overlap = (tail_len + head_len) - n_dig
                #carry = 1
                #print(f"overlap = {overlap}")
                for i in range(overlap):
                    if target_digs[i] != target_digs[target_n_dig - overlap + i]:
                        #print("hi2", i)
                        return 0
                if not tight:
                    #print(n_dig, i0, target_digs, head_len, tail_len, max(-overlap, 0))
                    return base ** max(-overlap, 0)
                # TODO
                #print("hi3")
                return 0
            #print("hi4")
            return 0
            """
        
        def countWhenTargetContainsWholeInteger(i0: int) -> int:
            i_start = n_dig - i0 if i0 else 0
            if not target_digs[i_start]:
                #print("hi0")
                return 0
            contained_int_digs_rev = [None] * n_dig
            contained_int = 0
            int_idx = 0
            is_base_pow = True
            for i in reversed(range(i_start, i_start + n_dig)):
                contained_int = base * contained_int + target_digs[i]
                contained_int_digs_rev[~(i - i_start)] = target_digs[i]
                if target_digs[i] != int(i == i_start):
                    is_base_pow = False
                #int_idx = int_idx * base + target_digs[i]
            for i in range(i_start, i_start + n_dig):
                int_idx = int_idx * base + target_digs[i]
            #idx = int_idx * n_dig - ((base - (n_dig + 1) * base ** (n_dig + 1) + n_dig * base ** (n_dig + 1)) // ((1 - base) ** 2)) - i_start
            #print(int_idx, contained_int)
            if is_base_pow and i_start:
                if not i_start or i + 1 == target_n_dig:
                    #print("hi1")
                    return 0 # Should not happen as this case should be handled by countWhenTargetStraddlesIntegers()
                if target_digs[i + 1]:
                    #print("hi2")
                    return 0
                contained_int *= base
                contained_int_digs_rev.append(1)
                int_idx *= base
            #print(int_idx)
            #idx = int_idx * n_dig - ((base - (n_dig + 1) * base ** (n_dig + 1) + n_dig * base ** (n_dig + 1)) // ((1 - base) ** 2)) - i_start #n_dig * (contained_int - base ** (n_dig - 1))
            idx = n_dig * (int_idx - base ** (n_dig - 1)) - i_start
            #print(f"idx = {idx}, idx_mx = {idx_mx}, target_digs = {target_digs}")
            #idx -= i_start
            if idx > idx_mx: return 0

            #print(f"contained_int_digs_rev = {contained_int_digs_rev}")

            carry = 1
            for i in range(i_start):
                carry, d = divmod(contained_int_digs_rev[i] - carry, base)
                #print(f"previous digit {i} = {d}")
                if d != target_digs[i_start - i - 1]:
                    #print("hi")
                    return 0
            curr_int_digs_rev = list(contained_int_digs_rev)
            i = i_start + len(curr_int_digs_rev)
            while i < target_n_dig:
                carry = 1
                for j in range(len(curr_int_digs_rev)):
                    carry, curr_int_digs_rev[j] = divmod(curr_int_digs_rev[j] + carry, base)
                    if not carry: break
                else: curr_int_digs_rev.append(1)
                #print(f"next digits reversed = {curr_int_digs_rev}")
                for j in range(min(len(curr_int_digs_rev), target_n_dig - i)):
                    if target_digs[i + j] != curr_int_digs_rev[~j]: return 0
                i += len(curr_int_digs_rev)
            return 1
            """
            if i_start:
                num2 = contained_int - 1
                for i in reversed(range(i_start)):
                    num2, d = divmod(num2, base)
                    if target_digs[i] != d: return 0
            num2 = contained_int
            i_start2 = i_start
            for j in range(i_start + n_dig, target_n_dig - n_dig, n_dig):
                num2 += 1
                num3 = num2
                i_start2 += n_dig
                for i in reversed(range(i_start2, i_start2 + n_dig)):
                    num3, d = divmod(num3, base)
                    if target_digs[i] != d: return 0
            if not (target_n_dig - i_start) % n_dig:
                return 1
            num2 += 1
            num3 = num2
            i_start2 += n_dig
            for i in reversed(range(i_start2, i_start2 + n_dig)):
                num3, d = divmod(num3, base)
                if i < target_n_dig and target_digs[i] != d: return 0
            return 1
            """
        
        
        res = 0

        trans1 = max(0, n_dig - target_n_dig + 1)
        trans2 = min(n_dig, max(0, 2 * n_dig - target_n_dig + 1))
        #print(f"n_dig = {n_dig}, trans1 = {trans1}, trans2 = {trans2}")
        for i0 in range(trans1):
            cnt1 = countWhenTargetInsideInteger(i0)
            res += cnt1
            #print(f"countWhenTargetInsideInteger({i0}) = {cnt1}")
        for i0 in range(trans1, trans2):
            cnt2 = countWhenTargetStraddlesIntegers(i0)
            res += cnt2
            #print(f"countWhenTargetStraddlesIntegers({i0}) = {cnt2}")
        for i0 in range(trans2, n_dig):
            cnt3 = countWhenTargetContainsWholeInteger(i0)
            res += cnt3
            #print(f"countWhenTargetContainsWholeInteger({i0}) = {cnt3}")

        int_mx, r = divmod(idx_mx, n_dig)
        #print(int_mx, r)
        int_mx += base ** (n_dig - 1)
        #print(f"nDigitsLECount(n_dig={n_dig}, idx_mx={idx_mx}) = {res} (int_mx = {int_mx}, dig_idx = {r})")

        return res
    
    n2 = n
    idx1 = -1
    res = 0
    for n_dig in itertools.count(1):
        #print(f"n_dig = {n_dig}, index count for lower n_dig = {res}, current n2 = {n2}")
        idx1 = n_dig * ((base - 1) * base ** (n_dig - 1)) - 1
        m = nDigitsLECount(n_dig, idx1)
        if m >= n2: break
        n2 -= m
        #print(f"new n2 = {n2}")
        res += idx1 + 1
    lo, hi = 0, idx1
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if nDigitsLECount(n_dig, mid) < n2:
            lo = mid + 1
        else: hi = mid
    return res + lo

def reflexivePositionsInNumberConcatenatorPowersSum(
    a: int=3,
    k_min: int=1,
    k_max: int=13,
    base: int=10,
) -> int:
    res = 0
    for k in range(k_min, k_max + 1):
        since = time.time()
        num = a ** k
        term = nthStartingPositionOfTargetInNumberConcatenator(
            n=num,
            target=num,
            base=base,
        )
        term += 1
        res += term
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

# Problem 307
def partitionsGenerator(
    num: int,
    part_size_min: Optional[int]=None,
    part_size_max: Optional[int]=None,
    n_part_min: Optional[int]=None,
    n_part_max: Optional[int]=None,
) -> Generator[Dict[int, int], None, None]:
    """
    Generator iterating over every possible unordered integer partition
    of the non-negative integer num, for which (if specified) all
    part have size between part_size_min and part_size_max inclusive
    and the number of parts in each partition is between
    (if specified) n_part_min and n_part_max inclusive.

    Args:
        Required positional:
        num (int): Non-negative integer giving the number to be
                partitioned (i.e. the number to which the size of each
                part in a yielded partition is to sum)
        
        Optional named:
        part_size_min (int or None): If specified as a non-negative integer,
                the smallest possible part size in any partition yielded,
                otherwise the smallest possible part size is 1.
            Default: None (i.e. the smallest part size is 1)
        part_size_max (int or None): If specified as an integer,
                the largest possible part size in any partition yielded,
                otherwise there is no upper bound on the part size
            Default: None (i.e. the part size has no upper bound)
        n_part_min (int or None): If specified as a non-negative integer,
                the smallest possible number of parts in any partition yielded,
                otherwise the smallest possible number of parts is 1.
            Default: None (i.e. the smallest number of parts is 1)
        n_part_max (int or None): If specified as an integer, the smallest
                possible number of parts in any partition yielded, otherwise
                the smallest possible number of parts is 1.
            Default: None (i.e. the number of parts has no upper bound)

    Yields:
    Dictionary representing an unordered integer partition of num, where the
    keys are integers giving the sizes of the parts and the corresponding
    values are strictly positive integers giving the number of occurrences
    of that part size in the given partition.
    Collectively, these represent all possible unordered integer partitions
    of num subject to the restrictions placed by the parameters part_size_min,
    part_size_max, n_part_min and n_part_max, with each such distinct
    unordered integer partition yielded exactly onced and in no particular
    order.
    """
    part_size_min = 1 if part_size_min is None else max(part_size_min, 1)
    n_part_min = 1 if n_part_min is None else max(n_part_min, 1)

    part_size_max = num if part_size_max is None else min(part_size_max, num // n_part_min)
    n_part_max = num if n_part_max is None else min(n_part_max, num // part_size_min)
    

    if part_size_min > part_size_max: return
    if n_part_min > n_part_max: return

    if part_size_min * n_part_min > num: return
    if part_size_max * n_part_max < num: return
    
    curr = {}

    def recur(remain: int, part_size: int, n_parts_remain_min: int, n_parts_remain_max: int) -> Generator[Dict[int, int], None, None]:
        #if not remain:
        #    print("hi1", part_size, curr)
        #    yield dict(curr)
        #    return
        if n_parts_remain_max <= 0 or remain < part_size_min or part_size * n_parts_remain_max < remain:
            return
        elif n_parts_remain_max == 1:
            if remain > part_size: return
            curr[remain] = 1
            #print("hi3", curr)
            yield dict(curr)
            curr.pop(remain)
            return
        elif part_size == part_size_min:
            f, r = divmod(remain, part_size)
            if r: return
            curr[part_size] = f
            #print("hi3", curr)
            yield dict(curr)
            curr.pop(part_size)
            return

        curr[part_size] = 0
        remain2 = remain
        f_mx = min(n_parts_remain_max, remain // part_size)
        f_mx = min(f_mx, (remain - n_parts_remain_min) // (part_size - part_size_min))
        for f in range(1, f_mx + 1):
            curr[part_size] += 1
            remain2 -= part_size
            if not remain2:
                yield dict(curr)
                break
            for part_size2 in reversed(range(part_size_min, part_size)):
                if part_size2 * n_parts_remain_max - f < remain2:
                    break
                yield from recur(remain2, part_size2, n_parts_remain_min - f, n_parts_remain_max - f)

        curr.pop(part_size)
    
    for part_size0 in range(part_size_min, part_size_max + 1):
        yield from recur(num, part_size0, n_part_min, n_part_max)
    return

def proportionOfBallAllocationsIntoBinsWithOneBinWithAtLeastGivenNumberFraction(
    n_bins: int,
    n_balls: int,
    n_balls_in_bin_maxmin: int,
) -> CustomFraction:
    """
    Calculates the proportion of all ways of allocating n_balls distinguishable
    balls among n_bins distinguishable bins for which at least one of the
    bins contains at least n_balls_in_bin_maxmin balls as a fraction.

    Args:
        Required positional:
        n_bins (int): Strictly positive integer giving the number of
                distinguishable bins among which the balls are to be
                distributed.
        n_balls (int): Non-negative integer giving the number of
                distinguishable balls to be distributed among the bins.
        n_balls_in_bin_maxmin (int): Non-negative integer giving the number
                of balls that must be in at least one of the bins for a
                given configuration to contribute to the returned
                proportion.
    
    Returns:
    CustomFraction object representing the proportion of all ways of
    allocating n_balls distinguishable balls among n_bins distinguishable
    bins for which at least one of the bins contains at least
    n_balls_in_bin_maxmin balls as a rational number.

    Outline of rationale:
    TODO
    """
    def multinomial(nums: List[int]) -> int:
        tot = sum(nums)
        res = 1
        for num in nums:
            res *= math.comb(tot, num)
            tot -= num
        return res

    tot_n_allocations = n_bins ** n_balls
    #print(tot_n_allocations)
    cnt = 0
    for part in partitionsGenerator(
        n_balls,
        part_size_min=None,
        part_size_max=n_balls_in_bin_maxmin - 1,
        n_part_min=None,
        n_part_max=n_bins,
    ):
        #print(part)
        remain = n_bins
        remain2 = n_balls
        curr = 1
        for num, f in part.items():
            #print(remain, remain2, f)
            curr *= math.comb(remain, f) * math.comb(remain2, f * num) * multinomial([num] * f)
            remain -= f
            remain2 -= f * num
        #print(part, curr)
        
        cnt += curr

    return CustomFraction(tot_n_allocations - cnt, tot_n_allocations)

def proportionOfBallAllocationsIntoBinsWithOneBinWithAtLeastGivenNumberFloat(
    n_bins: int=10 ** 6,
    n_balls: int=2 * 10 ** 4,
    n_balls_in_bin_maxmin: int=3,
) -> float:
    """
    Solution to Project Euler #307
    
    Calculates the proportion of all ways of allocating n_balls distinguishable
    balls among n_bins distinguishable bins for which at least one of the
    bins contains at least n_balls_in_bin_maxmin balls as a real number.

    Args:
        Optional named:
        n_bins (int): Strictly positive integer giving the number of
                distinguishable bins among which the balls are to be
                distributed.
            Default: 10 ** 6
        n_balls (int): Non-negative integer giving the number of
                distinguishable balls to be distributed among the bins.
            Default: 2 * 10 ** 4
        n_balls_in_bin_maxmin (int): Non-negative integer giving the number
                of balls that must be in at least one of the bins for a
                given configuration to contribute to the returned
                proportion.
            Default: 3
    
    Returns:
    Float representing the proportion of all ways of allocating n_balls
    distinguishable balls among n_bins distinguishable bins for which at
    least one of the bins contains at least n_balls_in_bin_maxmin balls
    as a real number.

    Outline of rationale:
    See outline of rationale section in the documentation for the function
    proportionOfBallAllocationsIntoBinsWithOneBinWithAtLeastGivenNumberFraction().
    """
    res = proportionOfBallAllocationsIntoBinsWithOneBinWithAtLeastGivenNumberFraction(
        n_bins,
        n_balls,
        n_balls_in_bin_maxmin,
    )
    #print(res)
    return res.numerator / res.denominator

# Problem 308
def conwayFractanPow2TransitionLengthBruteForce(pow2_init: int) -> Tuple[int, int]:
    state0 = {0: pow2_init}
    # prime indices:
    # 2: 0
    # 3: 1
    # 5: 2
    # 7: 3
    # 11: 4
    # 13: 5
    # 17: 6
    # 19: 7
    # 23: 8
    # 29: 9
    incr_dicts = [{6:  1}, {0: 1, 1: 1, 5: 1}, {7: 1}, {8: 1}, {9: 1}, {3: 1, 4: 1}, {2: 1, 7: 1}, {3: 1, 4: 1}, {}, {4: 1}, {5: 1}, {1: 1, 2: 1}, {}, {2: 1, 4: 1}]
    decr_dicts = [{3: 1, 5: 1}, {2: 1, 6: 1}, {1: 1, 6: 1}, {0: 1, 7: 1}, {1: 1, 4: 1}, {9: 1}, {8: 1}, {7: 1}, {6: 1}, {5: 1}, {4: 1}, {0: 1}, {3: 1}, {}]
    
    state = dict(state0)
    seq = []
    for i in itertools.count(1):
        for j in range(len(decr_dicts)):
            for p_idx, f in decr_dicts[j].items():
                if f > state.get(p_idx, 0): break
            else:
                seq.append(j)
                for p_idx, f in decr_dicts[j].items():
                    state[p_idx] -= f
                    if not state[p_idx]: state.pop(p_idx)
                for  p_idx, f in incr_dicts[j].items():
                    state[p_idx] = state.get(p_idx, 0) + f
                break
        #print(state)
        #print(state.get(0, 0))
        stt = [0] * 10
        for i, f in state.items():
            stt[i] = f
        print(stt)
        if len(state) == 1 and 0 in state.keys():
            break
    print(seq)
    return (i, state[0])

def conwayFractanPow2TransitionLength(pow2_init: int) -> Tuple[int, int]:

    # Review- look into the FRACTRAN programming language

    # Using https://oeis.org/wiki/Conway%27s_PRIMEGAME

    # prime indices:
    # 2: 0
    # 3: 1
    # 5: 2
    # 7: 3

    # State type indices (contains the given prime)
    # none: 0
    # 11: 1
    # 13: 2
    # 17: 3
    # 19: 4
    # 23: 5
    # 29: 6
    
    state = [0] * 4
    res = pow2_init + 1
    state[1] = pow2_init
    state[2] = pow2_init + 1
    state_typ = 1
    while state_typ or any(state[1:]):
        #print(state, state_typ)
        if not state_typ: # none
            if state[0]:
                res += state[0]
                state[1] += state[0]
                state[2] += state[0]
                state[0] = 0
            if state[3]:
                res += state[3]
                state[3] = 0
            res += 1
            state[2] += 1
            state_typ = 1
        elif state_typ == 1: # 11
            if state[1]:
                res += 2 * state[1]
                state[3] += state[1]
                state[1] = 0
            else:
                res += 1
                state_typ = 2
        elif state_typ == 2: # 13
            if state[3]:
                cnt = min(state[2], state[3])
                res += cnt * 2
                state[0] += cnt
                state[1] += cnt
                state[2] -= cnt
                state[3] -= cnt
                if state[3]:
                    res += 1
                    state[3] -= 1
                    state_typ = 3
                    continue
            res += 1
            state_typ = 1
            continue
        elif state_typ == 3: # 17
            #cnt = min(state[2], state[3])
            #res += 2 * cnt
            #state[0] += cnt
            #state[1] += cnt
            #state[2] -= cnt
            #state[3] -= cnt
            #if state[3]

            res += 1
            if state[2]:
                state[0] += 1
                state[1] += 1
                state[2] -= 1
                state_typ = 2
            elif state[1]:
                state[1] -= 1
                state_typ = 4
            else: state_typ = 0
        elif state_typ == 4: # 19
            cnt = state[0]
            res += 2 * cnt + 1
            state[2] += cnt
            state[0] = 0
            state[3] += 1
            state_typ = 1
        elif state_typ == 5: # 23
            res += 1
            state[2] += 1
            state_typ = 4
        else: # 29
            res += 1
            state[3] += 1
            state_typ = 1
    return (res, state[0])
    """
    # Stripping out the 2s
    state = [0] * 10
    res = pow2_init + 1
    state[1] = pow2_init
    state[2] = pow2_init + 1
    state[4] = 1
    print(f"state post stripping 2s = {state}")

    while state[1]:
        print(f"cycle state state = {state}")
        # Stripping the 3s
        res += state[1] * 2 + 1
        state[3] = state[1]
        state[1] = 0
        state[4] = 0
        state[5] = 1
        print(f"state post stripping 3s = {state}")

        # Stripping the 7s
        cnt = min(state[2], state[3])
        res += cnt * 2
        state[0] += cnt
        state[1] += cnt
        state[2] -= cnt
        state[3] -= cnt
        
        if state[3]:
            res += 1
            state[3] -= 1
            state[5] -= 1
            state[6] += 1
        print(f"state post stripping 7s = {state}")
        

        if not state[3] and state[5]:
            print("hi1")
            res += 1
            state[4] += 1
            state[5] -= 1
            print(f"state post 13-11 swap = {state}")
            #if state[1]: continue
            #if state[1]: continue
            if state[1]: continue
        if (state[3] and not state[5]) and (state[6] and not state[2]):
            state[6] -= 1
        print("hi2")

        #res += 2
        #state[3] -= 1
        #state[5] = 0
        #state[6] = 0
        print(state)
        if not state[1] and not state[2] and not state[3]:
            break
        cnt = min(state[0], state[3])
        res += cnt + 1
        state[0] -= cnt
        state[1] += cnt
        state[2] += cnt + 1
        state[3] -= cnt
        state[4] = 1
    print(f"final state = {state}")
    return (res, state[0])
    """

def conwayFractanPow2PrimeGeneratorStepCount(n_p: int=10 ** 5 + 1) -> int:
    """
    Solution to Project Euler #308
    """
    p = 1
    res = 0
    for i in range(1, n_p + 1):
        if not i % 100:
            print(f"prime number {i} of {n_p} found")
        #print(f"p = {p}")
        n_step, p = conwayFractanPow2TransitionLength(p)
        res += n_step
        #print(f"p = {p}, n_step = {n_step}, cumulative steps = {res}")
    return res

# Problem 309
def integerCrossingLaddersCount(len_max: int=10 ** 6 - 1) -> int:
    """
    Solution to Project Euler #309

    Calculates the number of values of the ordered triple (x, y, h)
    or strictly positive integers exist for which 0 < x < y <= len_max
    where x and y are the lengths of straight line segments between
    two semi-infinite vertical parallel lines an integer distance
    apart both extending upwards from a perpendicular horizontal line
    segment between them, the bottom of the two lines are each at the
    different points the horizontal line meets one of the vertical
    lines, the top ends of the lines falls on the opposite vertical
    line from its bottom end and the vertical distance from the
    horizontal line to the point at which the two lines meet is equal
    to h.
    
    Args:
        Optional named:
        len_max (int): Strictly positive integer giving the inclusive
                upper bound on the length of the two line segments
                in question (x and y).
            Default: 10 ** 6 - 1
    
    Returns:
    Integer (int) giving the number of possible values of the ordered
    triple (x, y, h) of strictly positive integers for which
    0 < x < y <= len_max that satisfy the constraints described above.

    Outline of rationale:
    TODO
    """
    # Review- wording of the documentation for clarity
    res = 0
    seen = {}
    prev_c0 = 0
    print_freq = 10 ** 4
    for triple, _ in pythagoreanTripleGeneratorByHypotenuse(primitive_only=False, max_hypotenuse=len_max):
        if (triple[2] // print_freq > prev_c0 // print_freq):
            print(f"hypotenuse = {triple[2]} (max {len_max})")
            prev_c0 = triple[2]
        for a, b1 in [(triple[0], triple[1]), (triple[1], triple[0])]:
            seen.setdefault(a, [])
            for b2 in seen[a]:
                res += not (b1 * b2) % (b1 + b2)
                #if not (b1 * b2) % (b1 + b2):
                #    print(a, (b1 * b2) // (b1 + b2))
            seen[a].append(b1)
    return res

# Problem 310
def squareNimbers(n_max: int) -> List[int]:
    """
    For a nim-type game with one pile of stones where on each
    turn a player may only remove a perfect square number
    of stones, where a player loses when there is no permitted
    number of stones they can remove, calculates the nimber
    for the game for each of the numbers of stones remaining
    between 0 and n_max inclusive.

    Args:
        Required positional:
        n_max (int): Non-negative integer giving the inclusive
                upper bound on the size of piles of stones for
                which the nimber in the described game is to
                be calculated.

    Returns:
    List of integers (int) with length (n_max + 1) where the
    element at index i (0-indexed) gives the nimber for the
    described game for a pile of i stones.
    """
    res = [0]
    for i in range(1, n_max + 1):
        seen = set()
        for j in range(1, isqrt(i) + 1):
            i2 = i - j * j
            seen.add(res[i2])
        seen = sorted(seen)
        #print(i, seen)
        lo, hi = 0, len(seen)
        while lo < hi:
            mid = lo + ((hi - lo) >> 1)
            if seen[mid] == mid:
                lo = mid + 1
            else: hi = mid
        res.append(lo)
    return res

def nimSquarePositionsLostByNextPlayerCount(n_max: int=10 ** 5) -> int:
    """
    Solution to Project Euler #310

    For a nim-type game with three pile of stones where on each
    turn a player may only remove a perfect square number
    of stones from one of the piles, where a player loses when
    there is no permitted number of stones they can remove from
    any of the piles, calculates the number of configurations of
    piles for which none of the piles has more than n_max stones
    for which with perfect play the player to move next cannot
    force a win, where configurations of piles that are permutations
    of each other are not considered to be distinct.

    Args:
        Optional named:
        n_max (int): Non-negative integer giving the inclusive
                upper bound on the size of piles of stones for
                which the number of states the player to move
                next cannot force a win is to be calculated.
            Default: 10 ** 5

    Returns:
    Integer (int) giving the number of distinct configurations of
    three piles of stones where no pile contains more than n_max
    stones and configurations that are permutations of each other
    are not considered distinct, for which with perfect play the
    player to move next cannot force a win.

    Outline of rationale:
    TODO
    """
    # Review- try to generalise to any number of piles

    # Using Sprague-Grundy
    nimbers = squareNimbers(n_max)
    #print(nimbers)
    print(max(nimbers))
    nimbers_f_lst = [0] * (max(nimbers) + 1)
    for num in nimbers:
        nimbers_f_lst[num] += 1
    #print(len(nimbers_f_lst))
    #print(nimbers_f_lst)
    f0 = nimbers_f_lst[0]
    res = (f0 * (f0 + 1) * (f0 + 2)) // 6
    for i2 in range(1, len(nimbers_f_lst)):
        f2 = nimbers_f_lst[i2]
        for i1 in range(i2):
            i3 = i1 ^ i2
            if i3 <= i2 or i3 >= len(nimbers_f_lst): continue
            res += nimbers_f_lst[i1] * f2 * nimbers_f_lst[i3]
        res += ((f2 * (f2 + 1)) >> 1) * nimbers_f_lst[0]
    
    return res

# Problem 311
def sumOfTwoSquaresSolutionsCount(
    target: int,
    ps: Optional[PrimeSPFsieve]=None,
) -> int:
    """
    Calculates the number of ordered integer pairs (a, b) for which:
        a ** 2 + b ** 2 = target
    Note that this includes solutions for which a and b are equal and
    for which a and/or be is/are neative.
    """
    res = 1
    if ps is not None:
        pf = calculatePrimeFactorisation(target, ps=ps)
        gaussian_pf = {}
        mult = 1
        for p, f in pf.items():
            if p == 2: continue
            residue = p % 4
            if residue == 3:
                if f & 1:
                    #print(p, f)
                    return 0
                continue
            res *= f + 1
    else:
        num = target
        while not num & 1:
            num >>= 1
        for p in range(3, isqrt(num) + 1, 2):
            if p * p > num: break
            f = 0
            num2, r = divmod(num, p)
            while not r:
                num = num2
                f += 1
                num2, r = divmod(num, p)
            if not f: continue
            if p & 3 == 3:
                if f & 1: return 0
                continue
            res *= f + 1
        if num > 1:
            if num & 3 == 3: return 0
            res <<= 1
    return res << 2

def sumOfTwoDistinctSquaresUnorderedPositiveSolutionsCount(
    target: int,
    ps: Optional[PrimeSPFsieve]=None,
) -> int:
    res = sumOfTwoSquaresSolutionsCount(
        target,
        ps=ps,
    )
    return res >> 3

def legendreSymbol(a: int, p: int, ps: Optional[PrimeSPFsieve]=None) -> int:
    # p must be an odd prime
    a %= p
    if not a: return 0
    elif a == 1: return 1
    elif a == p - 1: return 1 if p % 4 == 1 else -1
    pf = calculatePrimeFactorisation(a, ps=ps)
    res = 1
    for p2, f in pf.items():
        if not f & 1: continue
        if p2 == 2:
            res *= 1 if p % 8 in {1, 7} else -1
            continue
        res *= (-1) ** (((p - 1) * (p2 - 1)) >> 2) * legendreSymbol(p, p2)
    return res


def sumOfTwoSquaresEqualToPrime(p: int) -> Optional[Tuple[int, int]]:
    # Assumes that p is indeed prime
    if p == 2:
        return (1, 1)
    residue = p % 4
    if residue == 3:
        return None
    #for x in range(1, (p + 1) >> 1):
    #    if pow(x, 2, p) == p - 1:
    #        break
    # Using quadratic residues
    for a in range(2, p):
        if legendreSymbol(a, p) == -1:
            x = pow(a, (p - 1) >> 2, p)
            break
    else: raise ValueError(f"Could not find a quadratic non-residue of {p}")
    #print(f"{a}, {x}, {legendreSymbol(a, p)}")
    target = isqrt(p - 1)
    a, b = sorted([p, x])
    while b > target:
        a, b = sorted([a, b % a])
    return (a, b)
    

def sumOfTwoSquaresSolutionGenerator(
    target: int,
    ps: Optional[PrimeSPFsieve]=None,
) -> Generator[Tuple[int, int], None, None]:
    if ps is not None:
        pf = calculatePrimeFactorisation(target, ps=ps)
        gaussian_pf = {}
        mult = 1
        for p, f in pf.items():
            if p == 2:
                gaussian_pf[(1, 1)] = f
                continue
            residue = p % 4
            if residue == 3:
                if f & 1:
                    #print(p, f)
                    return
                mult *= p ** (f >> 1)
                continue
            pair = sumOfTwoSquaresEqualToPrime(p)
            gaussian_pf[pair] = f
    else:
        gaussian_pf = {}
        mult = 1
        num = target
        f = 0
        while not num & 1:
            f += 1
            num >>= 1
        if f:
            gaussian_pf[(1, 1)] = f
        odd_pf = {}
        #print(num)
        for p in range(3, isqrt(num) + 1, 2):
            if p * p > num: break
            f = 0
            num2, r = divmod(num, p)
            while not r:
                num = num2
                f += 1
                num2, r = divmod(num, p)
            if not f: continue
            if p & 3 == 3:
                if f & 1: return
                mult *= p ** (f >> 1)
                continue
            odd_pf[p] = f
        if num > 1:
            if num & 3 == 3: return
            odd_pf[num] = 1
        #print(target, odd_pf, mult)
        for p, f in odd_pf.items():
            pair = sumOfTwoSquaresEqualToPrime(p)
            gaussian_pf[pair] = f
    #print(target, gaussian_pf)
    def multiplyComplex(num1: Tuple[int, int], num2: Tuple[int, int]) -> Tuple[int, int]:
        #print(num1, num2)
        return (
            num1[0] * num2[0] - num1[1] * num2[1],
            num1[0] * num2[1] + num1[1] * num2[0],
        )
    
    def complexExponentiated(num: Tuple[int, int], exp: int) -> Tuple[int, int]:
        curr = num
        exp2 = exp
        res = num if exp2 & 1 else (1, 0)
        exp2 >>= 1
        while exp2:
            curr = multiplyComplex(curr, curr)
            if exp2 & 1:
                res = multiplyComplex(res, curr)
            exp2 >>= 1
        return res
    #print(gaussian_pf)
    p_lst = list(gaussian_pf.keys())
    #print(p_lst)
    #print(p_lst)
    f_lst = [gaussian_pf[p] for p in p_lst]
    #print(p_lst, f_lst)
    n_p = len(p_lst)
    seen = set()
    def recur(idx: int, curr: Tuple[int, int]) -> Generator[Tuple[int, int], None, None]:
        #print(idx, curr)
        if idx == n_p:
            ans = tuple(sorted(abs(x) for x in curr))
            if ans in seen:
                #print(f"repeat seen: {ans}")
                return
            seen.add(ans)
            yield ans
            return
        curr2 = curr
        p, f = p_lst[idx], f_lst[idx]
        p_conj = (p[0], -p[1])

        if not idx:
            # For the first prime factor, only consider one of each conjugate
            # pairs. This cannot be done for every prime factor due to how
            # the results change when replacing with its one or both of the
            # complex numbers in a product.
            # Review- can this be taken further than just one to avoid
            # calculating the same values repeatedly?
            curr2 = curr
            p, f = p_lst[idx], f_lst[idx]
            p_conj = (p[0], -p[1])
            #print(f"p = {p}, p_conj = {p_conj}, f = {f_lst[idx]}")
            for pos_f in range(f >> 1):
                neg_f = f - pos_f
                mult_neg = complexExponentiated(p_conj, neg_f)
                #print(f"pos_f = {pos_f}, curr2 = {curr2}, mult_neg = {mult_neg}")
                yield from recur(idx + 1, multiplyComplex(curr2, mult_neg))
                curr2 = multiplyComplex(curr2, p)
            pos_f = f >> 1
            neg_f = f - pos_f
            mult_neg = complexExponentiated(p_conj, neg_f)
            yield from recur(idx + 1, multiplyComplex(curr2, mult_neg))
            return
        #print(f"p = {p}, p_conj = {p_conj}, f = {f_lst[idx]}")
        for pos_f in range(f):
            neg_f = f - pos_f
            mult_neg = complexExponentiated(p_conj, neg_f)
            #print(f"pos_f = {pos_f}, curr2 = {curr2}, mult_neg = {mult_neg}")
            yield from recur(idx + 1, multiplyComplex(curr2, mult_neg))
            curr2 = multiplyComplex(curr2, p)
        pos_f = f
        yield from recur(idx + 1, curr2)
        
        return
        
    #print(gaussian_pf)
    #print(f"mult = {mult}")
    yield from recur(0, (mult, 0))
    #cnt = 0
    #for ans in recur(0, (mult, 0)):
    #    cnt += 1
    #print(f"for {target}, with the number of ways of representing as the sum of squares of two non-negative integers = {cnt}")
    return

def biclinicIntegralQuadrilateralCountBruteForce(
    squared_side_length_sum_max: int=10 ** 10,
    ps: Optional[PrimeSPFsieve]=None,
) -> int:
    # Using Apollonius's theorem
    res = 0
    mx_sq_sm = squared_side_length_sum_max >> 2
    hlf_diag_mx = isqrt(mx_sq_sm)
    #mx = isqrt(squared_side_length_sum_max >> 2)
    for hlf_diag in range(1, hlf_diag_mx + 1):
        if not hlf_diag % 100:
            print(f"half diagonal = {hlf_diag} (max {hlf_diag_mx})")
        hlf_diag_sq = hlf_diag * hlf_diag
        for off_diag_dist in range(1, min(hlf_diag, isqrt(mx_sq_sm - hlf_diag_sq)) + 1):
            sm_diff_sq_sm = (hlf_diag_sq + (off_diag_dist * off_diag_dist))
            #cnt_ub = sumOfTwoSquaresSolutionsCount(
            #    sq_sm,
            #    ps=ps,
            #)
            #if cnt_ub <= 1: continue
            cnt = 0
            #sols = []
            for (a, b) in sumOfTwoSquaresSolutionGenerator(
                sm_diff_sq_sm,
                ps=ps,
            ):
                if not a or b <= hlf_diag: continue
                #u = (b + a) >> 1
                #v = (b - a) >> 1
                #if a == b or a + off_diag_dist <= hlf_diag: continue
                #if a + off_diag_dist <= hlf_diag or a + b <= 2 * hlf_diag: continue
                cnt += 1
                #sols.append((a, b))
            if cnt > 1:
                #print(f"hlf_diag = {hlf_diag}, off_diag_dist = {off_diag_dist}, solutions:")
                #for pair in sols:
                #    print(pair)
                
                res += (cnt * (cnt - 1)) >> 1
                #print(f"current count = {res}")
            """
            n_sol = sumOfTwoDistinctSquaresUnorderedPositiveSolutionsCount(
                sq_sm,
                ps=ps,
            )
            print(hlf_diag, off_diag_dist, sq_sm, n_sol, (n_sol * (n_sol - 1)) >> 1)
            res += (n_sol * (n_sol - 1)) >> 1
            """
    return res

# Problem 312
def sierpinskiHamiltonianCyclesCountBruteForce(
    n: int,
    res_md: Optional[int]=None,
) -> int:
    if n < 0: return 0
    elif n <= 2: return 1
    curr = [3, 2]
    # curr[0] is the number of Hamiltonian paths from one outer vertex
    #         to another skipping the third
    # curr[1] is the number of Hamiltonian paths from one outer vertex
    #         to another including the third
    for n in range(3, n):
        if not n % 10 ** 4:
            print(f"n = {n}")
        curr = [2 * curr[0] ** 2 * curr[1], 2 * curr[0] * curr[1] ** 2]
        if res_md is not None:
            curr = [curr[0] % res_md, curr[1] % res_md]
        #print(curr)
    res = curr[1] ** 3
    return res if res_md is None else res % res_md

def sierpinskiHamiltonianCyclesCount(
    n: int,
    res_md: Optional[int]=None,
) -> int:
    if n < 0: return 0
    elif n <= 2: return 1
    return (8 * pow(12, ((3 ** (n - 2) - 3) >> 1), res_md)) % res_md

def nestedSierpinskiHamiltonianCyclesModuloPPowCount(n0: int=10 ** 4, n_nest: int=3, p: int=13, p_pow: int=8) -> int:

    # Reveiw- prove this works for any prime power
    # Review- generalise to any modulo
    if p_pow == 1:
        md = p
        curr = 2
        seen = {}
        for i in itertools.count(0):
            if curr in seen:
                break
            seen[curr] = i
            curr = pow(3 * curr, 3, md)
        cycle_len = (i - seen[curr])
    else:
        # Find cycle length for p ** 2
        md = p ** 2
        curr = 2
        seen = {}
        for i in itertools.count(0):
            if curr in seen:
                break
            seen[curr] = i
            curr = pow(3 * curr, 3, md)
        print(i, seen[curr], i - seen[curr])
        cycle_len = (i - seen[curr]) * p ** (p_pow - 2)
    
    res_md = p ** p_pow
    res = n0 % cycle_len
    print(f"cycle_len = {cycle_len}, res_md = {res_md}, initial value = {res}")
    for _ in range(n_nest - 1):
        print(res)
        res = sierpinskiHamiltonianCyclesCount(res, res_md=cycle_len)
    print(res)
    return sierpinskiHamiltonianCyclesCount(res, res_md=res_md)

def nestedSierpinskiHamiltonianCyclesModuloCount(n0: int=10 ** 4, n_nest: int=3, res_md: int=13 ** 8) -> int:
    """
    Solution to Project Euler #312
    """
    vals = [1, 1, 1, 2]
    seen = {}
    for i in itertools.count(4):
        if not i % 10 ** 6: print(f"i = {i}")
        num = pow(3 * vals[-1], 3, res_md)
        if num in seen.keys(): break
        vals.append(num)
        seen[num] = i
    cycle_len = len(vals) - seen[num]
    cycle_start = seen[num]
    print(f"cycle_start = {cycle_start}, cycle_len = {cycle_len}")

    res = n0 % cycle_len
    print(f"cycle_len = {cycle_len}, res_md = {res_md}, initial value = {res}")
    for _ in range(n_nest - 1):
        print(res)
        res = sierpinskiHamiltonianCyclesCount(res, res_md=cycle_len)
    print(res)
    return sierpinskiHamiltonianCyclesCount(res, res_md=res_md)
    """
    res = n0
    for _ in range(n_nest):
        if res < len(vals):
            res = vals[res]
            continue
        idx = res % cycle_len
        print(idx)
        idx += cycle_len * max(0, (cycle_start - idx - 1) // cycle_len + 1)
        print(cycle_start, cycle_len, idx)
        res = vals[idx]
    return res
    """

# Problem 313
def calculateSlidingPuzzleMinimumMoves(n_rows: int, n_cols: int) -> int:
    """
    Calculates the minimum number of moves of a slide
    puzzle required to move the square in the top left of
    the grid to the bottom right of the grid for a slide
    puzzle with dimensions n_rows x n_cols, where the
    empty space is initially in the bottom right corner.

    A slide puzzle is a rectangular grid with integer
    side lengths where each 1 x 1 section of the grid except
    one (the so-called empty space) is covered by a 1 x 1
    square, oriented so its edges are parallel to that of the
    grid. A move consists of moving one of the squares
    that is orthogonally adjacent to the empty space into the
    empty space, effectively moving the empty space into the
    previous position of the chosen square.

    Args:
        Required positional:
        n_rows (int): Strictly positive integer giving the
                number of rows in the slide puzzle grid.
        n_cols (int): Strictly positive integer giving the
                number of columns in the slide puzzle grid.
    
    Returns:
    Integer (int) giving the minimum number of moves required
    to get the square at the top left of the grid to the
    bottom right when the empty space is initially at the
    bottom right of the grid. If no such sequence of moves
    exists, -1 is returned.

    Outline of rationale:
    TODO
    """
    if n_rows > n_cols:
        n_rows, n_cols = n_cols, n_rows
    if n_rows <= 1: return 1
    #return (n_rows + n_cols - 2) + 3 * (n_rows + min(n_cols - 1, n_rows) - 2) + 5 * max(0, n_cols - n_rows - 1)
    if n_rows == n_cols:
        return 8 * n_rows - 11 # cannot be a square as no square is 5 modulo 8
    return 6 * n_cols + 2 * n_rows - 13

def calculateSlidingPuzzleMinimumMovesAPrimeSquareCount(p_max: int=10 ** 6 - 1) -> int:
    """
    Solution to Project Euler #313

    Calculates the number of different dimensions of
    slide puzzle grid exist for which minimum number of
    moves of a slide puzzle required to move the square
    in the top left of the grid to the bottom right of
    the grid for a slide puzzle with dimensions
    n_rows x n_cols, where the empty space is initially
    in the bottom right corner, is equal to the square of
    a prime number no greater than p_max.

    A slide puzzle is a rectangular grid with integer
    side lengths where each 1 x 1 section of the grid except
    one (the so-called empty space) is covered by a 1 x 1
    square, oriented so its edges are parallel to that of the
    grid. A move consists of moving one of the squares
    that is orthogonally adjacent to the empty space into the
    empty space, effectively moving the empty space into the
    previous position of the chosen square.

    Args:
        Optional named:
        p_max (int): Strictly positive integer giving the
                inclusive upper bound on the size of the prime
                for which slide puzzles whose minimum number
                of moves to solve is equal to that prime squared
                are to be counted.
    
    Returns:
    Integer (int) giving the number of different dimensions of
    slide puzzle grid exist for which minimum number of
    moves of a slide puzzle required to move the square
    in the top left of the grid to the bottom right of
    the grid for a slide puzzle with dimensions
    n_rows x n_cols, where the empty space is initially
    in the bottom right corner, is equal to the square of
    a prime number no greater than p_max.

    Outline of rationale:
    TODO
    """
    # Note the grid cannot be square or have either dimension less than 2
    ps = SimplePrimeSieve(p_max)
    x, y = extendedEuclideanAlgorithm(3, 1)[1]
    res = 0
    p_prev = 0
    for p in ps.p_lst[1:]:
        if p // 1000 != p_prev // 1000:
            print(f"p = {p}")
            p_prev = p
        
        p_sq = p * p
        rhs = (p_sq + 13) >> 1
        (dx, x0, dy, y0) = (1, x * rhs, -3, y * rhs)#solveLinearNonHomogeneousDiophantineEquation(3, 1, rhs)
        # x0 + m * dx > y0 + m * dy
        # m * (dx - dy) > y0 - x0  (note (dx - dy) is positive)
        # m > (y0 - x0) // (dx - dy)
        mult_mn = max((-x0  + 1) // dx, (y0 - x0) // (dx - dy)) + 1
        
        mult_mx = (y0 - 2) // (-dy)
        #for mult in range(mult_mn, mult_mx + 1):
        #    print(x0 + mult * dx, y0 + mult * dy, p_sq)
        res += max(0, mult_mx - mult_mn + 1)
    return res << 1

# Problem 314
def calculateMaximumAreaToPerimeterRatioInQuantisedSquare(
    square_side_length: int=500,
) -> float:
    """
    Solution to Project Euler #314

    Finds the maximum possible area to perimeter ratio for
    polygons whose vertices all have integer coordinate values
    (x, y) within the square region 0 <= x <= square_side_length,
    0 <= y <= square_side_length.

    Args:
        Optional named:
        square_side_length (int): Strictly positive integer
                giving the side of the square that encloses
                the permitted polygon vertex positions.
            Default: 500
    
    Returns:
    Float giving the largest possible area to perimeter ratio of
    the polygons whose vertices all have integer coordinate values
    (x, y) within the square region 0 <= x <= square_side_length,
    0 <= y <= square_side_length.

    Outline of rationale:
    TODO
    """
    # Assumes solution has the same symmetries as the square
    # (D4 group), that the shape is convex and that it includes
    # the midpoints of the four edges of the square (these
    # should be straightforward to prove, each by contradiction).

    # Try to make faster

    eps = 1e-5

    curr = [[(0, 0)]]
    for diag_len in range(1, (square_side_length >> 1) + 1):
        if not diag_len % 10:
            print(f"diag_len = {diag_len} of {(square_side_length >> 1)}")
            print(f"len(curr[-1]) = {len(curr[-1])}")
        add_perim = diag_len * math.sqrt(2)
        perim_area_lst = SortedList([(add_perim, 0)])


        for v2 in range(1, diag_len + 1):
            for v1 in range(min(v2, diag_len - v2 + 1)):
                if gcd(v1, v2) > 1:
                    continue
                v = [-v1, v2]
                diag_len0 = diag_len - v1 - v2
                thickness = v2 - v1
                add_area = (diag_len + diag_len0) * thickness
                #diag_len_diff = (diag_len - prev_diag_len) >> 1
                add_perim = 2 * math.sqrt(v1 * v1 + v2 * v2)
                #print(f"diag_len = {diag_len}, diag_len0 = {diag_len0}, v = {v}, diff = {diff}, add_area = {add_area}, add_perim = {add_perim}")
                # (1 + t) * v[1] = diag_len + (1 + t) * v[0]
                # (1 + t) * (v[1] - v[0]) = diag_len
                # t = diag_len / (v[1] - v[0]) - 1
                v_len = math.sqrt(v1 * v1 + v2 * v2)
                t = diag_len / (v2 + v1) - 1
                perim0_mx = v_len * t * 2 + eps
                for perim0, neg_area0 in curr[diag_len0]:
                    #print(f"perim0 = {perim0}, perim0_mx = {perim0_mx}, diag_len0_actual = {diag_len0 * math.sqrt(2)}")
                    if perim0 > perim0_mx:
                        #print("hi")
                        break
                    perim, neg_area = perim0 + add_perim, neg_area0 - add_area
                    j = perim_area_lst.bisect_right((perim, neg_area))
                    if j > 0 and neg_area >= perim_area_lst[j - 1][1]:
                        continue
                    while j < len(perim_area_lst) and neg_area <= perim_area_lst[j][1]:
                        perim_area_lst.pop(j)
                    perim_area_lst.add((perim, neg_area))
                #v = [x + 1 for x in v]
        """
        for prev_diag_len in range(diag_len):
            len_diff = diag_len - prev_diag_len
            len_diff_hlf = len_diff >> 1
            v0 = [-len_diff_hlf + (not len_diff & 1), len_diff_hlf + 1]
            v = v0
            for diff in range(len_diff & 1, len_diff + 1, 2):
                if gcd(*[abs(x) for x in v]) > 1:
                    v = [x + 1 for x in v]
                    continue
                add_area = (diag_len + prev_diag_len) * diff
                #diag_len_diff = (diag_len - prev_diag_len) >> 1
                add_perim = 2 * math.sqrt(sum(x * x for x in v))
                #print(f"diag_len = {diag_len}, prev_diag_len = {prev_diag_len}, v = {v}, diff = {diff}, add_area = {add_area}, add_perim = {add_perim}")
                for perim0, neg_area0 in curr[prev_diag_len]:
                    perim, neg_area = perim0 + add_perim, neg_area0 - add_area
                    j = perim_area_lst.bisect_right((perim, neg_area))
                    if j > 0 and neg_area >= perim_area_lst[j - 1][1]:
                        continue
                    while j < len(perim_area_lst) and neg_area <= perim_area_lst[j][1]:
                        perim_area_lst.pop(j)
                    perim_area_lst.add((perim, neg_area))
                v = [x + 1 for x in v]
        """
        curr.append(list(perim_area_lst))
        #print(diag_len, curr[-1])
    
    if square_side_length & 1:
        area0 = (square_side_length ** 2) - (((square_side_length >> 1) ** 2) << 1)
        perim0 = 4
    else:
        area0 = (square_side_length ** 2) >> 1
        perim0 = 0
    res = 0
    print(f"area0 = {area0}, perim0 = {perim0}")
    #print(curr[-1])
    for add_perim, add_neg_area in curr[-1]:
        perim, area = perim0 + (add_perim * 4), area0 - (add_neg_area * 2)
        ratio = area / perim
        if ratio > res:
            res = ratio
            print(f"add_perim = {add_perim}, perim0 = {perim0}, perim = {perim}, add_area = {-add_neg_area}, area0 = {area0}, area = {area}, ratio = {ratio}")
    return res


# Problem 315
def digitalRootDisplayPrimeTransitionsDifferenceCount(
    p_min: int=10 ** 7,
    p_max: int=2 * 10 ** 7,
) -> int:
    # Review- consider using bitmasks for digs_incl
    base = 10
    digs_incl = {
        None: set(),
        0: {0, 1, 2, 4, 5, 6},
        1: {2, 5},
        2: {0, 2, 3, 4, 6},
        3: {0, 2, 3, 5, 6},
        4: {1, 2, 3, 5},
        5: {0, 1, 3, 5, 6},
        6: {0, 1, 3, 4, 5, 6},
        7: {0, 1, 2, 5},
        8: {0, 1, 2, 3, 4, 5, 6},
        9: {0, 1, 2, 3, 5, 6},
    }

    digs_incl_bm = {}
    for d, st in digs_incl.items():
        digs_incl_bm[d] = 0
        for num in st:
            digs_incl_bm[d] |= 1 << num

    def digitTransitionDifference(d1: Optional[int], d2: Optional[int]) -> int:
        return ((digs_incl_bm[d1] & digs_incl_bm[d2]).bit_count()) << 1

    def integerTransitionDifference(num1: int, num2: int) -> int:
        res = 0
        while num1 and num2:
            num1, d1 = divmod(num1, base)
            num2, d2 = divmod(num2, base)
            res += digitTransitionDifference(d1, d2)
        return res

    def digitRoot(num: int) -> int:
        num2 = num
        res = 0
        while num2:
            num2, d = divmod(num2, base)
            res += d
        return res

    memo = {}
    def calculateDigitRootTransitionDifference(num: int) -> int:
        num2 = digitRoot(num)
        if num2 == num: return 0
        args = num
        if args in memo.keys(): return memo[args]
        res = integerTransitionDifference(num, num2) + calculateDigitRootTransitionDifference(num2)
        memo[args] = res
        return res

    ps = SimplePrimeSieve()
    def primeCheck(num: int) -> bool:
        return ps.millerRabinPrimalityTestWithKnownBounds(num, max_n_additional_trials_if_above_max=10)[0]

    res = 0
    for p in range(p_min + (not p_min & 1), p_max + 1, 2):
        if not primeCheck(p): continue
        num = p
        num2 = digitRoot(num)
        if num2 == num: continue
        res += integerTransitionDifference(num, num2) + calculateDigitRootTransitionDifference(num2)
    return res

# Problem 316
def gaussianEliminationFraction(mat: List[List[CustomFraction]], vec: List[CustomFraction]) -> List[CustomFraction]:
    # Assumes mat is a square matrix and vec is the same dimension as mat
    n = len(mat)
    for i1 in range(n):
        i1_ = i1
        if mat[i1][i1] == 0:
            for i1_ in range(i1 + 1, n):
                if mat[i1_][i1] != 0:
                    mat[i1], mat[i1_] = mat[i1_], mat[i1]
                    vec[i1], vec[i1_] = vec[i1_], vec[i1]
                    #print(f"swapped rows {i1} and {i1_}:")
                    #for row in mat:
                    #    print(row)
                    break
            else:
                print("matrix not invertible")
                print(f"matrix when found non-invertible (row {i1}):")
                for row in mat:
                    print(row)
                print(f"vector:")
                print(vec)
                return [] # the matrix is not invertible
        for i1_ in range(i1_ + 1, n):
            if mat[i1_][i1] == 0: continue
            mult = mat[i1_][i1] / mat[i1][i1]
            mat[i1_][i1] = CustomFraction(0, 1)
            for i2 in range(i1 + 1, n):
                mat[i1_][i2] -= mult * mat[i1][i2]
            vec[i1_] -= mult * vec[i1]
        #print(f"eliminated column {i1}:")
        #for row in mat:
        #    print(row)
    res = [CustomFraction(0, 1) for _ in range(n)]
    for i1 in reversed(range(n)):
        ans = vec[i1]
        for i2 in reversed(range(i1 + 1, n)):
            ans -= mat[i1][i2] * res[i2]
        res[i1] = ans / mat[i1][i1]
    return res

def calculateFirstOccurrenceOfIntegerInDigitSequenceExpectedValue(
    num: int,
    base: int=10,
) -> CustomFraction:

    num2 = num
    dig_lst = []
    while num2:
        num2, d = divmod(num2, base)
        dig_lst.append(d)
    n = len(dig_lst)
    dig_lst = dig_lst[::-1]
    kmp = KnuthMorrisPratt(dig_lst)
    lps = kmp.lps
    #print(lps)
    transf = [{} for _ in range(n + 1)]
    transf[0] = {0: CustomFraction(base - 1, base), 1: CustomFraction(1, base)}
    N_inv = [[CustomFraction(0, 1)] * n for _ in range(n)]
    N_inv[0][0] = -CustomFraction(base - 1, base)
    if 1 < n: N_inv[0][1] = -CustomFraction(1, base)
    for j in range(1, n):
        #print(f"j = {j}")
        #transf[j] = {j + 1: CustomFraction(1, base)}
        #if j + 1 < n:
        #    N_inv[j][j + 1] = -CustomFraction(1, base)
        #excl_nums = {dig_lst[j]}
        #print(dig_lst, excl_nums)
        excl_nums = set()
        j2 = j
        while True:
            if dig_lst[j2] not in excl_nums:
                excl_nums.add(dig_lst[j2])
                transf[j][j2 + 1] = CustomFraction(1, base)
                if j2 + 1 < n:
                    N_inv[j][j2 + 1] = -CustomFraction(1, base)
            if not j2: break
            j2 = lps[j2 - 1]
            #if not j2: break
            
        #print(excl_nums)
        if len(excl_nums) < base:
            transf[j][0] = CustomFraction(base - len(excl_nums), base)
            N_inv[j][0] = -CustomFraction(base - len(excl_nums), base)
    for i in range(n):
        N_inv[i][i] += CustomFraction(1, 1)
    #for row in N_inv:
    #    print(row)
    
    vec = [CustomFraction(1, 1) for _ in range(n)]
    sol = gaussianEliminationFraction(N_inv, vec)
    #print(sol)
    #if sol[0].denominator != 1:
    #    raise ValueError("the calculated solution is not an integer as was expected")
    return sol[0] - (n - 1)

def firstOccurrenceOfIntegerInDigitSequenceExpectedValueSum(
    numer: int=10 ** 16,
    denom_min: int=2,
    denom_max: int=10 ** 6 - 1,
    base: int=10,
) -> CustomFraction:
    """
    Solution to Project Euler #316
    """
    res = 0
    for n in range(denom_min, denom_max + 1):
        res += calculateFirstOccurrenceOfIntegerInDigitSequenceExpectedValue(
            numer // n,
            base=base,
        )
    return res

# Problem 317
def fircrackerVolume(h0: float=100., v0: float=20., g: float=9.81) -> int:
    """
    Solution to Project Euler #317

    For a firecracker that explodes at h0 metres above level ground
    into fragments with initial speeds of v0 metres per second in all
    directions under a uniform downward gravitational field with strength
    g metres per second squared, calculates the volume above ground that
    it is possible for at least one of the fragments of the firecracker
    to reach at some point during its trajectory in cubic metres.

    Args:
        Optional named:
        h0 (float): Strictly positive real number giving the height above
                the ground in metres at which the firecracker explodes
                in metres.
            Default: 100.
        v0 (float): Strictly positive real number giving the initial
                speed of the firecracker fragments in metres per second.
            Default: 20.
        g (float): Strictly positive real number giving the strength
                of the downwards uniform gravitational field in metres
                per second squared.
            Default: 9.81
    
    Returns:
    Float giving the volume above ground that it is possible for at least
    one of the fragments of the firecracker to reach at some point during
    its trajectory in cubic metres for the given values of h0, v0 and g.

    Outline of rationale:
    TODO
    """
    a = v0 ** 2 / (2 * g)
    return 2 * math.pi * a * (a + h0) ** 2

# Problem 318
def findPQValues(
    sum_max: int,
    base: int=10,
) -> list[tuple[int, int]]:
    n_base_minus_one_min = 3
    target = base ** n_base_minus_one_min - 1
    res = []
    for q in range(2, sum_max + 1):
        for p in range(1, min(sum_max - q + 1, q)):
            num = (math.sqrt(p) + math.sqrt(q)) ** 10
            frac = num - math.floor(num)
            frac_mul = math.floor(frac * base ** n_base_minus_one_min)
            #abs_diff = abs(math.sqrt(p) - math.sqrt(q))
            # (sqrt(p) - sqrt(q)) ** 2 < 1
            # p + q - 2 * sqrt(pq) < 1
            # 2 * sqrt(pq) > p + q - 1
            # 4 * pq > (p + q - 1) ** 2
            lhs = 4 * p * q
            rhs = (p + q - 1) ** 2
            print((p, q), (math.sqrt(p) + math.sqrt(q)) ** 2, lhs > rhs, frac_mul, target)
            if frac_mul == target:
                print("solution")
                res.append((p, q))
    return sorted(res)

def calculateMinimalNsForFractionalPartToStartWithMBaseMinusOneSum(
    sum_max: int=2011,
    n_base_minus_one: int=2011,
    base: int=10,
) -> int:
    """
    Solution to Project Euler #318
    """
    # Review- Look into Pisot-Vijayaraghavan numbers to prove that the
    # fractional part of increasing powers of a non-integer positive
    # real number tends to zero iff that number is less than 1.

    m = n_base_minus_one
    res = 0
    for q in range(2, sum_max):
        for p in reversed(range(1, min(q, sum_max - q + 1))):
            sqrt_diff = math.sqrt(q) - math.sqrt(p)
            if sqrt_diff >= 1:
                break
            cnt = math.ceil((-m) / (2 * math.log(sqrt_diff, base)))
            print((p, q), cnt)
            res += cnt
    return res

# Problem 319
def boundedSequenceGeneratorBruteForce(
    n_term_min: int,
    n_term_max: int,
) -> Generator[tuple[int], None, None]:
    
    if 1 >= n_term_min:
        yield (2,)
    #print("n_term = 1")
    curr = [(2,)]
    #print(f"number of bounded sequences length 1 = {len(curr)}")
    for n_term in range(2, n_term_max + 1):
        #print(f"n_term = {n_term}")
        prev = curr
        curr = []
        for seq in prev:
            rng = [-float("inf"), float("inf")]
            j = n_term
            for i, num in enumerate(seq, start=1):
                mn = integerNthRoot(num ** j, i)
                mx = integerNthRoot((num + 1) ** j - 1, i)
                #print(i, num, (mn, mx))
                rng[0] = max(rng[0], mn)
                rng[1] = min(rng[1], mx)
                if rng[0] > rng[1]: break
            else:
                if n_term >= n_term_min:
                    for num in range(rng[0], rng[1] + 1):
                        curr.append((*seq, num))
                        yield curr[-1]
                else:
                    for num in range(rng[0], rng[1] + 1):
                        curr.append((*seq, num))
    return



# Problem 320
def factorialPrimeFactorPower(p: int, n: int) -> int:
    """
    Calculates the exponent of the prime p in the prime
    factorisation of n! (n factorial).

    Args:
        Required positional:
        p (int): Prime number whose exponent in the prime
                factorisation of n! is to be found.
        n (int): Non-negative integer whose factorial prime
                factorisation the exponent of the prime p
                is to be found.
    
    Returns:
    Integer (int) giving the exponent of the prime p in the
    prime factorisation of n!.
    """
    res = 0
    n2 = n
    while n2:
        n2 //= p
        res += n2
    return res

def smallestFactorialDivisibleByPrimePower(
    p: int,
    exp: int,
) -> int:
    """
    Calculates the smallest non-negative integer n such that
    n! (n factorial) is divisible by p ** exp (p to the power
    of exp) where p is a prime.

    Args:
        Required positional:
        p (int): Prime number for which the returned value is
                the smallest non-negative integer whose factorial
                is divisible by this prime to the power of exp.
        exp (int): Non-negative integer for which the returned
                value is the smallest non-negative integer whose
                factorial is divisible by p to the power of this
                number.
    
    Returns:
    Integer (int) giving the smallest non-negative integer whose
    factorial is divisible by p ** exp (p to the power of exp).
    """
    if not exp: return 0
    lo, hi = 0, 1
    while factorialPrimeFactorPower(p, hi) < exp:
        lo = hi + 1
        hi = lo << 1
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if factorialPrimeFactorPower(p, mid) < exp:
            lo = mid + 1
        else: hi = mid
    return lo

def smallestFactorialDivisibleByPrimePowerWithLowerBound(
    p: int,
    exp: int,
    lo: int=0,
) -> int:
    """
    Calculates the smallest integer non-negative integer
    n >= lo such that n! (n factorial) is divisible by p ** exp
    (p to the power of exp) where p is a prime.

    Args:
        Required positional:
        p (int): Prime number for which the returned value is
                the smallest non-negative integer no less than
                lo whose factorial is divisible by this prime
                to the power of exp.
        exp (int): Non-negative integer for which the returned
                value is the smallest non-negative integer no
                less than lo whose factorial is divisible by p
                to the power of this number.

        Optional named:
        lo (int, optional): Integer giving the inclusive lower
                bound on the returned value.
            Default: 0
    
    Returns:
    Integer (int) giving the smallest non-negative integer no less
    than lo whose factorial is divisible by p ** exp (p to the
    power of exp).
    """
    lo = max(lo, 0)
    if factorialPrimeFactorPower(p, lo) >= exp: return lo
    lo += 1
    hi = lo << 1
    while factorialPrimeFactorPower(p, hi) < exp:
        #print(hi, factorialPrimeFactorPower(p, hi), exp)
        lo = hi + 1
        hi = lo << 1
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if factorialPrimeFactorPower(p, mid) < exp:
            lo = mid + 1
        else: hi = mid
    return lo

def smallestFactorialDivisibleByFactorialPower(
    m: int,
    exp: int,
    ps: Optional[SimplePrimeSieve]=None,
    lo: int=0,
) -> int:
    """
    Calculates the smallest non-negative integer n >= lo for
    which the n! (n factorial) is divisible by (m!) ** exp
    (m factorial to the power of exp).

    Args:
        Required positional:
        m (int): Non-negative integer for which the returned value
                is the smallest non-negative integer no less than
                lo whose factorial is divisible by the factorial
                of this integer to the power of exp.
        exp (int): Non-negative integer for which the returned value
                is the smallest non-negative integer no less than
                lo whose factorial is divisible by m! to the power
                of this number.
        
        Optional named:
        ps (Optional[SimplePrimeSieve], optional): If specified,
                a basic prime sieve (using the sieve of Eratosthenes)
                used to identify the primes up to (and including) m.
                If not specified (or given as None), such a sieve
                is created inside the function.
                The option to use a previously defined prime sieve is
                made available to avoid redundant creation of multiple
                prime sieve objects.
            Default: None
        lo (int, optional): Integer giving the inclusive lower bound
                on the returned value.
            Default: 0

    Returns:
    Integer (int) giving the smallest non-negative integer no less than
    lo whose factorial is divisible by m factorial to the power of exp.
    """
    if ps is None:
        ps = SimplePrimeSieve(m)
    else:
        ps.extendSieve(m)
    lo = max(lo, 0)
    p_i_mx = bisect.bisect_right(ps.p_lst, m)
    res = max(lo, 0)
    for p_i in range(p_i_mx):
        p = ps.p_lst[p_i]
        exp2 = factorialPrimeFactorPower(p, m) * exp
        res = smallestFactorialDivisibleByPrimePowerWithLowerBound(
            p,
            exp2,
            lo=res,
        )
    return res

def smallestFactorialDivisibleByFactorialPowerSum(
    m_min: int=10,
    m_max: int=10 ** 6,
    exp: int=1234567890,
    ps: Optional[PrimeSPFsieve]=None,
    res_md: Optional[int]=10 ** 18,
) -> int:
    """
    Solution to Project Euler #320

    Calculates the sum over the smallest non-negative integer n for
    which the n! (n factorial) is divisible by (m!) ** exp
    (m factorial to the power of exp) for each non-negative integer
    m between m_min and m_max inclusive. If res_md is given as a
    strictly positive integer, then this sum is returned modulo res_md.

    Args:
        Optional named:
        m_min (int): Non-negative integer giving the inclusive lower
                bound on the non-negative integer values for which
                the smallest non-negative integer whose factorial is
                divisible by the factorial the value to the power of
                exp is to be included in the returned sum.
            Default: 10
        m_max (int): Non-negative integer giving the inclusive upper
                bound on the non-negative integer values for which
                the smallest non-negative integer whose factorial is
                divisible by the factorial the value to the power of
                exp is to be included in the returned sum.
            Default: 10 ** 6
        exp (int): Non-negative integer for which the value included
                in the sum for a given integer m should be the smallest
                integer whose factorial is divisible by m! to the power
                of this number.
            Default: 1234567890
        ps (Optional[PrimeSPFsieve], optional): If specified, a smallest
                prime factor (SPF) prime sieve (based on the sieve of
                Eratosthenes) used to find primes and identify the prime
                factorisations of integers up to (and including) m_max.
                If not specified (or given as None), such a sieve
                is created inside the function.
                The option to use a previously defined prime sieve is
                made available to avoid redundant creation of multiple
                prime sieve objects.
            Default: None
        res_md (Optional[int], optional): If given as a strictly positive
                integer, the modulus to which the final sum should be
                taken when returned, otherwise the sum itself is returned.
            Default: 10 ** 18

    Returns:
    Integer (int) giving sum of the the smallest non-negative integer
    whose factorial is divisible by m factorial to the power of exp for
    each non-negative integer m between m_min and m_max inclusive.
    
    Outline of rationale:
    TODO
    """
    if ps is None: ps = PrimeSPFsieve(m_max)

    m_min = max(m_min, 0)
    if m_min > m_max: return 0

    addMod = (lambda x, y: x + y) if res_md is None else (lambda x, y: (x + y) % res_md)

    res = 0
    ans = smallestFactorialDivisibleByFactorialPower(
        m=m_min,
        exp=exp,
        ps=ps,
        lo=0,
    )
    res = addMod(res, ans)
    for m in range(m_min + 1, m_max + 1):
        if not m % 10 ** 4:
            print(f"m = {m} (max {m_max})")
        pf = ps.primeFactors(m)
        #print(m, pf)
        for p in ps.primeFactors(m):
            exp2 = factorialPrimeFactorPower(p, m) * exp
            ans = smallestFactorialDivisibleByPrimePowerWithLowerBound(
                p,
                exp2,
                lo=ans,
            )
        #print(m, ans)
        res = addMod(res, ans)
    return res

# Problem 321
def calculateFirstNCounterSwappingGamesEqualToTriangularNumber(
    n: int=40,
) -> List[int]:
    if n <= 0: return []
    res = []
    for (a, b) in generalisedPellSolutionGenerator(8, -7, excl_trivial=True):
        if b <= 1 or not a & 1: continue
        res.append(b - 1)
        if len(res) == n: break
    return res

def calculateFirstNCounterSwappingGamesEqualToTriangularNumberSum(
    n: int,
) -> int:
    """
    Solution to Project Euler #321
    """
    sol_lst = calculateFirstNCounterSwappingGamesEqualToTriangularNumber(n)
    print(sol_lst)
    return sum(sol_lst)

# Problem 322
def binomialCoefficientsNotDivisibleByPrimeForGivenKGenerator(
    n_max: int,
    k: int,
    p: int,
) -> Generator[int, None, None]:
    """
    Calculates the binomial coefficients (n choose k) that
    are not divisible by the prime p for n between k and n_max
    inclusive.
    """
    if n_max < k: return []

    n_max_base_p_digs = []
    n2 = n_max
    while n2:
        n2, d = divmod(n2, p)
        n_max_base_p_digs.append(d)
    n_digs = len(n_max_base_p_digs)
    #print(n_max_base_p_digs)

    k_base_p_digs = []
    k2 = k
    for _ in range(n_digs):
        k2, d = divmod(k2, p)
        k_base_p_digs.append(d)
    #print(k_base_p_digs)

    
    def recur(idx: int=0, curr: int=0, tight_hi: bool=True) -> Generator[int, None, None]:
        #print(idx, curr, tight_hi)
        if idx == n_digs:
            yield curr
            return
        curr *= p
        rng = [k_base_p_digs[~idx], n_max_base_p_digs[~idx] if tight_hi else p - 1]
        if rng[0] > rng[1]: return
        for d in range(*rng):
            yield from recur(idx=idx + 1, curr=curr + d, tight_hi=False)
        yield from recur(idx=idx + 1, curr=curr + rng[1], tight_hi=tight_hi)
        return
    yield from recur(idx=0, curr=0, tight_hi=True)
    #print(res)
    return

def binomialCoefficientsNotDivisibleByPrimePowerForGivenKGenerator(
    n_max: int,
    k: int,
    p: int,
    exp: int,
) -> Generator[int, None, None]:
    """
    Calculates the binomial coefficients (n choose k) that
    are divisible by the prime power p ** exp for n between
    k and n_max inclusive.
    """
    if exp < 0: raise ValueError("exp must be possible")
    elif not exp: return n_max - k + 1
    elif exp == 1:
        yield from binomialCoefficientsNotDivisibleByPrimeForGivenKGenerator(
            n_max,
            k,
            p,
        )
        return
    
    # Using Kummer's theorem

    if n_max < k: return []

    m_base_p_digs = []
    m2 = n_max - k
    while m2:
        m2, d = divmod(m2, p)
        m_base_p_digs.append(d)
    n_digs = len(m_base_p_digs)


    k_base_p_digs = []
    k2 = k
    for _ in range(n_digs):
        k2, d = divmod(k2, p)
        k_base_p_digs.append(d)

    
    def recur(idx: int=0, curr: int=0, carry: bool=False, carry_tot: int=0, tight_hi: bool=True) -> Generator[int, None, None]:
        if carry_tot >= exp: return
        if idx == n_digs:
            yield curr + k
            return
        curr *= p
        rng = [0, m_base_p_digs[~idx] if tight_hi else p - 1]
        if rng[0] > rng[1]: return
        for d in range(*rng):
            c = ((k_base_p_digs[~idx] + d + carry) >= p)
            yield from recur(idx=idx + 1, curr=curr + d, carry=c, carry_tot=carry_tot + c, tight_hi=False)
        c = ((k_base_p_digs[~idx] + rng[1] + carry) >= p)
        yield from recur(idx=idx + 1, curr=curr + rng[1], carry=c, carry_tot=carry_tot + c, tight_hi=tight_hi)
        return
    yield from recur(idx=0, curr=0, carry=False, carry_tot=0, tight_hi=True)
    return

def binomialCoefficientDivisibleByPrimePower(n: int, k: int, p: int, exp: int) -> bool:


    #print(n, k, p, exp)
    # Using Kummer's theorem
    if k < 0 or n < k or not exp: return True
    m = n - k
    m_base_p_digs = []
    m2 = m
    while m2:
        m2, d = divmod(m2, p)
        m_base_p_digs.append(d)
    n_digs = len(m_base_p_digs)


    k_base_p_digs = []
    k2 = k
    for _ in range(n_digs):
        k2, d = divmod(k2, p)
        k_base_p_digs.append(d)
    #print(m_base_p_digs, k_base_p_digs)
    
    carry = False
    curr = 0
    for idx in range(n_digs):
        carry = (m_base_p_digs[idx] + k_base_p_digs[idx] + carry >= p)
        if not carry: continue
        curr += 1
        if curr >= exp: return True
    return False

def binomialCoefficientsDivisibleByIntegerForGivenK(
    n_max: int,
    k: int,
    div: int,
    ps: Optional[PrimeSPFsieve]=None,
) -> Generator[int, None, None]:
    """
    Calculates the binomial coefficients (n choose k) that
    are divisible by the strictly positive integer div for n between
    k and n_max inclusive.
    """
    pf = calculatePrimeFactorisation(div, ps)
    n_p = len(pf)
    if not n_p:
        yield from range(k, n_max + 1)
        return
    elif len(pf) == 1:
        p, exp = next(iter(pf))
        yield from binomialCoefficientsNotDivisibleByPrimePowerForGivenKGenerator(
            n_max,
            k,
            p,
            exp,
        )
        return
    
    return

def binomialCoefficientsNotDivisibleByPrimeForGivenKCount(
    n_max: int,
    k: int,
    p: int,
) -> int:
    """
    Calculates the number of binomial coefficients (n choose k) that
    are not divisible by the prime p for n between k and n_max inclusive.
    """
    if n_max < k: return []

    n_max_base_p_digs = []
    n2 = n_max
    while n2:
        n2, d = divmod(n2, p)
        n_max_base_p_digs.append(d)
    n_digs = len(n_max_base_p_digs)
    #print(n_max_base_p_digs)

    k_base_p_digs = []
    k2 = k
    for _ in range(n_digs):
        k2, d = divmod(k2, p)
        k_base_p_digs.append(d)
    #print(k_base_p_digs)
    
    memo = {}
    def recur(idx: int=0, tight_hi: bool=True) -> int:
        if idx == n_digs:
            return 1
        args = (idx, tight_hi)
        if args in memo.keys(): return memo[args]
        rng = [k_base_p_digs[~idx], n_max_base_p_digs[~idx] if tight_hi else p - 1]
        if rng[0] > rng[1]: return 0
        res = 0
        res += (rng[1] - rng[0]) * recur(idx=idx + 1, tight_hi=False)
        res += recur(idx=idx + 1, tight_hi=tight_hi)
        memo[args] = res
        return res
    res = recur(idx=0, tight_hi=True)
    #print(res)
    #print(memo)
    return res

def binomialCoefficientsNotDivisibleByPrimePowerForGivenKCount(
    n_max: int,
    k: int,
    p: int,
    exp: int,
) -> int:
    """
    Calculates the number of binomial coefficients (n choose k) that
    are not divisible by the prime power p ** exp for n between k and
    n_max inclusive.
    """
    if exp < 0: raise ValueError("exp cannot be negative")
    elif not exp: return n_max - k + 1
    elif exp == 1:
        return binomialCoefficientsNotDivisibleByPrimeForGivenKCount(
            n_max,
            k,
            p,
        )
    return 0

def binomialCoefficientsNotDivisibleByIntegerForGivenKCount(
    n_max: int=10 ** 18 - 1,
    k: int=10 ** 12 - 10,
    div: int=10,
    ps: Optional[PrimeSPFsieve]=None,
) -> int:
    """
    Calculates the number of binomial coefficients (n choose k) that
    are not divisible by the strictly positive integer div for n between
    k and n_max inclusive.
    """
    # Review- try to make faster
    pf = calculatePrimeFactorisation(div, ps)
    n_p = len(pf)
    if not n_p:
        return list(range(k, n_max + 1))
    elif len(pf) == 1:
        p, exp = next(iter(pf))
        return binomialCoefficientsNotDivisibleByPrimePowerForGivenKCount(
            n_max,
            k,
            p,
            exp,
        )
    
    # Using inclusion-exclusion
    p_lst = sorted(pf.keys())
    p_map = {(1 << idx): p for idx, p in enumerate(p_lst)}
    #rng_tot = n_max - k + 1
    bm_cnts = [
        binomialCoefficientsNotDivisibleByPrimePowerForGivenKCount(
            n_max,
            k,
            p,
            pf[p],
        )
        for p in p_lst
    ]
    print(bm_cnts)
    res = sum(bm_cnts)
    print(f"n_set = 1: {res}")
    p_order = sorted(list(range(n_p)), key=(lambda x: bm_cnts[x]))
    curr_lsts = {}
    for i1 in range(n_p - 1):
        idx1 = p_order[i1]
        p1 = p_lst[idx1]
        bm1 = 1 << idx1
        bm2_dict = {}
        args_dict = {}
        for i2 in range(i1 + 1, n_p):
            idx2 = p_order[i2]
            p2 = p_lst[idx2]
            bm2_dict[i2] = bm1 ^ (1 << idx2)
            args_dict[i2] = [p2, pf[p2]]
            curr_lsts[bm2_dict[i2]] = []
        print(i1, args_dict)
        for num in binomialCoefficientsNotDivisibleByPrimePowerForGivenKGenerator(
            n_max,
            k,
            p1,
            pf[p1],
        ):
            for i2, args in args_dict.items():
                #print(f"num = {num}, i2 = {i2}")
                if not binomialCoefficientDivisibleByPrimePower(num, k, *args):
                    curr_lsts[bm2_dict[i2]].append(num)
        """
        print(idx1, len(lst1))
        for i2 in range(i1 + 1, n_p):
            idx2 = p_order[i2]
            p2 = p_lst[idx2]
            bm = bm1 ^ (1 << idx2)
            curr_lsts[bm] = []
            for num in lst1:
                if not binomialCoefficientDivisibleByPrimePower(num, k, p2, pf[p2]):
                    curr_lsts[bm].append(num)
            ans += len(curr_lsts[bm])
        """
    ans = sum(len(x) for x in curr_lsts.values())
    for bm in list(curr_lsts.keys()):
        if not curr_lsts[bm]: curr_lsts.pop(bm)
    print(f"n_set = 2: {ans}")
    #print(curr_lsts)
    res -= ans
            
    """
    print(bm_cnts)
    curr_lsts = {
        (1 << idx): binomialCoefficientsNotDivisibleByPrimePowerForGivenK(
            n_max,
            k,
            p,
            pf[p],
        )
        for idx, p in enumerate(p_lst)
    }
    res = sum(len(x) for x in curr_lsts.values())
    #print(1, curr_lsts)
    print(1, {p: len(lst) for p, lst in curr_lsts.items()})
    """
    for n_set in range(3, len(p_lst) + 1):
        ans = 0
        prev_lsts = curr_lsts
        curr_lsts = {}
        # Using Gosper's hack to iterate over all bitmasks with
        # n_p bits and exactly n_set set bits
        bm = (1 << n_set) - 1
        limit = (1 << n_p)
        while bm < limit:

            bm2 = bm
            mn_len = [float("inf"), -1]
            while bm2:
                lo_bit = bm2 & (-bm2)
                bm3 = bm ^ lo_bit
                mn_len = min(mn_len, [len(prev_lsts[bm3]), lo_bit])
                bm2 ^= lo_bit
            p = p_map[mn_len[1]]
            curr_lsts[bm] = []
            for num in prev_lsts[bm ^ mn_len[1]]:
                if not binomialCoefficientDivisibleByPrimePower(num, k, p, pf[p]):
                    curr_lsts[bm].append(num)
            ans += len(curr_lsts[bm])
            #for idx in range()
            #for :
            #    pass
        
            # Gosper's hack to transition to next bitmask
            lo_bit = bm & (-bm)
            lo_sm = bm + lo_bit
            shifted = bm ^ (bm + lo_sm)
            bm = lo_sm | ((shifted >> 2) // lo_bit)
        res += ans if (n_set & 1) else -ans
        print(n_set, {p: len(lst) for p, lst in curr_lsts.items()})
        print(f"n_set = {n_set}: {ans}")

    return res

def binomialCoefficientsDivisibleByIntegerForGivenKCount(
    n_max: int=10 ** 18 - 1,
    k: int=10 ** 12 - 10,
    div: int=10,
    ps: Optional[PrimeSPFsieve]=None,
) -> int:
    """
    Solution to Project Euler #322

    Calculates the number of binomial coefficients (n choose k) that
    are divisible by the strictly positive integer div for n between
    k and n_max inclusive.
    """
    res = binomialCoefficientsNotDivisibleByIntegerForGivenKCount(
        n_max=n_max,
        k=k,
        div=div,
        ps=ps,
    )
    print(res)
    return (n_max - k + 1) - res

# Problem 323
def randomSequenceBitwiseOrIsAllOnesExpectedValueFraction(
    n_bit: int,
) -> CustomFraction:
    """
    Calculates the expectation number of terms in the shortest
    prefix of an infinite sequence of random integers between 0
    and 2 ** (n_bit) - 1 inclusive, where each number in this
    range has the same probability of being selected, for
    which the bitwise or of every integer in that prefix is
    equal to 2 ** n_bit - 1 (i.e. the binary representation of
    the bitwise or of every element in the prefix without
    leading zeros consists of n_bit ones and no zeros).

    Args:
        Required positional
        n_bit (int): Strictly positive integer giving the
                range of the random numbers chosen, that
                being between 0 and 2 ** n_bit - 1, and
                the target value of the prefix bitwise or,
                that also being 2 ** n_bit - 1.
    
    Returns:
    CustomFraction object representing the rational expectation
    value for the length of the shortest prefix of the random
    sequences generated as described for which the bitwise or
    of all elements in that prefix is equal to (2 ** n_bit - 1).

    Outline of rationale:
    TODO
    """
    res = CustomFraction(0, 1)

    # P(number of steps to get all 1s <= n) = (1 - (1 / 2) ** n) ** n_bit
    # P(number of steps to get all 1s >= n) = 1 - P(number of steps to get all 1s <= (n - 1))
    #  = 1 - (1 - (1 / 2) ** (n - 1)) ** n_bit
    # so expected value is:
    # E(number of steps to get all ones) = sum(n = 1 to inf) (1 - (1 - (1 / 2) ** (n - 1)) ** n_bit)
    #  = sum(n = 0 to inf) (1 - (1 - (1 / 2) ** n)) ** n_bit)
    #  = sum(k = 1 to n_bit) (-1) ** (k - 1) * (n_bit choose k) / (1 - (1 / 2) ** k)

    res = CustomFraction(0, 1)
    for k in range(1, n_bit + 1):
        div = 1 - CustomFraction(1, 1 << k)
        term = math.comb(n_bit, k) / div
        res += term if k & 1 else -term
    return res

def randomSequenceBitwiseOrIsAllOnesExpectedValueFloat(
    n_bit: int=32,
) -> float:
    """
    Solution to Project Euler #323

    Calculates the expectation number of terms in the shortest
    prefix of an infinite sequence of random integers between 0
    and 2 ** (n_bit) - 1 inclusive, where each number in this
    range has the same probability of being selected, for
    which the bitwise or of every integer in that prefix is
    equal to 2 ** n_bit - 1 (i.e. the binary representation of
    the bitwise or of every element in the prefix without
    leading zeros consists of n_bit ones and no zeros).

    Args:
        Required positional
        n_bit (int): Strictly positive integer giving the
                range of the random numbers chosen, that
                being between 0 and 2 ** n_bit - 1, and
                the target value of the prefix bitwise or,
                that also being 2 ** n_bit - 1.
    
    Returns:
    Float representing the expectation value for the length of
    the shortest prefix of the random sequences generated as
    described for which the bitwise or of all elements in that
    prefix is equal to (2 ** n_bit - 1).

    Outline of rationale:
    See Outline of rationale section in the documentation for
    randomSequenceBitwiseOrIsAllOnesExpectedValueFraction().
    """
    res = randomSequenceBitwiseOrIsAllOnesExpectedValueFraction(n_bit)
    return res.numerator / res.denominator

# Problem 324
def blockTowerConfigurationsCount(
    dims: tuple[int, int, int], 
    res_md: Optional[int]=10 ** 8 + 7,
) -> int:

    # Review- when the layer rectangle has an odd area
    # (i.e. both sides are of odd length), consider
    # traversing two layers at a time.

    if dims[0] & 1 and dims[1] & 1 and dims[2] & 1:
        return 0
    dims = sorted(dims)

    addMod = (lambda x, y: x + y) if res_md is None else (lambda x, y: (x + y) % res_md)
    mulMod = (lambda x, y: x * y) if res_md is None else (lambda x, y: (x * y) % res_md)
    
    rect_dims = dims[:2]
    h = dims[2]
    
    rect_area = rect_dims[0] * rect_dims[1]

    def standardiseLayerBitmask(bm_raw: int) -> int:
        res = bm_raw
        bm_lst = [bm_raw]
        bm_set = {bm_raw}
        # reflection across horizontal reflection line
        for i in range(len(bm_lst)):
            bm2 = bm_lst[i]
            ans = 0
            bm3 = (1 << rect_dims[1]) - 1
            for _ in range(rect_dims[0]):
                ans = (ans << rect_dims[1]) ^ (bm2 & bm3)
                bm2 >>= rect_dims[1]
            if ans not in bm_set:
                bm_lst.append(ans)
                res = min(res, bm_lst[-1])

        # rotation through 180 degrees about centre (of both the original
        # and the reflection, the latter giving the reflection across the
        # vertical reflection line)
        for i in range(len(bm_lst)):
            bm2 = bm_lst[i]
            ans = 0
            for _ in range(rect_area):
                ans = (ans << 1) ^ (bm2 & 1)
                bm2 >>= 1
            if ans not in bm_set:
                bm_lst.append(ans)
                res = min(res, bm_lst[-1])
        #print(bm_lst)
        # If the layer rectangle is not a square, does not have
        # symmetry through 90 degree rotations or diagonal reflections
        if rect_dims[0] != rect_dims[1]:
            return res
        
        # Reflect across leading diagonal reflection line (which
        # is the diagonal that passes through the squre represented
        # by the least significant digit of the bitmask integer)
        for i in range(len(bm_lst)):
            bms = [0] * rect_dims[0]
            bm2 = bm_lst[i]
            for _ in range(rect_dims[0]):
                for j in range(rect_dims[1]):
                    bms[j] = (bms[j] << 1) ^ (bm2 & 1)
                    bm2 >>= 1
            ans = 0
            for sub_bm in bms:
                ans = (ans << rect_dims[1]) ^ sub_bm
            res = min(res, ans)
        return res
    
    memo = {}
    def getTransferFunction(bm0: int) -> dict[int, int]:
        bm0 = standardiseLayerBitmask(bm0)
        if bm0 in memo.keys(): return memo[bm0]
        res = {}

        def recur(idx: int, bm: int, cover: set[int]) -> None:
            if idx == rect_area:
                bm2 = standardiseLayerBitmask(bm)
                res[bm2] = addMod(res.get(bm2, 0), 1)
                return
            
            if idx in cover:
                return recur(idx + 1, bm, cover)
            bm2 = 1 << idx
            recur(idx + 1, bm | bm2, cover)
            i1, i2 = divmod(idx, rect_dims[1])
            if i2 < rect_dims[1] - 1:
                idx2 = idx + 1
                if idx2 not in cover:
                    cover.add(idx2)
                    recur(idx + 1, bm, cover)
                    cover.remove(idx2)
            if i1 < rect_dims[0] - 1:
                idx2 = idx + rect_dims[1]
                if idx2 not in cover:
                    cover.add(idx2)
                    recur(idx + 1, bm, cover)
                    cover.remove(idx2)
            return
        cover = set()
        bm2 = bm
        for i in range(rect_area):
            if bm2 & 1: cover.add(i)
            bm2 >>= 1
        recur(0, 0, cover)
        memo[bm0] = res
        return res

    transfers = {}
    wait_bms = {0}
    while wait_bms:
        bm = next(iter(wait_bms))
        wait_bms.remove(bm)
        transf_dict = getTransferFunction(bm)
        transfers[bm] = transf_dict
        for bm2 in transf_dict.keys():
            if bm2 in transfers.keys() or bm2 in wait_bms:
                continue
            wait_bms.add(bm2)

    #print(transfers)
    #print(sum(transfers[0].values()))
    #print(len(transfers))
    #print([format(bm, "b") for bm in sorted(transfers.keys())])
    
    def applyTransfer(
        transf: dict[int, dict[int, int]],
        state: dict[int, int],
    ) -> dict[int, int]:
        #print(transf, state)
        res = {}
        for bm1, f1 in state.items():
            if not f1: continue
            for bm2, f2 in transf.get(bm1, {}).items():
                if not f2: continue
                res[bm2] = addMod(res.get(bm2, 0), mulMod(f1, f2))
        return res

    def composeTransfers(
        transf1: dict[int, dict[int, int]],
        transf2: dict[int, dict[int, int]],
    ) -> dict[int, dict[int, int]]:
        res = {}
        for bm1, bm1_dict in transf1.items():
            res[bm1] = {}
            for bm2, f2 in bm1_dict.items():
                if not f2: continue
                for bm3, f3 in transf2.get(bm2, {}).items():
                    if not f3: continue
                    res[bm1][bm3] = addMod(res[bm1].get(bm3, 0), mulMod(f2, f3))
            if not res[bm1]: res.pop(bm1)
        return res

    transf_pow2 = transfers
    res = {0: 1}
    h2 = h
    while True:
        if h2 & 1:
            res = applyTransfer(transf_pow2, res)
        h2 >>= 1
        if not h2: break
        transf_pow2 = composeTransfers(transf_pow2, transf_pow2)
        #print(transf_pow2)
    #print(res)
    return res.get(0)

# Problem 327
def calculateNumberOfCardsNeededToProgress(
    card_carry_capacity: int,
    n_rooms: int,
) -> int:
    
    if card_carry_capacity > n_rooms:
        return n_rooms + 1
    if card_carry_capacity <= 2:
        return -1

    memo = {}
    def recur(idx: int, n_cards: int) -> int:
        if not idx:
            return 0
        args = (idx, n_cards)
        if args in memo.keys():
            return memo[args]
        res = 0
        n_backtracks = max(0, (n_cards - 2) // (card_carry_capacity - 2))
        res = recur(idx - 1, n_cards + (2 * n_backtracks + 1)) + (2 * n_backtracks + 1)

        memo[args] = res
        return res
    res = recur(n_rooms + 1, 0)
    #print(memo)
    return res

def calculateNumberOfCardsNeededToProgressSum(
    card_carry_capacity_min: int=3,
    card_carry_capacity_max: int=40,
    n_rooms: int=30,
) -> int:
    """
    Solution to Project Euler #327
    """
    if card_carry_capacity_min <= 2 and card_carry_capacity_min <= n_rooms:
        return -1
    res = 0
    for ccc in range(card_carry_capacity_min, card_carry_capacity_max + 1):
        res += calculateNumberOfCardsNeededToProgress(ccc, n_rooms)
    return res

# Problem 329
def croakSequenceProbability(
    n_squares: int=500,
    seq: str="PPPPNNPPPNPPNPN",
    p_croak_P_if_prime: CustomFraction=CustomFraction(2, 3),
    p_croak_P_if_nonprime: CustomFraction=CustomFraction(1, 3),
) -> CustomFraction:
    """
    Solution to Project Euler #329
    """
    pf = SimplePrimeSieve(n_squares)
    p_arr = [False] * n_squares
    for p in pf.p_lst:
        if p > len(p_arr): break
        p_arr[p - 1] = True
    p_prime = p_croak_P_if_prime
    p_nonprime = p_croak_P_if_nonprime
    if seq[0] != "P":
        p_prime = 1 - p_prime
        p_nonprime = 1 - p_nonprime
    curr = [CustomFraction(1, n_squares) * (p_prime if x else p_nonprime) for x in p_arr]
    #print(curr)
    for l in seq[1:]:
        prev = curr
        
        p_prime = p_croak_P_if_prime
        p_nonprime = p_croak_P_if_nonprime
        if l != "P":
            p_prime = 1 - p_prime
            p_nonprime = 1 - p_nonprime
        curr = [CustomFraction(0, 1)] * n_squares
        curr[1] += prev[0]
        for i in range(1, n_squares - 1):
            curr[i + 1] += CustomFraction(1, 2) * prev[i]
            curr[i - 1] += CustomFraction(1, 2) * prev[i]
        curr[n_squares - 2] += prev[n_squares - 1]
        
        for i in range(n_squares):
            curr[i] *= (p_prime if p_arr[i] else p_nonprime)
        #print(curr)

    return sum(curr)

# Problem 330
def eulerSequenceTermGenerator(
    n_max: Optional[int]=None,
) -> Generator[tuple[int, int], None, None]:

    # Note term index 1 follows OEIS A337000
    # Furthermore it appears that:
    #   2 * pair[1] + pair[0] - math.factorial(i) = 0
    head_sm = 0
    it = itertools.count(0) if n_max is None else range(n_max + 1)

    res = []
    for i in it:
        #print(i)
        head_sm = head_sm * i + 1
        i_fact = math.factorial(i)
        curr = [-head_sm, i_fact]
        for j, pair in enumerate(res):
            mult = math.comb(i, i - j)#i_fact // math.factorial(i - j)
            curr[0] += pair[0] * mult
            curr[1] += pair[1] * mult
        curr = tuple(curr)
        res.append(curr)
        yield curr
    return
"""
def eulerSequenceTermGenerator2(
    n_max: Optional[int]=None,
) -> Generator[tuple[int, int], None, None]:
    head_sm = 0
    it = itertools.count(0) if n_max is None else range(n_max + 1)

    res = []
    for i in it:
        head_sm = CustomFraction(0, 1)
        for j in range(i + 1):
            head_sm += CustomFraction(1, math.factorial(j))
        ans = 

        #print(i)
        head_sm = head_sm * i + 1
        i_fact = math.factorial(i)
        curr = [-head_sm, i_fact]
        for j, pair in enumerate(res):
            mult = i_fact // math.factorial(i - j)
            curr[0] += pair[0] * mult
            curr[1] += pair[1] * mult
        curr = tuple(curr)
        res.append(curr)
        yield curr
    return"""

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

    if 303 in eval_nums:
        since = time.time()
        res = smallestMultiplierGivingMultipleWithDigitValueUpperBoundSum(
            num_max=10 ** 4,
            max_dig_val=2,
            base=10,
        )
        print(f"Solution to Project Euler #303 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 304 in eval_nums:
        since = time.time()
        res = primonacciSum(
            n_max=10 ** 5,
            prime_start=10 ** 14,
            res_md=1234567891011,
        )
        print(f"Solution to Project Euler #304 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 305 in eval_nums:
        since = time.time()
        res = reflexivePositionsInNumberConcatenatorPowersSum(
            a=3,
            k_min=1,
            k_max=13,
            base=10,
        )
        print(f"Solution to Project Euler #305 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 306 in eval_nums:
        since = time.time()
        res = paperStripGamePlayer1WinsWithPerfectPlayCount(n_max=10 ** 6)
        print(f"Solution to Project Euler #301 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 307 in eval_nums:
        since = time.time()
        res = proportionOfBallAllocationsIntoBinsWithOneBinWithAtLeastGivenNumberFloat(
            n_bins=10 ** 6,
            n_balls=2 * 10 ** 4,
            n_balls_in_bin_maxmin=3,
        )
        print(f"Solution to Project Euler #307 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 308 in eval_nums:
        since = time.time()
        res = conwayFractanPow2PrimeGeneratorStepCount(n_p=10 ** 4 + 1)
        print(f"Solution to Project Euler #309 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 309 in eval_nums:
        since = time.time()
        res = integerCrossingLaddersCount(len_max=10 ** 6 - 1)
        print(f"Solution to Project Euler #309 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 310 in eval_nums:
        since = time.time()
        res = nimSquarePositionsLostByNextPlayerCount(n_max=10 ** 5)
        print(f"Solution to Project Euler #310 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 311 in eval_nums:
        since = time.time()
        res = biclinicIntegralQuadrilateralCountBruteForce(
            squared_side_length_sum_max=10 ** 10,
            ps=None,
        )
        print(f"Solution to Project Euler #311 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 312 in eval_nums:
        since = time.time()
        res = nestedSierpinskiHamiltonianCyclesModuloCount(n0=10 ** 4, n_nest=3, res_md=13 ** 8)
        # res = nestedSierpinskiHamiltonianCyclesModuloPPowCount(n0=10 ** 4, n_nest=3, p=13, p_pow=8)
        print(f"Solution to Project Euler #312 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 313 in eval_nums:
        since = time.time()
        res = calculateSlidingPuzzleMinimumMovesAPrimeSquareCount(p_max=10 ** 6 - 1)
        print(f"Solution to Project Euler #313 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 314 in eval_nums:
        since = time.time()
        res = calculateMaximumAreaToPerimeterRatioInQuantisedSquare(
            square_side_length=500,
        )
        print(f"Solution to Project Euler #314 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 315 in eval_nums:
        since = time.time()
        res = digitalRootDisplayPrimeTransitionsDifferenceCount(
            p_min=10 ** 7,
            p_max=2 * 10 ** 7,
        )
        print(f"Solution to Project Euler #315 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 316 in eval_nums:
        since = time.time()
        res = firstOccurrenceOfIntegerInDigitSequenceExpectedValueSum(
            numer=10 ** 16,
            denom_min=2,
            denom_max=10 ** 6 - 1,
            base=10,
        )
        print(f"Solution to Project Euler #316 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 317 in eval_nums:
        since = time.time()
        res = fircrackerVolume(h0=100, v0=20, g=9.81)
        print(f"Solution to Project Euler #317 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 318 in eval_nums:
        since = time.time()
        res = calculateMinimalNsForFractionalPartToStartWithMBaseMinusOneSum(
            sum_max=2011,
            n_base_minus_one=2011,
            base=10,
        )
        print(f"Solution to Project Euler #318 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 320 in eval_nums:
        since = time.time()
        res = smallestFactorialDivisibleByFactorialPowerSum(
            m_min=10,
            m_max=10 ** 6,
            exp=1234567890,
            ps=None,
            res_md=10 ** 18,
        )
        print(f"Solution to Project Euler #320 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 321 in eval_nums:
        since = time.time()
        res = calculateFirstNCounterSwappingGamesEqualToTriangularNumberSum(
            n=40,
        )
        print(f"Solution to Project Euler #321 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 322 in eval_nums:
        since = time.time()
        res = binomialCoefficientsDivisibleByIntegerForGivenKCount(
            n_max=10 ** 18 - 1,
            k=10 ** 12 - 10,
            div=10,
            ps=None,
        )
        print(f"Solution to Project Euler #322 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 323 in eval_nums:
        since = time.time()
        res = randomSequenceBitwiseOrIsAllOnesExpectedValueFloat(
            n_bit=32,
        )
        print(f"Solution to Project Euler #323 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 324 in eval_nums:
        since = time.time()
        res = blockTowerConfigurationsCount(
            dims=(3, 3, 10 ** (10 ** 4)),
            res_md=10 ** 8 + 7,
        )
        print(f"Solution to Project Euler #324 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 327 in eval_nums:
        since = time.time()
        res = calculateNumberOfCardsNeededToProgressSum(
            card_carry_capacity_min=3,
            card_carry_capacity_max=40,
            n_rooms=30,
        )
        print(f"Solution to Project Euler #327 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 329 in eval_nums:
        since = time.time()
        res = croakSequenceProbability(
            n_squares=500,
            seq="PPPPNNPPPNPPNPN",
            p_croak_P_if_prime=CustomFraction(2, 3),
            p_croak_P_if_nonprime=CustomFraction(1, 3),
        )
        print(f"Solution to Project Euler #329 = {res}, calculated in {time.time() - since:.4f} seconds")

    print(f"Total time taken = {time.time() - since0:.4f} seconds")

    

if __name__ == "__main__":
    eval_nums = {3290}
    evaluateProjectEulerSolutions251to300(eval_nums)



#pow2_init = 7
#conwayFractanPow2TransitionLengthBruteForce(pow2_init)
#print(conwayFractanPow2TransitionLength(pow2_init))
"""
n_inds = 7780
num = 7780
for i, idx in zip(range(n_inds), startingPositionsInNumberConcatenator(
    num,
    base=10,
)):
    print(f"index {i + 1} = {idx}")
"""
"""
sum_max = 10

for pair in findPQValues(
    sum_max=sum_max,
    base=10,
):
    print(pair)
"""


"""
base = 10
for target, n in [(1, 1), (2, 2), (3, 3), (4, 4), (2, 56456)]:
    for _, res1 in zip(range(n), startingPositionsInNumberConcatenator(
        target,
        base=base,
    )):
        pass

    res2 = nthStartingPositionOfTargetInNumberConcatenator(n, target, base=base)
    print(f"{n}:th occurrence of {target}: {res1}, {res2}")
"""
"""
base = 10
target_rng = [1, 10 ** 5]
n_mx = 3
for target in range(target_rng[0], target_rng[1] + 1):
    if not target % 100:
        print(f"testing target = {target} (max {target_rng[1]})")
    #print(f"\ntarget = {target}")
    for n in range(1, n_mx + 1):
        for _, res1 in zip(range(n), startingPositionsInNumberConcatenator(
            target,
            base=base,
        )):
            pass

        res2 = nthStartingPositionOfTargetInNumberConcatenator(n, target, base=base)
        if res1 == res2:
            #print(f"solution for target = {target}, n = {n}: {res1}")
            continue
        print(f"mismatch for target = {target}, n = {n}: brute force solution = {res1}, calculated solution = {res2}")
"""

"""
n_term = 4

cnt = 0
#curr_seq_len = 0
for seq in boundedSequenceGeneratorBruteForce(
    n_term_min=n_term,
    n_term_max=n_term,
):
    cnt += 1
    print(seq)
print(f"count = {cnt}")
"""

for i, pair in enumerate(eulerSequenceTermGenerator(
    n_max=50,
)):
    print(i, pair, -pair[0] / pair[1], 2 * pair[1] + pair[0] - math.factorial(i))