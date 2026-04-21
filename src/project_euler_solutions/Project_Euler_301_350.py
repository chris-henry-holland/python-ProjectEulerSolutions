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
    """
    curr = [1, 1]
    for _ in range(pow2):
        curr = [curr[1], sum(curr)]
    return curr[1]

# Problem 302
def strongAchillesNumberCountBruteForce(n_max: int, ps: Optional[PrimeSPFsieve]=None) -> int:

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
    
    digs = []
    num2 = num
    while num2:
        num2, d = divmod(num2, base)
        digs.append(d)
    n_dig = len(digs)

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
    res = 0
    for num in range(1, num_max + 1):
        mul = smallestMultiplierGivingMultipleWithDigitValueUpperBound(
            num,
            max_dig_val,
            base=base,
        )
        print(num, mul, num * mul)
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
    """
    res = proportionOfBallAllocationsIntoBinsWithOneBinWithAtLeastGivenNumberFraction(
        n_bins,
        n_balls,
        n_balls_in_bin_maxmin,
    )
    #print(res)
    return res.numerator / res.denominator

# Problem 309
def integerCrossingLaddersCount(len_max: int=10 ** 6 - 1) -> int:
    """
    Solution to Project Euler #309
    """
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


# Problem 313
def calculateSlidingPuzzleMinimumMoves(n_rows: int, n_cols: int) -> int:
    if n_rows > n_cols:
        n_rows, n_cols = n_cols, n_rows
    #return (n_rows + n_cols - 2) + 3 * (n_rows + min(n_cols - 1, n_rows) - 2) + 5 * max(0, n_cols - n_rows - 1)
    if n_rows == n_cols:
        return 8 * n_rows - 11 # cannot be a square as no square is 5 modulo 8
    return 6 * n_cols + 2 * n_rows - 13

def calculateSlidingPuzzleMinimumMovesAPrimeSquareCount(p_max: int=10 ** 6 - 1) -> int:
    """
    Solution to Project Euler #313
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

    if 309 in eval_nums:
        since = time.time()
        res = integerCrossingLaddersCount(len_max=10 ** 6 - 1)
        print(f"Solution to Project Euler #309 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 310 in eval_nums:
        since = time.time()
        res = nimSquarePositionsLostByNextPlayerCount(n_max=10 ** 5)
        print(f"Solution to Project Euler #310 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 313 in eval_nums:
        since = time.time()
        res = calculateSlidingPuzzleMinimumMovesAPrimeSquareCount(p_max=10 ** 6 - 1)
        print(f"Solution to Project Euler #313 = {res}, calculated in {time.time() - since:.4f} seconds")

    print(f"Total time taken = {time.time() - since0:.4f} seconds")

if __name__ == "__main__":
    eval_nums = {313}
    evaluateProjectEulerSolutions251to300(eval_nums)
