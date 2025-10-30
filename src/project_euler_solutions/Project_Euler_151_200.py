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
import heapq
import itertools
import math
import os
import random
import sys
import time

from collections import deque
from sortedcontainers import SortedDict, SortedList

from data_structures.prime_sieves import PrimeSPFsieve, SimplePrimeSieve
from data_structures.addition_chains import AdditionChainCalculator
from data_structures.fractions import CustomFraction

from algorithms.string_searching_algorithms import rollingHashWithValue, KnuthMorrisPratt
from algorithms.Pythagorean_triple_generators import pythagoreanTripleGeneratorByHypotenuse
from algorithms.continued_fractions_and_Pell_equations import sqrtBestRationalApproximation
from algorithms.geometry_algorithms import (
    twoDimensionalLineSegmentPairCrossing,
    BentleyOttmannAlgorithmIntegerEndpoints,
)
from algorithms.number_theory_algorithms import (
    gcd,
    lcm,
    isqrt,
    integerNthRoot,
)

def addFractions(frac1: Tuple[int, int], frac2: Tuple[int, int]) -> Tuple[int, int]:
    """
    Finds the sum of two fractions in lowest terms (i.e. such that
    the numerator and denominator are coprime)

    Args:
        frac1 (2-tuple of ints): The first of the fractions to sum,
                in the form (numerator, denominator)
        frac2 (2-tuple of ints): The second of the fractions to sum,
                in the form (numerator, denominator)
    
    Returns:
    2-tuple of ints giving the sum of frac1 and frac2 in the form (numerator,
    denominator). If the result is negative then the numerator is negative
    and the denominator positive.
    """
    denom = lcm(abs(frac1[1]), abs(frac2[1]))
    numer = (frac1[0] * denom // frac1[1]) + (frac2[0] * denom // frac2[1])
    g = gcd(numer, denom)
    #print(frac1, frac2, (numer // g, denom // g))
    return (numer // g, denom // g)

def multiplyFractions(frac1: Tuple[int, int], frac2: Tuple[int, int]) -> Tuple[int, int]:
    """
    Finds the product of two fractions in lowest terms (i.e. such that
    the numerator and denominator are coprime)

    Args:
        frac1 (2-tuple of ints): The first of the fractions to multiply,
                in the form (numerator, denominator)
        frac2 (2-tuple of ints): The second of the fractions to multiply,
                in the form (numerator, denominator)
    
    Returns:
    2-tuple of ints giving the product of frac1 and frac2 in the form (numerator,
    denominator). If the result is negative then the numerator is negative
    and the denominator positive.
    """
    neg = (frac1[0] < 0) ^ (frac1[1] < 0) ^ (frac2[0] < 0) ^ (frac2[1] < 0)
    frac_prov = (abs(frac1[0] * frac2[0]), abs(frac1[1] * frac2[1]))
    #print(frac_prov)
    g = gcd(frac_prov[0], frac_prov[1])
    return (-(frac_prov[0] // g) if neg else (frac_prov[0] // g), frac_prov[1] // g)

def compareFractions(frac1: Tuple[int, int], frac2: Tuple[int, int]) -> int:
    """
    Compares two fractions, identifying which is larger or if they are the
    same size.

    Args:
        frac1 (2-tuple of ints): The first of the fractions to compare,
                in the form (numerator, denominator)
        frac2 (2-tuple of ints): The second of the fractions to compare,
                in the form (numerator, denominator)
    
    Returns:
    Integer (int) giving 1 if frac1 is larger, -1 if frac2 is larger and 0 if
    frac1 and frac2 are equal.
    """
    num = frac1[0] * frac2[1] - frac2[0] * frac1[1]
    if num > 0: return 1
    elif num < 0: return -1
    return 0

def floorHarmonicSeries(n: int) -> int:
    """
    Calculates the value of the sum:
        (sum i from 1 to n) floor(n / i)
    using the identity that:
        (sum i from 1 to n) floor(n / i) = ((sum i from 1 to k) floor(n / i)) - k ** 2
    where k = floor(sqrt(n))
    
    Args:
        Required positional:
        n (int): Strictly positive integer giving the value
                for which the value of the above formula is
                to be calculated.
    
    Returns:
    Integer (int) giving the value of:
        sum (i from 1 to n) floor(n / i)
    for the given value of n.
    """
    k = isqrt(n)
    return sum(n // i for i in range(1, k + 1)) - k ** 2

# Problem 151
def singleSheetCountExpectedValueFraction(
    n_halvings: int,
) -> CustomFraction:
    """
    Consider a sheet of paper in an envelope. This sheet should
    produce 2 ** n_halvings sheets of paper, each 1 / 2 ** n_halvings
    the size of the original sheet (by area) by repeatedly cutting a
    sheet derived from the original in half.

    This is achieved in multiple stages. Each stage consists of
    randomly selecting a sheet currently in the envelope (where
    all sheets in the envelope have equal probability of being
    selected), and termed as the current sheet. The following
    process is then followed until completion:
        1) If the current sheet is the desired size (i.e.
           1 / 2 ** n_halvings the size of the original sheet)
           then the process is complete. Otherwise, go to step
           2.
        2) Cut the current sheet in half. One half is placed
           back in the envelope, the other half is now the
           current sheet. Go to step 1.
    
    This process is repeated until the envelope is empty.
    Regardless of the order in which the sheets are selected,
    the process is guaranteed to be completed in exactly
    2 ** (n_halvings + 1) - 1 steps.

    This function calculates the expected number of stages in
    this process excluding the first and last stage that start
    with exactly one sheet in the envelope, giving the value
    as a fraction.
    
    Args:
        Required positional:
        n_halvings (int): Non-negative integer giving the
                relative size (in terms of area) of the desired
                sheet size compared to the original sheet size,
                expressed as the number of repeated halvings of
                the original sheet are required to achieve the
                desired size.
    
    Returns:
    2-tuple of integers (ints), giving the expected value for
    the number of stages in the described process excluding the
    first and last stage that start with exactly one sheet in
    the envelope, expressed as a fraction in lowest terms (i.e.
    numerator and denominator are coprime), where index 0 is
    a non-negative integer giving the numerator and index 1
    is a strictly positive integer giving the denominator.

    Outline of rationale:
    We define a state as (where the number of each size sheet
    of paper in the envelope.
    We then observe that given that in any given run through
    of the process, the total area of paper is strictly
    decreasing with every stage (as no paper is added and a
    sheet of the desired size is always removed at the end
    of the stage), a state cannot occur more than once in
    the same process.
    Consequently the solution is then simply the sum of the
    probabilities of encountering each state where there is
    only one sheet of paper that is not the original size or
    the desired size.
    These probabilities are calculated with top-down dynamic
    programming with memoisation, working backwards from each
    chosen state  to a state containing only a single sheet
    of the original size to calculate the proportion of the
    possible applications of the process encounter that state
    at some stage. Given that sheets are selected uniform
    randomly is equal to the probability that that state is
    encountered in a single run through of the process, the
    number required.
    TODO- revise for clarity
    """
    #mx_counts = [1 << i for i in range(n_halvings)]
    mx_tot = 1 << n_halvings

    memo = {}
    def recur(state: List[int], tot: int, sm: int) -> CustomFraction:
        if tot >= mx_tot:
            return CustomFraction(int(tot == mx_tot and state[0] == 1), 1)
        args = tuple(state)
        #print(args, sm, tot)
        if args in memo.keys(): return memo[args]
        res = CustomFraction(0, 1)
        for i in reversed(range(n_halvings + 1)):
            #if state[i] == mx_counts[i]:
            #    sub = 1
            #    state[i] -= sub
            #    if state[i] < 0: break
            #    sm -= sub * (1 << i)
            #    continue
            add = 1
            tot += add * (1 << (n_halvings - i))
            sm += add
            state[i] += add
            #print(i)
            res += recur(state, tot, sm) * CustomFraction(state[i], sm)
            sub = 2
            state[i] -= sub
            if state[i] < 0: break
            sm -= sub
            tot -= sub * (1 << (n_halvings - i))
        else: i = 0
        for i in range(i, len(state)):
            state[i] += 1
            sm -= tot
            tot += (1 << (n_halvings - i))
        memo[args] = res
        return res

    state = [0] * (n_halvings + 1)
    state[0] = 1
    res = CustomFraction(0, 1)
    for i in range(1, n_halvings):
        state[i] += 1
        state[i - 1] -= 1
        #print(state)
        ans = recur(state, 1 << (n_halvings - i), 1)
        #print(ans)
        res += ans
    #print(memo)
    return res

def singleSheetCountExpectedValueFloat(n_halvings: int=5) -> float:
    """
    Solution to Project Euler #151

    Consider a sheet of paper in an envelope. This sheet should
    produce 2 ** n_halvings sheets of paper, each 1 / 2 ** n_halvings
    the size of the original sheet (by area) by repeatedly cutting a
    sheet derived from the original in half.

    This is achieved in multiple stages. Each stage consists of
    randomly selecting a sheet currently in the envelope (where
    all sheets in the envelope have equal probability of being
    selected), and termed as the current sheet. The following
    process is then followed until completion:
        1) If the current sheet is the desired size (i.e.
           1 / 2 ** n_halvings the size of the original sheet)
           then the process is complete. Otherwise, go to step
           2.
        2) Cut the current sheet in half. One half is placed
           back in the envelope, the other half is now the
           current sheet. Go to step 1.
    
    This process is repeated until the envelope is empty.
    Regardless of the order in which the sheets are selected,
    the process is guaranteed to be completed in exactly
    2 ** (n_halvings + 1) - 1 steps.

    This function calculates the expected number of stages in
    this process excluding the first and last stage that start
    with exactly one sheet in the envelope, giving the value
    as a float.
    
    Args:
        Required positional:
        n_halvings (int): Non-negative integer giving the
                relative size (in terms of area) of the desired
                sheet size compared to the original sheet size,
                expressed as the number of repeated halvings of
                the original sheet are required to achieve the
                desired size.
    
    Returns:
    Float, giving the expected value for the number of stages in
    the described process excluding the first and last stage that
    start with exactly one sheet in the envelope.

    Outline of rationale:
    See doumentation for singleSheetCountExpectedValueFraction().
    """
    #since = time.time()
    frac = singleSheetCountExpectedValueFraction(n_halvings=n_halvings)
    res = frac.numerator / frac.denominator
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 152
def sumsOfSquareReciprocals(
    target: CustomFraction=CustomFraction(1, 2),
    denom_min: int=2,
    denom_max: int=80,
) -> Set[Tuple[int]]:
    """
    Finds the distinct ways the fraction target can be constructed
    by summing recpirocals of squares of integers, where the
    integers being squared are between denom_min and denom_max
    inclusive, and no two integers in the sum are the same. Two
    such sums are considered distinct if and only if there is a
    term that is present in one sum that is not present in the
    other sum (so two sums where the terms are simply a permutation
    of each other are not distinct).

    Args:
        Optional named:
        target (CustomFraction object): A fraction representing
                the rational number the sums must equal.
            Default: CustomFraction(1, 2)- representing a half
        denom_min (int): Strictly positive integer giving the
                smallest integer for whose squared reciprocal
                can appear in the sum.
            Default: 2
        denom_max (int): Strictly positive integer giving the
                largest integer for whose squared reciprocal
                can appear in the sum.
            Default: 80
    
    Returns:
    Set of tuples of ints, where each tuple contains a distinct
    list of integers in strictly increasing order (each distinct
    and between denom_min and denom_max) such that the sum of the
    reciprocals of their squares equals target, and collectively
    they represent all such distinct sums.

    Example:
        >>> sumsOfSquareReciprocals(target=CustomFraction(1, 2), denom_min=2, denom_max=45))
        {(2, 3, 4, 6, 7, 9, 10, 20, 28, 35, 36, 45), (2, 3, 4, 6, 7, 9, 12, 15, 28, 30, 35, 36, 45), (2, 3, 4, 5, 7, 12, 15, 20, 28, 35)}

        This indicates that there are exactly three distinct sums
        of distinct integer square reciprocals equal to a half,
        such that each integer in between 2 and 45 inclusive,
        including:
            1 / 2 ** 2 + 1 / 3 ** 2 + 1 / 4 ** 2 + 1 / 5 ** 2
                + 1 / 7 ** 2 + 1 / 9 ** 2 + 1 / 10 ** 2 + 20 / 2 ** 2
                + 1 / 28 ** 2 + 1 / 35 ** 2 + 1 / 36 ** 2 + 1 / 45 ** 2
        which it can be verified is indeed equal to a half.
    
    Outline of rationale:
    TODO
    """
    # TODO- need to check that finds all solutions for any
    # target
    # Add check for whether the smallest denominators must be
    # included (i.e. without them the sum cannot reach the
    # target) or cannot be included (i.e. the value of that
    # term alone exceeds the target) and adjust the target
    # and other parameters accordingly
    # Try to make more efficient and rule out more combinations
    # at an early stage

    #g = gcd(*target)
    #target = tuple(t // g for t in target)

    #since = time.time()

    def checkSolution(denoms: Tuple[int]) -> bool:
        res = CustomFraction(0, 1)
        for d in denoms:
            res += CustomFraction(1, d ** 2)
        #print(f"solution check: {denoms}, {res}")
        return res == target
    
    # Find the maximum prime factor of denominators possible
    curr = 0
    prod = 1
    i = 1
    while True:
        curr = curr * i ** 2 + prod
        #print(i, curr)
        if curr * i >= denom_max ** 2:
            break
        prod *= i ** 2
        i += 1
    #print(denom_max // i)

    ps = PrimeSPFsieve(n_max=denom_max // i)

    p_set = set(ps.p_lst)
    target_denom_pf = ps.primeFactorisation(target.denominator)
    p_lst = sorted(p_set.union(target_denom_pf.keys()))

    p_split_idx = 2
    #print(ps.p_lst)
    p_lst1 = p_lst[:p_split_idx]
    p_lst2 = p_lst[p_split_idx:]

    
    #print(p_lst)
    p_opts = {}
    if p_lst2:
        for bm in range(1, 1 << (denom_max // p_lst2[0])):
            bm2 = bm
            i = 1
            curr = 0
            prod = 1
            mx = 0
            #print(bm2)
            while bm2:
                if bm2 & 1:
                    curr = curr * i ** 2 + prod
                    prod *= i ** 2
                    mx = i
                i += 1
                bm2 >>= 1
            for p in p_lst2:
                if mx * p > denom_max: break
                #if p == 7 and bm == 7:
                #    print(curr)
                #print(p, curr, bm)
                if p in target_denom_pf:
                    exp = 1
                    bm2 = bm
                    i = 1
                    while bm2:
                        if bm2 & 1:
                            i2, r = divmod(i, p)
                            while not r:
                                exp += 1
                                i2, r = divmod(i2, p)
                        i += 1
                        bm2 >>= 1
                    if 2 * exp < target_denom_pf[p]: continue
                    q, r = divmod(curr, p ** (2 * exp - target_denom_pf[p]))
                    if r or not q % p:
                        continue
                    #print(p, curr, mx_pow, bm)
                    p_opts.setdefault(p, set())
                    p_opts[p].add(bm)
                    continue
                if curr % (p ** 2): continue
                exp = 1
                bm2 = bm
                i = 1
                while bm2:
                    if bm2 & 1:
                        i2, r = divmod(i, p)
                        while not r:
                            exp += 1
                            i2, r = divmod(i2, p)
                    i += 1
                    bm2 >>= 1
                if exp > 1 and curr % (p ** (2 * exp)):
                    continue
                #print(p, curr, mx_pow, bm)
                p_opts.setdefault(p, set())
                p_opts[p].add(bm)
    if not set(target_denom_pf.keys()).issubset(set(p_lst1).union(set(p_opts.keys()))):
        return []
    for st in p_opts.values():
        st.add(0)
    #print(p_opts)
    
    n_pairs = 0
    p_lst2_2 = sorted(p_opts.keys())
    #print(p_lst2_2)
    allowed_pairs = {}
    for i2 in reversed(range(1, len(p_lst2_2))):
        p2 = p_lst2_2[i2]
        allowed_pairs[p2] = {}
        for i1 in reversed(range(i2)):
            p1 = p_lst2_2[i1]
            allowed_pairs[p2][p1] = {}
            for bm1 in p_opts[p1]:
                for bm2 in p_opts[p2]:
                    bm1_ = bm1 >> (p2 - 1)
                    bm2_ = bm2 >> (p1 - 1)
                    while bm1_ or bm2_:
                        if bm1_ & 1 != bm2_ & 1: break
                        bm1_ >>= p2
                        bm2_ >>= p1
                    else:
                        allowed_pairs[p2][p1].setdefault(bm2, set())
                        allowed_pairs[p2][p1][bm2].add(bm1)
                        n_pairs += 1

    if p_lst2_2:
        #print(allowed_pairs)
        #print(n_pairs)
        curr_allowed = [(x,) for x in p_opts[p_lst2_2[-1]]]
        tot = 0
        for i1 in reversed(range(len(p_lst2_2) - 1)):
            prev_allowed = curr_allowed
            curr_allowed = []
            p1 = p_lst2_2[i1]
            for bm_lst in prev_allowed:
                curr_set = set(allowed_pairs[p_lst2_2[-1]][p1].get(bm_lst[-1], set()))
                if not curr_set: break
                for j in range(1, len(bm_lst)):
                    p2 = p_lst2_2[~j]
                    bm2 = bm_lst[~j]
                    curr_set &= allowed_pairs[p2][p1].get(bm2, set())
                    if not curr_set: break
                for bm in curr_set:
                    curr_allowed.append((bm, *bm_lst))
                    tot += 1
    else: curr_allowed = [()]
    #print(len(curr_allowed), len(set(curr_allowed)))
    #print(curr_allowed)
    #print(tot)
    #frac_counts = {}
    frac_sets = {}
    seen = {}
    for bm_lst in curr_allowed:
        denom_set = set()
        for p, bm in zip(p_lst2_2, bm_lst):
            mult = 1
            while bm:
                if bm & 1:
                    denom_set.add(p * mult)
                bm >>= 1
                mult += 1
        denom_tup = tuple(sorted(denom_set))
        #print(denom_tup)
        if denom_tup and denom_tup[0] < denom_min: continue
        tot = target
        for denom in denom_set:
            tot -= CustomFraction(1, denom ** 2)
        #frac_counts[tot] = frac_counts.get(tot, 0) + 1
        frac_sets.setdefault(tot, set())
        #if denom_tup in frac_sets[tot]: print(f"repeated denominator tuple: {denom_tup}")
        frac_sets[tot].add(denom_tup)
        if denom_tup in seen.keys():
            print(f"repeated denominator tuple: {denom_tup}, bm_lsts: {seen[denom_tup]}, {bm_lst}")
        else: seen[denom_tup] = bm_lst
        #print(tot[1])
        #print(tot, denom_set, bm_lst, ps.primeFactorisation(tot[1]))
    #print(frac_sets)
    #print("finished stage 1")

    n_p1 = len(p_lst1)
    curr = {CustomFraction(0, 1): [()]}
    seen = set()
    # Ensure the results that already equal the target with only
    # the denominators considered in the previous step are included
    # in the answer
    res = list(frac_sets.get(CustomFraction(0, 1), set()))
    #print(res)

    def recur(p_idx: int=0, denom: int=1) -> None:
        #print(p_idx, denom)
        if p_idx == n_p1:
            if denom < denom_min: return
            #print(f"denom = {denom}")
            val = CustomFraction(1, denom ** 2)
            prev = {x: list(y) for x, y in curr.items()}
            #print(prev)
            for frac, tups in prev.items():
                frac2 = frac + val
                #if frac2 == (31, 72): print("found")
                if frac2 > target: continue
                curr.setdefault(frac2, [])
                for tup in tups:
                    denom_tup2 = (*tup, denom)
                    curr[frac2].append(denom_tup2)
                    #print(pow2 * 3 ** exp3)
                    for denom_tup1 in frac_sets.get(frac2, set()):
                        ans = tuple(sorted([*denom_tup1, *denom_tup2]))
                        if ans in seen:
                            print(f"repeated solution: {ans}")
                            continue
                        if not checkSolution(ans):
                            print(f"Incorrect solution: {ans}")
                        elif ans[-1] > denom_max:
                            print(f"Solution has denominator that is too large: {ans}")
                        else:
                            seen.add(ans)
                            res.append(ans)
                        #print(exp2, exp3, nxt2[frac2], frac2)
                        #print("solution:", sorted(denom_set.union(set(denom_tup))))
            return
        p = p_lst1[p_idx]
        #denom2 = 2 * denom
        #pow_mx = 0
        #while denom2 < denom_max:
        #    pow_mx += 1
        #    denom2 *= p
        d = denom
        d_mx = denom_max# >> 1
        while d <= d_mx:
            recur(p_idx=p_idx + 1, denom=d)
            d *= p
        return

    recur(p_idx=0, denom=1)
    #print(res)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return set(res)
    """
    pow2_mx = 0
    num = denom_max // 3
    while num:
        pow2_mx += 1
        #print(num, pow2_mx)
        num >>= 1
    
    res = []
    curr2 = {(0, 1): [()]}
    seen = set()
    for exp2 in range(pow2_mx + 1):
        pow2 = 1 << exp2
        pow3_mx = denom_max // pow2
        #print(exp2, denom_max, pow3_mx)
        exp3_mx = -1
        pow3_mx2 = pow3_mx
        while pow3_mx2:
            pow3_mx2 //= 3
            exp3_mx += 1
        #print(f"exp3_mx for exp2 = {exp2} is {exp3_mx}")
        for exp3 in range(not exp2, exp3_mx + 1):
            #print("powers: ", exp2, exp3)
            d = pow2 * (3 ** exp3)
            if d < denom_min: continue
            val = (1, d ** 2)
            nxt2 = {x: list(y) for x, y in curr2.items()}
            for frac, tups in curr2.items():
                frac2 = addFractions(frac, val)
                #if frac2 == (31, 72): print("found")
                if frac2[0] * 2 >= frac2[1]: continue
                nxt2.setdefault(frac2, [])
                for tup in tups:
                    denom_tup2 = (*tup, d)
                    nxt2[frac2].append(denom_tup2)
                    #print(pow2 * 3 ** exp3)
                    for denom_tup1 in frac_sets.get(frac2, set()):
                        ans = tuple(sorted([*denom_tup1, *denom_tup2]))
                        if ans in seen:
                            print(f"repeated solution: {ans}")
                            continue
                        if not checkSolution(ans):
                            print(f"Incorrect solution: {ans}")
                        elif ans[-1] > denom_max:
                            print(f"Solution has denominator that is too large: {ans}")
                        else:
                            seen.add(ans)
                            res.append(ans)
                        #print(exp2, exp3, nxt2[frac2], frac2)
                        #print("solution:", sorted(denom_set.union(set(denom_tup))))
            curr2 = nxt2
        #print(exp2, curr)
    """
    """
    opts3_0 = []
    for bm in range(1 << (pow3_mx + 1)):
        mx_denom = 1
        bm2 = bm
        curr = 1
        frac = (0, 1)
        while bm2:
            if bm2 & 1:
                frac = addFractions(frac, (1, curr ** 2))
                mx_denom = curr
            curr *= 3
            bm2 >>= 1
        opts3_0.append((mx_denom, frac))
    opts3_0.sort()
    
    opts3_1 = []
    for bm in range(1 << pow3_mx):
        mx_denom = 1
        bm2 = bm
        curr = 3
        frac = (0, 1)
        while bm2:
            if bm2 & 1:
                frac = addFractions(frac, (1, curr ** 2))
                mx_denom = curr
            curr *= 3
            bm2 >>= 1
        opts3_1.append((mx_denom, frac))
    opts3_1.sort()
    res = 0
    for bm in range(1 << (pow2_mx + 1)):
        mx_denom = 1
        bm2 = bm
        curr = 1
        frac = (0, 1)
        while bm2:
            if bm2 & 1:
                frac = addFractions(frac, (1, curr ** 2))
                mx_denom = curr
            curr *= 2
            bm2 >>= 1
        mx_denom3 = denom_max // mx_denom
        for opt3 in (opts3_1 if bm & 1 else opts3_0):
            if opt3[0] > mx_denom3: break
            frac2 = multiplyFractions(frac, opt3[1])
            print(frac, opt3[1], frac2)
            res += frac_counts.get(frac2, 0)
    #print(pow2_mx, pow3_mx, 2 ** pow2_mx * 3 ** pow3_mx)

    #for bm2 in range()

    print(len(frac_counts))
    """

def sumsOfSquareReciprocalsCount(
    target: (1, 2),
    denom_min: int=2,
    denom_max: int=80,
) -> int:
    """
    Solution to Project Euler #152

    Finds the number of distinct ways the fraction target can be
    constructed by summing recpirocals of squares of integers,
    where the integers being squared are between denom_min and
    denom_max inclusive, and no two integers in the sum are the
    same. Two such sums are considered distinct if and only if
    there is a term that is present in one sum that is not
    present in the other sum (so two sums where the terms are
    simply a permutation of each other are not distinct).

    Args:
        Optional named:
        target (2-tuple of ints): An ordered pair of strictly
                positive integers, representing the numerator
                and denominator respectively of the target
                fraction that the sums must equal.
            Default: (1, 2)- representing a half
        denom_min (int): Strictly positive integer giving the
                smallest integer for whose squared reciprocal
                can appear in the sum.
            Default: 2
        denom_max (int): Strictly positive integer giving the
                largest integer for whose squared reciprocal
                can appear in the sum.
            Default: 80
    
    Returns:
    Integer (int) giving the number of distinct sums of square
    reciprocals of different integers between denom_min and denom_max
    inclusive that are equal to target.
    """
    #since = time.time()
    sols = set(sumsOfSquareReciprocals(
        target=CustomFraction(target[0], target[1]),
        denom_min=denom_min,
        denom_max=denom_max,
    ))
    #for sol in sols:
    #    print(sol)
    res = len(sols)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 153
"""
def findIntegerCountGaussianIntegerDivides(a: int, b: int, n_max: int) -> int:
    if not b: return n_max // a
    k = gcd(a, b)
    a_, b_ = a // k, b // k
    res = n_max // ((a_ ** 2 + b_ ** 2) * k)
    #print((a, b), res)
    return res
"""

def findRealPartSumOverGaussianIntegerDivisors(
    n_max: int=10 ** 8,
) -> int:
    """
    Solution to Project Euler #153

    Calculates the sum of the real part of all Gaussian integer
    factors with positive real part over all strictly positive
    integers no greater than
    n_max.

    Args:
        Optional named:
        n_max (int): The largest integer whose Gaussian integer
                factors are considered in the sum
            Default: 10 ** 8
    
    Returns:
    Integer (int) representing the sum of the real part of all
    Gaussian integer factors with positive real part over all
    strictly positive integers no greater that n_max.

    Example:
        >>> findRealPartSumOverGaussianIntegerDivisors(n_max=5)
        35

        This indicates that the sum over the Gaussian integer
        factors of the integers between 1 and 5 inclusive is
        35. The Gaussian integer factors of these integers with
        positive real part are:
            1: 1
            2: 1, 1 + i, 1 - i, 2
            3: 1, 3
            4: 1, 1 + i, 1 - i, 2, 2 + 2 * i, 2 - 2 * i, 4
            5: 1, 1 + 2 * i, 1 - 2 * i, 2 + i, 2 - i, 5
        The sum over the real part of all of these factors is
        indeed 35.
    
    Brief outline of rationale:
    We consider this sum from the perspective of the Gaussian
    integer factors rather than the integers, rephrasing the
    question into the equivalent one of, calculate the sum over
    all Gaussian integers with positive real part of the
    real part multiplied by the number of strictly positive
    integers no greater than n_max for which that Gaussian
    integer is a factor.
    We observe that (a + bi) is a factor of an integer m if
    and only if (a - bi), (b + ai) and (b - ai) are factors
    of m (TODO- prove this)
    There are three cases to consider: Gaussian integers with no
    imaginary part (i.e. integers); Gaussian integers that are
    positive integer multiples of (1 + i) and (1 - i); and the
    other Gaussian integers with positive real part (i.e. those
    with non-zero imaginary parts whose real and imaginary parts
    are different sizes).
    Case 1: For Gaussian integers with no imaginary parts, the
    number of strictly positive integers no greater than n_max
    for which this is a factor is simply the number of positive
    integer multiples of this number that are no greater than
    n_max, which is given by the floor of n_max divided by
    the number. Therefore the contribution of these numbers to
    the overall sum is simply:
        (sum a from 1 to n_max) (a * (n_max // a))
    Case 2: As previously noted, an integer is divisible by (1 + i)
    if and only if it is also divisible by (1 - i). As such,
    an integer is divisible by (1 + i) or (1 - i) if and only if
    it is divisible by (1 + i) * (1 - i) = 2. It follows that for
    positive integer a, the positive integer m is divisible by
    a * (1 + i) if and only if it is divisible by a * (1 - i) and
    is a multiple of 2 * a. Consequently, the contribution to the
    sum of all Gaussian integers of the forms a * (1 + i) and
    a * (1 - i) for positive integer a is:
        2 * (sum a from 1 to n_max) a * (n_max // (2 * a))
    Case 3: TODO
    """
    #since = time.time()

    memo = {}
    def divisorSum(mag_sq: int) -> int:
        
        args = mag_sq
        if args in memo.keys(): return memo[args]
        
        def countSum(m: int) -> int:
            return (m * (m + 1)) >> 1

        rt = isqrt(n_max // mag_sq)

        res = 0
        for i in range(1, (n_max // (mag_sq * (1 + rt))) + 1):
            res += i * (n_max // (i * mag_sq))
        for i in range(1, rt + 1):
            res += i * (countSum(n_max // (i * mag_sq)) - countSum(n_max // ((i + 1) * mag_sq)))
        """
        j1, j2 = n_max // mag_sq, n_max // (2 * mag_sq)
        res = ((j1 * (j1 + 1)) >> 1) - ((j2 * (j2 + 1)) >> 1)
        for i in range(1, (n_max // (2 * mag_sq)) + 1):
            res += (n_max // (i * mag_sq)) * i
        """
        memo[args] = res
        return res

    res = 0
    for a in range(1, n_max + 1):
        res += a * (n_max // a)
        a_sq = a ** 2
        for b in range(1, a):
            mag_sq = a_sq + b ** 2
            if mag_sq > n_max: break
            if gcd(a, b) != 1: continue
            res += 2 * (a + b) * divisorSum(mag_sq)
            
        #print(a, res)
    res += 2 * divisorSum(2)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 154
def multinomialCoefficientMultiplesCount(
    n: int=2 * 10 ** 5,
    n_k: int=3,
    factor_p_factorisation: Dict[int, int]={2: 12, 5: 12},
) -> int:
    """
    For integers n and n_k and a prime factorisation
    factor_p_factorisation, finds the number of non-zero n_k-nomial
    coefficients with n as the number chosen from that are divisible
    by the integer with the prime factorisation factor_p_factorisation.

    For any list of n_k integers:
        (k_1, k_2, ..., k_(n_k))
    the n_k-nomial coefficient (also termed n_k multinomial coefficients)
    is the number of ways n items can be partitioned into n_k sets with
    sizes corresponding to the integers in the above list, where n is
    the number chosen from, given by the sum of the list of integers,
    the ordering within the sets does not matter and the sets themselves
    are considered to be distinguishable (so that if two of the list of
    n_k integers have the same value, the partitionings created by
    swapping the items from one of the corresponding sets to the
    other results in a different partitioning, despite each item
    being grouped with the same items in both partitionings). If
    any of the n_k integers in the list are negative, then the
    corresponding n_k-nomial coefficient is defined to be zero.
    
    The value of this coefficient for non-negative k_1, k_2, ...,
    k_(n_k) and n being the sum of these integers is given by:
        n! / (k_1! * k_2! * ... * k_(n_k)!)

    Args:
        Optional named:
        n (int): Non-negative integer giving the number chosen from
                for the n_k-nomial coefficients considered, i.e. the
                sum of the partition sizes in the n_k-nomial
                coefficients considered.
            Default: 2 * 10 ** 5
        n_k (int): Strictly positive integer giving the number of
                partitions in the multinomial coefficients considered.
            Default: 3
        factor_p_factorisation (dict): Dictionary representing a
                prime factorisation of the strictly positive integer
                by which the n_k-nomial coefficients for given n
                and n_k counted are to be divisible, whose keys are
                the prime numbers that appear in the prime
                factorisation of the integer in question, with the
                corresponding value being the number of times that
                prime appears in the factorisation (i.e. the power
                of that prime in the prime factorisation of the
                integer). An empty dictionary corresponds to the
                multiplicative identity (i.e. 1).
            Default: {2: 12, 5: 12} (the prime factorisation of 10 ** 12)

    Returns:
    Integer (int) giving the number of non-zero n_k-nomial coefficients
    with n as the number chosen from that are divisible by the integer with
    the prime factorisation factor_p_factorisation.

    TODO- reword for clarity

    Outline of rationale:
    TODO
    """
    #since = time.time()
    #ps = PrimeSPFsieve(n_max=factor)
    #factor_p_factorisation = ps.primeFactorisation(factor)
    p_lst = sorted(factor_p_factorisation.keys())
    n_p = len(p_lst)
    target = [factor_p_factorisation[p] for p in p_lst]
    print(p_lst, target)
    #for p in p_lst:
    #    n2 = n
    #    target.append(-factor_p_factorisation[p])
    #    while n2:
    #        n2 //= p
    #        target[-1] += n2
    
    
    def createNDimensionalArray(shape: List[int]) -> list:
        n_dims = len(shape)
        def recur(lst: list, dim_idx: int) -> list:
            if dim_idx == n_dims - 1:
                for _ in range(shape[dim_idx]):
                    lst.append(0)
                return
            for _ in range(shape[dim_idx]):
                lst.append([])
                recur(lst[-1], dim_idx + 1)
            return lst
        return recur([], 0)

    def getNDimensionalArrayElement(arr: list, inds: Tuple[int]) -> int:
        res = arr
        for idx in inds:
            res = res[idx]
        return res

    def setNDimensionalArrayElement(arr: list, inds: Tuple[int], val: int) -> None:
        lst = arr
        for idx in inds[:-1]:
            lst = lst[idx]
        lst[inds[-1]] = val
        return
    
    def modifyNDimensionalArrayElement(arr: list, inds: Tuple[int], delta: int) -> None:
        lst = arr
        for idx in inds[:-1]:
            lst = lst[idx]
        lst[inds[-1]] += delta
        return
    
    def findNDimensionalArrayShape(arr: list) -> Tuple[int]:

        lst = arr
        res = []
        while not isinstance(lst, int):
            res.append(len(lst))
            if not lst: break
            lst = lst[0]
        return tuple(res)
    
    def deepCopyNDimensionalArray(arr: list) -> list:
        shape = findNDimensionalArrayShape(arr)
        res = createNDimensionalArray(shape)

        def recur(inds: List[int], lst1: Union[list, int], lst2: Union[list, int]) -> None:
            dim_idx = len(inds)
            if dim_idx == len(shape) - 1:
                for idx in range(shape[-1]):
                    lst2[idx] = lst1[idx]
                return
            inds.append(0)
            for idx in range(shape[dim_idx]):
                recur(inds, lst1[idx], lst2[idx])
            inds.pop()
            return
        recur([], arr, res)
        return res
    
    def addDisplacedNDimensionalArray(arr1: list, arr2: list, displacement: Tuple[int]) -> None:
        shape1 = findNDimensionalArrayShape(arr1)
        shape2 = findNDimensionalArrayShape(arr2)

        def recur(inds: List[int], lst1: Union[list, int], lst2: Union[list, int]) -> None:
            dim_idx = len(inds)
            if dim_idx == len(shape1) - 1:
                for idx2 in range(shape2[-1], shape1[-1] - displacement[-1]):
                    lst1[idx2 + displacement[-1]] += lst2[idx2]
                return
            inds.append(0)
            for idx2 in range(shape2[dim_idx], shape1[dim_idx] - displacement[dim_idx]):
                recur(inds, lst1[idx2 + displacement[dim_idx]], lst2[dim_idx])
            inds.pop()
            return
        recur([], arr1, arr2)
        return

    def convertSumAllIndicesNoLessThanArray(arr: list) -> list:
        shape = findNDimensionalArrayShape(arr)
        #print(arr)
        #print(shape)
        def recur(inds: Tuple[int], lst: Union[list, int], lst2: Union[list, int], dim_idx: int, add_cnt_parity: bool=False, any_add: bool=False) -> None:
            idx = inds[dim_idx]
            if dim_idx == len(shape) - 1:
                if any_add:
                    lst[idx] += (lst2[idx] if add_cnt_parity else -lst2[idx])
                    if idx + 1 < shape[dim_idx]:
                        lst[idx] += (-lst2[idx + 1] if add_cnt_parity else lst2[idx + 1])
                    return
                    #lst[shape[-1] - 1] += -lst2[shape[-1] - 1] if add_cnt_parity else lst2[shape[-1] - 1]
                    #for idx in reversed(range(shape[-1] - 1)):
                    #    lst[idx] += (lst2[idx] if add_cnt_parity else -lst2[idx]) - (lst2[idx + 1] if add_cnt_parity else -lst2[idx + 1])
                    #return
                #for idx in reversed(range(shape[-1] - 1)):
                #    lst[idx] += lst2[idx + 1]
                if idx + 1 < shape[dim_idx]:
                    lst[idx] += lst2[idx + 1]
                return
                #return lst2 if add_cnt_parity else -lst2
            #print(dim_idx)
            recur(inds, lst[idx], lst2[idx], dim_idx + 1, add_cnt_parity=add_cnt_parity, any_add=any_add)
            if idx + 1 < shape[dim_idx]:
                recur(inds, lst[idx], lst2[idx + 1], dim_idx + 1, add_cnt_parity=not add_cnt_parity, any_add=True)
            #recur(lst[shape[dim_idx] - 1], lst2[shape[dim_idx] - 1], dim_idx + 1, add_cnt_parity=add_cnt_parity, any_add=any_add)
            #for idx in reversed(range(shape[dim_idx] - 1)):
            #    recur(lst[idx], lst2[idx], dim_idx + 1, add_cnt_parity=add_cnt_parity, any_add=any_add)
            #    recur(lst[idx], lst2[idx + 1], dim_idx + 1, add_cnt_parity=add_cnt_parity, any_add=True)
            return
        
        def recur2(inds: List[int]) -> None:
            #print(f"recur2(): {inds}")
            dim_idx = len(inds)
            if dim_idx == len(shape):
                #print(inds)
                recur(inds, arr, arr, 0, add_cnt_parity=False, any_add=False)
                #print(arr)
                return
            inds.append(0)
            for idx in reversed(range(shape[dim_idx])):
                inds[-1] = idx
                recur2(inds)
            inds.pop()
            return
        #recur(arr, arr, 0, add_cnt_parity=False, any_add=False)
        recur2([])
        return arr
        """
        def inclExcl(inds: Tuple[int]) -> int:
            
            def recur2(lst: Union[list, int], dim_idx: int, add_cnt_parity: bool=False) -> int:
                if dim_idx == len(shape):
                    return lst if add_cnt_parity else -lst
                res = recur2(lst[inds[dim_idx]], dim_idx + 1, add_cnt_parity=add_cnt_parity)
                if inds[dim_idx] < shape[dim_idx] - 1:
                    res += recur2(lst[inds[dim_idx] + 1], dim_idx + 1, add_cnt_parity=not add_cnt_parity)
                return res
            return recur2(arr, 0, add_cnt_parity=False)
        
        def recur(inds: List[int], lst: Union[list, int]) -> None:
            if len(inds) == len(shape):
                val = inclExcl(inds)

                return
            inds.append(0)
            for idx in reversed(range(len(inds))):
                inds[-1] = idx
            return
        """
    
    
    shape = tuple(x + 1 for x in target)
    curr = [createNDimensionalArray(shape) for _ in range(2)]
    setNDimensionalArrayElement(curr[0], [0] * len(shape), 1)
    setNDimensionalArrayElement(curr[1], [0] * len(shape), 2)
    counts = [tuple([0] * n_p)] * 2
    counts_curr = [0] * n_p
    for n2 in range(2, n + 1):
        print(f"n2 = {n2}")
        for i, p in enumerate(p_lst):
            n3 = n2
            while True:
                n3, r = divmod(n3, p)
                if r: break
                counts_curr[i] += 1
        counts.append(tuple(counts_curr))
        curr.append(createNDimensionalArray(shape))
        for k in range(n2 - (n2 >> 1)):
            #print(f"k = {k}")
            inds = tuple(min(t, x - y - z) for t, x, y, z in zip(target, counts[n2], counts[k], counts[n2 - k]))
            #print(inds)
            modifyNDimensionalArrayElement(curr[-1], inds, 2)
        if not n2 & 1:
            #print(f"k = {n2 >> 1}")
            inds = tuple(min(t, x - 2 * y) for t, x, y in zip(target, counts[n2], counts[n2 >> 1]))
            #print(inds)
            modifyNDimensionalArrayElement(curr[-1], inds, 1)
        #print(f"n2 = {n2}:")
        #print(curr[-1])
        convertSumAllIndicesNoLessThanArray(curr[-1])
        #print(curr[-1])
    
    #print(curr[n])

    if n_k == 2:
        return getNDimensionalArrayElement(curr[n], target)

    for _ in range(3, n_k):
        for n2 in reversed(range(n + 1)):
            arr = createNDimensionalArray(shape)
            for k in reversed(range(n2 + 1)):
                delta = tuple(x - y - z for x, y, z in zip(counts[n2], counts[k], counts[n2 - k]))
                addDisplacedNDimensionalArray(arr, curr[k], delta)
            curr[n2] = arr
    
    res = 0
    for k in range(n + 1):
        print(f"k = {k}")
        inds = tuple(max(t - (x - y - z), 0) for t, x, y, z in zip(target, counts[n], counts[k], counts[n - k]))
        #print(inds, target)
        res += getNDimensionalArrayElement(curr[k], inds)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
    """
    arr = createNDimensionalArray((2, 2, 2))
    setNDimensionalArrayElement(arr, (1, 1, 1), 3)
    setNDimensionalArrayElement(arr, (0, 1, 0), 4)
    setNDimensionalArrayElement(arr, (0, 0, 1), 5)
    print(arr)
    convertSumAllIndicesNoLessThanArray(arr)
    print(arr)
    """

def multinomialCoefficientMultiplesCount2(
    n: int=2 * 10 ** 5,
    n_k: int=3,
    factor_p_factorisation: Dict[int, int]={2: 12, 5: 12},
) -> int:
    """
    Solution to Project Euler #154

    For integers n and n_k and a prime factorisation
    factor_p_factorisation, finds the number of non-zero n_k-nomial
    coefficients with n as the number chosen from that are divisible
    by the integer with the prime factorisation factor_p_factorisation.

    For any list of n_k integers:
        (k_1, k_2, ..., k_(n_k))
    the n_k-nomial coefficient (also termed n_k multinomial coefficients)
    is the number of ways n items can be partitioned into n_k sets with
    sizes corresponding to the integers in the above list, where n is
    the number chosen from, given by the sum of the list of integers,
    the ordering within the sets does not matter and the sets themselves
    are considered to be distinguishable (so that if two of the list of
    n_k integers have the same value, the partitionings created by
    swapping the items from one of the corresponding sets to the
    other results in a different partitioning, despite each item
    being grouped with the same items in both partitionings). If
    any of the n_k integers in the list are negative, then the
    corresponding n_k-nomial coefficient is defined to be zero.
    
    The value of this coefficient for non-negative k_1, k_2, ...,
    k_(n_k) and n being the sum of these integers is given by:
        n! / (k_1! * k_2! * ... * k_(n_k)!)

    Args:
        Optional named:
        n (int): Non-negative integer giving the number chosen from
                for the n_k-nomial coefficients considered, i.e. the
                sum of the partition sizes in the n_k-nomial
                coefficients considered.
            Default: 2 * 10 ** 5
        n_k (int): Strictly positive integer giving the number of
                partitions in the multinomial coefficients considered.
            Default: 3
        factor_p_factorisation (dict): Dictionary representing a
                prime factorisation of the strictly positive integer
                by which the n_k-nomial coefficients for given n
                and n_k counted are to be divisible, whose keys are
                the prime numbers that appear in the prime
                factorisation of the integer in question, with the
                corresponding value being the number of times that
                prime appears in the factorisation (i.e. the power
                of that prime in the prime factorisation of the
                integer). An empty dictionary corresponds to the
                multiplicative identity (i.e. 1).
            Default: {2: 12, 5: 12} (the prime factorisation of 10 ** 12)

    Returns:
    Integer (int) giving the number of non-zero n_k-nomial coefficients
    with n as the number chosen from that are divisible by the integer with
    the prime factorisation factor_p_factorisation.

    TODO- reword for clarity

    Outline of rationale:
    TODO
    """
    # TODO- try to make faster
    #since = time.time()
    p_lst = sorted(factor_p_factorisation.keys())
    n_p = len(p_lst)
    target = [factor_p_factorisation[p] for p in p_lst]
    #print(p_lst, target)

    counts_n = []
    for p in p_lst:
        counts_n.append(0)
        n_ = n
        while n_:
            n_ //= p
            counts_n[-1] += n_
    #print(f"counts_n = {counts_n}")
    counts = [tuple([0] * n_p)] * 2
    counts_curr = [0] * n_p
    k1_mn = ((n - 1) // n_k) + 1

    def recur(k_num_remain: int, k_sum_remain: int, last_k: int, repeat_streak: int, mult: int, facts_remain: List[int]) -> int:
        #print(facts_remain, target)
        if any(x < y for x, y in zip(facts_remain, target)): return 0
        k1_mn = ((k_sum_remain - 1) // k_num_remain) + 1
        #print(f"k_num_remain = {k_num_remain}, k_sum_remain = {k_sum_remain}, last_k = {last_k}, k1_mn = {k1_mn}")
        if k_num_remain == 2:
            #print("hi")
            res = 0
            if not k_sum_remain & 1:
                k1 = k_sum_remain >> 1
                #print(f"k2 = {k1}, {[x - 2 * y for x, y in zip(facts_remain, counts[k1])]}")
                if all(x - 2 * y >= t for t, x, y in zip(target, facts_remain, counts[k1])):
                    #print("hello1")
                    r_streak = (repeat_streak if k1 == last_k else 0) + 1
                    ans = mult * r_streak * (r_streak + 1)
                    #print(f"k2 = {k1}, k3 = {}, a")
                    res +=  mult // (r_streak * (r_streak + 1))
            k1_mn2 = k1_mn + (k1_mn << 1 == k_sum_remain)
            if last_k >= k1_mn2 and last_k <= k_sum_remain and all(x - y - z >= t for t, x, y, z in zip(target, facts_remain, counts[last_k], counts[k_sum_remain - last_k])):
                res += mult // (repeat_streak + 1)
                
            for k1 in range(k1_mn2, min(last_k, k_sum_remain + 1)):
                #print(f"k2 = {k1}, {[x - y - z for x, y, z in zip(facts_remain, counts[k1], counts[k_sum_remain - k1])]}")
                if all(x - y - z >= t for t, x, y, z in zip(target, facts_remain, counts[k1], counts[k_sum_remain - k1])):
                    #print("hello2")
                    r_streak = (repeat_streak if k1 == last_k else 0) + 1
                    res += mult // r_streak
            return res
        if last_k >= k1_mn and last_k <= k_sum_remain:
            k1 = last_k
            k_num_remain2 = k_num_remain - 1
            k_sum_remain2 = k_sum_remain - k1
            last_k2 = last_k
            repeat_streak2 = repeat_streak + 1
            mult2 //= repeat_streak2
            facts_remain2 = [x - y for x, y in zip(facts_remain, counts[k1])]
            res += recur(k_num_remain2, k_sum_remain2, last_k2, repeat_streak2, mult2, facts_remain2)
        for k1 in range(k1_mn, min(last_k, k_sum_remain + 1)):
            k_num_remain2 = k_num_remain - 1
            k_sum_remain2 = k_sum_remain - k1
            last_k2 = last_k
            facts_remain2 = [x - y for x, y in zip(facts_remain, counts[k1])]
            res += recur(k_num_remain2, k_sum_remain2, last_k2, 1, mult, facts_remain2)
        return res
    res = 0
    #print(f"k1_mn = {k1_mn}")
    for k1 in range(2, n + 1):
        print(f"k1 = {k1}")
        for i, p in enumerate(p_lst):
            k1_ = k1
            while True:
                k1_, r = divmod(k1_, p)
                if r: break
                counts_curr[i] += 1
        counts.append(tuple(counts_curr))
        if k1 < k1_mn: continue
        #print("hello", n_k - 1, n - k1, facts_remain)
        res += recur(n_k - 1, n - k1, k1, 1, math.factorial(n_k), [x - y for x, y in zip(counts_n, counts_curr)])
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 155
def findNewCapacitorCombinationValuesNoLessThanOne(
    max_n_capacitors: int,
) -> List[Set[Tuple[int, int]]]:
    """
    Finds the values of capacitances that are possible for
    combinations of a number of capacitors with unit capacitance
    for each number of capacitors up to max_n_capacitors, only
    including values that are not possible for any smaller
    number of capacitors.

    Two capacitor combinations with a certain number of capacitors
    can be combined to form a third by connecting the two either
    in series or in parallel. If the two capacitor combinations
    being combined together have capacitances C1 and C2, and
    then if they are combined in parallel then the resultant
    overall capacitance is (C1 + C2), while if they are combined
    in series then the resultant capacitance is 1 / (1 / C1 + 1 / C2).
    The combinations are built up from individual capacitors with
    unit capacitors.

    For example, if we consider the combinations that contain no
    more than two unit capacitors, the possible capacitances result
    from a single capacitor, with capacitance of 1 (trivially),
    a combination of two capacitors in parallel with capacitance
    (1 + 1) = 2, and a combination of two capacitors in series
    with capacitance 1 / (1 / 1 + 1 / 1) = 1 / 2.

    Args:
        Required positional:
        max_n_capacitors (int): The largest number of capacitors
                considered.
    
    Returns:
    List of sets of 2-tuples of ints, where the i:th element in
    the list is the set of fractions representing the new capacitances
    possible for i capacitors.
    """
    seen = {CustomFraction(1, 1)}
    curr = [set(), {CustomFraction(1, 1)}]
    for i in range(2, max_n_capacitors + 1):
        st = set()
        for j in range(1, (i >> 1) + 1):
            for frac1 in curr[j]:
                frac1_inv = CustomFraction(frac1.denominator, frac1.numerator)
                for frac2 in curr[i - j]:
                    frac2_inv = CustomFraction(frac2.denominator, frac2.numerator)
                    for frac3 in (frac1 + frac2, frac1 + frac2_inv, frac1_inv + frac2):
                        if frac3 not in seen:
                            st.add(frac3)
                            seen.add(frac3)
                    frac3 = frac1_inv + frac2_inv
                    if frac3 not in seen and frac3.numerator >= frac3.denominator:
                        st.add(frac3)
                        seen.add(frac3)
                    frac3_inv = CustomFraction(frac3.denominator, frac3.numerator)
                    if frac3_inv not in seen and frac3_inv.numerator >= frac3_inv.denominator:
                        st.add(frac3_inv)
                        seen.add(frac3_inv)
        curr.append(st)
        print(f"n = {i}, number of new capacitances no less than one = {len(st)}")
        #print(i, st)
        #tot_set |= st
        print(f"n_capacitors = {i}, number of new capacitances no less than one = {len(st)}, cumulative n_combinations = {2 * sum(len(x) for x in curr) - 1}")
    return curr


def possibleCapacitorCombinationValuesNoLessThanOne(
    max_n_capacitors: int,
) -> List[Set[Tuple[int, int]]]:
    """
    Finds the values of capacitances that are possible for
    combinations of a number of capacitors with unit capacitance
    for each exact number of capacitors up to max_n_capacitors.

    Two capacitor combinations with a certain number of capacitors
    can be combined to form a third by connecting the two either
    in series or in parallel. If the two capacitor combinations
    being combined together have capacitances C1 and C2, and
    then if they are combined in parallel then the resultant
    overall capacitance is (C1 + C2), while if they are combined
    in series then the resultant capacitance is 1 / (1 / C1 + 1 / C2).
    The combinations are built up from individual capacitors with
    unit capacitors.

    For example, if we consider the combinations that contain no
    more than two unit capacitors, the possible capacitances result
    from a single capacitor, with capacitance of 1 (trivially),
    a combination of two capacitors in parallel with capacitance
    (1 + 1) = 2, and a combination of two capacitors in series
    with capacitance 1 / (1 / 1 + 1 / 1) = 1 / 2.

    Args:
        Required positional:
        max_n_capacitors (int): The largest number of capacitors
                considered.
    
    Returns:
    List of sets of 2-tuples of ints, where the i:th element in
    the list is the set of fractions representing the capacitances
    possible for combinations of exactly i unit capacitors.
    """
    curr = [set(), {CustomFraction(1, 1)}]
    cumu = 1
    tot_set = {CustomFraction(1, 1)}
    for i in range(2, max_n_capacitors + 1):
        st = set()
        for j in range(1, (i >> 1) + 1):
            for frac1 in curr[j]:
                frac1_inv = CustomFraction(frac1.denominator, frac1.numerator)
                for frac2 in curr[i - j]:
                    frac2_inv = CustomFraction(frac2.denominator, frac2.numerator)
                    st.add(frac1 + frac2)
                    st.add(frac1 + frac2_inv)
                    st.add(frac1_inv + frac2)
                    frac3 = frac1_inv + frac2_inv
                    if frac3.numerator >= frac3.numerator:
                        st.add(frac3)
                    frac3_inv = CustomFraction(frac3.denominator, frac3.numerator)
                    if frac3_inv.numerator >= frac3_inv.denominator:
                        st.add(frac3_inv)
        curr.append(st)
        tot_set |= st
        print(f"n_capacitors = {i}, n_combinations = {2 * len(st) - (CustomFraction(1, 1) in st)}, cumulative n_combinations = {2 * len(tot_set) - ((1, 1) in tot_set)}")
    return curr

def countDistinctCapacitorCombinationValues(max_n_capacitors: int=18) -> int:
    """
    Solution to Project Euler #155

    Finds the number of distinct values of capacitances that are
    possible from at most max_n_capacitors combinations of capacitors
    with unit capacitance when connecting them in parallel or
    series.

    Two capacitor combinations with a certain number of capacitors
    can be combined to form a third by connecting the two either
    in series or in parallel. If the two capacitor combinations
    being combined together have capacitances C1 and C2, and
    then if they are combined in parallel then the resultant
    overall capacitance is (C1 + C2), while if they are combined
    in series then the resultant capacitance is 1 / (1 / C1 + 1 / C2).
    The combinations are built up from individual capacitors with
    unit capacitors.

    For example, if we consider the combinations that contain no
    more than two unit capacitors, the possible capacitances result
    from a single capacitor, with capacitance of 1 (trivially),
    a combination of two capacitors in parallel with capacitance
    (1 + 1) = 2, and a combination of two capacitors in series
    with capacitance 1 / (1 / 1 + 1 / 1) = 1 / 2.

    The solution to this problem is equivalent to the sequence given
    by OEIS A153588.

    Args:
        Optional named:
        max_n_capacitors (int): The largest number of capacitors
                in any combination considered.
            Default: 18
    
    Returns:
    Integer (int) giving the number of distinct capacitance values
    possible from combining up to max_n_capacitors capacitors with unit
    capacitors by combinating them in parallel and in series.
    
    The solution to this problem is equivalent to the sequence given
    by OEIS A153588.
    """
    #since = time.time()
    new_combs_lst = findNewCapacitorCombinationValuesNoLessThanOne(max_n_capacitors)
    res = 2 * sum(len(x) for x in new_combs_lst) - 1
    """
    combs_lst = possibleCapacitorCombinationValuesNoLessThanOne(max_n_capacitors)
    tot_combs = combs_lst[0]
    for comb in combs_lst[1:]:
        tot_combs |= comb
    res = 2 * len(tot_combs) - ((1, 1) in tot_combs)
    """
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 156
def cumulativeDigitCount(d: int, n_max: int, base: int=10) -> int:
    """
    For a given base and the value of a non-zero digit in that base (i.e.
    an integer between 1 and (base - 1) inclusive), finds the number of
    times the digit appears in the collective representations in the chosen
    base of all the strictly positive integers no greater than a chosen
    integer.

    Args:
        Required positional:
        d (int): Non-zero integer between 1 and (base - 1) inclusive
                giving the value of the digit of interest in the chosen
                base.
        n_max (int): Integer giving the largest number considered when
                counting the occurrences of the digit d.
        
        Optional named:
        base (int): Integer strictly greater than 1 giving the base
                in which d is a digit and in which all numbers are
                to be expressed when counting the number of
                occurrences of the digit d.
            Defualt: 10
    
    Returns:
    Integer (int) giving the number of occurrences of the digit with
    value d in the collective representations in the base base of all
    the strictly positive integers no greater than n_max.
    """

    def countNumsLTBasePow(base_exponent: int) -> int:
        #if base_exponent == 0: return 0
        #return countNumsLTBasePow(base_exponent - 1) + base ** (base_exponent - 1)
        return 0 if base_exponent <= 0 else base_exponent * base ** (base_exponent - 1)
    
    res = 0
    n2 = n_max + 1
    base_exp = 0
    num2 = 0
    while n2:
        n2, d2 = divmod(n2, base)
        #print(base_exp, countNumsLTBasePow(base_exp))
        res += countNumsLTBasePow(base_exp) * d2
        if d2 > d:
            res += base ** (base_exp)
        elif d2 == d:
            res += num2
        num2 += d2 * base ** base_exp
        base_exp += 1
        #print(num2, res)
    
    return res

def cumulativeNonZeroDigitCountEqualsNumber(d: int, base: int=10) -> List[int]:
    """
    For a given base and the value of a non-zero digit in that base (i.e.
    an integer between 1 and (base - 1) inclusive), finds the all the
    integers such that the total of the number of occurrences of the digit
    d over the collective representations of all the strictly positive
    integers no greater than that integer in the chosen base is equal to
    that number.

    Args:
        Required positional:
        d (int): Non-zero integer between 1 and (base - 1) inclusive
                giving the value of the digit of interest in the chosen
                base.
        
        Optional named:
        base (int): Integer strictly greater than 1 giving the base
                in which d is a digit and in which all numbers are
                to be expressed when counting the number of
                occurrences of the digit d.
            Default: 10
    
    Returns:
    List of integers (ints) giving the value of all the integers with
    the described property for the chosen base and digit in that base
    in strictly increasing order.

    Outline of rationale:
    TODO- particularly the justification of exp_max
    """
    exp_max = base
    res = []

    def recur(exp: int, pref: int=0, pref_d_count: int=0) -> None:
        #print(f"exp = {exp}, pref = {pref}, pref_d_count = {pref_d_count}")
        num = -1
        num2 = pref * base# + (not pref)
        while num2 > num and num2 < (pref + 1) * base:
            num = num2
            num2 = cumulativeDigitCount(d, num * base ** exp, base=base) // (base ** exp)
        if num2 >= (pref + 1) * base: return
        mn = num
        pref2 = num * base ** exp
        num = float("inf")
        num2 = (pref + 1) * base - 1
        while num2 < num and num2 >= mn:
            num = num2
            num2 = cumulativeDigitCount(d, (num + 1) * base ** exp - 1, base=base) // (base ** exp)
        if num2 < mn: return
        pref3 = num * base ** exp

        """
        lft, rgt = pref * base + (not pref), (pref + 1) * base
        while lft < rgt:
            mid = lft + ((rgt - lft) >> 1)
            cnt = cumulativeDigitCount(d, (mid + 1) * base ** exp - 1, base=base)
            if cnt < mid * base ** exp:
                lft = mid + 1
            else: rgt = mid
        if lft == (pref + 1) * base: return
        print("hi")
        pref2 = lft * base ** exp
        lft, rgt = lft, (pref + 1) * base - 1
        while lft < rgt:
            mid = rgt - ((rgt - lft) >> 1)
            cnt = cumulativeDigitCount(d, (mid) * base ** exp, base=base)
            print(f"exp = {exp}, mid = {mid}, num = {(mid + 1) * base ** exp - 1}, cnt = {cnt}")
            if cnt > (mid + 1) * base ** exp:
                rgt = mid - 1
            else: lft = mid
        pref3 = lft * base ** exp
        """
        rng = [pref2, pref3]
        #print(rng)
        #pref2 = (pref * base + (not pref)) * base ** exp
        #pref3 = ((pref + 1) * base ** (exp + 1)) - 1
        pref2_cnt = cumulativeDigitCount(d, pref2, base=base)
        #pref3_cnt = cumulativeDigitCount(d, pref3, base=base)
        #rng = [max(pref2, pref2_cnt), min(pref3, pref3_cnt)]
        #print(rng)
        #if rng[0] > rng[1]: return
        #print(rng, pref2, pref2_cnt, pref3, pref3_cnt)
        #pref0 = (pref // base) * base
        if not exp:
            # Review- optimise (consider cases pref_d_count = 0, pref_d_count > 1
            # and pref_d_count = 1 separately, with the latter being the most
            # complicated)
            #d0 = max(rng[0], int(pref == 0))
            #print(f"pref2_cnt = {pref2_cnt}")
            d0 = rng[0] % base
            d1 = rng[1] % base
            cnt = pref2_cnt + d0 * pref_d_count + (d < d0)
            #cnt_start = cumulativeDigitCount(d, start, base=base)
            num = pref2
            if num == cnt:
                #print("found!")
                #print(num, cumulativeDigitCount(d, num, base=10))
                res.append(num)
            #print(f"num = {num}, cnt = {cnt}")
            #print(d0, rng[1] + 1)
            for d2 in range(d0 + 1, d1 + 1):
                num += 1
                cnt += pref_d_count + (d2 == d)
                #print(d2, cnt, num)
                if cnt == num:
                    #print("found!")
                    #print(num, cumulativeDigitCount(d, num, base=10))
                    res.append(num)
            return
        rng2 = (rng[0] // base ** (exp), rng[1] // base ** (exp))
        #print(rng2[0], rng2[1] + 1)
        for pref4 in range(rng2[0], rng2[1] + 1):
            recur(exp=exp - 1, pref=pref4, pref_d_count=pref_d_count + (pref4 % base == d))
        return
    
    #for exp in range(0, exp_max + 1):
    #    recur(exp, pref=0, pref_d_count=0)
    recur(exp_max, pref=0, pref_d_count=0)
    #print(d, res)
    #print(sum(res))
    return res

def cumulativeNonZeroDigitCountEqualsNumberSum(base: int=10) -> int:
    """
    Solution to Project Euler #156

    Finds the sum of the following sums for all non-zero digit values
    in the chosen base (i.e. the integers between 1 and (base - 1)
    inclusive:
    
    For each non-zero digit value in the chosen base (i.e. an integer
    between 1 and (base - 1) inclusive), calculates the sum of all the
    integers such that the total of the number of occurrences of the digit
    d over the collective representations of all the strictly positive
    integers no greater than that integer in the chosen base is equal to
    that number. These sums are then added together to produce the final
    result.

    Note that if an integer has the described property for more than
    one digit, it will be included in the final sum a number of times
    equal to the number of digits for which it has that property.

    Args:
        Optional named:
        base (int): Integer strictly greater than 1 giving the base
                in which d is a digit and in which all numbers are
                to be expressed when counting the number of
                occurrences of the digit d.
            Default: 10
    
    Returns:
    Integer (int) giving the sum of sums described.

    Outline of rationale:
    See Outline of rationale in the documentation for the function
    cumulativeNonZeroDigitCountEqualsNumber().
    """
    since = time.time()
    res = 0
    for d in range(1, base):
        res += sum(cumulativeNonZeroDigitCountEqualsNumber(d, base=base))
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 157
def reciprocalPartnerSumsEqualToMultipleOfReciprocal(a: int, reciprocal: int) -> List[int]:
    """
    Given the strictly positive integers a and reciprocal, identifies
    all strictly positive integers b such that:
        1 / a + 1 / b = p / reciprocal
    for some strictly positive integer p.

    Args:
        Required positional:
        a (int): Strictly positive integer giving the value of a in
                the above equation for which the possible values of
                b should be found.
        reciprocal (int): Strictly positive integer giving the value
                of reciprocal in the above equation for which the
                possible values of b are to be found.
    
    Returns:
    List of strictly positive integers (int) giving all strictly
    positive b for which there exists strictly positive integer
    p for which the above equation holds for the given values
    of a and reciprocal. The list is sorted in increasing size
    of the identified values of b.
    """
    mult_min = (reciprocal) // a + 1
    b_max = (a * reciprocal) // (mult_min * a - reciprocal)
    b_step = a // gcd(a, reciprocal)
    b_min = b_step * (((a - 1) // b_step) + 1)
    #print(b_step, b_min, b_max)
    
    res = []
    for b in range(b_min, b_max + 1, b_step):
        if not reciprocal % addFractions((1, a), (1, b))[1]:
            res.append(b)
    return res

def reciprocalPairSumsEqualToMultipleOfReciprocal(reciprocal: int) -> List[Tuple[int]]:
    """
    Given the strictly positive integer reciprocal, identifies all
    pairs of strictly positive integers (a, b) such that b is no
    less than a and:
        1 / a + 1 / b = p / reciprocal
    for some strictly positive integer p.

    Args:
        Required positional:
        reciprocal (int): Strictly positive integer giving the value
                of reciprocal in the above equation for which the
                possible strictly positive integer pairs (a, b) are
                to be found.
    
    Returns:
    List of 2-tuples of strictly positive integers (int) giving all
    ordered pairs of strictly positive integers (a, b) such that b is
    no less than a and there exists strictly positive integer p for
    which the above equation holds for the given value of reciprocal.
    The list is sorted in increasing size of a, and pairs with the
    same value of a these are sorted in increasing size of b.
    """
    a_step = 1
    a_max = reciprocal * 2
    a_min = 1
    
    res = []
    for a in range(a_min, a_max + 1, a_step):
        for b in reciprocalPartnerSumsEqualToMultipleOfReciprocal(a, reciprocal):
            res.append((a, b))
    return res

def reciprocalPairSumsEqualToFraction(frac: Tuple[int]) -> List[Tuple[int]]:
    """
    For a strictly positive rational number frac, finds all ordered
    pairs of strictly positive integers (a, b) such that b is no
    less than a and:
        1 / a + 1 / b = frac
    
    Args:
        Required positional:
        frac (2-tuple of ints): The strictly positive rational number
                for which the ordered pairs (a, b) satisfying the
                above equation are to be sought, represented as a
                fraction with the strictly positive integers at
                indices 0 and 1 represent the numerator and denominator
                of the fraction respectively.

    Returns: 
    List of 2-tuples of strictly positive integers (int) giving all
    ordered pairs of strictly positive integers (a, b) such that b is
    no less than a and the above equation holds for the given value
    of frac. The list is sorted in increasing size of a.
    """
    g = gcd(*frac)
    frac = tuple(x // g for x in frac)

    if frac[0] > frac[1]:
        return [(1, frac[1])] if frac[0] == frac[1] + 1 else []
    res = []
    for q in range((frac[1] // frac[0]) + 1, (2 * frac[1] // frac[0]) + 1):
        frac2 = addFractions(frac, (-1, q))
        if frac2[0] == 1:
            res.append((q, frac2[1]))
    return res

def countReciprocalPairSumsEqualToFraction(frac: Tuple[int]) -> int:
    """
    For a strictly positive rational number frac, finds the number
    of distinct ordered pairs of strictly positive integers (a, b)
    that exist such that b is no less than a and:
        1 / a + 1 / b = frac
    
    Args:
        Required positional:
        frac (2-tuple of ints): The strictly positive rational number
                for which the number of ordered pairs (a, b) satisfying
                the above equation is to be sought, represented as a
                fraction with the strictly positive integers at
                indices 0 and 1 represent the numerator and denominator
                of the fraction respectively.

    Returns: 
    Integer (int) giving the number of distinct ordered pairs of
    strictly positive integers (a, b) that exists such tha b is no less
    than a and the above equation holds for the given value of frac.
    """
    g = gcd(*frac)
    frac = tuple(x // g for x in frac)

    if frac[0] > frac[1]:
        return int(frac[0] == frac[1] + 1)
    res = 0
    for q in range((frac[1] // frac[0]) + 1, (2 * frac[1] // frac[0]) + 1):
        frac2 = addFractions(frac, (-1, q))
        res += (frac2[0] == 1)
    return res

def reciprocalPairSumsMultipleOfReciprocal2(q_factorisation: Dict[int, int]) -> List[Tuple[int, int]]:
    """
    Given the prime factorisation of a strictly positive integer
    q_factorisation, identifies all distinct ordered pairs of
    strictly positive integers (a, b) such that b is no less than
    a and:
        1 / a + 1 / b = p / q
    for some strictly positive integer p, where q is the integer
    whose prime factorisation is q_factorisation.

    Args:
        Required positional:
        q_factorisation (dict): Dictionary representing a prime
                factorisation of the strictly positive integer q
                in the above equation, whose keys are the prime
                numbers that appear in the prime factorisation of
                q, with the corresponding value being the
                number of times that prime appears in the
                factorisation (i.e. the power of that prime in the
                prime factorisation of q). An empty dictionary
                corresponds to the multiplicative identity (i.e. 1).
    
    Returns:
    List of 2-tuples of strictly positive integers (int) giving all
    ordered pairs of strictly positive integers (a, b) such that b is
    no less than a and there exists strictly positive integer p for
    which the above equation holds for the positive integer q having
    the prime factorisation represented by q_factorisation.
    The list is sorted in increasing size of a, and pairs with the
    same value of a these are sorted in increasing size of b.
    """
    q = 1
    for k, v in q_factorisation.items():
        q *= k ** v
    #print(f"q = {q}")
    res = []
    for p in range(1, (5 * q) // 6 + 1):
        ans = reciprocalPairSumsEqualToFraction((p, q))
        #print(f"{p} / {q}: {ans}")
        res.extend(ans)
    res.append((2, 2))
    p_lst = list(q_factorisation.keys())
    def factorGenerator(idx: int, val: int) -> Generator[int, None, None]:
        if idx == len(p_lst):
            yield val
            return
        val2 = val
        for exp in range(q_factorisation[p_lst[idx]] + 1):
            yield from factorGenerator(idx + 1, val2)
            val2 *= p_lst[idx]
        return

    for factor in factorGenerator(0, 1):
        res.append((1, factor))
    #res += 1 + sum(v + 1 for v in q_factorisation.values())
    return res

def countReciprocalPairSumsMultipleOfReciprocal2(q_factorisation: Dict[int, int]) -> int:
    """
    Given the prime factorisation of a strictly positive integer
    q_factorisation, finds the number of distinct ordered pairs
    of strictly positive integers (a, b) such that b is no less
    than a and:
        1 / a + 1 / b = p / q
    for some strictly positive integer p, where q is the integer
    whose prime factorisation is q_factorisation.

    Args:
        Required positional:
        q_factorisation (dict): Dictionary representing a prime
                factorisation of the strictly positive integer q
                in the above equation, whose keys are the prime
                numbers that appear in the prime factorisation of
                q, with the corresponding value being the
                number of times that prime appears in the
                factorisation (i.e. the power of that prime in the
                prime factorisation of q). An empty dictionary
                corresponds to the multiplicative identity (i.e. 1).
    
    Returns:
    Integer (int) giving the number of distinct ordered pairs of
    strictly positive integers (a, b) that exists such tha b is no less
    than a and the above equation holds for some strictly positive
    integer p with the positive integer q having the prime factorisation
    represented by q_factorisation.
    """
    q = 1
    for k, v in q_factorisation.items():
        q *= k ** v
    #print(f"q = {q}")
    res = 0
    for p in range(1, (5 * q) // 6 + 1):
        ans = reciprocalPairSumsEqualToFraction((p, q))
        #print(f"{p} / {q}: {ans}")
        res += len(ans)
    print(res)
    term = 1
    for v in q_factorisation.values():
        term *= v + 1
    #res += 1 + sum(v + 1 for v in q_factorisation.values())
    return res + term + 1

def factorFactorisationsGenerator(num_p_factorisation: Dict[int, int]) -> Generator[Dict[int, int], None, None]:
    """
    Generator yielding the prime factorisations of each
    distinct positive integer factor of the strictly positive
    integer with the prime factorisation num_p_factorisation,
    with each factor being represented by exactly one yielded.
    value.

    Args:
        Required positional:
        num_p_factorisation (dict): Dictionary representing a prime
                factorisation of the strictly positive integer for
                which the prime factorisations of its factors are
                to be generated, whose keys are the prime numbers
                that appear in the prime factorisation of the integer
                in question, with the corresponding value being the
                number of times that prime appears in the factorisation
                (i.e. the power of that prime in the prime
                factorisation of the integer in question). An empty
                dictionary corresponds to the multiplicative identity
                (i.e. 1).
    
    Yields:
    Dictionary (dict) giving the prime factorisation of a positive
    integer factor of the positive integer with prime factorisation
    num_p_factorisation, whose keys are strictly positive integers
    (int) giving the prime numbers that appear in the prime
    factorisation of the factor in question, with the corresponding
    value being a strictly positive integer (int) giving the number
    of times that prime appears in the factorisation (i.e. the power
    of that prime in the prime factorisation of the factor in
    question). An empty dictionary corresponds to the multiplicative
    identity (i.e. 1).
    The prime factorisation of each distinct positive integer factor
    is yielded exactly once. As such, the number of values yielded
    is equal to the number of positive integer factors of the integer
    (for instance, for prime numbers, exactly two factorisations
    are yielded, represenging to the multiplicative identity and
    the number itself).
    Note that while generally later values yielded tend to represent
    larger integers than those of earlier values yielded (with the
    multiplicative identity factorisation guaranteed to be yielded
    first and the original factorisation last), it is possible for
    factorisations of integers to be yielded after factorisations
    of larger integers, and as such, this ordering should not be
    relied on.
    """
    p_lst = list(num_p_factorisation.keys())
    curr = {}
    def recur(idx: int) -> Generator[int, None, None]:
        if idx == len(p_lst):
            yield dict(curr)
            return
        p = p_lst[idx]
        for exp in range(num_p_factorisation[p] + 1):
            yield from recur(idx + 1)
            curr[p] = curr.get(p, 0) + 1
        if p in curr.keys(): curr.pop(p)
        return
    
    yield from recur(0)
    return

def calculatePrimeFactorisation(num: int) -> Dict[int, int]:
    """
    For a strictly positive integer, calculates its prime
    factorisation.

    This is performed using direct division.

    Args:
        Required positional:
        num (int): The strictly positive integer whose prime
                factorisation is to be calculated.
    
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

def countReciprocalPairSumsMultipleOfReciprocal(
    q_factorisation: Dict[int, int],
) -> int:
    """
    Given the prime factorisation of a strictly positive integer
    q_factorisation, finds the number of distinct ordered pairs
    of strictly positive integers (a, b) such that b is no less
    than a and:
        1 / a + 1 / b = p / q
    for some strictly positive integer p, where q is the integer
    whose prime factorisation is q_factorisation.

    Args:
        Required positional:
        q_factorisation (dict): Dictionary representing a prime
                factorisation of the strictly positive integer q
                in the above equation, whose keys are the prime
                numbers that appear in the prime factorisation of
                q, with the corresponding value being the
                number of times that prime appears in the
                factorisation (i.e. the power of that prime in the
                prime factorisation of q). An empty dictionary
                corresponds to the multiplicative identity (i.e. 1).
    
    Returns:
    Integer (int) giving the number of distinct ordered pairs of
    strictly positive integers (a, b) that exists such tha b is no less
    than a and the above equation holds for some strictly positive
    integer p with the positive integer q having the prime factorisation
    represented by q_factorisation.
    """
    q = 1
    for k, v in q_factorisation.items():
        q *= k ** v
    res = 0
    #print(f"q = {q}")
    for d_fact in factorFactorisationsGenerator({k: 2 * v for k, v in q_factorisation.items()}):
        d = 1
        for k, v in d_fact.items():
            d *= k ** v
        if d > q: continue
        #print(d)
        #d_inv_fact = {k: 2 * v - d_fact.get(k, 0) for k, v in q_factorisation.items()}
        p_mult_fact = {k: min(v, 2 * q_factorisation[k] - v) for k, v in d_fact.items()}
        q_div = 1
        for k, v in q_factorisation.items(): q_div *= k ** (v - p_mult_fact.get(k, 0))

        numer1 = 1
        numer2 = 1
        for k, v in q_factorisation.items():
            numer1 *= k ** (d_fact.get(k, 0) - p_mult_fact.get(k, 0))
            numer2 *= k ** (2 * v - d_fact.get(k, 0) - p_mult_fact.get(k, 0))
        numer1 += q_div
        numer2 += q_div
        g = gcd(numer1, numer2)
        #print(f"d = {d}, g = {g}")
        #pf = ps.factorCount(g)
        #for p_fact in factorFactorisationsGenerator(p_mult_fact):
        #    p = 1
        #    for k, v in p_fact.items(): p *= k ** v
        #    print(p, d, ((q + d) // p, (q + q ** 2 // d) // p), ((q - d) // p, (q - q ** 2 // d) // p))
        pf = calculatePrimeFactorisation(g)
        for k, v in pf.items():
            p_mult_fact[k] = p_mult_fact.get(k, 0) + v
        ans = 1
        for v in p_mult_fact.values():
            ans *= v + 1
        res += ans
    print(f"result for q = {q}: {res}")
    return res


def countReciprocalPairSumsMultipleOfReciprocalPower2(
    reciprocal_factorisation: Dict[int, int]={2: 1, 5: 1},
    min_power: int=1,
    max_power: int=9,
) -> int:
    """
    Given the prime factorisation of a strictly positive integer
    reciprocal_factorisation, finds the number of distinct ordered
    triples (a, b, n) such that a and b are strictly positive
    integers, b is no less than a, n is a non-negative integer
    between min_power and max_power inclusive and:
        1 / a + 1 / b = p / q ^ n
    for some strictly positive integer p, where q is the integer
    whose prime factorisation is reciprocal_factorisation.

    Args:
        Optional named:
        reciprocal_factorisation (dict): Dictionary representing a
                prime factorisation of the strictly positive integer
                q in the above equation, whose keys are the prime
                numbers that appear in the prime factorisation of
                q, with the corresponding value being the
                number of times that prime appears in the
                factorisation (i.e. the power of that prime in the
                prime factorisation of q). An empty dictionary
                corresponds to the multiplicative identity (i.e. 1).
            Default: {2: 1, 5: 1} (the prime factorisation of 10)
        min_power (int): Non-negative integer giving the smallest
                value of n considered for counted solutions.
            Default: 1
        max_power (int): Non-negative integer giving the largest
                value of n considered for counted solutions.
            Default: 9
    
    Returns:
    Integer (int) giving the number of distinct ordered triples
    (a, b, n) such that a and b are strictly positive integers,
    b is no less than a, n is a non-negative integer between
    min_power and max_power inclusive and the above equation holds
    for some strictly positive integer p with the positive integer
    q having the prime factorisation represented by
    reciprocal_factorisation.
    """
    since = time.time()
    b = 1
    b2 = 1
    for k, v in reciprocal_factorisation.items():
        b *= k ** (v * max_power)
        b2 *= k ** v
    #print(f"b = {b}, b2 = {b2}")
    res = 0
    for a in range(1, (5 * b) // 6 + 1):
        if not a % 1000:
            print(f"a = {a}, b = {b}")
        g = gcd(a, b)
        (a_, b_) = (a // g, b // g)
        ans = countReciprocalPairSumsEqualToFraction((a_, b_))
        if not ans: continue
        mx = (max_power - min_power)
        
        for p, v in reciprocal_factorisation.items():
            curr = g
            num = p ** v
            for i in range(mx + 1):
                curr, r = divmod(curr, num)
                if r: break
            else: continue
            mx = i
        #print(f"{p} / {q}: {ans}")
        res += ans * (mx + 1)
    #print(res)
    res += (max_power - min_power + 1)
    p_lst = list(reciprocal_factorisation.keys())
    def factorGenerator(idx: int, val: int) -> Generator[int, None, None]:
        if idx == len(p_lst):
            yield val
            return
        val2 = val
        for exp in range(reciprocal_factorisation[p_lst[idx]] * max_power + 1):
            yield from factorGenerator(idx + 1, val2)
            val2 *= p_lst[idx]
        return

    for factor in factorGenerator(0, 1):
        curr = factor
        r = 0
        ans = 0
        while not r and ans < (max_power - min_power + 1):
            ans += 1
            curr, r = divmod(curr, b2)
        res += ans
    #res += 1 + sum(v + 1 for v in q_factorisation.values())
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

def countReciprocalPairSumsMultipleOfReciprocalPower(reciprocal_factorisation: Dict[int, int]={2: 1, 5: 1}, min_power: int=1, max_power: int=9) -> int:
    """
    Solution to Project Euler #157

    Given the prime factorisation of a strictly positive integer
    reciprocal_factorisation, finds the number of distinct ordered
    triples (a, b, n) such that a and b are strictly positive
    integers, b is no less than a, n is a non-negative integer
    between min_power and max_power inclusive and:
        1 / a + 1 / b = p / q ^ n
    for some strictly positive integer p, where q is the integer
    whose prime factorisation is reciprocal_factorisation.

    Args:
        Optional named:
        reciprocal_factorisation (dict): Dictionary representing a
                prime factorisation of the strictly positive integer
                q in the above equation, whose keys are the prime
                numbers that appear in the prime factorisation of
                q, with the corresponding value being the
                number of times that prime appears in the
                factorisation (i.e. the power of that prime in the
                prime factorisation of q). An empty dictionary
                corresponds to the multiplicative identity (i.e. 1).
            Default: {2: 1, 5: 1} (the prime factorisation of 10)
        min_power (int): Non-negative integer giving the smallest
                value of n considered for counted solutions.
            Default: 1
        max_power (int): Non-negative integer giving the largest
                value of n considered for counted solutions.
            Default: 9
    
    Returns:
    Integer (int) giving the number of distinct ordered triples
    (a, b, n) such that a and b are strictly positive integers,
    b is no less than a, n is a non-negative integer between
    min_power and max_power inclusive and the above equation holds
    for some strictly positive integer p with the positive integer
    q having the prime factorisation represented by
    reciprocal_factorisation.
    """
    since = time.time()
    res = 0
    for exp in range(min_power, max_power + 1):
        q_factorisation = {k: v * exp for k, v in reciprocal_factorisation.items()}
        res += countReciprocalPairSumsMultipleOfReciprocal(q_factorisation)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 158
def countAllDifferentLetterStringsWithNSmallerLeftNeighbours(n_chars: int, max_len: int, n_smaller_left_neighbours: int) -> List[int]:
    """
    For strings constructed from an ordered alphabet consisting
    of n_chars characters such that no character is used more
    than once, calculates the number of such strings that can be
    formed where exactly n_smaller_left_neighbours of the letters
    in the string have the letter directly to its left be a
    character that appears earlier in the alphabet, for all string
    lengths not exceeding max_len.

    Args:
        Required positional:
        n_chars (int): The number of characters in the alphabet
                being considered.
        max_len (int): The largest length string considered.
                Should be between 1 and n_chars inclusive.
        n_smaller_left_neighbours (int): The exact number of
                letters in the strings counted for which the
                letter to its left in the string is a character
                that appears earlier in the alphabet that that
                of the letter in question.
    
    Returns:
    List of integers (ints) with length max_len + 1 where for
    non-negative integer i <= max_len, the i:th element (0-indexed)
    of the list gives the count of strings satisfying the conditions
    for strings with length i.
    """
    # TODO- try to derive the exact formula for n_smaller_left_neighbours = 1:
    #  (2^n - n - 1) * (26 choose n)
    # and generalise to any n_smaller_left_neighbours- look into Eulerian
    # numbers
    res = [0, 0] if n_smaller_left_neighbours else [1, n_chars]
    curr = [[1] * n_chars]
    for length in range(2, max_len + 1):
        prev = curr
        curr = []
        curr.append([0] * (n_chars - length + 1))
        curr[0][n_chars - length] = prev[0][n_chars - length + 1]
        for i in reversed(range(n_chars - length)):
            curr[0][i] = curr[0][i + 1] + prev[0][i + 1]
        for j in range(1, min(n_smaller_left_neighbours, len(prev)) + 1):
            curr.append([0] * (n_chars - length + 1))
            if j < len(prev):
                curr[j][n_chars - length] = prev[j][n_chars - length + 1]
                for i in reversed(range(n_chars - length)):
                    curr[j][i] = curr[j][i + 1] + prev[j][i + 1]
            # Transition from the previous with one fewer smaller neighbours
            cumu = 0
            for i in range(n_chars - length + 1):
                cumu += prev[j - 1][i]
                curr[j][i] += cumu
        #print(curr)
        res.append(sum(curr[-1]))
    return res

def maximumDifferentLetterStringsWithNSmallerLeftNeighbours(n_chars: int=26, max_len: int=26, n_smaller_left_neighbours: int=1) -> List[int]:
    """
    Solution to Project Euler #158

    For strings constructed from an ordered alphabet consisting
    of n_chars characters such that no character is used more
    than once, calculates the largest number of such strings of a 
    given length not exceeding max_len that can be formed where
    exactly n_smaller_left_neighbours of the letters in the string
    have the letter directly to its left be a character that
    appears earlier in the alphabet, for all string.

    Args:
        Optional named:
        n_chars (int): The number of characters in the alphabet
                being considered.
            Default: 26
        max_len (int): The largest length string considered.
                Should be between 1 and n_chars inclusive.
            Default: 26
        n_smaller_left_neighbours (int): The exact number of
                letters in the strings counted for which the
                letter to its left in the string is a character
                that appears earlier in the alphabet that that
                of the letter in question.
            Default: 1
    
    Returns:
    Integers (int) giving the largest count of strings of a given
    length no greater than max_len satisfying the specified
    conditions.
    """
    since = time.time()
    res = max(countAllDifferentLetterStringsWithNSmallerLeftNeighbours(n_chars, max_len, n_smaller_left_neighbours))
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 159
def digitalRoot(num: int, base: int=10) -> int:
    """
    For a non-negative integer num, finds the digital
    root of that integer in a chosen base.

    The digital root of a non-negative integer in a
    given base is defined recursively as follows:
     1) If the integer is strictly less than base, then
        it is equal to the value of that integer.
     2) Otherwise, it is equal to the digital root of the
        integer calculated by summing the value of the
        digits of the representation of the intger in the
        given base.
    For any base greater than 1, this definition uniquely
    defines the digital root, given that for any integer
    no less than base, the sum of its digits in the chosen
    base has an integer value no less than 1 and strictly
    less than the value of the integer. This guarantees
    that step 2 will only need to be applied a finite number
    of times before reaching a number between 1 and
    (base - 1) inclusive for any integer (with an upper
    bound on the number of applications being the value of
    the integer itself).

    Args:
        Required positional:
        num (int): The strictly positive integer whose
                digital root is to be calculated.
        
        Optional named:
        base (int): Integer strictly greater than 1 giving the
                base in which the integers are to be represented
                when finding the sum of digits.
            Default: 10
    
    Returns:
    Integer (int) between 1 and (base - 1) inclusive giving
    the digital root of num in the chosen base, as defined
    above.
    """
    #num0 = num
    while num >= base:
        num2 = 0
        while num:
            num, r = divmod(num, base)
            num2 += r
        num = num2
    #print(num0, num)
    return num

def maximalDigitalRootFactorisations(n_max: int, base: int=10) -> List[int]:
    """
    Calculates the maximal digital root sum in the chosen
    base for every non-negative integer no greater than
    n_max.

    The digital root of a strictly positive integer in a
    chosen base is defined recursively as follows:
     1) If the integer is strictly less than base, then
        it is equal to the value of that integer.
     2) Otherwise, it is equal to the digital root of the
        integer calculated by summing the value of the
        digits of the representation of the intger in the
        chosen base.
    For any base greater than 1, this definition uniquely
    defines the digital root, given that for any integer
    no less than base, the sum of its digits in the chosen
    base has an integer value no less than 1 and strictly
    less than the value of the integer. This guarantees
    that step 2 will only need to be applied a finite number
    of times before reaching a number between 1 and
    (base - 1) inclusive for any integer (with an upper
    bound on the number of applications being the value of
    the integer itself).

    The digital root sum of a postive integer factorisation is
    the sum of the digital roots of the terms in the
    factorisation, where repeated factors are included in the
    sum as many times as they appear in the factorisation.

    The maximal digital root sum of a non-negative integer is
    defined to be 0 for the integers 0 and 1 and the largest
    digital root sum of its possible integer factorisations
    in which 1 does not appear as a factor.

    Args:
        Required positional:
        n_max (int): The largest non-negative integer for
                which the maximal digit root sum is to be
                calculated.
        
        Optional named:
        base (int): Integer strictly greater than 1 giving the
                base in which the integers are to be represented
                when finding the digital roots.
            Default: 10
    
    Returns:
    List of integers (int) with length (n_max + 1), for which
    the i:th index (using 0-indexing) gives the maximal digit
    root sum of the integer i in the chosen base.
    
    Outline of rationale:
    For any integer strictly greater than 1, all possible
    positive integer factorisations with no 1 factors except
    for the factorisation consisting of the integer itself
    (with no other factors and the integer appearing only once)
    include an integer between 1 and the original integer
    exclusive. For any of the latter factorisations, consider
    the result obtained by partitioning the factors into two
    non-empty partitions. The result is two factorisations of 
    positive integers greater than 1 and strictly less than the
    original integer.
    Suppose the original factorisation gives rise to a maximal
    digital root sum for the original integer. By the definition
    of the digital root sum of a positive integer factorisation,
    the value of this maximal digital root sum is equal to the
    sum of the digital root sums of the two new factorisations.
    Suppose the digital root sum of one of these new factorisations
    is not a maximal digital root sum for the integer for which it
    is a factorisation (which, as previously observed, is strictly
    smaller than the original integer). If we now consider a
    positive integer factorisation of this integer with no 1 factors
    whose digital root sum is a maximal digital root sum for that
    integer, its digital root sum is by definition larger than that
    of the previous factorisation. Re-inserting the integers in the
    other partition into this factorisation, we obtain a new
    factorisation of the original integer, whose digital root
    sum is equal to that digital root sum plus the digital root sum
    of the factorisation in the other partition, giving a digital
    root sum strictly greater than the original factorisation of the
    original integer, which we supposed was the maximal digital root
    sum of the integer. Given that this new factorisation contains
    only positive integers and no 1s, by the definition of the
    maximal digital root sum of the integer its digital root sum
    cannot exceed that of the maximal digital root sum, so we have a
    contradiction.
    Consequently, the maximal digital root sum of an integer strictly
    greater than 1 is either the digital sum of the integer itself
    or the sum of the maximal digit sum of two smaller integers
    whose product is equal to that integer.
    Thus, once all the maximal digital root sums of smaller integers
    strictly greater than 1 are known, we can calculate the maximal
    digit root sum of an integer by taking the maximum values of
    the digit sum of the integer itself and for each non-unit factor
    pair (i.e. pairs of integers strictly greater than 1 whose product
    is the integer in question), the sum of the two factors' maximal
    digital root sums. This enables iterative calculation of all the
    maximal digital root sums up to an arbitrarily large value.
    We further optimise this process by using a sieve approach
    (similar to the sieve of Eratosthenes in prime searching)
    to circumvent the need to find factors (which would involve the
    expensive integer division operation), whereby after each maximal
    digital root sum is calculated, the multiples up to the maximum
    value or the square of the current integer (whichever is smaller)
    are updated to take the maximum value of their current value
    and the sum of the maximal digital root sum just calculated and
    that of the integer being multiplied by (which, given that it
    is no larger than the integer being considered will have already
    been calculated). Thus, when reaching a new integer, all that
    needs to be calculated to find the maximal digital root sum is
    the digital sum of that integer, to check whether that exceeds
    the existing maximum value.
    TODO- revise wording of rationale for clarity
    """
    res = [0] * (n_max + 1)
    for n in range(2, n_max + 1):
        res[n] = max(res[n], digitalRoot(n, base=base))
        for n2 in range(2, min(n, n_max // n) + 1):
            m = n2 * n
            res[m] = max(res[m], res[n] + res[n2])
    return res

def maximalDigitalRootFactorisationsSum(n_min: int=2, n_max: int=10 ** 6 - 1, base: int=10) -> int:
    """
    Solution to Project Euler #159

    Calculates the sum of the maximal digital root sums in
    the chosen base over every non-negative integer between
    n_min and n_max inclusive.

    The digital root of a strictly positive integer in a
    chosen base is defined recursively as follows:
     1) If the integer is strictly less than base, then
        it is equal to the value of that integer.
     2) Otherwise, it is equal to the digital root of the
        integer calculated by summing the value of the
        digits of the representation of the intger in the
        chosen base.
    For any base greater than 1, this definition uniquely
    defines the digital root, given that for any integer
    no less than base, the sum of its digits in the chosen
    base has an integer value no less than 1 and strictly
    less than the value of the integer. This guarantees
    that step 2 will only need to be applied a finite number
    of times before reaching a number between 1 and
    (base - 1) inclusive for any integer (with an upper
    bound on the number of applications being the value of
    the integer itself).

    The digital root sum of a postive integer factorisation is
    the sum of the digital roots of the terms in the
    factorisation, where repeated factors are included in the
    sum as many times as they appear in the factorisation.

    The maximal digital root sum of a non-negative integer is
    defined to be 0 for the integers 0 and 1 and the largest
    digital root sum of its possible integer factorisations
    in which 1 does not appear as a factor.

    Args:
        Optional named:
        n_min (int): The smallest strictly positive integer for
                which the maximal digit root sum is to be
                included in the sum.
            Default: 2
        n_max (int): The largest strictly positive integer for
                which the maximal digit root sum is to be
                included in the sum.
            Default: 10 ** 6 - 1
        base (int): Integer strictly greater than 1 giving the
                base in which the integers are to be represented
                when finding the digital roots.
            Default: 10
    
    Returns:
    Integer (int) giving the sum of the maximal digital root sums
    in the chosen base over every non-negative integer between
    n_min and n_max inclusive.

    Outline of rationale:
    See documentation for maximalDigitalRootFactorisations().
    """
    since = time.time()
    arr = maximalDigitalRootFactorisations(n_max, base=10)
    #print(arr)
    res = sum(arr[n_min:])
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 160
def factorialPrimeFactorCount(n: int, p: int) -> int:
    """
    For a strictly positive integer n and a prime number p, calculates
    the number of times p occurs in the prime factorisation of n!
    (n factorial).

    Args:
        Required positional
        n (int): Strictly positive integer whose factorial is being
                assesed for the number of times p occurs in its
                prime factorisation.
        p (int): Strictly positive integer giving the prime number
                whose number of occurrences in n! is to be calculated.
    
    Returns:
    Integer (int) giving the number of times the prime p occurs in
    the prime factorisation of n!.
    """
    res = 0
    while n:
        n //= p
        res += n
    return res

def factorialFinalDigitsBeforeTrailingZeros(n: int=10 ** 12, n_digs: int=5, base: int=10) -> int:
    """
    Solution to Project Euler #160

    Calculates the value of the final n_digs digits of n! before the
    trailing zeros when expressed in the chosen base, giving this
    as the value of these digits in the same order when interpreted
    in the chosen base.

    Args:
        Optional named:
        n (int): Strictly positive integer for which the final n_digs
                digits before the trailing zeros of its factorial when
                represented in the chosen base are to be found.
            Default: 10 ** 12
        n_digs (int): Strictly positive integer giving the number of
                digits before the trailing zeros of n! expressed in
                the chosen base are to be found.
            Default: 5
        base (int): Integer strictly greater than 1 giving the base
                in n! to be represented when finding its last n_digs
                digits before the trailing zeros of n!.
            Default: 10
    
    Returns:
    Integer (int) giving the value when interpreted in the chosen
    base of the last n_digs digits before the trailing zeros of
    n! when expressed in the chosen base.

    Outline of rationale:
    We are essentially looking for:
        n! / (base ** n_zeros) (mod base ** n_digs)
    where n_zeros is the number of trailing zeros in the prime
    factorisation of n! when represented in the chosen base.
    The prime factorisation of the base is first calculated. The
    primes in this factorisation account for the trailing zeros
    and so are handled separately. We can calculate the number
    of times these occur in the prime factorisation of n! using
    the function factorialPrimeFactorCount(). The number of
    trailing zeros is then the minimum value among all of those
    prime factors of the floor of the number of times it occurs
    in the prime factorisation of n! divided by the number of
    times it occurs in the prime factorisation of base. For each
    prime, the remainder of occurrences in the prime factorisation
    of n! not accounted for by the trailing zeros can then be used
    to contribute to the result by taking the product of the powers
    of each prime to its remainder occurrence count, all modulo
    (base ** n_digs).
    All that remains is to calculate the contribution of all integers
    between 1 and n inclusive, where for each integer all prime
    factors of base are removed (i.e. for the given integer and each
    of those primes successively, the integer is repeatedly divided by
    the prime until it is no longer divisible by that prime), all
    modulo (base ** n_digs). These are then all multiplied together
    with the value for the contribution of those primes found previously
    and taken modulo (base ** n_digs) to give the final answer.
    
    To calculate the overall result of doing this for all integers
    from 1 to n inclusive, we work backwards from each integer from
    1 to (base ** n_digs - 1) inclusive that do not divide any of the
    prime factors of base and work out for how many of the integers
    from 1 to n inclusive give a contribution which modulo
    (base ** n_digs) is equal to that number. The result is then the
    product of these numbers, each to the power of the count of integers
    for which it represents the contribution, again all modulo
    (base ** n_digs).
    There are two interlinked considerations when doing this. The first
    is that each number represents not only itself but all of the
    positive integers equal to it modulo (base ** n_digs) that are
    no greater than n (which we refer to as the number's associates).
    The second is that each number and its associates also represent
    the contribution of all of the integers no greater than n found by
    multiplying the number or associate by a product of powers of the
    prime factors of base.
    TODO
    """
    since = time.time()
    base_pf = calculatePrimeFactorisation(base)
    #print(base_pf)
    fact_cnts = {}
    md = base ** n_digs
        
    n_zeros = float("inf")
    for p, cnt1 in base_pf.items():
        cnt2 = factorialPrimeFactorCount(n, p)
        #print(p, cnt2)
        fact_cnts[p] = cnt2
        n_zeros = min(n_zeros, cnt2 // cnt1)
    res = 1
    #print(f"n zeros = {n_zeros}")
    for p, cnt1 in base_pf.items():
        num = fact_cnts[p] - cnt1 * n_zeros
        if num: res *= pow(p, fact_cnts[p] - cnt1 * n_zeros, md)
        #print(p, num)
    #print(res)
    base_p_prods = [1]
    for p in base_pf.keys():
        i = 0
        while i < len(base_p_prods):
            num = base_p_prods[i] * p
            if num <= n:
                base_p_prods.append(num)
            i += 1
        base_p_prods.sort()
    #base_p_prods = list(base_p_prods)
    #print(base_p_prods)
    avoid_heap = list((p, p) for p in base_pf.keys())
    heapq.heapify(avoid_heap)
    for i in range(1, md):
        if i == avoid_heap[0][0]:
            while i == avoid_heap[0][0]:
                tup = avoid_heap[0]
                heapq.heappushpop(avoid_heap, (tup[0] + tup[1], tup[1]))
            continue
        j = 0
        exp = 0
        k = 0
        for j in reversed(range(len(base_p_prods))):
            k2 = ((n // base_p_prods[j]) - i) // md
            
            exp += (j + 1) * (k2 - k)
            k = k2
            
            #print(f"k = {k}")
        res = (res * pow(i, exp, md)) % md
    #print(res)
    #print(math.factorial(n))
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 161
def nextTriominoStates(state: List[Tuple[int, bool]], rows_remain: int=-1, nxt_insert: Optional[int]=None) -> List[Tuple[List[Tuple[int, bool]], int, int]]:
    """
    For filling a series of rows of fixed width with triominos
    without gaps, finds the possible states of the rows after
    the insertions of the possible triominos that fills the
    leftmost empty square on the first row with unfilled squares
    (such that no unfillable gaps are left).

    For rows of width n units (where n is an integer), the state
    of the rows is described by a length n list of 2-tuples, with index
    i (for non-negative integer i < n) representing the i:th column
    from the left (starting at 0 for the leftmost column), where
    index 0 of the 2-tuple contains an integer giving the number
    of consecutive squares filled in that column starting from the
    first row with unfilled squares, and index 1 contains a boolean
    representing whether for this column, there exists a filled
    square after a non-filled square in the previous row (described
    as an overhang).

    Args:
        Required positional:
        state (list of 2-tuples of an ordered int bool pair): The
                initial state from which the possible states reached
                by inserting a triomino are to be calculated (see
                above for the representation of a state).
        
        Optional named:
        rows_remain (int): Integer which, if non-negative, gives the
                number of rows that remain to be filled (starting with
                the row labelled 0 in state). If negative then there is
                no limit on the number of rows to fill.
            Default: -1
        nxt_insert (int or None): If specified as an integer, gives the
                index of the leftmost column for which the square in the
                first row with any unfilled quares is unfilled, (assumed
                to be correct and have been calculated previously, to
                reduce repeated calculation). If given as None, this
                will be directly calculated from state.
            Default: None

    Returns:
    List of 3-tuples, representing the possible states that can be reached
    from the initial state by inserting a triomino in such a way that the
    leftmost unfilled square of the first row with unfilled squares of the
    original state is filled and no gaps are left that are unfillable by
    triominos.
    Each 3-tuple representing such a state contains in index 0 a list of
    2-tuples of an ordered int bool pair representing the state reached
    (see above), in index 1 the number of rows the row labelled 0 is displaced
    relative to that of the initial state (note that this shift is calculated
    such that the row labelled 0 is the first row with any unfilled squares),
    and in index 2 the index of the leftmost column for which the square in
    the row labelled 0 (i.e. the first row with any unfilled squares) is
    unfilled.
    """
    # TODO- review documentation for clarity
    # Boolean represents whether there is an overhang caused by
    # an L-piece
    if rows_remain < 0:
        rows_remain = float("inf")
    n = len(state)
    lvl_delta0 = 0
    if nxt_insert is None or state[nxt_insert][0]:
        mn_state = min(x[0] for x in state)
        if mn_state:
            lvl_delta0 = mn_state
            state = [(x[0] - mn_state, x[1]) for x in state]
        for i in range(n):
            if not state[i][0]:
                nxt_insert = i
                break
    
    res = []
    def addState(state2: List[Tuple[int, bool]], piece_mn: Tuple[int, int], piece_rng: Tuple[int, int]) -> None:
        
        mn = piece_mn
        for i in range(piece_rng[1] + 1, n):
            mn = min(mn, (state2[i][0], i))
            if not mn: break
        if mn[0] >= 1:
            for i in range(piece_rng[0]):
                mn = min(mn, (state2[i][0], i))
                if state2[i][0] <= 1: break
        state2 = tuple((x[0] - mn[0], x[1]) for x in state2)
        #if state2 == ((1, False), (1, False), (0, False), (0, False), (0, False), (1, False)):
        #    print(piece_rng, piece_mn, mn)
        res.append((state2, lvl_delta0 + mn[0], mn[1]))
        return
    
    # Pieces that can fill in an overhang space
    #if state == ((1, False), (1, False), (0, False), (0, False), (0, False), (1, False)):
    #    print("hello", nxt_insert)
    # Flat line piece
    if rows_remain >= 1 and nxt_insert + 2 < n and not state[nxt_insert + 1][0] and not state[nxt_insert + 2][0] and\
            not (nxt_insert + 3 < n and state[nxt_insert + 3] == (0, True)):
        state2 = list(state)
        mn = (float("inf"), 0)
        for i in range(nxt_insert, nxt_insert + 3):
            state2[i] = (1 + state[i][1], False)
            mn = min(mn, (state2[i][0], i))
        addState(state2, mn, (nxt_insert, nxt_insert + 2))
        #print(state)
        """
        for i in range(nxt_insert + 3, n):
            mn = min(mn, state2[i][0])
            if not mn: break
        if mn > 1:
            for i in range(nxt_insert):
                mn = min(mn, state2[i][0])
                if mn <= 1: break
        if mn: state2 = [(x[0] - mn, x[1]) for x in state2]
        res.append((state2, lvl_delta0 + mn))
        """
    if rows_remain <= 1: return res

    # Backward R-piece
    if nxt_insert + 1 < n and not state[nxt_insert + 1][0] and not state[nxt_insert + 1][1] and\
            not (nxt_insert + 2 < n and state[nxt_insert + 2] == (0, True)):
        state2 = list(state)
        state2[nxt_insert] = (1 + state[nxt_insert][1], False)
        state2[nxt_insert + 1] = (2, False)
        mn = min((state2[nxt_insert][0], nxt_insert), (2, nxt_insert + 1))
        addState(state2, mn, (nxt_insert, nxt_insert + 1))

    if state[nxt_insert][1]: return res

    # The other pieces

    # Upright line piece
    if rows_remain >= 3 and not (nxt_insert + 1 < n and state[nxt_insert + 1] == (0, True)):
        state2 = list(state)
        state2[nxt_insert] = (3, False)
        mn = (3, nxt_insert)
        addState(state2, mn, (nxt_insert, nxt_insert))
    
    # R piece
    if nxt_insert + 1 < n and not state[nxt_insert + 1][0] and\
            not (nxt_insert + 2 < n and state[nxt_insert + 2] == (0, True)):
        state2 = list(state)
        state2[nxt_insert] = (2, False)
        state2[nxt_insert + 1] = (1 + state[nxt_insert + 1][1], False)
        mn = min((2, nxt_insert), (state2[nxt_insert + 1][0], nxt_insert + 1))
        addState(state2, mn, (nxt_insert, nxt_insert + 1))
    
    # L piece
    if nxt_insert + 1 < n and (state[nxt_insert + 1][0] <= 1 and state[nxt_insert + 1] != (0, True)) and\
            not (nxt_insert + 2 < n and state[nxt_insert + 2] == (0, True)) and\
            (state[nxt_insert + 1][0] == 1 or (nxt_insert + 2 < n and state[nxt_insert + 2] == (0, False))):
        state2 = list(state)
        state2[nxt_insert] = (2, False)
        state2[nxt_insert + 1] = (2, False) if state[nxt_insert + 1][0] == 1 else (0, True)
        mn = min((2, nxt_insert), (state2[nxt_insert + 1][0], nxt_insert + 1))
        addState(state2, mn, (nxt_insert, nxt_insert + 1))
    
    # Backwards L piece
    if nxt_insert > 0 and (state[nxt_insert - 1][0] == 1):
        state2 = list(state)
        state2[nxt_insert - 1] = (2, False)
        state2[nxt_insert] = (2, False)
        mn = (2, nxt_insert - 1)
        addState(state2, mn, (nxt_insert - 1, nxt_insert))
    
    return res
"""
def triominoStateComplementRotated(state: Tuple[Tuple[int, bool]]) -> Tuple[Tuple[Tuple[int, bool]], int]:
    # Assumes state1 has at least one entry at level 0
    h = max(x[0] for x in state)
    state2 = []
    for x, b in state:
        if not b: state2.append((h - x, False))
        else: state2.append((h - x - 2, True))
    return (tuple(state2[::-1]), h)
"""
def triominoAreaFillCombinations(grid_width: int=9, grid_height: int=12) -> int:
    """
    Solution to Project Euler #161

    Calculates the number of distinct ways a rectangular grid with dimensions
    n_rows x n_cols can be completely filled by non-overlapping triominos
    constructed from squares of unit length.

    A triomino is a shape consisting of exactly three squares, all with the
    same dimensions, connected edge to edge such that it is possible traverse
    from any of the three squares to any other via these edge to edge
    connections.

    Args:
        Optional named:
        grid_width (int): Strictly positive integer giving the width of the
                rectangular grid in terms of the dimensions of the squares
                from which the triominos are constructed.
            Default: 9
        grid_height (int): Strictly positive integer giving the height of the
                rectangular grid in terms of the dimensions of the squares
                from which the triominos are constructed.
            Default: 12
    
    Returns:
    Integer (int) giving the number of distinct ways a rectangular grid with
    dimensions width x height can be completely filled with non-overlapping
    triominos constructed from squares of unit length.

    Outline of rationale:
    TODO
    """
    since = time.time()
    n_triominos, r = divmod(grid_width * grid_height, 3)
    if r: return 0
    #print(n_triominos)
    n_cols, n_rows = sorted([grid_width, grid_height])
    #n_crow, n_rows = sorted([n_rows, n_cols])
    states_counts = {(tuple((0, False) for _ in range(n_cols)), 0, 0): 1}
    
    for _ in range(n_triominos):#(n_triominos) >> 1):
        new_states_counts = {}
        for (state, lvl, nxt_insert), cnt in states_counts.items():
            if lvl > n_rows - 3 and lvl + max(x[0] + 2 * x[1] for x in state) > n_rows:
                continue
            for (state2, lvl_delta, nxt_insert2) in nextTriominoStates(state, n_rows - lvl, nxt_insert=nxt_insert):
                lvl2 = lvl + lvl_delta
                x = (state2, lvl2, nxt_insert2)
                new_states_counts[x] = new_states_counts.get(x, 0) + cnt
        states_counts = new_states_counts
        #print(states_counts)
    res = sum(states_counts.values())
    """
    print("states:")
    for (state, lvl, nxt_insert), cnt in states_counts.items():
        print(state, lvl, nxt_insert, cnt)
    #print(states_counts)
    res = 0
    if n_triominos & 1:
        states_complement_counts = {}
        complements_map = {}
        for (state, lvl, nxt_insert), cnt in states_counts.items():
            state_c, h = triominoStateComplementRotated(state)
            complements_map[state] = state_c
            print(f"state = {state}")
            print(f"state_c = {state_c}")
            #states_complement_counts[(state_c, n_rows - lvl - h)] = cnt
            states_complement_counts[state_c] = cnt
        for (state, lvl, nxt_insert), cnt in states_counts.items():
            #state_c = complements_map[state]
            for (state2, lvl_delta, nxt_insert2) in nextTriominoStates(state, nxt_insert=nxt_insert):
                print(f"centre state = {state}")
                #if states_complement_counts.get(state2, 0) != cnt or state_c != state2:
                res += states_complement_counts.get(state2, 0) * cnt
                #else: res += cnt * (cnt - 1)
    else:
        #print("middle states:")
        #states_complement_counts = {}
        for (state, lvl, nxt_insert), cnt in states_counts.items():
            state_c, h = triominoStateComplementRotated(state)
            print(state, state_c)
            if state_c == state:
                print(f"self-match: {state}, {cnt}")
                res += cnt ** 2
            elif state_c in states_counts.keys():
                print(f"match: {state}, {cnt}, {state_c}, {states_counts.get(state_c, 0)}")
                res += states_counts.get(state_c, 0) * cnt
    """
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 162
def intAsHexadecimal(num: int) -> str:
    """
    For a non-negative integer, finds its representation as a
    hexadecimal number, where the digits with value between 10
    and 15 inclusive are represented by "A", "B", "C", "D", "E"
    and "F" in that order. The answer is given without leading
    zeros (except in the case of zero itself which is returned
    as "0").

    Args:
        Required positional:
        num (int): Non-negative integer giving the integer whose
                representation as a hexadecimal number is being
                sought.
    
    Returns:
    String (str) giving the hexadecimal representation of num
    without leading zeros.
    """
    conv_dict = {x: str(x) for x in range(10)}
    for i, l in enumerate("ABCDEF", start=10):
        conv_dict[i] = l
    res = []
    while num:
        num, d = divmod(num, 16)
        res.append(conv_dict[d])
    res = res[::-1]
    return "".join(res) if res else "0"

def countIntegersContainGivenDigits(max_n_dig: int, n_contained_digs: int, contained_includes_zero: bool, base: int=10) -> int:
    """
    For a given base, calculates the number of strictly positive
    integers which when expressed in the chosen base with no
    leading zeros contain at most max_n_dig digits and contain each
    of n_contained_digs selected, distinct digits (with zero one of
    these digits if and only if contained_includes_zero is given
    as True) at least once.

    Args:
        Required positional:
        max_n_dig (int): Strictly positive integer giving the
                maximum number of digits the numbers considered
                can have when expressed in the chosen base
                without leading zeros.
        n_contained_digs (int): Non-negative integer giving the
                number of selected  distinct digits that must each
                be present in the representation in the chosen
                base of the integers counted without leading zeros.
        contained_includes_zero (bool): If given as True then
                zero is one of the digits in the n_contained_digs
                selected that must be present at least once in each
                number counted when expressed in the chosen base,
                otherwise zero is not one of these digits.
        
        Optional named:
        base (int): The base in which integers are to be expressed
                when judging how many digits it has and whether it
                contains all of the selected distinct digits.
    
    Returns:
    Integer (int) giving the number of strictly positive integers
    satisfying the requirements described above.
    """

    if n_contained_digs > base - (not contained_includes_zero):
        return 0
    memo = {}
    def recur(n_dig: int, n_contained_digs: int, contained_includes_zero: bool, first: bool=True) -> int:
        if n_contained_digs > n_dig or (not n_contained_digs and contained_includes_zero): return 0
        elif not n_dig: return 1
        args = (n_dig, n_contained_digs, contained_includes_zero, first)
        if args in memo.keys(): return memo[args]
        n_opts = base - first
        n_other_opts = n_opts - n_contained_digs + (first and contained_includes_zero)
        n_nonzero_contained_opts = n_contained_digs - contained_includes_zero
        res = n_other_opts * recur(n_dig - 1, n_contained_digs, contained_includes_zero, first=False)
        res += n_nonzero_contained_opts * recur(n_dig - 1, n_contained_digs - 1, contained_includes_zero, first=False)
        if not first and contained_includes_zero:
            res += recur(n_dig - 1, n_contained_digs - 1, False, first=False)
        memo[args] = res
        return res

    res = sum(recur(n_dig, n_contained_digs, contained_includes_zero, first=True) for n_dig in range(n_contained_digs, max_n_dig + 1))
    return res
    
def countHexadecimalIntegersContainGivenDigits(max_n_dig: int=16, n_contained_digs: int=3, contained_includes_zero: bool=True) -> str:
    """
    Solution to Project Euler #162

    Calculates the number of strictly positive integers which when
    represented as a hexadecimal number with no leading zeros contain
    at most max_n_dig digits and contain each of n_contained_digs
    selected, distinct digits (with zero one of these digits if and
    only if contained_includes_zero is given as True) at least once.
    The answer is given as a string giving the hexadecimal
    representation of this total (see documentation of
    intAsHexadecimal() for more detail about this representation).

    Args:
        Required positional:
        max_n_dig (int): Strictly positive integer giving the
                maximum number of digits the numbers considered
                can have when represented as a h
                without leading zeros.
        n_contained_digs (int): Non-negative integer giving the
                number of selected  distinct digits that must each
                be present in the hexadecimal representation of the
                integers counted without leading zeros.
        contained_includes_zero (bool): If given as True then
                zero is one of the digits in the n_contained_digs
                selected that must be present at least once in each
                number counted when expressed as a hexadecimal
                number, otherwise zero is not one of these digits.
    
    Returns:
    String (str) giving hexadecimal representation of the number of
    strictly positive integers satisfying the requirements described
    above.
    """
    since = time.time()
    res1 = countIntegersContainGivenDigits(max_n_dig, n_contained_digs, contained_includes_zero, base=16)
    print(res1)
    res2 = intAsHexadecimal(res1)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res2

# Problem 163
def countCrossHatchedTrianglesUsingBottomLayer(n_layers: int) -> int:
    # Doesn't work for n_layers >= 4- have not been able to find error.
    # Most likely a possible triangle type that has not been accounted
    # for
    n = n_layers
    res = 0
    # Bottom corners to bottom edges
    ans = 0
    #for i in range(1, n + 1):
    #    ans += i - 1 - ((i - 1) // 4) # Acute angle
    #    ans += (i + 1) // 2 # 60 deg angle
    for i in range(n):
        ans += min(3 * i + 2, n - i)#i - 1 - ((i - 1) // 4) # Acute angle
        ans += min(i + 1, n - i)#(i + 1) // 2 # 60 deg angle
    res += 2 * ans
    #res += 2 * n * (n + 1)
    print(2 * ans, res)
    # Bottom corners to bottom corners
    ans = (n * (n + 1)) # Two angles no greater than 60 degrees
    res += 2 * ans
    print(2 * ans, res)
    ans = 0
    #for i in range(n + 1):
    #    ans += i - ((i + 1) // 4) # Acute and right angle
    #    ans += min(i, n - i) # 60 degrees and right angle
    #    ans += (n - i) >> 1 # Acute and 120 degrees
    for i in range(n):
        ans += min(3 * i, n - i) # Acute and vertical
        ans += min(i, n - i) # 60 degrees and vertical
        ans += min(i, n - i) # Acute and 120 degrees
    res += 2 * ans
    print(2 * ans, res)

    # Bottom corners with both edges from that corner going up, one edge going acutely
    ans = 0
    for i in range(1, n + 1):
        # Other edge from bottom at 60 degrees in same direction
        ans += i >> 1 # last edge is horizontal
        ans += ((3 * i) >> 1) # last edge is acute
        ans += i # last edge is 60 degrees
        ans += i # last edge is vertical
        #ans += ((9 * (i + 1)) >> 1) - 1 # Other edge from bottom at 60 degrees in same direction
    res += 2 * ans
    print(2 * ans, res)
    ans = 0
    for i in range(1, n):
        # Other edge from bottom is vertical
        ans += min(i, n - i) # last edge is 60 degrees
        ans += min(3 * i, ((3 * (n - i)) >> 1)) # last edge is acute
        ans += min(i, (n - i) >> 1) # last edge is horizontal
        #ans += min(i, n - i) # last edge is 60 degrees
        #ans += min() # last edge is acute
        #for j in range(1, i + 1):
        #    ans += min(3 * j, ((3 * (n - i)) >> 1)) # last edge is acute
    res += 2 * ans
    print(2 * ans, res)
    ans = 0
    for i in range(1, n):
        # Other edge from bottom is 60 degrees in other direction
        ans += min(i, (n - i) >> 1) # last edge is horizontal
        ans += min(i, ((3 * (n - i)) >> 1)) # last edge is acute
    res += 2 * ans
    print(2 * ans, res)
    ans = 0
    for i in range(1, n):
        # Other edge from bottom is acute in other direction
        ans += min(i >> 1, (n - i) >> 1) # last edge is horizontal
    res += ans
    print(ans, res)

    # Bottom corners with both edges from that corner going upwards, one edge going 60 degrees the other at least 60 degrees
    ans = 0
    for i in range(1, n):
        ans += min(i, n - i) # Upside down equilateral (i.e. last edge is horizontal)
    res += ans
    print(ans, res)
    ans = 0
    for i in range(1, n):
        ans += min(2 * i, n - i) # other edge is 60 degrees, last edge is acute
        ans += min(2 * i, 3 * (n - i)) # other edge is vertical, last edge is acute from the 60 edge
        ans += min(i, 3 * (n - i)) # other edge is vertical, last edge is acute from the vertical edge
        ans += min(i, (n - i)) # other edge is vertical, last edge is 60 degrees
        ans += min(2 * i, (n - i)) # other edge is vertical, last edge is horizontal
    res += 2 * ans
    print(2 * ans, res)

    # Upright triangle centre with both edges from that corner going upwards
    ans = 0
    for i in range(1, n - 1):
        # both edges from triangle middle corner acute
        ans += min((i + 1) >> 1, (n - i) >> 1) 
    res += ans
    print(ans, res)
    ans = 0
    for i in range(n):
        # one edge from triangle middle corner acute, the other vertical
        ans += min(2 * i + 1, (n - i) >> 1) # last edge is horizontal
        ans += min(3 * i + 1, ((3 * (n - i)) >> 1) - 1) # last edge is acute
        ans += min(i + 1, n - i) # last edge is 60 degrees
    res += 2 * ans
    print(2 * ans, res)

    # Middle upside down triangle centre with both edges from that corner going upwards
    ans = 0
    for i in range(1, n):
        ans += min((i + 1) >> 1, (n - i + 1) >> 1) # both edges from triangle middle corner acute
    res += ans
    print(ans, res)
    ans = 0
    for i in range(1, n):
        # one edge from triangle middle corner acute, the other vertical
        ans += min(2 * i, (n - i + 1) >> 1) # last edge is horizontal
        ans += min(3 * i - 1, ((3 * (n - i - 1)) >> 1) + 1) # last edge is acute
        ans += min(i, n - i) # last edge is 60 degrees
    res += 2 * ans
    print(2 * ans, res)
    
    # Edge between layer triangle centre with both edges from that corner going upwards
    ans = 0
    for i in range(1, n):
        ans += min(i, (n - i + 1) >> 1) # last edge horizontal
        ans += min(i, ((3 * (n - i - 1)) >> 1) + 2) # last edge acute
    res += 2 * ans
    print(2 * ans, res)

    print(f"n_layers = {n_layers}, ans = {res}")
    return res

def countCrossHatchedTriangles2(n_layers: int=36) -> int:
    """
    Alternative solution attempt for Project Euler #163. Does not
    give correct answer for n_layers >= 4 (unclear where the mistake
    is)
    """
    since = time.time()
    res = sum(countCrossHatchedTrianglesUsingBottomLayer(i) for i in range(1, n_layers + 1))
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

def countCrossHatchedTriangles(n_layers: int=36) -> int:
    """
    Solution to Project Euler #163
    """
    since = time.time()
    n = n_layers
    
    hs1_crossings = []
    hs2_crossings = []
    ha1_crossings = []
    ha2_crossings = []
    hv_crossings = []
    for i in range(n):
        hs1_crossings.append(set(range(min(i + 2, n))))
        hs2_crossings.append(set(range(min(i + 2, n))))
        ha1_crossings.append(set(range(i, min(2 * i + 1, 2 * n - 2) + 1)))
        ha2_crossings.append(set(range(i, min(2 * i + 1, 2 * n - 2) + 1)))
        hv_crossings.append(set(range(-min(n - 1, i + 1), min(n, i + 2))))
    #print(hs1_crossings)
    #print(hs2_crossings)
    #print(ha1_crossings)
    #print(ha2_crossings)
    #print(hv_crossings)
    s1s2_crossings = []
    s1a1_crossings = []
    s1a2_crossings = []
    s1v_crossings = []
    for i in range(n):
        s1s2_crossings.append(set(range(min(n + 1 - i, n))))
        s1a1_crossings.append(set(range(max(2 * i - 1, 0), min(n + i - 1, 2 * n - 2) + 1)))
        s1a2_crossings.append(set(range(max(i - 1, 0), min(2 * n - i - 1, 2 * n - 2) + 1)))
        s1v_crossings.append(set(range(-i, min(n - 2 * i, n - 1) + 1)))
    #print(s1s2_crossings)
    #print(s1a1_crossings)
    #print(s1a2_crossings)
    #print(s1v_crossings)
    s2a1_crossings = []
    s2a2_crossings = []
    s2v_crossings = []
    for i in range(n):
        s2a1_crossings.append(set(range(max(i - 1, 0), min(2 * n - i - 1, 2 * n - 2) + 1)))
        s2a2_crossings.append(set(range(max(2 * i - 1, 0), min(n + i - 1, 2 * n - 2) + 1)))
        s2v_crossings.append(set(range(-min(n - 2 * i, n - 1), i + 1)))
    #print(s2a1_crossings)
    #print(s2a2_crossings)
    #print(s2v_crossings)
    a1a2_crossings = []
    a1v_crossings = []
    for i in range(2 * n - 1):
        a1a2_crossings.append(set(range(i >> 1, min(2 * i + 1, 3 * n - 2 - i, 2 * n - 2) + 1)))
        a1v_crossings.append(set(range(-min((i + 1) >> 1, n - 1), min(i + 1, 3 * n - 2 * i - 2, n - 1) + 1)))
    #print(a1a2_crossings)
    #print(a1v_crossings)
    a2v_crossings = []
    for i in range(2 * n - 1):
        a2v_crossings.append(set(range(-min(i + 1, 3 * n - 2 * i - 2, n - 1), min((i + 1) >> 1, n - 1) + 1)))
    #print(a2v_crossings)

    xings = [[],
        [a2v_crossings],
        [a1v_crossings, a1a2_crossings],
        [s2v_crossings, s2a2_crossings, s2a1_crossings],
        [s1v_crossings, s1a2_crossings, s1a1_crossings, s1s2_crossings],
        [hv_crossings, ha2_crossings, ha1_crossings, hs2_crossings, hs1_crossings],
    ]

    res = 0
    
    for i1 in reversed(range(len(xings))):
        for i2 in reversed(range(i1)):
            for i3 in reversed(range(i2)):
                ans2 = 0
                for j1, j2_set in enumerate(xings[i1][i2]):
                    for j2 in j2_set:
                        
                        ans = len(xings[i1][i3][j1].intersection(xings[i2][i3][j2]))
                        ans2 += ans
                        #print(i1, i2, i3, j2, j2_set, xings[i1][i2], xings[i1][i3], xings[i2][i3], ans)
                        res += ans
                #print(i1, i2, i3, ans2)
    
    # Subtract the cases where all three edges intersect at a single point

    # Triangle centres
    ans = n ** 2 * math.comb(3, 3)
    #print(ans)
    res -= ans

    # Triangle corners (excluding the corners of the large traingle)
    #if n >= 2:
    ans = ((((n + 1) * (n + 2)) >> 1) - 3) * math.comb(6, 3)
    #print(ans)
    res -= ans

    # Large triangle corners
    res -= 3 * math.comb(3, 3)

    # Triangle edges (included for completeness, gives answer zero as only 2 lines intersect here)
    res -= (3 * ((n * (n - 1)) >> 1)) * math.comb(2, 3)

    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 164
def countIntegersConsecutiveDigitSumCapped(
        n_digs: int=20,
        n_consec: int=3,
        consec_sum_cap: int=9,
        base: int=10,
) -> int:
    """
    Solution to Project Euler #164

    For a given base, finds the number of strictly positive integers
    which when represented in that base, contains exactly n_digs digits
    and the sum of any n_consec consecutive digits in that representation
    has a sum no greater than consec_sum_cap.

    Args:
        Optional named:
        n_digs (int): The number of digits (without leading zeros) in
                the representation in the chosen base of integers
                considered.
            Default: 20
        n_consec (int): The number of consecutive digits in the
                representation in the chosen base of any of the
                integers considered that must never exceed consec_sum_cap
                in order for the integer to be counted.
            Default: 3
        consec_sum_cap (int): The maximum value any n_consec consecutive
                digits in the representation in the chosen base of
                any of the integers considered can have in order for the
                integer to be counted.
            Default: 9
        base (int): Integer strictly greater than 1 giving the base
                in which the integers are to be represented when selecting
                which integers to be considered and which of these are
                to be counted.
            Default: 10

    Returns:
    Integer (int) giving the number of strictly positive integers which
    when represented in that base, contains exactly n_digs digits and the
    sum of any n_consec consecutive digits in that representation has a sum
    no greater than consec_sum_cap.
            
    Brief outline of rationale:
    This is calculated using bottom-up dynamic programming over the
    digits in the representation of the integer in the chosen base from
    left to right, keeping and updating the record of the number of valid
    integers with each combination of the latest (n_consec - 2) digits
    (including the digit under current consideration) and whose digit
    (n_consec - 2) places to the left of the current digit is no greater
    than each possible digit, as we consider each digit in turn from left
    to right.
    """
    since = time.time()
    if consec_sum_cap >= (base - 1) * n_consec:
        return (base - 1) * base ** (n_digs - 1)
    if n_consec == 1:
        return consec_sum_cap * (consec_sum_cap + 1) ** (n_digs - 1)

    def buildNDimensionalArrayIndexSumCapped(n_dim: int, index_max: int, index_sum_max: int) -> list:
        index_max = min(index_max, index_sum_max)
        #if n_dim == 1:
        #    return [0] * (min(index_max, index_sum_max) + 1)

        def recur(dim_idx: int, idx_sum: int) -> Union[list, int]:
            if dim_idx == n_dim - 1:
                return [0] * (min(index_max, index_sum_max - idx_sum) + 1)
            res = []
            for idx in range(min(index_max, index_sum_max - idx_sum) + 1):
                res.append(recur(dim_idx + 1, idx_sum + idx))
            return res
        return recur(0, 0)
    
    def arraySliceGenerator(arr: list, max_n_indices: Optional[int]) -> Generator[Tuple[Union[list, int], Tuple[int]], None, None]:
        #print("hi")
        if max_n_indices is None: max_n_indices = float("inf")
        #print(max_n_indices)
        curr = []
        def recur(arr_slice: Union[int, list], dim_idx: int) -> Generator[Tuple[Union[list, int], Tuple[int]], None, None]:
            #print("hello")
            #print(arr_slice, dim_idx)
            if isinstance(arr_slice, int) or dim_idx == max_n_indices:
                yield (arr_slice, tuple(curr))
                return
            res = []
            curr.append(0)
            for idx in range(len(arr_slice)):
                #print(f"idx = {idx}")
                curr[-1] = idx
                yield from recur(arr_slice[idx], dim_idx + 1)
            curr.pop()
            return
        yield from recur(arr, 0)
    
    def getNDimensionalArraySlice(arr: list, inds: Tuple[int]) -> Union[list, int]:
        arr2 = arr
        for idx in inds:
            arr2 = arr2[idx]
        return arr2
    
    def accumulateLastDimension(arr: list, n_dim: int) -> None:
        for arr_slice, _ in arraySliceGenerator(arr, n_dim - 1):
            for idx in range(1, len(arr_slice)):
                arr_slice[idx] += arr_slice[idx - 1]
        return
    
    arr = buildNDimensionalArrayIndexSumCapped(n_consec - 1, base - 1, consec_sum_cap)
    #print(arr)
    #for arr_slice, idx in arraySliceGenerator(arr, n_consec - 1):
    #    print(idx, arr_slice)
    base_eff = min(base, consec_sum_cap + 1)
    curr = buildNDimensionalArrayIndexSumCapped(n_consec - 1, base - 1, consec_sum_cap)#[[0] * min(base_eff, consec_sum_cap - d1 + 1) for d1 in range(base_eff)]
    if n_consec == 2:
        for d in range(len(curr)):
            curr[d] = d
    else:
        inds = [0] * (n_consec - 2)
        #for curr_slice, curr_inds in arraySliceGenerator(curr, n_consec - 2):
        #arr_slice = getNDimensionalArraySlice(curr, inds)
        for d1 in range(1, len(curr)):
            inds[0] = d1
            arr_slice = getNDimensionalArraySlice(curr, inds)
            for d2 in range(len(arr_slice)):
                arr_slice[d2] = 1
    #for curr_slice, curr_inds in arraySliceGenerator(curr, n_consec - 2):
        #inds[0] = d1
        #arr_slice = getNDimensionalArraySlice(curr, inds)
        #for d2 in range(1, len(curr_slice)):
        #    curr_slice[d2] = 1
    #for d1 in range(1, base_eff):
    #    for d2 in range(len(curr[d1])):
    #        curr[d1][d2] = 1
    #print(curr)
    for _ in range(n_digs - 1):
        prev = curr
        curr = buildNDimensionalArrayIndexSumCapped(n_consec - 1, base - 1, consec_sum_cap)#[[0] * min(base_eff, consec_sum_cap - d1 + 1) for d1 in range(base_eff)]
        for curr_slice, curr_inds in arraySliceGenerator(curr, n_consec - 2):
            curr_inds_sum = sum(curr_inds)
            for d2 in range(min(base_eff, consec_sum_cap - curr_inds_sum + 1)):
                prev_inds = (*curr_inds[1:], d2) if n_consec > 2 else ()
                prev_slice = getNDimensionalArraySlice(prev, prev_inds)
                curr_slice[d2] += prev_slice[consec_sum_cap - curr_inds_sum - d2]
        #for d1 in range(base_eff):
        #    for d2 in range(min(base_eff, consec_sum_cap - d1 + 1)):
        #        curr[d1][d2] += prev[d2][consec_sum_cap - d1 - d2]
        #print(curr)
        accumulateLastDimension(curr, n_consec - 1)
        #for d1 in range(base_eff):
        #    for d2 in range(1, len(curr[d1])):
        #        curr[d1][d2] += curr[d1][d2 - 1]
        #print(curr)
    #res = sum(curr[d][-1] for d in range(base_eff))
    res = sum(x[0][-1] for x in arraySliceGenerator(curr, n_consec - 2))
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res


    """
    consec_sum_counts = [[0] * (min((base - 1) * (i + 1), consec_sum_cap) + 1) for i in range(n_consec - 1)]
    for d in range(1, min(consec_sum_cap + 1, base)):
        consec_sum_counts[0][d] = 1
    res = 0
    for _ in range(n_digs - 1):
        #prev_consec_sum_counts = consec_sum_counts
        #consec_sum_counts = [[0] * (min((base - 1) * i, consec_sum_cap) + 1) for i in range(n_consec - 1)]
        for j in range(len(consec_sum_counts[n_consec - 2])):
            res += min(base, consec_sum_cap) * consec_sum_counts[n_consec - 2][j]
        #for d in range(min(consec_sum_cap + 1, base)):
        #    res += consec_sum_counts[n_consec - 1][consec_sum_cap - d]
        for i in reversed(range(1, n_consec - 1)):
            consec_sum_counts[i] = [0] * (min((base - 1) * (i + 1), consec_sum_cap) + 1)
            for j in range(len(consec_sum_counts[i])):
                for d in range(min(base, j + 1)):
                    consec_sum_counts[i][j] += consec_sum_counts[i - 1][j - d]
        consec_sum_counts[0][0] = 1
        print(consec_sum_counts, res)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    
    return res
    """

# Problem 165
def blumBlumShubPseudoRandomGenerator(s_0: int=290797, s_mod: int=50515093, t_min: int=0, t_max: int=499) -> Generator[int, None, None]:
    """
    Generator iterating over the terms in a Blum Blum Shub sequence
    for a given seed value, modulus and value range.

    To find the terms in the Blum Blum Shub sequence for a given seed
    value, modulus and value range the following sequence s_i is used,
    calculated from the recurrence relation:
        s_(i + 1) = s_i * s_i (mod s_mod)
    where s_0 is the seed value and s_mod is the modulus for this Blum
    Blum Shub sequence.
    For strictly positive integers i, the i:th term in the Blum Blum Shub
    sequence, t_i is given by:
        t_n = s_i + t_min (mod t_max - t_min + 1)
    where t_min and t_max are the minimum and maximum values respectively
    of the value range.

    For well chosen values of s_0 and s_mod, this can potentially be used
    as a generator of pseudo-random integers between t_min and t_max
    inclusive.

    Note that the generator never terminates and thus any
    iterator over this generator must include provision to
    terminate (e.g. a break or return statement), otherwise
    it would result in an infinite loop.

    Args:
        Optional named:
        s_0 (int): Integer giving the seed value for this Blum Blum Shub
                sequence.
            Default: 290797
        s_mod (int): Integer strictly greater than 1 giving the modulus
                for this Blum Blum Shub sequence.
            Default: 50515093
        t_min (int): Integer giving the smallest value possible
                for terms in the Blum Blum Shub sequence.
            Default: 0
        t_max (int): Integer giving the smallest value possible
                for terms in the Blum Blum Shub sequence. Must
                be no smaller than t_min.
            Default: 499
    
    Yields:
    Integer (int) between min_value and max_value inclusive,
    with the i:th term yielded (for strictly positive integer
    i) representing the i:th term in the Blum Blum Shub sequence
    for the given seed value (s_0), modulus (s_mod) and value range
    ([t_min, t_max]).
    """
    s = s_0
    t_mod = t_max - t_min + 1
    while True:
        s = s ** 2 % s_mod
        yield (s % t_mod) + t_min
    return

def blumBlueShubPseudoRandomLineSegmentGenerator(
        n_dim: int=2,
        s_0: int=290797,
        s_mod: int=50515093,
        t_min: int=0,
        t_max: int=499,
) -> Generator[Tuple[Tuple[int, int], Tuple[int, int]], None, None]:
    """
    Generator yielding pseudo-random line segments in n_dim-dimensional
    space whose end points have integer Cartesian coordinates based
    on the terms in a Blum Blum Shub sequence for a given seed value,
    modulus and value range.

    Each group of consecutive (2 * n_dim) values in the specified Blum Blum
    Shub sequence produces a line segment, with the first n_dim values in
    the group giving the coordinates of one end point (in the same order they
    appear in the Blum Blum Shub sequence) and the last n_dim values giving
    the coordinates of the other end point.

    To find the terms in the Blum Blum Shub sequence for a given seed
    value, modulus and value range the following sequence s_i is used,
    calculated from the recurrence relation:
        s_(i + 1) = s_i * s_i (mod s_mod)
    where s_0 is the seed value and s_mod is the modulus for this Blum
    Blum Shub sequence.
    For strictly positive integers i, the i:th term in the Blum Blum Shub
    sequence, t_i is given by:
        t_n = s_i + t_min (mod t_max - t_min + 1)
    where t_min and t_max are the minimum and maximum values respectively
    of the value range.

    Note that the generator never terminates and thus any
    iterator over this generator must include provision to
    terminate (e.g. a break or return statement), otherwise
    it would result in an infinite loop.

    Args:
        Optional named:
        n_dim (int): The number of dimensions of the space for which
                line segments are to be generated.
            Default: 2
        s_0 (int): Integer giving the seed value for this Blum Blum Shub
                sequence.
            Default: 290797
        s_mod (int): Integer strictly greater than 1 giving the modulus
                for this Blum Blum Shub sequence.
            Default: 50515093
        t_min (int): Integer giving the smallest value possible
                for terms in the Blum Blum Shub sequence, and so the
                smallest value possible for any coordinate for the
                end points of the line segments.
            Default: 0
        t_max (int): Integer giving the smallest value possible
                for terms in the Blum Blum Shub sequence, and so the
                largest value possible for any coordinate for the
                end points of the line segments. Must be no smaller
                than t_min.
            Default: 499
    
    Yields:
    2-tuple of n_dim-tuples of ints, with each integer value being
    between min_value and max_value inclusive, each representing a
    line segment based on the given Blum Blum Shub sequence as
    specified above.
    """
    it = iter(blumBlumShubPseudoRandomGenerator(s_0=s_0, s_mod=s_mod, t_min=t_min, t_max=t_max))
    while True:
        ans = []
        for i in range(2):
            ans.append(tuple(next(it) for _ in range(n_dim)))
        yield tuple(ans)
    return



def twoDimensionalLineSegmentsCountInternalCrossings(
        line_segments: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        use_bentley_ottmann: bool=False,
) -> int:
    """
    For a given collection of line segments in two dimensional space whose
    endpoints have integer Cartesian coordinates, finds how many unique
    points are the site of an internal crossing between at least one pair
    of the line segments.

    A line segment is a straight line extending between two points (these
    points referred to as the end points of the line segment).

    An internal crossing of two line segments is a point that is on both line
    segments but not on either endpoint of either line segment (note that by
    Euclidean geometry at most one such line crossing can occur). Two line
    segments may be referred to as crossing internally if and only if there
    exists an internal crossing for those two line segments.

    Args:
        Required positional:
        line_segments (List of 2-tuples of 2-tuples of ints): The line
                segments whose number of unique points representing internal
                crossings are to be counted, where each line segment is
                represented by the Cartesian coordinates of its two end
                points.
        
        Optional named:
        use_bentley_ottmann (bool): Specifies the method to use. If True
                then uses an implementation of the Bentley-Ottmann line
                sweep algorithm. Otherwise checks every pair of line segments
                whose bounding boxes (with edges parallel to the x- and
                y-axes) overlap. The former method is recommended if the
                number of crossings is expected to be small compared to the
                number of edges, otherwise (e.g. for randomly generated line
                segments) the latter method is recommended.
            Default: False
    
    Returns:
    Integer (int) giving the number of unique points in two dimensional
    space that is the site of an internal crossing of at least one pair of
    the line segments given by line_segments.
    """
    #for seg in line_segments:
    #    print(f"{seg[0][0]}, {seg[0][1]}, {seg[1][0]}, {seg[1][1]}")

    if use_bentley_ottmann:
        # Bentley-Ottmann algorithm

        xings_final = BentleyOttmannAlgorithmIntegerEndpoints(line_segments)

        tot_cnt = 0
        for e_lst in xings_final.values():
            cnt = 0
            ans = 0
            for e_set in e_lst:
                l = len(e_set)
                cnt += l
                ans -= (l * (l - 1)) >> 1
            ans += (cnt * (cnt - 1)) >> 1
            tot_cnt += ans
        unique_cnt = len(xings_final)
        print(f"total number of line crossings (including duplicates) = {tot_cnt}")
        #print(f"total number of line crossing points = {unique_cnt}")
        return unique_cnt
    #return len(res)

    #print("edges crossing edge (2, 14):")
    #for edge in xings.get((2, 14)):
    #    print(f"edge, {points[edge[0]]} to {points[edge[1]]}")

    line_segments_sorted = sorted([sorted(x) for x  in line_segments])
    x_ends = SortedList()
    
    #xings2 = {}
    res = {}
    #res2 = 0
    tot_cnt2 = 0
    for i, seg in enumerate(line_segments_sorted):
        #if not (i + 1) % 50:
        #    print(f"{i + 1} edges processed, number of crossings found = {res2}")
        seg_sort = tuple(sorted(seg))
        for x2, i2 in reversed(x_ends):
            if x2 < seg_sort[0][0]: break
            intersect = twoDimensionalLineSegmentPairCrossing(
                seg_sort,
                line_segments_sorted[i2],
                internal_only=True,
                allow_endpoint_to_endpoint=False
            )
            if intersect is not None:
                res.setdefault(intersect, set())
                #res.add(intersect)
                """
                edge1 = (points_dict[seg_sort[0]], points_dict[seg_sort[1]])
                edge2 = (points_dict[line_segments_sorted[i2][0]], points_dict[line_segments_sorted[i2][1]])
                res2.setdefault(intersect, set())
                res2[intersect].add(edge1)
                res2[intersect].add(edge2)
                xings2.setdefault(edge1, set())
                xings2[edge1].add(edge2)
                xings2.setdefault(edge2, set())
                xings2[edge2].add(edge1)
                """
                tot_cnt2 += 1
            #res += twoDimensionalLineSegmentPairCrossing(seg_sort, line_segments_sorted[i2], internal_only=True, allow_endpoint_to_endpoint=False)
        x_ends.add((seg[1][0], i))
    
    #for intersect in sorted(res2.keys()):
    #    edges = res2[intersect]
    #    edges2 = xings.get(intersect, set())
    #    if len(edges) == len(edges2): continue
    #    print(f"missing intersections at {intersect} between the edges: {edges - edges2}")
    #    print(f"all edges intersecting at {intersect}: {edges}")
    
    #for edge1, edge2_set in xings2.items():
    #    for edge2 in edge2_set - xings.get(edge1, set()):
    #        if edge2 > edge1:
    #            print(f"missing intersection: {edge1} ({points[edge1[0]]} to {points[edge1[1]]}) with {edge2} ({points[edge2[0]]} to {points[edge2[1]]})")
    #print(xings2)
    print(f"total number of line crossings (including duplicates) = {tot_cnt2}")
    return len(res)
    

def blumBlumShubPseudoRandomTwoDimensionalLineSegmentsCountInternalCrossings(
        n_line_segments: int=5000,
        blumblumshub_s_0: int=290797,
        blumblumshub_s_mod: int=50515093,
        coord_min: int=0,
        coord_max: int=499,
        use_bentley_ottmann: bool=False,
) -> int:
    """
    Solution to Project Euler #165

    For a collection of n_line_segments line segments in two dimensional space
    whose endpoints have integer Cartesian coordinates generated by a
    Blum Blum Shub pseudo-random number generator as per the generator
    blumBlueShubPseudoRandomLineSegmentGenerator() for the given parameters,
    finds how many unique points are the site of an internal crossing between at
    least one pair of the line segments.

    A line segment is a straight line extending between two points (these
    points referred to as the end points of the line segment).

    An internal crossing of two line segments is a point that is on both line
    segments but not on either endpoint of either line segment (note that by
    Euclidean geometry at most one such line crossing can occur). Two line
    segments may be referred to as crossing internally if and only if there
    exists an internal crossing for those two line segments.

    For details regarding the line segment generation, see the documentation
    of blumBlueShubPseudoRandomLineSegmentGenerator(), from which (for the
    given Blum Blum Shub parameters) the first n_line_segments values are
    taken as the line segments to be analysed.
    
    Args:
        Optional named:
        n_line_segments (int): The number of line segments to be generated and
                for which the number of unique points representing internal
                crossings are to be counted.
            Default=5000
        blumblumshub_s_0 (int): Integer giving the seed value for the Blum
                Blum Shub sequence to be used to generate the line segments.
            Default=290797
        blumblumshub_s_mod (int): Integer strictly greater than 1 giving the
                modulus for the Blum Blum Shub sequence to be used to generate
                the line segments.
            Default=50515093
        coord_min (int): Integer giving the smalles value possible for either
                x or y coordinate for the endpoints of the line segments.
                Note that changes to this value will in general completely
                change the line segments generated.
            Default=0
        coord_max (int): Integer giving the smalles value possible for either
                x or y coordinate for the endpoints of the line segments.
                Note that changes to this value will in general completely
                change the line segments generated. Must be no smaller than
                coord_min.
            Default=499
        use_bentley_ottmann (bool): Specifies the method to use. If True
                then uses an implementation of the Bentley-Ottmann line
                sweep algorithm. Otherwise checks every pair of line segments
                whose bounding boxes (with edges parallel to the x- and
                y-axes) overlap. In this case (i.e. effectively randomly
                generated line segments) the latter method is recommended
                from the perspective of speed.
            Default: False
    
    Returns:
    Integer (int) giving the number of unique points in two dimensional
    space that is the site of an internal crossing of at least one pair of
    the n_line_segments line segments generated by
    blumBlumShubPseudoRandomLineSegmentGenerator().
    """
    since = time.time()
    it = iter(blumBlueShubPseudoRandomLineSegmentGenerator(
        n_dim=2,
        s_0=blumblumshub_s_0,
        s_mod=blumblumshub_s_mod,
        t_min=coord_min,
        t_max=coord_max,
    ))
    line_segments = [next(it) for _ in range(n_line_segments)]
    res = twoDimensionalLineSegmentsCountInternalCrossings(line_segments, use_bentley_ottmann=use_bentley_ottmann)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 166
def magicSquareWithRepeatsCount(square_side_length: int=4, val_max: int=9) -> int:
    """
    Solution to Problem 166
    
    Calculates the number of magic squares with a given side length
    that can use the integers between 0 and val_max inclusive, when
    there is no restriction on repeated values in the magic square.

    A magic square with a given side length is a square array of
    integers where the sum of elements in each column, in each row
    and in both long diagonals (i.e. the diagonals from one corner
    to the opposite corner, including every element directly in
    between the corners) are all the same.

    Args:
        Optional named:
        square_side_length (int): Strictly positive integer giving
                the side length of the magic squares in question.
            Default: 4
        val_max (int): Non-negative integer giving the largest
                possible values in the magic squares that can
                be counted.
            Default: 9
    
    Returns:
    Integer (int) giving the number of magic squares with side length
    square_side_length containing only the integers between 0 and
    val_max inclusive where there is no restriction on repeated
    values in the square.

    Brief outline of rationale:
    Possible values along the leading diagonal are iterated over,
    reducing the search space using the properties of magic squares
    to deduce certain functions on the square arrays that map to
    magic squares and only magic squares to other magic squares
    which we term symmetries. The types of symmetries used are
    certain permutations of row and column orders and the
    replacement of the numbers by val_max minus the original number.
    These are used to partition combinations of leading diagonal
    entries related by these symmetries together (i.e. partitioning
    into equivalence classes), multiplying the count found for one
    in each partition by the size of the partition.
    For the chosen entry of each partition, the number of
    magic squares with that leading diagonal is found using a
    backtracking algorithm, along with a function to simplify
    the current state of the incomplete magic square (finding the
    current restrictions on the ranges of values each incomplete
    square can take).

    TODO- add more detail to the outline of rationale, particularly
    about the symmetries used and why they can be used.
    """
    # Review- try to make faster by utilising more symmetries
    since = time.time()
    # TODO- generalise to arbitrary size grids
    base = val_max + 1
    n = square_side_length
    m = n ** 2

    def idx2Grid(idx: int) -> Tuple[int, int]:
        res = []
        while idx:
            idx, r = divmod(idx, n)
            res.append(r)
        res.extend([0] * (2 - len(res)))
        return tuple(reversed(res))
    
    def grid2Idx(pos: Tuple[int]) -> int:
        res = 0
        for i in pos:
            res = res * n + i
        return res
    
    idx_lines_map = [set() for _ in range(m)]
    lines = []
    for i1 in range(n):
        j = len(lines)
        lines.append(set())
        for i2 in range(n):
            idx = grid2Idx((i1, i2))
            lines[j].add(idx)
            idx_lines_map[idx].add(j)
    for i2 in range(n):
        j = len(lines)
        lines.append(set())
        for i1 in range(n):
            idx = grid2Idx((i1, i2))
            lines[j].add(idx)
            idx_lines_map[idx].add(j)
    j = len(lines)
    lines.extend([set(), set()])
    remain_idx = set(range(m))
    lead_diag_idx_lst = []
    lead_diag_line = j
    for i1 in range(n):
        idx = grid2Idx((i1, i1))
        lines[j].add(idx)
        idx_lines_map[idx].add(j)
        remain_idx.remove(idx)
        lead_diag_idx_lst.append(idx)
        idx = grid2Idx((i1, n - i1 - 1))
        lines[j + 1].add(idx)
        idx_lines_map[idx].add(j + 1)
    #print(lines)
    #print(idx_lines_map)

    res = 0
    def updateRanges(idx_ranges: Dict[int, Tuple[int]], line_deficits: Dict[int, int], updated_inds: Set[int]) -> bool:
        #, set_idx_vals: Dict[int, int],
        #print(updated_inds)
        lines_remain = {i: 0 for i in line_deficits.keys()}
        for idx in idx_ranges.keys():
            for line in idx_lines_map[idx]:
                lines_remain[line] += 1
        for line in list(lines_remain.keys()):
            cnt = lines_remain[line]
            if cnt: continue
            if line_deficits[line]: return False
            line_deficits.pop(line)
            lines_remain.pop(line)


        def setIndex(idx: int, val: int) -> bool:
            #print(f"setting index {idx} to {val}")
            idx_ranges.pop(idx)
            for line in idx_lines_map[idx].intersection(lines_remain.keys()):
                lines_remain[line] -= 1
                
                line_deficits[line] -= val
                if line_deficits[line] < 0: return False
                #print(line, lines_remain, line_deficits)
                if not lines_remain[line]:
                    r = line_deficits.pop(line, 0)
                    if r:
                        #print("nonzero line deficit in empty line")
                        #print(line, r)
                        #print(idx, line, idx_ranges, line_deficits, lines_remain)
                        return False
                    lines_remain.pop(line)
            return True
        #print(idx_ranges)
        line_update_inds = {}
        update_lines = set()
        for idx in updated_inds:
            for line in idx_lines_map[idx]:
                #line_update_inds.setdefault(line, set())
                #line_update_inds[line].add(idx)
                update_lines.add(line)
            if idx_ranges[idx][0] == idx_ranges[idx][1]:
                #for line in idx_lines_map[idx]:
                #    line_deficits[line] -= idx_ranges[idx][0]
                #    if line_deficits[line] < 0: return False
                #print(f"index {idx} set to {idx_ranges[idx][0]}")
                if not setIndex(idx, idx_ranges[idx][0]): return False
                
        update_qu = deque(update_lines)
        #print(update_qu, line_deficits)
        while update_qu:
            #print(update_qu)
            #print(idx_ranges, line_deficits, updated_inds)
            line = update_qu.popleft()
            update_lines.remove(line)
            if line not in line_deficits.keys(): continue
            #inds = line_update_inds.get(line, set())
            unset_line_min = 0
            unset_line_max = 0
            unset_line_count = 0
            update_inds = []
            neg_rng_sz_lst = SortedList()
            for idx in lines[line]:
                if idx not in idx_ranges.keys(): continue
                update_inds.append(idx)
                unset_line_count += 1
                unset_line_min += idx_ranges[idx][0]
                unset_line_max += idx_ranges[idx][1]
                neg_rng_sz_lst.add((idx_ranges[idx][0] - idx_ranges[idx][1], idx))
            if unset_line_min > line_deficits[line]: return False
            if unset_line_max < line_deficits[line]: return False
            if not update_inds:
                if line_deficits.pop(line): return False
                continue
            elif len(update_inds) == 1:
                val = line_deficits[line]
                idx2 = update_inds[0]
                rng = idx_ranges[idx2]
                #print(f"index {idx2} set to {val}")
                if rng[0] > val or rng[1] < val:
                    return False
                if not setIndex(idx2, val): return False
                #for line2 in idx_lines_map[idx2].intersection(line_deficits.keys()):
                #    line_deficits[line2] -= val
                #    if line_deficits[line2] < 0: return False
                continue
            slacks = [line_deficits[line] - unset_line_min, unset_line_max - line_deficits[line]]
            zero_slack = False
            for i, slack in enumerate(slacks):
                if slack: continue
                #line_deficits.pop(line)
                new_line_el_vals = {}
                func = min if i else max
                for idx2 in update_inds:
                    #print(f"idx2 = {idx2}, {idx_lines_map[idx2]}")
                    for line2 in idx_lines_map[idx2].intersection(line_deficits):
                        #print(f"line2 = {line2}")
                        new_line_el_vals[line2] = max(new_line_el_vals.get(line2, 0), idx_ranges[idx2][i]) if i else min(new_line_el_vals.get(line2, float("inf")), idx_ranges[idx2][i]) 
                    #print(f"index {idx2} set to {idx_ranges[idx2][i]}")
                    #idx_ranges.pop(idx2)
                    if not setIndex(idx2, idx_ranges[idx2][i]): return False
                    
                for line2, val in new_line_el_vals.items():
                    #if line2 not in line_deficits.keys(): continue
                    #print(f"subtracting {val} from line {line2}")
                    #line_deficits[line2] -= val
                    #if line_deficits[line2] < 0: return False
                    if line2 not in update_lines:
                        update_lines.add(line2)
                        update_qu.append(line2)
                zero_slack = True
                break
            if zero_slack: continue
            updated = False
            first = True
            updated_idx = set()
            #print(slacks, neg_rng_sz_lst)
            while first or updated:
                updated = False
                while neg_rng_sz_lst:
                    neg_rng_sz, idx2 = neg_rng_sz_lst[0]
                    #rng_sz = -neg_rng_sz
                    xs = -neg_rng_sz - slacks[0]
                    if xs <= 0: break
                    neg_rng_sz_lst.pop(0)
                    updated = True
                    updated_idx.add(idx2)
                    idx_ranges[idx2] = (idx_ranges[idx2][0], idx_ranges[idx2][0] + slacks[0])
                if not first and not updated: break
                first = False
                updated = False
                while neg_rng_sz_lst:
                    neg_rng_sz, idx2 = neg_rng_sz_lst[0]
                    #rng_sz = -neg_rng_sz
                    xs = -neg_rng_sz - slacks[1]
                    if xs <= 0: break
                    #print("hi2")
                    neg_rng_sz_lst.pop(0)
                    updated = True
                    updated_idx.add(idx2)
                    orig = idx_ranges[idx2]
                    idx_ranges[idx2] = (idx_ranges[idx2][1] - slacks[1], idx_ranges[idx2][1])
                    #if (orig[0] >= 0 and idx_ranges[idx2][0] < 0):
                    #    print("error")
                    #    print(orig, idx_ranges[idx2], slacks, neg_rng_sz, xs)
            new_updated_lines = set()
            for idx2 in updated_idx:
                new_updated_lines |= idx_lines_map[idx2].intersection(line_deficits.keys()) - update_lines
            new_updated_lines.discard(line)
            for line2 in new_updated_lines:
                update_lines.add(line2)
                update_qu.append(line2)
        return True

    idx_order = sorted(remain_idx, key=lambda x: -len(idx_lines_map[x]))

    def backtrack(j: int, idx_ranges: Dict[int, Tuple[int]], line_deficits: Dict[int, int]) -> int:
        #print("backtrack", j, idx_order[j] if j < len(idx_order) else -1, idx_ranges, line_deficits)
        if not idx_ranges:
            #print("solution found")
            return 1
        elif j == len(idx_order): return 0
        idx = idx_order[j]
        if idx not in idx_ranges.keys():
            return backtrack(j + 1, idx_ranges, line_deficits)
        
        res = 0
        #print(f"j = {j}, idx = {idx}, range = {idx_ranges[idx]}")
        for val in range(idx_ranges[idx][0], idx_ranges[idx][1] + 1):
            idx_ranges2 = dict(idx_ranges)
            line_deficits2 = dict(line_deficits)
            #for line in idx_lines_map[idx]:
            #    if line in line_deficits2.keys():
            #        line_deficits2[line] -= val
            idx_ranges2[idx] = (val, val)
            if not updateRanges(idx_ranges2, line_deficits2, {idx}):
                #print("setting value:", idx, val)
                continue
            #print("set value:", idx, val)
            res += backtrack(j + 1, idx_ranges2, line_deficits2)
        return res
    
    def diagGenerator() -> Generator[Tuple[Tuple[int], int], None, None]:

        hlf1_curr = [0] * (n >> 1)
        hlf1_rpt_cnt = [0]
        mult0 = math.factorial(n >> 1)
        def hlf1_recur(idx: int, mult: int=mult0):
            if idx == len(hlf1_curr):
                print(hlf1_curr, mult, hlf1_rpt_cnt[0])
                yield (hlf1_curr, mult // math.factorial(hlf1_rpt_cnt[0]))
                return
            if not idx:
                hlf1_rpt_cnt[0] = 1
                for i in range(val_max):
                    hlf1_curr[0] = i
                    yield from hlf1_recur(idx + 1, mult)
                return
            for i in range(hlf1_curr[idx - 1]):
                print(mult, hlf1_curr, hlf1_rpt_cnt[0])
                mult2 = mult // math.factorial(hlf1_rpt_cnt[0])
                hlf1_rpt_cnt[0] = 1
                hlf1_curr[idx] = i
                yield from hlf1_recur(idx + 1, mult2)
            hlf1_rpt_cnt[0] += 1
            hlf1_curr[idx] = hlf1_curr[idx - 1]
            yield from hlf1_recur(idx + 1, mult)
            return

        hlf2_curr = [0] * (n >> 1)
        hlf2_rpt_cnt = [0]
        def hlf2_recur(idx: int, hlf1: List[int], mult: int=1, lt_hlf1: bool=False):
            #print(idx, hlf1, mult, lt_hlf1)
            if idx == len(hlf2_curr):
                print(hlf2_curr, mult, hlf2_rpt_cnt[0])
                for hlf2 in set(itertools.permutations(hlf2_curr)):
                    yield (hlf2, (1 + lt_hlf1))
                return
            if not idx:
                hlf2_rpt_cnt[0] = 1
                for i in range(hlf2_curr[0]):
                    hlf2_curr[0] = i
                    yield from hlf2_recur(idx + 1, hlf1, mult=mult, lt_hlf1=True)
                hlf2_curr[0] = hlf1[0]
                yield from hlf2_recur(idx + 1, hlf1, mult=mult, lt_hlf1=False)
                return
            if lt_hlf1:
                for i in range(hlf2_curr[idx - 1]):
                    mult //= math.factorial(hlf2_rpt_cnt[0])
                    hlf2_rpt_cnt[0] = 1
                    hlf2_curr[idx] = i
                    yield from hlf2_recur(idx + 1, hlf1, mult=mult, lt_hlf1=True)
                hlf2_rpt_cnt[0] += 1
                hlf2_curr[idx] = hlf2_curr[idx - 1]
                yield from hlf2_recur(idx + 1, hlf1, mult, lt_hlf1=True)
            else:
                #print(idx, hlf2_curr)
                for i in range(min(hlf1[idx] + 1, hlf2_curr[idx - 1])):
                    print(mult, hlf2_curr, hlf2_rpt_cnt[0])
                    mult2 = mult // math.factorial(hlf2_rpt_cnt[0])
                    hlf2_rpt_cnt[0] = 1
                    hlf2_curr[idx] = i
                    yield from hlf2_recur(idx + 1, hlf1, mult2, lt_hlf1=(hlf2_curr[idx] < hlf1[idx]))
                if hlf2_curr[idx - 1] <= hlf1[idx]:
                    hlf2_rpt_cnt[0] += 1
                    hlf2_curr[idx] = hlf2_curr[idx - 1]
                    yield from hlf2_recur(idx + 1, hlf1, mult, lt_hlf1=(hlf2_curr[idx] < hlf1[idx]))
            return
        
        if not n & 1:
            for hlf1, mult1 in hlf1_recur(0, mult=mult0):
                print(hlf1, mult1)
                for hlf2, mult2 in hlf2_recur(0, hlf1, mult=mult0, lt_hlf1=0):
                    print(hlf2, mult2)
                    yield ((*hlf1, *hlf2), mult1 * mult2)
        else:
            for hlf1, mult1 in hlf1_recur(0, mult=mult0):
                print(hlf1, mult1)
                for hlf2, mult2 in hlf2_recur(0, hlf1, mult=mult0, lt_hlf1=0):
                    print(hlf2, mult2)
                    for i in range((val_max + 1) >> 1):
                        yield ((*hlf1, i, *hlf2), mult1 * mult2 * 2)
                    if not val_max & 1:
                        yield ((*hlf1, (val_max >> 1), *hlf2), mult1 * mult2)
        print(f"mult0 = {mult0}")
        return
    
    
    #lead_diag_idx_st = set(lead_diag_idx_lst)
    #print(lead_diag_idx_lst)
    #print(idx_order)
    #print(lead_diag_line)
    # Iterating over the possible number combinations in the leading diagonal
    res = 0
    tot_dict = {}
    mx_sm = val_max * n
    for num in range(base ** n):#diag1, mult in diagGenerator():
        #print(diag1, mult)
        
        diag1 = []
        while num:
            num, r = divmod(num, base)
            diag1.append(r)
        diag1.extend([0] * (n - len(diag1)))
        #print(diag1)
        #if diag1 != [2, 2, 2]: continue
        # Utilise 180 degree rotational symmetry
        
        hlf1 = diag1[:(n >> 1)]
        hlf2 = diag1[::-1][:(n >> 1)]
        #if hlf1 > hlf2: continue
        #mult = 1 + (hlf1 != hlf2)
        # Using the symmetry on permuting the rows and
        # columns in the same way such that each permutation
        # is symetrical about the midline in that direction
        mult = math.factorial(len(hlf1))
        cont = False
        cnt = 1
        for i in range(1, len(hlf1)):
            if hlf1[i] > hlf1[i - 1]:
                cont = True
                break
            elif hlf1[i] == hlf1[i - 1]:
                cnt += 1
                mult //= cnt
            else: cnt = 1
        # Using the symmtery on swapping any row and column
        # pair (where the row and column have the same index)
        # with the ones opposite in the grid and reversing
        # their orders
        if cont: continue
        for x1, x2 in zip(hlf1, hlf2):
            if x1 > x2:
                cont = True
                break
            elif x1 != x2: mult <<= 1
        if cont: continue

        """
        mult2 = math.factorial(len(hlf1))
        cont = True
        cnt = 1
        for i in range(1, len(hlf1)):
            if hlf1[i] > hlf1[i - 1]:
                cont = False
                break
            if hlf1[i] == hlf1[i - 1]:
                cnt += 1
                mult2 //= cnt
            else: cnt = 1
        if not cont: continue
        mult *= mult2
        """
        #if n >= 2:
        #    mid = n - (n >> 1)
        #    if hlf1[0] > mid: continue
        #    for lst in (hlf1, hlf2):
        #        parts = []
        #        curr = lst[0]
        
        tot = sum(diag1)
        tot2 = tot * 2
        if tot2 > mx_sm: continue
        elif tot2 < mx_sm: mult *= 2

        mn = max(0, tot - (n - 1) * val_max)
        mx = min(val_max, tot)
        
        idx_ranges = {idx: (mn, mx) for idx in range(m)}
        line_deficits = {i: tot for i in range(len(lines))}
        for val, idx in zip(diag1, lead_diag_idx_lst):
            idx_ranges[idx] = (val, val)
        #print(idx_ranges)
        updated_inds = set(lead_diag_idx_lst)
        #print(updated_inds)
        #print("pre:")
        #print(idx_ranges, line_deficits)#, updated_inds)
        #print(diag1)
        if not updateRanges(idx_ranges, line_deficits, updated_inds): continue
        #print(diag1)
        #print("post:")
        #print(idx_ranges, line_deficits)
        ans = backtrack(0, idx_ranges, line_deficits)
        if ans:
            tot_dict[tot] = tot_dict.get(tot, 0) + ans
            print("nonzero answer:", diag1, ans, mult)
        res += mult * ans
        #break
    print(tot_dict)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
    """
    base = val_max + 1
    res = 0
    for num in range(base ** 4):
        row1 = []
        while num:
            num, r = divmod(num, base)
            row1.append(r)
        tot = sum(row1)
        row2 = [None] * 4
        for r2_0 in range(max(0, tot - 3 * ))
    """

# Problem 167
def ulamSequenceGenerator(a1: int, a2: int) -> Generator[int, None, None]:
    """
    Generator yielding the terms in the Ulam sequence U(a1, a2) in
    order.

    For strictly positive integers a1, a2, the Ulam sequence U(a1, a2) is
    the sequence such that the first and second terms are a1 and a2
    respectively and the other terms are defined to be the smallest
    non-negative integer that can be expressed as the sum of any two
    distinct previous terms in the sequence in exactly one way.

    Args:
        Required positional:
        a1 (int): Strictly positive integer giving the first term
                in the Ulam sequence U(a1, a2).
        a2 (int): Strictly positive integer giving the second term
                in the Ulam sequence U(a1, a2)
    
    Yields:
    The terms in the Ulam sequence U(a1, a2) in the order they
    appear in that sequence.
    """
    seq = [a1, a2]
    yield a1
    yield a2
    seen_once = {a1 + a2}
    seen_multiple = set()
    candidates = SortedList([a1 + a2])
    while candidates:
        num = candidates.pop(0)
        yield num
        for num2 in seq:
            num3 = num + num2
            if num3 in seen_multiple: continue
            elif num3 in seen_once:
                seen_once.remove(num3)
                candidates.remove(num3)
                seen_multiple.add(num3)
            else:
                seen_once.add(num3)
                candidates.add(num3)
        seq.append(num)
        #print(seq, seen_once, candidates, )
    return

def ulamSequenceTwoOddPattern(a2: int) -> Tuple[List[int], List[int]]:
    """
    Finds a complete description of the Ulam sequence U(2, a2) where
    a2 is an odd integer strictly greater than 3.

    For strictly positive integers a1, a2, the Ulam sequence U(a1, a2) is
    the sequence such that the first and second terms are a1 and a2
    respectively and the other terms are defined to be the smallest
    non-negative integer that can be expressed as the sum of any two
    distinct previous terms in the sequence in exactly one way.

    It can be shown that for Ulam sequences of the form U(2, a2) where
    a2 is an odd integer strictly greater than 3 that:
        1. U(2, a2) contains exactly 2 even terms, 2 and 2 * (a2 + 1)
        2. The difference sequence of U(2, a2) (i.e. the sequence
            constructed from the difference between successive terms
            in U(2, a2)) is eventually cyclic (i.e. after a certain
            term, the sequence consists of a finite sequence that
            repeats back-to-back endlessly).
    
    It is therefore possible to completely describe the sequence U(2, a2)
    with a finite sequence of its initial terms until it becomes cyclic,
    and then the finite contiguous subsequence of the difference sequence
    from that term onwards until the first cycle ends.

    Args:
        Required positional:
        a2 (int): Odd integer strictly greater than 3 for which the
                Ulam sequence U(2, a2) is to be described.
    
    Returns:
    2-tuple of lists of integers, whose index 0 contains the list of
    the initial terms of the Ulam sequence U(2, a2), starting with 2, a2
    up until the first term in the sequence where the difference sequence
    becomes cyclic, and whose index 1 contains the list of difference
    sequence terms from the difference between the last term in the list
    at index 0 and the next term, up until the end of the first cycle
    of the differnce sequence, so that the difference term after the
    final difference term in this list is the first difference term in
    the list, followed by the second and so on, endlessly cycling.
            
    Outline of rationale:
    Given that there are only 2 even terms in the sequence 2 and m where
    m = 2 * (a2  + 1) and the sum of two integers is odd if and only if
    one is odd and the other even, an odd number num is in the Ulam sequence
    if and only if exactly one of (num - 2) and (num - m) is in the Ulam
    sequence.
    It follows that any terms with value greater than a given term num where
    num exceeds 2 * m (so that num - m > m) are only dependent on the two even
    terms (which are constant) and the preceding terms with value no less
    than (num - m + 2). The number of odd numbers between (num - m + 2) and num
    is (m / 2) = a2 + 1. Therefore, the remainder of the Ulam sequence and
    its difference sequence can be completely calculated from this point on
    from a maximum of the a2 + 1 preceding terms.
    It therefore follows that for terms in the Ulam exceeding 2 * (a2 + 1),
    once we encounter a contiguous subsequence of length a2 + 1 of differences
    between successive terms that has occurred before in the sequence (also
    where the terms in the corresponding Ulam sequence exceeds 2 * (a2 + 1)),
    then we have found a cycle in the difference terms.
    A rolling hash with length (a2 + 1) over the difference sequence is used
    to efficiently detect such repetitions and so cycles.
    """
    # It appears that the cycle always seems to begin at the soonest possible
    # point, but have not proved this and so we do not use this result.
    even_pair = [2, 2 * (a2 + 1)]
    #last_even_idx = (even_pair[-1] >> 1) + 3
    rh_length = even_pair[-1] >> 1

    def diffSequence() -> Generator[int, None, None]:
        #idx0 = last_even_idx + 1
        latest_odd = deque(range(a2, even_pair[1] + 2, 2))
        #print(latest_odd)
        for num in itertools.count(latest_odd[-1] + 2, step=2):
            if latest_odd[0] < num - even_pair[1]:
                latest_odd.popleft()
            if not latest_odd: break
            if not (latest_odd[0] + even_pair[1] == num) ^ (latest_odd[-1] + even_pair[0] == num):
                continue
            #print(num)
            yield (num - latest_odd[-1])
            latest_odd.append(num)
        return

    rh = rollingHashWithValue(diffSequence(), length=rh_length, p_lst=(37, 53),
            md=10 ** 9 + 7, func=(lambda x: x >> 1))

    seen_hsh = {}
    diffs = []
    rpt_start_idx = -1
    for i, (diff, hsh) in enumerate(rh):
        #print(i, diff, hsh)
        diffs.append(diff)
        if hsh is None: continue
        seen_hsh.setdefault(hsh, [])
        for i2 in seen_hsh[hsh]:
            for j in range(rh_length):
                if diffs[i - rh_length + j + 1] != diffs[i2 - rh_length + j + 1]:
                    break
            else:
                rpt_start_idx = i2 - rh_length + 1
                break
        else:
            seen_hsh[hsh].append(i)
            continue
        break
    #print(rpt_start_idx)
    for _ in range(rh_length):
        diffs.pop()
    
    initial = [2]
    for num in range(a2, even_pair[1], 2):
        initial.append(num)
    initial.append(even_pair[1])
    initial.append(even_pair[1] + 1)
    for i in range(rpt_start_idx):
        initial.append(initial[-1] + diffs[i])
    return (initial, diffs[rpt_start_idx:])

def ulamSequenceTwoOddTermValue(a2: int, term_number: int) -> int:
    """
    Finds the a given term of the Ulam sequence U(2, a2) where
    a2 is an odd integer strictly greater than 3.

    For strictly positive integers a1, a2, the Ulam sequence U(a1, a2) is
    the sequence such that the first and second terms are a1 and a2
    respectively and the other terms are defined to be the smallest
    non-negative integer that can be expressed as the sum of any two
    distinct previous terms in the sequence in exactly one way.

    Args:
        Required positional:
        a2 (int): Odd integer strictly greater than 3 for which the
                desired term in the Ulam sequence U(2, a2) is to be found.
        term_number (int): Strictly positive integer giving the which
                term to be returned, where term_number 1 corresponds
                to the first term, i.e. 2, term_number 2 corresponds
                to the second term, i.e. a2, and so on.
    
    Returns:
    Integer (int) giving the term_number:th term in the Ulam sequence
    U(2, a2).
                
    Outline of rationale:
    From ulamSequenceTwoOddPattern(), we can get a finite number of initial
    terms and a finite number of difference between successive later terms
    (starting with the difference between the last of the listed initial
    terms and the next term) that cycles endlessly to describe the rest
    of the terms in the (infinite) sequence (see the function's outline of
    rationale for justification of this way of representing the Ulam sequence
    U(2, a2) for odd a2 strictly greater than 3).
    This can be used to find a specific term in the Ulam sequence by returning
    term_number:th term in the initial terms if this list has length at least
    term_number, or if term_number exceeds this length, by adding the final
    initial term to the sum of the first (term_number - l_i) term differences
    in the cycles, where l_i is the number of initial terms. This can be
    calculated by finding the integers q and r such that:
        q * l_c + r = term_number - l_i
    where l_c is the number of term differences in the identified cycle, such
    that 0 <= r < l_c. Then the final solution is just:
        (final initial term) + q * sum(all cycle differences) +
            (sum of first r cycle differences)
    """

    #idx = term_number - 1
    initial, rpt_diffs = ulamSequenceTwoOddPattern(a2)
    #print(initial, rpt_diffs)
    if term_number <= len(initial):
        return initial[term_number - 1]
    idx2 = term_number - len(initial)
    q, r = divmod(idx2, len(rpt_diffs))
    #print(q, r)
    res = initial[-1]
    if q:
        res += q * sum(rpt_diffs)
    if r:
        res += sum(rpt_diffs[:r])
    #print(res)
    return res

def ulamSequenceTwoOddTermValueSum(a2_min: int=5, a2_max: int=21, term_number: int=10 ** 11) -> int:
    """
    Finds the sum of the term_number:th term for each of the Ulam sequences
    U(2, a2) for which a2 is an odd integers strictly greater than 3 and
    a2_min <= a2 <= a2_max.

    For strictly positive integers a1, a2, the Ulam sequence U(a1, a2) is
    the sequence such that the first and second terms are a1 and a2
    respectively and the other terms are defined to be the smallest
    non-negative integer that can be expressed as the sum of any two
    distinct previous terms in the sequence in exactly one way.

    Args:
        Optional_named:
        a2_min (int): Integer giving the lower bound on the values of
                the odd integers a2 strictly greater than 3 for which the
                term in the Ulam sequence U(2, a2) is to be included in
                the sum.
            Default: 5
        a2_max (int): Integer giving the upper bound on the values of
                the odd integers a2 strictly greater than 3 for which the
                term in the Ulam sequence U(2, a2) is to be included in
                the sum.
            Default: 21
        term_number (int): Strictly positive integer giving the which
                term in the specified Ulam sequences the sum is to be
                performed, where term_number 1 corresponds to the first
                term, i.e. 2, term_number 2 corresponds to the second
                term, and so on.
            Default: 10 ** 11
    
    Returns:
    Integer (int) giving the sum of the term_number:th term of each of
    the Ulam sequences U(2, a2) for which a2 is an odd number strictly
    greater than 3 and a2_min <= a2 <= a2_max.
                
    Outline of rationale:
    See outline of rationale for ulamSequenceTwoOddTermValue().
    """
    since = time.time()
    res = 0
    for a2 in range(max(5, a2_min + (not a2_min & 1)), a2_max + 1, 2):
        res += ulamSequenceTwoOddTermValue(a2, term_number)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 168
def rightRotationMultiplesSum(min_n_digs: int=2, max_n_digs: int=100, n_tail_digs: Optional[int]=5, base: int=10) -> int:
    """
    Solution to Project Euler #168

    Calculates the sum of all strictly positive integers whose
    representation in the chosen base without leading zeros contains
    between min_n_digs and max_n_digs inclusive and is an integer
    multiple of its right rotation, returning either that number (if
    n_tail_digs is given as None) or the integer whose representation
    in the chosen base is the same as the rightmost n_tail_digs of
    the representation in the chosen base of this sum in the same order.

    A right rotation of the representation of an integer in a chosen
    base is the integer whose representation in the chosen base
    takes as its leftmost digit, the rightmost digit of the original
    integer and its remaining digits are the other digits in the
    original integer in the same order, such that the second leftmost
    digit is the same as the leftmost digit of the representation
    of the original integer and the rightmost digit is the same as
    the second rightmost digit of the representation of the original
    integer.

    Args:
        Optional named:
        min_n_digs (int): Strictly positive integer giving the minimum
                number of digits in the representation in the chosen
                base without leading zeros of integers considered for
                inclusion in the sum.
            Default: 2
        max_n_digs (int): Strictly positive integer giving the maximum
                number of digits in the representation in the chosen
                base without leading zeros of integers considered for
                inclusion in the sum.
            Default: 100
        n_tail_digs (int or None): If given as a strictly positive
                integer, then the integer whose representation in the
                chosen base is this number of the rightmost digits of
                the representation of the sum value in the chosen base
                in the same order. If given as None, then the value of
                the sum itself is returned.
            Default: 5
        base (int): Integer strictly greater than 1 giving the base
                in which the integers are to be represented when
                assessing how many digits they contain and whether each
                on is a multiple of its right rotation.
            Default: 10
    
    Returns:
    Integer (int) giving the sum of all strictly positive integers whose
    representation in the chosen base without leading zeros contains
    between min_n_digs and max_n_digs inclusive and is an integer
    multiple of its right rotation, returning either that number (if
    n_tail_digs is given as None) or the integer whose representation
    in the chosen base is the same as the rightmost n_tail_digs of
    the representation in the chosen base of this sum in the same order.

    Outline of rationale:
    TODO
    """
    since = time.time()
    md = None if n_tail_digs is None else base ** n_tail_digs

    def recur(mult: int, carry: int=0) -> Optional[Tuple[int]]:
        if len(curr) > 1 and not carry and curr[-1] == curr[0] and curr[-2]:
            return tuple(curr[-2::-1])
        if len(curr) > max_n_digs: return None
        
        num = curr[-1] * mult + carry
        q, r = divmod(num, base)
        curr.append(r)
        return recur(mult, carry=q)

    res = 0
    for num0 in range(1, base):
        curr = [num0]
        for m in range(1, base):
            curr = [num0]
            ans = recur(m, carry=0)
            if ans is None: continue
            #print(m, ans)
            n_dig = len(ans)
            max_n_incl = max(0, max_n_digs // n_dig)
            min_n_incl = max(0, (min_n_digs - 1) // n_dig + 1)
            if min_n_incl > max_n_incl or not max_n_incl: continue
            #n_incl_rng = max_n_incl - min_n_incl + 1
            mult0 = base ** n_dig
            mult = 1
            if n_tail_digs is None:
                num = 0
                for d in ans:
                    num = num * base + d
                mult = 1#mult0 ** min_n_incl
                for i in range(max_n_incl):
                    res += num * mult * (max_n_incl - max(i, min_n_incl - 1))
                    mult *= mult0
                continue
            q, r = divmod(max_n_digs, n_dig)
            #print(q, r)
            if q:
                a = 0
                for d in ans:
                    a = a * base + d
                #print(f"a = {a}")
                #res = (res + a * (mult ** (q + 1) - 1) // (mult - 1)) % md
                for i in range(q):
                    res = (res + a * mult * (max_n_incl - max(i, min_n_incl - 1))) % md
                    mult *= mult0
            if r:
                a = 0
                for i in reversed(range(r)):
                    a = a * base + ans[-i]
                res = (res + a * mult * (max_n_incl - max(q, min_n_incl - 1))) % md
            #print(res)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res



# Problem 169
def sumOfPowersOfTwoWithMaxRepeats(num: int=10 ** 25, max_rpt: int=2) -> int:
    """
    For a non-negative integer, finds the number of ways that number
    can be expressed as the sum of powers of two, where each power
    of two can appear in the sum at most max_rpt times.

    Args:
        Optional named:
        num (int): The non-negative integer for which the number of
                ways it can be expressed as the sum of powers of two
                as described above is to be found.
            Default: 10 ** 25
        max_rpt (int): The maximum number of times each power of two
                can appear in any of the sums counted.
            Default: 2
    
    Returns:
    Integer (int) giving the number of ways that num can be expressed
    as the sum of powers of two, where each power of two can appear in
    the sum at most max_rpt times.

    Outline of rationale:
    We use top-down dynamic programming with memoisation going through
    the digits of num when expressed in binary from left to right (i.e.
    from most to least significant), with the parameters idx,
    representing position of the digit in question when read from right
    to left starting at 0 (i.e. the power to which 2 is taken for that
    digit) and higher_req, representing the number of multiples of the
    next higher power of 2 that needs to be accounted for by the current
    and smaller powers of 2.
    The recurrence consists of adding together the results when accounting
    for the quantity passed to the current stage (multiplied by 2 as the
    current power  of 2 is half the size of that from which it was passed
    on), plus one in the case that the current digit is 1 with the
    different possible multiples of the current power of 2 (the multiples
    being in the integer range 0 to max_rpt inclusive) and passing the
    remainder to the smaller powers of 2. The base case is after the units
    digit (from which no value can be passed on but also has no
    corresponding digit meaning it does not add any value to be accounted
    for), so this gives 1 (representing the empty sum) if higher_req is
    0, otherwise 0.
    We exclude some unnecessary calls by noting that the sum of
    all powers of 2 up to but not including a given power of 2
    is exactly one less than that power of 2. Therefore, an upper
    bound on the number of multiples of a higher power that can
    be accounted for by the smaller powers of 2 is:
        floor((max_rpt * (2 ** idx - 1)) / 2 ** idx)
    """
    # Review- Try to implement bottom up dynamic programming solution
    digs = []
    while num:
        digs.append(num & 1)
        num >>= 1
    n_dig = len(digs)
    memo = {}
    def recur(idx: int, higher_req: int=0) -> int:
        if idx == -1:
            # This cannot contribute anything, so is only non-zero
            # if higher_req is zero, in which case there is only
            # one option, no powers of 2 (giving an empty sum with
            # value 0).
            return 1 if not higher_req else 0
        
        #if (max_rpt * (exp - 1)) // exp < higher_req: return 0
        args = (idx, higher_req)
        if args in memo.keys(): return memo[args]
        #res = 0
        exp = 1 << (idx)
        higher_req_min = max(0, 2 * higher_req + digs[idx] - max_rpt)
        higher_req_max = min(2 * higher_req + digs[idx], (max_rpt * (exp - 1)) // exp)
        #for i in range(min(2 * higher_req + digs[idx], max_rpt) + 1):
        #    res += recur(idx - 1, higher_req=2 * higher_req + digs[idx] - i)
        res = sum(recur(idx - 1, higher_req=i) for i in range(higher_req_min, higher_req_max + 1))

        """
        if digs[idx]:
            #res = recur(idx - 1, higher_req=True) if higher_req else recur(idx - 1, higher_req=False) + recur(idx - 1, higher_req=True)
            res = recur(idx - 1, higher_req=higher_req * 2) + 
            if not higher_req: res = recur(idx - 1, higher_req=0) + recur(idx - 1, higher_req=2)
            elif higher_req == 1: res = recur(idx - 1, higher_req=0) + recur(idx - 1, higher_req=3)
            #print(f"hi1")
        else:
            #if not higher_req: res = recur(idx - 1, higher_req=0)
            #elif higher_req == 1: res = recur(idx - 1, higher_req=0) + recur(idx - 1, higher_req=2)
            #else: res = recur(idx - 1, higher_req=2)
            #print(f"hi2")
            #res = recur(idx - 1, higher_req=False) + recur(idx - 1, higher_req=True) if higher_req else recur(idx - 1, higher_req=False) + recur(idx - 1, higher_req=True)
        """
        memo[args] = res
        return res
    
    res = recur(n_dig - 1, higher_req=0)
    return res

def sumOfPowersOfTwoEachMaxTwice(num: int=10 ** 25) -> int:
    """
    Solution to Project Euler #169

    For a non-negative integer, finds the number of ways that number
    can be expressed as the sum of powers of two, where each power
    of two can appear in the sum at most 2 times.

    Args:
        Optional named:
        num (int): The non-negative integer for which the number of
                ways it can be expressed as the sum of powers of two
                as described above is to be found.
            Default: 10 ** 25
    
    Returns:
    Integer (int) giving the number of ways that num can be expressed
    as the sum of powers of two, where each power of two can appear in
    the sum at most twice.

    Outline of rationale:
    See documentation for sumOfPowersOfTwoWithMaxRepeats().
    """
    since = time.time()
    res = sumOfPowersOfTwoWithMaxRepeats(num=num, max_rpt=2)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
    """
    print(format(num, "b"))
    dp = [[0, 1]]
    while num:
        if not num & 1:
            curr.append(curr[-1])
        else:
            curr.append(sum(curr))
        #curr = [curr[0], sum(curr)] if num & 1 else [sum(curr), 0]
        num >>= 1
        print(curr)
    return curr[-1]
    """

# Problem 170
def largestPandigitalConcatenatingProduct(min_n_prods: int=2, incl_zero: bool=True, base: int=10) -> int:
    """
    Solution to Project Euler #170

    For a given base, finds the largest 0-(base - 1) or 1-(base - 1)
    pandigital concatenated product whose input numbers when concatenated
    with no leading zeros also forms a similar pandigital number where
    the number of products is no less than min_n_prods.

    A concatenated product in a given base is a strictly positive integer
    constructed by concatenating the result of a number of products
    between a common positive integer (the multiplier) and other
    strictly positive input integers.
    
    In this case, for such a concatenated product to be considered, it
    must be 0-(base - 1) or 1-(base - 1) (depending on incl_zero),
    it must have been concatenated from at least min_n_prods integers and
    the collection of digits of the multiplier and other input integers
    used to construct the concatenated product, when expressed in the
    chosen base without leading zeros, must consist of each of the
    digits from 0 to (base - 1) or 1 to (base - 1) respectively each
    exactly once.

    Args:
        Optional named:
        min_in_prods (int): Strictly positive integer giving the minimum
                number of products from which any considered concatenated
                product must consist.
            Default: 2
        incl_zero (bool): Boolean, which if True siginifies that the
                concatenated product and the concatenation of the
                multiplier and other input integers should be
                0-(base - 1) pandigital, otherwise 1-(base - 1)
                pandigital.
            Default: True
        base (int): Integer strictly greater than 1 giving the base
                in which the integers are to be represented when
                concatenating and when assessing whether integers
                or concatenations of integers are pandigital.
            Default: 10

    Returns:
    Integer (int) giving the largest 0-(base - 1) or 1-(base - 1)
    pandigital concatenated product whose input numbers when concatenated
    with no leading zeros also forms a similar pandigital number where
    the number of products is no less than min_n_prods.

    Outline of rationale:
    
    TODO
    """

    # Review- consider optimising by ensuring each separate
    # number in the concatenation is smaller than the previous
    # one by making the first digit be less than the first
    # digit in the previous number.
    since = time.time()

    if min_n_prods == 1:
        mult_max_n_digs = base >> 1
    elif min_n_prods == 2:
        mult_max_n_digs = 2
    else:
        mult_max_n_digs = 1
    print(mult_max_n_digs)
    #num2_digs = []
    curr = []
    res_digs = [None]
    num1_digs_remain = SortedList(range(1 - incl_zero, base))
    num2_digs_remain = SortedList(list(range(1 - incl_zero, base)))
    tot_n_digs = base + incl_zero - 1

    def orderedUniqueListGenerator(n_select: int, nums: Set[int], first_not_zero: bool=False) -> Generator[Tuple[int], None, None]:
        if n_select > len(nums): return
        remain = set(nums)
        curr = []
        def recur(idx: int) -> Generator[Tuple[int], None, None]:
            if idx == n_select:
                yield tuple(curr)
                return
            curr.append(0)
            rmn = set(remain) if idx or not first_not_zero else set(remain) - {0}
            for num in rmn:
                curr[-1] = num
                remain.remove(num)
                yield from recur(idx + 1)
                remain.add(num)
            curr.pop()
            return
        yield from recur(0)
        return

    def recur1(idx: int=0, num2: int=0, num2_n_digs: int=0, prec_digs_eq_best: bool=False) -> bool:
        if idx >= tot_n_digs - min_n_prods + 1: return False
        #print(idx, curr)
        #res = ()
        min_dig = max(res[0][idx], not num2) if prec_digs_eq_best else 0 + (not incl_zero or not num2)
        #print(f"min_dig = {min_dig}")
        curr.append(0)
        num2_n_digs2 = num2_n_digs + 1
        b = False
        lst = list(num2_digs_remain)# if num2 else [num2_digs_remain[-1]]
        for d in reversed(lst):
            if d < min_dig: break
            #print(curr, lst, d, num2_digs_remain)
            num2_digs_remain.remove(d)
            num2_2 = num2 * base + d
            curr[-1] = d
            b = recur1(idx=idx + 1, num2=num2_2, num2_n_digs=num2_n_digs2, prec_digs_eq_best=prec_digs_eq_best and d == min_dig)
            for mult_n_dig in range(1, mult_max_n_digs + 1):
                for mult_digs in orderedUniqueListGenerator(mult_n_dig, set(num1_digs_remain), first_not_zero=True):
                    #if not mult_digs[0]: continue
                    mult = 0
                    for d1 in mult_digs:
                        mult = mult * base + d1
                        
                    if mult < 2: continue
                    #print(mult, mult_n_dig)
                    num1, r = divmod(num2_2, mult)
                    #if curr == [7, 6, 3, 8]:
                    #    print(curr, mult, num2_2, num1, r)
                    if r: continue
                    #print(mult_digs)
                    for d1 in mult_digs:
                        num1_digs_remain.remove(d1)
                    #num1_digs_remain.remove(mult)
                    num1_digs = []
                    num1_2 = num1
                    while num1_2:
                        num1_2, d1 = divmod(num1_2, base)
                        if d1 not in num1_digs_remain:
                            break
                        num1_digs_remain.remove(d1)
                        num1_digs.append(d1)
                    else:
                        num1_n_digs = len(num1_digs)
                        slack2 = mult_n_dig - (num2_n_digs2 - num1_n_digs)
                        #if num1_n_digs >= num2_n_digs2 - 1:
                        if slack2 >= (idx < tot_n_digs - 1) * (mult_n_dig - 1):
                        #ans = ()
                        
                            #if mult == 6:
                            #    print(num2, mult)
                            b2 = recur2(1, idx + 1, 0, num2_n_digs=0, mult=mult, mult_n_dig=mult_n_dig, slack=slack2, prec_digs_eq_best=(b or (prec_digs_eq_best and d == min_dig)))
                            if b2:
                                #print(num2_2)
                                b = b2
                    for d1 in num1_digs:
                        num1_digs_remain.add(d1)
                    
                    #num1_digs_remain.add(mult)
                    for d1 in mult_digs:
                        num1_digs_remain.add(d1)
            num2_digs_remain.add(d)
            if b: break
        curr.pop()
        return b

    
    def recur2(prod_idx: int, idx: int, num2: int, num2_n_digs: int, mult: int, mult_n_dig: int, slack: int, prec_digs_eq_best: bool=False) -> bool:
        #if num2 == 0:
        #    print(idx, curr, res_digs[0], prec_digs_eq_best, mult, num2_digs_remain)
        if idx == tot_n_digs:
            if not num2 and not slack and prod_idx >= min_n_prods:
                print("solution:", curr, mult, num2, slack)
                res_digs[0] = tuple(curr)
                return True
            return False
        elif min_n_prods - prod_idx > tot_n_digs - idx:
            return False
        #res = ()
        min_dig = max(res_digs[0][idx], not num2) if prec_digs_eq_best else 0 + (not incl_zero or not num2)
        curr.append(0)
        num2_n_digs2 = num2_n_digs + 1
        b = False
        lst = list(num2_digs_remain)# if num2 else [num2_digs_remain[-1]]
        for d in reversed(lst):
            if d < min_dig: break
            num2_digs_remain.remove(d)
            num2_2 = num2 * base + d
            curr[-1] = d
            
            b = recur2(prod_idx=prod_idx, idx=idx + 1, num2=num2_2, num2_n_digs=num2_n_digs2, mult=mult, mult_n_dig=mult_n_dig, slack=slack, prec_digs_eq_best=(prec_digs_eq_best and d == min_dig))
            num1, r = divmod(num2_2, mult)
            if r:
                num2_digs_remain.add(d)
                if b: break
                continue
            num1_digs = []
            num1_2 = num1
            while num1_2:
                num1_2, d1 = divmod(num1_2, base)
                if d1 not in num1_digs_remain:
                    break
                num1_digs_remain.remove(d1)
                num1_digs.append(d1)
            else:
                num1_n_digs = len(num1_digs)
                #ans = ()
                b2 = False
                slack2 = slack - (num2_n_digs2 - num1_n_digs)
                #if num1_n_digs >= num2_n_digs2 - 1:
                if slack2 >= max(0, idx < tot_n_digs - 1, (min_n_prods - prod_idx - 1)) * (mult_n_dig - 1):#(idx < tot_n_digs - 1) * (mult_n_dig - 1):
                    b2 = recur2(prod_idx=prod_idx + 1, idx=idx + 1, num2=0, num2_n_digs=0, mult=mult, mult_n_dig=mult_n_dig, slack=slack2, prec_digs_eq_best=(b or (prec_digs_eq_best and d == min_dig)))
                #if n_digs_reduced:
                #    if num1_n_digs == num2_n_digs2:
                #        b2 = recur2(idx + 1, 0, num2_n_digs=0, mult=mult, n_digs_reduced=True, prec_digs_eq_best=(b or (prec_digs_eq_best and d == min_dig)))
                #elif num1_n_digs >= num2_n_digs2 - 1:
                #    b2 = recur2(idx + 1, 0, num2_n_digs=0, mult=mult, n_digs_reduced=(num1_n_digs == num2_n_digs2 - 1), prec_digs_eq_best=(b or (prec_digs_eq_best and d == min_dig)))
                if b2:
                    #print(num2_2)
                    b = b2
            for d1 in num1_digs:
                num1_digs_remain.add(d1)
            num2_digs_remain.add(d)
            if b: break
        curr.pop()
        return b

    if not recur1(): return -1
    res = 0
    for d in res_digs[0]:
        res = res * base + d
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res


# Problem 171
def sumSquareOfTheDigitalSquares(max_n_dig: int=20, n_tail_digs: Optional[int]=9, base: int=10) -> int:
    """
    Solution to Project Euler #171

    For a given base, finds the sum of strictly positive integers
    which, when expressed in that base contain at most max_n_dig
    digits and whose sum of squares of those digits is a perfect
    square, returning that sum (if n_tail_digs is None) or the
    value of the rightmost n_tail_digs of that sum (if n_tail_digs is a
    strictly positive integers) when interpreted as an integer in
    the chosen base.

    Args:
        Optional named:
        max_n_dig (int): Strictly positive integer giving the
                maximum number of digits when expressed in the chosen
                base of the strictly positive ntegers considered for
                inclusion in the sum.
            Default: 20
        n_tail_digs (int or None): If specified as a strictly
                positive integer, gives the number of rightmost
                digits of the sum whose value is to be returned.
                Otherwise, the sum itself is returned.
            Default: 9
        base (int): Integer strictly greater than 1 giving the base
                in which the integers are to be represented when
                assessing which integers have fewer than max_n_dig
                digits, have a sum of squares of digits equal to
                a perfect square, and when finding the value of
                the rightmost n_tail_digs digits of the final
                sum for the returned value (if n_tail_digs given as
                a strictly positive integer).
            Default: 10

    Returns:
    Integer (int) giving the value of the rightmost n_tail_digs
    digits when interpreted as an integer in the chosen base or
    (if n_tail_digs is None) the value of the sum of all strictly
    positive integers which when expressed in the chosen base
    contain no more than max_n_dig digits and the sum of the
    squares of those digits is itself a perfect square.
    """
    since = time.time()
    md = None if n_tail_digs is None else base ** n_tail_digs

    curr = [0] * (base - 1)
    def digitalSquaresNonZeroDigitCountsGenerator(d: int=1, remain: int=max_n_dig, curr_dig_sq_sum: int=0) -> Generator[Tuple[int], None, None]:
        if d == base or not remain:
            rt = isqrt(curr_dig_sq_sum)
            if rt ** 2 != curr_dig_sq_sum: return
            yield tuple(curr)
            return
        
        sq = d ** 2
        dig_sq_sum = curr_dig_sq_sum
        for cnt in range(remain + 1):
            yield from digitalSquaresNonZeroDigitCountsGenerator(d=d + 1, remain=remain - cnt, curr_dig_sq_sum=dig_sq_sum)
            curr[d - 1] += 1
            dig_sq_sum += sq
        curr[d - 1] = 0
        return
    
    res = 0
    mult = ((base ** min(max_n_dig, n_tail_digs) - 1) // (base - 1))
    for nz_dig_counts in digitalSquaresNonZeroDigitCountsGenerator(d=1, remain=max_n_dig, curr_dig_sq_sum=0):
        nz_cnt = sum(nz_dig_counts)
        if not nz_cnt: continue
        z_cnt = max_n_dig - sum(nz_dig_counts)
        multinom = math.factorial(max_n_dig) // math.factorial(z_cnt)
        for num in nz_dig_counts:
            multinom //= math.factorial(num)
        
        for d in range(1, base):
            cnt = nz_dig_counts[d - 1]
            if not cnt: continue
            res = (res + mult * d * ((multinom * cnt) // max_n_dig))
            if md is not None: res %= md
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
        

# Problem 172
def countNumbersWithDigitRepeatCap(n_dig: int=18, max_dig_rpt: int=3, base: int=10) -> int:
    """
    Solution to Project Euler #172

    Calculates the number of strictly positive integers which,
    whose representation in the chosen base with no leading zeros
    contains exactly n_dig digits with no single digit value appearing
    more than max_dig_rpt times.

    Args:
        Optional named:
        n_dig (int): Strictly positive integer specifying the number
                of digits the integers considered must contain when
                expressed in the chosen base with no leading zeros.
            Default: 18
        max_dig_rpt (int): Strictly positive integer specifying the
                maximum number of times any digit value may appear
                in the representation of any integer considered
                in the chosen base with no leading zeros.
            Default: 3
        base (int): Integer strictly greater than 1 giving the base
                in which the integers are to be represented (with
                no leading zeros) when assessing whether an it contains
                exactly n_dig digits and the same digit value no more
                than max_dig_rpt times.
            Default: 10
    
    Returns:
    Integer (int) giving the number of strictly positive integers which,
    whose representation in the chosen base with no leading zeros
    contains exactly n_dig digits with no single digit value appearing
    more than max_dig_rpt times.

    Outline of rationale:
    We take a bottom up dynamic programming approach by consider
    constructing the numbers to be counted one digit at a time from
    left to right, keeping track of the number of prefixes of that
    length and of different types exist.
    We observe that after the first digit (which cannot be 0), when going
    from one digit to the next, the order of the digits previously
    encountered does not matter, only their frequencies. This allows
    for grouping together of the prefixes encountered based on the
    frequency of the digits encountered, considerably reducing the number
    of required calculations.
    We can further observe that since all possible digit values are
    treated equally after the first digit (which again cannot be zero),
    we can further group together prefixes based on the number of
    times each digit frequency occurs over the possible digit values,
    without needing to keep track of which frequency corresponds to
    which digit value (i.e. recording a frequency of frequencies).
    As we iterate over the digits from left to right, we keep track
    of these frequency of frequencies in a dictionary, whose keys
    are a tuple of length no greater than (max_dig_rpt + 1) containing
    the frequency of each digit frequency encountered that this
    represents (with the indices of the tuple element giving the
    frequency to which it corresponds) as keys and the corresponding
    value being the number of prefixes of the current length that this
    tuple represents (accounting for possible the different allocation
    of digit values to each frequency and the different possible
    orderings of these digits).
    Care must be taken with the first digit, which may not be zero,
    but this can be accounted for easily due to the simplicity of
    only considering single digit numbers by noting that the single
    digit prefixes are just the integers from 1 to (base - 1)
    inclusive, which trivially results in the frequency of frequencies
    dictionary:
        {(base - 1, 1): base - 1}
    representing that there are (base - 1) different ways of
    constructing single digit numbers in this base such that
    (base - 1) digit values in this base occur 0 times and 1
    digit value occurs once (with the minus one in the value being
    due to the digit not being allowed to be zero).
    """
    since = time.time()
    if max_dig_rpt * base < n_dig: return 0
    f_lst = [0] * (max_dig_rpt + 1)
    f_lst[0] = base - 1
    f_lst[1] = 1

    curr = {tuple(f_lst): base - 1}
    for _ in range(n_dig - 1):
        prev = curr
        curr = {}
        for f_tup, cnt in prev.items():
            f_lst = list(f_tup)
            for f in range(len(f_tup) - 1):
                if not f_tup[f]: continue
                cnt2 = f_lst[f] * cnt
                f_lst[f] -= 1
                f_lst[f + 1] += 1
                f_tup2 = tuple(f_lst)
                curr[f_tup2] = curr.get(f_tup2, 0) + cnt2
                f_lst[f] += 1
                f_lst[f + 1] -= 1
        #print(curr)
    res = sum(curr.values())
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 173

def hollowSquareLaminaCount(max_n_squares: int=10 ** 6) -> int:
    """
    Solution to Project Euler #173

    Finds the number of different square laminae that can be
    constructed by placing at most max_n_squares in a
    non-overlapping arrangement.

    A square lamina is a shape completely enclosing exactly
    on single square region, whose outer border is also a square,
    such that each of the diagonals of the outer border square
    pass through two corners of the empty inner square region
    (so the outer square border and the inner square empty
    region are aligned and have the same center).

    Args:
        Optional named:
        max_n_squares (int): The maximum number of squares that
                can be used to form a square lamina arrangement.
            Default: 10 ** 6

    Returns:
    Integer (int) giving the number of distinct square lamina
    patterns that can be constructed by placing at most
    max_n_squares squares in a non-overlapping arrangement.


    Outline of rationale:
    By splitting the square lamina into four rectangular sections
    with edges equal to the thickness of the lamina (i.e. the
    perpendicular distance from the border of the empty region to
    the outer border) and length of the outer square border minus
    the thickness of the lamina, and considering without loss of
    generality the squares used to construct the pattern to be
    unit squares, it is apparent that this problem is equivalent
    to finding the number of non-similar non-square rectangles
    with integer side lengths and area no greater than a quarter
    of max_n_squares. In other words, we wish to find the number
    of ordered pairs of strictly positive integers (a, b) such
    that a > b and 4 * a * b <= max_n_squares.
    """
    since = time.time()
    res = 0
    max_prod = max_n_squares >> 2
    for b in range(1, isqrt(max_prod)):
        a = max_prod // b
        res += a - b
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

def hollowSquareLaminaCount2(max_n_squares: int=10 ** 6) -> int:
    """
    Alternative (less efficient) solution to Project Euler #173.

    Finds the number of different square laminae that can be
    constructed by placing at most max_n_squares in a
    non-overlapping arrangement.

    A square lamina is a shape completely enclosing exactly
    on single square region, whose outer border is also a square,
    such that each of the diagonals of the outer border square
    pass through two corners of the empty inner square region
    (so the outer square border and the inner square empty
    region are aligned and have the same center).

    Args:
        Optional named:
        max_n_squares (int): The maximum number of squares that
                can be used to form a square lamina arrangement.
            Default: 10 ** 6

    Returns:
    Integer (int) giving the number of distinct square lamina
    patterns that can be constructed by placing at most
    max_n_squares squares in a non-overlapping arrangement.
    """

    # Review- try to make more efficient
    since = time.time()
    i0 = isqrt(max_n_squares)
    i0_hlf = i0 >> 1
    res = (i0_hlf * (i0_hlf - 1))
    #print(res)
    i = i0 + 1
    while True:
        j = isqrt(i ** 2 - max_n_squares - 1)
        if j >= i - 2: break
        ans = (i - j - 1) >> 1
        #print(i, j, ans)
        res += ans
        i += 1
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 174
def hollowSquareLaminaTypeCounts(max_n_squares: int) -> Dict[int, int]:
    """
    Solution to Project Euler #173

    Finds the number of distinct square laminae that can be
    constructed by placing exactly a given number of squares in
    a non-overlapping arrangement for all numbers of squares
    between 1 and max_n_squares inclusive.

    A square lamina is a shape completely enclosing exactly
    on single square region, whose outer border is also a square,
    such that each of the diagonals of the outer border square
    pass through two corners of the empty inner square region
    (so the outer square border and the inner square empty
    region are aligned and have the same center).

    Args:
        Required positional:
        max_n_squares (int): The maximum number of squares considered
                when counting the number of ways of creating a distinct
                square lamina arrangement with a given number of
                squares.

    Returns:
    Dictionary (dict) whose keys are integers (int) between 1 and
    max_n_squares inclusive for which that number of squares
    can be placed into at least one non-overlapping arrangement
    forming a square lamina, and whose corresponding value
    is a strictly positive integer (int) giving the number of
    distinct square laminae exactly that number of squares
    can so form.

    Outline of rationale:
    Similarly to the outline of rationale in the documentation
    of the function, hollowSquareLaminaCount() by splitting the
    square lamina into four rectangular sections with edges equal
    to the thickness of the lamina (i.e. the perpendicular distance
    from the border of the empty region to the outer border) and
    length of the outer square border minus the thickness of the
    lamina, and considering without loss of generality the squares
    used to construct the pattern to be unit squares, it is apparent
    that the number of distinct square laminae with exactly a given
    number of squares is the number of non-similar and non-square
    rectangles with integer side length and area equal to exactly
    a quarter of the number of squares.
    As such, we need only consider multiples of four. Furthermore,
    the number of such rectangles for a given multiple of four
    squares will be the floor of half the number of positive
    integer factors of that number divided by four (note that
    taking the floor of half accounts for the possibility of
    perfect squares as they will be the only unpaired factor
    and therefore not be counted).
    We therefore use a prime sieve and the properties of the
    divisor function (i.e. powers of primes have exactly one
    plus that power divisors and the number of divisors of two
    coprime positive integers is the product of the number of
    divisors of each of the integers) to find the divisor count
    of all integers up to max_n_squares divided by four, and
    assign the floor of half that number to four times the
    original integer.
    """
    # Review- try to make more efficient
    
    max_prod = max_n_squares >> 2
    ps = PrimeSPFsieve(max_prod)
    res = {}
    for i in range(1, max_prod + 1):
        n_facts = ps.factorCount(i)
        n_pairs = n_facts >> 1
        #if n_pairs > max_type:
        #print(i, pf, n_facts, n_pairs)
        res[n_pairs] = res.get(n_pairs, 0) + 1
    
    return res

def hollowSquareLaminaTypeCountSum(max_n_squares: int=10 ** 6, min_type: int=1, max_type: int=10) -> int:
    """
    Solution to Project Euler #174

    Finds the number of integers between 1 and max_n_squares
    inclusive for which exactly that number of squares can be
    placed in a non-overlapping arrangement forming a square
    lamina in no less than min_type and no more than max_type
    distinct ways.

    A square lamina is a shape completely enclosing exactly
    on single square region, whose outer border is also a square,
    such that each of the diagonals of the outer border square
    pass through two corners of the empty inner square region
    (so the outer square border and the inner square empty
    region are aligned and have the same center).

    Args:
        Optional named:
        max_n_squares (int): The maximum number of squares considered
                when counting the number of ways of creating a distinct
                square lamina arrangement with a given number of
                squares.
            Default: 10 ** 6
        min_type (int): Strictly positive integer giving the smallest
                number of distinct ways a given number of squares
                should be able to form a square lamina to be counted
                in the sum.
            Default: 1
        max_type (int): Strictly positive integer no less than min_type
                giving the largest number of distinct ways a given
                number of squares should be able to form a square lamina
                to be counted in the sum.

    Returns:
    Integer (int) giving the number of integers between 1 and
    max_n_squares inclusive for which exactly that number of squares can
    be placed in a non-overlapping arrangement forming a square
    lamina in no less than min_type and no more than max_type distinct
    ways.

    Outline of rationale:
    See outline of rationale section of documentation for function
    hollowSquareLaminaTypeCounts().
    """
    since = time.time()
    cnts = hollowSquareLaminaTypeCounts(max_n_squares)
    res = sum(cnts.get(i, 0) for i in range(min_type, max_type + 1))
    if min_type < 1 and max_type >= 1:
        res += max_n_squares - sum(cnts.values())
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 175
def fractionsAndSumOfPowersOfTwo(numerator: int, denominator: int) -> int:
    
    def recur(p: int, q: int) -> int:
        #print(p, q)
        if not q: return 0
        elif p > q:
            p2 = (p - 1) % q + 1
            res = recur(p2, q)
            n_zeros = (p - p2) // q
            res <<= n_zeros
        else:
            q2 = q % p
            res = recur(p, q2)
            n_ones = (q - q2) // p
            res <<= n_ones
            res |= (1 << n_ones) - 1
        return res
    return recur(numerator, denominator)

def fractionsAndSumOfPowersOfTwoShortenedBinary(numerator: int=123456789, denominator: int=987654321) -> List[int]:
    """
    Solution to Project Euler #175

    Calkin-Wilf Tree
    """
    since = time.time()
    
    def recur(p: int, q: int) -> List[int]:
        #print(p, q)
        if not q: return []
        elif p > q:
            p2 = (p - 1) % q + 1
            res = recur(p2, q)
            n_zeros = (p - p2) // q
            res.append(n_zeros)
        else:
            q2 = q % p
            res = recur(p, q2)
            n_ones = (q - q2) // p
            res.append(n_ones)
        return res
    res = recur(numerator, denominator)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 176
def smallestCathetusCommonToNRightAngledTriangles(n_common: int=47547) -> int:
    """
    Solution to Project Euler #176

    Finds the smallest strictly positive integer that is the length
    of a cathetus of exactly n_common different right-angled triangles
    whose side lengths are all integers.

    A cathetus of a right-angled triangle is one of its shorter sides
    (i.e. a side that is not the hypotenuse).

    Args:
        Optional named:
        n_common (int): Strictly positive integer giving the exact
                number of catheti for different right-angled triangles
                to which the returned value must be equal.
            Default: 47547
    
    Returns:
    Integer (int) giving the smallest strictly positive integer that
    is equal to the length of a cathetus of exactly n_common different
    right-angled triangles whose side lengths are all integers.

    Outline of rationale:
    Use difference of squares formula, which implies that for
    give a, the number of positive integer solutions (x, y) to:
        (x ** 2 - y ** 2) = (x + y) * (x - y) = a ** 2
    is the number of ways of expressing a ** 2 as the product
    of two strictly positive integers of the same parity (where
    the order does not matter).
    We consider the cases of odd and even a separately.
    For odd a ...
    For even a ...
    TODO
    """
    since = time.time()
    pf = calculatePrimeFactorisation((n_common << 1) + 1)
    ps = PrimeSPFsieve()
    pg = iter(ps.endlessPrimeGenerator())
    p_facts = sorted(pf.keys(), reverse=True)
    res = next(pg) ** ((p_facts[0] + 1) >> 1)
    for _ in range(pf[p_facts[0]] - 1):
        res *= next(pg) ** ((p_facts[0] - 1) >> 1)
    for i in range(1, len(p_facts)):
        for _ in range(pf[p_facts[i]]):
            res *= next(pg) ** ((p_facts[i] - 1) >> 1)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 177
def integerAngledQuadrilaterals(tol: float=10 ** -9):
    """
    Solution to Project Euler #177

    Finds the number of convex quadrilaterals whose angles between
    edges and diagonals are all integer degrees, to within a
    tolerance of tol.

    Args:
        Optional named:
        tol (float): The tolerance for numbers to be considered
                integers (i.e. a number is considered to be
                an integer if the absolute difference between it
                and the nearest integer is strictly less than
                this number).
            Default: 10 ** -9
    
    Returns:
    Integer (int) giving the number of convex quadrilaterals whose
    angles between edges and diagonals are all integer degrees to
    within the tolerance tol.

    Outline of rationale:
    TODO
    """
    since = time.time()
    #res = 0
    seen = set()
    for a1 in range(45, 179):
        print(f"a1 = {a1}")
        for a2 in range(1, min(a1 + 1, 180 - a1)):
            #print(if a1 == 80 and a2 == 50): print("hi1")
            for b1 in range(1, min(180 - (a1 + a2), a1 + 1)):
                #print(if a1 == 80 and a2 == 50 and b1 == ): print("hi1")
                b2_rng = min(180 - (a1 + a2), a1 + 1)
                if a1 == a2:
                    b2_rng = min(b2_rng, b1 + 1)
                for b2 in range(1, b2_rng):
                    #sin_angle = math.sin(math.radians(a1 + a2)) * math.sin(math.radians(a1 + a2 + b1)) * math.sin(math.radians(a1 + a2 + b2)) / \
                    #        (math.sin(math.radians(a2)) * math.sin(math.radians(a2 + b1)))
                    phi = math.radians(a1 + a2 + b1)
                    cot_angle = -(math.sin(math.radians(a1)) * math.sin(math.radians(b2)) + math.cos(phi) * math.sin(math.radians(b1)) * math.sin(math.radians(a2 + b2))) / \
                            (math.sin(math.radians(b1)) * math.sin(math.radians(a2 + b2)) * math.sin(phi))
                    #if abs(sin_angle) > 1: continue
                    #print(a1, a2, b1, b2, sin_angle)
                    angle = 90 if cot_angle == 0 else math.degrees(math.atan(1 / cot_angle))
                    
                    #if a1 == 80 and a2 == 30 and b1 == 50 and b2 == 40:
                    #    print(cot_angle, angle)
                    #if a1 == 45 and a2 == 45 and b1 == 45 and b2 == 45:
                    #    print(math.sin(math.radians(a1)) * math.sin(math.radians(b2)), math.cos(phi) * math.sin(math.radians(b1)) * math.sin(math.radians(a2 + b2)), (math.sin(math.radians(b1)) * math.sin(math.radians(a2 + b2)) * math.sin(phi))) 
                    #    print(cot_angle, angle)
                    theta = round(angle)
                    if abs(angle - theta) >= tol:
                        continue
                        #res += 1
                        #print((a1, a2, b1, b2, angle))
                    if theta < 0:
                        #print(theta)
                        theta += 180
                    if theta < 0:
                        print(theta)
                    c1 = 180 - (a1 + a2 + b1)
                    if c1 > a1: continue
                    c2 = 180 - (a1 + a2 + b2)
                    if c2 > a1: continue
                    d1 = theta - c1
                    if d1 > a1: continue
                    d2 = 360 - (a1 + a2 + b1 + b2 + c1 + c2 + d1)
                    if d2 > a1: continue
                    mx = a1
                    angles = (a1, a2, b2, c2, d2, d1, c1, b1)
                    angles_final = angles
                    for i in range(2, len(angles), 2):
                        if angles[i] < mx: continue
                        angles2 = (*angles[i:], *angles[:i])
                        angles_final = max(angles_final, angles2)
                    for i in range(1, len(angles), 2):
                        if angles[i] < mx: continue
                        angles2 = (*angles[:i + 1][::-1], *angles[i + 1:][::-1])
                        angles_final = max(angles_final, angles2)
                    #if (max(angles) != a1):
                    #    print("error")
                    if a1 == 80 and a2 == 30 and b1 == 50 and b2 == 40:
                        print(angles_final)
                    #print(angles)
                    seen.add(angles_final)
    res = len(seen)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 178
def countPandigitalStepNumbers(max_n_digs: int=40, incl_zero: bool=True, base: int=10) -> int:
    """
    Solution to Project Euler #178

    For a given base, finds the number of integers that whose
    representation in the chosen base contains at most max_n_digs
    and is simultaneously 0-(base - 1) (if incl_zero is True) or
    1-(base - 1) (if incl_zero is False) pandigital and a step
    number.

    An integer is a 0-(base - 1) pandigital number in a given base if
    and only if it is non-negative and when represented in that base
    without leading zeros it contains each of the digits from 0 to
    (base - 1) at least once.

    An integer is a 1-(base - 1) pandigital number in a given base if
    and only if it is non-negative and when represented in that base
    without leading zeros it contains each of the digits from 1 to
    (base - 1) at least once and contains no zeros.

    An integer is a step number in a given base if and only if it is
    non-negative and when represented in that base without leading
    zeros, any neighbouring digits differ by exactly one. Note that
    we are not using modular arithmethic, so 0 and (base - 1) are
    not considered to differ by 1 unless base is 2.

    Args:
        Optional named:
        max_n_digs (int): The maximum number of digits of non-negative
                integers considered when represented in the chosen
                base without leading zeros.
            Default: 40
        incl_zero (bool): If True then 0-(base - 1) pandigital numbers
                are considered, otherwise 1-(base - 1) pandigital
                numbers are considered
            Default: True
        base (int): Integer strictly greater than 1 giving the base
                in which the integers are to be represented when
                assessing whether an integer contains no more than
                max_n_digs digits, is 0-(base - 1) or 1-(base - 1)
                pandigital and whether it is a step number.
            Default: 10
    
    Returns:
    Integer (int) giving the number of integers that whose
    representation in the chosen base contains at most max_n_digs
    and is simultaneously 0-(base - 1) (if incl_zero is True) or
    1-(base - 1) (if incl_zero is False) pandigital and a step
    number.

    Outline of rationale:
    Top-down dynamic programming with memoisation is used, going one
    digit at a time from left to right, keeping track of the index
    of the current digit (where the leftmost is index 0 and each
    step to the right increments the index by 1), the last digit
    encountered, and whether or not the digit 0 (if incl_zero is True)
    or 1 (if incl_zero is False) is present in the number so far
    and whether or not the digit (base - 1) is present in the number
    so far.
    Note that since the numbers are step numbers, if both of those
    digits are present then all digits in between must also be present,
    guaranteeing that the current number is 0-(base - 1) pandigital
    (if incl_zero is True) or, as long as 0 has not been included,
    1-(base - 1) pandigital.
    """

    since = time.time()
    lower = int(not incl_zero)
    upper = base - 1
    #print(max_n_digs, lower, upper)
    curr = []
    memo = {}
    def recur(idx: int, last_dig: int, incl_lower: bool=False, incl_upper: bool=False) -> int:
        if last_dig == lower: incl_lower = True
        if last_dig == upper: incl_upper = True
        if not incl_lower and max_n_digs - idx - 1 < last_dig - lower: return 0
        if not incl_upper and max_n_digs - idx - 1 < upper - last_dig: return 0
        curr.append(last_dig)
        res = int(incl_lower and incl_upper)
        if idx == max_n_digs - 1:
            #if res:
            #    print(curr)
            #    print(res)
            curr.pop()
            return res
        args = (idx, last_dig, incl_lower, incl_upper)
        if args in memo.keys():
            curr.pop()
            return memo[args]
        if last_dig > lower:
            res += recur(idx + 1, last_dig - 1, incl_lower=incl_lower, incl_upper=incl_upper)
        if last_dig < upper:
            res += recur(idx + 1, last_dig + 1, incl_lower=incl_lower, incl_upper=incl_upper)
        memo[args] = res
        curr.pop()
        return res
    #res = 0
    #for i in range(max(lower, 1), upper + 1):
    #    ans = recur(0, i, incl_lower=False, incl_upper=False)
    #    print(i, ans)
    #    res += ans
    res = sum(recur(0, i, incl_lower=False, incl_upper=False) for i in range(max(lower, 1), upper + 1))
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 179
def countConsecutiveNumberPositiveDivisorsMatch(n_max: int=10 ** 7) -> int:
    """
    Solution to Project Euler #179

    Finds the number of integers between 1 and n_max inclusive for
    which the number of positive integer factors is equal to that
    of the next integer (i.e. the one exactly one greater).

    Args:
        Optional named:
        n_max (int): The largest number considered in the count.
            Default: 10 ** 7
    
    Returns:
    Integer (int) giving the number of integers between 1 and
    n_max inclusive for which the number of positive integer factors
    is equal to that of the next integer.

    Outline of rationale:
    A prime sieve recording the largest prime factor and its
    power in the prime factorisation of each positive integer up
    to (n_max + 1) is used to efficiently calculate the number
    of factors. Then each integer from 1 to n_max is iterated
    over, using that prime sieve to calculate the number of
    factors and compare this with that of the next integer to
    see whether it should contribute to the count.
    """
    since = time.time()
    print(f"n_max = {n_max}")
    ps = PrimeSPFsieve(n_max + 1)
    print("created prime sieve")
    prev = 1
    res = 0
    for num in range(2, n_max + 2):
        cnt = ps.factorCount(num)
        res += (cnt == prev)
        #if cnt == prev:
        #    print(num - 1)
        prev = cnt
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 180
def goldenTriplets(max_order: int) -> List[Tuple[int, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]]:
    """
    Consider the families of functions for integer n:
        f_(1, n)(x, y, z) = x ** (n + 1) + y ** (n + 1) - z ** (n + 1)
        f_(2, n)(x, y, z) = (x * y + y * z + z * x) * (x ** (n - 1) + y ** (n - 1) - z ** (n - 1))
        f_(3, n)(x, y, z) = x * y * z * (x ** (n - 2) + y ** (n - 2) - z ** (n - 2))
    Use these to define the functions for integer n:
        f_n(x, y, z) = f_(1, n)(x, y, z) + f_(2, n)(x, y, z) - f_(3, n)(x, y, z)
    
    This function finds all the integer values of n and the rational
    triples (x, y, z) such that f_n(x, y, z) = 0, x < y and when expressed
    as a fraciton in lowest form, each of x, y and z are all between
    zero and one exclusive and have a denominator no greater than
    max_order.

    Args:
        Required positional:
        max_order (int): Strictly positive integer giving the largest
                denominator for any of x, y or z of rational triples
                (x, y, z) considered for inclusion in the result.
    
    Returns:
    List of 2-tuples, each of which represents values of n and (x, y, z)
    collectively satisfying the stated constraints, whose index 0 contains
    an integer (int) giving the value of n for that solution, and whose
    index 1 contains a 3-tuple of 2-tuples, representing the values of
    x, y and z respectively for the solution as fractions in lowest
    form where indices 0 and 1 contains an integers giving the numerator
    and denominator respectively.

    Outline of rationale:
    Through algebraic manipulation, the equation f_n(x, y, z) = 0 can be
    shown to be equivalent to:
        (x ** n + y ** n - z ** n) * (x + y + z) = 0
    This can be zero if and only if either (x ** n + y ** n - z ** n) = 0
    or (x + y + z) = 0. As x, y and z must all be strictly positive,
    x + y + z must also be strictly positive and so can never be zero.
    Consequently, for strictly positive x, y, z, f_n(x, y, z) = 0 is
    equivalent to:
        x ** n + y ** n = z ** n
    It follows from Fermat's last theorem that if n is an integer, this
    has no solutions over rational triples (x, y, z) unless abs(n) <= 2.
    Furthermore, for any rational x, y, z, x ** 0 + y ** 0 = 1 + 1 = 2
    and z ** 0 = 1, so this also has no solutions for n = 0.
    Therefore, we need only find solutions for when n is -2, -1, 1 or 2.
    This is done by checking each permitted value of x and y and
    checking whether for each of these values of n it produces a permitted
    value of z (i.e. a rational value between 0 and 1 exclusive).
    """
    res = []
    sqrts = {i ** 2: i for i in range(1, max_order + 1)}
    for b_x in range(2, max_order + 1):
        for a_x in range(1, b_x):
            if gcd(a_x, b_x) != 1: continue
            for b_y in range(b_x, max_order + 1):
                for a_y in range(1, a_x + 1 if b_y == b_x else b_y):
                    if gcd(a_y, b_y) != 1: continue
                    a_z1, b_z1 = addFractions((a_x, b_x), (a_y, b_y))
                    if a_z1 < b_z1 and b_z1 <= max_order:
                        if a_x * b_y <= a_y * b_x:
                            res.append((1, ((a_x, b_x), (a_y, b_y), (a_z1, b_z1))))
                        else:
                            res.append((1, ((a_y, b_y), (a_x, b_x), (a_z1, b_z1))))
                    b_z2, a_z2 = addFractions((b_x, a_x), (b_y, a_y))
                    if a_z2 < b_z2 and b_z2 <= max_order:
                        if a_x * b_y <= a_y * b_x:
                            res.append((-1, ((a_x, b_x), (a_y, b_y), (a_z2, b_z2))))
                        else:
                            res.append((-1, ((a_y, b_y), (a_x, b_x), (a_z2, b_z2))))
                    a_zsq1, b_zsq1 = addFractions((a_x ** 2, b_x ** 2), (a_y ** 2, b_y ** 2))
                    if a_zsq1 < b_zsq1 and a_zsq1 in sqrts.keys() and b_zsq1 in sqrts.keys():
                        a_z, b_z = sqrts[a_zsq1], sqrts[b_zsq1]
                        if a_x * b_y <= a_y * b_x:
                            res.append((2,( (a_x, b_x), (a_y, b_y), (a_z, b_z))))
                        else:
                            res.append((2, ((a_y, b_y), (a_x, b_x), (a_z, b_z))))
                    b_zsq2, a_zsq2 = addFractions((b_x ** 2, a_x ** 2), (b_y ** 2, a_y ** 2))
                    if a_zsq2 < b_zsq2 and a_zsq2 in sqrts.keys() and b_zsq2 in sqrts.keys():
                        a_z, b_z = sqrts[a_zsq2], sqrts[b_zsq2]
                        if a_x * b_y <= a_y * b_x:
                            res.append((-2, ((a_x, b_x), (a_y, b_y), (a_z, b_z))))
                        else:
                            res.append((-2, ((a_y, b_y), (a_x, b_x), (a_z, b_z))))
    

    """
    #print(len(res))
    seen = set()
    ps = PrimeSPFsieve(max_order)
    for (a, b, c), _ in pythagoreanTripleGeneratorByHypotenuse(primitive_only=True, max_hypotenuse=max_order):
        c_facts = sorted(ps.factors(c))
        for d in range(c + 1, max_order + 1):
            for fact in c_facts:
                c2 = c // fact
                d_c = d
                #if not d_c % c: continue
                g_a1 = gcd(a, fact)
                g_b1 = gcd(b, fact)
                #if min(g_a1, g_b1) != 1: continue
                #if d * fact // g_a1 > max_order or d * fact // g_b1 > max_order:
                #    continue
                d_a = d * (fact // g_a1)
                d_b = d * (fact // g_b1)
                a2 = a // g_a1
                b2 = b // g_b1
                #if (a, b, c) == (3, 4, 5): print(c, fact, d_a, d_b, d_c, d // c2)
                #if not d_c % c and not d_b % c and not d_a % c:
                #    print(c, fact, d_a, d_b, d_c, d // c2)
                #    continue
                #if (c, fact) == (5, 5):
                #    print("hello")
                #    print(a, b, c)
                #    print(c, fact, d_a, d_b, d_c, d // c2)
                #if (d // c2) * c == d_c:
                #    print(c, fact, d_a, d_b, d_c, d // c2)
                for num in range(1, (d // c2) + 1):
                    if gcd(num, fact) != 1: continue
                    a3, b3, c3 = a2 * num, b2 * num, c2 * num
                    #if (a, b, c) == (3, 4, 5): print(num, ((a3, d_a), (b3, d_b), (c3, d_c)))
                    g1 = gcd(a3, d_a)
                    g2 = gcd(b3, d_b)
                    g3 = gcd(c3, d_c)
                    #if gcd(g1, gcd(g2, g3)) != 1: continue
                    if min(g1, g2, g3) > 1:
                        #print(f"no unit gcd: {g1, g2, g3}")
                        continue
                    d_a2, d_b2, d_c2 = d_a // g1, d_b // g2, d_c // g3
                    if max(d_a2, d_b2, d_c2) > max_order: continue
                    ans = ((a3 // g1, d_a2), (b3 // g2, d_b2), (c3 // g3, d_c2))
                    if ans in seen:
                        print(f"repeat seen: {ans}")
                        print(g1, g2, g3)
                    seen.add(ans)
                    #if (a, b, c) == (3, 4, 5): print(num, (2, ans))
                    res.append((2, *ans))
                    #if not d_c % c and not d_b % c and not d_a % c:
                    #    print(c, fact, d_a, d_b, d_c, d // c2)
                    #    print(g1, g2, g3)
                    #    print(num, ans)
        for num in range(1, max_order + 1):
            g1 = gcd(a, num)
            g2 = gcd(b, num)
            g3 = gcd(c, num)
            a2, num_a = a // g1, num // g1
            b2, num_b = b // g2, num // g2
            c2, num_c = c // g3, num // g3
            for d_mult in range((num // a) + 1, (max_order // (max(a2, b2, c2))) + 1):
                if gcd(d_mult, num) != 1: continue
                #if gcd(num_a, d_mult) != 1 and gcd(num_b, d_mult) != 1 and gcd(num_c, d_mult) != 1: continue
                a3 = a2 * d_mult
                b3 = b2 * d_mult
                c3 = c2 * d_mult
                g1 = gcd(a3, num_a)
                g2 = gcd(b3, num_b)
                g3 = gcd(c3, num_c)
                res.append((-2, (num_b // g2, b3 // g2), (num_a // g1, a3 // g1), (num_c // g3, c3 // g3)))
    #print(res)
    print()
    """
    return res

def goldenTripletsSum(max_order: int) -> Tuple[int]:
    """
    Consider the families of functions for integer n:
        f_(1, n)(x, y, z) = x ** (n + 1) + y ** (n + 1) - z ** (n + 1)
        f_(2, n)(x, y, z) = (x * y + y * z + z * x) * (x ** (n - 1) + y ** (n - 1) - z ** (n - 1))
        f_(3, n)(x, y, z) = x * y * z * (x ** (n - 2) + y ** (n - 2) - z ** (n - 2))
    Use these to define the functions for integer n:
        f_n(x, y, z) = f_(1, n)(x, y, z) + f_(2, n)(x, y, z) - f_(3, n)(x, y, z)
    
    This function finds the sum over the integer values of n and the rational
    triples (x, y, z) such that f_n(x, y, z) = 0, x < y and when expressed
    as a fraciton in lowest form, each of x, y and z are all between
    zero and one exclusive and have a denominator no greater than
    max_order of (x + y + z).

    Args:
        Required positional:
        max_order (int): Strictly positive integer giving the largest
                denominator for any of x, y or z of rational triples
                (x, y, z) considered for inclusion in the result.
    
    Returns:
    2-tuple of integers (int), giving the rational number result
    of the sum over (x + y + z) for all rational ordered triples (x, y, z)
    and integers n satisfying the above stated properties. This is
    represented as a fraction in lowest terms, where index 0 contains
    a non-negative integer giving the numerator of this fraction and index 1
    contains a strictly positive integer giving the denominator of this
    fraction.

    Outline of rationale:
    See outline of rationale section in the function goldenTriplets()
    for details of the method of finding the values of (x, y, z) and
    n satisfying the given properties.
    """

    triplets = goldenTriplets(max_order)
    res = (0, 1)
    tot1 = 0
    tot2 = 0
    tots = {}
    seen = set()
    tots_breakdown = {}
    for n, (x, y, z) in triplets:
        mult = 1 + (x != y)
        tot1 += mult
        tot2 += 1
        tots[n] = tots.get(n, 0) + 1
        
        add = addFractions(x, addFractions(y, z))
        add = (add[0], add[1])
        tots_breakdown.setdefault(n, set())
        tots_breakdown[n].add((x, y, z))
        #if n == 2:
        #    print(n, (x, y, z), add)
        if add in seen: continue
        res = addFractions(res, add)
        seen.add(add)
        #print(n, (x, y, z), add, res)
        #res = addFractions(res, add)
    #print(f"total1 = {tot1}, total2 = {tot2}, unique = {len(seen)}")
    #print(f"totals breakdown = {tots}")
    tots_breakdown_counts = {x: len(y) for x, y in tots_breakdown.items()}
    #print(f"totals unique breakdown = {tots_breakdown_counts}")
    #print(res)
    return res

def goldenTripletsSumTotalNumeratorDenominator(max_order: int=35) -> int:
    """
    Solution to Project Euler #180

    Consider the families of functions for integer n:
        f_(1, n)(x, y, z) = x ** (n + 1) + y ** (n + 1) - z ** (n + 1)
        f_(2, n)(x, y, z) = (x * y + y * z + z * x) * (x ** (n - 1) + y ** (n - 1) - z ** (n - 1))
        f_(3, n)(x, y, z) = x * y * z * (x ** (n - 2) + y ** (n - 2) - z ** (n - 2))
    Use these to define the functions for integer n:
        f_n(x, y, z) = f_(1, n)(x, y, z) + f_(2, n)(x, y, z) - f_(3, n)(x, y, z)
    
    This function finds the sum over the integer values of n and the rational
    triples (x, y, z) such that f_n(x, y, z) = 0, x < y and when expressed
    as a fraciton in lowest form, each of x, y and z are all between
    zero and one exclusive and have a denominator no greater than
    max_order of (x + y + z), returning the sum of the numerator and
    denominator of this value when it is expressed as a fraction in
    lowest terms.

    Args:
        Required positional:
        max_order (int): Strictly positive integer giving the largest
                denominator for any of x, y or z of rational triples
                (x, y, z) considered for inclusion in the result.
    
    Returns:
    2-tuple of integers (int), giving the sum of the numerator and denominator
    of the sum over (x + y + z) for all rational ordered triples (x, y, z)
    and integers n satisfying the above stated properties, when represented
    as a fraction in lowest terms (i.e. such that the denominator is strictly
    positive and the numerator and denominator are coprime).

    Outline of rationale:
    See outline of rationale section in the function goldenTriplets()
    for details of the method of finding the values of (x, y, z) and
    n satisfying the given properties.
    """
    since = time.time()
    frac = goldenTripletsSum(max_order)
    res = sum(frac)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 181
def groupingNDifferentColouredObjects(colour_counts: List[int]=[40, 60]) -> int:
    """
    Solution to Project Euler Problem #181

    Calculates the number of ways of partitioning numbers of coloured objects
    where there is a given number of objects of each different colour, objects
    of the same colour are indistinguishable, two partitions with the same
    contents are considered the same, regardless of the order of the objects
    and two partitionings where the partitions are permutations of each other
    are considered the same.

    Note that if colour_counts contains only one entry, this is just the
    partition function.

    Args:
        Optional named:
        colour_counts (list of ints): The number of objects of each different
                colour for which the number of distinct partitionings is
                being sought.
            Default: [40, 60]
    
    Returns:
    Integer (int) giving the number of distinct partitionings of objects
    with the number of objects of each different colour as given by
    colour_counts, where objects of the same colour are indistinguishable,
    two partitions with the same contents are considered the same, regardless
    of the order of the objects and two partitionings where the partitions
    are permutations of each other are considered the same.

    Brief outline of rationale:
    This is solved using top-down dynamic programming, ensuring that
    the different partitionings are only counted once by only considering
    partitions that are lexicographically smaller than the previous
    partition (where equal partitions are handled by considering the
    different possible multiples of that partition).
    """
    # Review- currently very slow
    since = time.time()
    remain = sorted(colour_counts, reverse=True)
    while remain and not remain[-1]: remain.pop()
    #print(remain)
    n_colours = len(remain)
    if not n_colours: return 1
    
    def lexicographicallySmallerGroupingGenerator(remain: List[int], lex_max: List[int], first_remain_nonzero: bool=False) -> Generator[Tuple[Tuple[int], int], None, None]:
        curr = [0] * n_colours
        for idx_mx in reversed(range(n_colours)):
            if remain[idx_mx]: break
        else: return
        for idx_mx2 in reversed(range(idx_mx, n_colours)):
            if lex_max[idx_mx2]: break
        #print(f"idx_mx = {idx_mx}")
        #n_tail_zeros = n_colours - idx_mx - 1
        def recur(idx: int, pref_eq: bool=True, seen_remain: bool=False, max_mult: Union[int, float]=float("inf"), seen_nonzero: bool=False) -> Generator[Tuple[Tuple[int], int], None, None]:
            if idx == idx_mx + 1:
                #print(curr)
                yield (tuple(curr), max_mult)
                return
            curr[idx] = 0
            if not remain[idx]:
                if seen_nonzero or (idx < idx_mx):
                    yield from recur(idx + 1, pref_eq=pref_eq and (not lex_max[idx]), seen_remain=seen_remain, max_mult=max_mult, seen_nonzero=seen_nonzero)
                return
            mn = 1 - bool((seen_nonzero or (idx < idx_mx)) and (not first_remain_nonzero or seen_remain))
            #seen_remain = True
            #print(f"mn = {mn}", (seen_nonzero or (idx < idx_mx)))
            mx = min(remain[idx], lex_max[idx] - (idx == idx_mx2)) if pref_eq else remain[idx]
            #print(f"idx = {idx}, curr = {curr}, pref_eq = {pref_eq}, max_mult = {max_mult}, mn = {mn}, mx = {mx}, lex_max = {lex_max}, remain = {remain}")
            #seen_remain2 = seen_remain or (remain[idx] != 0)
            
            for cnt in range(mn, mx + 1):
                curr[idx] = cnt
                max_mult2 = min(max_mult, remain[idx] // curr[idx]) if curr[idx] else max_mult
                #print(f"max_mult2 = {max_mult2}, remain[idx] = {remain[idx]}, max_mult = {max_mult}, curr[-1] = {curr[-1]}, mn = {mn}, mx = {mx}")
                if not max_mult2: break
                yield from recur(idx + 1, pref_eq=(pref_eq and cnt == lex_max[idx]), seen_remain=True, max_mult=max_mult2, seen_nonzero=(seen_nonzero or bool(cnt)))
            curr[idx] = 0
            return
        #grps = []
        yield from recur(0)
        return

    #nonzero_set = set(range(n_colours))
    memo = {}
    def recur(prev_group: Tuple[int]) -> int:
        #if not nonzero_set: return 1
        for last_idx in reversed(range(len(remain))):
            if remain[last_idx] > 0: break
        else: return 1
        args = (tuple(min(prev_group[idx], remain[idx] + 1) for idx in range(n_colours)), tuple(remain))
        if args in memo.keys(): return memo[args]
        res = 0
        #grps.append(None)
        for grp, max_mult in lexicographicallySmallerGroupingGenerator(remain, prev_group, first_remain_nonzero=True):
            #print(grp, max_mult)
            for mult in range(1, max_mult + 1):
                for i in range(n_colours):
                    remain[i] -= grp[i]
                #print(prev_group, grp, mult, remain)
                #grps[-1] = (grp, mult)
                res += recur(grp)
            if max_mult > 0:
                for i in range(n_colours):
                    remain[i] += grp[i] * max_mult
            #print(remain)
        #grps.pop()
        memo[args] = res
        return res
    #grps = []
    res = recur(tuple(float("inf") for _ in range(n_colours)))
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 182
def exponentsMinimisingRSAEncryptionUnconcealed(p: int, q: int) -> List[int]:
    """
    An RSA encryption based on the distinct prime numbers p and q of
    an integer m between 0 and (n - 1) inclusive can be calculated as
    follows:
     1) Define n = p * q, phi = (p - 1) * (q - 1)
     2) Select an integer e between 1 and phi exclusive that is coprime
        with phi (i.e. the greatest common divisor with phi is 1).
     3) The encryption applied to an integer m between 0 and (n - 1)
        inclusive is then:
         m ** e (mod n)

    For RSA encryptions based on given distinct primes p and q, this
    function calculates the possible values of e (between 1 and phi
    exclusive and coprime with phi, where phi = (p - 1) * (q - 1))
    for which there are no other choices of such values of e for which
    the RSA encryptions result in fewer integers between 0 and (n - 1)
    (where n = p * q) being encrypted to themselves (so-called
    uncovered messages).
         
    Args:
        Required positional:
        p (int): Prime number giving the first of the two primes on
                which the RSA encryptions of interest are to be
                based. Should be different from q.
        q (int): Prime number giving the second of the two primes on
                which the RSA encryptions of interest are to be
                based. Should be different from p.
    
    Returns:
    List of integers (ints) between 1 and phi exclusive (where phi
    is (p - 1) * (q - 1)) giving the possible choices of e for RSA
    encryptions based on p and q for which there are no other 
    possible choices of e result in fewer integers between 0 and
    (n - 1) (where n = p * q) being encrypted to themselves by the
    corresponding RSA encryption.

    Outline of rationale:
    It can be shown that for 1 < e < phi, the number of integers
    in the range [0, n - 1] that map to themselves is:
        (1 + gcd(e - 1, p - 1)) * (1 + gcd(e - 1, q - 1))
    We therefore iterate over the values of e between 2 and (phi - 1)
    inclusive such that gcd(e, phi) = 1, keeping track of the
    minimum value of the above expression encountered and a
    list of the values of e encountered for which that expression
    takes on that value.
    """
    # Review- derive the expression given in Outline of rationale.

    # For 1 < e < phi, the number of unconcealed integers on
    # the range [0, n - 1] is:
    #  (1 + gcd(e - 1, p - 1)) * (1 + gcd(e - 1, q - 1))
    # Therefore need to find the values of e such that
    # gcd(e - 1, p - 1) = 1, gcd(e, p - 1) = 1, gcd(e - 1, q - 1) = 1
    # and gcd(e, p - 1) = 1.

    mn_cnt = float("inf")
    res = []
    phi = (p - 1) * (q - 1)
    for e in range(2, phi):
        if gcd(e, phi) != 1: continue
        cnt = (1 + gcd(e - 1, p - 1)) * (1 + gcd(e - 1, q - 1))
        if cnt > mn_cnt: continue
        if cnt < mn_cnt:
            mn_cnt = cnt
            res = []
        res.append(e)
    print(mn_cnt)#, res)
    return res

    """
    less1_factors = [[], []]
    for i, n in enumerate((p - 1, q - 1)):
        if not n & 1:
            less1_factors[i].append(2)
            n >>= 1
        while not n & 1:
            n >>= 1
        for m in range(3, isqrt(n) + 1, 2):
            if m ** 2 > n: break
            n2, r = divmod(n, m)
            if r: continue
            less1_factors[i].append(m)
            while not r:
                n = n2
                n2, r = divmod(n, m)
        if n > 1: less1_factors[i].append(n)
    print("hello")
    phi = (p - 1) * (q - 1)
    sieve = [True] * phi
    all_p_factors = sorted(set(less1_factors[0]).union(less1_factors[1]))
    print(p - 1, q - 1, all_p_factors)
    for n in all_p_factors:
        for idx in range(n, phi, n):
            sieve[idx] = False
    print(sieve)
    res = []
    for e in range(2, phi):
        if sieve[e - 1] and sieve[e]: res.append(e)
    print(res)
    return res
    """

def exponentsMinimisingRSAEncryptionUnconcealedSum(p: int=1009, q: int=3643) -> int:
    """
    Solution to Project Euler #182

    An RSA encryption based on the distinct prime numbers p and q of
    an integer m between 0 and (n - 1) inclusive can be calculated as
    follows:
     1) Define n = p * q, phi = (p - 1) * (q - 1)
     2) Select an integer e between 1 and phi exclusive that is coprime
        with phi (i.e. the greatest common divisor with phi is 1).
     3) The encryption applied to an integer m between 0 and (n - 1)
        inclusive is then:
         m ** e (mod n)

    For RSA encryptions based on given distinct primes p and q, this
    function calculates the sum of the possible values of e (between 1 and
    phi exclusive and coprime with phi, where phi = (p - 1) * (q - 1))
    for which there are no other choices of such values of e for which
    the RSA encryptions result in fewer integers between 0 and (n - 1)
    (where n = p * q) being encrypted to themselves (so-called
    uncovered messages).
         
    Args:
        Optional named:
        p (int): Prime number giving the first of the two primes on
                which the RSA encryptions of interest are to be
                based. Should be different from q.
            Default: 1009
        q (int): Prime number giving the second of the two primes on
                which the RSA encryptions of interest are to be
                based. Should be different from p.
            Default: 3643
    
    Returns:
    Integer (int) giving the sum of the possible choices of e for RSA
    encryptions based on p and q for which there are no other 
    possible choices of e result in fewer integers between 0 and
    (n - 1) (where n = p * q) being encrypted to themselves by the
    corresponding RSA encryption.

    Outline of rationale:
    See outline of rationale for the function
    exponentsMinimisingRSAEncryptionUnconcealed().
    """
    since = time.time()
    res = sum(exponentsMinimisingRSAEncryptionUnconcealed(p, q))
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 183
def partsCountMaximisingProductOfParts(n: int) -> int:
    """
    For a given strictly positive integer, identifies the number of
    equal parts that integer can be divided into that maximises
    the size of the parts to the power of the number of parts.

    Args:
        Required positional:
        n (int): Strictly positive integer for which the number
                of parts maximising the part size to the power
                of the number of parts is to be found.
    
    Returns:
    Integer (int) giving the number of equal parts n is divided
    into that maximises the size of the parts to the power of the
    number of parts.
    """
    """
    num1 = isqrt(n)
    if num1 * num1 == n: return num1
    if (n / (num1 + 1)) ** (num1 + 1) > (n / (num1)) ** (num1):
        return num1 + 1
    return num1
    """
    lft, rgt = 1, n
    while lft < rgt:
        mid = rgt - ((rgt - lft) >> 1)
        num1 = (mid - 1) * math.log(n / (mid - 1)) 
        num2 = mid * math.log(n / mid)
        #print(mid, num1, num2)
        if num1 > num2:
            rgt = mid - 1
            continue
        num3 = (mid + 1) * math.log(n / (mid + 1))
        #print(num3)
        if num3 > num2:
            lft = mid + 1
            continue
        return mid
    return lft

def maximumProductOfPartsTerminatingSum(n_min: int=5, n_max: int=10 ** 4, base: int=10) -> int:
    """
    Solution to Problem 183

    For a strictly positive integers n, define the function M(n) to be the
    maximum value of r ** k where k is a strictly positive integer and
    k * r = n.

    For strictly positive integers n and integers strictly greater than 1
    base, define the function D(n, base) to be equal to -n if M(n) has a
    terminating base-mal representation (i.e. the equivalent of a decimal
    representation, but using the integer base as the base rather than base
    10) and n if M(n) has a non-terminating base-mal representation.

    This function calculates the sum:
        (sum n from n_min to n_max) D(n, base)
    
    Args:
        Optional named:
        n_min (int): Strictly positive integer giving the smallest value of
                n included in the sum.
            Default: 5
        n_max (int): Strictly positive integer giving the largest value of
                n included in the sum.
            Default: 10 ** 4
        base (int): Integer strictly greater than 1 giving the base in which
                the value of function M(n) is to be represented when assessing
                whether its representation is terminating or non-terminating
                for each value of n in the sum over D(n, base).
            Default: 10
    
    Returns:
    Integer (int) giving the value of the sum:
        (sum n from n_min to n_max) D(n, base)
    where D(n, base) is defined above.
    """
    since = time.time()
    base_pf = calculatePrimeFactorisation(base)
    base_p_facts = set(base_pf.keys())
    res = 0
    for num in range(n_min, n_max + 1):
        n_parts = partsCountMaximisingProductOfParts(num)
        #print(num, n_parts)
        n_parts2 = n_parts // gcd(num, n_parts)
        for p in base_p_facts:
            q = n_parts2
            r = 0
            while not r:
                n_parts2 = q
                q, r = divmod(n_parts2, p)
            if n_parts2 == 1: break
        else:
            res += num
            continue
        res -= num
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 184
def orderedFractionsWithMaxNumeratorDenominatorSquareSum(max_numerator_denominator_square_sum: int, reverse: bool=False, incl_zero: bool=True, incl_one: bool=True) -> Generator[Tuple[int, int], None, None]:
    """
    Generator yielding all rational numbers between zero and one
    for which, when expressed as a fraction in lowest terms the
    sum of the squares of the numerator and denominator does not
    exceed max_numerator_denominator_square_sum. These are
    yielded in order of increasing (if reverse is False) or
    decreasing (if reverse is True) size.

    Args:
        Required positional:
        max_numerator_denominator_square_sum (int): Strictly positive
                integer giving the largest value for the squared sum
                of the numerator and denominator for any yielded
                rational number when expressed as a fraction in lowest
                terms.
        
        Optional named:
        reverse (bool): Boolean indicating whether the rational numbers
                are to be yielded in increasing (if False) or decreasing
                (if True) order of size.
            Default: False
        incl_zero (bool): Boolean indicating whether the value 0 should
                be yielded (as either the first or last value yielded
                depending on reverse)
            Default: True
        incl_one (bool): Boolean indicating whether the value 1 should
                be yielded (as either the last or first value yielded
                depending on reverse)
            Default: True

    Yields:
    2-tuples of integers (ints) representing a rational number as a
    fraction in lowest terms, where index 0 contains a non-negative integer
    representing the numerator and index 1 contains a strictly positive
    integer representing the denominator. The collection of values
    yielded represent all rational numbers greater than (or greater than or
    equal to if incl_zero is True) zero and less than (or less than or
    equal to if incl_one is True) one which when represented as a fraction
    in lowest terms have the squared sum of numerator and denominator not
    exceeding max_numerator_denominator_square_sum, which are yielded
    in strictly increasing (if reverse is False) or strictly decreasing
    (if reverse is True) order of size.

    Brief outline of rationale:
    The terms are calculated in a manner similar to that used to
    calculate Farey sequences, by starting with the sequence consisting
    of the two fractions 0 / 1 and 1 / 1 and between each pair of
    successive terms in the sequence placing the fraction with numerator
    and denominator being the sums of the numerators and the denominators
    respecively of the fractions between which it is being placed, until
    the squared sum of these values exceeds the allowed maximum.
    """
    
    # Using Farey sequences
    bounds = [(1, 1), (0, 1)] if reverse else [(0, 1), (1, 1)]
    curr = bounds[0]
    incl_first, incl_last = (incl_one, incl_zero) if reverse else (incl_zero, incl_one)
    if incl_first:
        yield curr
    stk = [bounds[1]]
    while stk:
        nxt = (curr[0] + stk[-1][0], curr[1] + stk[-1][1])
        if nxt[0] ** 2 + nxt[1] ** 2 <= max_numerator_denominator_square_sum:
            stk.append(nxt)
            continue
        curr = stk.pop()
        if incl_last or curr != bounds[1]:
            yield curr
    return

def latticeTrianglesContainingOriginCount(lattice_radius: int=105, incl_edge: bool=False) -> int:
    """
    Solution to Project Euler #184

    Consider the set of lattice points on a Cartesian plane (i.e. points
    that have integer x and y coordinates) that are within a distance of
    lattice_radius of the origin, inclusive if incl_edge is True else
    exclusive. This function calculates the number of distinct triangles
    that can be formed whose vertices are each from the described set of
    lattice points and which completely enclose the origin.

    A triangle completely encloses a point if and only if the point is
    inside the triangle and is not one of the vertices or on one of the
    edges of the triangle.

    Two triangles are distinct if and only if there is at least one vertex
    of one of the triangles whose Cartesian coordinates does not describe
    a vertex in the other triangle.

    Args:
        Optional named:
        lattice_radius (int): Strictly poisitve integer specifying the
                Euclidean distance in the Cartesian plane within which
                lattice points should be to be allowed to be used as
                positions for triangle vertices.
            Default: 105
        incl_edge (bool): Boolean specifying whether lattice points at
                a distance of exactly lattice_radius should be allowed
                to be used as positions for triangle vertices, with
                such vertices being allowed to be used if and only if
                this is given as True.
            Defualt: False
    
    Returns:
    Integer (int) giving the number of distinct triangles whose vertices
    are each at one of the described set of lattice points and which
    completely enclose the origin.

    Outline of rationale:
    We can partition the lattice points into:
     1) the origin
     2) those in the upper half plane (where y > 0) along with the points
        on the positive x axis (where y = 0 and x > 0)
     3) those in the lower half plane (where y < 0) along with the points
        on the negative x axis (where y = 0 and x < 0)
    Given our definitions, the origin cannot be one of the vertices of a
    triangle enclosing the origin, so partition 1 can be ignored.
    Additionally, partitions 2 and 3 can be mapped onto each other by a
    rotation by pi about the origin.
    For any given triangle enclosing the origin, either two or more of the
    vertices are in partition 2 or two or more of the vertices are in
    partition 3. Given the symmetry of these partitions, the total number
    of triangles enclosing the origin in each of these two cases is the
    same, and so we only need to calculate the number of such triangles
    for one of these cases and (since there is no overlap in the cases)
    the final answer is simply double that count. We choose the case where
    at least two of the vertices are in partition 2.
    Now suppose that two vertices of the triangle have been placed in
    partition 2 and consider where the third vertex can be placed such that
    the triangle encloses the origin. If we imagine looking from the third
    vertex towards the origin, we find that the triangle encloses the origin
    if and only if from this perspective one of the points is strictly to
    the left of the origin and the other is strictly to the right of the
    origin. The points for which this is the case can be constructed as
    follows.
    Construct a line from each of the two points in partition 2 that pass
    through the origin and extend to infinity, forming a wedge in the lower
    half plane with its tip at the origin. The points which those two vertices
    appear strictly to the left and strictly to the right of the origin
    respectively are precisely those points within that wedge, not including
    its borders.
    We now observe that by symmetry, the number of lattice points within
    that wedge is equal to the number of lattice points in the wedge when
    rotated by pi about the origin. This is the number of lattice points in
    partition 2 whose position vector with respect to the origin makes an
    angle strictly between that made by the position vectors of the two
    other vertices.
    Consequently, for each pair of lattice points in partition 2 comprising
    two vertices in a triangle, the number of possible lattice points for the
    third vertex resulting in a triangle enclosing the origin is the number
    of valid lattice points whose position vector from the origin makes an
    angle strictly between that of the two chosen lattice points.
    
    This implies that we can find the number of triangles for which two
    of the vertices are in partition 2 as follows:
    Iterate over all the angles from 0 (inclusive) to pi (exclusive) in
    order of increasing size for the angles at which there is at least one
    lattice point at that angle to the x-axis from the origin. During this
    process we maintain three counts:
     1) The number of lattice points encountered (this corresponds to the
        number of lattice points in partition 2 with an angle no greater
        than the current angle)
     2) The sum over all encountered lattice points of the number of other
        lattice points encountered with angle strictly less than that lattice
        point (this corresponds to the number of pairs of lattice points
        that form two vertices, one of which has an anticlockwise angle from
        the x-axis about the origin no greater than this angle, another has
        an anticlockwise angle from the x-axis about the origin between pi
        and pi plus this angle inclusive, which all form triangles if the
        third vertex has an anticlockwise angle from the x-axis about the
        origin between the current angle and pi exclusive).
     3) The sum over all encountered lattice points of the number of unordered
        pairs of lattice points encountered with angles strictly less than
        that lattice point and angles different from each other (this
        corresponds to the number of triangles containing the origin for
        which two of the lattice points make an anticlockwise angle about the
        x-axis no greater than this angle).
    Note that count after the final iteration, the final count 3 corresponds
    to the number of triangles enclosing the origin for which at least two
    vertices are in partition 2, which as previously observed is half the
    final answer.
    These counts can be maintained as follows. Each count starts at 0. For
    each angle, find the total number of lattice points at that angle,
    which we refer to as the degeneracy of that angle. Mutliply the degeneracy
    by the current count 2 (which is the count 2 for the previous angle), and
    add this to count 3. Similarly, multiply the degeneracy by the current
    count 1 (which is similarly the count 1 for the previous angle), and add
    this to count 2. Finally, add the degeneracy to count 1. After this we
    can proceed to the next angle and repeat this process until the angle
    pi is reached, before which the process is terminated.
    All that remains is to establish the angles between 0 and pi anticlockwise
    from the x-axis correspond to lattice points and how many.
    For this, we split the set of angles into the ranges [0, pi / 4),
    [pi / 4, pi / 2), [pi / 2, 3 * pi / 4) and [3 * pi / 4, pi). Recall that
    for all lattice points eligible to be triangle vertices, (x, y), we have
    x ** 2 + y ** 2 is less than (or equal to if incl_edge is True)
    lattice_radius ** 2.
    For the first range, we additionally require y >= 0 and x > y.
    As such, we can map (x, y) onto the rational numbers smaller than 1 through
    the function f(x, y) = y / x. Note that the angle with the x-axis is given
    by the principal value of arctan(y / x) which is strictly increasing. This
    also implies that all lattice points for which f(x, y) is the same have equal
    angle with the x-axis. We can therefore find the lattice points by iterating
    over the rational numbers in increasing order between 0 (inclusive) and 1
    (exclusive) for which when expressed as a fraction in lowest terms, the sum
    of the squared numerator and denominator is less than (or equal to)
    lattice_radius ** 2. Furthermore, the degeneracy of the angle corresponding
    to each such rational number will be the number of ways it can be represented
    as a fraction such that the squared numerator and denominator is less than
    (or equal to) lattice_radius ** 2, which is simply the largest integer a
    such that a ** 2 * (squared sum of squared numerator and denominator in lowest
    terms) is less than (or equal to) lattice_radius ** 2, which in the less than
    case is:
      floor(sqrt(lattice_radius ** 2 / (sum of squared numerator and denominator in lowest terms)))
    and in the less than or equal to case is:
      floor(sqrt((lattice_radius ** 2 - 1) / (sum of squared numerator and denominator in lowest terms)))
    The sequence of such rational numbers from 0 to 1 is generated by the
    generator orderedFractionsWithMaxNumeratorDenominatorSquareSum() in a manner
    similar to the calculation of Farey sequences (see the outline of rationale
    of that generator).
    The other ranges can be iterated over similarly, with f(x, y) = -y / x for
    the range [3 * pi / 4, pi) (resulting in the angle being the principal value
    of -arctan(f(x, y)) which is strictly decreasing with increasing f(x, y), so
    we iterate over the same rational numbers from 1 to 0 instead of from 0 to 1),
    f(x, y) = x / y for the range [pi / 4, pi / 2) (resulting in the angle being the
    principal value of pi / 2 - arctan(f(x, y)), which is strictly decreasing with
    increasing f(x, y) and so again iterating over the same rational numbers from
    1 to 0) and finally f(x, y) = -x / y for the range [pi / 2, 3 * pi / 4)
    (resulting in the angle being the principal value of pi / 2 + arctan(f(x, y))),
    which is strictly increasing with increasing f(x, y) and so iterating over the
    same rational numbers from 0 to 1 like with the range [0, pi / 4].
    The rational numbers for all of these four ranges can be generated in the
    correct order by the generator orderedFractionsWithMaxNumeratorDenominatorSquareSum()
    for the appropriate inputs, and so we can chain these ranges together in
    order to find the rational numbers corresponding to each angle at which there
    is at least one lattice point at that angle anticlockwise from the x-axis
    in order of increasing angle. Using each fraction thus obtained, we can
    calculate the degeneracy and finally use these values in the process
    detailed above, maintaining the three counts through the iterations to
    finally get the total number of triangles with vertices on permitted
    lattice points that surround the origin for which at least two of the
    vertices are in partition 2 of the lattice points, which can be doubled
    to obtain the final answer.
    """
    # Review- Outline of rationale for clarity and verbosity.
    since = time.time()
    r_sq = lattice_radius ** 2
    cnt1 = 0
    cnt2 = 0
    res = 0
    it0 = lambda b1, b2, b3: orderedFractionsWithMaxNumeratorDenominatorSquareSum(r_sq - (not incl_edge), reverse=b1, incl_zero=b2, incl_one=b3)
    for it in (it0(False, True, False), it0(True, False, True), it0(False, True, False), it0(True, False, True)):
        for frac in it:
            #print(frac)
            l_sq = frac[0] ** 2 + frac[1] ** 2
            degen = isqrt((r_sq - (not incl_edge)) // l_sq)
            #print(frac, degen)
            res += cnt2 * degen
            cnt2 += cnt1 * degen
            cnt1 += degen

    res <<= 1
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res


# Problem 185
def numberMindSimulatedAnnealing(alphabet: str="0123456789", n_trials: int=20, guesses: List[Tuple[str, int]]=[
    ("5616185650518293", 2),
    ("3847439647293047", 1),
    ("5855462940810587", 3),
    ("9742855507068353", 3),
    ("4296849643607543", 3),
    ("3174248439465858", 1),
    ("4513559094146117", 2),
    ("7890971548908067", 3),
    ("8157356344118483", 1),
    ("2615250744386899", 2),
    ("8690095851526254", 3),
    ("6375711915077050", 1),
    ("6913859173121360", 1),
    ("6442889055042768", 2),
    ("2321386104303845", 0),
    ("2326509471271448", 2),
    ("5251583379644322", 2),
    ("1748270476758276", 3),
    ("4895722652190306", 1),
    ("3041631117224635", 3),
    ("1841236454324589", 3),
    ("2659862637316867", 2),
]) -> List[str]:
    """
    Solution to Project Euler #185

    Using a given set of characters (alphabet), a string of a given length has
    been constructed. A sequence of similarly constructed strings are given
    (each with the same length and using the same set of characters), each having
    an associated number corresponding to the number of positions in the string
    for which the character in that position in the string is the same as the
    character at the equivalent position in the original string (referred to
    as the number of matches).

    This function produces a string which is could have been the original string
    based on that sequence of similarly constructed strings with the associated
    number.

    This is performed using simulated annealing, using as a score the sum of
    the absolute difference between the actual number of matches and the required
    number of matches for each string (referret to as the score), with at each
    step making at most n_trials attempts at random single character changes at
    each position to improve the score before making a random single character
    change regardless of its effect on the score.

    Args:
        Optional named:
        alphabet (str): A string containing the characters that may be used
                in the original string.
            Default: "0123456789"
        n_trials (int): Strictly positive integer giving the number of attempts
                made during the simulated annealing to make a change that improves
                the score before making a random single character change regardless
                of its effect on the score.
            Default: 20
        guesses (list of 2-tuples whose index 0 contains a string and whose index
                1 contains an int): The sequence of strings with the same length
                as the original string and consisting of the characters in
                alphabet, each with the corresponding number of matches, with
                which the returned string must be consistent for all of the
                contained strings and number of matches.
            Default: [
                        ("5616185650518293", 2),
                        ("3847439647293047", 1),
                        ("5855462940810587", 3),
                        ("9742855507068353", 3),
                        ("4296849643607543", 3),
                        ("3174248439465858", 1),
                        ("4513559094146117", 2),
                        ("7890971548908067", 3),
                        ("8157356344118483", 1),
                        ("2615250744386899", 2),
                        ("8690095851526254", 3),
                        ("6375711915077050", 1),
                        ("6913859173121360", 1),
                        ("6442889055042768", 2),
                        ("2321386104303845", 0),
                        ("2326509471271448", 2),
                        ("5251583379644322", 2),
                        ("1748270476758276", 3),
                        ("4895722652190306", 1),
                        ("3041631117224635", 3),
                        ("1841236454324589", 3),
                        ("2659862637316867", 2),
                    ]
    
    Returns:
    A string (str) consisting of characters in alphabet, with the string
    having the same length as each of the strings in guesses that is
    consistent with all of the number of matches with all of the
    strings in guesses. Note that while this may be the original string,
    it is not guaranteed.
    """
    since = time.time()
    n = len(guesses[0][0])
    #print(n)
    alpha_dict = {l: i for i, l in enumerate(alphabet)}

    def encode(l: str) -> int:
        return alpha_dict[l]

    def decode(num: int) -> str:
        return alphabet[num]
    
    guess_enc = []
    for s, g in guesses:
        guess_enc.append(([encode(l) for l in s], g))

    def dist(s: str) -> int:
        res = 0
        for s2, g in guess_enc:
            #print(s, s2)
            #print(s, s2, sum(num1 == num2 for num1, num2 in zip(s, s2)))
            res += abs(sum(num1 == num2 for num1, num2 in zip(s, s2)) - g)
        #print(s, res)
        return res

    def randomStep(curr: List[int]) -> List[int]:
        idx = random.randrange(0, n)
        num = random.randrange(0, len(alphabet) - 1)
        num += (num >= curr[idx])
        res = list(curr)
        res[idx] = num
        return res

    curr = [random.randrange(0, len(alphabet)) for _ in range(n)]
    d = dist(curr)
    #print(d)
    #if not d: return "".join(decode(x) for x in curr)
    best = float("inf")
    while d:
        #print(d, curr)
        for _ in range(n_trials):
            improved = False
            order = list(range(n))
            random.shuffle(order)
            #order = random.shuffle(range(n))
            for i in order:
                prev = curr[i]
                num = random.randrange(0, len(alphabet) - 1)
                num += num >= curr[i]
                curr[i] = num
                d2 = dist(curr)
                if d2 < d:
                    d = d2
                    improved = True
                else: curr[i] = prev
            if improved:
                if d < best:
                    best = d
                    print(f"new best = {best}")
                break
        else:
            curr = randomStep(curr)
            d = dist(curr)
            if d < best:
                best = d
                print(f"new best = {d}")
    
    res = "".join(decode(x) for x in curr)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

def numberMindExact(alphabet: str="0123456789", guesses: List[Tuple[str, int]]=[
    ("5616185650518293", 2),
    ("3847439647293047", 1),
    ("5855462940810587", 3),
    ("9742855507068353", 3),
    ("4296849643607543", 3),
    ("3174248439465858", 1),
    ("4513559094146117", 2),
    ("7890971548908067", 3),
    ("8157356344118483", 1),
    ("2615250744386899", 2),
    ("8690095851526254", 3),
    ("6375711915077050", 1),
    ("6913859173121360", 1),
    ("6442889055042768", 2),
    ("2321386104303845", 0),
    ("2326509471271448", 2),
    ("5251583379644322", 2),
    ("1748270476758276", 3),
    ("4895722652190306", 1),
    ("3041631117224635", 3),
    ("1841236454324589", 3),
    ("2659862637316867", 2),
]) -> List[str]:
    """
    Alternative solution to Project Euler #185, giving all possible solutions
    rather than just one possible solution.

    Using a given set of characters (alphabet), a string of a given length has
    been constructed. A sequence of similarly constructed strings are given
    (each with the same length and using the same set of characters), each having
    an associated number corresponding to the number of positions in the string
    for which the character in that position in the string is the same as the
    character at the equivalent position in the original string (referred to
    as the number of matches).

    This function produces a list of all of the strings that could have been the
    original string based on that sequence of similarly constructed strings with
    the associated number.

    This is currently not a viable solution for the specific inputs of Project
    Euler #185 (the default inputs) as it has too large a search space.

    Args:
        Optional named:
        alphabet (str): A string containing the characters that may be used
                in the original string.
            Default: "0123456789"
        guesses (list of 2-tuples whose index 0 contains a string and whose index
                1 contains an int): The sequence of strings with the same length
                as the original string and consisting of the characters in
                alphabet, each with the corresponding number of matches, with
                which the returned string must be consistent for all of the
                contained strings and number of matches.
            Default: [
                        ("5616185650518293", 2),
                        ("3847439647293047", 1),
                        ("5855462940810587", 3),
                        ("9742855507068353", 3),
                        ("4296849643607543", 3),
                        ("3174248439465858", 1),
                        ("4513559094146117", 2),
                        ("7890971548908067", 3),
                        ("8157356344118483", 1),
                        ("2615250744386899", 2),
                        ("8690095851526254", 3),
                        ("6375711915077050", 1),
                        ("6913859173121360", 1),
                        ("6442889055042768", 2),
                        ("2321386104303845", 0),
                        ("2326509471271448", 2),
                        ("5251583379644322", 2),
                        ("1748270476758276", 3),
                        ("4895722652190306", 1),
                        ("3041631117224635", 3),
                        ("1841236454324589", 3),
                        ("2659862637316867", 2),
                    ]
    
    Returns:
    List of strings (str) with each string consisting of characters in
    alphabet, of the same length as each of the strings in guesses and 
    is consistent with all of the number of matches with all of the
    strings in guesses. All such strings are included exactly once, and as
    such the original string used to give the number of matches for all
    of the strings in guesses is guaranteed to be in this list.
    """
    # Review- try finding ways to prune the search space to 
    # make this method viable

    since = time.time()
    n = len(guesses[0][0])

    f_sets = [set() for _ in range(max(x[1] for x in guesses) + 1)]
    l_dicts = [{} for _ in range(n)]
    f_lst = []
    for i, (s, f) in enumerate(guesses):
        f_sets[f].add(i)
        for idx in range(n):
            l = s[idx]
            l_dicts[idx].setdefault(l, [])
            l_dicts[idx][l].append(i)
        f_lst.append(f)
    #print(f_sets)
    #print(l_dicts)
    
    curr = []
    res = []

    def recur(idx: int=n - 1) -> None:
        if idx < 0:
            res.append("".join(curr[::-1]))
            return
        if len(f_sets) > idx + 1 and f_sets[idx + 1]:
            opts = {guesses[i][0][idx] for i in f_sets[idx + 1]}
            if len(opts) > 1: return
        else: opts = set(alphabet)
        if f_sets:
            for i in f_sets[0]:
                opts.discard(guesses[i][0][idx])
        if not opts: return
        curr.append(None)
        #print(curr, opts)
        for l in opts:
            curr[-1] = l
            shifted = []
            for i in l_dicts[idx].get(l, []):
                f = f_lst[i]
                f_sets[f].remove(i)
                f_sets[f - 1].add(i)
                shifted.append(i)
                f_lst[i] -= 1
            recur(idx - 1)
            for i in shifted:
                f = f_lst[i]
                f_lst[i] += 1
                f_sets[f].remove(i)
                f_sets[f + 1].add(i)
        curr.pop()
        return
    
    recur(idx=n - 1)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 186
class UnionFind:
    def __init__(self, n: int):
        self.n = n
        self.root = list(range(n))
        self.rank = [1] * n
        self.size = [1] * n
    
    def find(self, i: int) -> int:
        r = self.root[i]
        if r == i: return i
        res = self.find(r)
        self.root[i] = res
        return res
    
    def union(self, i1: int, i2: int) -> None:
        r1, r2 = list(map(self.find, (i1, i2)))
        if r1 == r2: return
        d = self.rank[r1] - self.rank[r2]
        if d < 0: r1, r2 = r2, r1
        elif not d: self.rank[r1] += 1
        self.root[r2] = r1
        self.size[r1] += self.size[r2]
        self.size[r2] = 0
        return
    
    def connected(self, i1: int, i2: int) -> bool:
        return self.find(i1) == self.find(i2)

    def connectedComponentSize(self, i: int) -> int:
        return self.size[self.find(i)]

def edgeCountForVertexToConnectToProportionOfGraph(
    n_vertices: int,
    vertex: int,
    target_proportion: float,
    edge_iterator: Iterable[Tuple[int, int]],
    ignore_self_edges: bool=True,
) -> int:
    """
    For an undirected graph initially consisting of n_vertices and no
    edges, where the vertices are labelled with the integers from 0 to
    (n_vertices - 1) inclusive, undirected edges between the vertices
    are inserted one by one as provided by the iterator edge_iterator,
    which produces an ordered sequence of edges in the form of 2-tuples
    with the labels of the vertices the edge should connect.

    This function calculates the smallest number of edges to be
    inserted before the vertex with integer label vertex is connected
    to a proportion of the vertices in the graph (including itself)
    no less than target_proportion.

    In an undirected graph, a vertex is connected to another vertex if
    and only if there exists an unbroken path using edges of the
    graph from the first vertex to the second.

    Args:
        Required positional:
        n_vertices (int): Strictly positive integer giving the number
                of vertices in the graph.
        vertex (int): Integer between 0 and (n_vertices - 1) inclusive
                specifying for which vertex the proportion of vertices
                in the graph is being monitored as edges are inserted.
        target_proportion (float): The proportion of the vertices in
                the graph the vertex labelled vertex is to be connected
                when the number of edges inserted is to be returned
                (including the vertex itself).
        edge_iterator (iterator of 2-tuples of ints): Ordered iterator
                producing a finite or infinite number of 2-tuples of
                ints, both of which are between 0 and (n_vertices - 1)
                inclusive, each corresponding to an edge connecting
                the vertices labelled with the two contained integers.

        Optional named:
        ignore_self_edges (bool): Boolean which if True signifies that
                if an edge produced by the iterator edge_iterator
                connects a vertex to itself, it should ignored (and so
                that edge is not included in the count of edges),
                but if False such edges are still included.
            Default: True

    Returns:
    Integer (int) giving the number of edges inserted into the graph by
    the iterator edge_iterator (if ignore_self_edges is True, not including
    edges connecting an edge to itself) before the vertex labelled with the
    integer vertex is first connected to a proportion of the vertices in the
    graph (including itself) no less than target_proportion. If edge_iterator
    produces a finite number of edges and even with all the edges in this
    iterator are included that proportion of the graph is never reached then
    -1 is returned.
            
    Outline of rationale:
    This is a simple application of the Union Find (or Disjoint Set Union)
    data structure with path compression and union by rank to efficiently
    keep track of the connected components of the graph (i.e. the partitions
    of the vertices in which every vertex is connected to every other
    vertex within a partition and is connected to no vertex in another
    partition) as the edges are added.
    """
    target = math.floor(n_vertices * target_proportion)
    if target <= 1: return 0
    uf = UnionFind(n_vertices)
    prev_sz = 1
    self_edge_count = 0
    for i, e in enumerate(edge_iterator):
        if e[0] == e[1]:
            self_edge_count += 1
        uf.union(*e)
        sz = uf.connectedComponentSize(vertex)
        if sz > prev_sz:
            #print(i, e, sz, target)
            prev_sz = sz
        if sz >= target:
            return i + 1 - (self_edge_count if ignore_self_edges else 0)
    return -1

def generalisedLaggedFibonacciGenerator(poly_coeffs: Tuple[int]=(100003, -200003, 0, 300007), lags: Tuple[int]=(24, 55), min_val: int=0, max_val: int=10 ** 6 - 1) -> Generator[int, None, None]:
    """
    Generator iterating over the terms in a generalisation
    of a lagged Fibonacci generator sequence for given for a
    given initial polynomial and given lag lengths within n a
    given range.

    The generalisation of the lagged Fibonacci generator
    sequence for the given tuple of integers poly_coeffs, tuple
    of strictly positive integers lags, and the integers min_val
    and max_val is the sequence such that for integer i >= 1,
    the i:th term in the sequence is:
        t_i = (sum j from 0 to len(poly_coeffs) - 1) (poly_coeffs[j] * i ** j) % md + min_val
                for i <= max(lags)
              ((sum j fro 0 to len(lags) - 1) (t_(i - lags[i]))) % md + min_val
                otherwise
    where md is one greater than the difference between
    min_value and max_value and % signifies modular division
    (i.e. the remainder of the integer preceding that symbol
    by the integer succeeding it). This sequence contains integer
    values between min_value and max_value inclusive.

    The terms where i <= max(lags) are referred as the polynomial
    terms and the terms where i > max(lags) are referred to as the
    recursive terms.

    In the case that lags is length 2 with those two elements
    distinct, this is a traditional lagged Fibonacci generator
    sequence.

    For well chosen values of poly_coeffs and lags for given
    min_value and max_value, this can potentially be used as a
    generator of pseudo-random integers between min-value and
    max_value inclusive.

    Note that the generator never terminates and thus any
    iterator over this generator must include provision to
    terminate (e.g. a break or return statement), otherwise
    it would result in an infinite loop.

    Args:
        Optional named:
        poly_coeffs (tuple of ints): Tuple of integers giving
                the coefficients of the polynomial used to
                generate the polynomial terms.
            Default: (100003, -200003, 0, 300007)
        lags (tuple of ints): Strictly positive integers,
                which when generating the recursive terms,
                indicates how many steps back in the sequence
                the previous terms summed should each be
                from the position of the term being generated.
                Additionally, the maximum value determines
                at which term the transition from the polynomial
                terms to the recursive terms will occur.
            Default: (24, 55)
        min_value (int): Integer giving the smallest value
                possible for terms in the sequence.
        max_value (int): Integer giving the largest value
                possible for terms in the sequence. Must
                be no smaller than min_value.
    
    Yields:
    Integer (int) between min_value and max_value inclusive,
    with the i:th term yielded (for strictly positive integer
    i) representing the i:th term in the generalisation of
    the lagged Fibonacci generator defined above for the
    given parameters.
    """
    qu = deque()
    md = max_val - min_val + 1
    #print(md)
    lags = sorted(lags)
    max_lag = lags[-1]
    for k in range(1, max_lag + 1):
        num = (sum(c * k ** i for i, c in enumerate(poly_coeffs)) % md) + min_val
        #num = ((100003 - 200003 * k + 300007 * k ** 3) % md) - 5 * 10 ** 5
        #print(num)
        qu.append(num)
        yield num
    #cnt = 0
    while True:
        num = qu.popleft()
        for i in range(len(lags) - 1):
            num += qu[-lags[i]]
        num = (num % md) + min_val
        #if cnt < 10:
        #    print(num)
        #cnt += 1
        #num = ((qu[-24] + qu.popleft() + 10 ** 6) % md) - 5 * 10 ** 5
        qu.append(num)
        yield num
    return

def laggedFibonacciGraphEdgeGenerator(
    n_vertices: int=10 ** 6,
    n_edges: Optional[int]=None,
    l_fib_poly_coeffs: Tuple[int]=(100003, -200003, 0, 300007),
    l_fib_lags: Tuple[int]=(24, 55),
) -> Generator[Tuple[int, int], None, None]:
    """
    Generator yielding edges in an unweighted undirected graph consisting
    of n_vertices vertices labelled with the integers from 0 to
    (n_vertices - 1) inclusive based on the terms of a generalisation of
    a lagged Fibonacci generator sequence for given polynomial coefficients
    and lags. If n_edges is specified as a non-negative integer, this gives
    the number of edges yielded, otherwise the iterator does not of itself
    terminate.

    Each pair of consecutive values in the generalisation of a lagged Fibonacci
    generator sequence with the specified parameters represents an edge by the
    integer labels of the vertices the edge connects.

    The generalisation of the lagged Fibonacci generator sequence for the
    given tuple of integers l_fib_poly_coeffs, tuple of strictly positive integers
    l_fib_lags, and the integers min_val and max_val is the sequence such that
    for integer i >= 1, the i:th term in the sequence is:
        t_i = (sum j from 0 to len(l_fib_poly_coeffs) - 1) (l_fib_poly_coeffs[j] * i ** j) % n_vertices
                for i <= max(l_fib_lags)
              ((sum j fro 0 to len(l_fib_lags) - 1) (t_(i - l_fib_lags[i]))) % n_vertices
                otherwise
    where % signifies modular division (i.e. the remainder of the integer
    preceding that symbol by the integer succeeding it). This sequence contains
    integer values between 0 and (n_vertices - 1) inclusive.

    The terms where i <= max(l_fib_lags) are referred as the polynomial
    terms and the terms where i > max(l_fib_lags) are referred to as the
    recursive terms.

    Note that if n_edges is not specified, the generator never terminates and
    thus any iterator over this generator must include provision to terminate
    (e.g. a break or return statement), otherwise it would result in an infinite
    loop.

    Args:
        Optional named:
        n_vertices (int): Strictly positive integer giving the number of
                vertices in the graph whose edges are to be generated.
        n_edges (int or None): If given as a non-negative integer, the
                number of edges the generator should yield before
                terminating, otherwise the generator does not of itself
                terminate
        l_fib_poly_coeffs (tuple of ints): Tuple of integers giving the
                coefficients of the polynomial used to calculate the
                polynomial terms of the generalisation of the lagged
                Fibonacci generator sequence used to generate the
                edges.
            Default: (100003, -200003, 0, 300007)
        l_fib_lags (tuple of ints): Tuple of strictly positive integers,
                which when calculating the recursive terms of the
                generlisation of the lagged Fibonacci generator sequence
                used to generate the edges, indicates how many steps back
                in the sequence the previous terms summed should each be
                from the position of the term being generated. Additionally,
                the maximum value determines at which term the transition
                from the polynomial terms to the recursive terms will occur
                in this sequence.
            Default: (24, 55)
    
    Yields:
    2-tuple of ints between 0 and (n_vertices - 1) inclusive representing
    an edge of the graph, where the two integers give the labels of the
    two vertices the edge connects, based on the generalisation of the
    given generalisation of the lagged Fibonacci generator sequence as
    specified above.
    If n_edges is specified as a non-negative integer then (unless
    externally terminated first) exactly n_edges such values are yielded,
    otherwise the generator never of itself terminates.
    """
    it = generalisedLaggedFibonacciGenerator(poly_coeffs=l_fib_poly_coeffs, lags=l_fib_lags, min_val=0, max_val=n_vertices - 1)
    it2 = range(n_edges) if isinstance(n_edges, int) and n_edges >= 0 else itertools.count(0)
    for _ in it2:
        edge = []
        edge.append(next(it))
        edge.append(next(it))
        yield tuple(edge)
    return 

def laggedFibonacciGraphEdgeCountForVertexToConnectToProportionOfGraph(
    n_vertices: int=10 ** 6,
    vertex: int=524287,
    target_proportion: float=.99,
    l_fib_poly_coeffs: Tuple[int]=(100003, -200003, 0, 300007),
    l_fib_lags: Tuple[int]=(24, 55),
    ignore_self_edges: bool=True,
) -> int:
    """
    Solution to Project Euler #186

    For an undirected graph initially consisting of n_vertices and no
    edges, where the vertices are labelled with the integers from 0 to
    (n_vertices - 1) inclusive, undirected edges between the vertices
    are inserted one by one as generated by a lagged Fibonacci generator
    sequence as per the generator laggedFibonacciGraphEdgeGenerator()
    for the given parameters, which produces an ordered sequence of edges
    in the form of 2-tuples with the labels of the vertices the edge
    should connect.

    This function calculates the smallest number of edges to be
    inserted before the vertex with integer label vertex is connected
    to a proportion of the vertices in the graph (including itself)
    no less than target_proportion.

    In an undirected graph, a vertex is connected to another vertex if
    and only if there exists an unbroken path using edges of the
    graph from the first vertex to the second.

    For details regarding the edge generation, see the documentation
    of laggedFibonacciGraphEdgeGenerator(), from which (for the given
    polynomial coefficients and lags) the ordered sequence of edges is
    produced.

    Args:
        Optional named:
        n_vertices (int): Strictly positive integer giving the number
                of vertices in the graph.
            Default: 10 ** 6
        vertex (int): Integer between 0 and (n_vertices - 1) inclusive
                specifying for which vertex the proportion of vertices
                in the graph is being monitored as edges are inserted.
            Default: 524287
        target_proportion (float): The proportion of the vertices in
                the graph the vertex labelled vertex is to be connected
                when the number of edges inserted is to be returned
                (including the vertex itself).
            Default: 0.99
        l_fib_poly_coeffs (tuple of ints): Tuple of integers giving the
                coefficients of the polynomial used to generate the
                terms of the generalisation of the lagged Fibonacci
                generator sequence used to generate the edges.
            Default: (100003, -200003, 0, 300007)
        l_fib_lags (tuple of ints): Tuple of strictly positive integers,
                which when calculating the terms of the generalisation of
                the lagged Fibonacci generator sequence used to generate
                the edges.
            Default: (24, 55)

        Optional named:
        ignore_self_edges (bool): Boolean which if True signifies that
                if an edge produced by the generator
                laggedFibonacciGraphEdgeGenerator() connects a vertex to
                itself, it should ignored (and so that edge is not included
                in the count of edges), but if False such edges are still
                included.
            Default: True

    Returns:
    Integer (int) giving the number of edges inserted in into the graph by
    the generator laggedFibonacciGraphEdgeGenerator() for the given parameters
    (if ignore_self_edges is True, not including edges connecting an edge to
    itself) before the vertex labelled with the integer vertex is first
    connected to a proportion of the vertices in the graph (including itself)
    no less than target_proportion.
    """
    since = time.time()
    mx_n_edges = None
    res = edgeCountForVertexToConnectToProportionOfGraph(
            n_vertices,
            vertex,
            target_proportion,
            laggedFibonacciGraphEdgeGenerator(n_vertices=n_vertices, n_edges=mx_n_edges, l_fib_poly_coeffs=l_fib_poly_coeffs, l_fib_lags=l_fib_lags),
            ignore_self_edges=ignore_self_edges,
        )
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
    

# Problem 187
def semiPrimeCount(n_max: int=10 ** 8 - 1) -> int:
    """
    Finds the number of strictly positive integers no greater
    than n_max that are semi-prime.

    A semi-prime is a strictly positive integer that is the
    product of exactly two prime numbers, where the two
    primes are not necessarily distinct.

    Args:
        Optional named:
        n_max (int): The largest integer whose status as
                a semi-prime is considered in the count.
            Default: 10 ** 8 - 1
    
    Returns:
    Integer (int) giving the number of strictly positive
    integers no greater than n_max that are semi-prime.
    """
    since = time.time()
    pf = SimplePrimeSieve(n_max >> 1)
    p_mx = isqrt(n_max)
    i2 = len(pf.p_lst) - 1
    #print(pf.p_lst)
    res = 0
    for i1, p1 in enumerate(pf.p_lst):
        if p1 > p_mx: break
        for i2 in reversed(range(i1, i2 + 1)):
            p2 = pf.p_lst[i2]
            if p1 * p2 <= n_max:
                res += i2 - i1 + 1
                break
        else: break
        #print(p1, res)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 188
def primeFactorisation(n: int) -> Dict[int, int]:
    """
    Finds the prime factorisation of the strictly positive integer
    n.

    Args:
        Required positional:
        n (int): Strictly positive integer whose prime factorisation
                is to be found.
    
    Returns:
    Dictionary (dict) whose keys are all the prime numbers
    that divide n with corresponding value as the number of
    times that prime divides n (i.e. the power of that prime in the
    prime factorisation of n).
    """
    res = {}
    if n < 2: return {}
    n2 = n
    if not n2 & 1:
        res[2] = 1
        n2 >>= 1
        while not n2 & 1:
            n2 >>= 1
            res[2] += 1
    for p in range(3, isqrt(n2), 2):
        if p > isqrt(n2): break
        n3, r = divmod(n2, p)
        if r: continue
        n2 = n3
        res[p] = 1
        n3, r = divmod(n2, p)
        while not r:
            n2 = n3
            res[p] += 1
            n3, r = divmod(n2, p)
    if n2 > 1:
        res[n2] = 1
    return res

def eulerTotientFunction(n: int) -> int:
    """
    Finds the Euler totient function for the strictly positive
    integer n.

    For each strictly positive integer, the Euler totient function
    at that integer is the number of strictly positive integers
    less than that integer to which the integer is coprime (i.e.
    the greatest common denominator of the two integers is 1).
    Note that for a prime number p, the Euler totient function
    for p is (p - 1).

    Given that the Euler totient function is multiplicative (i.e.
    the value of the function for the product of two coprime
    integers is equal to the product of the value of the Euler
    totient function for those two integers) and the Euler
    totient function for a prime p to the power of a positive
    integer k is (p - 1) * p ** (k - 1), if n has the prime
    factorisation:
        n = p_1 ** k_1 * p_2 ** k_2 * ...
    it follows that the Euler totient function for n is:
        (p_1 - 1) * p_1 ** (k_1 - 1) * (p_2 - 1) * p_2 ** (k_2 - 1) * ...
    This is the method used by this function to calculate
    the Euler totient function for given n, using the function
    primeFactorisation() to find the prime factorisation of n.

    Args:
        Required positional:
        n (int): Strictly positive integer for which the value
                of the Euler totient function is to be found.
    
    Returns:
    Integer (int) giving the value of the Euler totient function
    (as defined above) for n.
    """
    p_fact = primeFactorisation(n)
    #print(p_fact)
    res = 1
    for p, k in p_fact.items():
        res *= (p - 1) * (p ** (k - 1))
    return res

def modPower(base: int, exp: int, md: int, e_tot_md: Optional[int]=None) -> int:
    return pow(base, exp, md)
    #if gcd(base, md) != 1: return (base ** exp) % md
    #print("hi")
    #if e_tot_md is None:
    #    e_tot_md = eulerTotientFunction(md)
    #exp %= e_tot_md
    #print(exp)
    #return (base ** exp) % md

def modTetration(base: int=1777, tetr: int=1855, md: int=10 ** 8) -> int:
    """
    Solution to Project Euler #188

    Computes the strictly positive integer base tetrated to the non-negative
    integer tetr modulo md (i.e. the remainder when base to the tetr:th
    tetration is divided by md).

    Tetration for a strictly positive integer is defined such that
    the tetration of that integer to 0 is 1 and for any positive
    integer k, the tetration of the integer to k (or equivalently the
    k:th tetration of the integer) is equal to the integer to the
    power of the tetration of the integer to (k - 1).

    Args:
        Optional named:
        base (int): Strictly positive integer to be tetrated to the
                integer tetr modulo md.
            Default: 1777
        tetr (int): Non-negative integer giving the number to which
                base is to be tetrated modulo md.
            Default: 1855
        md (int): Strictly positive integer giving the modulus to
                which the tetration of base to tetr is to be taken.

    Returns:
    Integer (int) giving the tetration of base to k modulo md.

    Outline of rationale:
    TODO (explain the reduction of each intermediate modulo the
    Euler totient function of md due to the Euler-Fermat theorem
    and the checking for cycles to potentially reduce the number of
    iterations needing to be performed)
    """
    since = time.time()
    e_tot_md = eulerTotientFunction(md)
    #print(md, e_tot_md)
    if not tetr: return 1
    res = base
    seen_dict = {1: 0, base: 1}
    seen_lst = [0, base]
    for i in range(2, tetr + 1):
        res = modPower(base, res, md, e_tot_md=e_tot_md)
        if res in seen_dict.keys():
            i0 = seen_dict[res]
            cycle_len = i - i0
            r = (tetr - i) % cycle_len
            res = seen_lst[i0 + r]
            break
        seen_lst.append(res)
        seen_dict[res] = i
        #print(i, res)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 189
def numberOfTriangularGridColourings(n_colours: int=3, n_rows: int=8) -> int:
    """
    Solution to Project Euler #189

    For a collection of equilateral triangles of the same size
    arranged in a tesselating pattern itself forming an equilateral
    triangle with n_rows rows of alternating upward pointing and
    downward pointing triangles, calculates the number of ways of
    colouring the colleciont of triangles such that up to n_colours
    different colours are used and no two adjacent triangles (i.e.
    triangles for which the edge of one lies entirely up against
    the edge of another) have the same colour.

    In this, symmetries are not taken into account (so for example,
    if one pattern can be rotated to form another pattern that was
    different from the first in its initial orientation, the two
    patterns are still considered to be different).

    Args:
        Optional named:
        n_colours (int): Strictly positive integer giving the
                maximum number of different colours that may be
                used to colour the collection of equilateral triangles
                for colourings counted in the total.
            Default: 3
        n_rows (int): Strictly positive integer giving the number of
                rows of alternating upward and downward pointing
                equilateral triangles in the overall equilateral
                triangle pattern.
            Default: 8
    
    Returns:
    Integer (int) giving the number of different possible colourings
    of the tesselating equilateral subject to the rules for allowed
    colourings stated with given n_colours and n_rows.

    Brief outline of rationale:
    This uses top-down dynamic programming with memoisation, going
    row by row from a vertex of the overall triangle pattern towards
    the opposite edge, going from one upward pointing triangle to the
    next (and so accounting for one or two triangles at a time)
    keeping track of the position in the overall pattern (by row and
    index) and the colourings of the triangles that will be adjacent
    to triangles yet to be placed, recalibrating the colour labelling
    at each step such that the colours of those triangles are labelled
    according to where they were first encountered amoung those, so
    that for instance the first triangle recorded and all others in
    recorded with the same colour are labelled 0, the next different
    colour recorded and all others with the same colour are labelled
    1 and so on. This reduces the possibility space by recognising
    different colourings that can be mapped onto each other by a
    bijection as equivalent (as it is only whether two colours are
    the same or different that matter), so enabling us to just
    calculate each equivalent colouring once and saving duplicated
    calculations.
    """
    since = time.time()
    if n_rows == 1: return n_colours
    if n_colours <= 1: return 0
    elif n_colours == 2: return n_colours
    stk = []

    memo = {}
    def recur(row_idx: int, triangle_idx: int) -> int:
        if triangle_idx == row_idx + 1:
            row_idx += 1
            if row_idx == n_rows: return 1
            triangle_idx = 0
        n_prev = max(0, row_idx + 1 - (not triangle_idx))
        colour_map = {}
        nxt = 0
        #print(row_idx, triangle_idx, n_prev, stk)
        for i in range(n_prev):
            c0 = stk[-n_prev + i]
            if c0 in colour_map.keys():
                continue
            colour_map[c0] = nxt
            nxt += 1
            if nxt == n_colours: break
        prev = tuple(colour_map[stk[-n_prev + i]] for i in range(n_prev))
        #print(prev)
        args = (row_idx, triangle_idx, prev)
        if args in memo.keys(): return memo[args]
        res = 0
        stk.append(0)
        if triangle_idx:
            c_opts = {c: 0 for c in range(n_colours)}
            for c0 in set(range(n_colours)) - {prev[0], prev[-1]}:
                for c in set(range(n_colours)) - {c0}:
                    c_opts[c] += 1
        else: c_opts = {c: 1 for c in range(n_colours)}
        #print(c_opts)
        for c, mult in c_opts.items():
            stk[-1] = c
            res += mult * recur(row_idx, triangle_idx + 1)
        stk.pop()

        memo[args] = res
        return res

    res = recur(0, 0)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res


# Problem 190
def maximisedRestrictedPowerProduct(n: int) -> Tuple[int, int]:
    """
    For the strictly postive integer n, finds the maximum value of
    the function:
        P_n(x_1, x_2, ..., x_n) = (prod i from 1 to n) x_i ** i
    where for integers 1 <= i <= n, x_i is real and:
        (sum i from 1 to n) x_i = n

    It can be shown that this value must be rational (see outline
    of rationale below), and so the answer is given as a fraction
    in lowest terms.
    
    Args:
        Required positional:
        n (int): Strictly positive integer for which the above
                function P_n(x_1, x_2, ..., x_n) should be maximised
                subject to the stated constraints on x_1, x_2, ...,
                x_n for this n.
    
    Returns:
    2-tuple of integers (int) representing the rational number that
    is the maximum value of P_n(x_1, x_2, ..., x_n) subject to the
    stated constraints on x_1, x_2, ..., x_n for the given n, given
    as a fraction in lowest terms where index 0 and 1 contain the
    numerator and denominator respectively of this fraction.

    Outline of rationale:
    Solved using Lagrange multipliers. For real values, this has the
    Lagrangian (where x is the n-dimensional vector (x_1, x_2, ..., x_n))
    and l is a real valued parameter, the Lagrange multiplier:
        L(x, l) = f(x) + l * g(x)
    where:
        f(x) = (prod i = 1 to n) x_i ** i
        g(x) = ((sum i = 1 to n) x_i) - n
    The extrema of f(x) subject to the constraint g(x) = 0 are the
    values of x of the solutions of the simultaneous equations:
        (partial d / dx_i) L(x) = 0
    for i = 1, 2, ..., n and
        (partial d / dl) L(x) = 0
    We find that:
        (partial d / dx_i) L(x) = j * f(x) / x_j + l
    and:
        (partial d / dl) L(x) = g(x)
    Therefore, the simultaneous equations become:
        f(x) = l * x_i / i
    for i = 1, 2, ..., n, and
        (sum i = 1 to n) x_i = n
    For 1 < i <= n, this gives:
        x_i / x_(i - 1) = i / (i - 1)
    From this it can be shown by induction (or intuitively by
    telescopic cancellation) that for 1 <= i <= n:
        x_i = i * x_1
    Consequently:
        (sum i = 1 to n) x_i = x_1 * (sum i = 1 to n) i
                             = x_1 * n * (n + 1) / 2
    Thus, given that this must equal n:
        x_1 = 2 / (n + 1)
    and so for 1 <= i <= n:
        x_i = 2 * i / (n + 1)
    Therefore, the maximum value of P_n(x) for the given constraint
    is:
        (prod i from 1 to n) (2 * i / (n + 1)) ** i
        = (2 / (n + 1)) ** (n * (n + 1) / 2) * (prod i from 1 to n) i ** i
    Since n and i are integers and the product contains a finite
    number of terms, this value is rational, justifying the
    returned value being given as a fraction.
    """
    def powerProduct(num):
        res = 1
        for i in range(2, num + 1):
            res *= i ** i
        return res

    numer = powerProduct(n)
    exp = ((n * (n + 1)) >> 1)
    if not n & 1:
        numer *= 2 ** exp
        denom = (n + 1) ** exp
    else:
        denom = ((n + 1) >> 1) ** exp
    g = gcd(numer, denom)
    res = (numer // g, denom // g)
    #print(res, res[0] / res[1])
    return res

def sumFloorMaximisedRestrictedPowerProduct(n_min: int=2, n_max: int=15) -> int:
    """
    Solution to Project Euler #190

    For the positive integers n, define the family of functions:
        P_n(x_1, x_2, ..., x_n) = (prod i from 1 to n) x_i ** i
    where for integers 1 <= i <= n, x_i are all real.
    
    This function for each integer n between n_min and n_max
    inclusive finds the floor of the maximum of P_n(x_1, x_2, ..., x_n)
    such that:
        (sum i from 1 to n) x_i = n
    and returns the sum of these values.
    
    Args:
        Required positional:
        n_min (int): Strictly positive integer giving the smallest
                value of n for which the floor of the maximum of the
                function P_n(x_1, x_2, ..., x_n) subject to the
                stated constraints on x_1, x_2, ..., x_n should be
                included in the sum.
            Default: 2
        n_max (int): Integer no less than n_min giving the largest
                value of n for which the floor of the maximum of the
                function P_n(x_1, x_2, ..., x_n) subject to the
                stated constraints on x_1, x_2, ..., x_n should be
                included in the sum.
    
    Returns:
    Integer (int) giving the sum of the floors of the maximum values of
    P_n(x_1, x_2, ..., x_n) where for integers 1 <= i <= n, x_i is
    real and sum to n, over the integers n from n_min to n_max
    inclusive.

    Outline of rationale:
    See outline of rationale in the documentation for the function
    maximisedRestrictedPowerProduct().
    """
    since = time.time()
    res = 0
    for i in range(n_min, n_max + 1):
        frac = maximisedRestrictedPowerProduct(i)
        res += math.floor(frac[0] / frac[1])
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 191
def attendancePrizeStringCount(n_days: int=30, n_consec_absent: int=3, n_late: int=2) -> int:
    """
    Solution to Project Euler #191

    Over a period of n_days, count the number of ways of achieving
    a daily attendance record with no consecutive absences of
    n_consec_absent and no more than n_late total lates, where
    the possible outcomes each day are present, absent or late.

    Two attendance records over n_days are distinct if and only
    if the attendance marked on at least one of the days is
    different between the two records.

    Args:
        Optional named:
        n_days (int): The total number of days of the attendance
                record.
            Default: 30
        n_consec_absent (int): The number of consecutive absences
                in an attendance record that will result in that
                attendance record not being counted in the total.
            Default: 3
        n_late (int): The total number of lates in an attendance
                record that will result in the that attendance
                record not being counted in the total.
            Default: 2
    
    Returns:
    The total number of distinct attendance records possible over
    n_days subject to the given constraints.
    
    Brief outline of rationale:
    Solved using bottom-up dynamic programming, iterating over the
    days in increasing order, at each stage finding the number
    of ways of after that day having each non-negative integer up to
    (n_consec_absent - 1) consecutive absences up to the current day
    without having any previous consecutive n_consec_absent days
    absent and each non-negative integer up to (n_late - 1) total
    days late, based on these counts for the previous day.
    The solution is then the total of these numbers for day number
    n_days.
    """
    since = time.time()
    curr = [[0] * n_consec_absent for _ in range(n_late)]
    curr[0][0] = 1
    for _ in range(n_days):
        prev = curr
        curr = [[0] * n_consec_absent for _ in range(n_late)]
        for i_a in range(1, n_consec_absent):
            for i_l in range(n_late):
                curr[i_l][i_a] = prev[i_l][i_a - 1]
        for i_l in range(n_late - 1):
            sm = sum(prev[i_l])
            curr[i_l][0] += sm
            curr[i_l + 1][0] += sm
        i_l = n_late - 1
        curr[i_l][0] += sum(prev[i_l])
    res = sum(sum(x) for x in curr)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 192
def bestSqrtApproximationsDenominatorSum(n_max: int=10 ** 5, denom_bound: int=10 ** 12) -> int:
    """
    Finds the sum of the denominators of the best rational approximations
    for the denominator bound denom_bound of the square roots of all
    non-square integers between 1 and n_max inclusive.

    The best rational approximation of a non-negative number num
    for a given denominator bound denom_bound is the unique fraction
    p / q (where p and q are integers) such that q < denom_bound and
    for any integers p2, q2:
        if abs(num - p2 / q2) < abs(num - p / q)
        then q2 > denom_bound.
    
    Args:
        Optional named:
        n_max (int): Non-negative integer such that the denominator of
                the best rational approximation of no integers larger
                than this are included in the denominator sum.
            Default: 10 ** 5
        denom_bound (int): Strictly positive integer giving the maximum
                value of denominator the best rational approximations
                may have when expressed as a fraction in lowest terms.
            Default: 10 ** 12
        
    
    Returns:
    The sum of the denominators of the described best rational approximations.

    Outline of rationale:
    See documentation for continuedFractionConvergentGenerator() and
    https://shreevatsa.wordpress.com/2011/01/10/not-all-best-rational-approximations-are-the-convergents-of-the-continued-fraction/
    """
    # See https://shreevatsa.wordpress.com/2011/01/10/not-all-best-rational-approximations-are-the-convergents-of-the-continued-fraction/
    since = time.time()
    m = 2
    nxt_sq = m ** 2
    res = 0
    for num in range(2, n_max + 1):
        if num == nxt_sq:
            m += 1
            nxt_sq = m ** 2
            continue
        frac = sqrtBestRationalApproximation(denom_bound, num)
        res += frac[1]
        #print(num, frac)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 193
def squareFreeNumberCount(n_max: int=2 ** 50 - 1) -> int:
    """
    Solution to Project Euler #193

    Finds the number of square free numbers no greater than n_max.

    A square free number is a strictly positive integer that is not
    divisible by any squared prime number.

    Args:
        Optional named:
        n_max (int): Strictly positive integer for which no integer
                greater than this is considered for inclusion in the
                count.
            Default: 2 ** 50 - 1
    
    Returns:
    Integer (int) giving the number of square free numbers no greater
    than n_max.

    Outline of rationale:
    Instead of counting the strictly positive integers no greater than
    n_max that are square free we count the strictly positive integers
    no greater than n_max that are not square free and subtract this
    from n_max.
    Consider a prime p and the multiples of p ** 2. These are not
    square free, and the number of these no greater than n_max is
    the floor of n_max / p ** 2. This suggests that we might find the
    count of numbers that are not square free no greater than n_max
    by adding together n_max / p ** 2 all primes p no greater than
    the square root of n_max.
    But this would cause double counting, as for two distinct primes
    p1 and p2, the multiples of p1 * p2 will be counted twice.
    This can be corrected by the inclusion-exclusion principle, by
    taking all integers between 2 and the square root of n_max
    inclusive that are not divisible by the square of a prime, and
    for each such number m finding n_max / m ** 2 and, if m is
    divisible by an odd number of distinct primes adding this to
    the total and if m is divisible by an even number of distinct
    primes subtracting this from the total. This will produce
    the number of strictly positive integers no greater than n_max
    that are not square free, which as previously noted can be
    subtracted from n_max to get the desired result.
    We use a prime sieve that records the smallest prime factor
    of each integer to efficiently identify the integers no greater
    than the square root of n_max that do not have any repeated prime
    factors and to find the number of distinct prime factors of those
    integers.
    """
    # Review- consider making the explanation more rigerous
    # Review- Try to find a faster method (look into Mobius function)
    since = time.time()
    mx = isqrt(n_max)
    ps = PrimeSPFsieve(mx)
    res = 0
    for n in range(2, mx + 1):
        p_cnt = 0
        n2 = n
        while n2 > 1:
            if ps.sieve[n2][1] != 1: break
            p_cnt += 1
            n2 = ps.sieve[n2][2]
        else:
            contrib = n_max // n ** 2
            #print(n, contrib, p_cnt)
            res += contrib if (p_cnt & 1) else -contrib
    #print(res)
    res = n_max - res
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 194
def allowedColouredConfigurationsCount(type_a_count: int=25, type_b_count: int=75, n_colours: int=1984, md: Optional[int]=10 ** 8) -> int:
    """
    Solution to Project Euler #194
    
    Consider the two undirected unweighted graphs A and B, each consisting
    of 7 vertices which, when labelled with the integers from 0 to 6
    inclusive each have the edges between the vertex pairs:
        (0, 1)
        (0, 2)
        (0, 4)
        (1, 5)
        (2, 3)
        (2, 4)
        (3, 5)
        (4, 6)
        (5, 6)
    with graph A having an additional edge between the vertex pair:
        (1, 3)
    Two such graphs (A and A, A and B, B and A or B and B) can be
    fused to form a larger graph by letting the vertices labelled 2
    and 3 (as per the labelling above) in the first of the two graphs
    be the same vertices as those labelled 0 and 1 respectively in
    the second, and similarly letting the edge between vertex pair
    (2, 3) in the first graph be the same as the edge between vertex
    pair (0, 1) in the second graph.

    This process can then be repeated by using the second graph in
    the above fusion as the first graph in the next fusion and a new
    graph A or B as the second and performing another fusion in the
    same way. We can then continue this process indefinitely,
    effectively producing a chain of A and B graphs.
    
    Consider all graphs that can be formed in this way consisting
    of the fusion of type_a_count graphs of type A and type_b_count
    graphs of type B in any order. This function identifies the
    sum of the number of ways of colouring the vertices each such
    graph, such that there are n_colours different colours available
    and no two adjacent vertices (i.e. no two vertices with an edge
    between them) have the same colour, with the answer given modulo
    md if md is given.

    Args:
        Optional named:
        type_a_count (int): The number of A graphs (as defined above)
                used in the construction of the graphs of interest
                through the repeated fusion of A and B graphs.
            Default: 25
        type_b_count (int): The number of B graphs (as defined above)
                used in the construction of the graphs of interest
                through the repeated fusion of A and B graphs.
            Default: 75
        n_colours (int): Strictly positive integer giving the number
                of colours available for colouring vertices.
        md (int or None): If specified as a strictly positive integer,
                the modulus to which the final count is to be returned
                (i.e. the answer is the remainder of the count when
                divided by this value).
            Default: 10 ** 8

    Returns:
    Integer (int) giving the sum over the number of vertex colourings
    allowed for all graphs that can be constructed from type_a_count A
    graphs and type_b_count B graphs as outlined above such that
    n_colours different colours are available and no two adjacent vertices
    can be the same colour, (with the count being given modulo md if md is
    given as a strictly positive integer).

    Outline of rationale:
    We first observe that all of the n_colours different colours are
    treated equivalently, so any valid colouring formed by taking a valid
    colouring and permuting the colours in any way is also valid.
    We further observe that in both graph A and graph B the vertices labelled
    0 and 1 are adjacent and the vertices labelled 2 and 3 are connected,
    in any valid colouring, for any A or B graph making up the graph, vertices
    labelled 0 and 1 are of different colours from each other and the vertices
    labelled 2 and 3 are also different colours from each other.
    Combining these two observations, consider when constructing the graph
    the step of fusing a new A or B graph to the existing graph, the only
    restrictions on the new A or B graph is that vertices labelled 0 and
    1 are already coloured, and are already guaranteed to have different
    colours. Given that permutations of the colours on a valid colouring
    result in a valid colouring, and the newly fused A or B graph does
    not connect to the existing subgraph in any other way, regradless
    of how the existing subgraph has been coloured, the number of ways
    of colouring the A or B graph to get a valid colouring is simply
    the number of ways to colour that type of graph where the colours of
    vertices labelled 0 and 1 have already been chosen (and are necessarily
    different colours). Furthermore, this number does not depend on which of
    the available colours have been chosen (due to the invariance of colour
    validity on permuting the colours). We therefore conclude that the
    number of colourings of a subgraph constructed from A and B graphs
    as described fused to a new A or B graph as per the described construction
    is equal to the number of valid colourings of the existing
    subgraph multiplied by the number of valid ways of colouring the A or
    B graph being fused such that the colours of vertices labelled 0 and 1
    have already been selected and are different from each other.
    We can treat the first A or B graph used to construct the graph in the same
    way by starting the graph with two connected vertices which are already
    coloured (differently from each other) and fusing vertices 0 and 1 and
    their connecting edge with the first A or B with these vertices and
    edge in a similar manner to the connection to the vertices 2 and 3 of
    another A or B graph. Similarly to previously, the number of valid ways
    of colouring this graph is the number of ways of colouring the two
    initial vertices (which is (n_colours * (n_colours - 1))) multiplied
    by the number of valid ways of colouring an A or B graph (depending
    on which type was chosen) such that the colours of vertices labelled
    0 and 1 have already been selected and are different from each other.
    These observations constitute an induction step and a basis for induction
    respectively that allow us to conclude by induction that for graphs
    constructed by the iterative fusion of the given numbers A and B graphs
    in a particular order, the number of valid colourings is equal to:
        n_colours * (n_colours - 1) *
        (number of valid colourings of graph A where the colour of vertices
        0 and 1 have been selected and are different) ** type_a_count *
        (number of valid colourings of graph B where the colour of vertices
        0 and 1 have been selected and are different) ** type_b_count
    Note that this does not depend on the actual ordering of A and B graphs,
    and all orderings are allowed. Accounting for the fact that all A graphs
    are equivalent to each other and all B graphs are equivalent to each other,
    but A and B graphs are distinct from each other, the number of possible
    orderings is simply (type_a_count + type_b_count) choose type_a_count.
    Therefore, the total is simply the above expression multiplied by
    ((type_a_count + type_b_count) choose type_a_count).
    All that remains is to find the number of valid colourings of each of
    graph A and B where the colour of vertices 0 and 1 have been selected
    and are different.
    We refer to the colours by numbers, where by definition the already coloured
    vertices 0 and 1 are also labelled 0 and 1.
    We also exploit the equivalence of validity of colouring under permutations
    of the colours by only considering colourings such that each vertex takes
    a colour number which is either one already used by a vertex with a smaller
    label or is exactly one greater than the largest number used, and multiplying
    this number by the number of different assignments of colours to the number
    labels this represents.
    By categorising the different colourings based on the exact number of
    colour labels used, we can deduce that if a number labelling uses exactly m
    different numbers then the number of possible colourings this represents
    (recalling that vertices 0 and 1 already have fixed colours) is given by:
        (n_colours - 2)_(m - 2)
    where integer a and non-negative integer b, a_b is the falling factorial:
        a_b = (product i from 0 to (b - 1)) (a - i)
    The restrictions on labelling given and the simplicity of graphs A and B
    enable the number of possible labellings involving the differnt number of
    colours to be counted by hand.
    For graph A:
        number of labellings with fewer than 3 different labels = 0
        number of labellings with exactly 3 different labels = 4
        number of labellings with exactly 4 different labels = 27
        number of labellings with exactly 5 different labels = 33
        number of labellings with exactly 6 different labels = 11
        number of labellings with exactly 7 different labels = 1
        number of labellings with more than 7 different labels = 0
    For graph B:
        number of labellings with fewer than 3 different labels = 0
        number of labellings with exactly 3 different labels = 6
        number of labellings with exactly 4 different labels = 38
        number of labellings with exactly 5 different labels = 40
        number of labellings with exactly 6 different labels = 12
        number of labellings with exactly 7 different labels = 1
        number of labellings with more than 7 different labels = 0
    Combining these numbers and the number of colourings corresponding
    to each labelling using a specific number of different labels
    and appropriately inserting these into the formula found previously
    for the total number of colourings, we find an explicit formula
    for the sum over the different number of different valid colourings
    from n_colours available different colours over all graphs
    constructed as described by the fusion of type_a_count A graphs
    and type_b_count B graphs in all posisble orders:
        ((type_a_count + type_b_count) choose type_a_count) *
        n_colours * (n_colours - 1) *
        (4 * (n_colours - 2)_1 + 27 * (n_colours - 2)_2 +
            33 * (n_colours - 2)_3 + 11 * (n_colours - 2)_4 +
            (n_colours - 2)_5) ** type_a_count *
        (6 * (n_colours - 2)_1 + 38 * (n_colours - 2)_2 +
            40 * (n_colours - 2)_3 + 12 * (n_colours - 2)_4 +
            (n_colours - 2)_5) ** type_b_count *
    """
    # Review- generalise to different graph structures
    since = time.time()
    if not type_a_count and not type_b_count: return 1
    if n_colours < 3: return 0

    n = n_colours
    
    res = math.comb(type_a_count + type_b_count, type_a_count) * n * (n - 1) * \
            ((n - 2) * (4 + (n - 3) * (27 + (n - 4) * (33 + (n - 5) * (11 + (n - 6)))))) ** type_a_count *\
            ((n - 2) * (6 + (n - 3) * (38 + (n - 4) * (40 + (n - 5) * (12 + (n - 6)))))) ** type_b_count
    if md is not None:
        res %= md
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 195
def integerSideSixtyDegreeTrianglesWithMaxInscribedCircleRadiusCount(radius_max: int=1053779) -> int:
    """
    Solution to Project Euler #195
    """
    since = time.time()
    r_sq_mx = radius_max ** 2
    """
    
    m_max = isqrt(48 * r_sq_mx)
    print(f"m_max = {m_max}")
    res = 0
    seen = set()
    for m in range(2, m_max + 1):
        #print(f"m = {m}")
        for n in range(1, m):
            if gcd(m, n) != 1: continue
            a0, b0, c0 = 2 * m * n - n ** 2, m ** 2 - n ** 2, m ** 2 - m * n + n ** 2
            #print(a0, b0, c0)
            g = gcd(gcd(a0, b0), c0)
            #if g != 1:
            #    print((m, n), (a0, b0, c0), g)
            a, b, c = a0 // g, b0 // g, c0 // g
            tup = tuple(sorted([a, b, c]))
            if tup in seen  or tup[0] == tup[-1]: continue
            seen.add(tup)
            #print(a, b, c)
            r_sq = 3 * (n * (m - n)) ** 2
            comp1 = r_sq_mx * (4 * 3 ** 2)
            if r_sq > comp1: break
            comp2 = r_sq_mx * (4 * g ** 2)
            #print((a, b, c), g)
            #print(r_sq / r_sq_mx, r_sq, r_sq_mx)
            k_sq_mx = comp2 // r_sq
            #if not k_sq_mx: break
            res += isqrt(k_sq_mx)
    """
    # Why is the following slower?
    #print(f"m_max = {m_max}")
    res = 0
    seen = {}
    m_max1 = isqrt((16 * r_sq_mx) // 3)
    for m in range(2, m_max1 + 1):
        #print(f"m = {m}")
        start = (-(m - 1) % 3)
        for n0 in range(start, ((m + 1) >> 1), 3):
            for n in range(n0, n0 + 2):
                if not n or gcd(m, n) != 1: continue
                a, b, c = 2 * m * n - n ** 2, m ** 2 - n ** 2, m ** 2 - m * n + n ** 2
                g = gcd(gcd(a, b), c)
                if g != 1: print(1, g, (m, n), a, b, c)
                #print(a0, b0, c0)
                #g = gcd(gcd(a0, b0), c0)
                #if g != 1:
                #    print((m, n), (a0, b0, c0), g)
                #a, b, c = a0 // g, b0 // g, c0 // g
                tup = tuple(sorted([a, b, c]))
                if tup in seen or tup[0] == tup[-1]:
                    #print(f"already seen, {(m, n)}; {seen[tup]}")
                    continue
                seen[tup] = (m, n)
                #print(a, b, c)
                r_sq = 3 * (n * (m - n)) ** 2
                #comp1 = r_sq_mx * (4 * 3 ** 2)
                #if r_sq > comp1: break
                #comp2 = r_sq_mx * (4 * g ** 2)
                comp = r_sq_mx * 4
                if r_sq > comp: break
                #print((a, b, c), g)
                #print(r_sq / r_sq_mx, r_sq, r_sq_mx)
                k_sq_mx = comp // r_sq
                #if not k_sq_mx: break
                res += isqrt(k_sq_mx)
            else: continue
            break
    since1 = time.time()
    print(since1 - since)
    m_max2 = isqrt(48 * r_sq_mx)
    for m0 in range(1, m_max2 + 1, 3):
        for i, m in enumerate((m0, m0 + 1)):
            for n in range(2 - i, (m >> 1), 3):
                if gcd(m, n) != 1: continue
                a, b, c = (2 * m * n - n ** 2) // 3, (m ** 2 - n ** 2) // 3, (m ** 2 - m * n + n ** 2) // 3
                g = gcd(gcd(a, b), c)
                if g != 1: print(2, g, (m, n), a, b, c)
                #print(a0, b0, c0)
                #g = gcd(gcd(a0, b0), c0)
                #if g != 1:
                #    print((m, n), (a0, b0, c0), g)
                #a, b, c = a0 // g, b0 // g, c0 // g
                tup = tuple(sorted([a, b, c]))
                if tup in seen or tup[0] == tup[-1]:
                    print(f"already seen, {(m, n)}; {seen[tup]}")
                    continue
                seen[tup] = (m, n)
                #print(a, b, c)
                r_sq = (n * (m - n)) ** 2
                #comp1 = r_sq_mx * (4 * 3 ** 2)
                #if r_sq > comp1: break
                #comp2 = r_sq_mx * (4 * g ** 2)
                comp = r_sq_mx * 12
                if r_sq > comp:
                    #print(m, n)
                    break
                #print((a, b, c), g)
                #print(r_sq / r_sq_mx, r_sq, r_sq_mx)
                k_sq_mx = comp // r_sq
                #if not k_sq_mx: break
                res += isqrt(k_sq_mx)
    """
    for m in range(2, m_max2 + 1):
        if not m % 3: continue
        print(f"m = {m}")
        start = (-m % 3)
        if not start: start += 3
        for n in range(start, (m >> 1), 3):
            #print(m, n)
            if gcd(m, n) != 1: continue
            a, b, c = (2 * m * n - n ** 2) // 3, (m ** 2 - n ** 2) // 3, (m ** 2 - m * n + n ** 2) // 3
            g = gcd(gcd(a, b), c)
            if g != 1: print(2, g, (m, n), a, b, c)
            #print(a0, b0, c0)
            #g = gcd(gcd(a0, b0), c0)
            #if g != 1:
            #    print((m, n), (a0, b0, c0), g)
            #a, b, c = a0 // g, b0 // g, c0 // g
            tup = tuple(sorted([a, b, c]))
            if tup in seen or tup[0] == tup[-1]:
                print(f"already seen, {(m, n)}; {seen[tup]}")
                continue
            seen[tup] = (m, n)
            #print(a, b, c)
            r_sq = (n * (m - n)) ** 2
            #comp1 = r_sq_mx * (4 * 3 ** 2)
            #if r_sq > comp1: break
            #comp2 = r_sq_mx * (4 * g ** 2)
            comp = r_sq_mx * 12
            if r_sq > comp:
                print(m, n)
                break
            #print((a, b, c), g)
            #print(r_sq / r_sq_mx, r_sq, r_sq_mx)
            k_sq_mx = comp // r_sq
            #if not k_sq_mx: break
            res += isqrt(k_sq_mx)
    """
    print(time.time() - since1)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 196
def numberTrianglePrimeTripletRowSum(row_num: int, ps: Optional[SimplePrimeSieve]=None) -> int:
    """
    Consider a sequence of lists of integers such that the first list
    is the single integer 1, and each subsequent list contains exactly
    one more element than the list immediately preceding it, and whose
    first element is one greater than the last element of the preceding
    list and every other element in the list is exactly one greater than
    the preceding list element.

    In this construction, a prime triplet is defined to be three distinct
    prime numbers such that relative to the location of one of those
    numbers in the construction (referred to as the centre prime), each
    of the other two are in the same list, the preceding list or the
    succeeding list and its position in that list is either the same, one
    before or one after the position of the centre prime in its list.

    This function calculates the sum of numbers that appear in the
    row_num:th list and are constituent primes in at least one prime
    triplet.

    Args:
        Required positional:
        row_num (int): The number of the list in the sequence of lists
                in the described construction for which the elements
                that are part of at least one prime triplet are to be
                summed.

        Optional named:
        ps (SimplePrimeSieve or None): If given, a prime sieve used
                to establish which numbers are prime.
    
    Returns:
    Integer (int) giving the sum of elements in the row_num:th list
    in the construction that are constituent primes in at least one prime
    triplet.

    Brief outline of rationale:
    For each prime number in the row, we consider the possible places the
    prime. We reduce the search space by considering the parity of the
    elements (i.e. whether they are odd or even), which for rows beyond
    the 4th results in a splitting into two cases: odd row numbers
    (for which numbers directly above have the same parity and directly
    below have different parity from the current number) and even row
    numbers (for which numbers directly above have different parity and
    directly below have the same parity as the current number). Additionally,
    the first odd number in each row is treated separately due to it
    encountering the left border of the triangle, which further splits
    into the cases where the first element of the row is even or odd.
    We also note that the last element of each row is a triangle number,
    and triangle numbers larger than 3 are all composite, enabling a
    further minor reduction in the search space.
    We use the Miller-Rabin primality test with 10 trials to establish
    whether a number is prime or not. This is not a probabalistic rather
    than an exact test, and can falsely identify non-primes as primes (though
    not the reverse), but appears to be reliable enough to give the
    correct answer and for the numbers that are involved in this problem
    (> 10 ** 10) a full check is either prohibitavely slow or requires
    a prohibitively large amount of memory.
    """
    if row_num == 1: return 0
    elif row_num in {2, 3}: return 5
    elif row_num == 4: return 7
    
    end_idx = (row_num * (row_num + 1)) >> 1
    start_idx = end_idx - row_num + 1
    mx_num = end_idx + 2 * row_num + 1

    mx_num_sqrt = isqrt(mx_num)

    if ps is None:
        ps = SimplePrimeSieve()
    #ps.extendSieve(min(10 ** 7, mx_num_sqrt))

    memo = {}
    def primeCheck(num: int) -> bool:
        args = num
        if args in memo.keys(): return memo[args]
        res = ps.millerRabinPrimalityTest(num, n_trials=10)
        # res = ps.isPrime(num, extend_sieve=False, extend_sieve_sqrt=False, use_miller_rabin_screening=True, n_miller_rabin_trials=3)
        memo[args] = res
        return res

    res = 0
    if row_num & 1:
        if start_idx & 1:
            num = start_idx
            if primeCheck(num):
                num2 = num - row_num + 1
                num3 = num + row_num + 1
                num4 = num2 - row_num + 3
                num5 = num3 + row_num + 1
                num6 = num + 2
                if (primeCheck(num2) and primeCheck(num3)) or\
                        (primeCheck(num2) and (primeCheck(num4))) or\
                        (primeCheck(num3) and (primeCheck(num5) or primeCheck(num6))):
                    res += num
        else:
            num = start_idx + 1
            if primeCheck(num):
                num2 = num - row_num + 1
                num3 = num + row_num - 1
                num4 = num3 + 2
                num5 = num2 - row_num + 1
                num6 = num5 + 2
                num7 = num3 + row_num - 1
                num8 = num7 + 2
                num9 = num + 2
                if (primeCheck(num2) + primeCheck(num3) + primeCheck(num4) > 1) or\
                        (primeCheck(num2) and (primeCheck(num5) or primeCheck(num6))) or\
                        (primeCheck(num3) and primeCheck(num7)) or\
                        (primeCheck(num4) and (primeCheck(num8) or primeCheck(num9))):
                    res += num
        for num in range(num + 2, end_idx - 2, 2):
            if not primeCheck(num): continue
            num2 = num - row_num + 1
            num3 = num + row_num - 1
            num4 = num3 + 2
            if (primeCheck(num2) + primeCheck(num3) + primeCheck(num4) > 1):
                res += num
                continue
            num5 = num2 - row_num + 1
            num6 = num5 + 2
            if primeCheck(num2) and (primeCheck(num5) or primeCheck(num6)):
                res += num
                continue
            num7 = num3 + row_num + 1
            num8 = num - 2
            if primeCheck(num3) and (primeCheck(num7) or primeCheck(num8)):
                res += num
                continue
            num9 = num7 + 2
            num10 = num + 2
            if primeCheck(num4) and (primeCheck(num9) or primeCheck(num10)):
                res += num
                continue
        num += 2
        if not primeCheck(num): return res
        num2 = num - row_num + 1
        num3 = num + row_num - 1
        num4 = num3 + 2
        if (primeCheck(num2) + primeCheck(num3) + primeCheck(num4) > 1):
            return res + num
        num5 = num2 - row_num + 1
        if primeCheck(num2) and primeCheck(num5):
            return res + num
        num7 = num3 + row_num + 1
        num8 = num - 2
        if primeCheck(num3) and (primeCheck(num7) or primeCheck(num8)):
            return res + num
        num9 = num7 + 2
        num10 = num + 2
        if primeCheck(num4) and (primeCheck(num9) or primeCheck(num10)):
            return res + num
        return res
    if start_idx & 1:
        num = start_idx
        if primeCheck(num):
            num2 = num - row_num + 2
            num3 = num + row_num
            num4 = num2 - row_num + 2
            num5 = num3 + row_num + 2
            num6 = num + 2
            if (primeCheck(num2) and primeCheck(num3)) or\
                    (primeCheck(num2) and (primeCheck(num4) or primeCheck(num6))) or\
                    (primeCheck(num3) and primeCheck(num5)):
                res += num
    else:
        num = start_idx + 1
        if primeCheck(num):
            num2 = num - row_num
            num3 = num2 + 2
            num4 = num + row_num
            num5 = num2 - row_num + 2
            num6 = num5 + 2
            num7 = num4 + row_num
            num8 = num7 + 2
            num9 = num + 2
            if (primeCheck(num2) + primeCheck(num3) + primeCheck(num4) > 1) or\
                    (primeCheck(num2) and primeCheck(num5)) or\
                    (primeCheck(num3) and (primeCheck(num6) or primeCheck(num9))) or\
                    (primeCheck(num4) and (primeCheck(num7) or primeCheck(num8))):
                res += num
        num += 2
    for num in range(num + 2, end_idx - 2, 2):
        if not primeCheck(num): continue
        num2 = num - row_num
        num3 = num2 + 2
        num4 = num + row_num
        if (primeCheck(num2) + primeCheck(num3) + primeCheck(num4) > 1):
            res += num
            continue
        num5 = num2 - row_num + 2
        num6 = num - 2
        if primeCheck(num2) and (primeCheck(num5) or primeCheck(num6)):
            res += num
            continue
        num7 = num5 + 2
        num8 = num + 2
        if primeCheck(num3) and (primeCheck(num7) or primeCheck(num8)):
            res += num
            continue
        num9 = num4 + row_num
        num10 = num9 + 2
        if primeCheck(num4) and (primeCheck(num9) or primeCheck(num10)):
            res += num
            continue
    num += 2
    if not primeCheck(num): return res
    if end_idx & 1:
        num2 = num - row_num
        num3 = num2 + 2
        num4 = num + row_num
        if (primeCheck(num2) + primeCheck(num3) + primeCheck(num4) > 1):
            return res + num
        num5 = num2 - row_num + 2
        num6 = num - 2
        if primeCheck(num2) and (primeCheck(num5) or primeCheck(num6)):
            return res + num
        num8 = num + 2
        if primeCheck(num3) and (primeCheck(num8)):
            return res + num
        num9 = num4 + row_num
        num10 = num9 + 2
        if primeCheck(num4) and (primeCheck(num9) or primeCheck(num10)):
            return res + num
        return res
    num2 = num - row_num
    num4 = num + row_num
    if (primeCheck(num2) and primeCheck(num4)):
        return res + num
    num5 = num2 - row_num + 2
    num6 = num - 2
    if primeCheck(num2) and (primeCheck(num5) or primeCheck(num6)):
        return res + num
    num9 = num4 + row_num
    num10 = num9 + 2
    if primeCheck(num4) and (primeCheck(num9) or primeCheck(num10)):
        return res + num
    return res
    
    """
    res = 0

    candidate_primes = set()
    candidate_prime_sum = 0
    prime_counts_qu = deque()
    prime_counts_curr = [0] * 3
    is_prime = [False] * 5
    for i in range(3):
        prev_is_prime = is_prime[1: -1]
        num3 = start_idx + i
        num4 = num3 + row_num
        num5 = num4 + row_num + 1
        num2 = num3 - row_num + 1
        num1 = num2 - row_num + 2
        nums = [num1, num2, num3, num4, num5]
        is_prime = [primeCheck(num) for num in nums]
        pc = [sum(is_prime[:3])]
        for j in range(3, len(is_prime)):
            pc.append(pc[-1] + is_prime[j] - is_prime[j - 3])
        prime_counts_curr = [x + y for x, y in zip(prime_counts_curr, pc)]
        prime_counts_qu.append(pc)
        if is_prime[2]:
            candidate_primes.add(nums[2])
            candidate_prime_sum += nums[2]
        if max([x * y for x, y in zip(prime_counts_curr, prev_is_prime)]) >= 3:
            res += candidate_prime_sum
            candidate_primes = set()
            candidate_prime_sum = 0
    #print(nums)
    #print(2, prev_is_prime, nums[2], res)

    for i in range(3, row_num):
        prev_is_prime = is_prime[1: -1]
        nums = [num + 1 for num in nums]
        #print(nums)
        is_prime = [primeCheck(num) for num in nums]
        pc = [sum(is_prime[:3])]
        for j in range(3, len(is_prime)):
            pc.append(pc[-1] + is_prime[j] - is_prime[j - 3])
        prime_counts_qu.append(pc)
        prime_counts_curr = [x + y - z for x, y, z in zip(prime_counts_curr, pc, prime_counts_qu.popleft())]
        if is_prime[2]:
            candidate_primes.add(nums[2])
            candidate_prime_sum += nums[2]
        if nums[2] - 3 in candidate_primes:
            candidate_primes.remove(nums[2] - 3)
            candidate_prime_sum -= nums[2] - 3
        if max([x * y for x, y in zip(prime_counts_curr, prev_is_prime)]) >= 3:
            res += candidate_prime_sum
            candidate_primes = set()
            candidate_prime_sum = 0
        #print(i, prev_is_prime, nums[2], res)
    
    return res
    """
    
def numberTrianglePrimeTripletRowsSum(row_nums: List[int]=[5678027, 7208785]) -> int:
    """
    Solution to Project Euler #196

    Consider a sequence of lists of integers such that the first list
    is the single integer 1, and each subsequent list contains exactly
    one more element than the list immediately preceding it, and whose
    first element is one greater than the last element of the preceding
    list and every other element in the list is exactly one greater than
    the preceding list element.

    In this construction, a prime triplet is defined to be three distinct
    prime numbers such that relative to the location of one of those
    numbers in the construction (referred to as the centre prime), each
    of the other two are in the same list, the preceding list or the
    succeeding list and its position in that list is either the same, one
    before or one after the position of the centre prime in its list.

    This function calculates the sum of numbers that appear in any of
    the lists with sequence numbers in row_nums and are constituent primes
    in at least one prime triplet.

    Args:
        Optional named:
        row_nums (list of ints): The sequence numbers of the lists in the
                described construction for which the elements that are
                part of at least one prime triplet are to be summed.
            Default: [5678027, 7208785]
    
    Returns:
    Integer (int) giving the sum of elements the lists in the construction
    with sequence numbers in row_nums that are constituent primes in at least
    one prime triplet.

    Brief outline of rationale:
    See Brief outline of rationale for the function primeTriangleTripletRowSum().
    """
    # Review- consider rewording documentation for clarity- esp consider
    # defining a row in the array, as this is the terminology used in the
    # outline of rationale in primeTriangleTripletRowSum() and is probably
    # more intuitive.
    since = time.time()
    ps = SimplePrimeSieve()
    res = sum(numberTrianglePrimeTripletRowSum(row_num=row_num, ps=ps) for row_num in set(row_nums))
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 197
def findFloorRecursiveSequenceLoop(u0: float, max_term: int, base: int=2, a: float=-1., b: float=0., c: float=30.403243784, div: int=10 ** 9) -> Tuple[int, Tuple[float], Tuple[float]]:
    """
    Identifies the initial terms and cycling terms of the sequence u_n
    for strictly positive n defined by the recurrence relation:
        u_0 = u0
        u_(n + 1) = f(u_n)
    where for real x:
        f(x) = floor(base ** (a * x ** 2 + b * x + c)) / div
    where base is an integer strictly greater than 1, a is a strictly
    negative float, b and c are floats and div is a strictly positive
    integer.

    It can be shown that for such values in the expression for f(x),
    the sequence is guaranteed to eventually cycle through the same
    finite subsequence of terms indefinitely (see outline of rationale),
    justifying this representation of the sequence.

    Args:
        Required positional:
        u0 (float): The initial term of the sequence defined by the
                above recurrence relation.

        Optional named:
        base (int): Integer strictly greater than 1 giving the value of
                base in the above expression for f(x).
            Default: 2
        a (float): Strictly negaive integer giving the value of a in
                the above expression for f(x).
            Default: -1.
        b (float): The value of b in the above expression for f(x).
            Default: 0.f
        c (float): The value of c in the above expression for f(x).
            Default: 30.403243784
        div (int): Strictly positive integer giving the value of div
                in the above expression for f(x).
            Default: 10 ** 9
    
    Returns:
    3-tuple whose index 0 contains div, index 1 contains the sequence
    of initial terms (those before the eventual sequence cycle has been
    reached) in order, multiplied by div and given as integers, and
    index 2 contains the sequence of cycling terms in order starting
    with the first term encountered in the cycle, also multiplied by
    div and given as integers.

    Outline of rationale:
    For any quadratic equation in real numbers with a negative square
    coefficient, there is a finite global maximum. Additionally, a
    positive number to the power of a real number is always strictly
    positive, implying that the floor of a positive number to the power
    of a real number is non-negative.
    Consequently, for real x, strictly positive base, real a, b and
    c and negative a the expression:
        floor(base ** (a * x ** 2 + b * x + c))
    is an integer bounded above and below, and thus can only take
    a finite number of values, and thus there must exist some
    positive integer n for which there is some positive integer m < n
    such that u_n = u_m.
    Given that for integer n > 0 u_(n + 1) only depends on the value
    of u_n, if two for positive integers m and n such that m < n,
    u_n and u_m are equal then for any non-negative integer a,
    u_(n + a) = u_(m + a). Furthermore, this implies that for any
    strictly positive integer b, u_(m + b * (n - m)) = u_m, and
    so for any non-negative integer a and strictly positive integer
    b, u_(m + b * (n - m) + a) = u_(m + a), in other words there
    exists a cycle in the sequence of length b * (n - m).
    As previously established, such values of n and m exist for this
    sequence, and so this sequence must eventually end up in a finite
    subsequence of terms that cycle indefinitely.
    """
    seen = {}
    u = u0 * div
    i = 0
    res = []
    i0 = max_term
    #print(max_term)
    for i in range(max_term + 1):
        if u in seen.keys():
            i0 = seen[u]
            break
        seen[u] = i
        res.append(u)
        x = u / div
        u = math.floor(base ** (a * x ** 2 + b * x + c))
        i += 1
    
    return (div, tuple(res[:i0]), tuple(res[i0:]))
        
def findFloorRecursiveSequenceTermSum(term_numbers: List[int]=[10 ** 12, 10 ** 12 + 1], u0: float=-1, base: int=2, a: float=-1., b: float=0., c: float=30.403243784, div: int=10 ** 9) -> float:
    """
    Solution to Project Euler #197

    Identifies the sum of the terms u_k for the values of nkin the
    list term_numbers, where for strictly positive integers n,
    u_n is a term in the sequence defined by the recurrence relation:
        u_0 = u0
        u_(n + 1) = f(u_n)
    where for real x:
        f(x) = floor(base ** (a * x ** 2 + b * x + c)) / div
    where base is an integer strictly greater than 1, a is a strictly
    negative float, b and c are floats and div is a strictly positive
    integer.

    Args:
        Optional named:
        term_numbers (list of ints): The values of k for which the sum
                over u_k is to be found.
            Default: [10 ** 12, 10 ** 12 + 1]
        u0 (float): The initial term of the sequence defined by the
                above recurrence relation.
            Default: -1.
        base (int): Integer strictly greater than 1 giving the value of
                base in the above expression for f(x).
            Default: 2
        a (float): Strictly negaive integer giving the value of a in
                the above expression for f(x).
            Default: -1.
        b (float): The value of b in the above expression for f(x).
            Default: 0.f
        c (float): The value of c in the above expression for f(x).
            Default: 30.403243784
        div (int): Strictly positive integer giving the value of div
                in the above expression for f(x).
            Default: 10 ** 9
    
    Returns:
    Float giving the sum over k in term_numbers of the terms of the
    sequence u_k defined by the above recurrence relation.

    Outline of rationale:
    As explained in the outline of rationale for the function
    findFloorRecursiveSequenceLoop(), for the given restrictions on
    the values of a, b, c, base and div, the sequence eventually
    cycles indefinitely. We can therefore find the terms by using
    findFloorRecursiveSequenceLoop() to find the initial terms and
    the cycle terms, and for each value of k, if k is one of the
    initial terms, returning that, otherwise returning the index:
        (k - (number of initial terms) % len(number of cycle_terms)
    (0-indexed) from the list of cycle terms.
    """
    since = time.time()
    _, init_terms, loop_terms = findFloorRecursiveSequenceLoop(u0, max_term=max(term_numbers), base=base, a=a, b=b, c=c, div=div)
    #print(init_terms, loop_terms)
    res = 0
    for i in term_numbers:
        if i < len(init_terms):
            res += init_terms[i]
            continue
        j = (i - len(init_terms)) % len(loop_terms)
        res += loop_terms[j]
    res /= div
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 198
def orderedFareyFractionPairsWithMaxDenominatorProductGenerator(max_denominator_product: int, max_lower: Tuple[int, int]=(1, 1), incl_zero: bool=True) -> Generator[Tuple[Tuple[int, int], Tuple[int, int]], None, None]:
    """
    Generator yielding all of the Farey fraction pairs with the product
    of the denominators in the pair no greater than max_denominator_product
    and the smaller of the two fractions having a value no less than
    max_lower, with or without pairs that include zero depending on
    incl_zero.

    A Farey pair is a pair of fractions that appear adjacent to each other
    in a Farey sequence of some order.

    In the context used here, the Farey sequence of order n (where n is
    a strcitly positive integer) is the sequence of rational numbers between
    0 and 1 inclusive in order of ascending value which when expressed
    as a fraction in lowest terms has a denominator of at most n.

    The Farey sequence of order 1 is simply:
        0 / 1, 1 / 1
    
    For n greater than 1, the Farey sequence of order n can be constructed
    from the Farey sequence of order (n - 1) by identifying all consecutive
    pairs of elements in that sequence, say p1 / q1 and p2 / q2 whose
    denominators when expressed in lowest terms sum to exactly n (i.e.
    q1 + q2 = n) and inserting between those two elements the fraction
    whose numerator and denominator are the sum of the numerators and
    denominators respectively of the pair of fractions (i.e.
    (p1 + p2) / (q1 + q2)). Note that the fraction this construction
    produces is guaranteed to already be in lowest terms.

    Args:
        Required positional:
        max_denominator_product (int): The largest product of denominators
                for the Farey fraction pairs
        
        Optional named:
        max_lower (2-tuple of ints): Rational number between 0 and 1
                inclusive giving the largest value of the smaller of the
                pair for the Farey pairs that are to be yielded, represented
                as a fraction whose numerator is given in index 0 and whose
                denominator is given in index 1 of the 2-tuple. The former
                should be a non-negative integer and the latter a strictly
                positive integer with a value no less than the former.
            Default: (1, 1) (representing 1)
        incl_zero (bool): Boolean which, if True indicates that Farey
                pairs including 0 (or (0, 1) as a fraction) are to be
                yielded, and if False such pairs are not to be yielded.
            Default: True

    Yields:
    2-tuple of 2-tuples of ints, representing a Farey pair where the
    2-tuple in index 0 is the smaller of the pair and the 2-tuple in index
    1 is the larger of the pair. Both numbers in the pair are represented
    as fractions, with index 0 and 1 in the 2-tuple are the numerator
    and denominator respectively of the represented fraction in lowest
    terms.
    """
    
    # Using Farey sequences
    
    mn_lower_reciprocal = ((max_lower[1] - 1) // max_lower[0]) + 1
    if incl_zero:
        for denom in range(mn_lower_reciprocal, max_denominator_product + 1):
            yield ((0, 1), (1, denom))

    pair = ((0, 1), (1, 1))
    #yield pair
    stk = []
    mx_lower_reciprocal = isqrt(max_denominator_product)
    stk = [((1, d), (1, d - 1)) for d in range(mn_lower_reciprocal, mx_lower_reciprocal + 1)]
    print(mn_lower_reciprocal, mx_lower_reciprocal)
    print(stk)
    while stk:
        pair = stk.pop()
        if pair[0][1] * pair[1][1] > max_denominator_product:
            continue
        yield pair
        frac = (pair[0][0] + pair[1][0], pair[0][1] + pair[1][1])
        if (frac[0] * max_lower[1] <= max_lower[0] * frac[1]):
            stk.append((frac, pair[1]))
        stk.append((pair[0], frac))
    return

def ambiguousNumberCountUpToFraction(max_denominator: int=10 ** 8, upper_bound: Tuple[int, int]=(1, 100), incl_upper_bound: bool=False) -> int:
    """
    Solution to Project Euler Problem #198

    Calculates the number of strictly positive ambiguous numbers
    less than (or less than or equal to) upper_bound which, when
    expressed as a fraction in lowest terms, have a denominator
    no greater than max_denominator.

    An ambiguous number is defined to be a number for which there
    exists a strictly positive integer such that of all rational
    numbers that can be expressed as fractions with a denominator
    no larger than that integer, for exactly two of them the
    absolute difference between them and the ambiguous number is
    exactly equal and is smaller than that of every other such
    rational number.

    Given that the number between and equidistant from two
    rational numbers is guaranteed to be rational, all ambiguous
    numbers are rational, meaning that for any ambiguous number
    we can ask whether when expressed in lowest terms its
    denominator is no greater than max_denominator.

    Args:
        Optional named:
        max_denominator (int): Strictly positive integer giving
                the largest denominator of the rational numbers
                considered for inclusion in the count (when
                expressed as a fraction in lowest terms).
            Default: 10 ** 8
        upper_bound (2-tuple of ints): A non-negative rational
                number giving the upper bound for ambiguous numbers
                considered for inclusion in the count. This number
                is to be given as a fraction represented as a
                2-tuple of ints where index 0 contains the fraction's
                numerator and index 1 the denominator. The numerator
                is to be non-negative and the denominator is to be
                strictly positive, 
            Default: (1, 100) (representing one one-hundredth)
        incl_upper_bound (bool): Boolean which, if True and
                the number upper_bound is ambiguous is included
                in the count, and if False this number is not
                included in the count even if it is ambiguous.
            Default: False
    
    Returns:
    Integer (int) giving the number of strictly positive ambiguous
    numbers less than (if incl_upper_bound is False) or less than
    or equal to (if incl_upper_bound is True) upper_bound which,
    when expressed as a fraction in lowest terms, have a denominator
    no greater than max_denominator.

    Outline of rationale:
    TODO
    """
    since = time.time()

    g = gcd(*upper_bound)
    if g != 1:
        upper_bound = tuple(x // g for x in upper_bound)

    comp = (lambda x: x[0] * upper_bound[1] <= x[1] * upper_bound[0]) if incl_upper_bound else (lambda x: x[0] * upper_bound[1] < x[1] * upper_bound[0])

    mn_lower_reciprocal = ((upper_bound[1] - 1) // upper_bound[0]) + 1
    mx_denom_prod = max_denominator >> 1

    #yield pair
    mx_lower_reciprocal = isqrt(mx_denom_prod)
    safe_stk = [(d, d - 1) for d in range(mn_lower_reciprocal + 1, mx_lower_reciprocal + 1)]

    unsafe_stk = [] if (1, mn_lower_reciprocal) == upper_bound else [((1, mn_lower_reciprocal), (1, mn_lower_reciprocal - 1))]
    #print(mn_lower_reciprocal, mx_lower_reciprocal)
    #print(stk)
    #print(safe_stk)
    #print(unsafe_stk)
    # The number of reciprocals with even denominator and value
    # greater than (or greater than or equal to if incl_upper_bound)
    # upper_bound and denominator no greater than max_denominator.
    res = mx_denom_prod - (mn_lower_reciprocal >> 1) + (incl_upper_bound and upper_bound[0] == 1 and upper_bound[1] <= max_denominator and not upper_bound[1] & 1)
    #print(res)
    while unsafe_stk:
        pair = unsafe_stk.pop()
        denom_hlf = pair[0][1] * pair[1][1]
        if pair[0][1] * pair[1][1] > mx_denom_prod:
            continue
        numer = pair[0][0] * pair[1][1] + pair[1][0] * pair[0][1]
        res += comp((numer, denom_hlf << 1))

        frac = (pair[0][0] + pair[1][0], pair[0][1] + pair[1][1])
        if (frac[0] * upper_bound[1] <= upper_bound[0] * frac[1]):
            unsafe_stk.append((frac, pair[1]))
            safe_stk.append((pair[0][1], frac[1]))
        else:
            unsafe_stk.append((pair[0], frac))
    
    while safe_stk:
        pair = safe_stk.pop()

        if pair[0] * pair[1] > mx_denom_prod:
            continue
        res += 1
        d = pair[0] + pair[1]
        safe_stk.append((pair[0], d))
        safe_stk.append((d, pair[1]))
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

    """
    res = 0
    mx = max_denominator >> 1
    comp = (lambda x: x[0] * upper_bound[1] <= x[1] * upper_bound[0]) if incl_upper_bound else (lambda x: x[0] * upper_bound[1] < x[1] * upper_bound[0])
    for pair in orderedFareyFractionPairsWithMaxDenominatorProductGenerator(mx, max_lower=(1, 100), incl_zero=False):
        if not comp(pair[0]): break
        numer = pair[0][0] * pair[1][1] + pair[1][0] * pair[0][1]
        denom = (pair[0][1] * pair[1][1]) << 1
        #if numer & 1:
        #    denom <<= 1
        #    if denom > max_denominator: continue
        #else:
        #    numer >>= 1
        if comp((numer, denom)):
            res += 1
            #print((numer, denom), pair)
        #res += comp((numer, denom))
    res += mx - ((((upper_bound[1] - 1) // upper_bound[0]) + 1) >> 1)
    """
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

def ambiguousNumberCountUpToReciprocal(max_denominator: int=10 ** 8, upper_bound_reciprocal=100, incl_upper_bound: bool=False) -> int:
    """
    Alternative solution to Project Euler #198

    Calculates the number of strictly positive ambiguous numbers
    less than (or less than or equal to) a given integer reciprocal
    which, when expressed as a fraction in lowest terms, have a
    denominator no greater than max_denominator.

    An ambiguous number is defined to be a number for which there
    exists a strictly positive integer such that of all rational
    numbers that can be expressed as fractions with a denominator
    no larger than that integer, for exactly two of them the
    absolute difference between them and the ambiguous number is
    exactly equal and is smaller than that of every other such
    rational number.

    Given that the number between and equidistant from two
    rational numbers is guaranteed to be rational, all ambiguous
    numbers are rational, meaning that for any ambiguous number
    we can ask whether when expressed in lowest terms its
    denominator is no greater than max_denominator.

    Args:
        Optional named:
        max_denominator (int): Strictly positive integer giving
                the largest denominator of the rational numbers
                considered for inclusion in the count (when
                expressed as a fraction in lowest terms).
            Default: 10 ** 8
        upper_bound_reciprocal (int): A strictly positive integer,
                the reciprocal of which (i.e. one divided by this
                number) gives the upper bound for ambiguous numbers
                considered for inclusion in the count.
            Default: (1, 100) (representing one one-hundredth)
        incl_upper_bound (bool): Boolean which, if True and
                the reciprocal of the number upper_bound is
                ambiguous is included in the count, and if False
                this number is not included in the count even if
                it is ambiguous.
            Default: False
    
    Returns:
    Integer (int) giving the number of strictly positive ambiguous
    numbers less than (if incl_upper_bound is False) or less than
    or equal to (if incl_upper_bound is True) the reciprocal of
    upper_bound which, when expressed as a fraction in lowest terms,
    have a denominator no greater than max_denominator.
    """
    # Alternative solution that only works when the upper bound is a reciprocal
    since = time.time()
    
    mx = max_denominator >> 1

    # The reciprocals with even denominator greater than (or greater than or
    # equal to if incl_upper_bound is True) upper_bound_reciprocal and no
    # greater than max_denominator
    res = mx - (upper_bound_reciprocal >> 1) + (incl_upper_bound and upper_bound_reciprocal <= max_denominator and not upper_bound_reciprocal & 1)
    
    
    denom1 = isqrt(mx)
    stk = list(range(upper_bound_reciprocal, denom1))
    #print(stk)
    
    while stk:
        denom2 = stk[-1]
        if denom1 * denom2 > mx:
            denom1 = stk.pop()
            continue
        res += 1
        stk.append(denom1 + denom2)
    
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 199
def IterativeCirclePackingUncoveredAreaProportion(n_iter: int=10) -> float:
    """
    Solution to Project Euler #199

    Consider a circle containing three identically sized circles such that
    the three circles do not overlap with each other, are externally tangent with
    each other and is externally to internally tangent to the original circle.
    Further consider a process whereby, starting with this construction, at each
    step (which we refer to as an iteration), into each contiguous region inside
    the original circle ot covered by another circle place a circle sized and
    positioned such that it is tangent to each circle on the boundary of that
    region, that tangent being external to internal in the case of the original
    circle and external otherwise (note that for this construction, the region
    will be bounded by exactly three circles, possibly including the boundary
    of the original circle).

    This function calculates the proportion of the area of the original circle
    that is not covered by any other circle after n_iter iterations of this
    process.

    Args:
        Optional named:
        n_iter (int): The number of iterations of the described process that
                should have been completed after which the proportion of the
                area of the original circle uncovered by another circle
                it to be calculated.
            Default: 10
    
    Returns:
    Float giving the proportion of the area of the original circle left
    uncovered by any other circle after n_iter iterations of the above
    described process.

    Outline of rationale:
    We use Descarte's theorem, which states that for four circles that are
    each tangent to each other (with all tangencies at different points)
    with curvatures k1, k2, k3 and k4, the following identity holds:
        (k1 + k2 + k3 + k4) ** 2 = 2 * (k1 ** 2 + k2 ** 2 + k3 ** 2 + k4 ** 2)
    The absolute value of the curvature is the reciprocal of the radius
    of the circle.
    Note that it is only possible to make such a construction if the four
    circles do not overlap at all (and so are all internally tangent to
    each other) or one circle completely contains the other circles, which
    do not overlap with each other (and so the other circles are externally
    to internally tangent to that circle and externally tangent to each
    other). In the former case, all the curvatures are positive, and in
    the latter case, the curvature of the outer circle is negative and the
    others positive.
    Applying this to the construction in the problem, we find that the
    curvature of the original circle should be taken to be negative (without
    loss of generality, -1) and all others positive.
    For a given gap between three circles of known curvature, we can use this
    equation to find the curvature of the circle to be placed.
    If we solve the equation for k4 given known k1, k2 and k3, we get:
        k4 = (k1 + k2 + k3) +/- 2 * sqrt(k1 * k2 + k1 * k3 + k2 * k3)
    It can be shown that if the negative branch is chosen for nonzero k1, k2,
    k3 then k4 will be negative. As noted, other than the original circle
    the curvature should be positive, and so the positive branch should
    always be used.
    We can find the curvatures of the initial identical three circles
    with basic trigonometry to get a curvature of:
        (2 / sqrt(3) + 1)
    recalling that we set the curvature of the original circle as -1,
    corresponding to a radius of 1.
    All that remains is to identify the gaps in each iteration. For the first
    iteration, there are three gaps between the original circle and two of
    the three identical circles, and a single gap between the three identical
    circles.
    Gaps for subsequent iterations can be identified during the previous
    iteration, by noticing that when a circle is placed in a gap, it produces
    exactly three gaps, those being between the circle being placed and two
    of the circles on the boundary. In fact, this accounts for all of the
    gaps as any others that exist will have been filled in that iteration.
    We can therefore simply make a record of these new gaps when a new circle
    is inserted, ready for the next iteration.
    This enables the problem to be solved, by taking the areas of all the
    circles produced during the n_iter iterations, and (since these circles
    do not intersect with each other) summing these areas and diving them
    by the area of the original circle (which is pi) to get the proportion
    of the area covered and subtracting this from 1 to get the proportion
    uncovered, the desired result.
    This can be slightly optimised by noting that all the areas end up
    including a factor of pi, which is cancelled when taking the ratio of
    the interior circle areas with the original circle area. We can therefore
    ignore the factor of pi.
    A further optimisation is made by noting that the curvature and so radius
    of each circle is only dependent on the curvatures of the three forming
    the gap into which it is being placed. Given the symmetry of the
    construction there will inevitably be repeats of these curvatures. We
    therefore can avoid repeating the same calculations by keeping track
    of the counts of circles formed from different precursor combinations
    (these precursors being curvatures previously encountered), with the
    actual curvatures and subsequently radii and areas calculated later,
    with the areas multiplied by the number of occurrences of that circle
    type in the final calculation.
    """
    # Review- consider rewording the documentation for clarity
    since = time.time()
    n_initial_internal_circles = 3
    if n_initial_internal_circles < 2: return 0.
    counts = [0, n_initial_internal_circles]
    precursors = [(), ()]
    precursor_dict = {}
    curr_dict = {(0, 1, 1): n_initial_internal_circles, (1, 1, 1): 1}
    for i in range(n_iter):
        prev_dict = curr_dict
        curr_dict = {}
        for prec, f in prev_dict.items():
            if prec in precursor_dict.keys():
                idx = precursor_dict[prec]
            else:
                idx = len(precursors)
                precursors.append(prec)
                counts.append(0)
                precursor_dict[prec] = idx
            counts[idx] += f
            if i == n_iter - 1: continue
            for prec2 in ((prec[0], prec[1], idx), (prec[0], prec[2], idx), (prec[1], prec[2], idx)):
                curr_dict[prec2] = curr_dict.get(prec2, 0) + f
    radii_inv = [-1, 1 + 2 / math.sqrt(3)]
    #print(precursors)
    #print(counts)
    
    res = 1 - counts[1] / radii_inv[1] ** 2
    for i in range(2, len(counts)):
        f = counts[i]
        prec = precursors[i]
        prec_r_inv = [radii_inv[j] for j in prec]
        #print(prec_r_inv)
        r_inv = sum(prec_r_inv) + 2 * math.sqrt(prec_r_inv[0] * prec_r_inv[1] + prec_r_inv[0] * prec_r_inv[2] + prec_r_inv[1] * prec_r_inv[2])
        radii_inv.append(r_inv)
        res -= f / r_inv ** 2
    #print(radii_inv)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 200
def squbeGenerator(ps: Optional[SimplePrimeSieve]=None, filter_func: Optional[Callable[[int], bool]]=None) -> Generator[int, None, None]:
    """
    Generator yielding all sqube numbers in order that give 
    (if such a function is given) a value of True when applied
    to the function filter_func.

    A sqube is a strictly positive integer that can be expressed
    as p ** 2 * q ** 3 where p and q are distinct prime numbers.

    Args:
        Optional named:
        ps (SimplePrimeSieve object or None): If specified, a
                simple prime sieve object used to produce prime
                numbers.
            Default: None
        filter_func (function with a single integer argument and
                outputting a bool or None): If specified, a boolean
                function used to filter the squbes that are yielded.
                If not specified, then no squbes are omitted.
            Default: None (resulting in a function that always
                returns True)
    
    Yields:
    Integers (int) representing the squbes that return True to
    filter_func (if that is required) in ascending order.
    

    Outline of rationale:
    In order to yield the numbers in increasing order, the
    sqube numbers are produced in batches, each producing
    the squbes between successive powers of 10. Each batch
    is then sorted and yielded in this sorted order.
    """
    if ps is None: ps = SimplePrimeSieve()
    if filter_func is None:
        filter_func = lambda x: True
    
    rng_mult = 10
    
    curr_rng = [1, rng_mult]
    p_gen = iter(ps.endlessPrimeGenerator())
    p0 = next(p_gen)
    p_nxt = p0
    
    p_lst = []
    i2_0 = 1
    while True:
        #print(f"rng = {curr_rng}")
        p_mx = isqrt(curr_rng[1] // p0 ** 3)
        if p_nxt <= p_mx:
            p_lst.append(p_nxt)
            for p in p_gen:
                if p > p_mx:
                    p_nxt = p
                    break
                p_lst.append(p)
        
        sqube_lst = []
        #i1_0 = 0
        for i2 in range(i2_0, len(p_lst)):
            p2 = p_lst[i2]
            p2_sq = p2 ** 2
            i1_0 = bisect.bisect_right(p_lst, integerNthRoot((curr_rng[0] - 1) // p2_sq, 3), hi=i2)
            i1_1 = bisect.bisect_right(p_lst, integerNthRoot((curr_rng[1] - 1) // p2_sq, 3), lo=i1_0, hi=i2)
            for i1 in range(i1_0, i1_1):
                p1 = p_lst[i1]
                num = p2_sq * p1 ** 3
                if filter_func(num): sqube_lst.append(num)
            
            p2_cub = p2_sq * p2
            i1_0 = bisect.bisect_right(p_lst, isqrt((curr_rng[0] - 1) // p2_cub), hi=i2)
            i1_1 = bisect.bisect_right(p_lst, isqrt((curr_rng[1] - 1) // p2_cub), lo=i1_0, hi=i2)
            for i1 in range(i1_0, i1_1):
                p1 = p_lst[i1]
                num = p2_cub * p1 ** 2
                if filter_func(num): sqube_lst.append(num)
        
        #sqube_lst.sort()
        for num in sorted(sqube_lst):
            yield num

        curr_rng = [curr_rng[1], curr_rng[1] * rng_mult]

    return
    """
    while True:
        if p_max is None: p_max = float("inf")
        for p2 in :
            if p2 > p_max: break
            p2_sq = p2 ** 2
            while sqube_heap and sqube_heap[0] < p2_sq * p_lst[0] ** 3:
                yield heapq.heappop(sqube_heap)
            for p1 in p_lst:
                common = p1 ** 2 * p2_sq
                num2 = common * p1
                if filter_func(num2):
                    heapq.heappush(sqube_heap, num2)
                num3 = common * p2
                if filter_func(num3):
                    heapq.heappush(sqube_heap, num3)
            p_lst.append(p2)
        print("reached largest allowed prime")
        while sqube_heap:
            yield heapq.heappop(sqube_heap)
    return
    """
    

def isPrimeFree(num: int, primeChecker: Callable[[int], bool], base: int=10) -> bool:
    """
    Assesses whether a strictly positive integer num is prime free
    when expressed in the chosen base, checking for prime status
    using the functon primeChecker().

    A strictly positive integer is prime free in a given base if
    and only if it is not prime and when represented in the chosen base,
    all strictly positive integers that when represented in the
    chosen base have the same number of digits as the original number
    (with no leading zeros) and exactly one digit is different, are
    also not prime.

    Args:
        Required positional:
        num (int): Strictly positive integer whose status as prime free
                in the chosen base is being assessed.
        primeChecker (function taking an integer as input and outputting
                a boolean): The function used to check whether a given
                strictly positive integer is prime. For a given strictly
                positive integer as input, should returns True if that
                integer is a prime and False otherwise.
        
        Optional named:
        base (int): Integer strictly greater than 1 giving the base
                in which the given integer should be represented when
                assessing whether it is prime free.
            Defualt: 10

    Returns:
    Boolean (bool), with the value True if num is prime free in the chosen
    base, otherwise False.
    """
    if primeChecker(num): return False
    num2 = num
    mult = 1
    while num2:
        num2, d = divmod(num2, base)
        #print(num2 == 0, d)
        for d2 in range(num2 == 0, d):
            #print(num + (d2 - d) * mult)
            if primeChecker(num + (d2 - d) * mult): return False
        #print(d + 1, base)
        for d2 in range(d + 1, base):
            #print(num + (d2 - d) * mult)
            if primeChecker(num + (d2 - d) * mult): return False
        mult *= base
    return True

def intContainsSubstring(num: int, substr_rightleft_kmp: KnuthMorrisPratt, base: int=10) -> bool:
    """
    Assesses whether a strictly positive integer num when expressed
    in the chosen base without leading zeros contains as a contiguous
    substring the digit value pattern contained in the KnuthMorrisPratt
    object substr_rightleft_kmp.

    The digit value pattern is represented in the KnuthMorrisPratt as a
    list of integers, each between 0 and (base - 1) inclusive representing
    the value of the corresponding digit, in right to left order.
    
    Args:
        Required positional:
        num (int): The strictly positive integer being assessed for
                whether it contains as a substring the digit pattern
        substr_rightleft_kmp (KnuthMorrisPratt): An object encapsulating
                the digit value pattern to be matched in right to left order
                and an implementation of the Knuth-Morris-Pratt algorithm
                for that pattern.
            
        Optional named:
        base (int): Integer strictly greater than 1 giving the base
                in which the given integer should be represented when
                assessing whether it contains the digit value pattern
                contained in substr_rightleft_kmp as a contiguous substring.
            Defualt: 10

    Returns:
    Boolean (bool) with value True if num when expressed in the chosen base
    without leading zeros contains as a contiguous substring the digit value
    pattern contained in the KnuthMorrisPratt object substr_rightleft_kmp.
    """
    s = []
    num2 = num
    while num2:
        num2, d = divmod(num2, base)
        s.append(d)
    #print(s, substr_lst[::-1])
    #kmp = KnuthMorrisPratt(substr_lst[::-1])
    for _ in substr_rightleft_kmp.matchStartGenerator(s):
        #print("contains substring")
        return True
    else: return False

def findNthPrimeProofSqubeWithSubstring(substr_num: int=200, substr_num_lead_zeros_count: int=0, n: int=200, base: int=10) -> int:
    """
    Solution to Project Euler #200

    Finds the n:th smallest strictly positive integer that is a sqube and
    when represented in the chosen base is prime free and contains as a
    contiguous substring
    the digit string consisting of the digits of strictly positive integer
    substr_num when represented in the same base with exactly
    substr_num_lead_zeros_count leading zeros.

    A sqube is a strictly positive integer that can be expressed as
    p ** 2 * q ** 3 where p and q are distinct prime numbers.

    A strictly positive integer is prime free in a given base if
    and only if it is not prime and when represented in the chosen base,
    all strictly positive integers that when represented in the
    chosen base have the same number of digits as the original number
    (with no leading zeros) and exactly one digit is different, are
    also not prime.

    Args:
        Optional named:
        substr_num (int): The value of the digit string when interpreted
                in the chosen base with substr_num_lead_zeros_count leading
                zeros that any sqube prime free integers when represented in
                the chosen base without leading zeros must contain as a
                contiguous substring.
            Default: 200
        substr_num_lead_zeros_count (int): Non-negative integer giving the
                number of leading zeros the digit string with value substr_num
                should have when expressed in the chosen base as a digit
                string that any sqube prime free integers when represented in
                the chosen base without leading zeros must contain as a
                contiguous substring.
            Default: 0
        n (int): Strictly positive integer specifying which of the integers
                satisfying the given constraints should be returned, that
                being the n:th smallest such integer.
            Default: 200
        base (int): Integer strictly greater than 1 giving the base
                in which the given integer should be represented when
                assessing whether it contains the above specified digit
                string as a contiguous substring (with that digit string,
                when interpreted in this base having the value substr_num).
            Defualt: 10
    
    Returns:
    Integer (int) giving the n:th smallest strictly positive integer that is
    a sqube and when represented in the chosen base is prime free and contains
    as a contiguous substring the digit string consisting of the digits of
    strictly positive integer substr_num when represented in the same base with
    exactly substr_num_lead_zeros_count leading zeros.
    """
    since = time.time()
    ps = SimplePrimeSieve()

    memo = {}
    def primeChecker(num: int) -> bool:
        args = num
        if args in memo.keys(): return memo[args]
        res = ps.millerRabinPrimalityTest(num, n_trials=10)
        # res = ps.isPrime(num, extend_sieve=False, extend_sieve_sqrt=False, use_miller_rabin_screening=True, n_miller_rabin_trials=3)
        memo[args] = res
        return res

    count = 0
    p = []
    num2 = substr_num
    while num2:
        num2, d = divmod(num2, base)
        p.append(d)
    p.extend([0] * substr_num_lead_zeros_count)
    kmp = KnuthMorrisPratt(p)

    def squbeFilterFunction(num: int) -> bool:
        return intContainsSubstring(num, kmp, base=base)

    #p = p[::-1]
    found = False
    for num in squbeGenerator(ps=ps, filter_func=squbeFilterFunction):
        #print(num)
        if not isPrimeFree(num, primeChecker, base=base): continue
        count += 1
        print(count, num)
        if count == n:
            found = True
            break
    res = num if found else -1
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

##############
project_euler_num_range = (151, 200)

def evaluateProjectEulerSolutions151to200(eval_nums: Optional[Set[int]]=None) -> None:
    if not eval_nums:
        eval_nums = set(range(project_euler_num_range[0], project_euler_num_range[1] + 1))

    if 151 in eval_nums:
        since = time.time()
        res = singleSheetCountExpectedValueFloat(n_halvings=4)
        print(f"Solution to Project Euler #151 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 152 in eval_nums:
        since = time.time()
        res = sumsOfSquareReciprocalsCount(target=(1, 2), denom_min=2, denom_max=80)
        print(f"Solution to Project Euler #152 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 153 in eval_nums:
        since = time.time()
        res = findRealPartSumOverGaussianIntegerDivisors(n_max=10 ** 8)
        print(f"Solution to Project Euler #153 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 154 in eval_nums:
        since = time.time()
        res = multinomialCoefficientMultiplesCount2(n=2 * 10 ** 5, n_k=3, factor_p_factorisation={2: 12, 5: 12})
        print(f"Solution to Project Euler #154 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 155 in eval_nums:
        since = time.time()
        res = countDistinctCapacitorCombinationValues(max_n_capacitors=18)
        print(f"Solution to Project Euler #155 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 156 in eval_nums:
        since = time.time()
        res = cumulativeNonZeroDigitCountEqualsNumberSum(base=10)
        print(f"Solution to Project Euler #156 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 157 in eval_nums:
        since = time.time()
        res = countReciprocalPairSumsMultipleOfReciprocalPower(reciprocal_factorisation={2: 1, 5: 1}, min_power=1, max_power=9)
        print(f"Solution to Project Euler #157 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 158 in eval_nums:
        since = time.time()
        res = maximumDifferentLetterStringsWithNSmallerLeftNeighbours(n_chars=26, max_len=26, n_smaller_left_neighbours=1)
        print(f"Solution to Project Euler #158 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 159 in eval_nums:
        since = time.time()
        res = maximalDigitalRootFactorisationsSum(n_min=2, n_max=10 ** 6 - 1, base=10)
        print(f"Solution to Project Euler #159 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 160 in eval_nums:
        since = time.time()
        res = factorialFinalDigitsBeforeTrailingZeros(n=10 ** 12, n_digs=5, base=10)
        print(f"Solution to Project Euler #160 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 161 in eval_nums:
        since = time.time()
        res = triominoAreaFillCombinations(n_rows=9, n_cols=12)
        print(f"Solution to Project Euler #161 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 162 in eval_nums:
        since = time.time()
        res = countHexadecimalIntegersContainGivenDigits(max_n_dig=16, n_contained_digs=3, contained_includes_zero=True)
        print(f"Solution to Project Euler #162 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 163 in eval_nums:
        since = time.time()
        res = countCrossHatchedTriangles(n_layers=36)
        print(f"Solution to Project Euler #163 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 164 in eval_nums:
        since = time.time()
        res = countIntegersConsecutiveDigitSumCapped(n_digs=20, n_consec=3, consec_sum_cap=9, base=10)
        print(f"Solution to Project Euler #164 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 165 in eval_nums:
        since = time.time()
        res = blumBlumShubPseudoRandomTwoDimensionalLineSegmentsCountInternalCrossings(
            n_line_segments=5000,
            blumblumshub_s_0=290797,
            blumblumshub_s_mod=50515093,
            coord_min=0,
            coord_max=499,
            use_bentley_ottmann=False,
        )
        #res = twoDimensionalLineSegmentsCountInternalCrossings([((27, 44), (12, 32)), ((46, 53), (17, 62)), ((46, 70), (22, 40))])
        print(f"Solution to Project Euler #165 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 166 in eval_nums:
        since = time.time()
        res = magicSquareWithRepeatsCount(square_side_length=4, val_max=9)
        print(f"Solution to Project Euler #166 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 167 in eval_nums:
        since = time.time()
        res = ulamSequenceTwoOddTermValueSum(a2_min=5, a2_max=21, term_number=10 ** 11)
        print(f"Solution to Project Euler #167 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 168 in eval_nums:
        since = time.time()
        res = rightRotationMultiplesSum(min_n_digs=2, max_n_digs=100, n_tail_digs=5, base=10)
        print(f"Solution to Project Euler #168 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 169 in eval_nums:
        since = time.time()
        res = sumOfPowersOfTwoEachMaxTwice(num=10 ** 25)
        print(f"Solution to Project Euler #169 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 170 in eval_nums:
        since = time.time()
        res = largestPandigitalConcatenatingProduct(min_n_prods=3, incl_zero=True, base=10)
        print(f"Solution to Project Euler #170 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 171 in eval_nums:
        since = time.time()
        res = sumSquareOfTheDigitalSquares(max_n_dig=20, n_tail_digs=9, base=10)
        print(f"Solution to Project Euler #171 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 172 in eval_nums:
        since = time.time()
        res = countNumbersWithDigitRepeatCap(n_dig=18, max_dig_rpt=3, base=10)
        print(f"Solution to Project Euler #172 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 173 in eval_nums:
        since = time.time()
        res = hollowSquareLaminaCount(max_n_squares=10 ** 6)
        print(f"Solution to Project Euler #173 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 174 in eval_nums:
        since = time.time()
        res = hollowSquareLaminaTypeCountSum(max_n_squares=10 ** 6, min_type=1, max_type=10)
        print(f"Solution to Project Euler #174 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 175 in eval_nums:
        since = time.time()
        res = fractionsAndSumOfPowersOfTwoShortenedBinary(numerator=123456789, denominator=987654321)
        print(f"Solution to Project Euler #175 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 176 in eval_nums:
        since = time.time()
        res = smallestCathetusCommonToNRightAngledTriangles(n_common=47547)
        print(f"Solution to Project Euler #176 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 177 in eval_nums:
        since = time.time()
        res = integerAngledQuadrilaterals(tol=10 ** -9)
        print(f"Solution to Project Euler #177 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 178 in eval_nums:
        since = time.time()
        res = countPandigitalStepNumbers(max_n_digs=40, incl_zero=True, base=10)
        print(f"Solution to Project Euler #178 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 179 in eval_nums:
        since = time.time()
        res = countConsecutiveNumberPositiveDivisorsMatch(n_max=10 ** 7)
        print(f"Solution to Project Euler #179 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 180 in eval_nums:
        since = time.time()
        res = goldenTripletsSumTotalNumeratorDenominator(max_order=35)
        print(f"Solution to Project Euler #180 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 181 in eval_nums:
        since = time.time()
        res = groupingNDifferentColouredObjects(colour_counts=[60, 40])
        print(f"Solution to Project Euler #181 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 182 in eval_nums:
        since = time.time()
        res = exponentsMinimisingRSAEncryptionUnconcealedSum(p=1009, q=3643) 
        print(f"Solution to Project Euler #182 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 183 in eval_nums:
        since = time.time()
        res = maximumProductOfPartsTerminatingSum(n_min=5, n_max=10 ** 4, base=10)
        print(f"Solution to Project Euler #183 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 184 in eval_nums:
        since = time.time()
        res = latticeTrianglesContainingOriginCount(lattice_radius=105, incl_edge=False)
        print(f"Solution to Project Euler #184 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 185 in eval_nums:
        since = time.time()
        res = numberMindSimulatedAnnealing(alphabet="0123456789", n_trials=20, guesses=[
            ("5616185650518293", 2),
            ("3847439647293047", 1),
            ("5855462940810587", 3),
            ("9742855507068353", 3),
            ("4296849643607543", 3),
            ("3174248439465858", 1),
            ("4513559094146117", 2),
            ("7890971548908067", 3),
            ("8157356344118483", 1),
            ("2615250744386899", 2),
            ("8690095851526254", 3),
            ("6375711915077050", 1),
            ("6913859173121360", 1),
            ("6442889055042768", 2),
            ("2321386104303845", 0),
            ("2326509471271448", 2),
            ("5251583379644322", 2),
            ("1748270476758276", 3),
            ("4895722652190306", 1),
            ("3041631117224635", 3),
            ("1841236454324589", 3),
            ("2659862637316867", 2),
        ])
        print(f"Solution to Project Euler #185 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 186 in eval_nums:
        since = time.time()
        res = laggedFibonacciGraphEdgeCountForVertexToConnectToProportionOfGraph(
                n_vertices=10 ** 6,
                vertex=524287,
                target_proportion=.99,
                l_fib_poly_coeffs=(100003, -200003, 0, 300007),
                l_fib_lags=(24, 55),
                ignore_self_edges=True,
            )
        print(f"Solution to Project Euler #186 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 187 in eval_nums:
        since = time.time()
        res = semiPrimeCount(n_max=10 ** 8 - 1)
        print(f"Solution to Project Euler #187 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 188 in eval_nums:
        since = time.time()
        res = modTetration(base=1777, tetr=100, md=10 ** 8)
        print(f"Solution to Project Euler #188 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 189 in eval_nums:
        since = time.time()
        res = numberOfTriangularGridColourings(n_colours=3, n_rows=8)
        print(f"Solution to Project Euler #189 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 190 in eval_nums:
        since = time.time()
        res = sumFloorMaximisedRestrictedPowerProduct(n_min=2, n_max=15)
        print(f"Solution to Project Euler #190 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 191 in eval_nums:
        since = time.time()
        res = attendancePrizeStringCount(n_days=30, n_consec_absent=3, n_late=2)
        print(f"Solution to Project Euler #191 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 192 in eval_nums:
        since = time.time()
        res = bestSqrtApproximationsDenominatorSum(n_max=10 ** 5, denom_bound=10 ** 12)
        print(f"Solution to Project Euler #192 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 193 in eval_nums:
        since = time.time()
        res = squareFreeNumberCount(n_max=2 ** 50 - 1)
        print(f"Solution to Project Euler #193 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 194 in eval_nums:
        since = time.time()
        res = allowedColouredConfigurationsCount(type_a_count=25, type_b_count=75, n_colours=1984, md=10 ** 8)
        print(f"Solution to Project Euler #194 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 195 in eval_nums:
        since = time.time()
        res = integerSideSixtyDegreeTrianglesWithMaxInscribedCircleRadiusCount(radius_max=1053779)
        print(f"Solution to Project Euler #195 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 196 in eval_nums:
        since = time.time()
        res = numberTrianglePrimeTripletRowsSum(row_nums=[5678027, 7208785])
        print(f"Solution to Project Euler #196 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 197 in eval_nums:
        since = time.time()
        res = findFloorRecursiveSequenceTermSum(term_numbers=[10 ** 12, 10 ** 12 + 1], u0=-1, base=2, a=-1., b=0., c=30.403243784, div=10 ** 9)
        print(f"Solution to Project Euler #197 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 198 in eval_nums:
        since = time.time()
        res = ambiguousNumberCountUpToFraction(max_denominator=10 ** 8, upper_bound=(1, 100), incl_upper_bound=False)
        #res = ambiguousNumberCountUpToReciprocal(max_denominator=10 ** 8, upper_bound_reciprocal=100, incl_upper_bound=False)
        print(f"Solution to Project Euler #198 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 199 in eval_nums:
        since = time.time()
        res = IterativeCirclePackingUncoveredAreaProportion(n_iter=10)
        print(f"Solution to Project Euler #199 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 200 in eval_nums:
        since = time.time()
        res = findNthPrimeProofSqubeWithSubstring(substr_num=200, substr_num_lead_zeros_count=0, n=200, base=10)
        print(f"Solution to Project Euler #200 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    return

if __name__ == "__main__":
    eval_nums = {155}
    evaluateProjectEulerSolutions151to200(eval_nums)