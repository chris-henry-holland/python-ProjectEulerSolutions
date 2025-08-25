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
    Hashable,
)

import bisect
import copy
import heapq
import itertools
import math
import os
import random
import time

import numpy as np

from sortedcontainers import SortedList, SortedDict

from graph_classes import GridWeightedDirectedGraph, Grid

from data_structures.prime_sieves import SimplePrimeSieve, PrimeSPFsieve

from algorithms.number_theory_algorithms import gcd, isqrt
from algorithms.random_selection_algorithms import uniformRandomDistinctIntegers
from algorithms.string_searching_algorithms import AhoCorasick

from project_euler_solutions.utils import (
    loadTextFromFile,
    loadStringsFromFile,
)
from project_euler_solutions.Project_Euler_1_50 import triangleMaxSum

Real = Union[int, float]

# Problem 51
def smallestPrimeWhenReplacingGivenDigits(
    dig_lst: List[int],
    dig_replace_inds: List[int],
    base: int=10,
    p_sieve: Optional[SimplePrimeSieve]=None,
    disallowed_last_dig: Optional[Set[int]]=None,
) -> int:
    """
    Finds the smallest prime which has len(dig_lst) digits in the
    chosen base with no leading zeroes, and (indexing the digits
    of its representation in the chosen base from left to right
    starting at 0), for each index i that does not appear in
    dig_replace_inds, the ith digit equals dig_lst[i].
    
    Args:
        Required positional:
        dig_lst (list of ints): List containing digits between 0
                and (base - 1) inclusive, representing the
                digits that the representation of the output
                in the chose base must match (excluding those
                with index in dig_replace_inds)
        dig_replace_inds (list of ints): List of integers between
                0 and (len(dig_lst) - 1) in strictly increasing
                order representing the indices where the digits of
                the representation of the output in the chosen
                base are not required to match the corresponding
                index of dig_lst and may be replaced by another
                digit.
        
        Optional named:
        base (int): Integer strictly greater than 1 giving the base
                in which the primes are to be represented.
            Default: 10
        p_sieve (SimplePrimeSieve or None): A SimplePrimeSieve object
                used to facilitate assessment of whether a
                number is prime. If not given or given as None, a
                new SimplePrimeSieve is created. By specifying
                this and using the same prime sieve object in
                multiple calculation and functions, it can prevent
                the same calculations from being repeated.
            Default: None
        disallowed_last_dig (set of ints or None): If given
                should be None or the set of digits which no
                prime with at least two digits when expressed
                in the chosen base has a final digit in this
                set. This is the set of integers between 2 and
                (base - 1) inclusive which are not coprime with
                base. For example, for base 10, this is the
                set {2, 4, 5, 6, 8} (i.e. all integers from 2 to
                9 inclusive divisible by 2 and 5, the prime
                factors of 10).
                If not given or is given as None, and len(dig_lst)
                is more than one, this set is calculated from
                scratch.
                Inclusion of this as an option is intended to avoid
                repeated calculation of this set if it is needed in
                more than one place in the larger calculation.
            Default: None

    Returns:
    Integer (int) representing the smallest prime that satisfies
    the conditions stated above. If no such prime exists then
    returns -1.
    """
    # Assumes without checking that dig_lst contains only integers
    # between 0 and (base - 1) inclusive and permut_inds only is a
    # strictly increasing list of integers between 0 and
    # (len(dig_lst) - 1) inclusive.
    n_replace = len(dig_replace_inds)
    n_dig = len(dig_lst)
    if not n_dig: return -1
    elif not dig_lst[0] and (not n_replace or dig_replace_inds[0]):
        return -1
    if p_sieve is None: p_sieve = SimplePrimeSieve()
    if not n_replace:
        res = 0
        for d in reversed(dig_lst):
            res = base * res + d
        return res if p_sieve.isPrime(res) else -1
    ps_mx = base ** n_dig - 1 if dig_replace_inds and\
            not dig_replace_inds[0]\
            else (dig_lst[0] + 1) * base ** (n_dig - 1) - 1
    p_sieve.extendSieve(ps_mx)
    if n_dig == 1:
        res = 2
        return res if res < base else -1
    #print(p_sieve.p_lst[-1])
    if dig_replace_inds[-1] == n_dig - 1:
        if disallowed_last_dig is None:
            disallowed_last_dig = set()
            for p in range(2, base):
                if p * p > base: break
                elif base % p: continue
                for d in range(p, base, p):
                    disallowed_last_dig.add(d)
        if dig_lst[1] in disallowed_last_dig:
            return -1
        last_dig_iter = lambda: [x for x in range(base) if x not in\
                disallowed_last_dig]
    else:
        last_dig_iter = lambda: range(base)
    for num in range(base ** (n_replace - 2) if not dig_replace_inds[0]\
            else 0, base ** (n_replace - 1)):
        vals = [0] * n_replace
        num2 = num
        for i in range(1, n_replace):
            num2, d2 = divmod(num2, base)
            vals[~i] = d2
            if not num2: break
        for last_dig in last_dig_iter():
            vals[-1] = last_dig
            p = 0
            j = 0
            for j2, val in zip(dig_replace_inds, vals):
                for j in range(j, j2):
                    p = p * base + dig_lst[j]
                p = p * base + val
                j = j2 + 1
            
            for j in range(j, len(dig_lst)):
                p = p * base + dig_lst[j]
            #print(vals, dig_lst, p)
            if p_sieve.isPrime(p):
                return p
    return -1

def primeDigitReplacementFamilies(
    n_dig: int,
    family_min_n_primes: int,
    base: int=10,
    p_sieve: Optional[SimplePrimeSieve]=None,
    prime_disallowed_last_dig: Optional[Set[int]]=None,
) -> List[Tuple[List[int], List[int], List[int]]]:
    """
    Calculates all n_dig digit prime digit replacement families
    for the chosen base with at least family_min_n_prime
    members, returning for each family every member, and the
    values and indices of the replaced digits.
    
    A n_dig digit prime digit replacement family for a given
    base is defined to be the set of prime numbers which,
    when expressed in that base contain n_dig digits (without
    leading zeros) and these representations differ by each
    other at specific digit positions, with for each prime
    the digits at those positions all having the same value
    (an necessarily a different value from all of the other
    primes). The number of members of the family is the number
    of such primes that exist for that number of digits (in
    the chosen base) and digit positions.
    
    Args:
        Required positional:
        n_dig (int): The number of digits (without leading zeros)
                that the primes in the returned families should
                all have when represented in the chosen base.
        family_min_n_primes (int): The minimum number of members
                of an n_dig digit prime digit replacement family
                for the chosen base required for it to be included.
        
        Optional named:
        base (int): Integer strictly greater than 1 giving the base
                in which the primes are to be represented.
            Default: 10
        p_sieve (SimplePrimeSieve or None): A SimplePrimeSieve object
                used to facilitate assessment of whether a
                number is prime. If not given or given as None, a
                new SimplePrimeSieve is created. By specifying
                this and using the same prime sieve object in
                multiple calculation and functions, it can prevent
                the same calculations from being repeated.
            Default: None
        disallowed_last_dig (set of ints or None): If given
                should be None or the set of digits which no
                prime with at least two digits when expressed
                in the chosen base has a final digit in this
                set. This is the set of integers between 2 and
                (base - 1) inclusive which are not coprime with
                base. For example, for base 10, this is the
                set {2, 4, 5, 6, 8} (i.e. all integers from 2 to
                9 inclusive divisible by 2 and 5, the prime
                factors of 10).
                If not given or is given as None, and len(dig_lst)
                is more than one, this set is calculated from
                scratch.
                Inclusion of this as an option is intended to avoid
                repeated calculation of this set if it is needed in
                more than one place in the larger calculation.
            Default: None
    
    Returns:
    List of 3-tuples of lists representing the n_dig digit prime
    digit replacement families for the chosen base with at least
    family_min_n_prime members. Each 3-tuple represents one such
    family, where:
     - Index 0 contains a list of ints giving the prime numbers
       in the family in increasing order.
     - Index 1 contains a list of ints between 0 and (base - 1)
       inclusive with the same length as the list in index 0, each
       giving the values of the replacing digits for the
       corresponding prime in the list in index 0
     - Index 2 contains a list of ints giving the indices of the
       replaced digits for the primes when expressed in the
       chosen base without leading zeros, when read from right to
       left starting at index 0.
    The families are provided in order of the smallest prime in
    the family in increasing order (with ties resolved by the order
    of the second smallest prime in the family).
    """
    #since = time.time()
    if family_min_n_primes > base: return []
    if p_sieve is None:
        p_sieve = SimplePrimeSieve(base ** n_dig - 1)
    else:
        p_sieve.extendSieve(base ** n_dig - 1)
    if n_dig == 1:
        i2 = bisect.bisect_left(p_sieve.p_lst, base)
        return [(p_sieve.p_lst[:i2], [0])] if i2 >= family_min_n_primes else []
    if prime_disallowed_last_dig is None:
        base_p_factors = p_sieve.primeFactors(base)
        prime_disallowed_last_dig = set()
        for p in base_p_factors:
            for i in range(p, base, p):
                prime_disallowed_last_dig.add(i)

    def candidatesGenerator(
        n: int,
        mn_lst_sz: int=0,
        only_poss_primes: bool=False,
        base: int=10,
        prime_disallowed_last_dig: Optional[Set[int]]=None,
    ) -> Generator[Tuple[List[int]], None, None]:
        # Consider adding option to exclude all candidates which give
        # rise to numbers guaranteed to exceed a given number
        dig_lst = []
        n2 = n
        while n2:
            n2, r = divmod(n2, base)
            dig_lst.append(r)
        dig_lst = dig_lst[::-1]
        n_dig = len(dig_lst)
        ind_lsts = [[] for _ in range(base)]
        if only_poss_primes:
            if prime_disallowed_last_dig is None:
                disallowed_last_dig = set()
                for p in range(2, base):
                    if p * p > base: break
                    elif base % p: continue
                    for i in range(p, base, p):
                        disallowed_last_dig.add(i)
            else: disallowed_last_dig = prime_disallowed_last_dig
        else: disallowed_last_dig = set()
        for i, d in enumerate(dig_lst):
            ind_lsts[d].append(i)
        for d in range(base - mn_lst_sz):
            #print(f"d = {d}")
            has_last_dig = only_poss_primes and ind_lsts[d] and\
                            ind_lsts[d][-1] == n_dig - 1
            for bm in range(1, (1 << len(ind_lsts[d]))):
                if has_last_dig and (bm & (1 << (len(ind_lsts[d]) - 1))):
                    dig_vals = [d]
                    for d2 in range(d + 1, base):
                        if has_last_dig and d2 in disallowed_last_dig:
                            continue
                        dig_vals.append(d)
                    if len(dig_vals) < mn_lst_sz: break
                else: dig_vals = list(range(d, base))
                dig_inds = []
                for j in range(len(ind_lsts[d])):
                    if not bm & (1 << j): continue
                    dig_inds.append(ind_lsts[d][j])
                yield (dig_lst, dig_vals, dig_inds)
        return


    #i = 0
    res = []
    
    #curr = None
    i1 = bisect.bisect_left(p_sieve.p_lst, base ** (n_dig - 1))
    i2 = bisect.bisect_left(p_sieve.p_lst, base ** (n_dig), lo=i1)
    #print(i1, i2, p_sieve.p_lst)
    for i in range(i1, i2):
        p = p_sieve.p_lst[i]
        for dig_lst, dig_vals, dig_inds in candidatesGenerator(
            p, mn_lst_sz=family_min_n_primes,
            only_poss_primes=True,
            base=base,
            prime_disallowed_last_dig=prime_disallowed_last_dig,
        ):
            p_lst = [p]
            dig_vals2 = [dig_vals[0]]
            for d_i in range(1, len(dig_vals)):
                d = dig_vals[d_i]
                idx = 0
                p2 = 0
                for j, d2 in enumerate(dig_lst):
                    if j == dig_inds[idx]:
                        p2 = base * p2 + d
                        idx += 1
                        if idx == len(dig_inds): break
                    else: p2 = base * p2 + d2
                for j in range(j + 1, n_dig):
                    p2 = base * p2 + dig_lst[j]
                if p_sieve.isPrime(p2):
                    p_lst.append(p2)
                    dig_vals2.append(d)
                #idx = bisect.bisect_left(ps.p_lst, p2, lo=i)
                #if idx < i2 and ps.p_lst[idx] == p2:
                #    p_lst.append(p2)
            if len(p_lst) >= family_min_n_primes and p_lst[0] == min(p_lst):
                #print(p_lst, dig_lst, dig_vals, dig_vals2, dig_inds)
                res.append((p_lst, dig_vals2, dig_inds))
            """
            elif ans_in_family:
                res = [p_lst[0], p_lst]
                break
            p2 = smallestPrimeGivenDigits(
                p_lst,
                dig_inds,
                base=base,
                p_sieve=ps,
                disallowed_last_dig=prime_disallowed_last_dig,
            )
            if p2 < res[0]:
                res = [p2, p_lst]
                curr = [0] * n_dig
                for i in range(n_dig):
                    p2, r = divmod(p2, base)
                    curr[~i] = r
            """
        #else: continue
        #break
    #if isinstance(res[0], int):
    #    break
    #i = i2
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    #return res if isinstance(res[0], int) else ()
    return sorted(res)

def smallestPrimeDigitReplacementsPrime(
    family_min_n_primes: int=8,
    base: int=10,
    n_dig_max: Optional[int]=None,
    ans_in_family: bool=True,
) -> int:
    """
    Solution to Project Euler #51

    Calculates the smallest prime number for which there
    exists an n_dig digit prime digit replacement family
    for the chosen base with at least family_min_n_primes
    members (where n_dig is the number of digits in the
    prime when represented in the chosen base without leading
    zeros), where the representation of the prime number
    in the chosen base only differs from that of the prime
    numbers in the family in the digit positions for which
    they differ from each other, and if ans_in_family is True,
    the prime is a member of the family.
    
    A n_dig digit prime digit replacement family for a given
    base is defined to be the set of prime numbers which,
    when expressed in that base contain n_dig digits (without
    leading zeros) and these representations differ by each
    other at specific digit positions, with for each prime
    the digits at those positions all having the same value
    (an necessarily a different value from all of the other
    primes). The number of members of the family is the number
    of such primes that exist for that number of digits (in
    the chosen base) and digit positions.
    
     Args:
        Optional named:
        family_min_n_primes (int): The minimum number of members
                of an n_dig digit prime digit replacement family
                for the chosen base required for it to be included.
            Default: 8
        base (int): Integer strictly greater than 1 giving the base
                in which the primes are to be represented.
            Default: 10
        n_dig_max (int or None): If given as a strictly positive
                integer, the maximum number of digits in the
                representation in the chosen base of prime numbers
                considered. Otherwise, there is no such restriction
                and if there is no reason for the prime not to
                exist (for instance, due to family_min_n_primes
                exceeding base), there is no upper bound on the
                search.
            Default: None
        ans_in_family (bool): Whether the prime is required to be
                a member of the corresponding family.
            Default: True
    
    Returns:
    Integer (int) giving the smallest prime number satisfying the
    described conditions with (if n_dig_max is given as a strictly
    positive integer) no more than n_dig_max in its representation
    in the chosen base, if such a prime number exists. Otherwise,
    returns -1.
    """
    p_sieve = PrimeSPFsieve(base)
    base_p_factors = p_sieve.primeFactors(base)
    prime_disallowed_last_dig = set()
    for p in base_p_factors:
        for i in range(p, base, p):
            prime_disallowed_last_dig.add(i)
    it = itertools.count(1) if n_dig_max is None else range(1, n_dig_max + 1)
    for n_dig in it:
        #print(f"n_dig = {n_dig}")
        lsts = primeDigitReplacementFamilies(
            n_dig,
            family_min_n_primes,
            base=base,
            p_sieve=p_sieve,
            prime_disallowed_last_dig=prime_disallowed_last_dig,
        )
        if lsts: break
    else: return -1
    #print(lsts)
    
    if ans_in_family:
        return min(lsts)[0][0]
    
    res = float("inf")
    for p_lst, dig_vals, dig_inds in lsts:
        dig_lst = []
        p2 = p_lst[0]
        for _ in range(n_dig):
            p2, d = divmod(p2, base)
            dig_lst.append(d)
        dig_lst = dig_lst[::-1]
        p2 = smallestPrimeWhenReplacingGivenDigits(
            dig_lst,
            dig_inds,
            base=base,
            p_sieve=p_sieve,
            disallowed_last_dig=prime_disallowed_last_dig,
        )
        res = min(res, p2)
    return res
    """
    for dig_lst, dig_vals, dig_inds in lsts:
        d0 = min(dig_vals)
        j = 0
        num = 0
        for i, d in enumerate(dig_lst):
            if j < len(dig_inds) and i == dig_inds[j]:
                num = num * base + d0
                j += 1
                continue
            num = num * base + d
        res = min(res, num)
    return res
    """

# Problem 52
def digitFrequency(num: int, base: int=10) -> Dict[int, int]:
    """
    Finds the frequency of each digit of non-negative integer num
    in its representation in the chosen base.
    
    Args:
        Required positional:
        num (int): Non-negative integer of which the digit frequency
                is being calculated
        
        Optional named:
        base (int): The base in which num is to be represented
            Default: 10
    
    Returns:
    Dictionary (dict) whose keys are the digits which appear in
    the representation of num in the chosen base and whose
    corresponding values are the number of times this digit
    appears in the representation.
    """
    if not num: return {0: 1}
    res = {}
    while num:
        num, r = divmod(num, base)
        res[r] = res.get(r, 0) + 1
    return res

def digitFrequencyComp(num: int, comp: Dict[int, int], base: int=10) -> bool:
    """
    Finds whether the representation of a non-negative integer in the
    chosen base contains the same digits as another number whose digit
    frequency in that base has been calculated (and given by comp). In
    other words, finds whether the representation of num in the chosen
    base is a permutation of the digits of the representation of another
    non-negative integer.
    
    Args:
        Required positional:
        num (int): Non-negative integer whose digit in the chosen base
                are being compared.
        comp (dict with int keys and values): Frequency dictionary of
                the digits of non-negative integer in the chosen base
                with which the digits of num are being compared
        
        Optional named:
        base (int): Strictly positive integer giving the base in which
                both numbers are to be represented.
            Default: 10
    
    Returns:
    Boolean (bool) representing whether the digits of both non-negative
    integers contain the same digits in their representations in the
    chosen base.
    """
    if not num: return comp == {0: 1}
    res = {}
    while num:
        num, r = divmod(num, base)
        res[r] = res.get(r, 0) + 1
        if res[r] > comp.get(r, 0): return False
    return res == comp

def permutedMultiples(
    n_permutes: int=6,
    n_dig_mx: int=10,
    base: int=10,
) -> int:
    """
    Solution to Project Euler #52

    Searches for the smallest strictly positive integer n such that the
    digits of the representations of i * x for 1 <= i <= n_permutes
    in the chosen base are all composed of exactly the same collection
    of digits.
    
    Args:
        Optional named:
        n_permutes (int): The largest multiple (in the above description
                largest value of i) considered.
            Default: 6
        n_dig_mx (int): The maximum number of digits in the chosen base
                of all numbers searched (to ensure the search terminates)
            Default: 10
        base (int): Strictly positive integer giving the base in which
                all of the numbers are to be represented.
            Default: 10
    
    Returns:
    Integer (int) giving the smallest strictly positive integer which
    satisifes the described requirements.
    If it is certain that no solution exists then returns 0.
    If no solution is found in the numbers searched (i.e. up to
    base ** n_dig_mx - 1) then returns -1
    """
    #since = time.time()
    if n_permutes > base: return 0
    for n_dig in range(1, n_dig_mx + 1):
        rng = (base ** (n_dig - 1), -((-base ** n_dig) // n_permutes))
        for num in range(*rng):
            f_dict = digitFrequency(num, base=base)
            num2 = num
            for i in range(2, n_permutes + 1):
                num2 += num
                if not digitFrequencyComp(num2, f_dict, base=base):
                    break
            else:
                #print(f"Time taken = {time.time() - since:.4f} seconds")
                return num
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return -1

# Problem 53
def combinatoricSelections(
    n_mn: int=1,
    n_mx: int=100,
    cutoff: int=10 ** 6,
) -> int:
    """
    Solution to Project Euler #53

    Calculates the number of binomial coefficients of the form
    n choose r such that n is between n_mn and inclusive n_mx
    that have value at least cutoff,.
    
    Args:
        Optional named:
        n_mn: The smallest value of n for the binomial coefficients
                n choose r that are included
            Default: 1
        n_mx: The largest value of n for the binomial coefficients
                n choose r that are included
            Default: 100
        cutoff: The value which the binomial coefficients are to
                strictly greater than to be counted in the total.
            Default: 10 ** 6
    
    Returns:
    Non-negative integer (int) representing the number of binomial
    coefficients that satisfy the above conditions.
    """
    #since = time.time()
    if n_mn > n_mx: return 0
    n_mn = max(n_mn, 0)
    if not cutoff:
        return ((n_mx + 1) * (n_mx + 2) - n_mn * (n_mn + 1)) >> 1
    elif n_mn > cutoff:
        # In this case, for all rows considered, all but the outer
        # two elements are guaranteed to exceed the cutoff, while
        # the outer two elements (which are always 1) are
        # guaranteed not to. Therefore, the row corresponding
        # to integer n between n_mn and n_mx inclusive contributes
        # exactly n - 1 to the sum.
        return (n_mx * (n_mx - 1) - (n_mn - 1) * (n_mn - 2)) >> 1
    curr = [1]
    # Iterating over rows of Pascal's triangle, producing roughly
    # half of each row until find a value that exceeds the cutoff
    for i in range(1, min(n_mx, cutoff) + 1, 2):
        prev = curr
        curr = [0] * len(prev)
        curr[0] = 1
        for j in range(len(prev) - 1):
            curr[j + 1] = prev[j] + prev[j + 1]
            if curr[j + 1] > cutoff:
                i2 = i
                break
        else:
            #print(curr, 0)
            prev = curr
            curr = [0] * (len(prev) + 1)
            curr[0] = 1
            for j in range(len(prev) - 1):
                curr[j + 1] = prev[j] + prev[j + 1]
                if curr[j + 1] > cutoff:
                    break
            curr[-1] = prev[-1] << 1
            if curr[-1] <= cutoff: continue
            i2 = i + 1
        break
    else:
        #print(f"Time taken = {time.time() - since:.4f} seconds")
        return 0
    while curr and not curr[-1] or curr[-1] > cutoff:
        curr.pop()
    res = i2 - (len(curr) << 1) + 1 if i2 >= n_mn else 0
    # Finding the index of the first element in each row of
    # Pascal's triangle that exceeds the cutoff. Using the
    # fact that a row of Pascal's triangle is unimodal and
    # symmetric, if the row n-value is at least n_mn then
    # if the index of the first element exceeding the cutoff
    # is j for row i, then the number of elements in the row
    # exceedig the cutoff for row n is exactly (i + 1 - j * 2).
    # Repeats until gets to row n_mx or all elements of the
    # rows of Pascal's triangle other than the outer two
    # exceed the cutoff.
    for i in range(i2 + 1, min(n_mx, cutoff) + 1):
        prev = curr
        curr = [1]
        for j in range(len(prev) - 1):
            term = prev[j] + prev[j + 1]
            if term > cutoff:
                break
            curr.append(term)
        res += i - (len(curr) << 1) + 1 if i >= n_mn else 0
    if cutoff < n_mx:
        # Summing over the remaining rows, using the fact that
        # once the second element in a row exceeds the cutoff,
        # as long as the cutoff is not zero (which has been
        # handled) all elements of this and every subsequent
        # row except the first and last exceed the cutoff, 
        # while the first and last (which are always 1) will
        # never exceed the cutoff for any row. Thus for this
        # and every subsequent row the number of elements
        # exceeding the cutoff is exactly the row length minus
        # two, i.e. n - 1.
        # Note that given the case n_mn > cutoff handled at
        # the beginning, if this stage has been reached the
        # rows being considered here are all guaranteed to be
        # at least at row n_mn, so they are all counted
        res += (n_mx * (n_mx - 1) - (cutoff) * (cutoff - 1)) >> 1
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 54
class PokerHand(object):
    """
    Class representing poker hands so that they can be directly
    compared.
    This defines the dunder methods __lt__() (i.e. <) and __le__()
    (i.e. <=) which when used to compare two poker hands indicates
    whether the first hand is strictly worse than (i.e. loses to)
    the second and whether the first hand is no better than (i.e.
    does not win against, but may tie with) the second respectively.
    Note that two hands are considered equal only if they contain
    exactly the same cards, so __eq__() (i.e. ==) should not be used
    to check if two hands have the same value (i.e. the two hands
    would tie). This could however be accomplished by checking that
    the first is less than or equal to the second and the second
    is less than or equal to the first.
    
    Initialisation args:
        Required positional:
        hand (5-tuple of strs): Defines the cards in the poker hand
                being represented and sets the attribute hand.
                Each element of the tuple should represent a card and
                the identity of the card is encoded as a string which
                consists of a pair of alphanumeric characters, where
                the first characters is a digit from 2 to 9 inclusive
                representing that the card has that numeric value, or
                the upper case alphabet character 'T', 'J', 'Q', 'K' or
                'A' representing that the card is a 10, Jack, Queen,
                King or Ace respectively, and the second character is
                an upper case character 'C', 'D', 'H' or 'S'
                representing that the card is a club, diamond, heart or
                spade respectively.
    
    Attributes:
        hand (5-tuple of strs): Tuple containing the five cards in the
                poker hand the instance represents, with each card
                encoded as a string as outlined in the description
                of Initialisation argument hand. The initialisation
                argument hand sets this attribute.
        hand_set (set of strs): Set containing the same elements as the
                fellow attribute hand (i.e. the set of cards in the
                hand encoded as strings as outlined in the description
                of Initialisation argument hand).
        hand_repr (tuple of ints): Encodes the strength of the poker
                hand represented in such a way that enables comparison
                of different hands. The different possible hands are
                encoded as follows (where the numeric value of a card
                is its number for numbered cards, 11 for Jack, 12 for
                Queen, 13 for King and 14 for Ace):
                 - High card: 6-tuple whose index 0 contains the value
                        0 and whose remaining entries are the numeric
                        values of the cards in the hands in decreasing
                        order of size.
                 - Pair: 5-tuple whose index 0 contains the value 1,
                        index 1 contains the numeric value of the
                        cards in the pair and whose remaining entries
                        contain the numeric value of the remaining
                        cards in decreasing order of size.
                 - Two pair: 4-tuple whose index 0 contains the value
                        2, index 1 and 2 contain the larger and smaller
                        numeric values of the cards in the two pairs
                        respectively and index 3 contains the numeric
                        value of the remaining card.
                 - Three of a kind: 4-tuple whose index 0 contains the
                        value 3, index 1 contains the numeric value of
                        the cards in the triple and whose remaining
                        entries contain the numeric value of the
                        remaining cards in decreasing order of size.
                 - Straight: 2-tuple whose index 0 contains the value
                        4 and index 1 contains the largest numeric
                        value of the cards in the hand (noting that
                        in the case of an Ace to 5 straight then, as
                        the Ace is unambiguously being used as a 1,
                        this value will be 5 rather than 14).
                 - Flush: 6-tuple whose index 0 contains the value
                        5 and whose remaining entries are the numeric
                        values of the cards in the hands in decreasing
                        order of size.
                 - Full house: 3-tuple whose index 0 contains the value
                        6, index 1 contains the numeric value of the
                        cards in the triple and index 2 contains the
                        numeric value of the cards in the pair.
                 - Four of a kind: 2-tuple whose index 0 contains the
                        value 7, index 1 contains the numeric value of
                        the cards in the quadruple and index 2 contains
                        the numeric value of the remaining card.
                 - Straight/royal flush: 2-tuple whose index 0 contains
                        the value 8 and index 1 contains the largest
                        numeric value of the cards in the hand (noting
                        that in the case of an Ace to 5 straight then,
                        as the Ace is unambiguously being used as a 1,
                        this value will be 5 rather than 14).
                This representation of the strength of the hands has
                been designed such that, when this attribute is
                compared between two hands, the result of the
                comparison between the tuples (e.g. less than or less
                than or equal to) gives the same result as would be
                expected when comparing the relative strengths of
                the corresponding hands (e.g. whether the first would
                lose to or not beat the other).
    """
    val_dict = {str(x): x for x in range(2, 10)}
    val_dict["T"] = 10
    val_dict["J"] = 11
    val_dict["Q"] = 12
    val_dict["K"] = 13
    val_dict["A"] = 14
    
    str_dict = {str(x): str(x) for x in range(2, 10)}
    str_dict["T"] = "10"
    str_dict["J"] = "J"
    str_dict["Q"] = "Q"
    str_dict["K"] = "K"
    str_dict["A"] = "A"
    
    def __init__(self, hand: Tuple[str]):
        self.hand = hand
        self.hand_set = set(hand)
    
    @property
    def hand_repr(self):
        if hasattr(self, "_hand_repr"):
            return self._hand_repr
        res = self._handRepr()
        self._hand_repr = res
        return res
    
    def __str__(self):
        return "Poker hand " + ", ".join(\
                [f"{self.str_dict[x[0]]}{x[1]}" for x in self.hand])
    
    def _handRepr(self) -> Tuple[int]:
        vals = sorted(self.val_dict[x[:-1]] for x in self.hand)
        is_flush = (len(set(x[-1] for x in self.hand)) == 1)
        is_straight = all(vals[i] + 1 == vals[i + 1] for i in range(4))\
                or (vals[-1] == 14 and all(vals[i] == i + 2 for i in range(4)))
        
        # Straight/Royal flush
        if is_flush and is_straight:
            return (8, vals[-1]) if vals[-1] != 14 or vals[0] != 2\
                    else (8, vals[-2])
        
        # Four of a kind
        if vals[0] == vals[3]:
            return (7, vals[0], vals[4])
        elif vals[1] == vals[4]:
            return (7, vals[1], vals[0])
        
        # Full house
        if vals[0] == vals[1] and vals[2] == vals[4]:
            return (6, vals[4], vals[0])
        elif vals[0] == vals[2] and vals[3] == vals[4]:
            return (6, vals[0], vals[4])
        
        # Flush
        if is_flush: return (5, *vals[::-1])
        
        # Straight
        if is_straight:
            return (4, vals[-1]) if vals[-1] != 14 or vals[0] != 2\
                    else (4, vals[-2])
        
        # Three of a kind
        if vals[0] == vals[2]: return (3, vals[0], vals[4], vals[3])
        elif vals[1] == vals[3]: return (3, vals[1], vals[4], vals[0])
        elif vals[2] == vals[4]: return (3, vals[2], vals[1], vals[0])
        
        # Two pair and One pair
        if vals[0] == vals[1]:
            if vals[2] == vals[3]:
                return (2, vals[2], vals[0], vals[4])
            elif vals[3] == vals[4]:
                return (2, vals[3], vals[0], vals[2])
            else: return (1, vals[0], vals[4], vals[3], vals[2])
        elif vals[1] == vals[2]:
            if vals[3] == vals[4]:
                return (2, vals[3], vals[1], vals[0])
            else: return (1, vals[1], vals[4], vals[3], vals[0])
        for i in range(2, 4):
            if vals[i] != vals[i + 1]: continue
            res = [1, vals[i]]
            for j in reversed(range(i + 2, 5)):
                res.append(vals[j])
            for j in reversed(range(i)):
                res.append(vals[j])
            return tuple(res)
        
        # High card
        return (0, *vals[::-1])
        
    def __eq__(self, other) -> bool:
        if not isinstance(other, PokerHand):
            return False
        return self.hand_set == other.hand_set
    
    def __lt__(self, other) -> bool:
        if not isinstance(other, PokerHand):
            raise TypeError("An instance of PokerHand may only "
                    "have its size compared with another "
                    "instance of PokerHand.")
        return self.hand_repr < other.hand_repr
    
    def __le__(self, other) -> bool:
        if not isinstance(other, PokerHand):
            raise TypeError("An instance of PokerHand may only "
                    "have its size compared with another "
                    "instance of PokerHand.")
        return self.hand_repr <= other.hand_repr

def loadPokerHands(
    doc: str,
    rel_package_src: bool=False,
) -> List[Tuple[Tuple[str]]]:
    """
    From the .txt file at the relative or absolute location doc
    containing a list of pairs of poker hands, returns these hands.
    Each pair of poker hands in the file should be separated by
    a line break ('\\n') and consist of ten pairs of alphanumeric
    characters separated by single spaces (' '), with each pair
    representing a card, and each such pair consisting of a digit
    from 2 to 9 inclusive representing that the card has that numeric
    value or the upper case alphabet character 'T', 'J', 'Q', 'K' or
    'A' representing that the card is a 10, Jack, Queen, King or
    Ace respectively, followed by an upper case character 'C',
    'D', 'H' or 'S' representing that the card is a club, diamond,
    heart or spade respectively.
    In each line (representing a pair of hands), the first five pairs
    and second five pairs of alphanumeric characters represent the
    cards in the first hand and second hand respectively.
    
    Args:
        Required positional:
        doc (str): Relative or absolue path to the .txt file containing
                the pairs of poker hands.
        
        Optional named:
        rel_package_src (bool): Whether a relative path given by doc
                is relative to the current directory (False) or the
                package directory (True).
            Default: False
    
    Returns:
    List of 2-tuples of 5-tuples of strings, where the elements of the
    list represent each of the pairs of poker hands contained in the
    .txt file at the location doc (in the same order). For each pair
    of poker hands, the corresponding element of the list is a 2-tuple
    with indices 0 and 1 containing a 5-tuple of strings (str)
    representing the cards in the first and second of the pair of
    hands respectively. Each element of these 5-tuples is a string
    representing a card in the represented hand, with the card values
    encoded as a pair of alphanumeric characters the manner described
    above.
    """
    #doc = doc.strip()
    #if not os.path.isfile(doc):
    #    raise FileNotFoundError(f"There is no file at location {doc}")
    #with open(doc) as f:
    #    txt = f.read()
    txt = loadTextFromFile(doc, rel_package_src=rel_package_src)
    res = []
    for i, line in enumerate(txt.split("\n"), start=1):
        if not line: continue
        cards = line.split(" ")
        if len(cards) != 10:
            raise ValueError(f"Incorrect number of cards in line {i}: "
                    f"{line}. Expected 10 cards.")
        res.append((tuple(cards[:5]), tuple(cards[5:])))
    return res
        
def numberOfPokerHandsWon(
    hand_file: str="project_euler_problem_data_files/p054_poker.txt",
    rel_package_src: bool=True,
) -> int:
    """
    Solution to Project Euler #54

    Function that takes a list of pairs of poker hands given by the
    .txt file at location specified by input argument hand_file and
    calculates the number of these pairs for which if the two hands in
    each pair were put against each other head-to-head the first hand
    of the pair would beat the second.
    The pairs of poker hands given in the file at location hand_file
    should be formatted as outlined in the documentation of
    loadPokerHands().
    
    Args:
        Optional named:
        hand_file (str): String representing the absolute or relative
                location of the .txt file containing the pairs of
                poker hands.
            Default: "project_euler_problem_data_files/p054_poker.txt"
        rel_package_src (bool): Whether a relative path given by
                hand_file is relative to the current directory (False)
                or the package src directory (True).
            Default: True
    
    Returns:
    Integer (int) giving the number of pairs of hands in the .txt
    file at location hand_file for which the hand that appears first
    in the pair beats the hand that appears second in the pair.
    """
    #since = time.time()
    res = 0
    for i, (hand1_cards, hand2_cards) in enumerate(
        loadPokerHands(hand_file, rel_package_src=rel_package_src),
        start=1,
    ):
        hand1 = PokerHand(hand1_cards)
        hand2 = PokerHand(hand2_cards)
        res += hand1 > hand2
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 55
def reverseAdd(num: int, base: int=10) -> int:
    """
    Performs the reverse-add function to non-negative integer
    num in the chosen base. The reverse-add function takes the
    representation of non-negative integer num in the
    chosen base, forms a new non-negative integer by reversing
    the digits of this representation of num and treating this
    as the representation of a new number, num2 in the chosen
    base, then adds these two numbers together.
    
    Args:
        Required positional:
        num (int): The non-negative integer to which the reverse-add
                function is to be applied.
        
        base (int): Strictly positive integer giving the base in which
                num and its reverse are to be represented.
            Default: 10
    
    Returns:
    Integer (int) representing the result of applying the reverse-add
    function to the non-negative integer num.
    """
    num_lst = []
    num2 = num
    while num2:
        num2, r = divmod(num2, base)
        num_lst.append(r)
    mult = 1
    for i in range(len(num_lst)):
        num += num_lst[~i] * mult
        mult *= base
    return num

def isPalindrome(num: int, base: int=10) -> bool:
    """
    Checks whether the non-negative integer num is a palindrome in the
    chosen base (i.e. if the digits in the chosen base read the same
    forwards and backwards).
    
    Args:
        Required positional:
        num (int): The non-negative integer being tested.
        
        Optional named:
        base (int): Strictly positive integer giving the base in which
                num is to be represented.
            Default: 10
    
    Returns:
    Boolean (bool) which is True if num is a palindrome in the chosen
    base, otherwise False.
    """
    num_lst = []
    while num:
        num, r = divmod(num, base)
        num_lst.append(r)
    for i in range(len(num_lst) >> 1):
        if num_lst[i] != num_lst[~i]:
            return False
    return True

def isLychrel(
    num: int,
    seen: Dict[int, int],
    max_iter: int=50,
    base: int=10,
) -> bool:
    """
    Assesses whether a non-negative integer num is not Lychrel or assumed to
    be Lychrel in the chosen base. A number is Lychrel in a given base if and
    only if there does not exist a strictly positive (finite) integer n such
    that starting with num and sequentially applying the reverse-add function
    using that base (see reverseAdd()) a total of n times, the resulting
    number is a palindrome in that base (see isPalindrome()).
    In order for the process to guarantee terminating, for a given base and
    non-negative integer num, if a palindrome in that base is not encountered
    after max_iter or fewer sequential applications of the reverse-add
    function in that base to num, it is assumed that num is Lychrel in the
    chosen base.
    
    Args:
        Required positional:
        num (int): Non-negative integer which is being tested to see if it
                is not Lychrel or is assumed to be Lychrel.
        seen (dict whose keys and values are ints): A dictionary whose keys
                are non-negative integers whose Lychrel status has already
                been assessed. The corresponding value to a given key is
                -1 if the key has been assumed to be Lychrel and a positive
                integer if not, with that integer representing the smallest
                number of sequential applications of the reverse-add function
                before a palindromic number in the chosen base is encountered
                Note that this function updates seen if and when the Lychrel
                status of any new non-negative integers (including num)
                is found.
        
        Optional named:
        max_iter (int): The maximum number of iterations of the reverse-add
                function applied before, if a palindrome has not been
                encountered, num is assumed to be Lychrel.
            Defualt: 50
        base (int): Strictly positive integer giving the base in which
                num is to be represented.
            Default: 10
    
    Returns:
    Boolean (bool) value, with True representing that num is assumed to
    be Lychrel in the chosen base and False that num is definitely not
    Lychrel in the chosen base.
    """
    if num in seen.keys():
        return (seen[num] == -1)
    path = [num]
    for i in range(max_iter):
        num = reverseAdd(num)
        if isPalindrome(num, base=base):
            for j in range(i + 1):
                seen[path[~j]] = j + 1
            #print(f"palindrome found ({num}):")
            #print([(x, seen[x]) for x in path])
            return False
        if num not in seen.keys():
            path.append(num)
            continue
        if seen[num] == -1:
            for num2 in path:
                seen[num2] = -1
            #print("Lychrel encountered:")
            #print([(x, seen[x]) for x in path])
        else:
            #print(f"Non-Lychrel encountered (num = {num}, "
            #        "{seen[num]} moves away from palindrome)")
            add = seen[num] + 1
            for j in range(min(max_iter - add + 1, i + 2)):
                seen[path[-j]] = j + add
            for j in range(max_iter - add + 1, i + 2):
                seen[path[-j]] = -1
        break
    else:
        #print(f"Lychrel found: {path[0]}")
        seen[path[0]] = -1
        return True
    #print([(x, seen[x]) for x in path])
    return seen[num] == -1

def countLychrelNumbers(
    n_max: int=10 ** 4,
    iter_cap: int=50,
    base: int=10,
) -> int:
    """
    Solution to Project Euler #55
    
    Counts the number of strictly positive integers no greater than n_max
    that are assumed to be Lychrel in the chosen base. See documentation
    of isLychrel() and reverseAdd() for more detail. A non-negative integer
    num is assessed not to be Lychrel (and indeed is definitively not
    Lychrel) if there exists a strictly positive integer n < iter_cap such
    that n sequential applications of the reverse-add function in
    the chosen base (see reverseAdd()) to num results in a palindromic
    number in the chosen base (see isPalindrome()). Otherwise, num is
    assumed to be Lychrel in the chosen base.
    
    Args:
        Optional named:
        n_max (int): The strictly positive integer up to which the assumed
                Lychrel numbers in the chosen base are to be counted.
            Default: 10 ** 4
        iter_cap (int): The strictly positive number for which, for a given
                non-negative integer num, if a palindromic number in the
                chosen base is not encountered in fewer than this number
                of sequential applications of the reverse-add function, num
                is assumed to by Lychrel.
            Default: 50
        base (int): Strictly positive integer giving the base chosen for
                the assessment of palindrome status of integers and
                application of the reverse-add function.
            Default: 10
    
    Returns:
    Integer (int) giving the number of strictly positive integers no
    greater than n_max which are assumed to be Lychrel based on the
    specified parameters and assessment process outlined above.
    """
    #since = time.time()
    seen = {}
    res = sum(isLychrel(x, seen, max_iter=iter_cap - 1, base=base)\
            for x in range(1, n_max + 1))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 56
def digitSum(num: int, base: int=10) -> int:
    """
    Finds the sum of digits of a non-negative integer num in a given
    base.
    
    Args:
        Required positional:
        num (int): The integer whose digits are to be summed
        
        Optional named:
        base (int): The base in which the digits of num are to be
                calculated
            Default: 10
    
    Returns:
    Integer (int) with the sum of digits of num in the chosen base.
    """
    res = 0
    while num:
        num, r = divmod(num, base)
        res += r
    return res

def powerfulDigitSum(a_mx: int=99, b_mx: int=99, base: int=10) -> int:
    """
    Solution to Project Euler #56

    Finds the maximum sum of digits in the given base of all numbers
    of the form a ** b where 0 <= a <= a_mx and 0 <= b <= b_mx.
    
    Args:
        Optional named:
        a_mx (int): The largest value of a considered
            Default: 99
        b_mx(int): The largest value of b considered
            Default: 99
        base (int): The base in which the sum of digits is to be
                calculated
            Default: 10
    
    Returns:
    Integer (int) with the largest sum of digits in the given base
    for all the numbers of the given form.
    """
    #since = time.time()
    res = 1
    for a in range(a_mx + 1):
        num = 1
        for b in range(1, b_mx + 1):
            num *= a
            res = max(res, digitSum(num, base=base))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 57
#def gcd(a: int, b: int) -> int:
#    return a if not b else gcd(b, a % b)

def nextRoot2MinusOneConvergent(prev: Tuple[int]) -> Tuple[int]:
    """
    Given a convergent for root 2 minus one, calculates the
    next convergent
    
    Args:
        Required positional:
        prev (2-tuple of ints): The previous convergent of
                root 2 minus one where the 0th index is
                the numerator and 1st index is the
                denominator
    
    Returns:
    The convergent of root 2 minus one following prev,
    represented by a 2-tuple of ints similarly to prev
    """
    return (prev[1], 2 * prev[1] + prev[0])
    #curr = (prev[1], 2 * prev[1] + prev[0])
    #g = gcd(*curr)
    #return tuple(x // g for x in curr)

def countDigits(num: int, base: int=10) -> int:
    """
    Given a non-negative integer num, finds the number
    of digits in a given base
    
    Args:
        Required positional:
        num (int): The non-negative integer of which
                the number of digits is to be calculated
        
        Optional named:
        base (int): The chosen base
            Default: 10
    
    Returns:
    Integer (int) with the number of digits of num in
    the chosen base
    """
    if not num: return 1
    res = 0
    while num:
        num //= base
        res += 1
    return res

def squareRootTwoConvergents(
    n_expansions: int=1000,
    base: int=10,
) -> int:
    """
    Solution to Project Euler #57

    Calculates the number of the first n_expansions convergents of
    the square root of two for which the number of digits in the
    given base of the numerator exceeds that of the denominator.
    
    Args:
        Optional named:
        n_expansions (int): The number of convergents of the
                square root of two being considered/
            Default: 1000
        base (int): The chosen base in which the number of digits
                is to be calculated.
            Default: 10
    
    Returns:
    Integer (int) with the number of the first n_expansions
    convergents of the square root of two for which the number
    of digits in the given base of the numerator exceeds that
    of the denominator.
    """
    #since = time.time()
    curr = (0, 1)
    res = 0
    for _ in range(n_expansions):
        curr = nextRoot2MinusOneConvergent(curr)
        res += (countDigits(sum(curr), base=base) >\
                countDigits(curr[1], base=base))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 58
def spiralDiagonalGenerator() -> Generator[List[int], None, None]:
    """
    Suppose the natural numbers (starting with 1) are arranged in
    a square spiral. This generator yields the numbers on the diagonals
    of this spiral one layer at a time moving outwards.   
        
    Yields:
    List of ints giving the values on the diagonals of the current
    layer starting at 1 and moving out one layer after each yielded
    value. Note that the first layer yields [1] (as the first layer
    is just the integer 1, so is on all four diagonals for this
    layer) and all subsequent layers a list of length exactly four
    is yielded.
    
    Example:
    >>> for num in spiralDiagonalGenerator():
    >>>     if num[0] > 40: break
    >>>     print(num)
    [1]
    [3, 5, 7, 9]
    [13, 17, 21, 25]
    [31, 37, 43, 49]
    """
    yield [1]
    curr = 1
    n_steps = 0
    while True:
        n_steps += 2
        ans = []
        for _ in range(4):
            curr += n_steps
            ans.append(curr)
        yield ans
    return

def spiralPrimes(target_ratio: Real=10) -> int:
    """
    Solution to Project Euler #58

    Suppose the natural numbers (starting with 1) are arranged in
    a square spiral, with one layer added at a time. This function
    finds how many layers of this spiral need to be added before
    the proportion of diagonal elements that are primes is less
    than 1 / target_ratio (with both diagonals counted)
    
    Args:
        Optional named:
        target_ratio (int/float): The target for the inverse
                proportion of primes on the diagonals of the
                spiral.
            Default: 10
    
    Returns:
    Integer (int) giving the smallest number of layers that need
    to be added before the proportion of diagonal elements that
    are prime is less than 1 / target_ratio.
    """
    #since = time.time()

    ps = SimplePrimeSieve()
    def primeCheck(num: int) -> int:
        return ps.millerRabinPrimalityTestWithKnownBounds(num)[0]

    numer = 0
    denom = 1
    iter_obj = spiralDiagonalGenerator()
    next(iter_obj)
    for i, diag_lst in enumerate(iter_obj):
        denom += len(diag_lst)
        # Note the final element of the list is a square so
        # cannot be prime
        numer += sum(primeCheck(x) for x in diag_lst[:-1])
        #print(denom, numer, denom / numer)
        if numer * target_ratio < denom:
            break
    res = (i << 1) + 3
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 59
# Review- try to make faster (consider reducing the search space
#         using frequency analysis)
def decryptGenerator(
    key: Tuple[int],
    encrypt: List[int],
) -> Generator[str, None, None]:
    """
    Generator which, for a given key and encrypted text encrypt,
    performs rolling XOR decryption on encrypt and yields the
    decrypted letters one at a time.
    
    Args:
        Required positional:
        key (tuple of ints): The key of the rolling XOR encryption.
        encrypt (list of ints): The encrypted text.
    
    Yields:
    String (str) of length 1 providing each character of the decrypted
    text in order.
    """
    i = 0
    m = len(key)
    n = len(encrypt)
    for i in range(0, n, m):
        for j in range(m):
            yield chr(encrypt[i + j] ^ key[j])
    i += len(key)
    for j in range(i, n):
        yield chr(encrypt[j] ^ key[j - i])
    return

def xorDecryption(
    doc: str="project_euler_problem_data_files/p059_cipher.txt",
    key_len: int=3,
    auto_select: bool=True,
    n_candidates: int=3,
    rel_package_src: bool=True,
) -> int:
    """
    Solution to Project Euler #59

    Using cribbing to decrypt the string contained in the text file doc,
    given that the text has been encrypted based on an XOR cypher using
    ASCII encoding of characters with a key of known length (key_len)
    consisting of lower case characters.
    
    Args:
        Optional named:
        doc (str): Relative or absolue path to the .txt file containing
                the encrypted text
            Default: "project_euler_problem_data_files/p059_cipher.txt"
        key_len (int): The length of the key used to encrypt the text
            Default: 3
        auto_select (bool): If True, automatically selects the most
                likely encryption key (i.e. the key that gives rise
                to the most matches with crib words). Otherwise, gives
                a selection of the most likely keys with their associated
                decoding to the user to decide (in this case requiring
                a user prompt to complete).
            Default: True
        n_candidates (int): If auto_select is given as False, the number
                of candidate keys with their associated decoding given
                to the user to select from (given in order of the
                number of crib words the decoded text matches).
            Default: 3
        Optional named:
        rel_package_src (bool): Whether a relative path given by doc
                is relative to the current directory (False) or the
                package directory (True).
            Default: False
    
    Returns:
    Integer (int) giving the sum of ASCII values of the characters in
    the decoded text.
    
    Outline of rationale:
    Iterates over all possible keys (i.e. all strings of lower case
    characters with given length). For each key, XOR decrypts the
    text. The Aho Corasick algorithm is then applied to the
    decrypted text using variants of common words (with spaces on
    either side, with space on the left and comma/full stop and
    a spece on the right, and the same variants with the first
    letter capitalised), which counts the total number of matches of
    the decrypted text to these variants of common words. If auto_select
    is True, then judges the key with the largest such count to be the
    correct key and uses this for the final decryption, otherwise
    displays the n_candidates keys along with their corresponding
    decryption to the user to select the decryption that gives rise
    to a sensible text.
    """
    #since = time.time()
    cribs = {"the", "a", "an", "and", "it", "I", "me", "you", "he",\
            "him", "she", "her", "we", "us", "they", "them", "is",\
            "are", "am", "was", "were", "have", "has", "had", "how",\
            "where", "what", "why", "who", "from", "to",}
    if auto_select: n_candidates = 1
    
    #doc = doc.strip()
    #if not os.path.isfile(doc):
    #    raise FileNotFoundError(f"There is no file at location {doc}")
    #with open(doc) as f:
    #    txt = f.read()
    txt = loadTextFromFile(doc, rel_package_src=rel_package_src)
    encrypt = [int(x) for x in txt.split(",")]
    #print(encrypt)
    w_lst = []
    for w in cribs:
        w_pair = [w]
        if w[0].islower():
            w2 = list(w)
            w2[0] = w2[0].upper()
            w_pair.append("".join(w2))
        for w2 in w_pair:
            w_lst.append(f" {w2} ")
            w_lst.append(f" {w2}. ")
            w_lst.append(f" {w2}, ")
    ac = AhoCorasick(w_lst)
    n_letters = ord("z") - ord("a") + 1
    best_scores = []
    for num in range(n_letters ** key_len):
        key = []
        for _ in range(key_len):
            num, r = divmod(num, n_letters)
            key.append(r + ord("a"))
        #print(list(decryptGenerator(key, encrypt)))
        occur_dict = ac.search(decryptGenerator(key, encrypt))
        score = sum(len(x) for x in occur_dict.values())
        #if score:
        #    print(key, score)
        if len(best_scores) < n_candidates:
            heapq.heappush(best_scores, (score, key))
        else: heapq.heappushpop(best_scores, (score, key))
    
    if auto_select:
        i = 0
    else:
        best_scores.sort(reverse=True)
        # User prompt
        print("The following decryptions have been identified as "
                "the best candidates (the higher the score the "
                "better the candidate). Please select one "
                "decrypting from the following:")
        for j, (score, key) in enumerate(best_scores):
            print(f"\nOption {j + 1} (score {score}):\n"
                f"{''.join(list(decryptGenerator(key, encrypt)))}")
        i = int(input(f"Please select one of the options 1 "
                f"to {len(best_scores)}: ").strip()) - 1
    key = best_scores[i][1]
    res = sum(ord(l) for l in decryptGenerator(key, encrypt))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 60
def minimumPrimePairSetsSum(n_pair: int=5, base: int=10) -> int:
    """
    Solution to Project Euler #60

    Consider the collection of sets containing n_pair distinct
    primes such that concatenating any of the primes in the set with
    any other primes in the set in the chosen base in either order
    results in another prime. This function gives the smallest possible
    sum of elements of all the sets in this collection.
    
    Args:
        Optional named:
        n_pair (int): The size of the sets of primes under
                consideration.
            Default: 5
        base (int): The base used for the concatenation operation
            Default: 10
    
    Returns:
    Integer (int) giving the smallest possible sum of elements of
    all the sets in the collection described above.
    """
    #since = time.time()
    ps = SimplePrimeSieve()

    def primeCheck(num: int) -> int:
        return ps.millerRabinPrimalityTestWithKnownBounds(num)[0]

    best_sums = [0, 2]
    best_sets = [set(), {2}]
    rm_heaps = [[], []]
    res = float("inf")
    groups = {}
    available_grp_i = []
    group_membership = {}
    curr2 = base
    p_lst = SortedList()
    
    for i2, p2 in enumerate(ps.endlessPrimeGenerator()):
        #print(f"\ni2 = {i2}, p2 = {p2}, "
        #    f"group count = {len(groups)}, p_count = {len(p_lst)}")
        #print(f"best_sums = {best_sums}")
        #print(f"best_sets = {best_sets}")
        if len(best_sums) == n_pair + 1:
            p = p2
            ub = best_sums[-1]
            finish = True
            for j in reversed(range(n_pair)):
                ub -= p
                p += 2
                while rm_heaps[j] and -rm_heaps[j][0][0] >= ub:
                    grp_i = heapq.heappop(rm_heaps[j])[1]
                    if grp_i not in groups.keys() or\
                            len(groups[grp_i]) != j:
                        continue
                    group = groups.pop(grp_i)
                    for p in group:
                        group_membership[p].remove(grp_i)
                        if not group_membership[p]:
                            group_membership.pop(p)
                            p_lst.remove(p)
                    heapq.heappush(available_grp_i, grp_i)
                while rm_heaps[j]:
                    grp_i = rm_heaps[j][0][1]
                    if grp_i not in groups.keys() or\
                            len(groups[grp_i]) != j:
                        heapq.heappop(rm_heaps[j])
                    else: break
                if rm_heaps[j]: finish = False
            if finish:
                res = best_sums[-1]
                break
        if p2 >= curr2:
            curr2 *= base
        curr1 = base
        edges = set()
        for p1 in p_lst:
            if p1 >= curr1:
                curr1 *= base
            if not primeCheck(p2 + p1 * curr2) or not primeCheck(p1 + p2 * curr1):
                continue
            edges.add(p1)
        p_lst.add(p2)
        if not edges:
            i = heapq.heappop(available_grp_i) if available_grp_i\
                    else len(groups)
            group_membership[p2] = {i}
            groups[i] = {p2}
            heapq.heappush(rm_heaps[1], (p2, i))
            continue
        seen_groups = set()
        group_membership[p2] = set()
        cnt = 0
        seen_connected = set()
        for p1 in edges:
            for i in list(group_membership.get(p1, [])):
                if i in seen_groups or len(groups[i]) == n_pair:
                    continue
                cnt += 1
                seen_groups.add(i)
                connected = groups[i].intersection(edges)
                connected_tup = tuple(sorted(connected))
                if connected_tup in seen_connected:
                    continue
                seen_connected.add(connected_tup)
                
                if len(connected) == len(groups[i]):
                    if len(groups[i]) == n_pair - 1:
                        heapq.heappush(available_grp_i, i)
                        group = groups.pop(i)
                        for p in group:
                            group_membership[p].remove(i)
                            if not group_membership[p]:
                                group_membership.pop(p)
                                p_lst.remove(p)
                    else:
                        group = groups[i]
                        group_membership[p2].add(i)
                    group.add(p2)
                    ans = sum(group)
                    j = len(group)
                    if j == len(best_sums):
                        best_sums.append(float("inf"))
                        best_sets.append(set())
                        rm_heaps.append([])
                    heapq.heappush(rm_heaps[j], (-ans, i))
                    if ans < best_sums[j]:
                        best_sets[j] = set(group)
                        best_sums[j] = ans
                    continue
                i2 = heapq.heappop(available_grp_i) if available_grp_i\
                        else len(groups)
                connected.add(p2)
                for p in connected:
                    group_membership[p].add(i2)
                groups[i2] = connected
                heapq.heappush(rm_heaps[len(connected)],\
                        (-sum(connected), i2))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 61
def cyclicalFigurateNumbersSequences(
    n_dig: int,
    k_min: int,
    k_max: int,
    base: int=10,
) -> List[Tuple[int]]:
    """
    A cyclical sequence of length m (where m is a non-negative integer) in
    a given base is a sequence of integers such that each integer has the
    same even number of digits n when expressed in the chosen base (with no
    leading zeros) and the last n / 2 digits of any of the numbers in the
    chosen base are the same and in the same order as the first n  / 2 digits
    of the next number in the sequence or, for the final number in the
    sequence, of the first number in the sequence.
    This function identifies every cyclical sequence (up to cycling of elements
    in the list) in the given base of length (k_max - k_min + 1) for which
    every number in the sequence has n_dig digits (where n_dig is an even
    strictly positive integer) such that each element of the sequence
    corresponds to a different integer k between k_min and k_max inclusive
    and each element of the sequence is a k-oganal number (where k is
    the integer to which it corresponds). Such a sequence is referred to as
    a cyclical figurate numbers sequence in the given base for n_dig digit
    numbers for k between k_min and k_max inclusive.
    A k-oganal number (for k an integer no less than 3) is a number for which
    that number of dots can be arranged into the shape of a regular polygon with
    k sides.
    For example, the 3-ogonal numbers are the triangle numbers: 1, 3, 6, 10, etc
    and the 4-ogonal numbers are the square numbers: 1, 4, 9, 16, etc.
    The formula for the rth k-gonal number (where r is a stictly positive integer)
    is:
        r * ((k - 2) * r - k + 4)
    From this it can be deduced that a strictly positive integer a is k-oganal
    if and only if:
        k ** 2 + 8 * (a - 1) * (k - 2)
    is the square of an integer and:
        (k - 4) + sqrt(k ** 2 + 8 * (a - 1) * (k - 2))
    is a multiple of (2 * k - 4), where sqrt refers to the positive square root.
    
    
    Args:
        Required positional:
        n_dig (int): Even strictly positive integer giving the number of
                digits in the chosen base of the numbers being considered.
        k_min (int): Integer no less than 3 giving the smallest k
                of the k-oganal numbers required to appear in the
                sequence.
        k_max (int): Integer no less than k_min giving the largest k
                of the k-oganal numbers required to appear in the
                sequence.
        
        Optional named:
        base(int): The base used to express the integers.
            Default: 10
    
    Returns:
    Lists of (k_max - k_min + 1)-tuples giving all the possible sequences
    (up to cycling of elements in the list) which satisfy the conditions
    described above.   
    """
    if n_dig <= 0 or n_dig & 1:
        raise ValueError("n_dig must be a strictly positive even "
                f"integer. The value given was {n_dig}.")
    elif k_min < 3:
        raise ValueError("k_min must be an integer no less than 3. "
                f"The value given was {k_min}.")
    elif k_max < k_min:
        raise ValueError("k_min cannot exceed k_max. The value of "
                f"k_min was given as {k_min} while value given was "
                f"for k_max was {k_max}.")
    
    n_dig_hlf = n_dig >> 1
    md = base ** n_dig_hlf
    cycle_mn = base ** (n_dig_hlf - 1)
    
    def kPolygonal(n: int, k: int=3) -> int:
        return (n * ((k - 2) * n - k + 4)) >> 1
    
    def isKPolygonal(num: int, k: int=3) -> bool:
        num2 = k ** 2 + 8 * (k - 2) * (num - 1)
        num3 = isqrt(num2)
        if num3 ** 2 != num2: return False
        num4 = (k - 4) + num3
        return not (num4 % (2 * k - 4))
    
    def kPolygonalInRange(rng_mn: int, rng_mx: int, k: int=3)\
            -> Generator[int, None, None]:
        n = ((k - 4 + isqrt(k ** 2 + 8 * (k - 2) * (rng_mn - 1)))\
                // (2 * k - 4)) + 1
        if kPolygonal(n - 1, k=k) == rng_mn:
            yield rng_mn
        while True:
            num = kPolygonal(n, k=k)
            if num >= rng_mx: break
            yield num
            n += 1
        return
    
    def nextRange(num: int) -> Optional[Tuple[int]]:
        num2 = num % md
        if num2 < cycle_mn:
            return None
        num2 *= md
        return (num2, num2 + md)
    
    k_set = set(range(k_min, k_max))
    curr = []
    res = []
    def backtrack(num: int) -> None:
        rng = nextRange(num)
        if rng is None:
            return
        curr.append(num)
        if len(k_set) == 1:
            num2 = ((curr[-1] % md) * md) + (curr[0] // md)
            k = next(iter(k_set))
            if isKPolygonal(num2, k=k):
                curr.append(num2)
                res.append(tuple(curr))
                curr.pop()
            curr.pop()
            return
        for k in list(k_set):
            k_set.remove(k)
            for num2 in kPolygonalInRange(*rng, k=k):
                backtrack(num2)
            k_set.add(k)
        curr.pop()
        return
    
    for num in kPolygonalInRange(base ** (n_dig - 1), base ** n_dig,\
            k=k_max):
        backtrack(num)
    return res

def cyclicalFigurateNumbersSum(
    n_dig: int=4,
    k_min: int=3,
    k_max: int=8,
    base: int=10,
) -> int:
    """
    Solution to Project Euler #61

    Finds the smallest sum of terms out of all cyclical figurate
    numbers sequences in the given base for n_dig digit numbers for k
    between k_min and k_max inclusive (see documentation of
    cyclicalFigurateNumbersSequences() for more detail).
    
    Args:
        Optional named:
        n_dig (int): Even strictly positive integer giving the number of
                digits in the chosen base of the numbers being considered.
            Default: 4
        k_min (int): Integer no less than 3 giving the smallest k
                of the k-oganal numbers required to appear in the
                sequence.
            Default: 3
        k_max (int): Integer no less than k_min giving the largest k
                of the k-oganal numbers required to appear in the
                sequence.
            Default: 8
        base(int): The base used to express the integers.
            Default: 10
    
    Returns:
    Integer (int) giving the smallest sum of terms of all sequences
    fulfilling the described conditions. If no such sequences exist,
    returns -1.
    """
    #since = time.time()
    seqs = cyclicalFigurateNumbersSequences(n_dig=n_dig, k_min=k_min,\
            k_max=k_max, base=base)
    
    res = min(sum(x) for x in seqs) if seqs else -1
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 62
def iNthRoot(a: Union[int, float], n: int) -> int:
    """
    For a non-negative real number a and a strictly positive integer
    n, finds the largest non-negative integer b such that b ** n
    is no larger than num (or equivalently, the floor of the nth
    root of a).
    
    Args:
        Required positional:
        a (int/float): Non-negative integer giving the upper bound
                for the result when taken to the power of n.
        n (int): Strictly positive integer giving the exponent
    
    Returns:
    Integer (int) giving the largest integer b such that b ** n
    is no larger than a.
    """
    if n <= 0: raise ValueError("n must be strictly positive")
    if a < 0: raise ValueError("a must be non-negative")
    elif n == 1 or a < 2: return int(a)
    
    lft, rgt = 1, int(a)
    while lft < rgt:
        mid = lft - ((lft - rgt) >> 1)
        if mid ** n > a: rgt = mid - 1
        else: lft = mid
    return lft
"""
def numPermutationsGenerator(num_lst: int, base: int=10, skip_identity: bool=False)\
        -> Generator[int, None, None]:
    seen = set()
    if not skip_identity:
        yield num
    iter_obj = itertools.permutations(num_lst)
    seen.add(next(iter_obj)) # skip identity permutation
    for num_lst2 in iter_obj:
        if num_lst2 in seen or not num_lst2[-1]:
            continue
        seen.add(num_lst2)
        ans = 0
        for d in reversed(num_lst2):
            ans = ans * base + d
        yield ans
    return

def countNthRootPermutations(a: int, a_lst: Tuple[int], n: int, cubes: Set[int], base: int=10) -> Tuple[int]:
    #a0 = 110592
    res = 0
    a0 = a
    for a2 in numPermutationsGenerator(a_lst, base=base, skip_identity=True):
        if a2 in cubes:
            res += 1
            a0 = min(a0, a2)
        #b = iNthRoot(a2, n)
        #if a == a0: print(a2, b ** n == a2)
        #res += (b ** n == a2)
    return res, a0

def SmallestWithMNthPowerPermutations(m: int=5, n: int=3, base: int=10) -> int:
    b = 1
    cubes = set()
    while True:
        a = b ** n
        a_lst = []
        a2 = a
        while a2:
            a2, r = divmod(a2, base)
            a_lst.append(r)
        d0 = a_lst[-1]
        #print(a_lst)
        for i in range(len(a_lst) - 1):
            if a_lst[i] > d0:
                break
        else:
            cnt, a0 = countNthRootPermutations(a, a_lst, n, cubes, base=base)
            cnt += 1
            print(b, a, a0, cnt)
            if cnt == m:
                return a0
        cubes.add(a)
        b += 1
    return -1
"""
def smallestWithMNthPowerPermutations(
    m: int=5,
    n: int=3,
    base: int=10,
) -> int:
    """
    Solution to Project Euler #62

    Finds the smallest non-negative integer which is itself the nth power
    of an integer and exactly m of the numbers formed by permutations of
    its digits in the given base (excluding those with leading zeros but
    including itself) are each the nth power of an integer.
    
    Args:
        Optional named:
        m (int): The number of permutations of the digits of the solution
                that should fulfill the requirements.
            Default: 5
        n (int): The power used
            Default: 3
        base (int): The base used
            Default: 10
    
    Returns:
    Integer (int) giving the smallest non-negative integer fulfilling
    the stated requirements.
    """
    #since = time.time()
    if n <= 1: return 0
    b = 1
    n_dig = 0
    opts = set()
    res = -1
    while True:
        a = b ** n
        #print(b, a)
        a_lst = []
        a2 = a
        while a2:
            a2, r = divmod(a2, base)
            a_lst.append(r)
        if len(a_lst) > n_dig:
            if opts:
                res = min(opts)
                break
            n_dig = len(a_lst)
            seen = {}
            opts = set()
        k = tuple(sorted(a_lst))
        seen.setdefault(k, [a, 0])
        if seen[k][1] == m - 1:
            opts.add(seen[k][0])
        elif seen[k][1] == m:
            opts.remove(seen[k][0])
        seen[k][1] += 1
        b += 1
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 63
def powerfulDigits(base: int=10) -> int:
    """
    Solution to Project Euler #63

    The number of n-digit positive integers in the chosen base
    that are also the nth power of an integer for some positive
    integer n.
    
    Args:
        Optional named:
        base (int): The base used
            Default: 10
    
    Returns:
    Integer (int) giving the number of integers satisfying the
    stated requirements.
    
    Outline of rationale:
    In any base, base ** n is the smallest integer with n + 1
    digits. Therefore, there are no integers equal to a ** n
    with n digits in any base such that a >= base.
    Furthermore, if a ** n has n digits, then every
    integer b such that a <= b < base (in the given base)
    has n digits.
    Finally, if for positive integer a, a ** n has fewer
    than n digits in the given base, then given that this
    implies that a < base, a ** (n + 1) can have at most
    one more digit than a ** n, so a ** (n + 1) has fewer
    than n + 1 digits. It is then easy to show by induction
    that a ** m has fewer than m digits for all integers
    m >= n.
    """
    #since = time.time()
    k = 1
    res = 0
    n = 1
    mn = 1
    while True:
        while k ** n < mn:
            k += 1
        if k == base: break
        res += base - k
        #print(n, k)
        n += 1
        mn *= base
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 64
def sqrtCF(num: int) -> Tuple[Union[Tuple[int], int]]:
    """
    Finds the continued fraction representation of the
    square root of num
    
    Args:
        Required positional:
        num (int): The number whose square root to be
                represented as a continued fraction
    
    Returns:
    2-tuple whose 0th index contains a tuple of ints which is
    the sequence of terms in the continued fraction
    representation up to the point where the sequence repeats
    and whose 1st index contains the index of the sequence
    the repetition goes back to (where 0-indexing is used).
    For any positive integer that is not an exact square,
    the sequence is guaranteed to repeat.
    If num is an exact square (e.g. 1, 4, 9, ...) then the
    1st index contains -1.
    """
    seen = {}
    res = []
    curr = (0, 1)
    rt = isqrt(num)
    if rt ** 2 == num: return ((rt,), -1)
    while True:
        if curr in seen.keys():
            return (tuple(res), seen[curr])
        seen[curr] = len(res)
        a = (sqrt + curr[0]) // curr[1]
        res.append(a)
        b = curr[0] - a * curr[1]
        curr = (-b, (num - b ** 2) // curr[1])
        prev = curr
    return ()

def sqrtCFCycleLengthParity(num: int) -> bool:
    """
    Calculates the parity of the period of the continued
    fraction sequence of non-negative integer num.
    
    Args:
        Required positional:
        num (int): The integer whose square root continued
                fraction under consideration
    
    Returns:
    Boolean (bool) giving the parity of the period of the continued
    fraction sequence of the square root of positive integer num,
    with True representing odd period and False representing even
    period.
    If the continued fraction does not have a cycle (which occurs if
    and only if num is an exact square, i.e. 0, 1, 4, 9, ...) then
    False is returned.
    """
    cf = sqrtCF(num)
    if cf[1] == -1: return False
    return bool((len(cf[0]) - cf[1]) & 1)

def sqrtCFCycleLengthOddTotal(mx: int=10 ** 4) -> int:
    """
    Solution to Project Euler #64

    Calculates the number of positive integers not exceeding the
    integer mx whose continued fraction sequence cycles with an odd
    period.
    
    Args:
        Optional named:
        mx (int): The largest integer considered
            Default: 10 ** 4
    
    Returns:
    Integer (int) giving the number of positive integers not
    exceeding mx whose continued fraction sequence cycles with an
    odd period.
    """
    #since = time.time()
    res = sum(sqrtCFCycleLengthParity(i) for i in range(1, mx + 1))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 65
def nthConvergent(n: int, cf_func: Callable[[int], int]) -> Tuple[int]:
    """
    Finds the nth convergent of a given continued fraction
    representation of a non-negative number, with terms as
    given in cf_func()
    
    Args:
        Required positional:
        n (int): Strictly positive integer giving the
                convergent of the continued fraction is to
                be calculated
        cf_func (callable): A function accepting a single int
                as an argument. For given i, this should
                return the ith value in the 0-indexed continued
                fraction sequence. If the sequence has
                terminated before the ith index, it should
                return -1. Note that cf_func(0) must be a
                non-negative integer.
    
    Returns:
    2-tuple of ints where index 0 is the numerator and
    index 1 is the denominator
    """
    if n < 1: raise ValueError("n should be strictly positive")
    for i in reversed(range(n)):
        a = cf_func(i)
        if a != -1: break
    else: raise ValueError("The function cf_func returned -1 "
            "for 0, which is not allowed.")
    res = (a, 1)
    for i in reversed(range(i)):
        res = (res[1] + cf_func(i) * res[0], res[0])
    return res

def convergentGenerator(
    cf_func: Callable[[int], int],
) -> Generator[Tuple[int], None, None]:
    """
    Generates the convergents in order of a given continued fraction
    representation of a non-negative number with terms as given in
    the function cf_func() giving the terms in the continued fraction
    sequence under consideration.
    Note that if the continued fraction is infinite, this generator
    will not itself terminate, so any loop over this generator would
    in such a case need to contain a break or return statement.
    
    Args:
        cf_func (callable): A function accepting a single int
                as an argument. For given i, this should
                return the ith value in the 0-indexed continued
                fraction sequence. If the sequence has
                terminated before the ith index, it should
                return -1. Note that cf_func(0) must be a
                non-negative integer.
    
    Yields:
    Each convergent of the continued fraction with terms given by
    cf_func in turn.
    
    Outline of rationale:
    For continued fraction sequence, it is a known result (see
    https://pi.math.cornell.edu/~gautam/ContinuedFractions.pdf) that
    if p_i and q_i represent the numerator and denominator in lowest
    terms (i.e. gcd(p_i, q_i) = 1) of the ith convergent of a given
    continued fraction [a_0, a_1, a_2, ...] for any non-negative
    integer i:
        p_0 = a_0, q_0 = 1
        p_1 = a_1 * a_0 + 1, q_1 = a_1
        and for n >= 2:
        p_n = a_n * p_(n - 1) + p_(n - 2)
        q_n = a_n * q_(n - 1) + q_(n - 2)
    """
    a_0 = cf_func(0)
    curr = [(a_0, 1)]
    yield curr[0]
    a_1 = cf_func(1)
    if a_1 == -1: return
    curr.append((a_1 * a_0 + 1, a_1))
    yield curr[-1]
    n = 2
    while True:
        a_n = cf_func(n)
        if a_n == -1: break
        curr = [curr[1], (a_n * curr[-1][0] + curr[-2][0],\
                a_n * curr[-1][1] + curr[-2][1])]
        yield curr[-1]
        n += 1
    return

def eCFSequenceValue(i: int) -> int:
    """
    Gives the ith index (0-indexed) value of the continued fraction
    sequence of e (Euler's number).
    
    Args:
        Required positional:
        i (int): The index of the continued fraction sequence to be
                returned
        
    Returns:
    The ith index value of the continued fraction sequence of e.
    """
    if i % 3 == 2:
        return ((i // 3) + 1) << 1
    return 1 + (i == 0)

def nthEConvergent(n: int) -> Tuple[int]:
    """
    Finds the nth convergent of e
    
    Args:
        Required positional:
        n (int): Strictly positive integer giving the
                convergent of e to be calculated
    
    Returns:
    2-tuple of ints where index 0 is the numerator and
    index 1 is the denominator
    """
    return nthConvergent(n, cf_func=eCFSequenceValue)


def convergentENumeratorDigitSum(n: int=100, base: int=10) -> int:
    """
    Solution to Project Euler #65

    Finds the sum of digits of the numerator of the nth
    covergent of e in the chosen base
    
    Args:
        Optional named:
        n (int): Strictly positive integer giving the
                convergent of e considered
            Default: 100
        base (int): The base in which the digit sum is to
                be calculated
            Default: 10
    
    Returns:
    Integer (int) giving the sum of digits of the numerator
    of the nth convergent of e in the chosen base
    """
    #since = time.time()
    convergent = nthEConvergent(n)
    res = digitSum(convergent[0], base=base)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 66
def sqrtCFSequenceValue(i: int, num: int) -> int:
    """
    Gives the ith index (0-indexed) value of the continued fraction
    sequence of the square root of non-negative integer num.
    
    Args:
        Required positional:
        i (int): The index of the continued fraction sequence to be
                returned
        num (int): The non-negative integer whose square root the
                continued fraction sequence represents.
        
    Returns:
    The ith index value of the continued fraction sequence of the
    square root of num.
    If the sequence has terminated before the ith index (which is the
    case only for exact squares for i > 0), -1 is returned.
    
    Examples:
    >>> [sqrtCFSequenceValue(i, 2) for i in range(5)]
    [1, 2, 2, 2, 2]
    >>> [sqrtCFSequenceValue(i, 7) for i in range(10)]
    [2, 1, 1, 1, 4, 1, 1, 1, 4, 1]
    >>> [sqrtCFSequenceValue(i, 4) for i in range(5)]
    [2, -1, -1, -1, -1]
    """
    num_cf = sqrtCF(num)
    if i < len(num_cf[0]): return num_cf[0][i]
    if num_cf[1] == -1:
        return -1
    j = num_cf[1]
    return num_cf[0][j + (i - j) % (len(num_cf[0]) - j)]

def pellFundamentalSolution(D: int) -> Tuple[int]:
    """
    Finds the solution to Pell's equation and Pell's
    negative equation (if they exist):
        x ** 2 - D * y ** 2 = 1
        and
        x ** 2 - D * y ** 2 = -1
    respectively for given strictly positive integer D and
    strictly positive integers x and y such that there does
    not exist another such solution with smaller x (the
    so-called fundamental solution).
    Uses the standard method of solving Pell's equation using
    continued fractions.
    
    Args:
        Required positional:
        D (int): Strictly positive integer that is the value
                of D in the above equation
    
    Returns:
    2-tuple, giving the fundamental solution to Pell's equation
    (index 0) and Pell's negative equation (index 1). Each
    solution is either None (if no solution for strictly
    positive x and y exists for this D) or in the form of a
    2-tuple where indices 0 and 1 are the values of x and y
    for the fundamental solution.
    If D is a square, there is no solution with both strictly
    positive integers x and y, for either Pell's equation or
    Pell's negative equation so (None, None) is returned.
    Otherwise, there is always a solution to Pell's equation,
    so index 0 gives the value None if and only if D is a
    square.
    """
    #(since (1, 0) is the only possible solution to Pell's
    #equation with square D and integer x and y, but not
    #one satisfying the requirement that x and y are strictly
    #positive, and there is no solution not even a trivial one
    #to Pell's negative equation with square D and integer x and y).
    D_cf = sqrtCF(D)
    if D_cf[1] == -1:
        return (None, None)#(1, 0)
    def cf_func(i: int) -> int:
        if i < len(D_cf[0]): return D_cf[0][i]
        j = D_cf[1]
        return D_cf[0][j + (i - j) % (len(D_cf[0]) - j)]
        
    res = nthConvergent(len(D_cf[0]) - 1, cf_func)
    if (len(D_cf[0]) - D_cf[1]) & 1:
        # For continued fractions with odd cycle lengths
        # the solution found is the fundamental solution for:
        #  x ** 2 + D * y ** 2 = -1
        # The following converts this into the fundamental
        # solution for:
        #  x ** 2 + D * y ** 2 = 1
        x, y = res
        res2 = ((x ** 2 + D * y ** 2), (2 * x * y))
        return (res2, res)
    return (res, None)
    """
    # Solution checking every convergent in order
    cf_func = lambda i: sqrtCFSequenceValue(i, D)
    nth_convergent_func = lambda n: nthConvergent(n, cf_func)
    
    i = 1
    while True:
        x, y = nth_convergent_func(i)
        #print(x, y, x ** 2 - D * y ** 2)
        if x ** 2 - D * y ** 2 == 1:
            return (x, y)
        i += 1
    return -1
    """

def pellSolutionGenerator(
    D: int,
    negative: bool=False
) -> Generator[Tuple[int], None, None]:
    """
    Generator that yields the positive integer solutions to Pell's
    equation or Pell's negative equation:
        x ** 2 - D * y ** 2 = 1 (Pell's equation)
        or x ** 2 - D * y ** 2 = - 1 (Pell's negative equation)
    for given strictly positive integer D, in order of increasing
    size of x.
    Note that these equations either has no integer solutions or
    an infinite number of integer solutions. Pell's equation has
    and infinite number of integer solutions for any non-square
    positive integer value of D, while for square values of D
    it has no integer solutions, while Pell's negative equation
    some non-square positive integer values of D give an infinite
    number of integer solutions, while the other non-square positive
    integer values of D and all square values of D give no
    integer solutions.
    Given that for many values of D the generator does not by
    itself terminate, any loop over this generator should contain a
    break or return statement.
    
    Args:
        Required positional:
        D (int): The strictly positive integer number D used in
                Pell's equation or Pell's negative equation
        
        Optional named:
        negative (bool): If True, iterates over the solutions to
                Pell's equation for the given value of D, otherwise
                iterates over the solutions to Pell's negative
                equation for the given value of D.
            Default: False
    
    Yields:
    2-tuple of ints with the 0th index containing the value of x and
    1st index containing the value of y for the current solution to
    Pell's equation or Pell's negative equation (based on the input
    argument negative given).
    These solutions are yielded in increasing size of x (which by the
    form of Pell's equation and Pell's negative equation and the
    requirement that x and y are strictly positive implies the solutions
    are also yielded in increasing size of y), and it if the generator
    terminates, there are no solutions other than those yielded, and
    for any two consecutive solutions yielded (x1, y1) and (x2, y2), for
    any integer x where x1 < x < x2 there does not exist a positive
    integer y such that (x, y) is also a solution.
    """
    f_sol_pair = pellFundamentalSolution(D=D)
    f_sol = f_sol_pair[negative]
    if f_sol is None:
        # No solution
        return
    curr = f_sol
    if not negative:
        while True:
            yield curr
            curr = (curr[0] * f_sol[0] + D * curr[1] * f_sol[1],\
                    curr[1] * f_sol[0] + curr[0] * f_sol[1])
        return
    while True:
        yield curr
        curr = (curr[0] * f_sol[0] ** 2 + D * curr[0] * f_sol[1] ** 2\
                + 2 * D * curr[1] * f_sol[1] * f_sol[0],\
                curr[1] * f_sol[0] ** 2 + D * curr[1] * f_sol[1] ** 2\
                + 2 * curr[0] * f_sol[1] * f_sol[0])
    return

def pellLargestFundamentalSolution(D_max: int=1000) -> int:
    """
    Solution to Project Euler #66

    Finds the value of strictly positive non-square integer
    value of D no greater than D_max, for which the x-value
    of the fundamental solution of Pell's equation:
        x ** 2 - D * y ** 2 = 1
    for strictly positive integers x and y is largest.
    The fundamental solution of Pell's equation is the solution
    (x, y) where x and y are strictly positive integers and there
    does not exist strictly positive integers (x2, y2) such that
    x2 < x and (x2, y2) is a solution to Pell's equation (see
    documentation of pellFundamentalSolution() for more detail).
    
    Args:
        Optional named:
        D_max (int): Strictly positive integer representing the
                upper bound (inclusive) of the values of D
            Default: 1000
    
    Returns:
    Integer (int) with the value of D for which the x-value
    of the fundamental solution of Pell's equation for strictly
    positive integers x and y is largest of all strictly positive
    non-square D no greater than D_max.
    """
    #since = time.time()
    m = 2
    sq = m ** 2
    mx_x = -float("inf")
    res = -1
    for D in range(2, D_max + 1):
        
        if D == sq:
            sq += 2 * m + 1
            m += 1
            continue
        x = pellFundamentalSolution(D)[0][0]
        if x > mx_x:
            mx_x = x
            res = D
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 67
def loadTriangleFromFile(
    doc: str,
    rel_package_src: bool=False,
) -> List[List[int]]:
    """
    Loads triangle of integers from .txt file located at doc.
    The file should contain the rows of the triangle in order,
    separated by line breaks ('\\n') and the integers in each
    row separated by single spaces. For rows labelled in order
    starting from 1, each row must contain exactly the same
    number of elements as its row number.
    
    Args:
        Required positional:
        doc (str): The relative or absolution location of the .txt
                file containing the triangle of integers.

        Optional named:
        rel_package_src (bool): Whether a relative path given by doc
                is relative to the current directory (False) or
                the package src directory (True).
            Default: False
    
    Returns:
    List of lists of integers (int), with each list in the outer list
    representing a row of the triangle, as a list of integers with
    length equal to the row number (starting at 1).
    """
    txt = loadTextFromFile(doc, rel_package_src=rel_package_src)
    res = txt.split("\n")
    while not res[-1]: res.pop()
    return [[int(x) for x in row.split(" ")] for row in res]

def triangleMaxSumFromFile(
    triangle_doc: str="project_euler_problem_data_files/p067_triangle.txt",
    rel_package_src: bool=True,
) -> int:
    """
    Solution to Project Euler #67

    For a given triangle of integers contained in the .txt file
    triangle_doc, (arranged as a grid where the first row has length
    1 and every other row has length exactly one more than the
    previous row) finds the maximum total that can be achieved by
    travelling along a path from the top to the bottom, starting on
    the single element of the first row and each step moving to the
    next row, to either the element of that row with the same index
    as the index on the current row or the element with index one
    greater, and taking the total of the numbers of all elements of
    the triangle this path encounters.
    
    Args:
        Optional named:
        triangle_doc (str): The relative or absolute path to the .txt
                file containing the triangle of integers. In this file
                the triangle should be represented as a series of
                rows each separated by a single line break ('\n'),
                with each integer in each row separated by a single
                space (' ').
            Default: "project_euler_problem_data_files/p067_triangle.txt"
        rel_package_src (bool): Whether a relative path given by
                triangle_doc is relative to the current directory (False)
                or the package src directory (True).
            Default: False
        
    Returns:
    The maximum sum (int) that can be achieved by any traversal
    of the triangle contained in the .txt file triangle_doc following
    the restrictions given.
    """
    #since = time.time()
    """
    if isinstance(triangle, str):
        triangle = loadTriangle(triangle)
        preserve_triangle = False
    # Converting the triangle into lists of integers
    if isinstance(triangle[0], str):
        if preserve_triangle:
            triangle = list(triangle)
        for i, row in enumerate(triangle):
            triangle[i] = [int(x) for x in row.strip().split(" ")]
    """
    triangle = loadTriangleFromFile(
        triangle_doc,
        rel_package_src=rel_package_src,
    )
    return triangleMaxSum(
        triangle=triangle,
        preserve_triangle=False,
    )

# Problem 68
"""
Can easily find without a computer. This can be done by
noting that if there exists a solution where 6-10 are all
in the external nodes then this then the largest 16-digit
solution must have 6-10 all in the external nodes (as such
an arrangement would have its first digit as 6, and an
arrangement with any number 1-5 in the external nodes will
have first digit less than 6, which given that the numbers
must have 16 digits means that the latter will necessarily
be smaller). For such an arrangement, each triplet must sum
to ((6+7+8+9+10)+2*(1+2+3+4+5))/5 = 14. Similarly, of these
arrangements, if there exists a 16-digit solution where the
node adjacent to 6 is 5 then the largest 16-digit solution
must have the node adjacent to 6 being 5. Placing these
essentially forces the rest of the numbers, and gives the
following solution (numbers separated by commas and semicolons
for clarity):
6,5,3;10,3,1;9,1,4;8,4,2;7,2,5
Concatenated (as is required of the solution):
6531031914842725
"""
def magic5gonRing() -> int:
    """
    Solution to Project Euler #68

    TODO
    """
    return 6531031914842725

# Problem 69
def totientFunction(num: int, ps: Optional[PrimeSPFsieve]=None) -> int:
    """
    Calculates the value of the Euler totient function for strictly
    positive integer num (i.e. phi(num)).
    For strictly positive integer n, the Euler totient function applied
    to n (i.e. phi(n)) gives the number of strictly positive integers m
    not exceeding n such that the greatest common divisor (gcd) of m and
    n is exactly 1 (i.e. m and n are relatively prime).
    
    Args:
        Required positional:
        num (int): The strictly positive integer for which the Euler
                totient function is to be calculated
        
        Optional named:
        ps (PrimeSPFsieve or None): If specified, a PrimeSPFsieve
                object to facilitate the calculation of the prime
                factorisation of num (which is used to calculate
                the Euler totient function value for num).
                If this is given, then this object may be updated
                by this function, in particular it may extend the
                prime sieve it contains.
            Default: None
    
    Returns:
    Integer (int) giving the value of the Euler totient function
    when applied to num.
    """
    if ps is None: ps = PrimeSPFsieve()
    factorisation = ps.primeFactorisation(num)
    res = 1
    for p, n in factorisation.items():
        res *= (p - 1) * p ** (n - 1)
    return res

def totientMaximum(n_max: int=10 ** 6) -> int:
    """
    Solution to Project Euler #69

    Finds the strictly positive integer n not exceeding n_max
    such that n / phi(n) is maximised, where phi(n) is the Euler
    totient function.
    For strictly positive integer n, the Euler totient function applied
    to n gives the number of strictly positive integers m not exceeding
    n such that the greatest common divisor (gcd) of m and n is exactly
    1.
    
    Args:
        Optional named:
        n_max (int): The upper bound (inclusive) of the range
                considered.
            Default: 10 ** 6
        
    Returns:
    Integer (int) giving the strictly positive integer n not
    exceeding n_max that gives the largest value of n / phi(n)
    of all strictly positive integers not exceeding n_max.
    
    Outline of rationale:
    We use phi(n) = n * (product over primes p that divide n)
                    of (1 - 1 / p)
    which can be rearranged to find:
    n / phi(n) = (product over primes p that divide n) p / (p - 1)
    Note that for positive if p1, p2 > 0 and p1 > p2 then
    p1 / (p1 - 1) < p2 / (p2 - 1)
    From this it can be derived that the value of 0 < n <= n_max
    such that n / phi(n) is maximised is:
     (product over all primes p < m) p
    where m is the largest integer such that this product is no
    larger than n_max.
    """
    ps = PrimeSPFsieve()
    curr = 1
    for p in ps.endlessPrimeGenerator():
        nxt = curr * p
        if nxt > n_max: break
        curr = nxt
    return curr
    """
    # Brute force calculation of the value of n / phi(n) for
    every strictly positive integer not exceeding n_max
    ps = PrimeSPFsieve(n_max)
    best = -float("inf")
    for num in range(1, n_max + 1):
        totient = totientFunction(num, ps)
        #print(f"{num}: {totient}")
        ans = num / totient
        if ans > best:
            best = ans
            res = num
    return res
    """

# Problem 70
# Review- ?can be faster (most of the time is currently taken
#         constructing the prime sieve (~22s out of ~32s total).
#         Is there an alternative method that can avoid using
#         the prime sieve with overall faster runtime?
def numDigitDict(num: int, base: int=10) -> Dict[int, int]:
    """
    Creates a frequency dictionary of the digits of non-negative
    integer num when expressed in the chosen base without leading
    zeros.
    
    Args:
        Required positional:
        num (int): The non-negative integer whose digit frequency
                in the chosen base is to be calculated.
        
        Optional named:
        base (int): The base in which the integer num is to be
                expressed.
            Default: 10
    
    Returns:
    Dictionary whose keys and values are the digits (as ints)
    that appear in the representation of num in the chosen base
    (without leading zeros) and the corresponding count of that
    digit in the representation repectively.
    """
    if not num: return {0: 1}
    res = {}
    while num:
        num, r = divmod(num, base)
        res[r] = res.get(r, 0) + 1
    return res

def numDigitsMatch(
    num: int,
    comp: Union[int, Dict[int, int]],
    base: int=10
) -> bool:
    """
    Checks that the digits of non-negative integer num when expressed
    in the chosen base with no leading zeros contains the exact same
    digits as comp when expressed in the chosen base (if comp is an
    integer) or exactly the same frequency of each digit as given by
    comp (if comp is a dictionary).
    
    Args:
        Required positional:
        num (int): The non-negative integer whose digits in the chosen
                base are to be assessed.
        comp (int or dict): Either a non-negative integer whose digits
                in the chosen base are to be compared to num, or a
                frequency dictionary whose keys and values are the
                digits (as ints) that appear in the representation of
                some non-negative integer in the chosen base and the
                corresponding count of that digit in the representation.
        
        Optional named:
        base (int): The base in which num and comp (or the integer the
                dictionary comp is calculated from) are to be expressed.
            Default: 10
        
    Returns:
    Boolean (bool) indicating whether the digit count of num in the
    chosen base is a match for comp.
    """
    if isinstance(comp, int): comp = numDigitDict(comp, base=base)
    res = {}
    while num:
        num, r = divmod(num, base)
        res[r] = res.get(r, 0) + 1
        if res[r] > comp.get(r, 0): return False
    return res == comp

def totientPermutation(
    num_mn: int=2,
    num_mx: int=10 ** 7 - 1,
    base: int=10,
) -> int:
    """    
    Solution to Project Euler #70

    Finds the integer n between num_mn and num_mx inclusive such that
    the representation of n in the chosen base contains the same digits
    in the same frequency as the representation of the value of phi(n)
    (where phi is the Euler totient function) in the chosen base (with
    no leading zeros in either representation) such that there is no
    other integer n' that satisfies the same conditions for which
    n' / phi(n') < n / phi(n) or n' / phi(n') = n / phi(n) and n' > n.
    
    Args:
        Optional named:
        num_mn (int): The lower bound (inclusive) of the integers
                considered.
            Default: 2
        num_mx (int): The upper bound (inclusive) of the integers
                considered.
            Default: 10 ** 7 - 1
        base (int): The base used for the digit counts.
    
    Returns:
    Integer (int) representing the value of the unique integer
    satisfying the described conditions.
    """
    #since = time.time()
    ps = PrimeSPFsieve(num_mx)
    #print(f"Prime sieve took {time.time() - since:.4f} seconds")
    sieve = ps.sieve
    best_ratio = -1
    for num in range(num_mn, num_mx + 1):
        # Primes cannot have their totient function be a permutation
        # of the digits of itself if base > 2
        if sieve[num][0] == num and base > 2: continue
        ratio = 1 # phi(num) / num, which we want to maximise
                  # (note that the ratio is flipped over relative
                  # to the problem statement and the documentation,
                  # so seeking to maximise rather than minimise)
        num2 = num
        tf = 1
        while num2 > 1 and ratio > best_ratio:
            p, m, num2 = sieve[num2]
            ratio *= (1 - 1 / p)
            if ratio <= best_ratio: break
            tf *= (p - 1) * p ** (m - 1)
        else:
            if not numDigitsMatch(tf, num): continue
            #print(num, ratio)
            best_ratio = ratio
            res = num
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 71
# Review- consider adding Farey sequence algorithms into
#         the data_structures_and_algorithms package

def solveLinearHomogeneousDiophantineEquation(
    a: int,
    b: int,
) -> Tuple[int, int]:
    """
    Finds the set of solutions to the equation:
        a * x + b * y = 0
    for integer x and y
    
    Args:
        Required positional:
        a (int): The coefficient of x in the equation
        b (int): The coefficient of y in the equation
    
    Returns:
    2-tuple of ints, (x0, y0) such that the set of solutions are:
        x = x0 * k, y = y0 * k, integer k
    """
    g = gcd(a, b)
    return (b // g, -a // g)

def extendedEuclideanAlgorithm(a: int, b: int) -> Tuple[int, int, int]:
    """
    Finds integers x and y such that:
        a * x + b * y = gcd(a, b)
    
    Args:
        Required positional:
        a (int): The coefficient of x in the equation
        b (int): The coefficient of y in the equation
    
    Returns:
    3-tuple of ints where indices 0 and 1 contain the values
    of x and y respectively of the solution and index 2 contains
    gcd(a, b)
    """
    rev = False
    if a < b:
        a, b = b, a
        rev = True
    stk = [(0, b, a)]
    while stk[-1][1]:
        stk.append((*divmod(stk[-1][2], stk[-1][1]), stk[-1][1]))
    g = stk.pop()[2]
    x, y = 1, 0
    while stk:
        x, y = y - stk.pop()[0] * x, x
    return (y, x, g) if rev else (x, y, g)

def solveLinearNonHomogeneousDiophantineEquation(
    a: int,
    b: int,
    c: int,
) -> Tuple[int]:
    """
    Finds the general solution for integers x and y in the equation:
        a * x + b * y = c
    for integers a, b, c where a and b are non-zero, if such a solution
    exists
    
    Args:
        Required positional:
        a (int): The coefficient of x in the equation. Must be non-zero.
        b (int): The coefficient of y in the equation. Must be non-zero.
        c (int): The value on the right hand side of the equation
    
    Returns:
    4-tuple of ints if a solution exists, otherwise an empty tuple.
    The general solution is of the form:
        x = dx * k + x0, y = dy * k + y0
    for integers dx, x0, dy, y0 and k can take any integer value.
    Any solution of the equation for integer x, y is of this form.
    In terms of these integers, the returned 4-tuple is:
        (dx, x0, dy, y)
    """
    if not a or not b: raise ValueError("a and b must be non-zero")
    g = gcd(a, b)
    if c % g: return ()
    a, b, c = (a // g, b // g, c // g)
    x, y, _ = extendedEuclideanAlgorithm(a, b)
    return (b, x * c, -a, y * c)

def adjacentFarey(
    frac: Tuple[int],
    max_denom: int,
    nxt: bool=True,
    frac_farey: bool=False,
) -> Tuple[int, int]:
    """
    For any fraction frac between 0 and 1 inclusive with denominator in
    lowest terms no greater than max_denom, finds the largest fraction
    strictly less than frac and no less than 0 (if nxt given as False)
    or the smallest fraction strictly greater than frac and no greater
    than 1 (if nxt given as True) with denominator no greater than
    max_denom (or equivalently, finds the fraction that precedes/succeeds
    frac in the Farey sequence of order max_denom).
    
    Args:
        Required positional:
        frac (2-tuple of ints): The fraction strictly between 0 and 1,
                represented by a 2-tuple of ints whose 0th index contains
                the numerator and whose 1st index contains the denominator
        max_denom (int): The upper bound (inclusive) of the denominator
                for both frac and the solution in lowest terms
        
        Optional named:
        nxt (bool): If True then finds the next element in the Farey
                sequence of order max_denom (i.e. the smallest fraction
                strictly greater than frac with denominator no greater
                than max_denom), otherwise the previous element in the Farey
                sequence of order max_denom (i.e. the largest fraction
                strictly less than frac with denominator no greater
                than max_denom).
            Default: True
        frac_farey (bool): If True then the input frac is guaranteed
                to represent an element of the Farey sequence in lowest
                terms.
            Default: False
    
    Returns:
    2-tuple of ints representing the unique fraction in lowest terms that
    satisfies the given requirements, with index 0 and 1 containing the
    numerator and denominator respectively. If no such fraction exists
    (i.e frac = (1, 1) and nxt = True or frac = (0, 1) and nxt = False)
    then returns an empty tuple.
    
    Outline of rationale:
    This uses the fact that two consecutive elements of a Farey sequence
    p1 / q1 and p2 / q2 (with the former preceding the latter in the
    sequence) satisfy:
        p2 * q1 - p1 * q2 = 1
    and for the Farey sequence of order n, for an element fo that Farey
    sequence p2 / q2, if p2 / q2 < 1 then the next element, p3 / q3 is
    the integer solution to:
        p3 * q2 - p2 * q3 = 1
    where q3 <= n and there does not exist another solution with
    denominator larger than q3 and no greater than n, and if p2 / q2 > 0
    then the previous element p1 / p2 is the integer solution to:
        p2 * q1 - p1 * q2 = 1
    where q1 <= n and there does not exist another solution with
    denominator larger than q1.
    """
    if not frac_farey:
        if frac[0] < 0 and frac[1] < 0:
            frac = tuple(-x for x in frac)
        if frac[0] <= 0 or frac[0] >= frac[1]:
            raise ValueError("The input parameter frac must represent a "
                    "fraction that is strictly greater than zero and "
                    "strictly less than one.")
        g = gcd(*frac)
        frac = tuple(x // g for x in frac)
        if frac[1] > max_denom:
            raise ValueError("The input parameter frac must represent a "
                    "fraction which, when expressed in lowest terms, has a "
                    "denominator no greater thn max_denom")
    if (nxt and (frac[0] == frac[1])) or (not nxt and frac[0] == 0):
        return ()
    elif nxt and not frac[0]:
        return (1, max_denom)
    # General solution to Diophantine equation a * x - b * y = 1 where
    # frac = (b, a)
    args0 = (frac[0], -frac[1]) if nxt\
            else (-frac[0], frac[1])
    #print(args0)
    dx, x0, dy, y0 = solveLinearNonHomogeneousDiophantineEquation(*args0, 1)
    #print(x0, dx, y0, dy)
    neg = ((max_denom - x0) < 0) ^ (dx < 0)
    #print(neg)
    k = abs(max_denom - x0) // abs(dx)
    if neg: k = -k
    return (y0 + dy * k, x0 + dx * k)

def orderedFractions(
    frac: Tuple[int]=(3, 7),
    max_denom: int=10 ** 6,
) -> int:
    """
    Solution to Project Euler #71

    Finds the numerator of the largest fraction (in lowest terms) smaller
    than frac with denominator no greater than max_denom.
    
    Optional named:
        Required positional:
        frac (2-tuple of ints): A fraction no larger than 1 and strictly
                greater than 0, represented by a 2-tuple of ints whose
                0th index contains the numerator and whose 1st index
                contains the denominator.
            Default: (3, 7)
        max_denom (int): The upper bound (inclusive) of the denominator
                for both frac and the solution in lowest terms.
            Default: 10 ** 6
    
    Returns:
    Integer (int) giving the numerator of the largest fraction (in lowest
    terms) smaller than frac with denominator no greater than max_denom.
    """
    #since = time.time()
    res = adjacentFarey(frac, max_denom, nxt=False)[0]
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
    """
    # Brute force solution
    res = (0, 1)
    for denom in range(2, max_denom + 1):
        numer = (denom * frac[0]) // frac[1]
        numer -= (denom * frac[0] == numer * frac[1])
        if numer * res[1] > denom * res[0]:
            res = (numer, denom)
    return res[0]
    """

# Problem 72
def countingFractions(max_denom: int=10 ** 6) -> int:
    """
    Solution to Project Euler #72

    Finds the number of distinct fractions with denominator no greater
    than max_denom and value between 0 and 1 exclusive
    
    Args:
        Optional named:
        max_denom (int): The largest denominator allowed for fractions
                considered.
            Default: 10 ** 6
    
    Returns:
    Integer (int) giving the number of fractions satisfying the
    requirements given.
    """
    #since = time.time()
    res = 0
    ps = PrimeSPFsieve(max_denom)
    for num in range(2, max_denom + 1):
        res += totientFunction(num, ps)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 73- try to make faster
# Look into https://people.csail.mit.edu/mip/papers/farey/talk.pdf

def fareyNext(
    n: int,
    curr: Tuple[int]=(0, 1),
    lowest_terms: bool=False,
) -> Tuple[int]:
    """
    Finds the smallest element of the Farey sequence of order n which
    is strictly larger than the fraction curr.
    Note that this is distinct from adjacentFarey with nxt=True as this
    does not require that curr is itself in the Farey sequence of order
    n.
    
    Args:
        Required positional:
        n (int): The order of the Farey sequence in question
        
        Optional named:
        curr (2-tuple of ints): A fraction no less than 0 and strictly
                less than one, represented by a 2-tuple of ints whose
                0th index contains the numerator and whose 1st index
                contains the denominator, giving the fraction for
                which the smallest element of the Farey sequence with
                value strictly greater than this fraction is being
                sought.
            Default: (0, 1)
        lowest_terms (bool): Whether the fraction curr is guaranteed
                to be in lowest terms (i.e. gcd(curr[0], curr[1]) = 1
                and curr[1] > 0)
            Default: False
    
    Returns:
    2-tuple of ints representing smallest the fraction in the Farey
    sequence of order n whose value is strictly greater than the
    fraction curr whose 0th index contains the numerator and whose
    1st index contains the denominator. If no such element exists,
    returns an empty tuple.
    """
    if not lowest_terms:
        if curr[0] ^ curr[1]: return (0, 1)
        elif curr[0] < 0: curr = tuple(-x for x in curr)
        if curr[0] >= curr[1]: return ()
        g = gcd(*curr)
        curr = tuple(x // g for x in curr)
    elif curr[0] < 0: return (0, 1)
    elif curr[0] >= curr[1]: return ()
    if curr == (0, 1): return (1, n)
    elif curr[0] == 1 and (curr[1] << 1) > n:
        return (1, curr[1] - 1)
    elif curr[0] == curr[1] - 1 and ((curr[1] + 1) << 1) > n:
        if curr[1] >= n: return (1, 1)
        return (curr[0] + 1, curr[1] + 1)
    if curr[1] <= n:
        # curr is in the Farey sequence of order n
        return adjacentFarey(curr, n, nxt=True, frac_farey=True)
    res = (1, 1)
    for denom in range(2, n + 1):
        numer = ((curr[0] * denom) // curr[1]) + 1
        if numer * res[1] < denom * res[0]:
            res = (numer, denom)
    return res

def fareySequence(
    n: int,
    mn: Tuple[int]=(0, 1),
    mx: Tuple[int]=(1, 1)
) -> Generator[Tuple[int], None, None]:
    """
    Generator iterating over the elements of the Farey sequence of order
    n for elements with value between mn and mx (inclusive) in
    increasing order.
    
    Args:
        Required positional:
        n (int): The order of the Farey sequence being iterated over
        
        Optional named:
        mn (2-tuple of ints): A fraction between 0 and 1 inclusive,
                represented by a 2-tuple of ints whose 0th index
                contains the numerator and whose 1st index contains
                the denominator, giving the smallest value of
                fractions that may be yielded by the generator.
            Default: (0, 1)
        mx (2-tuple of ints): A fraction between 0 and 1 inclusive,
                represented by a 2-tuple of ints whose 0th index
                contains the numerator and whose 1st index contains
                the denominator, giving the largest value of
                fractions that may be yielded by the generator.
            Default: (1, 1)
    
    Yields:
    Each element of the Farey sequence of order n that is no less than
    the fraction mn and no greater than the fraction mx in order of
    increasing size and in lowest terms, represented by a 2-tuple of
    ints whose 0th index contains the numerator and whose 1st index
    contains the denominator.
    """
    # Using https://en.wikipedia.org/wiki/Farey_sequence
    if mx[0] > mx[1]: mx = (1, 0)
    if (mn[0] < 0) ^ (mn[1] < 0): mn = (0, 1)
    elif mn[0] < 0: mn = tuple(-x for x in mn)
    if mn[0] * mx[1] > mn[1] * mx[0] or mn[0] > mn[1]: return
    g = gcd(*mn)
    mn = tuple(x // g for x in mn)
    if mn[1] <= n: prev = mn
    else:
        prev = fareyNext(n, curr=mn, lowest_terms=True)
        if not prev or prev[0] * mx[1] > prev[1] * mx[0]: return
    yield prev
    curr = adjacentFarey(prev, n, nxt=True, frac_farey=True)
    while curr[0] * mx[1] <= curr[1] * mx[0]:
        yield curr
        mult = (n + prev[1]) // curr[1]
        prev, curr = curr, (mult * curr[0] - prev[0],\
                            mult * curr[1] - prev[1])
    return
"""
def fareySequenceDenom(n: int) -> Generator[int, None, None]:
    prev, curr = 1, n
    yield prev
    yield curr
    while curr > 1:
        prev, curr = curr, ((n + prev) // curr) * curr - prev
        yield curr
    return
"""
def countingFractionsRange(
    lower_frac: Tuple[int]=(1, 3),
    upper_frac: Tuple[int]=(1, 2),
    max_denom: int=12 * 10 ** 3,
) -> int:
    """
    Solution to Project Euler #73
    
    Counts the number of fractions with value between the fractions
    lower_frac and upper_frac exclusive with denominator no greater
    than max_denom.
    
    Args:
        Optional named:
        lower_frac (2-tuple of ints): A fraction between 0 and 1
                inclusive, represented by a 2-tuple of ints whose 0th index
                contains the numerator and whose 1st index contains
                the denominator, where the denominator in lowest terms
                is no greater than max_denom, giving the lower bound
                (exclusive) of the value of fractions counted.
            Default: (1, 3)
        upper_frac (2-tuple of ints): A fraction between 0 and 1
                inclusive, represented by a 2-tuple of ints whose 0th index
                contains the numerator and whose 1st index contains
                the denominator, where the denominator in lowest terms
                is no greater than max_denom, giving the upper bound
                (exclusive) of the value of fractions counted.
            Default: (1, 2)
        max_denom (int): The largest denominator allowed for fractions
                considered.
            Default: 12 * 10 ** 3
    
    Returns:
    Integer (int) giving the number of fractions satisfying the given
    requirements.
    """
    #since = time.time()
    g1 = gcd(*lower_frac)
    lower_frac = tuple(x // g1 for x in lower_frac)
    g2 = gcd(*upper_frac)
    upper_frac = tuple(x // g2 for x in upper_frac)
    res = 0
    for _ in fareySequence(max_denom, lower_frac, upper_frac): res += 1
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res - (lower_frac[1] <= max_denom and lower_frac[0] <= lower_frac[1])\
            - (upper_frac[1] <= max_denom and upper_frac[0] <= lower_frac[1])
    
    """
    if lower_frac[0] * upper_frac[1] >= lower_frac[1] * upper_frac[0]:
        return 0
    since = time.time()
    lower_count = 0
    upper_count = 0
    i1 = -1
    for i2, denom in enumerate(fareySequenceDenom(max_denom)):
        if denom == lower_frac[1]:
            lower_count += 1
            if lower_count == lower_frac[0]:
                i1 = i2
        if denom == upper_frac[1]:
            upper_count += 1
            if upper_count == upper_frac[0]:
                break
    else: raise ValueError
    if i1 == -1: raise ValueError
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return i2 - i1 - 1
    """
    """
    since = time.time()
    res = 0
    for denom in range(2, max_denom + 1):
        numer_start = ((denom * lower_frac[0]) // lower_frac[1]) + 1
        numer_end = -((-denom * upper_frac[0]) // upper_frac[1])
        # create sieve
        excl_set = set()
        for numer in range(numer_start, numer_end):
            if numer in excl_set: continue
            g = gcd(numer, denom)
            if g > 1:
                for i in range(numer, numer_end, g):
                    excl_set.add(i)
            #if gcd(numer, denom) == 1:
            #    res += 1
            #    print((numer, denom))
        res += numer_end - numer_start - len(excl_set)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
    """

# Problem 74
# Review- try to make faster
def digitFactorial(num: int, base: int=10) -> int:
    """
    The sum of the digits of the representation of non-negative integer
    num in the chosen base.
    
    Args:
        Required positional:
        num (int): The non-negative integer being considered
        
        Optional named:
        base (int): The base in which num is to be represented
    
    Returns:
    Integer (int) giving the sum of the digits of the representation
    of num in the chosen base.
    """
    res = 0
    while num:
        num, r = divmod(num, base)
        res += math.factorial(r)
    return res

def digitFactorialChainLength(
    num: int,
    seen: Dict[int, int],
    base: int=10,
) -> int:
    """
    Consider the sequence starting at num where a term in the
    sequence after the first is calculated by finding the sum of
    the factorials of the digits of the previous term when expressed
    in the chosen base with no leading zeros. This function finds
    the number of terms in this sequence before a term that has
    already appeared in the sequence is encountered (i.e. the length
    of the longest non-repeating chain of terms of this sequence).
    This is referred to as the digit factorial chain length of num
    in the chosen base.
    
    Args:
        Required positional:
        num (int): The non-negative integer for which the length of
                the digit factorial chain in the chosen base is being
                sought.
        seen (dict): Dictionary representing the non-negative integers
                whose digit factorial chain lengths in the chosen
                base have already been evaluated, with the keys and
                values being those integers and the corresponding digit
                factorial chain lengths in the chosen base.
                Note that seen will be updated in-place with any new
                results evaluated during the function call.
        
        Optional named:
        base (int): The base in which the digit factorial chain length
                is to be calculated.
            Default: 10
    
    Returns:
    Integer (int) giving the digit factorial chain length of num in
    the chosen base.
    """
    path = []
    path_dict = {}
    num0 = num
    while num not in path_dict.keys() and num not in seen.keys():
        path_dict[num] = len(path)
        path.append(num)
        num = digitFactorial(num, base=base)
    if num in path_dict.keys():
        i = path_dict[num]
        cycle_len = len(path) - i
        for j in range(i):
            seen[path[j]] = len(path) - j
        for j in range(i, len(path)):
            seen[path[j]] = cycle_len
    else:
        add = seen[num] + 1
        for j in range(len(path)):
            seen[path[~j]] = j + add
    return seen[num0]

def countDigitFactorialChains(
    chain_len: int=60,
    n_max: int=10 ** 6 - 1,
    base: int=10,
) -> int:
    """
    Solution to Project Euler #74

    Finds the number of integers between 1 and n_max (inclusive) whose
    digit factorial chain length in the chosen base (see documentation
    of digitFactorialChainLength() for more details) is exactly
    chain_len.
    
    Args:
        Optional named:
        chain_len (int): The length of digit factorial chain length
                in the chosen base of integers counted.
            Default: 60
        n_max (int): The largest integer considered.
            Default: 10 ** 6 - 1
        base (int): The base used for calculation of the digit factorial
            Default: 10
    
    Returns:
    Integer (int) giving the number of integers between 1 and n_max
    inclusive whose digit factorial chain length in the chosen base
    is exactly chain_len.
    """
    #since = time.time()
    seen = {}
    res = 0
    for i in range(1, n_max + 1):
        res += digitFactorialChainLength(i, seen, base=base) == chain_len
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
        
# Problem 75
def countUniquePythagoreanTripleSums(n_max: int=15 * 10 ** 5) -> int:
    """
    Solution to Project Euler #75

    Calculates the number of strictly positive integers not exceeding
    n_max such that there is exactly one Pythagorean triple whose sum
    is equal to this value.
    
    Args:
        Optional named:
        n_max (int): The strictly positive integer value for which
                Pythagorean triples whose sum exceeds this value are
                not counted.
            Default: 15 * 10 ** 5
    
    Returns:
    Integer (int) giving the number of strictly positive integers not
    exceeding n_max such that there is exactly one Pythagorean triple
    whose sum is equal to this value.
    
    Outline of rationale:
    Uses the fact that every distinct Pythagorean triple corresponds to
    a unique set of integers k, m, n such that m > n, exactly one of
    m and n is even and gcd(m, n) = 1. The Pythangorean triple this
    set of integers corresponds to is:
        2 * k * m * n, k * (m ** 2 - n ** 2), k * (m ** 2 + n ** 2)
    Summing these side lengths for the given valid k, m, n gives:
        2 * k * m * (m + n)
    We therefore calculate this value for all the k, m, n values
    collectively satisfying the above requirements for which this
    value does not exceed n_max, counting the number of occurrences
    of each result and counting those that occur exactly once.
    """
    #since = time.time()
    f_dict = {}
    for m in range(1, (n_max >> 1) + 1):
        for n in range(1 + (m & 1),\
                min((n_max // (2 * m)) - m + 1, m), 2):
            if gcd(m, n) != 1: continue
            basic = (m * (m + n)) << 1
            for k in range(1, (n_max // basic) + 1):
                ans = basic * k
                f_dict[ans] = f_dict.get(ans, 0) + 1
    res = sum(x == 1 for x in f_dict.values())
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 76
class Partition(object):
    """
    Class whose instances calculate the partition function for
    non-negative integers (if specified, modulo a given integer).
    The partition function is the function which has the domain
    of the non-negative integers, and for non-negative integer
    n returns the number of ways that number can be expressed
    as the sum of strictly positive integers (with different
    orderings of the same multiset of integers considered to
    be the same). 
    
    Initialisation args:
        Optional named:
        n (int): The integer up to which the partition function
                should be calculated during initialisation
            Default: 0
        md (int or None): The value given to the md attribute,
                which if given as an integer represents the
                modulus to which the values of the partition
                function returned should be given
            Default: None
    
    Attributes:
        md (int or None): If an integer, the modulus to which the
                values of the partition function returned should
                be given. If None, the values of the partition
                function are returned with the exact value, not
                taken to any modulus. 
        arr (list of ints): The values of the partition function
                (modulo self.md if attribute self.md is not None) that
                have been calculated, where the ith index contains
                the value of the partition function (modulo self.md
                if applicable) for the integer i.
    
    Function call:
        Returns the value of the partition function (modulo md if
        applicable) for a given non-negative integer
        
        Args:
            Required positional:
            n (int): The non-negative integer for which the value
                    of the partition function (modulo self.md if
                    applicable) is to be returned
        
        Returns:
        The value of the partition function (modulo self.md if
        applicable) for n.
    
    Methods:
            (see documentation for methods themselves for more detail)
        extend(): Calculates and stores all values for the partition
                function (modulo self.md if applicable) for all
                non-negative integers for which it has not already been
                calculated up to a given integer.
    """
    def __init__(self, n: int=0, md: int=None) -> int:
        self.arr = [1 if md is None else 1 % md]
        self.md = md
        self.extend(n)
    
    def extend(self, n: int) -> None:
        """
        Calcuates the value of the partition function (modulo self.md
        if self.md is an integer) for all integer  values from
        len(self.arr) to n (inclusive), extends self.arr to length n
        and storing the calculated values at the corresponding index
        of self.arr.
        
        Args:
            Required positional:
            n (int): The largest input value this method requires the
                    partition function to be calculated.
        
        Returns:
        None
        
        Outline of rationale:
        This method makes use of the recurrence relation (where for
        non-negative integer n, P(n) is the partition function for
        n):
            P(0) = 1
            P(n) = (sum over k=1 to n) (-1) ** (k + 1) *
                        [P(n - k * (3 * k - 1) / 2)
                        + P(n - k * (3 * k + 1) / 2)]
                   for n > 0
        """
        if self.md is not None:
            return self._extend_mod(n, md=self.md)
        for i in range(len(self.arr), n + 1):
            self.arr.append(0)
            for k in range(1, i + 2):
                j = i - ((k * (3 * k - 1)) >> 1)
                if j < 0: break
                if j < k:
                    self.arr[i] -= self.arr[j] * round((-1) ** k)
                    break
                self.arr[i] -= (self.arr[j] + self.arr[j - k]) * round((-1) ** k)
        return
    
    def _extend_mod(self, n: int, md: Optional[int]=None) -> None:
        if md is None: md = self.md
        for i in range(len(self.arr), n + 1):
            self.arr.append(0)
            for k in range(1, i + 2):
                j = i - ((k * (3 * k - 1)) >> 1)
                if j < 0: break
                if j < k:
                    self.arr[i] = (self.arr[i] -\
                                self.arr[j] * round((-1) ** k)) % md
                    break
                self.arr[i] = (self.arr[i] - (self.arr[j] +\
                                self.arr[j - k]) * round((-1) ** k)) % md
        return
    
    def __call__(self, n: int) -> int:
        self.extend(n)
        return self.arr[n]

def partitionFunctionNontrivial(n: int=100) -> int:
    """
    Solution to Project Euler #76

    Finds the number of ways that the non-negative integer n can be
    expressed as the sum of at least two strictly positive integers
    (with different orderings of the same multiset of integers
    considered to be the same)- i.e. the number of non-trivial
    partitions of n.
    
    Args:
        Optional positional:
        n (int): The non-negative integer for which the number of
                non-trivial partitions is to be calculated.
            Default: 100
    
    Returns:
    The number of non-trivial partitions of n
    
    Outline of rationale:
    This is simply the value of the partition function (see Partition
    class) for n minus one, as the requirement that the partition
    consists of at least two strictly positive integers excludes
    the partition n = n and no others.
    """
    # Using partition function recurrence relation (with Partition
    # class instance)
    # Time complexity O(n ** (3 / 2))
    #since = time.time()
    pf = Partition()
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return pf(n) - 1
    
    """
    # Using dynamic programming
    # Time complexity O(n ** 2)
    row = [0] * (n + 1)
    row[0] = 1
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            row[j] += row[j - i]
    
    return row[-1] - 1
    """

# Problem 77
def primeSummations(
    target_count: int=5000,
    batch_size: int=100,
) -> int:
    """
    Solution to Project Euler #77
    Finds the smallest strictly positive integer that can be expressed
    as the sum of primes in at least target_count different ways.
    Two sums of primes are considered the same if and only if they
    include the same primes and each prime appears the same number of
    times in both summations.
    
    Args:
        Optional named:
        target_count (int): Strictly positive integer giving the
                smallest number of different ways the integer returned
                should be expressible as the sum of primes.
            Default: 5000
        batch_size (int): The integers considered are examined in
                batches of increasing size. This strictly positive
                integer determines the size of these batches.
            Default: 100
    
    Returns:
    Integer (int) giving the smallest strictly positive integer that
    can be expressed as the sum of primes in at least target_count
    different ways.
    """
    # Try to optimise space usage by limiting the size of each row in the
    # dp array to max(batch_size, p) where p is the prime corresponding to
    # that row
    #since = time.time()
    if not target_count: return 0
    if target_count == 1: return 2
    ps = SimplePrimeSieve()
    res = float("inf")
    start = 1
    dp = []
    cnt = 0
    while not isinstance(res, int):
        #print(f"start = {start}")
        end = start + batch_size
        p_gen = ps.endlessPrimeGenerator()
        p = next(p_gen)
        if not dp:
            if p >= end:
                start = end
                continue
            dp.append([0] * (end))
            dp[0][0] = 1
        else: dp[0].extend([0] * batch_size)
        for i in range(max((-((-start) // p)) * p, p), end, p):
            dp[0][i] = 1
        for p_i, p in enumerate(p_gen, start=1):
            if p >= end: break
            if len(dp) == p_i:
                dp.append(list(dp[p_i - 1]))
                for i in range(p, end):
                    dp[p_i][i] += dp[p_i][i - p]
                    if dp[p_i][i] >= target_count:
                        res = min(res, i)
                continue
            dp[p_i].extend([0] * batch_size)
            for i in range(start, end):
                dp[p_i][i] = dp[p_i - 1][i] + dp[p_i][i - p]
                if dp[p_i][i] >= target_count:
                    res = min(res, i)
        if isinstance(res, int): break
        start = end
        #print(dp)
    #print(dp[-1])
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
        
    """
            if p >= end: break
            carry_len = max(increment, p)
            if len(carries) == p_i:
                carries.append([0] * carry_len)
                if not p_i: carries[0][p - start] = 1
            carry = carries[p_i]
            carry2 = [0] * carry_len
            if p_i:
                for i in range(len(carries[p_i - 1])):
                    carry2[~i] = carries[p_i - 1][~i]
            #elif p - start >= 0:
            #    carry2[p - start] += 1
            #if len(carry) != p:
            #    print(start, p, carry)
            for c_i2, j in enumerate(range(start, min(end, start + len(carry)))):
                #print(c_i, j)
                c_i = c_i2 + carry_len - p
                row[j] += carry[c_i]
                print(f"c_i = {c_i}, j = {j}, c_i2 = {c_i2}")
                carry2[c_i2] += carry[c_i]
                if row[j] >= target_count:
                    res = min(res, j)
            #print(row)
            #print(p, start + len(carry))
            #print(p, start)
            for j in range(max(p, start + len(carry)), end):
                row[j] += row[j - p]
                #if j - start >= 0:
                #    print(p, j, j - start, j - p)
                
                c_i2 = j - start
                print(f"j = {j}, c_i2 = {c_i2}")
                carry2[c_i2] += row[j - p]
                if row[j] >= target_count:
                    res = min(res, j)
            #print(row)
            carries2.append(carry2)
        start = end
        carries = carries2
        print(carries)
        print(row)
        if cnt >= 7: break
        cnt += 1
    print(row)
    return res
    """

# Problem 78
def coinPartitions(div: int=10 ** 6) -> int:
    """
    Solution to Project Euler #78

    The smallest integer such that the number of integer partitions
    of that integer is divisible by the strictly positive integer
    div.
    
    Args:
        Optional named:
        div (int): The strictly positive integer by which the
                solution should be divisible.
            Default: 10 ** 6
    
    Returns:
    Integer (int) representing the smallest integer such that the
    number of integer partitions of that integer is divisible by
    div.
    """
    #since = time.time()
    pf = Partition(md=div)
    i = 0
    while True:
        ans = pf(i)
        #print(i, ans)
        if not ans: break
        i += 1
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return i

# Problem 79
def loadStringsLineBreak(
    doc: str,
    rel_package_src: bool=False,
) -> List[str]:
    """
    Loads a list of strings from a .txt file at relative or absolute
    file location doc.
    In the .txt file at location doc, the different strings should each
    be separated by line breaks ('\\n').
    
    Args:
        Required positional:
        doc (str): The relative or absolute path to the .txt
                file containing the strings.
        
        Optional named:
        rel_package_src (bool): Whether a relative path given by doc
                is relative to the current directory (False) or
                the package src directory (True).
            Default: False
    
    Returns:
    List of strings (str), containing the strings in the .txt file
    at location doc in unchanged order.
    """
    txt = loadTextFromFile(doc, rel_package_src=rel_package_src)
    return [row for row in txt.split("\n") if row]

def shortestStringGivenSubsequences(subseqs: List[str]) -> str:
    """
    For a list of strings subseqs, finds the string such for which
    every string in subseqs is a subsequence, such that any other
    string also satisfying this property contains more characters
    or the same number of characters and is lexicographically larger.
    A subsequence of a string is a string that can be formed by
    deleting any number of characters from that string (including
    none) and leaving the remaining characters in their original order.
    
    Args:
        Required positional:
        subseqs (list of strs): A list containing the strings that
                must each be a subsequence of the returned string.
    
    Returns:
    String (str) for which every string in subseqs is a subsequence,
    such that any other string also satisfying this property contains
    more characters or the same number of characters and is
    lexicographically larger.
    """
    #print("Using shortestStringGivenSubsequences()")
    n = len(subseqs)
    #rng_lst = [[0, len(s)] for s in subseqs]
    subseq_lens = [len(s) for s in subseqs]
    nxt_dict = {}
    counts = {}
    for i, s in enumerate(subseqs):
        nxt_dict.setdefault(s[0], set())
        nxt_dict[s[0]].add((i, 1))
        for l in s:
            counts[l] = counts.get(l, 0) + 1
    curr = []
    best = [float("inf"), []]
    
    def backtrack() -> None:
        #if len(curr) == 1:
        #print(curr, nxt_dict, counts)
        if len(curr) + len(counts) > best[0]:
            return
        elif not counts:
            if counts:
                l = next(iter(counts.keys()))
                for _ in range(counts[l]):
                    curr.append(l)
            length = len(curr)
            if [length, curr] < best:
                best[0], best[1] = [length, list(curr)]
            if counts:
                for _ in range(counts[l]):
                    curr.pop()
            return
        l_lst = sorted(nxt_dict.keys(),\
                key=lambda x: -len(nxt_dict[x]))
        #print(l_lst)
        #print(counts)
        
        for l in l_lst:
            #print(f"counts pre = {counts}")
            inds = nxt_dict.pop(l)
            #print(l, inds)
            curr.append(l)
            add_lst = [(subseqs[i][j], (i, j + 1)) for i, j in inds\
                    if j < subseq_lens[i]]
            counts[l] -= len(inds)
            if not counts[l]: counts.pop(l)
            for l2, pair in add_lst:
                nxt_dict.setdefault(l2, set())
                nxt_dict[l2].add(pair)
            backtrack()
            for l2, pair in add_lst:
                nxt_dict[l2].remove(pair)
                if not nxt_dict[l2]:
                    nxt_dict.pop(l2)
            nxt_dict[l] = inds
            counts[l] = counts.get(l, 0) + len(inds)
            #print(f"counts post = {counts}")
            curr.pop()
        return
    backtrack()
    return "".join(best[1])

def shortestStringGivenSubsequencesTrie(subseqs: List[str]) -> str:
    """
    For a list of strings subseqs, finds the string such for which
    every string in subseqs is a subsequence, such that any other
    string also satisfying this property contains more characters
    or the same number of characters and is lexicographically larger.
    A subsequence of a string is a string that can be formed by
    deleting any number of characters from that string (including
    none) and leaving the remaining characters in their original order.
    
    This has the same effect as the function
    shortestStringGivenSubsequences() using a different methodology
    involving a trie. Seems to be generally slower than that function
    (approx 25-50% slower typically).
    
    Args:
        Required positional:
        subseqs (list of strs): A list containing the strings that
                must each be a subsequence of the returned string.
    
    Returns:
    String (str) for which every string in subseqs is a subsequence,
    such that any other string also satisfying this property contains
    more characters or the same number of characters and is
    lexicographically larger.
    """
    print("Using shortestStringGivenSubsequencesTrie()")
    trie = [{}]
    subtrie_sizes = [0]
    l_counts = {}
    for s in subseqs:
        # Checking if the substring has already been encountered.
        i = 0
        for l in s:
            if l not in trie[i].keys():
                break
            i = trie[i][l]
        else: continue
        i = 0
        subtrie_sizes[i] += 1
        for l in s:
            l_counts[l] = l_counts.get(l, 0) + 1
            if l not in trie[i].keys():
                trie[i][l] = len(trie)
                trie.append({})
                subtrie_sizes.append(0)
            i = trie[i][l]
            subtrie_sizes[i] += 1
    
    #print(subseqs)
    #print(trie)
    #print(l_counts)
    #print(subtrie_sizes)
    curr = []
    best = [float("inf"), []]
    #trie_inds = {0}
    nxt_letter_dict = {l: {i} for l, i in trie[0].items()}
    #print(nxt_letter_dict)
    last_letters = {}
    
    def backtrack() -> None:
        #print(curr, best[0], len(l_counts), nxt_letter_dict, last_letters)
        if len(curr) + len(l_counts) > best[0]:
            return
        if not nxt_letter_dict and not last_letters:
            length = len(curr)
            if [length, curr] < best:
                best[0], best[1] = [length, list(curr)]
                #print(best)
            return
        l_lst = sorted(set(nxt_letter_dict.keys())\
                .union(last_letters.keys()),\
                key=lambda x: -sum(subtrie_sizes[y]\
                for y in nxt_letter_dict.get(x, set())) -\
                last_letters.get(x, 0))
        #print(l_lst)
        curr.append("")
        for l in l_lst:
            inds = nxt_letter_dict.pop(l) if l in\
                    nxt_letter_dict.keys() else set()
            n_last = last_letters.pop(l) if l in last_letters.keys()\
                    else 0
            sub = sum(subtrie_sizes[i] for i in inds) + n_last
            l_counts[l] -= sub
            if not l_counts[l]: l_counts.pop(l)
            rm_lst = []
            last_letters_add = {}
            for i1 in inds:
                for l2, i2 in trie[i1].items():
                    if not trie[i2]:
                        last_letters[l2] = last_letters.get(l2, 0) + 1
                        last_letters_add[l2] =\
                                last_letters_add.get(l2, 0) + 1
                        continue
                    nxt_letter_dict.setdefault(l2, set())
                    nxt_letter_dict[l2].add(i2)
                    rm_lst.append((l2, i2))
            curr[-1] = l
            backtrack()
            for l2, i2 in rm_lst:
                nxt_letter_dict[l2].remove(i2)
                if not nxt_letter_dict[l2]:
                    nxt_letter_dict.pop(l2)
            for l2, f2 in last_letters_add.items():
                last_letters[l2] -= f2
                if not last_letters[l2]:
                    last_letters.pop(l2)
            #for i1 in inds:
            #    for l2, i2 in trie[i1].items():
            #        nxt_letter_dict[l2].remove(i2)
            #        if not nxt_letter_dict[l2]:
            #            nxt_letter_dict.pop(l2)
            l_counts[l] = l_counts.get(l, 0) + sub
            if n_last:
                last_letters[l] = n_last
            if inds:
                nxt_letter_dict[l] = inds
        curr.pop()
        return
    backtrack()
    return "".join(best[1])

def stringContainsSubsequencesCheck(
    s: str,
    subseqs: List[str]
) -> bool:
    """
    Checks whether a string s contains every one of a list of
    strings subseqs as subsequences.
    A subsequence of a string is a string that can be formed by
    deleting any number of characters from that string (including
    none) and leaving the remaining characters in their original order.
    
    Args:
        Required positional:
        s (str): The string being checked as to whether it contains
                all of the required subsequences.
        subseqs (list of strs): The strings, each of which should be
                a subsequence of s for the function to return True.
    
    Returns:
    Boolean (bool) giving True if every string in subseqs is a
    subsequence of s and False otherwise.
    """
    ind_lsts = {}
    for i, l in enumerate(s):
        ind_lsts.setdefault(l, [])
        ind_lsts[l].append(i)
    for ss in subseqs:
        idx = -1
        for l in ss[:-1]:
            if l not in ind_lsts.keys(): return False
            j = bisect.bisect_right(ind_lsts[l], idx)
            if j == len(ind_lsts[l]): return False
            idx = ind_lsts[l][j]
        l = ss[-1]
        if l not in ind_lsts.keys() or idx >= ind_lsts[l][-1]:
            print(ss)
            return False
    return True

def testRandomShortestStringGivenSubsequences(
    orig_len: int=15,
    n_subseqs: int=30,
    subseq_lens: int=3,
    use_trie_method: bool=False,
    verbosity: int=1,
) -> Tuple[Union[bool, str, Tuple[str]]]:
    """
    Generates a random string of digits from which a number of random
    subsequences are extracted to test one of the functions:
     - shortestStringGivenSubsequences() if use_trie_method False or
     - shortestStringGivenSubsequencesTrie() if use_trie_method True
    which for a given set of subsequences, finds the shortest string
    which contains all these subsequences.
    A subsequence of a string is a string that can be formed by
    deleting any number of characters from that string (including
    none) and leaving the remaining characters in their original order.
    This checks whether the string generated by the chosen function
    does indeed contain all of the required subsequences and also
    that it is either shorter than or the same length as and
    lexicographically no greater than the original random string.
    Note that while a passed test guarantees that the string found
    by the function contains all the required subsequences, it does
    not guarantee that the string found is the shortest such string.
    
    The test case given for Project Euler #79 is really straightforward
    (in particular, there are no character repeats) so this function
    is designed to provide more robust testing of the
    shortestStringGivenSubsequences() or
    shortestStringGivenSubsequencesTrie() functions with (in general)
    more challenging cases.
    
    Args:
        Optional named:
        orig_len (int): Strictly positive integer giving the length
                of the random string of digits from which the
                subsequences are to be extracted.
            Default: 15
        n_subseqs (int): Strictly positive integer giving the number
                of subsequences that are to be extracted from the
                random string and be used as the list of subsequences
                to test one of the functions
                shortestStringGivenSubsequences() or
                shortestStringGivenSubsequencesTrie()
            Default: 30
        subseq_lens (int): Strictly positive integer no greater than
                orig_len giving the lengths of the subsequences to
                be extracted from the random string and used to test
                one of the above functions.
            Default: 3
        use_trie_method (bool): If False then tests
                shortestStringGivenSubsequences(), otherwise tests
                shortestStringGivenSubsequencesTrie().
            Default: False
        verbosity (int): Non-negative integer representing how much
                information about the progress and results of the
                function to print to Terminal. The larger this number
                is, the more detail is printed. If 0 then nothing
                is printed to Terminal.
            Default: 1
    
    Returns:
    4-tuple whose index 0 contains a boolean (bool) indicating whether
    the test has been passed (with True specifying that the test was
    passed and False that it was failed), index 1 contains a string
    giving the random string from which the subsequences wre extracted,
    index 2 contains a n_subseqs-tuple of strings giving the random
    subsequences of the string in index 1 used to test the function,
    and index 3 contains the string the function being tested returned.
    """
    orig = "".join([str(random.randrange(0, 10)) for _ in\
            range(orig_len)])
    subseqs = ["".join([orig[i] for i in\
            uniformRandomDistinctIntegers(subseq_lens, 0,\
            orig_len - 1)]) for _ in range(n_subseqs)]
    func = shortestStringGivenSubsequencesTrie if use_trie_method else\
            shortestStringGivenSubsequences
    res = func(subseqs)
    b = True
    ps = ""
    if not stringContainsSubsequencesCheck(res, subseqs):
        ps = "Test failed- the derived string does not contain every "\
                "required subsequence."
        b = False
    elif len(res) > len(orig):
        ps = "Test failed- the derived string is has more characters "\
                "than the original string from which the "\
                "subsequences were drawn, so the derived string "\
                "cannot be the shortest possible string containing "\
                "all these subsequences."
        b = False
    elif len(res) == len(orig) and res > orig:
        ps = "Test failed- the derived string has the same number of "\
                "characters as and is lexicographically larger than "\
                "the original string from which the subsequences "\
                "were drawn, so the derived string cannot be the "\
                "correct solution."
        b = False
    else:
        ps = "Test passed- the derived string contains all of the "\
                "required subsequences and has no more characters "\
                "than the original string from which the "\
                "subsequences were drawn, so this may have the "\
                "correct solution for this set of subsequences "\
                "(though this test does not guarantee that "\
                "there does not exist a string containing all the "\
                "required subsequences that either has fewer "\
                "characters or the same number of characters and is "\
                "lexicographically smaller)."
    if verbosity > 0:
        print(f"\nOriginal string = {orig} (length {len(orig)})")
        print("Subsequences:")
        print(subseqs)
        print(f"Derived shortest possible string containing these "
                f"subsequences = {res} (length {len(res)})")
        print(ps)
    return (b, orig, tuple(subseqs), res)

def testMultipleRandomShortestStringGivenSubsequences(
    n_tests: int=20,
    orig_len: int=15,
    n_subseqs: int=30,
    subseq_lens: int=3,
    use_trie_method: bool=False,
    verbosity: int=1,
) -> bool:
    """
    Performs n_tests tests for one of the functions:
     - shortestStringGivenSubsequences() if use_trie_method False or
     - shortestStringGivenSubsequencesTrie() if use_trie_method True
    which for a given set of subsequences, finds the shortest string
    which contains all these subsequences.
    For each test (performed by testRandomShortestStringGivenSubsequences()
    a random string of digits is generated from which a number of
    random subsequences are extracted to test the chosen function.
    A subsequence of a string is a string that can be formed by
    deleting any number of characters from that string (including
    none) and leaving the remaining characters in their original order.
    This checks whether the string generated by the chosen function
    does indeed contain all of the required subsequences and also
    that it is either shorter than or the same length as and
    lexicographically no greater than the original random string.
    Note that while a passed test guarantees that the string found
    by the function contains all the required subsequences, it does
    not guarantee that the string found is the shortest such string.
    
    Args:
        Optional named:
        n_tests (int): The number of tests of the chosen function
                performed.
            Default: 20
        orig_len (int): Strictly positive integer giving the length
                of the random string of digits from which the
                subsequences are to be extracted.
            Default: 15
        n_subseqs (int): Strictly positive integer giving the number
                of subsequences that are to be extracted from the
                random string and be used as the list of subsequences
                to test one of the functions
                shortestStringGivenSubsequences() or
                shortestStringGivenSubsequencesTrie()
            Default: 30
        subseq_lens (int): Strictly positive integer no greater than
                orig_len giving the lengths of the subsequences to
                be extracted from the random string and used to test
                one of the above functions.
            Default: 3
        use_trie_method (bool): If False then tests
                shortestStringGivenSubsequences(), otherwise tests
                shortestStringGivenSubsequencesTrie().
            Default: False
        verbosity (int): Non-negative integer representing how much
                information about the progress and results of the
                function to print to Terminal. The larger this number
                is, the more detail is printed. If 0 then nothing
                is printed to Terminal.
            Default: 1
    
    Returns:
    Boolean (bool) giving True if every test performed on the chosen
    function was passed, and False if at least one of these tests was
    failed.
    """
    since = time.time()
    n_pass = 0
    for i in range(1, n_tests + 1):
        ans = testRandomShortestStringGivenSubsequences(\
                orig_len=orig_len, n_subseqs=n_subseqs,\
                subseq_lens=subseq_lens,\
                use_trie_method=use_trie_method,\
                verbosity=max(0, verbosity - 1))
        n_pass += ans[0]
        outcome_str = "passed" if ans[0] else "failed"
        if verbosity > 0:
            print(f"Test {i} of {n_tests} {outcome_str}.")
    if verbosity > 0:
        print(f"{n_pass} out of {n_tests} random codes passed check.")
        print(f"Time taken = {time.time() - since:.4f} seconds")
    return n_pass == n_tests

def shortestStringGivenSubsequencesCompareMethodTimes(
    orig_len: int=15,
    n_subseqs: int=30,
    subseq_lens: int=3,
    verbosity: int=1,
) -> Dict[str, float]:
    """
    TODO
    """
    orig = "".join([str(random.randrange(0, 10))\
            for _ in range(orig_len)])
    samples = ["".join([orig[i] for i in\
            uniformRandomDistinctIntegers(subseq_lens, 0,\
            orig_len - 1)]) for _ in range(n_subseqs)]
    funcs = ((shortestStringGivenSubsequencesTrie, "trie method"),
            (shortestStringGivenSubsequences, "standard method"))
    t_lst = [0] * len(funcs)
    res = [""] * len(funcs)
    if verbosity > 0:
        print(f"\nOriginal code = {orig} (length {len(orig)})")
        print("Samples:")
        print(samples)
    for i, func_pair in enumerate(funcs):
        func, s = func_pair
        since = time.time()
        res[i] = func(samples)
        t_lst[i] = time.time() - since
        print(f"Derived shortest possible codes for samples by {s} = "
                f"{res[i]} (length {len(res[i])})")
        if not stringContainsSubsequencesCheck(res[i], samples) or\
                len(res[i]) > len(orig):
            print("This is not an optimal solution")
            return []
        print("This may be an optimal solution.")
        print(f"Time taken by {s} = {t_lst[i]:.4f}") 
    return {f[1]: t for f, t in zip(funcs, t_lst)}

def passcodeDerivation(
    doc: str="project_euler_problem_data_files/0079_keylog.txt",
    rel_package_src: bool=True,
) -> str:
    """
    Solution to Project Euler #79

    For a set of strings contained in the .txt file at doc, finds the
    string such that every word in the set of strings is a subsequence
    of that string, and any other such string either contains more
    characters or the same number of characters and is
    lexicographically larger.
    A subsequence of a string is a string that can be formed by
    deleting any number of characters from that string (including
    none) and leaving the remaining characters in their original order.
    Each string in the file should be separated by a line break
    ('\\n').
    
    Args:
        Optional named:
        doc (str): String giving the relative or absolute location
                of the .txt file containing the set of strings that
                are each required to be a subsequence of the returned
                string.
            Default: "project_euler_problem_data_files/0079_keylog.txt"
        rel_package_src (bool): Whether a relative path given by doc
                is relative to the current directory (False) or
                the package src directory (True).
            Default: True
    
    Returns:
    The string (str) satisfying the given requirements.
    """
    #since = time.time()
    code_lst = loadStringsLineBreak(doc=doc, rel_package_src=rel_package_src)
    res = shortestStringGivenSubsequences(code_lst)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 80
def sqrtDecimal(x: Union[float, int], accuracy: float=10 ** -6) -> int:
    """
    Finds the positive square root real non-negative number x
    such that for calculated number y:
        abs(y ** 2 - x) < accuracy
    
    Args:
        Required positional:
        x (float/int): Non-negative real number whose square root
                is to be calculated.
        accuracy (float): The required accuracy of the square root.
    
    Returns:
    The positive square root of x up to the requred accuracy
    """
    # Newton's method
    if isinstance(x, int):
        y1 = isqrt(x)
        if y1 * y1 == x: return float(y1)
    else: y1 = x
    y2 = 0
    #print(y1, y2)
    while abs(y1 - y2) >= accuracy:
        y2 = y1
        y1 = (y2 + x / y2) / 2
        #print(y1, y2, abs(y1 - y2))
    return y1

def squareRootDigitalExpansionSum(
    num_max: int=100,
    n_dig: int=100,
    base: int=10,
) -> int:
    """
    Solution to Project Euler #80
    
    Finds the sum over the first n_dig digits of the representation
    in the chosen base (with no leading zeros) of the positive square
    root of all non-square positive integers no greater than num_max.
    
    Args:
        Optional named:
        num_max (int): The largest integer whose square root may be
                included in the sum
            Default: 100
        n_dig (int): The number of digits of each square root
                included in the sum
            Default: 100
        base (int): The base in which the square roots are to
                be represeted
            Default: 10
        
    Returns:
    Integer (int) giving the value of the sum over the first n_dig
    digits of the representation in the chosen base (with no leading
    zeros) of the positive square root of all non-square positive
    integers no greater than num_max.
    """
    #since = time.time()
    sq_set = set()
    num = 1
    while True:
        sq = num ** 2
        if sq > num_max: break
        sq_set.add(sq)
        num += 1
    res = 0
    for num in range(1, num_max + 1):
        if num in sq_set: continue
        ans = isqrt(num)
        cnt = 0
        while ans:
            ans //= base
            cnt += 1
        num *= base ** (2 * (n_dig - cnt))
        ans = isqrt(num)
        while ans:
            ans, r = divmod(ans, base)
            res += r
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 81
def loadGrid(
    doc: str,
    rel_package_src: bool=False,
) -> List[List[int]]:
    """
    Loads a rectangular grid from a .txt file at relative or
    absolute file location doc.
    The matrix should be stored in that file in the form of integer
    strings with rows separated by line breaks ('\\n') and columns
    separated by commas (',').
    
    Args:
        Required positional:
        doc (str): The relative or absolute path to the .txt
                file containing the rectangular grid
        
        Optional named:
        rel_package_src (bool): Whether a relative path given by doc
                is relative to the current directory (False) or
                the package src directory (True).
            Default: True
    
    Returns:
    A list of list of ints, representing the rectangular (2
    dimensional) grid.
    """
    txt = loadTextFromFile(doc, rel_package_src=rel_package_src)
    #doc = doc.strip()
    #if not os.path.isfile(doc):
    #    raise FileNotFoundError(f"There is no file at location {doc}.")
    #with open(doc) as f:
    #    txt = f.read()
    return [[int(x) for x in row.split(",")]\
            for row in txt.split("\n") if row]

def gridPathTwoWayMatrix(
    mat: List[List[int]],
    preserve_mat: bool=True,
) -> int:
    """
    For an integer grid mat with r rows and c columns, finds the
    minimum cost of traversing the grid from coordinates (0, 0) to
    coordinates (r - 1, c - 1) where for integers i and j for which
    0 <= i < r and 0 <= j < c, a step can go either to (i + 1, j)
    assuming i < r - 1 or to (i, j + 1) assuming j < c - 1, and
    the cost of a path is the sum the values of the grid whose
    coordinates the path travels through (including (0, 0) and
    (r - 1, c - 1)).
    Assumes that all entries in mat are non-negative integers.
    
    Args:
        Required positional:
        mat (list of lists of ints): The grid being traversed.
                Each list in mat must have the same length as
                each other and contain only non-negative integers.
        
        Optional named:
        preserve_mat (bool): Whether a deep copy of mat should be
                created, as the process changes mat and it will
                not be able to be used for any other purposes.
            Default: True
    
    Returns:
    Integer (int) giving the minimum possible cost of traversing
    the grid based on the described conditions.
    """
    if preserve_mat: mat = [list(row) for row in mat]
    shape = (len(mat), len(mat[0]))
    for i2 in range(1, shape[1]):
        mat[0][i2] += mat[0][i2 - 1]
    for i1 in range(1, shape[0]):
        mat[i1][0] += mat[i1 - 1][0]
        for i2 in range(1, shape[1]):
            mat[i1][i2] += min(mat[i1][i2 - 1], mat[i1 - 1][i2])
    return mat[-1][-1]

def gridPathTwoWayFromFile(
    doc: str="project_euler_problem_data_files/0081_matrix.txt",
    rel_package_src: bool=True,
) -> int:
    """
    Solution to Project Euler #81

    For a rectangular grid given in the .txt file doc, finds the
    smallest cost to traverse the grid from the top left hand
    corner to the bottom right hand corner while taking steps only
    to the right and down. A traversal of the grid involves taking
    a sequence of steps from one element of the grid to another
    for which one of the coordinates is exactly one greater than
    that of the first the first and the other coordinate is the
    same (i.e. elements that are directly adjacent, not
    including diagonals to the right and down). The cost of a given
    traversal is the sum of the values of the elements of the grid
    included in the sequence of steps, including the first and last
    (in this case, the top left and bottom right elements).
    Assumes that the grid does not contain any negative values.
    
    Args:
        Optional named:
        doc (str): String giving the relative or absolute location
                of the .txt file containing the rectangular grid.
            Default: "project_euler_problem_data_files/0081_matrix.txt"
        rel_package_src (bool): Whether a relative path given by doc
                is relative to the current directory (False) or
                the package src directory (True).
            Default: True
    
    Returns:
    This smallest cost of any traversal of the grid contained in
    doc from the top left hand corner to the bottom right hand
    corner.
    """
    #since = time.time()
    mat = loadGrid(doc, rel_package_src=rel_package_src)
    res = gridPathTwoWayMatrix(mat, preserve_mat=False)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 82
# Review- consider enabling the use of the shortest path faster
#         algorithms

def gridPathThreeWayFromFile(
    doc: str="project_euler_problem_data_files/0082_matrix.txt",
    alg: str="findShortestPath",
    bidirectional: bool=True,
    rel_package_src: bool=True,
) -> int:
    """
    Solution to Project Euler #82

    For a rectangular grid given in the .txt file doc, finds
    smallest cost to traverse the grid from any element along
    the left hand edge of the grid to any element along the
    right hand edge of the grid taking only steps up, down
    or right.  A traversal of the grid involves taking a sequence
    of steps from one element of the grid to another for which
    one of the coordinates differs by one from the first and the
    other coordinate is the same (i.e. elements that are directly
    adjacent, not including diagonals), excluding steps in which
    the second coordinate value reduces (i.e. steps to the left).
    The cost of a given traversal is the sum of the values of the
    elements of the grid included in the sequence of steps,
    including the first and last (in this case, the element on
    the left hand edge where the traversal starts and the element
    on the right hand edge where the traversal ends).
    Assumes that the grid does not contain any negative values.
    
    Args:
        Optional named:
        doc (str): String giving the relative or absolute location
                of the .txt file containing the rectangular grid.
            Default: "project_euler_problem_data_files/0082_matrix.txt"
        alg (str or None): String indicating which path finding
                algorithm to use. The options are:
                 - "dijkstra" (Uses the Dijkstra algorithms)
                 - "findShortestPath" (the graph_classes package selects
                    the most appropriate algorithm)
            Default: "findShortestPath"
        bidirectional (bool): Whether the search should be bidirectional
                (if given as True) or unidirectional (if given as False).
                Bidirectional is in general faster.
            Default: True
        rel_package_src (bool): Whether a relative path given by doc
                is relative to the current directory (False) or
                the package src directory (True).
            Default: True
    
    Returns:
    This smallest cost of any traversal of the grid contained in
    doc from the left hand edge to the right hand edge with steps
    only in the up, down and right directions.
    """
    #since = time.time()
    arr = loadGrid(doc, rel_package_src=rel_package_src)
    grid = Grid(2, arr)
    has_neg = any(x < 0 for x in grid.arr_flat)
    
    #def restrictDirect(grid_idx: int, mv_idx: int) -> bool:
    #    dim_i = 1
    #    restrict = (True, False)
    #    idx_pair = [grid_idx, mv_idx]
    #    idx_pair2 = [None, None]
    #    for j, idx in enumerate(idx_pair):
    #        for i in range(dim_i):
    #            idx //= grid.shape[i]
    #        idx_pair2[j] = idx % grid.shape[dim_i]
    #    if not idx_pair[1]: return True
    #    elif sum(idx_pair) >= grid.shape[dim_i]:
    #        return restrict[0]
    #    return restrict[1]
    
    #allowed_direct_idx_func = lambda grid_idx, mv_idx: mv_idx
    #move_kwargs = {"n_diag": 0, "n_step_cap": 1,\
    #        "allowed_direct_idx_func": restrictDirect}#{1: (True, False)}}
    move_kwargs = {"n_diag": 0, "n_step_cap": 1,\
            "directed_axis_restrict": {1: (True, False)}}
    
    weight_func = lambda grid, grid_idx1, state_idx1, grid_idx2,\
            state_idx2, mv, n_step: grid.arr_flat[grid_idx2]
    graph = GridWeightedDirectedGraph(
        grid,
        move_kwargs=move_kwargs,
        weight_func=weight_func,
        neg_weight_edge=has_neg
    )
    starts = {((x, 0), 0): grid[x, 0] for x in range(grid.shape[0])}
    ends = {((x, grid.shape[1] - 1), 0): 0 for x in range(grid.shape[0])}
    func = getattr(graph, alg)
    res = func(starts, ends, bidirectional=bidirectional)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res[0]
    """
    since = time.time()
    arr = loadGrid(doc)
    grid = Grid(2, arr)
    
    def _moveAdj3Way(self, idx: Tuple[int])\
            -> Generator[Tuple[Union[Tuple[int], int, float]], None, None]:
        if idx[0]:
            idx2 = (idx[0] - 1, idx[1])
            yield (idx2, self.grid[idx2])
        if idx[0] + 1 < self.shape[0]:
            idx2 = (idx[0] + 1, idx[1])
            yield (idx2, self.grid[idx2])
        if idx[1] + 1 < self.shape[1]:
            idx2 = (idx[0], idx[1] + 1)
            yield (idx2, self.grid[idx2])
        return
    
    move_str = "moveAdj3Way"
    GridGraph.addMoveGenerator(move_str, _moveAdj3Way)
    
    graph = GridGraph(grid, move_str=move_str)
    res = dijkstra(graph, {(x, 0): grid[x, 0] for x in range(grid.shape[0])},\
            {(x, grid.shape[1] - 1): 0 for x in range(grid.shape[0])})
    #print(res[1])
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res[0]
    """
    
# Problem 83
def gridPathFourWayFromFile(
    doc: str="project_euler_problem_data_files/0083_matrix.txt",
    alg: str="findShortestPath",
    bidirectional: bool=True,
    rel_package_src: bool=True,
) -> int:
    """
    Solution to Project Euler #83

    For a rectangular grid given in the .txt file doc, finds
    smallest cost to traverse the grid from the top left hand
    corner to the bottom right hand corner. A traversal of the
    grid involves taking a sequence of steps from one element of
    the grid to another for which one of the coordinates differs
    by one from the first and the other coordinate is the
    same (i.e. elements that are directly adjacent, not
    including diagonals). The cost of a given traversal is the
    sum of the values of the elements of the grid included in
    the sequence of steps, including the first and last (in this
    case, the top left and bottom right elements).
    Assumes that the grid does not contain any negative values.
    
    Args:
        Optional named:
        doc (str): String giving the relative or absolute location
                of the .txt file containing the rectangular grid.
            Default: "project_euler_problem_data_files/0083_matrix.txt"
        alg (str or None): String indicating which path finding
                algorithm to use. The options are:
                 - "dijkstra" (Uses the Dijkstra algorithms)
                 - "findShortestPath" (the graph_classes package selects
                    the most appropriate algorithm)
            Default: "findShortestPath"
        bidirectional (bool): Whether the search should be bidirectional
                (if given as True) or unidirectional (if given as False).
                Bidirectional is in general faster.
            Default: True
        rel_package_src (bool): Whether a relative path given by doc
                is relative to the current directory (False) or
                the package src directory (True).
            Default: True
    
    Returns:
    This smallest cost of any traversal of the grid contained in
    doc from the top left hand corner to the bottom right hand
    corner.
    """
    #since = time.time()
    arr = loadGrid(doc, rel_package_src=rel_package_src)
    grid = Grid(2, arr)
    has_neg = any(x < 0 for x in grid.arr_flat)
    if alg is None: alg = "findShortestPath"
    
    move_kwargs = {"n_diag": 0, "n_step_cap": 1}
    weight_func = lambda grid, grid_idx1, state_idx1, grid_idx2,\
            state_idx2, mv, n_step: grid.arr_flat[grid_idx2]
    graph = GridWeightedDirectedGraph(grid, move_kwargs=move_kwargs,\
                weight_func=weight_func, neg_weight_edge=has_neg)
    starts = {((0, 0), 0): grid[0, 0]}
    ends = {(tuple(x - 1 for x in grid.shape), 0): 0}
    func = getattr(graph, alg)
    res = func(starts, ends, bidirectional=bidirectional)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res[0]
    """
    since = time.time()
    arr = loadGrid(doc)
    grid = Grid(2, arr)
    graph = GridGraph(grid)
    res = dijkstra(graph, {(0, 0): grid[0, 0]}, {tuple(x - 1 for x in grid.shape): 0})
    #print(res[1])
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res[0]
    """

# Problem 84
def diceDistributionRepeats(
    n_faces: int,
    n_dice: int,
) -> Tuple[Union[Dict[int, Tuple[int]], int]]:
    # Review- simplify documentation. Repeated use of the word
    # value in multiple different contexts
    """
    Finds the distribution of the result of summing the rolls of
    n_dice independent fair n_faces faced dice, treating rolls with
    multiple dice with the same value separately from those where
    all the dice in a roll have different values.
    The dice each have the integers 1 to n_faces inclusive which
    are each equally likely to occur on a given roll.
    
    Args:
        Required positional:
        n_faces (int): Strictly positive integer giving the number
                of faces of the dice rolled.
        n_dice (int): Strictly positive integer giving the number
                of dice rolled whose results are summed.
    
    Returns:
    2-tuple whose index 0 contains a dictionary whose keys are integers
    giving the possible results of the sum of the dice rolled with
    the corresponding value being a 2-tuple of ints whose index 0 and
    1 are integers (ints) giving the number of distinct rolls that sum
    to that result which do not contain and do contain respectively
    multiple dice in the roll with the same value and whose index 1
    contains the total number of distinct rolls. In assessing whether
    two rolls are distinct, the n_dice dice are considered to be
    distinguishable, so for example with n_faces=6 and n_dice=2, the
    roll where die 1 is 1 and die 2 is 2 is distinct from the roll where
    die 1 is 2 and die 2 is 1.
    For fair and independent dice, each of the distinct rolls is
    equally likely. As such, the probability of a given integer being
    the sum of a roll of the dice as described where all the dice
    values in the roll are unique is given by index 0 of value in the
    dictionary contained in index 0 whose key is that integer (or 0 if
    the integer does not appear in the dictionary) divided by the value
    contained in index 1 and the probability of a given integer being
    the sum of a roll of the dice as described where there is at least
    one value where more than one die in the roll has that value is
    given by index 1 of value in the dictionary contained in index 1
    whose key is that integer (or 0 if the integer does not appear in
    the dictionary) divided by the value contained in index 1.
    Furthermore, given that the set of such dice rolls with a given
    sum for which all the dice values in the roll are unique and the
    set of such dice rolls with the same sum for which there is at
    least one value where more than one die in the roll has that value
    are mutually exclusive (i.e. are disjoint sets) and collectively
    cover all the rolls with the given sum (i.e. their union is the set
    of all such dice rolls with the given sum), the number of distinct
    rolls with the given sum is the total of the contents of indices 0
    and 1 of the value of the dictionary corresponding to the key equal
    to the given sum (or 0 if the given sum is not a key of the
    dictionary), with corresponding probability being this number
    divided by the value contained in index 1 of the returned 2-tuple.
    
    Examples:
    >>> diceDistributionRepeats(6, 2)
    ({2: (0, 1),
      3: (2, 0),
      4: (2, 1),
      5: (4, 0),
      6: (4, 1),
      7: (6, 0),
      8: (4, 1),
      9: (4, 0),
      10: (2, 1),
      11: (2, 0),
      12: (0, 1)},
     36)
    This signifies that when rolling two dice each with 6 faces
    labelled 1 to 6 inclusive there are exactly 36 distinct rolls. It
    also signifies that for instance there are exactly 4 distinct rolls
    that give the sum 6 for which all of the dice values in the roll
    are unique (which are the ordered pairs (1, 5), (2, 4), (4, 2) and
    (5, 1)), corresponding to a probability of 4 / 36 = 1 / 9, and
    exactly 1 distinct roll that gives the sum 6 for which there is
    a value that appears more than once among the dice rolled (the
    ordered pair (3, 3)) corresponding to a probability of 1 / 36.
    Given that these events are mutually exclusive and are collectively
    the only ways to get a sum of 6, we may conclude that there are
    exactly 4 + 1 distinct rolls that give the sum 6 corresponding
    to a probability of 5 / 36.
    
    >>> diceDistributionRepeats(6, 3)
    ({3: (0, 1),
      4: (0, 3),
      5: (0, 6),
      6: (6, 4),
      7: (6, 9),
      8: (12, 9),
      9: (18, 7),
      10: (18, 9),
      11: (18, 9),
      12: (18, 7),
      13: (12, 9),
      14: (6, 9),
      15: (6, 4),
      16: (0, 6),
      17: (0, 3),
      18: (0, 1)},
     216)
    This signifies that when rolling three dice each with 6 faces
    labelled 1 to 6 inclusive there are exactly 216 distinct rolls. It
    also signifies that for instance there are exactly 6 distinct rolls
    that give the sum 6 for which all of the dice values in the roll
    are unique (which are the ordered triples (1, 2, 3), (1, 3, 2),
    (2, 1, 3), (2, 3, 1), (3, 1, 2) and (3, 2, 1)), corresponding to a
    probability of 6 / 216 = 1 / 36, and exactly 4 distinct rolls that
    give the sum 6 for which there is a value that appears more than
    once among the dice rolled (the ordered triples (1, 1, 4),
    (1, 4, 1), (4, 1, 1), (2, 2, 2)) corresponding to a probability of
    4 / 216 = 1 / 54.
    Given that these events are mutually exclusive and are collectively
    the only ways to get a sum of 6, we may conclude that there are
    exactly 6 + 4 distinct rolls that give the sum 6 corresponding
    to a probability of 10 / 216 = 1 / 108.
    """
    # Brute force
    iter_objs = [range(1, n_faces + 1) for _ in range(n_dice)]
    tots = {}
    for nums in itertools.product(*iter_objs):
        k = sum(nums)
        tots.setdefault(k, [0, 0])
        tots[k][len(set(nums)) < len(nums)] += 1
    for k in tots.keys():
        tots[k] = tuple(tots[k])
    return (tots, n_faces ** n_dice)

def monopolyTransitionFunction(
    start_idx: int,\
    dice_distr: Tuple[Union[Dict[int, Tuple[int]], int]]
) -> Dict[Tuple[Union[int, bool]], float]:
    """
    From a given starting square on the Monopoly board, and a given
    probability distribution of dice roll results, determines the
    probabilities that the turn will end on the squares it is possible
    to end on taking into account the possibility of being sent to Jail
    and being sent to other squares by Chance and Community Chest
    cards, where for the Jail square, In Jail and Just Visiting are
    considered separate and whether or not the dice roll of that turn
    was a double also considered separately.
    Each square is represented by an index from 0 to 39 inclusive where
    the GO square corresponds to index 0 and each other square
    corresponds to the index exactly one greater than that of the
    previous square when traversing clockwise round the board.
    Each end state is represented by an ordered pair whose index
    0 contains the index corresponding to the square (as outlined
    above) and whose index 1 contains a boolean representing whether
    if the turn ends in Jail (with True if in Jail and False
    otherwise). This boolean can only be True if the index corresponds
    to the Jail square (i.e. 10).
    A double dice roll is one in which more than one dice in that
    roll has the same value.
    Note that the multiple dice throws resulting from doubles are
    counted as separate turns, and this ignores any previous
    consecutive doubles thrown (i.e. this transition function does
    not account for the rule stating that after a given number of
    consecutive doubles the player is sent to Jail).
    
    Args:
        Required positional:
        start_idx (int): Integer between 0 and 39 inclusive giving the
                index corresponding to the square from which the given
                turn starts.
                Each square is represented by an index from 0 to 39
                inclusive where the GO square corresponds to index 0
                and each other square corresponds to the index exactly
                one greater than that of the previous square when
                traversing clockwise round the board.
        dice_distr (2-tuple): 2-tuple giving the distribution of the
                sums of dice of each roll, separating the rolls
                are not doubles (i.e. those whose dice values are all
                different) from the rolls that are doubles (i.e. those
                for which two or more of the dice have the same value).
                Index 0 of the tuple should contain a dictionary whose
                keys are the possible dice roll sums with the
                corresponding value being a 2-tuple whose index 0 and
                1 contain the number of distinct rolls whose dice sum
                to the key that are not doubles and are doubles
                respectively. Meanwhile, index 1 of the tuple should
                contain the total number of distinct rolls possible
                (and should equal the sum of all the numbers of
                distinct rolls counted in the dictionary in index 1).
                It is assumed that each distinct dice roll is equally
                likely, so that the probability of a given outcome
                (i.e. a given sum with or without doubles) is the
                corresponding number in the dictionary in index 0
                divided by the total in index 1.
                This is intended to directly take the output of
                function diceDistributionRepeats() for a given number
                of dice faces and number of dice.
    
    Returns:
    Dictionary (dict) whose keys are 2-tuples representing the possible
    state after exactly one turn from the given starting square (as
    specified by start_idx) whose index 0 contains an integer between 0
    and 39 inclusive corresponding to a square it is possible to end on
    and whose index 1 contains a boolean (bool) which can be True only
    if that square is the Jail square, in which case it represents the
    turn ending in Jail as opposed to Just Visiting. For a given key,
    the corresponding value is a 2-tuple whose indices 0 and 1 contain
    floats giving the probability of a turn starting from the given
    starting square ending ends in the state represented by the key
    with a dice roll that is not a double and is a double respectively.
    Note that the sum over all the floats in all the 2-tuples in the
    values of the dictionary should sum to 1.0 (allowing for float
    rounding errors).
    Also note that any state that does not have a corresponding key
    in the dictionary is taken to have probability 0 of being the state
    after exactly one turn from the given starting square, both with
    a dice roll that is or is not a double.
    """
    n = 40 # The number of squares on the board
    go_idx = 0 # The index corresponding to the GO square
    jail_idx = 10 # The index corresponding to the Jail square
    gtj_idx = 30 # The index corresponding to the Go To Jail square
    n_chance = 16 # The total number of Chance cards
    chance_send_idx = {11, 24, 39, 5} # The indices a Chance card may
                                      # send the player
    chance_idx = {7, 22, 36} # The indices of the Chance squares
    n_cc = 16 # The total number of Community Chest cards
    cc_idx = {2, 17, 33} # The indices of the Community Chest squares
    rail_idx = (5, 15, 25, 35) # The indices of the Railway squares
    util_idx = (12, 28) # The indices of the Utility squares
    
    go_idx2 = (go_idx, False)
    jail_idx2 = (jail_idx, True)
    gtj_idx2 = (gtj_idx, False)
    
    def communityChest(idx: int) -> Tuple[Union[Dict[int, int], int]]:
        #print("cc")
        # Send to jail and send to GO
        res = {go_idx2: 1, jail_idx2: 1}
        # No move
        tot = sum(res.values())
        res[(idx, False)] = res.get((idx, False), 0) + n_cc - tot
        return (res, n_cc)
    
    def chance(idx: int) -> Tuple[Union[Dict[int, int], int]]:
        #print("chance")
        # Send to jail and send to GO
        res = {go_idx2: 1, jail_idx2: 1}
        # Send back 3 spaces
        idx2 = ((idx - 3) % n, False)
        res[idx2] =\
                res.get(idx2, 0) + 1
        # Send to specific square
        for i in chance_send_idx:
            idx2 = (i, False)
            res[idx2] = res.get(idx2, 0) + 1
        # Send to next railway
        j = bisect.bisect_left(rail_idx, idx) % len(rail_idx)
        idx2 = (rail_idx[j], False)
        res[idx2] =\
                res.get(idx2, 0) + 2
        # Send to next utility
        j = bisect.bisect_left(util_idx, idx) % len(util_idx)
        idx2 = (util_idx[j], False)
        res[idx2] =\
                res.get(idx2, 0) + 1
        # No move
        tot = sum(res.values())
        idx2 = (idx, False)
        res[idx2] = res.get(idx2, 0) + n_chance - tot
        return (res, n_chance)    
    
    
    res = {}
    div = dice_distr[1]
    for n_steps, p_pair in dice_distr[0].items():
        finished = False
        stk = [(((start_idx + n_steps) % n, False), p_pair, div)]
        while stk:
            idx2, p_pair2, div2 = stk.pop()
            if idx2 == gtj_idx2:
                res[jail_idx2] = tuple(p0 + p / div2 for p0, p in\
                        zip(res.get(jail_idx2, [0] * len(p_pair2)),\
                        p_pair2))
                continue
            for (idx_set, distr_func) in ((chance_idx, chance),\
                    (cc_idx, communityChest)):
                if idx2[0] not in idx_set: continue
                distr, div4 = distr_func(idx2[0])
                for idx4, p4 in distr.items():
                    if idx4 == idx2:
                        res[idx4] = tuple(p0 + p * p4 / (div2 * div4)\
                                for p0, p in zip(res.get(idx4,\
                                [0] * len(p_pair2)), p_pair2))
                    else:
                        stk.append((idx4, [p * p4 for p in p_pair2],\
                                div2 * div4))
                break
            else:
                res[idx2] = tuple(p0 + p / div2 for p0, p in\
                        zip(res.get(idx2, [0] * len(p_pair2)),\
                        p_pair2))
    return res

def monopolyOdds(
    n_dice_faces: int=6,
    n_dice: int=2,
    n_double_jail: Optional[int]=3,
    jail_resets_doubles: bool=True
) -> List[float]:
    """
    Calculates the expected proportion of turns in a Monopoly game
    that end on each square, where the sum of rolls of n_dice fair
    dice with n_dice_faces (where the faces are labelled with the
    integers from 1 to n_dice_faces inclusive and no two faces have
    the same label) is used to determine the number of squares to
    move forward on each turn.
    If n_double_jail is given as an integer, then this number of
    consecutive doubles are rolled by a player results in automatically
    being sent to Jail, where a double is defined as a roll of the dice
    for which two or more of the dice rolled give the same value. If
    n_double_jail is given as None then no number of consecutive
    doubles will automatically result in being sent to Jail.
    It is assumed that directly after being sent to jail, the necessary
    fee to be let out of Jail is immediately paid.
    Additionally this makes the minor simplification by assuming that
    the Chance and Community Chest decks are full and freshly shuffled
    prior to each draw of a card. This is different from typical
    monopoly play, but without this assumption the method used would
    become considerably more complicated and potentially the problem
    may not be solvable. However, this simplification should not
    affect the results greatly as the changes in probability from
    not freshly shuffling the chance and community chest decks should
    be extremely small, particularly in the context of all possible
    moves and regardless would likely largely cancel each other out.
    
    Args:
        Optional named:
        n_dice_faces (int): Strictly positive integer giving the number
                of faces on each die used. The dice used are fair and
                the faces are labelled each with an integer from 1 to
                n_dice_faces inclusive such that no two sides are
                labelled with the same number. Furthermore, the dice
                are assumed to be fair, so a roll of each die is
                equally likely to result in each face being rolled.
            Default: 6
        n_dice (int): Strictly positive integer giving the number of
                dice used in each roll. The result of the roll is the
                sum of the labels of the faces rolled for each die.
            Default: 2
        n_double_jail (int or None): If given as an integer, this gives
                the number of consecutive double dice rolls that result
                in automatically being sent to Jail. If given as None
                then there is no such restriction on consecutive double
                dice rolls.
                A dice roll is a double if and only if two or more of
                the dice being rolled have the same value.
            Default: 3
        jail_resets_doubles (bool): Boolean specifying whether being
                sent to Jail for any reason resets the count of
                consecutive double dice rolls to zero. If True, then
                being sent to Jail for any reason resets this count
                to zero, while if False then only being sent to Jail
                for n_double_jail consecutive double rolls or rolling
                a non-double roll resets the count to zero.
            Default: True
    
    Returns:
    List of floats of length 40, giving the expected proportion of
    turns that end on each of the squares on the Monopoly board.
    Each square is represented by an index from 0 to 39 inclusive where
    the GO square corresponds to index 0 and each other square
    corresponds to the index exactly one greater than that of the
    previous square when traversing clockwise round the board, and
    the expected proportion of turns that end at a given square is
    given by the entry at the corresponding index in the returned
    list.
    
    Outline of rationale:
    TODO
    
    
    Uses Markov chains
    Define a state of the board as being an ordered pair whose first
    element is a square represented by an index between 0 and 39
    inclusive (where the GO square is index 0 and each other square
    has index exactly one greater than the previous square when
    traversing clockwise round the board) whose second element is an
    integer between 0 and (n_double_jail - 1) inclusive representing
    the length of the current run of consecutive rolls with multiple
    dice having the same value (in the case of 2 dice, doubles)
    without having been sent to jail. There are exactly
    40 * n_double_jail such states. Furthermore, given the assumptions
    about the game, for given a pair of states, the probability
    of a move (i.e. a dice roll) from the first state in the pair
    ends up in the second state in the pair can be calculated
    by just the information in the two states and no additional
    information is required. In particular, the states prior to
    the first state in the pair have no influence on the result,
    in other words the game has no memory. This enables the use of
    Markov chains.
    We label the states from 0 to (40 * n_double_jail - 1) inclusive.
    For each ordered pair of states, we calculate the probability
    of a move from the first in the pair ending up at the second
    in the pair, and organise these numbers into a (40 * n_double_jail)
    x (40 * n_double_jail) matrix, which we call M, such that
    (using 0-indexing) for 0 <= i, j < (40 * n_double_jail), M_(i,j)
    is the probability of a move from the state labelled j ending
    up at the state labelled i. This is a so-called left stochastic
    matrix (i.e. a real square matrix with each column summing to 1).
    
    
     and these collectively describe
    the information of every single possible state a player can
    be in at any point in the game that influences their future
    moves.
    For ea
    
    For each square and each lenght of run of consecutive rolls with
    multiple dice with the same value (in the case of 2 dice, doubles)
    
    """
    eps = 10 ** -6
    n = 40
    jail_idx = 10
    
    if n_dice == 1:
        # Impossible to roll a double if only one die is used
        n_double_jail = None

    dice_distr = diceDistributionRepeats(n_dice_faces, n_dice)
    mat_sz = n * (1 if n_double_jail is None else n_double_jail)
    mat = np.zeros(tuple([mat_sz] * 2), dtype=float)
    
    if n_double_jail is None:
        for idx in range(n):
            tf = monopolyTransitionFunction(idx, dice_distr)
            for (idx2, b), p_pair in tf.items():
                mat[idx2, idx] += sum(p_pair)
    else:
        for idx in range(n):
            tf = monopolyTransitionFunction(idx, dice_distr)
            for i in range(n_double_jail - 1):
                add1 = i * n
                add2 = (i + 1) * n
                j1 = idx + add1
                for (idx2, b), p_pair in tf.items():
                    mat[idx2, j1] += p_pair[0]
                    mat[idx2 + (0 if b and jail_resets_doubles else\
                            add2), j1] += p_pair[1]
            i = n_double_jail - 1
            add1 = i * n
            add2 = (i + 1) * n
            j1 = idx + add1
            for (idx2, b), p_pair in tf.items():
                mat[idx2, j1] += p_pair[0]
                mat[jail_idx, j1] += p_pair[1]
    e_vals, e_vecs = np.linalg.eig(mat)
    for i, e_val in enumerate(e_vals):
        if abs(e_val.real - 1) < eps and abs(e_val.imag) < eps:
            break
    else:
        raise ValueError("The matrix does not have an eigenvalue close "
                "to 1")
    e_vec = e_vecs[:, i]
    e_vec_real = [v.real for v in e_vec]
    sm = sum(e_vec_real)
    e_vec_real = [v / sm for v in e_vec_real]
    res = [0] * n
    for i, v in enumerate(e_vec_real):
        res[i % n] += v
    return res

def monopolyOddsMostVisited(
    n_dice_faces: int=4,
    n_dice: int=2,
    n_double_jail: Optional[int]=3,
    jail_resets_doubles: bool=False,
    n_return: int=3,
) -> str:
    """
    Solution to Project Euler #84

    Finds the n_return squares on the Monopoly board on which the
    greatest expected proportion of turns end where the sum of rolls
    of n_dice fair dice with n_dice_faces (where the faces are labelled
    with the integers from 1 to n_dice_faces inclusive and no two faces
    have the same label) is used to determine the number of squares to
    move forward on each turn. The returned value is a string comprised
    of the concatenation of the indices corresponding to each of
    these squares in decreasing order of this expected proportion
    (where the index corresponding to the GO square is 0 and for every
    other square the corresponding index is exactly one greater than
    that of the previous square when traversing clockwise round the
    board).
    If n_double_jail is given as an integer, then this number of
    consecutive doubles are rolled by a player results in automatically
    being sent to Jail, where a double is defined as a roll of the dice
    for which two or more of the dice rolled give the same value. If
    n_double_jail is given as None then no number of consecutive
    doubles will automatically result in being sent to Jail.
    The Jail square accounts for both Jail itself and Just Visiting.
    It is assumed that directly after being sent to jail, the necessary
    fee to be let out of Jail is immediately paid.
    Additionally this makes the minor simplification by assuming that
    the Chance and Community Chest decks are full and freshly shuffled
    prior to each draw of a card. This is different from typical
    monopoly play, but without this assumption the method used would
    become considerably more complicated and potentially the problem
    may not be solvable. However, this simplification should not
    affect the results greatly as the changes in probability from
    not freshly shuffling the chance and community chest decks should
    be extremely small, particularly in the context of all possible
    moves and regardless would likely largely cancel each other out.
    
    Args:
        Optional named:
        n_dice_faces (int): Strictly positive integer giving the number
                of faces on each die used. The dice used are fair and
                the faces are labelled each with an integer from 1 to
                n_dice_faces inclusive such that no two sides are
                labelled with the same number. Furthermore, the dice
                are assumed to be fair, so a roll of each die is
                equally likely to result in each face being rolled.
            Default: 4
        n_dice (int): Strictly positive integer giving the number of
                dice used in each roll. The result of the roll is the
                sum of the labels of the faces rolled for each die.
            Default: 2
        n_double_jail (int or None): If given as an integer, this gives
                the number of consecutive double dice rolls that result
                in automatically being sent to Jail. If given as None
                then there is no such restriction on consecutive double
                dice rolls.
                A dice roll is a double if and only if two or more of
                the dice being rolled have the same value.
            Default: 3
        jail_resets_doubles (bool): Boolean specifying whether being
                sent to Jail for any reason resets the count of
                consecutive double dice rolls to zero. If True, then
                being sent to Jail for any reason resets this count
                to zero, while if False then only being sent to Jail
                for n_double_jail consecutive double rolls or rolling
                a non-double roll resets the count to zero.
            Default: False
        n_return (int): Strictly positive integer specifying the number
                of the squares with the highest proportion of turns
                ending on those squares whose indices are included in
                the returned string.
            Default: 3
    
    Returns:
    String (str) comprised of the concatenation of the indices
    representing the n_return squares on which the greatest expected
    proportion of turns end, in decreasing order of this value. In the
    case of a tie, the square with the smaller index is given first.
    The index of each square is an integer from 0 to 39 inclusive where
    the GO square corresponds to index 0 and each other square
    corresponds to the index exactly one greater than that of the
    previous square when traversing clockwise round the board.
    In the returned string, each index is expressed in base 10 and
    is padded to the left by zeros so that each index given has the
    same number of digits as the largest possible index (i.e. 39,
    meaning each index is padded to the left with zeros to ensure
    they all contain exactly 2 digits. For instance, 0 is included as
    "00" and 2 is included as "02").
    """
    #since = time.time()
    p_lst = monopolyOdds(n_dice_faces=n_dice_faces, n_dice=n_dice,\
            n_double_jail=n_double_jail,\
            jail_resets_doubles=jail_resets_doubles)
    p_lst_sort = sorted([(i, x) for i, x in enumerate(p_lst)],\
            key=lambda x: x[1])
    n = len(p_lst)
    return_num_lens = len(str(n - 1))
    res = []
    for i in range(min(n_return, n)):
        res.append(str(p_lst_sort[~i][0]).zfill(return_num_lens))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return "".join(res)

# Problem 85
def countingRectangles(target_count: int=2 * 10 ** 6) -> int:
    """
    Solution to Project Euler #85

    Consider a rectangular grid with integer side lengths and grid
    lines every unit starting from and parallel to each of the
    edges on the interior and edges of the rectangle. A certain
    number of distinct rectangles can be formed with their edges
    only on these grid lines, where two such rectangles are
    considered identical if all four of their edges exactly
    coincide (so there may be multiple distinct rectangles with
    the same dimensions and orientations, as long as they do
    not overlap exactly).
    This function finds the area of the rectangular grid such that
    the number of such distinct rectangles that can be formed
    in the grid in this way is closest to target_count (i.e. the
    rectangular grid for which abs(count - target_count) is
    minimised, where count is the number of distinct rectangles
    that can be formed in the grid in the manner described).
    
    Args:
        Optional named:
        target_count (int): The number of rectangles that can
                be formed in the manner described that is
                being targeted.
            Default: 2 * 10 ** 6
    
    Returns:
    Integer (int) giving the area of the rectangular grid for
    which the number of distinct rectangles that can be formed
    in the manner described is closest to target_count.
    """
    #since = time.time()
    m = 1
    best_diff = float("inf")
    res = 0
    n = float("inf")
    while n > m:
        c1 = (m * (m + 1)) >> 1
        a = target_count // c1
        n = (isqrt(1 + 8 * a) - 1) >> 1
        count1 = c1 * ((n * (n + 1)) >> 1)
        diff1 = target_count - count1
        if diff1 < best_diff:
            best_diff = diff1
            res = m * n
            if not diff1: return res
        count2 = c1 * (((n + 1) * (n + 2)) >> 1)
        diff2 = count2 - target_count
        
        if diff2 < best_diff:
            best_diff = diff2
            res = m * (n + 1)
            if not diff2: return res
        #print(m, n, a, diff1, diff2)
        m += 1
    #print(res, best_diff)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 86
def pythagoreanTripleABMaxRangeGenerator(
    ab_mx_max: Optional[int]=None,
    ab_mx_min: int=0,
) -> Generator[Tuple[int, int, int], None, None]:
    """
    Generator iterating over Pythagorean triples, yielding them
    in order of increasing size of the larger of the two
    catheti (i.e. the number that is neither the smallest nor
    largest value in the Pythagorean triple), for values of this
    cathetus between ab_mx_min and ab_mx_max inclusive (or
    endlessly if ab_mx_max is not given or given as None).

    A cathetus (pl. catheti) of a right angled triangle is one
    of the sides adjacent to the right angle (i.e. one of the
    two sides that are not the hypotenuse).

    Args:
        Optional named:
        ab_mx_max (None or int): If given, specifies the
                inclusive upper bound on the value of the
                larger cathetus of any Pythagorean triple
                yielded.
                If this is not given or given as None, the
                iterator will not self-terminate, so any loop
                based around this iterator must contain a
                mechanism to break the loop (e.g. break or
                return) to avoid an infinite loop.
            Default: None
        ab_mx_max (int): Specifies the inclusive lower bound
                on the value of the larger cathetus of
                any Pythagorean triple yielded.
            Default: 0
    
    Yields:
    3-tuple of integers (int) specifying the corresponding
    Pythagorean triple, with the 3 items ordered in increasing
    size (so the hypotenuse is last).
    The triples are yielded in order of increasing size of
    the middle of these three values, with triples with the
    same such value yielded in increasing order of the smallest
    of the three values (the shorter cathetus).
    """
    if ab_mx_max is None:
        ab_mx_max = float("inf")
    elif ab_mx_max < ab_mx_min: return

    h = []

    def yieldCurrent(ab_mx_ub: Real) -> Generator[Tuple[int, int, int], None, None]:
        while h and h[0][0] < ab_mx_ub:
            _, prim, k = heapq.heappop(h)
            yield tuple(k * x for x in prim)
            k2 = k + 1
            mid = prim[1] * k2
            if mid > ab_mx_max: continue
            heapq.heappush(h, (mid, prim, k2))
        return
    
    #sqrt = isqrt(ab_mx_max)
    #for m in range(1, sqrt + 1):
    
    for m in range(1, (ab_mx_max >> 1) + 1):
        n1 = isqrt(2 * m ** 2) - 1
        n1 -= not ((m - n1) & 1)
        n2 = n1 + 1
        ab_mx_lb = min(m ** 2 - n1 ** 2, 2 * m * n2)
        if ab_mx_lb > ab_mx_max: break
        yield from yieldCurrent(ab_mx_lb)
        for n in range(1 + (m & 1), m, 2):
            if gcd(m, n) != 1: continue
            primitive = sorted([m ** 2 - n ** 2, 2 * m * n,\
                    m ** 2 + n ** 2])
            primitive_ab_mx = primitive[1]
            k0 = max(1, ((ab_mx_min - 1) // primitive_ab_mx) + 1)
            mid = primitive_ab_mx * k0
            if mid <= ab_mx_max:
                heapq.heappush(h, (mid, primitive, k0))
            #for k in range(max(1,\
            #        (-((-ab_mx_min) // primitive_ab_mx))),\
            #        ab_mx_max // primitive_ab_mx + 1):
            #    yield tuple(k * x for x in primitive)
    yield from yieldCurrent(float("inf"))
    return

def pythagoreanTripleABMinRangeGenerator(
    ab_mn_max: int,
    ab_mn_min: int=0,
) -> Generator[Tuple[int], None, None]:
    """
    Generator iterating over Pythagorean triples, yielding them
    in order of increasing size of the smaller of the two
    catheti (i.e. the smallest value in the Pythagorean triple),
    for values of this cathetus between ab_mn_min and ab_mn_max
    inclusive (or endlessly if ab_mn_max is not given or given
    as None).

    A cathetus (pl. catheti) of a right angled triangle is one
    of the sides adjacent to the right angle (i.e. one of the
    two sides that are not the hypotenuse).

    Args:
        Optional named:
        ab_mn_max (None or int): If given, specifies the
                inclusive upper bound on the value of the
                smaller cathetus of any Pythagorean triple
                yielded.
                If this is not given or given as None, the
                iterator will not self-terminate, so any loop
                based around this iterator must contain a
                mechanism to break the loop (e.g. break or
                return) to avoid an infinite loop.
            Default: None
        ab_mn_max (int): Specifies the inclusive lower bound
                on the value of the larger cathetus of
                any Pythagorean triple yielded.
            Default: 0
    
    Yields:
    3-tuple of integers (int) specifying the corresponding
    Pythagorean triple, with the 3 items ordered in increasing
    size (so the hypotenuse is last).
    The triples are yielded in order of increasing size of
    the smallest of these three values, with triples with the
    same such value yielded in increasing order of the middle
    of the three values (the longer cathetus).
    """
    if ab_mn_max is None:
        ab_mn_max = float("inf")
    elif ab_mn_max < ab_mn_min: return

    h = []

    def yieldCurrent(ab_mn_ub: Real) -> Generator[Tuple[int, int, int], None, None]:
        while h and h[0][0] < ab_mn_ub:
            _, prim, k = heapq.heappop(h)
            yield tuple(k * x for x in prim)
            k2 = k + 1
            lo = prim[0] * k2
            if lo > ab_mn_max: continue
            heapq.heappush(h, (lo, prim, k2))
        return

    if not ab_mn_max or ab_mn_max < ab_mn_min: return
    for m in range(1, ((ab_mn_max + 1) >> 1) + 1):
        n1 = 1 + (m & 1)
        n2 = m - 1
        n2 -= not ((m - n2) & 1)
        ab_mn_lb = min(m ** 2 - n2 ** 2, 2 * m * n1)
        if ab_mn_lb > ab_mn_max: break
        yield from yieldCurrent(ab_mn_lb)
        for n in range(1 + (m & 1), m, 2):
            if gcd(m, n) != 1: continue
            primitive = sorted([m ** 2 - n ** 2, 2 * m * n,\
                    m ** 2 + n ** 2])
            primitive_ab_mn = primitive[0]
            k0 = max(1, ((ab_mn_min - 1) // primitive_ab_mn) + 1)
            lo = primitive_ab_mn * k0
            if lo <= ab_mn_max:
                heapq.heappush(h, (lo, primitive, k0))
            #for k in range(max(1,\
            #        (-((-ab_mn_min) // primitive_ab_mn))),\
            #        ab_mn_max // primitive_ab_mn + 1):
            #    yield tuple(k * x for x in primitive)
    return

def integerMinCuboidRoute(
    target_count: int=10 ** 6,
    increment: int=500,
) -> int:
    """
    Solution to Project Euler #86

    Consider the set of cuboids whose side lengths are all integers and the
    shortest path along the surface of the cuboid between opposite vertices
    is an integer. This function finds the smallest integer such that
    the number of cuboids in this set for which the largest side length
    does not exceed this integer is strictly greater than target_count.
    
    Args:
        Optional named:
        target_count (int): Non-negative integer giving the target number of
                cuboids.
            Default: 10 ** 6
        increment (int): The increment of the maximum side length considered
                at each step of the iteration (i.e. finds the cuboid counts
                for side lengths 0 up to increment - 1 (inclusive, if
                target_count is not exceeded then finds the cuboid counts
                for side lengths increment up to 2 * increment - 1 (inclusive)
                etc. until target_count is exceeded).
            Default: 500
    
    Returns:
    The smallest integer (int) such that the number of cuboids whose side
    lengths are integers not exceeding this number and shortest path along
    the surface of the cuboid between opposite vertices is also an integer
    exceeds target_count.
    
    Outline of rationale:
    TODO
    """
    #since = time.time()
    start = 0
    prev_cumu = 0
    while True:
        end = start + increment
        cumu = [0] * increment
        cumu[0] = prev_cumu
        #print(f"Range = ({start}, {end - 1})")
        #print(f"max:")
        for triple in pythagoreanTripleABMaxRangeGenerator(
            ab_mx_max=end - 1,
            ab_mx_min=start,
        ):
            a, b = triple[:2]
            cumu[b - start] += (a >> 1)
        #print(f"min:")
        for triple in pythagoreanTripleABMinRangeGenerator(
            ab_mn_max=end - 1,
            ab_mn_min=start,
        ):
            a, b = triple[:2]
            cumu[a - start] += max(0, (b >> 1) + a - b + 1)
        if cumu[0] > target_count:
            #print(cumu)
            #print(f"Time taken = {time.time() - since:.4f} seconds")
            return start
        for i in range(1, increment):
            cumu[i] += cumu[i - 1]
            if cumu[i] > target_count:
                #print(cumu)
                #print(f"Time taken = {time.time() - since:.4f} seconds")
                return start + i
        start = end
        prev_cumu = cumu[-1]
        #print(start, cumu)
    return -1

# Problem 87
# Review- try to make faster
def countPrimePowerNTuples(
    mx_sum: int=5 * 10 ** 7 - 1,
    mn_pow: int=2,
    mx_pow: int=4,
) -> int:
    """
    Solution to Project Euler #87

    Finds the number of integers not exceeding mx_sum which can be
    expressed as a sum of prime powers as follows:
        (sum from k=mn_pow to mx_pow) p_k ** k
    where for integers mn_pow <= k <= mx_pow p_k is a prime number.
    
    Args:
        Optional named:
        mx_sum (int): The largest integer considered
            Default: 5 * 10 ** 7 - 1
        mn_pow (int): The smallest prime power exponent in the sum
            Default: 2
        mx_pow (int): The largest prime power exponent in the sum
            Default: 4
    
    Returns:
    Integer (int) giving the number of integers no greater than mx_sum
    that can be expressed as the sum of prime powers in the manner
    described.
    """
    #since = time.time()
    m = mx_pow
    ps = PrimeSPFsieve()
    curr = {0}
    for m in reversed(range(mn_pow, mx_pow + 1)):
        prev = curr
        curr = set()
        for num in prev:
            for p in ps.endlessPrimeGenerator():
                num2 = num + p ** m
                if num2 > mx_sum: break
                curr.add(num2)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return len(curr)

# Problem 88
def factorisationGenerator(
    num: int,
    proper: bool=False,
    ps: Optional[PrimeSPFsieve]=None,
) -> Generator[Tuple[int], None, None]:
    """
    Generator iterating over the possible positive factorisations
    of a given positive integer such that each factor is greater
    than 1 (including or excluding the trivial factorisation as
    specified) with the factors in each factorisation ordered in
    decreasing size
    
    Args:
        Required positional:
        num (int): The positive integer to be factorised
        
        Optional named:
        proper (bool): If False, includes the trivial factorisation
                (i.e. (num,)), otherwise excludes this
            Default: False
        ps (PrimeSPFsieve or None): If specified, a prime sieve that
                aids in the calculation of the factorisations. If
                not given or given as None, creates its own prime
                sieve.
                Giving the option to provide an externally defined
                prime sieve may speed up the factorisations by
                removing the need to repeat the same calculations
                multiple times.
            Default: None
    
    Yields:
    Tuple of ints with each of the factors in a factorisation of
    num (so that the product of the integers in the tuple equals
    num) with the ints in decreasing order. All possible factorisations
    where the factors are strictly greater than 1 (and if proper
    given as True, strictly less than num) are yielded by the
    generator exactly once (unless the generator iteration is exited
    early).
    """
    # Not currently used- superceded by factorisationsGenerator method
    # of PrimeSPFsieve
    if ps is None: ps = PrimeSPFsieve(num)
    p_fact = ps.primeFactorisation(num)
    p_lst = sorted(p_fact.keys())
    exp_lst = [p_fact[p_lst[i]] for i in range(len(p_lst))]
    fact_lst = []
    nonzero = SortedList(range(len(p_lst)))
    def recur(i: int, curr: int=1) -> Generator[Tuple[int], None, None]:
        added = False
        if i == len(nonzero):
            #if nonzero and curr < nonzero[-1]:
            #    return
            fact_lst.append(curr)
            i = 0
            curr = 1
            added = True
        if not nonzero:
            if not proper or len(fact_lst) > 1:
                yield tuple(fact_lst)
            fact_lst.pop()
            return
        prev = curr
        idx = nonzero[i]
        exp = exp_lst[idx]
        start = int(curr == 1 and i == len(nonzero) - 1)
        curr *= (p_lst[idx] if start == 1 else 1)
        for j in range(start, exp_lst[idx]):
            if fact_lst and curr > fact_lst[-1]:
                break
            exp_lst[idx] = exp - j
            yield from recur(i + 1, curr)
            curr *= p_lst[idx]
        if not fact_lst or curr <= fact_lst[-1]:
            nonzero.pop(i)
            exp_lst[idx] = 0
            yield from recur(i, curr)
            nonzero.add(idx)
        curr = fact_lst.pop() if added else prev
        exp_lst[idx] = exp
        return
    yield from recur(0)
    return

def productSumNumbers(k_mn: int=2, k_mx: int=12000) -> int:
    """
    Solution to Project Euler #88

    Consider the set of all strictly positive integers such that there
    exists a multiset of between k_mn and k_mx (inclusive) strictly
    positive integers whose sum and product equals that integer.
    This function returns the sum of all elements in this set.
    
    Args:
        Optional named:
        k_mn (int): The smallest size of integer multisets considered
            Default: 2
        k_mx (int): The largest size of integer multisets considered
            Default: 12000
    
    Returns:
    The sum of all integers in the set defined above.
    
    Outline of rationale:
    TODO
    """
    #since = time.time()
    k_mn = max(k_mn, 1)
    seen_k = {1} if k_mn < 2 else set()
    res = {1} if k_mn < 2 else set()
    target_cnt = k_mx - k_mn + 1
    p_mx = k_mx * 2
    ps = PrimeSPFsieve(p_mx)
    factorisations = [[], []]
    for num in range(2, p_mx + 1):
        factors = ps.factors(num)
        factors -= {1, num}
        factors = sorted(factors)
        fact_lst = []
        for f in factors:
            num2 = num // f
            for fact2 in factorisations[num2]:
                if fact2[0][-1] > f: break
                fact = ([*fact2[0], f], fact2[1] + f)
                fact_lst.append(fact)
                k = len(fact[0]) + num - fact[1]
                if not (k_mn <= k <= k_mx) or k in seen_k:
                    continue
                seen_k.add(k)
                res.add(num)
        if len(seen_k) == target_cnt:
            #print(res)
            #print(factorisations)
            res2 = sum(res)
            break
        fact_lst.append(([num], num))
        factorisations.append(fact_lst)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res2
    """        
    since = time.time()
    seen_k = set()
    res = set()
    target_cnt = k_mx - k_mn + 1
    increment = k_mx * 2
    ps = PrimeSPFsieve()
    p_mx = 2
    factorisations = [[], []]
    while True:
        p_mx2 = p_mx + increment
        ps.extendSieve(p_mx2)
        for num in range(p_mx + 1, p_mx2 + 1):
            
            for fact in ps.factorisationsGenerator(num, proper=True):
                k = len(fact) + num - sum(fact)
                if not (k_mn <= k <= k_mx) or k in seen_k:
                    continue
                seen_k.add(k)
                res.add(num)
                if len(seen_k) == target_cnt:
                    print(f"Time taken = {time.time() - since:.4f} seconds")
                    return sum(res)
    """

# Problem 89- this is overkill. Try looking directly at patterns in
#             the numerals that can be shortened
def loadRoman(
    doc: str,
    rel_package_src: bool=False,
) -> List[str]:
    """
    Loads a sequence of numbers expressed as Roman numerals from the
    .txt file at relative or absolute location doc, where the different
    numbers are strings separated by line breaks ('\\n')
    
    Args:
        Required positional:
        doc (str): The relative or absolute location of the .txt file
                containing the sequence of numbers expressed as
                Roman numerals

        Optional named:
        rel_package_src (bool): Whether a relative path given by doc
                is relative to the current directory (False) or
                the package src directory (True).
            Default: True
    
    Returns:
    List of strings (str) with the numbers expressed as Roman numerals
    in the .txt file at location doc in the order they appear in that
    file.
    """
    txt = loadTextFromFile(doc, rel_package_src=rel_package_src)
    return [x.strip() for x in txt.split("\n")]

def romanToInt(numeral: str) -> int:
    """
    Finds the value of an integer from its expression as a Roman
    numeral.
    
    Args:
        Required positional:
        numeral (str): String representing the Roman numeral.
    
    Returns:
    Integer (int) giving the value of the Roman numeral.
    """
    pre_pairs = {"I": {"V", "X"}, "X": {"L", "C"}, "C": {"D", "M"}}
    l_vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100,\
            "D": 500, "M": 1000}
    res = 0
    n = len(numeral)
    for i, l in enumerate(numeral):
        if i + 1 < n and numeral[i + 1] in pre_pairs.get(l, set()):
            res -= l_vals[l]
        else: res += l_vals[l]
    return res

def intToRoman(num: int) -> str:
    """
    Finds a Roman numeral expression of the strictly positive integer
    num such that there is no shorter Roman numeral expression with
    that value.
    
    Args:
        Required positional:
        num (int): The strictly positive integer to be expressed as a
                Roman numeral
    
    Returns:
    String (str) giving a Roman numeral expression whose value is equal
    to num such that there is no shorter Roman numeral expression with
    that value.
    """
    numeral_vals = (("I", "V"), ("X", "L"), ("C", "D"), ("M",))

    i = 0
    res = []
    for i in range(len(numeral_vals) - (len(numeral_vals[-1]) == 1)):
        if not num: break
        num, r = divmod(num, 10)
        #print(num, r)
        if r == 9: res.extend([numeral_vals[i + 1][0], numeral_vals[i][0]])
        elif r == 4: res.extend([numeral_vals[i][1], numeral_vals[i][0]])
        elif r >= 5:
            res.extend([numeral_vals[i][0]] * (r - 5))
            res.append(numeral_vals[i][1])
        else: res.extend([numeral_vals[i][0]] * r)
    if num:
        if len(numeral_vals[-1]) == 1:
            res.extend([numeral_vals[-1][0]] * num)
        else:
            res.extend([numeral_vals[-1][1]] * (num * 2))
    return "".join(res[::-1])

def romanNumeralsSimplificationScoreFromFile(
    doc: str="project_euler_problem_data_files/0089_roman.txt",
    rel_package_src: bool=True,
) -> int:
    """
    Solution to Project Euler #89
    For a sequence of Roman numeral expressions in the .txt file doc,
    finds the total difference between the number of characters used
    in this sequence and the least possible number of characters with
    which the same sequence of corresponding numbers could have been
    expressed in Roman numerals.
    
    Args:
        Optional named:
        doc (str): The .txt file containing the sequence of Roman
                numeral expressions, with each expression separated
                by a line break ('\\n')
            Default: "project_euler_problem_data_files/0089_roman.txt"
        rel_package_src (bool): Whether a relative path given by doc
                is relative to the current directory (False) or
                the package src directory (True).
            Default: True
    
    Returns:
    Integer (int) giving the total number of characters used to express
    the Roman numerals in doc minus the smallest number of characters
    that could have been used to express the same sequence of values
    as Roman numerals.
    """
    #since = time.time()
    numerals = loadRoman(doc, rel_package_src=rel_package_src)
    res = 0
    for s in numerals:
        num = romanToInt(s)
        s2 = intToRoman(num)
        #print(s, num, s2)
        res += len(s) - len(s2)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 90
# Review- Needs tidying and/or simplifying
def polyhedraNumberedFacesFormingSquareNumbersCount(
    n_polyhedra: int=2,
    polyhedron_n_faces: int=6,
    base: int=10,
    interchangeable: Tuple[Set[int]]=({6, 9},),
) -> int:
    """
    Solution to Project Euler #90

    Consider n_polyhedra polyhedra, each with polyhedron_n_faces
    faces that are numbered with digits in the chosen base. This
    function calculates the number of ways such polyhedra can
    be labelled such that the representations of all the strictly
    positive square numbers in the chosen base with at most
    n_polyhedra digits (potentially with leading zeros) can be
    constructed by selecting the digit on exactly one face from
    each of the polyhedra, where:
     - Two polyhedra with the same counts of each digit over all
       of their faces are considered identical (so permutation of
       the digits between the faces does not result in a different
       polyhedron)
     - Digits in the same set in the collection of sets interchangeable
       can replace each other (for example, in base 10, 6 and 9
       can be used in place of one another by being turned upside
       down). However, if two polyhedra are labelled such that the
       only faces that differ after pairing all equal faces have
       different but interchangeable digits, the labellings are still
       considered to be different.
     - If two labellings such that each polyhedron in one labelling
       can be paired to a different polyhedron in the other labelling
       in such a way that every pair of polyhedra are the same, then
       those two labellings are the same (so permutation of the
       polyhedra does not result in a different labelling).

    Args:
        Optional named:
        n_polyhedra (int): The number of polyhedra used, equal to
                the largest number of digits allowed in the
                representation in the chosen base of the square
                numbers that must be able to be formed by any
                counted labelling.
            Default: 2
        polyhedron_n_faces (int): The number of faces of each
                polyhedron (for the platonic solids, this would
                be 4 for a tetrahedron, 6 for a cube, 8 for an
                octohedron, 12 for a dodecohedron and 20 for an
                icosohedron).
            Default: 6
        base (int): Integer no less than 2 giving the base in which
                the square numbers are to be represented (potentially
                with leading zeros).
            Default: 10
        interchangeable (tuple of sets of ints): Sets of digit values
                in the chosen base, whose representations are considered
                to be interchangeable (for instance, in arabic numerals
                6 and 9 can be considered interchangeable as their
                forms can be mapped onto each other by being turned
                upside down)
            Default: ({6, 9},)
        
    Returns:
    Integer (int) giving teh number of distinct possible labellings
    of the polyhedra such that the representation of all squares with
    no more than n_polyhedra digits in the chosen base can be formed
    (possibly with leading zeros) by selecting the digit (or one of
    its interchangeable counterparts) from exactly one face from each
    of the polyhedra.

    Outline of rationale:
    TODO
    """
    #since = time.time()
    interchange_dict = {}
    multiplicity = {x: 1 for x in range(base)}
    for nums in interchangeable:
        nums_lst = list(nums)
        for i in range(1, len(nums_lst)):
            interchange_dict[nums_lst[i]] = nums_lst[0]
            multiplicity.pop(nums_lst[i])
        multiplicity[nums_lst[0]] = len(nums)
    #print(multiplicity)
    targets = []
    rt = isqrt(base ** n_polyhedra)
    for i in range(1, rt):
        num = i ** 2
        targets.append({})
        for i in range(n_polyhedra):
            num, r = divmod(num, base)
            r2 = interchange_dict.get(r, r)
            targets[-1][r2] = targets[-1].get(r2, 0) + 1
        else: continue
        #targets[-1][0] = targets[-1].get(0, 0) + n_cubes - i
    targets = {tuple(sorted([(k, v) for k, v in x.items()])) for x in targets}
    #print(targets)

    # Reducing the number of options by ensuring that every cube has at least
    # one digit of every target number (which is a requirement)
    def possiblePolyhedra(
        targets: Set[Tuple[Tuple]],
        multiplicity: Dict[int, int],
        n_faces: int=6,
    ) -> List[Set[int]]:
        def addExtras(nums_lst: List[int], extra_lst: List[Tuple[int]])\
                -> Generator[List[Tuple[int]], None, None]:
            cumu_lst = [0] * (len(extra_lst) + 1)
            curr = 0
            for i in reversed(range(len(extra_lst))):
                curr += extra_lst[i][1]
                cumu_lst[i] = curr
            nums_dict = {x: 1 for x in nums_lst}
            def recur(remain: int, i: int) -> Generator[List[Tuple[int]], None, None]:
                #print(remain, len(nums_lst))
                if not remain:
                    yield sorted([(x, y) for x, y in nums_dict.items()])
                    return
                num = extra_lst[i][0]
                f_orig = nums_dict.get(num, 0)
                start = max(0, remain - cumu_lst[i + 1])
                end = min(remain, extra_lst[i][1])
                if end < start:
                    return
                if start: nums_dict[num] = nums_dict.get(num, 0) + start
                for j in range(start, end):
                    yield from recur(remain - j, i + 1)
                    nums_dict[num] = nums_dict.get(num, 0) + 1
                yield from recur(remain - end, i + 1)
                if num in nums_dict.keys():
                    if not f_orig: nums_dict.pop(num)
                    else: nums_dict[num] = f_orig
                return
            yield from recur(n_faces - len(nums_dict), 0)
            return

        def coreGenerator(mx_sz: int=n_faces) -> Generator[Tuple[int], None, None]:
            targets2 = [{y[0] for y in x} for x in targets]
            n_target = len(targets2)
            
            curr = set()
            def recur(i: int) -> Generator[Tuple[int], None, None]:
                if len(curr) == n_faces or i == n_target:
                    for j in range(i, n_target):
                        if curr.isdisjoint(targets2[i]):
                            break
                    else: yield tuple(sorted(curr))
                    return
                new_nums = targets2[i].difference(curr)
                if len(new_nums) < len(targets2[i]):
                    # i.e there is a non-zero intersection between
                    # targets2[i] and curr
                    yield from recur(i + 1)
                for num in new_nums:
                    curr.add(num)
                    yield from recur(i + 1)
                    curr.remove(num)
                return
            yield from recur(0)
            return

        res = set()
        for nums_tup in coreGenerator(mx_sz=n_faces):
            if len(nums_tup) == n_faces:
                res.add(tuple((x, 1) for x in nums_tup))
                continue
            extra_lst = []
            nums_set = set(nums_tup)
            for num in sorted(multiplicity.keys()):
                cnt = multiplicity[num] - (num in nums_set)
                if not cnt: continue
                extra_lst.append((num, cnt))
            for nums in addExtras(nums_tup, extra_lst):
                res.add(tuple(nums))
        return res

    cube_opts = list(possiblePolyhedra(targets, multiplicity, n_faces=polyhedron_n_faces))
    
    counts = [1] * len(cube_opts)
    for i, tup in enumerate(cube_opts):
        cnt = 1
        for num, f in tup:
            cnt *= math.comb(multiplicity[num], f)
        counts[i] = cnt
    cube_opts2 = [{y[0] for y in x} for x in cube_opts]
    n_opts = len(cube_opts2)
    
    def backtrack(cubes: List[Tuple[Tuple[int]]], target: Dict[int, int]) -> bool:
        inds = SortedList(range(len(cubes)))
        
        target_lst = []
        for num, f in target:
            target_lst.extend([num] * f)
        def recur(i: int) -> bool:
            if not inds: return True
            for j in list(inds):
                if target_lst[i] not in cubes[j]: continue
                inds.remove(j)
                if recur(i + 1): return True
                inds.add(j)
            return False
        return recur(0)
    
    res = 0
    for idx in itertools.combinations_with_replacement(range(n_opts), n_polyhedra):
            #combinationsWithReplacement(n_opts, n_cubes):
        cubes = [cube_opts2[i] for i in idx]
        for target in targets:
            if not backtrack(cubes, target):
                break
        else:
            mult = 1
            for i in idx:
                mult *= counts[i]
            res += mult
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
        

# Problem 91
def countRightTrianglesWithIntegerCoordinates(
    x_mx: int=50,
    y_mx: int=50,
) -> int:
    """
    Solution to Project Euler #91
    Identifies the number of distinct right angled triangles that can
    be drawn on a Cartesian plane such that one of its vertices is at
    the origin (i.e. (0, 0)) and its other vertices are at positions
    with integer coordinates such that the x coordinate is between 0
    and x_mx inclusive and the y coordinate is between 0 and y_mx. Two
    triangles are considered to be the same if and only if the set of
    positions of the vertices of both triangles are equal.
    
    Args:
        Optional named:
        x_mx (int): The largest allowed value of the x-coordinate for
                the vertices of the triangles
            Default: 50
        y_mx (int): The largest allowed value of the y-coordinate for
                the vertices of the triangles
            Default: 50
    
    Returns:
    Integer (int) giving the number of distinct right angled triangles
    that can be drawn subject to the given constraints.
    
    Outline of rationale:
    TODO
    """
    #since = time.time()
    res = 3 * x_mx * y_mx
    for x in range(1, x_mx + 1):
        for y in range(1, y_mx + 1):
            g = gcd(x, y)
            vec = (y // g, -x // g)
            res += min((x_mx - x) // vec[0], y // (-vec[1]))
            res += min(x // vec[0], (y_mx - y) // (-vec[1]))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 92
# Review- Try to make faster
def squareDigitSum(num: int, base: int=10) -> int:
    """
    Finds the sum of the squares of the digits of non-negative
    integer num when expressed in the chosen base.
    
    Args:
        Required positional:
        num (int): The non-negative integer for which the sum
                of squares of its digits when expressed in the
                chosen base is to be calculated
        
        Optional named:
        base (int): The base in which num is to be expressed
            Default: 10
    
    Returns:
    Integer giving the sum of the squares of the digits of the
    representation of num in the chosen base.
    """
    res = 0
    while num:
        num, r = divmod(num, base)
        res += r ** 2
    return res

def squareDigitChains(num_mx: int=10 ** 7 - 1) -> int:
    """
    Solution to Project Euler #92

    Consider the set of infinite integer sequences  where the first
    term is a strictly positive integer and every term after the first
    is the sum of the squares of the digits of the previous term when
    expressed in base 10.
    It can be shown that every sequence in this set contains either the
    number 1 or the number 89 will appear in the sequence.
    This function returns the number of strictly positive integers
    not exceeding num_mx for which the sequence in this set whose
    first element equals that integer contains 89.
    
    Args:
        Optional named:
        num_mx (int): The largest integer considered for the first
                term in the sequence
    
    Returns:
    Integer (int) giving the number of strictly positive integers
    not exceeding num_mx for which the sequence in the descibed set
    whose first element equals that integer contains 89.
    """
    # Note that in base 10 the digit square sum of any number is
    # smaller than that number for any number with at least 3
    # digits. Therefore, for integer n:
    #     squareDigitSum(num, base=10) <= max(num, 2 * 9 ** 2)
    #since = time.time()
    memo = [None for _ in range(max(num_mx, 2 * 9 ** 2)  + 1)]
    memo[1] = False
    memo[89] = True
    res = 0
    for num in range(1, num_mx + 1):
        stk = [num]
        while memo[stk[-1]] is None:
            stk.append(squareDigitSum(stk[-1], base=10))
        b = memo[stk.pop()]
        while stk: memo[stk.pop()] = b
        res += memo[num]
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res 
            
# Problem 93
def reachableIntegerRanges(nums: Set[int]) -> Tuple[Tuple[int]]:
    """
    For a set of distinct integers nums, finds the ranges of integers
    it are expressible as an arithmetic expression involving each
    of the integers in nums exactly once, the addition, subtraction
    and multiplication operations and brackets.
    Note that concatenation of the integers in nums is not permitted
    
    Args:
        Required positional:
        nums (set of ints): The set of distinct integers, each of which
                must appear exactly once in any valid arithmetic
                expression.
    
    Returns:
    A tuple of 2-tuples, where each 2-tuple is a range (a, b)
    (where a and b are integers and a <= b) signifying that all
    integers between a and b are expressible in the manner
    described. The 2-tuples are non-overlapping, are sorted
    in increasing value of the first element (i.e. the
    beginning of the range) and collectively cover exactly those
    integers that are expressible as an arithmetic expression
    as described.
    """
    res = SortedDict()
    for tup in itertools.permutations(nums):
        curr = {(tup[0], 1)}
        for i in range(1, len(tup)):
            prev = curr
            curr = set()
            for frac in prev:
                curr.add((frac[0] + tup[i] * frac[1], frac[1]))
                curr.add((frac[0] - tup[i] * frac[1], frac[1]))
                curr.add((tup[i] * frac[1] - frac[0], frac[1]))
                g1 = max(gcd(abs(tup[i]), abs(frac[1])), 1)
                curr.add((frac[0] * (tup[i] // g1), frac[1] // g1))
                g2 = max(gcd(abs(tup[i]), abs(frac[0])), 1)
                v1, v2 = frac[0] // g2, frac[1] * (tup[i] // g2)
                if v2 > 0: curr.add((v1, v2))
                elif v2 < 0: curr.add((-v1, -v2))
                if v1 > 0: curr.add((v2, v1))
                elif v1 < 0: curr.add((-v2, -v1))
        for frac in curr:
            if frac[1] != 1: continue
            #print(frac)
            i = res.bisect_right(frac[0]) - 1
            inserted = False
            if i >= 0:
                rng = res.peekitem(i)
                if frac[0] <= rng[1]:
                    continue
                elif frac[0] == rng[1] + 1:
                    res[rng[0]] += 1
                    inserted = True
            if not inserted:
                res[frac[0]] = frac[0]
                i += 1
                rng = (frac[0], frac[0])
            if i + 1 == len(res): continue
            rng2 = res.peekitem(i + 1)
            if rng2[0] != frac[0] + 1: continue
            res[rng[0]] = res.popitem(i + 1)[1]
    return tuple((x, y) for x, y in res.items())
    
def arithmeticExpressions(
    n_num: int=4,
    num_mn: int=1,
    num_mx: int=9,
) -> str:
    """
    Solution to Project Euler #93
    
    Finds a set of n_num distinct integers between num_mn and num_mx
    inclusive such that for an integer m, every integer between 1 and
    m inclusive can be expressed as an arithmetic expression involving
    each of the integers in the exactly once, the addition, subtraction
    and multiplication operations and brackets, and there are no sets
    of n_num distinct integers between num_mn and num_mx inclusive
    such that verey integer between 1 and m + 1 is similarly expressible.
    This set is returned as a string concatenating the elements of the
    set (expressed in base 10) in order of increasing size.
    Note that concatenation of the integers in nums is not permitted.
    
    Args:
        Optional named:
        n_num (int): The number of distinct integers in the sets
                considered.
            Default: 4
        num_mn (int): The smallest integer permitted in the sets
                considered.
            Default: 1
        num_mx (int): The largest integer permitted in the sets
                considered.
            Default: 9
    
    Returns:
    String (str) consisting of the expression in base 10 of the elements
    of the set of integers found concatenated in order of increasing
    size.
    """
    #since = time.time()
    if not n_num: return ""
    curr = [0] * n_num
    def recur(i: int=0, mn: int=num_mn) -> Generator[int, None, None]:
        if i == n_num:
            yield tuple(curr)
            return
        for num in range(mn, num_mx - (n_num - i) + 2):
            curr[i] = num
            yield from recur(i=i + 1, mn=num + 1)
        return
    best = (-1, ())
    for nums in recur():
        rngs = reachableIntegerRanges(nums)
        i = bisect.bisect_right(rngs, (1, float("inf"))) - 1
        if i < 0: continue
        best = max(best, (rngs[i][1], nums))
    if not best[1]: return ""
    #print(best)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return "".join([str(x) for x in best[1]])

# Problem 94
def almostEquilateralTriangles(
    perimeter_max: int=10 ** 9,
) -> int:
    """
    Solution to Project Euler #94

    An almost equilateral triangle is a triangle for which
    two sides are the same length and the third differs in
    length from the other two by no more than one unit.
    This function finds the sum of perimeters of all almost
    equlateral triangles with integer side lengths and integer
    areas with perimeter no greater than perimeter_max.
    
    Args:
        Optional named:
        perimeter_max (int): Non-negative integer giving the
                largest perimeter of a triangle that may be
                included in the sum.
            Default: 10 ** 9
    
    Returns:
    Integer (int) giving the sum of perimeters of all almost
    equlateral triangles with integer side lengths and integer
    areas with perimeter no greater than perimeter_max.
    
    Outline of proof and rationale of approach:
    Given that the height of an equalateral triangle is
    sqrt(3) / 2 times the side length, and sqrt(3) / 2 is
    irrational, there do not exist equalateral triangles with
    integer solutions, we are only looking for triangles with
    integer area such that two of its sides are equal integer
    length and the remaining side has length exactly one less
    or one greater than the length of the other two.
    Consider a triangle that satisfies these properties. Construct
    a new triangle whose vertices are the vertex shared by
    the two sides of equal length, the midpoint of the other
    side and one of the other vertices of the original triangle.
    This is a right angled triangle (with the right angle at
    the second of these vertices) whose hypotenuse is the length
    of the two equal length sides of the original triangle and
    whose area exactly half of that of the original triangle.
    Therefore, twice the area of this triangle must be an integer.
    Suppose the length of the unique side is odd, and consider
    a triangle which is similar to the constructed triangle and
    each of its sides double the length of the corresponding one
    in the constructed triangle. In this triangle, the area must
    be even (as its area is four times that of the constructed
    triangle), the hypotenuse must also be even and the side
    originating from half of the unique side of the original
    triangle is now equal in length to the length of that unique
    side and so must be (by our supposition) odd. Since the area
    must be even, the product of the shorter two sides must be
    a multiple of four (given that the area is half the product
    of these two sides), so the other shorter side must be a
    multiple of four. But by Pythagoras' theorem, the sum of
    the squares of the two shorter sides must equal the square
    of the hypotenuse. The sum of the squares of the two shorter
    sides is the sum of the squares of an even and an odd number,
    which is odd, while the square of the hypotenuse is the
    square of an even number, which is even. This is a
    contradiction, and therefore the length of the unique side
    must be even.
    Returning to the constructed triangle, we have now established
    that the hypotenuse and the side originating from the unique
    side of the original triangle (which is half the length of
    this unique side) are both integers. It therefore follows
    by Pythagoras' theorem that the square of the remaining side
    is also an integer. Furthermore, since the product of the
    two shorter sides is an integer and the other shorter side
    is an integer, it follows that this side must also be rational.
    Therefore, since the square root of an integer is either
    an integer or irrational, the remaining side is also an
    integer.
    Consequently, for any triangle with integer area and two of
    its sides equal integer length and the remaining side has
    length exactly one less or one greater than the length of the
    other two, the triangle constructed from this as described
    above is a right angled triangle with integer side lengths,
    i.e. a Pythagorean triple. For this Pythagorean triple,
    twice one of the shorter sides must be equal to one more
    (if the unique side or the original triangle is shorter
    than the other two) or one less (otherwise) than the
    hypotenuse. It follows that this construction maps every
    almost equalateral triangle to a Pythagorean triple for
    which twice one of the shorter sides is one more or one
    less than the length of the hypotenuse.
    
    We now try to reverse this mapping. We consider a given
    Pythagorean triple (a, b, c) where a < b < c and (so c
    corresponds to the hypotenuse and a ** 2 + b ** 2 = c ** 2,
    and recalling that there does not exist a Pythagorean
    triple such that two of the triple are equal),
    such that either 2 * a = c + 1, 2 * a = c - 1,
    2 * b = c + 1 or 2 * b = c - 1. First suppose
    2 * b = c +/- 1. Then:
        a ** 2 + b ** 2 = 4 * b ** 2 +/- 2 * b + 1
    so a ** 2 = 3 * b ** 2 +/- 2 * b + 1
    Given that b >= 1, b ** 2 >= b which implies that:
        a ** 2 >= b ** 2 + 1
    which, given that both a, b >= 1 implies that a > b, which
    is a contradiction. Therefore, there are no Pythagorean
    triples (a, b, c) where a < b < c and 2 * b = c +/- 1.
    We are therefore left with the Pythagorean triples for
    which 2 * a = c + 1 and 2 * a = c - 1. For 2 * a = c + 1,
    by reflecting the corresponding right angled triangle along
    the side corresponding to b and attaching the unreflected
    and reflected triangles two together, we get a triangle
    with sides c, c and 2 * a = c + 1 (which are all integers)
    and area 2 * (a * b / 2) = a * b, which is also an integer.
    This therefore corresponds to an almost equalateral
    triangle with integer area.
    Similarly, if 2 * a = c - 1 we can similarly construct
    a triangle with sides c, c, c - 1 (noting that for every
    Pythagorean triple c > 1 so c + 1 is strictly positive).
    Given that the triangle with sides c, c, c + 1 is certainly
    distinct from the triangle with sides c, c, c - 1, this
    gives a correspondence between every Pythagorean triple and
    to exactly two almost equalateral triangles with integer
    area. Furthermore, given no two distinct Pythagorean triples
    (a, b, c) where a < b < c can have the same value for both
    a and c (as known a and c together with
    a ** 2 + b ** 2 = c ** 2 fixes b if b is required to be
    positive), this correspondence does not link two distinct
    Pythagorean triples with the same almost equalateral triangles
    with integer area.
    Thus, we can find all almost equalateral triangles with
    integer area satisfying a given set of constraints by finding
    all Pythagorean triples (a, b, c) such that 2 * a = c + 1
    satisfying corresponding constraints and for each giving the
    triangle (c, c, c + 1), and finding all Pythagorean triples
    such that 2 * a = c - 1 satisfying corresponding constraints
    and giving the triangle (c, c, c + 1).
    
    We now attempt to find an efficient method for generating
    the Pythagorean triples such that 2 * a = c - 1. By Euclid's
    formula, the set of triplets of strictly positive integers
    (k, m, n) such that n < m and m and n are coprime (i.e.
    gcd(m, n) = 1) can be mapped bijectively onto every Pythagorean
    triple, where this mapping giving the triple (a, b, c) where:
        a' = k * (m ** 2 - n ** 2), b' = k * (2 * m * n),
        c = k * (m ** 2 + n ** 2)
    and a = min(a', b') while b = max(a', b'). We
    consider the Pythagorean triples satisfying the described
    conditions, i.e. those for which 2 * a = c +/- 1.
    Suppose k > 1. Then c is even, and so both c + 1 and c - 1
    are odd. However, given that a' and b' are integers, 2 * a =
    min(2 * a', 2 * b') must be even, which is a contradicts
    both the condition 2 * a = c + 1 and 2 * a = c - 1. Therefore,
    any Pythagorean triple for which 2 * a = c +/- 1 must have
    k = 1. We now divide the possibilities into four cases
    (the combinations of a' < b' or a' > b' and 2 * a = c + 1
    or 2 * a = c - 1) and examine each case in turn.
    
    First suppose a' < b' and 2 * a = c + 1. Given that k = 1 and
    so the possible values of a and c are:
        a = a' = m ** 2 - n ** 2 and c = m ** 2 + n ** 2
    for strictly positive coprime integers m, n for which m > n,
    we find:
        2 * (m ** 2 - n ** 2) = m ** 2 + n ** 2 + 1
    which can be rearranged to get:
        m ** 2 - 3 * n ** 2 = 1
    This is Pell's equation for D = 3. In this equation, we note
    that if m, n are not coprime, and so share some common factor
    g > 1 (g an integer), then the left hand side is dividible by
    g but the right hand side is not, which is a contradiction.
    Therefore, any integer solution to this equation has m, n
    coprime. Furthermore, if m and n are strictly positive integers,
    then m ** 2 = 3 * n ** 2 + 1 > n ** 2, implying that m > n.
    Additionally:
        b - a = b' - a' = n ** 2 + 2 * m * n - m ** 2
              = n ** 2 + 2 * m * n - 3 * n ** 2 - 1
              = 2 * n * (m - n) - 1
              >= 2 * n - 1 >= 1 > 0
    Thus, for any solution (m, n) to Pell's equation for D = 3 (with
    strictly positive integers m, n), m and n are coprime, m > n and
    m ** 2 - n ** 2 < 2 * m * n. Consequently, for positive integer
    c, the triangle with sides (c, c, c + 1) has integer area (or
    equivalently, is an almost equilateral triangle with integer
    area) if and only if there exist strictly positive integers
    m, n such that c = m ** 2 + n ** 2  and m ** 2 - 3 * n ** 2 = 1
    (i.e. (m, n) is a solution of Pell's equation for D = 3).
    We can also confirm that every distinct solution of Pell's equation
    (m, n) for strictly positive integer m, n gives a distinct value
    of c = m ** 2 + n ** 2 (and so corresponds to a distinct
    almost equilateral triangle with integer area). Suppose there
    exist two pairs (m, n), (m', n') where m, n, m', n' are strictly
    positive integers, m ** 2 - 3 * n ** 2 = 1, m' ** 2 - 3 * n' ** 2 = 1
    and m ** 2 + n ** 2 = m' ** 2 + n' ** 2 = c. Then we can substitute
    m ** 2 = c - n ** 2 and m' ** 2 = c - n' ** 2 into their respective
    Pell's equation to get:
        4 * n ** 2 = c - 1 = 4 * n' ** 2
    This implies that n ** 2 = n' ** 2 and since n and n' are non-negative
    that n = n'. Thus:
        m ** 2 = 3 * n ** 2 + 1 = 3 * n' ** 2 + 1 = m' ** 2
    which similarly (given that m and m' are non-negative) implies that
    m = m'. Thus, for any two distinct solutions to Pell's equation for D = 3
    with strictly positive integers (m, n) and (m, n'):
        m ** 2 + n ** 2 != m' ** 2 + n' ** 2
    We are therefore not at risk of double-counting solutions.
    
    Now suppose a' < b' and 2 * a = c - 1. Given that k = 1 and
    so the possible values of a and c are:
        a = a' = m ** 2 - n ** 2 and c = m ** 2 + n ** 2
    for strictly positive coprime integers m, n for which m > n,
    we find:
        2 * (m ** 2 - n ** 2) = m ** 2 + n ** 2 - 1
    which can be rearranged to get:
        m ** 2 - 3 * n ** 2 = -1
    This is Pell's negative equation for D = 3, for which there is
    no solution for strictly positive m and n. This can be seen
    by considering the equation modulo 4. Any square of an integer
    is either 0 or 1 modulo 4. Thus, both m ** 2 and n ** 2 can either
    be 0 or 1 modulo 4, implying their sum cannot be 3 modulo 4 as
    the right hand side is. Thus, there are no solutions for a' < b'
    and 2 * a = c - 1.
    We similarly suppose a' > b' and 2 * a = c + 1. Given that k = 1
    and so the possible values of a and c are:
        a = b' = 2 * m * n and c = m ** 2 + n ** 2
    for strictly positive coprime integers m, n for which m > n we
    find:
        4 * m * n = m ** 2 + n ** 2  + 1
    Taking this equation modulo 4 similarly to previously, we find
    that the left hand side is 0 modulo 4 and the right hand side
    is one of 1, 2, 3 modulo 4. Therefore, as before there are
    no solutions for a' > b' and 2 * a = c + 1.
    
    Finally suppose a' > b' and 2 * a = c - 1. Given that k = 1 and
    so the possible values of a and c are:
        a = b' = 2 * m * n and c = m ** 2 + n ** 2
    for strictly positive coprime integers m, n for which m > n,
    we find:
        4 * m * n = m ** 2 + n ** 2 - 1
    which rearranged gives:
        (m - 2 * n) ** 2 - 3 * n ** 2 = 1
    We make the substitution m' = m - 2 * n, which gives:
        m' ** 2 - 3 * n ** 2 = 1
    This is Pell's equation for D = 3 again, though in this case
    m' > -2 * n rather than being strictly positive. However,
    if we suppose -n <= m' <= 0 then (since n > 0)
    m' ** 2 <= n ** 2 so:
        n ** 2 - 3 * n ** 2 > 1
    implying that 2 * n ** 2 < 1, which given that n is a
    strictly positive integer is a contradiction. Therefore, if
    m' <= 0 then m' < -n, and so m = m' + 2 * n < n, which
    contradicts the requirement that m > n. Therefore, there are
    no solutions to m' ** 2 - 3 * n ** 2 = 1 for which
    -2 * n < m' <= 0 that give rise to an almost equalateral
    triangle with integer area, and thus in the case a' > b'
    and 2 * a = c - 1, any solution can be expressed as
    a = 2 * m * n and c = m ** 2 + n ** 2 where (m', n) are
    strictly positive integers such that m' ** 2 - 3 * n ** 2 = 1
    and m = m' + 2 * n.
    Similarly to previously, we wish to show the converse is
    also true, i.e. that if (m', n) is a solution to Pell's
    equation for D = 3 with strictly postivie integers m'
    and n, then for m = m' + 2 * n and c = m ** 2 + n ** 2
    then the triangle (c, c, c - 1) is an almost equilateral
    triangle with integer area. This is indeed the case if
    we can show for any solution to Pell's equation
    for D = 3 for strictly positive integers (m', n) and
    m = m' + 2 * n that m and n are coprime, m and n are
    strictly positive, m > n and a < b (where a = b' = 2 * m * n
    and b = a' = m ** 2 - n ** 2). From our analysis of the
    case a' < b' and 2 * a = c + 1, we know that m' and n are
    coprime, and since m = m' + 2 * n, this imples that m and
    n are coprime. Furthermore, since m' and n are strictly
    positive integers, so is m = m' + 2 * n. m = m' + 2 * n
    with m' and n strictly positive also implies that m > n.
    Finally:
        b - a = a' - b' = m ** 2 - 2 * m * n - n ** 2
              = (m' + 2 * n) ** 2 - 2 * (m' + 2 * n) * n - n ** 2
              = m' ** 2 + 2 * m' * n - n ** 2
              = 3 * n ** 2 + 1 + 2 * m' * n - n ** 2
              = 2 * n ** 2 + 2 * m' * n + 1 >= 5 > 0
    Thus, for any solution (m', n) to Pell's equation for D = 3 (with
    strictly positive integers m', n) and m = m' + n, m and n are
    strictly positive coprime integers, m > n and
    m ** 2 - n ** 2 < 2 * m * n. Consequently, for positive integer
    c, the triangle with sides (c, c, c - 1) has integer area (or
    equivalently, is an almost equilateral triangle with integer
    area) if and only if there exist strictly positive integers
    m', n such that c = (m' + 2 * n) ** 2 + n ** 2  and
    m' ** 2 - 3 * n ** 2 = 1 (i.e. (m', n) is a solution of Pell's
    equation for D = 3).
    We can show similarly to previously that if there exist two
    different pairs of strictly positive integers that satisfy
    Pell's equation that both correspond to the same value of
    c then they are the same, meaning that again double counting
    is not an issue.
    
    We therefore have two bijections, one which maps the set
    of solutions to Pell's equation for D = 3 to the set of almost
    equilateral triangles with integer side lengths and area where
    two sides are equal in length and the other is exactly one
    longer than the other two, which takes the form (where m, n
    are strictly positive integers such that m ** 2 - 3 * n ** 2 = 1:
        f(m, n) = triangle with side lengths c, c, c + 1 where
                  c = m ** 2 + n ** 2
    and one which maps the set of solutions to Pell's equation
    for D = 3 to the set of almost equilateral triangles with
    integer side lengths and area where two sides are equal in
    length and the other is exactly one shorter than the other two,
    which takes the form (where m', n are strictly positive integers
    such that m' ** 2 - 3 * n ** 2 = 1:
        f(m', n) = triangle with side lengths c, c, c - 1 where
                  c = (m' + 2 * n) ** 2 + n ** 2
    As we have already deduced, there is no equilateral triangle
    with integer edges and integer area, so all almost equilateral
    triangles with integer lengths and integer area must be in
    one of the two sets of triangles described above, and thus,
    given that the mappings from the solutions of Pell's equations
    for D = 3 are bijections and so surjections, the union of the
    image of the solutions of Pell's equation for D = 3 under the
    two mappings is precisely the set of all almost equilateral
    triangles with integer lengths and integer area. Furthermore,
    since the two sets of triangles are disjoint (as previously
    established) and the mappings are both bijections and so
    injections, each such almost equilateral triangle is the
    result of no more than one of the mappings being evaluated
    at no more than one solution to Pell's equation for D = 3.
    Thus, as we traverse through unique solutions of Pell's
    equation and evaluate both mappings for each, no single
    triangle is given as an output more than once by the two
    mappings combined, so there is no double counting.
    
    As such, we can find all almost equilateral triangles with
    perimeter not exceeding a given number by iterating over the
    solutions of Pell's equation for D = 3 in increasing size of
    m and n (noting that in Pell's equation, n increases as m
    increases and vice versa and in both mappings the larger the
    value of m and n, the larger the side length and so perimeter
    of the triangle mapped to) using pellSolutionGenerator(),
    evaluating both mappings to find the corresponding triangle
    and caluclate its perimeter, stopping the iteration once
    the perimeter exceeds the given maximum. Along the way,
    the desired quantity (in this case the sum over all the
    perimeters) can be evaluated.
    """
    #since = time.time()
    res = 0
    for x, y in pellSolutionGenerator(D=3, negative=False):
        #print(f"x = {x}, y = {y}")
        perim1 = 4 * x ** 2
        perim2 = 2 * (x + 3 * y) ** 2
        #print(perim1, perim2, perim1 < perim2)
        if perim1 <= perimeter_max:
            res += perim1
        if perim2 > perimeter_max: break
        res += perim2
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
        
# Problem 95
# Review- try to make faster
def properFactorSum(num: int, ps: Optional[PrimeSPFsieve]=None) -> int:
    """
    For a strictly positive integer num, returns the sum of its positive
    proper factors.
    
    Args:
        Required positional:
        num (int): The strictly positive integers whose proper factors
                are to be summed
        
        Optional named:
        ps (PrimeSPFsieve or None): If specified, a PrimeSPFsieve
                object to facilitate the calculation of the positive
                proper factors of num.
                If this is given, then this object may be updated
                by this function, in particular it may extend the
                prime sieve it contains.
            Default: None
    
    Returns:
    Integer (int) giving the sum of the proper factors of num
    """
    if ps is None: ps = PrimeSPFsieve()
    p_fact = ps.primeFactorisation(num)
    res = 1
    for p, k in p_fact.items():
        res *= (p ** (k + 1) - 1) // (p - 1)
    return res - num

def amicableChains(num_mx: int=10 ** 6) -> int:
    """
    Solution to Project Euler #95

    An amicable chain is a finite sequence of positive integers such
    that any element of the sequence other than the first is equal
    to the sum of proper positive factors of the preceding element
    of the sequence, and the first element of the sequence is
    equal to the sum of proper positive factors of the last element
    of the sequence.
    This function finds the smallest integer that belongs to an
    amicable chain of length m where none of the elements of that
    amicable chain exceeds num_mx, such that there does not exist
    an amicable chain of length m + 1 for which none of its elements
    exceed num_mx.
    
    Args:
        Optional named:
        num_mx (int): The largest integer that is allowed to
                be present in any amicable chain considered.
            Default: 10 ** 6
    
    Returns:
    Integer (int) giving the smallest integer that belongs to an
    amicable chain of length m where none of the elements of that
    amicable chain exceeds num_mx, such that there does not exist
    an amicable chain of length m + 1 for which none of its elements
    exceed num_mx.
    """
    #since = time.time()
    ps = PrimeSPFsieve(num_mx)
    seen = {0, 1}
    res = (0, ())
    for num in range(2, num_mx):
        if num in seen: continue
        seen2 = {}
        num2 = num
        stk = []
        while num2 < num_mx and num2 not in seen:
            seen2[num2] = len(stk)
            seen.add(num2)
            stk.append(num2)
            num2 = properFactorSum(num2, ps)
        if num2 not in seen2.keys(): continue
        i0 = seen2[num2]
        length = len(stk) - i0
        if length < res[0]: continue
        if i0:
            mn = (float("inf"),)
            for i in range(i0, len(stk)):
                mn = min(mn, (stk[i], i))
            i = mn[1]
        else: i = 0
        if length == res[0] and stk[i] > res[1][0]:
            continue
        chain = (stk[i:] + stk[i0:i]) if i else stk[i0:]
        res = (length, tuple(chain))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    #print(res)
    return res[1][0]

# Problem 96- try to incorporate more advanced solve techniques, e.g. pairs
def loadSudokusFromFile(
    doc: str,
    rel_package_src: bool=False,
) -> Dict[str, List[str]]:
    """
    Loads a list of sudokus from the .txt file at relative or absolute
    location doc. Each sudoku in the .txt file should be separated
    from the previous one (if applicable) by a line break ('\\n') and
    its first line be the name of the sudoku. Subsequent lines should
    be the concatenation of the digits in the rows of the sudoku
    in order, with "0" used for blank squares.
    
    Note that this currently does not support sudokus containing digits
    exceeding 9 (so for instance does not support 16x16 sudokus)
    
    Args:
        Required positional:
        doc (str): The relative or absolute path to the .txt
                file containing the sudokus

        Optional named:
        rel_package_src (bool): Whether a relative path given by doc
                is relative to the current directory (False) or the
                package directory (True).
            Default: False

    Returns:
    Dictionary whose keys are the names of the sudokus in the .txt file
    at location doc, and for each key the corrseponding value is that
    sudoku as a list of strings, with each entry in the list being
    a concatenation of the digits in the corresponding row of the
    sudoku with "0" used for blank squares.
    """
    txt = loadTextFromFile(doc, rel_package_src=rel_package_src)
    lines = [x for x in txt.split("\n") if x]
    sudoku_dict = {}
    curr = None
    for line in lines:
        line2 = line.strip()
        if line2.startswith("Grid"):
            curr = line2
            sudoku_dict[curr] = []
        elif curr is not None:
            sudoku_dict[curr].append(line2)
    return sudoku_dict

# Review- revise naming. In particular board, which at some points
#   refers to the symbol representation and at others the integer
#   numpy array representation.

# Review- consider splitting this off into its own project to
# experiment with more advanced logic, including pairs and triples
# to reduce the extent to which backtracking is required and so
# speed up evaluation time.
class Sudoku(object):
    """
    Class whose instances represent sudokus, containing methods to
    check that the sudoku is valid and solve the sudoku.
    
    Initialisation args:
        Required positional:
        board (list of lists of hashable objects): Representation of
                the sudoku board, with each list in board representing
                a row of the sudoku in order and each entry of these
                lists is either one of the items in fellow argument
                symbols or fellow argument empty_symbol.
                The length of board must be a square and the length
                of each entry in board must be equal to the length
                of board.
        symbols (set of hashable objects): The set of symbols used
                in the solved sudoku. Must be the same length as
                the fellow argument board.
        
        Optional named:
        empty_symbol (hashable object): The symbol representing an
                empty square in fellow argument board.
            Default: "0"
    
    Attributes:
        length (int): The number of small squares in each side of the
                sudoku board. Is a square number.
        square_len (int): The number of elements in each side of the squares.
                This is the square root of the attribute length.
        symbol_lst (list of hashable objects): A complete list of the
                symbols used in the sudoku including the symbol
                representing an empty square (i.e. initialsiation
                argument empty_symbol) which is the first element
                (index 0). It has no repeated elements and has
                length equal to one more than the attribute length.
                Each symbol in the solved sudoku corresponds to an
                integer between 1 and length inclusive, which is the
                index at which that symbol appears in this list.
        symbol_dict (dict): Dictionary whose keys and values are
                exactly the elements of attribute symbol_lst the index
                that symbol appears in symbol_lst respectively. For
                the symbols of the solved sudoku, this gives the
                integer from 1 to length inclusive to which that symbol
                corresponds and for the symbol representing an empty
                square this gives 0.
                This is effectively the inverse of symbol_lst in that
                for any symbol x in symbol_lst:
                 symbol_lst[symbol_dict[x]] = x
                and for any integer n between 0 and length inclusive:
                 symbol_dict[symbol_lst[n]] = n
        symbol_width_mx (int): Strictly positive integer giving the
                largest number of characters needed for the string
                representations of any of the elements of symbol_lst.
        board_orig_symbol (list of lists of hashable objects): A
                copy of the original board, as given by initialisation
                argument board. This should remain unaltered.
        board_orig (2-dimensional numpy array with unsigned integers):
                Representation of the original sudoku board with the
                symbols replaced by their corresponding integer
                between 0 and length (see attribute symbol_lst). This
                should remain unaltered.
    """
    bm_dict = {}
    idx_dtype = np.uint8
    val_dtype = np.uint8
    bm_dtype = np.uint16

    def __init__(self, board: List[List[Hashable]],\
            symbols: Set[Hashable], empty_symbol: Hashable="0"):
        length = len(board)
        for row in board:
            if len(row) != length:
                raise ValueError("board must be an array where every "
                        "row and column has the same length.")
        if length != len(symbols):
            raise ValueError("board must have dimensions equal to the "
                        "number of symbols")
        square_len = isqrt(length)
        if square_len * square_len != length:
            raise ValueError("board must have dimensions which equal "
                    "an exact square")
        self.square_len = square_len
        self.length = length
        for i in range(len(self.bm_dict), length):
            self.bm_dict[1 << i] = i + 1
        self.symbol_lst = [empty_symbol]
        self.symbol_dict = {empty_symbol: 0}
        for l in symbols:
            self.symbol_dict[l] = len(self.symbol_lst)
            self.symbol_lst.append(l)
        #self.symbol_width_mx = max(len(str(x)) for x in self.symbol_lst)
        self.board_orig_symbol = board
        self.board_orig = self.symbol2Array(self.board_orig_symbol)
        self.idx_arr = np.zeros((length, length, 2), dtype=self.idx_dtype)
        for i1 in range(length):
            self.idx_arr[i1, :, 0] = i1
        for i2 in range(length):
            self.idx_arr[:, i2, 1] = i2
        self.resetBoard()
        if not self.isValid():
            raise ValueError("board does not represent a valid sudoku "
                    "starting point")
    
    # Getters and setters
    @property
    def symbol_width_mx(self):
        res = getattr(self, "_symbol_width_mx", None)
        if res is None:
            res = max(len(str(x)) for x in self.symbol_lst)
            self._symbol_width_mx = res
        return res
    
    def array2Symbol(self, arr: np.ndarray) -> List[List[Hashable]]:
        """
        Method converting an integer array board representation into
        a list of list of symbols, based on self.symbol_list.

        Args:
            arr (self.length x self.length numpy array): An array
                    of integers representing the board state to be
                    converted.

        Returns:
        List of lists of hashable objects, representing the same board
        state in symbols as arr represents with integers based on the
        correspondence defined by the attributes self.symbol_list and
        self.symbol_dict. The outer list and every inner list have
        length self.length and contain as elements the contents of
        self.symbol_list.
        """
        res = [[0 for _ in range(self.length)] for _ in range(self.length)]
        for i1, row in enumerate(arr):
            for i2, num in enumerate(row):
                res[i1][i2] = self.symbol_lst[num]
        return res
    
    def symbol2Array(self, board: List[List[Hashable]]) -> np.ndarray:
        """
        Method converting a list of lists of symbols board representation
        into an integer array board representation, based on self.symbol_list.

        Args:
            board (list of lists of hashable objects): A list of lists of
                    symbols representing the board state to be converted,
                    with symbols all present in the attribute
                    self.symbol_list. The outer list and each inner list
                    should have length self.length.

        Returns:
        A self.length x self.length integer numpy array, representing the
        same board state as board represents with symbols based on the
        correspondence defined by the attributes self.symbol_list and
        self.symbol_dict.
        """
        res = np.zeros((self.length, self.length), dtype=self.val_dtype)
        for i1, row in enumerate(board):
            for i2, s in enumerate(row):
                res[(i1, i2)] = self.symbol_dict[s]
        return res
    
    def __str__(self) -> str:
        #print(self.board)
        symbol_width = self.symbol_width_mx
        res = []
        for i, row in enumerate(self.board):
            row_print = []
            for j, num in enumerate(row):
                s = self.symbol_lst[num]
                d = symbol_width - len(s)
                n_spc_pref = d >> 1
                n_spc_suff = d - n_spc_pref
                row_print.append("".join([" " * (n_spc_pref + bool(j)),\
                        s, " " * n_spc_suff]))
                if j % self.square_len == self.square_len - 1 and j != len(row) - 1:
                    row_print.append(' |')
            res.append(''.join(row_print))
            if i % self.square_len == self.square_len - 1 and i != len(self.board) - 1:
                res.append("-" * len(res[-1]))
        return "\n".join(res)
    
    def resetBoard(self) -> None:
        """
        Method resetting the sudoku to its original state.

        Args:
            None

        Returns:
        None
        """
        #print("reset")
        self.board = copy.deepcopy(self.board_orig)
        board = self.board
        length = self.length
        #empty_symbol = self.symbol_lst[0]
        self.board_bm = np.zeros((length, length), dtype=self.bm_dtype)
        self.board_n_opts = np.full((length, length), length,\
                                    dtype=self.val_dtype)
        self.unset_idx = {}
        bm_val = (1 << length) - 1
        self.row_bms = [bm_val] * length
        self.col_bms = [bm_val] * length
        self.sq_bms = [bm_val] * length
        for i1, row in enumerate(board):
            for i2, num in enumerate(row):
                idx = (i1, i2)
                if not num:
                    self.board_bm[idx] = bm_val
                    self.unset_idx.setdefault(i1, set())
                    self.unset_idx[i1].add(i2)
                else:
                    self.board_n_opts[idx] = 0
        return
    
    def rowIdxGenerator(self) -> Generator[Tuple[int, int], None, None]:
        """
        Generator yielding the indices of one element in each
        different row.

        Args:
            None

        Yields:
        2-tuple of ints giving the indices of an element of the
        board, with each yielded corresponding to a different
        row and all rows represented.
        """
        for i1 in range(self.length):
            yield (i1, 0)
        return
    
    def columnIdxGenerator(self) -> Generator[Tuple[int], None, None]:
        """
        Generator yielding the indices of one element in each
        different column.

        Args:
            None

        Yields:
        2-tuple of ints giving the indices of an element of the
        board, with each yielded corresponding to a different
        column and all columns represented.
        """
        for i2 in range(self.length):
            yield (0, i2)
        return
    
    def squareIdxGenerator(self) -> Generator[Tuple[int], None, None]:
        """
        Generator yielding the indices of one element in each
        different square.

        Args:
            None

        Yields:
        2-tuple of ints giving the indices of an element of the
        board, with each yielded corresponding to a different
        square and all squares represented.
        """
        for i1 in range(0, self.length, self.square_len):
            for i2 in range(0, self.length, self.square_len):
                yield (i1, i2)
        return
    
    def getRow(
        self,
        idx: Tuple[int, int],
        board: Optional[np.ndarray]=None,
    ) -> np.ndarray:
        """
        Method giving the section of the board belonging to the row
        containing the element at index idx, as a 1 x self.length
        integer numpy array.

        Args:
            Required positional:
            idx (2-tuple of ints): The indices of one of the elements
                    the row to be returned belongs.
            
            Optional named:
            board (self.length x self.length numpy array or None): If
                    specified, the board from which the row is to be
                    extracted. Otherwise, the row is extracted from
                    self.board.
        
        Returns:
        1 x self.length integer numpy array representing the state
        of the selected row.
        """
        if board is None: board = self.board
        slc = (slice(idx[0], idx[0] + 1), slice(None))
        return board[slc]
    
    def getColumn(
        self,
        idx: Tuple[int, int],
        board: Optional[np.ndarray]=None
    ) -> np.ndarray:
        """
        Method giving the section of the board belonging to the column
        containing the element at index idx, as a self.length x 1
        integer numpy array.

        Args:
            Required positional:
            idx (2-tuple of ints): The indices of one of the elements
                    the column to be returned belongs.
            
            Optional named:
            board (self.length x self.length numpy array or None): If
                    specified, the board from which the column is to be
                    extracted. Otherwise, the column is extracted from
                    self.board.
        
        Returns:
        self.length x 1 integer numpy array representing the state
        of the selected column.
        """
        if board is None: board = self.board
        slc = (slice(None), slice(idx[1], idx[1] + 1))
        return board[slc]
    
    def getSquare(
        self,
        idx: Tuple[int, int],
        board: Optional[np.ndarray]=None
    ) -> np.ndarray:
        """
        Method giving the section of the board belonging to the square
        containing the element at index idx, as a
        self.square_len x self.square_len integer numpy array.

        Args:
            Required positional:
            idx (2-tuple of ints): The indices of one of the elements
                    the square to be returned belongs.
            
            Optional named:
            board (self.length x self.length numpy array or None): If
                    specified, the board from which the square is to be
                    extracted. Otherwise, the square is extracted from
                    self.board.
        
        Returns:
        self.square_len x self.square_len integer numpy array representing
        the state of the selected square.
        """
        if board is None: board = self.board
        idx2 = tuple((x // self.square_len) * self.square_len for x in idx)
        slc = tuple(slice(x, x + self.square_len, 1) for x in idx2)
        return board[slc]
    
    def isValid(self) -> bool:
        """
        Method checking whether the current state contains any
        direct conflicts (i.e. the same element more than once
        in any square, row or column).
        Note that a returned value of True does not guarantee
        that a solution for the current state exists.

        Args:
            None

        Returns:
        Boolean (bool) giving False if the current state contains
        a direct conflict, True otherwise.
        """
        for (idx_gen, get_func) in ((self.rowIdxGenerator, self.getRow),\
                (self.columnIdxGenerator, self.getColumn),\
                (self.squareIdxGenerator, self.getSquare)):
            for idx in idx_gen():
                arr = get_func(idx, board=self.board)
                seen = set()
                for num in np.nditer(arr, order="C", op_dtypes=[self.idx_dtype]):
                    num = int(num)
                    if not num: continue
                    if num in seen: return False
                    seen.add(num)
        return True
    
    def checkSolution(self) -> bool:
        """
        Method checking whether the current state constitutes
        a valid solution for the initial state. This constitutes:
         1) Checking that the board still has the same dimensions
         2) The current state has no direct conflicts
         3) There are no unset elements
         4) All set elements in the initial state have their
            corresponding elements in the current state set to
            the same value.
        
        Args:
            None

        Returns:
        Boolean (bool) giving True if the current state constitutes
        a valid solution for the initial state, otherwise False.
        """
        if self.board.shape != self.board_orig.shape or 0 in self.board or not self.isValid():
            return False
        for row1, row2 in zip(self.board_orig, self.board):
            for num1, num2 in zip(row1, row2):
                if num1 and num1 != num2: return False
        return True
    
    def setValues(
        self,
        val_dict: Dict[Tuple[int], str],
        recursive: bool=True
    ) -> bool:
        """
        Method setting the values at given positions on the board,
        and (if recursive is set to True) propogating those changes
        forwards to see if they affect the possible values at other
        positions.

        Args:
            Required positional:
            val_dict (dict): Dictionary whose keys are 2-tuples of
                    ints indicating the indices of the position on
                    the board to be set and corresponding value being
                    the symbol to which that the position should be set.

            Optional named:
            recursive (bool): Indicates whether to recursively propogate
                    this result (if True) or simply update the values
                    directly affected (if False)
                Default: True
        
        Returns:
        Boolean (bool) indicating whether setting the given values gave
        rise to a valid state (if True) or gave rise to a contradiction
        (either through a direct conflict or a conflict arising through
        recursive propogation if the input recursive is True).
        """
        return self._setValue(
            {idx: self.symbol_dict[val] for idx, val in val_dict.items()},
            recursive=recursive
        )
    
    def _setValues(
        self,
        val_dict: Dict[Tuple[int], int],
        recursive: bool=True,
        find_forces: bool=True
    ) -> bool:
        # The potentially recursive implementation behind the setValues()
        # method that should be accessed through that function.
        if not val_dict: return True
        add_idx = set()
        for idx, val in val_dict.items():
            bm = 1 << (val - 1)
            if not self.board_bm[idx] & bm:
                return False
            self.board_bm[idx] = 0
            self.board[idx] = val
            self.board_n_opts[idx] = 0
            self.unset_idx[idx[0]].remove(idx[1])
            if not self.unset_idx[idx[0]]: self.unset_idx.pop(idx[0])
            for get_func in (self.getRow, self.getColumn, self.getSquare):
                val_arr = get_func(idx, board=self.board)
                n_opts_arr = get_func(idx, board=self.board_n_opts)
                bm_arr = get_func(idx, board=self.board_bm)
                n_opts_arr -= ((bm_arr & bm) != 0)
                bm2 = ((1 << self.length) - 1) & ~bm
                bm_arr &= np.array(bm2, dtype=self.bm_dtype)
                if np.any((val_arr == 0) & (n_opts_arr == 0)):
                    return False
                idx_arr = get_func(idx, board=self.idx_arr)
                add = {tuple(int(y) for y in x)\
                        for x in idx_arr[n_opts_arr == 1]}
                add_idx |= add
        if not add_idx or not recursive: return True
        val_dict_nxt = {idx: self.bm_dict[self.board_bm[idx]]\
                for idx in add_idx if idx not in val_dict.keys()}
        if find_forces:
            b, val_dict2 = self.findForces()
            if not b: return False
            for k, v in val_dict2.items():
                val_dict_nxt[k] = v
        return self._setValues(val_dict_nxt, recursive=True)
    
    def findForces(self) -> Tuple[bool, Dict[Tuple[int], int]]:
        """
        Method that for the current state calculates whether there
        are any positions that are not set and are directly prevented
        from taking any value (implying that the current state is
        unsolvable) and if not finds any positions that are not
        currently set but are directly prevented from taking all but
        one value (and so can and should be set to that value).

        Args:
            None
        
        Returns:
        2-tuple where:
         - Index 0 contains a boolean (bool) indicating whether all
           positions that are not set are able to take at least one
           value.
         - Index 1 contains a dictionary (dict). If the boolean in 
           index 0 is False, then the dictionary is empty. Otherwise,
           it contains as keys 2-tuples of ints giving the indices
           of the positions that are not currently set but are
           directly prevented from taking all but one value (and so
           can and should be set to that value), with the
           corresponding dictionary value being that value.
        """
        #print("using findForces()")
        val_dict_nxt = {}
        for (idx_gen, get_func) in (
            (self.rowIdxGenerator, self.getRow),
            (self.columnIdxGenerator, self.getColumn),
            (self.squareIdxGenerator, self.getSquare),
        ):
            for idx in idx_gen():
                bm_arr = get_func(idx, board=self.board_bm)
                bm_cumu = []
                curr = 0
                for i1, r in enumerate(bm_arr):
                    for i2, bm in enumerate(r):
                        bm_cumu.append(curr)
                        curr |= bm
                curr = 0
                j = len(bm_cumu) - 1
                for i1 in reversed(range(len(bm_arr))):
                    for i2 in reversed(range(len(bm_arr[i1]))):
                        bm = bm_arr[i1][i2]
                        bm2 = bm & ~(curr | bm_cumu[j])
                        j -= 1
                        curr |= bm
                        if not bm2: continue
                        if bm2 not in self.bm_dict.keys():
                            return (False, {})
                        val_dict_nxt[(idx[0] + i1, idx[1] + i2)] = self.bm_dict[bm2]
        return (True, val_dict_nxt)
    
    def setupBoard(self) -> bool:
        """
        Method initialising the variables related to the board
        in the initial state, in particular calculating which
        values each non-set position can take and recursively
        setting any non-set positions that logically can take
        one and only one value.

        Args:
            None

        Returns:
        Boolean (bool) indicating whether the starting state
        leads to an unavoidable conflict (in which case False is
        returned, indicating that there cannot exist solutions),
        or this initial setup did not encounter any unavoidable
        conflicts (in which case True is returned, indicating that
        there may or may not exist one or more solutions).
        """
        for (idx_gen, get_func) in ((self.rowIdxGenerator, self.getRow),\
                (self.columnIdxGenerator, self.getColumn),\
                (self.squareIdxGenerator, self.getSquare)):
            for idx in idx_gen():
                bm = 0
                for val in {int(x) for x in\
                        np.nditer(get_func(idx, board=self.board), order="C")}:
                    if val: bm |= 1 << (val - 1)
                if not bm: continue
                bm_arr = get_func(idx, board=self.board_bm)
                #print(bm_arr)
                #print(bm)
                bm2 = ((1 << self.length) - 1) & ~bm
                bm_arr &= np.array(bm2, dtype=self.bm_dtype)
        val_dict = {}
        for i1, i2_set in self.unset_idx.items():
            for i2 in i2_set:
                idx = (i1, i2)
                n_opts = 0
                bm = self.board_bm[idx]
                while bm:
                    bm &= (bm - 1)
                    n_opts += 1
                if not n_opts: return False
                self.board_n_opts[idx] = n_opts
                if n_opts > 1: continue
                val_dict[idx] = self.bm_dict[self.board_bm[idx]]
        return self._setValues(val_dict, recursive=True)
    
    
    
    def backtrackSolve(
        self,
        print_time: bool=False,
        check_solutions: bool=True
    ) -> Generator["Sudoku", None, None]:
        """
        Generator using backtracking with pruning through using
        simple logical deductions to rule out impossible states
        in order to yield all possible solutions to the sudoku
        given the initial state provided.

        Args:
            Optional named:
            print_time (bool): Indicates whether the time taken
                    to solve should be printed to console (True)
                    or not (False).
                Default: False
            check_solutions (bool): Whether each solution should
                    be double-checked to ensure it is a valid
                    solution to the initial state (if True then
                    yes, if False then no). This is for the purpose
                    of debugging, as if the method is working
                    correctly it should not be necessary. As such,
                    if this is set to True and an invalid solution
                    is identified, then a ValueError will be raised.
                Default: True
        Yields:
        A Sudoku object that is a valid solution to the initial
        state. Collectively, all possible such solutions are yielded.
        These are yielded in the order they are discovered, which
        has no special significance.
        """
        since = time.time()
        if not self.setupBoard():
            #print("hello")
            return # Not solvable
        def backtrack() -> Generator[np.ndarray, None, None]:
            if not self.unset_idx:
                if check_solutions and not self.checkSolution():
                    raise ValueError("State identified by backtrackSolve() "
                            "as a solution s not a valid "
                            f"solution:\n{str(self)}\nThis indicates an "
                            "error in one or more of the methods in the "
                            "Sudoku class.")
                yield copy.deepcopy(self)
                return
            best = (float("inf"),)
            for i1, i2_set in self.unset_idx.items():
                for i2 in i2_set:
                    idx = (i1, i2)
                    best = min(best, (self.board_n_opts[idx], idx))
                    if best[0] == 2: break
            idx = best[1]
            bm = self.board_bm[idx]
            board = copy.deepcopy(self.board)
            board_bm = copy.deepcopy(self.board_bm)
            board_n_opts = copy.deepcopy(self.board_n_opts)
            unset_idx = copy.deepcopy(self.unset_idx)
            for _ in range(best[0]):
                bm2 = bm & (bm - 1)
                val = self.bm_dict[bm & ~bm2]
                if self._setValues({idx: val}, recursive=True, find_forces=True):
                    yield from backtrack()
                bm = bm2
                # Reset
                self.board = copy.deepcopy(board)
                self.board_bm = copy.deepcopy(board_bm)
                self.board_n_opts = copy.deepcopy(board_n_opts)
                self.unset_idx = copy.deepcopy(unset_idx)
            return
        
        for i, arr in enumerate(backtrack(), start=1):
            if print_time:
                print(f"Solution {i} found after {time.time() - since:.4f} seconds")
            yield arr#self.array2Symbol(arr)
        if print_time:
            print(f"Solution search completed in {time.time() - since:.4f} seconds")
        return
            

def sudokusSolutionUpperLeftSumFromFile(
    sudoku_doc: str="project_euler_problem_data_files/p096_sudoku.txt",
    rel_package_src: bool=True,
) -> int:
    """
    Solution to Project Euler #96

    For a collection of sudokus contained in the .txt document
    sudoku_doc that are guaranteed to have unique solutions, calculates
    the sum of the top-leftmost entry over each sudoku for its
    unique solution.

    Each sudoku in the .txt file should be separated from the
    previous one (if applicable) by a line break ('\\n') and
    its first line be the name of the sudoku. Subsequent lines should
    be the concatenation of the digits in the rows of the sudoku
    in order, with "0" used for blank squares.

    Note that while currently does not support sudokus containing digits
    exceeding 9 (so for instance does not support 16x16 or larger sudokus).
    This limitation is due to parsing of the .txt file, not a limitation
    of the solution method.

    Args:
        Optional named:
        sudoku_doc (str): The relative or absolute path to the .txt
                file containing the sudokus
            Default: "project_euler_problem_data_files/p096_sudoku.txt"
        rel_package_src (bool): Whether a relative path given by
                sudoku_doc is relative to the current directory (False)
                or the package directory (True).
            Default: True

    Returns:
    Integer (int) giving the described sum of the top-leftmost entries
    of the solved sudokus.
    """
    #since = time.time()
    board_dict = loadSudokusFromFile(
        sudoku_doc,
        rel_package_src=rel_package_src,
    )
    empty_symbol = "0"
    
    sudoku_dict = {}
    res = 0
    for k, board in board_dict.items():
        #print(f"\n{k}")
        symbols = [str(x) for x in range(1, len(board) + 1)]
        sudoku_dict[k] = Sudoku(board, symbols,\
                empty_symbol=empty_symbol)
        #print("Original board:")
        #print(str(sudoku_dict[k]))
        solutions = list(sudoku_dict[k].backtrackSolve(
            print_time=False,
            check_solutions=True,
        ))
        if not solutions:
            raise ValueError(f"No solution found for {k}")
        elif len(solutions) > 1:
            sol_str = "\n\n".join(str(x) for x in solutions)
            raise ValueError(f"Multiple soluctions found for "
                    "{k}:{sol_str}")
        sol = solutions[0]
        #print("Solution:")
        #print(str(sol))
        sol_lst = sol.array2Symbol(sol.board)
        res += int("".join(sol_lst[0][:3]))
    #if print_time:
    #    print(f"\nTotal time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 97
def largePowerModulo(
    a: int,
    b: int,
    md: int,
    ps: Optional[PrimeSPFsieve]=None,
) -> int:
    """
    For integer a, non-negative integer b and strictly positive
    integer md, finds (a ** b) % md (i.e. a to the power of b
    modulo md).
    Note that this performs the same function as the inbuilt
    function pow() with a modulus, but using this would make the
    problem trivial.
    
    Args:
        Required positional:
        a (int): Integer a in the above equation, the base.
        b (int): Non-negative integer b in the above equation, the
                power.
        md (int): Strictly positive integer md in the above equation,
                the modulus.
        
        Optional named:
        ps (PrimeSPFsieve or None): If specified, a PrimeSPFsieve
                object to potentially facilitate the calculation of
                the Euler totient function of md
                If this is given, then this object may be updated
                by this function, in particular it may extend the
                prime sieve it contains.
        
    Returns:
    Integer (int) giving the value of (a ** b) % md
    """
    if md == 1: return 0
    a %= md
    if not a: return 0
    if gcd(a, md) == 1:
        # Using Euler's generalisation of Fermat's little theorem:
        #  (a ** phi(n)) % n = 1
        # if gcd(a, n) = 1 where phi is Euler's totient function
        b_md = totientFunction(md, ps=ps)
        b = b % b_md
    res = 1
    curr = a
    while b:
        if b & 1:
            res = (res * curr) % md
        curr = (curr * curr) % md
        b >>= 1
    return res

def largeNonMersennePrimeLastDigits(
    mult: int=28433,
    a: int=2,
    b: int=7830457,
    add: int=1,
    n_tail_dig: int=10,
    base: int=10,
) -> int:
    """
    Solution to Project Euler #97

    For integers mult, a, b and add, finds that rightmost n_dig digits of
    (mult * a ** b + add) when expressed in the chosen base.
    
    Args:
        Optional named:
        mult (int): The integer mult in the above equation
            Default: 28433
        a (int): The integer a in the above equation
            Default: 2
        b (int): The integer b in the above equation
            Default: 7830457
        add (int): The integer add in the above equation
            Default: 1
        n_tail_dig (int): The number of rightmost digits of the solution
                to return when represented in the chosen base.
            Default: 10
        base (int): The base in which the number is to be expressed
            Default: 10
    
    Returns:
    Integer (int) giving the value of the last n_dig digits of
    (mult * a ** b + add) when expressed in the chosen base
    """
    #since = time.time()
    md = base ** n_tail_dig
    res = largePowerModulo(a, b, md)
    res = (res * (mult % md)) % md
    res = (res + (add % md)) % md
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 98- tidy up and try to speed up

def anagramSets(words: List[str]) -> Dict[int, List[Set[str]]]:
    """
    Finds the strings in words that are the anagram of at least
    one other string in words (i.e. contain the same characters
    and the same number of each character) and partitions these
    strings into sets of strings that are anagrams of each other
    (noting that the anagram relation is an equivalence relation
    and so can act as a partition). These sets are themselves
    partitioned based on the length of the words in the set
    (noting that anagrams must have the same length).
    
    Args:
        Required positional:
        words (list of strs): The strings being partitioned
    
    Returns:
    Dictionary whose keys are the integers for which there
    exists at least one pair of strings in words that are anagrams
    of each other. For each such key, the corresponding value is
    a list containing the sets of words of that length that are
    anagrams of each other (with each such set containing at
    least 2 elements).
    
    Example:
    >>> anagramSets(["CAT", "SOUTH", "SHOUT", "ACT", "EXCEPT",\
            "POST", "DOG", "STOP", "GOD", "SPOT"])
    {3: [{'ACT', 'CAT'}, {'DOG', 'GOD'}],
     5: [{'SHOUT', 'SOUTH'}],
     4: [{'POST', 'SPOT', 'STOP'}]}
    """
    anagrams = {}
    for w in words:
        w_sorted = tuple(sorted(w))
        anagrams.setdefault(w_sorted, set())
        anagrams[w_sorted].add(w)
    res = {}
    mx_len = 0
    for w_sorted, anagram_set in anagrams.items():
        if len(anagram_set) == 1: continue
        mx_len = max(mx_len, len(anagram_set))
        k = len(w_sorted)
        res.setdefault(k, [])
        res[k].append(anagram_set)
    return res


def anagramicSquares(
    doc: str="project_euler_problem_data_files/0098_words.txt",
    base: int=10,
    rel_package_src: bool=True,
) -> int:
    # Review wording of documentation
    """
    Solution to Project Euler #98
    
    Given a list of words in the .txt file doc and a given base,
    consider the set of pairs of these words such that the two
    words in the pair are different and are anagrams of each other
    (i.e. contain the same characters and the same number of each
    character). For each pair in this set, consider the set of
    bijective mappings between the letters that appear at least
    once in the two words and the digits of the chosen base
    (i.e. 0 to base - 1) such that, if the digits of the two
    words are replaced by the corresponding digit and the
    value of the resulting series of digits when interpreted
    as an integer in the chosen base that for both words
    has no leading zeros and is a square.
    This function finds the largest number that either of
    the pair is mapped to in this way out of all the pair and
    bijection combinations.
    
    Args:
        Optional named:
        doc (str): The relative or absolute location of the
                .txt file that contains the words. The file
                should be contain the words separated by
                commas with each word surrounded by double
                quotation marks.
            Default: "project_euler_problem_data_files/0098_words.txt"
        base (int): The base from which the digits to which the
                characters of the words are mapped to and in
                which the resultant series of digits are
                interpreted.
                This must be at least as large as the number
                of distinct characters in any of the words
            Default: 10
        rel_package_src (bool): Whether a relative path given by
                doc is relative to the current directory (False)
                or the package directory (True).
            Default: True
    
    Returns:
    Integer (int) giving the largest value that any valid word pair
    and bijection combination maps one of the the words to when
    expressed in the chosen base. If there are no valid word pair and
    bijection combinations, returns 0.
    """
    #since = time.time()
    words = loadStringsFromFile(doc, rel_package_src=rel_package_src)
    anagrams = anagramSets(words)
    lengths = sorted(anagrams.keys())
    mx_n_dig = lengths[-1]
    
    def digitPattern(num: int, base: int=10) -> Tuple[int]:
        digs = []
        while num:
            num, r = divmod(num, base)
            digs.append(r)
        res = []
        seen = {}
        for d in reversed(digs):
            if d not in seen.keys(): seen[d] = len(seen)
            res.append(seen[d])
        return tuple(res)

    def wordPattern(w: str) -> Tuple[int]:
        res = []
        seen = {}
        for l in w:
            if l not in seen.keys(): seen[l] = len(seen)
            res.append(seen[l])
        return tuple(res)

    def num2wordMapping(num: int, w: str, base: int=10)\
            -> Tuple[Tuple[int]]:
        # Assumes a bijection between the digits in num in the
        # given base and the letters of w exists
        res = []
        i = len(w) - 1
        seen = set()
        while num:
            num, d = divmod(num, base)
            if d not in seen:
                res.append((d, w[i]))
            i -= 1
        return tuple(sorted(res))
    
    word2pattern = {}
    pattern2words = {}
    for w_sets in anagrams.values():
        for w_set in w_sets:
            for w in w_set:
                pattern = wordPattern(w)
                pattern2words.setdefault(pattern, set())
                pattern2words[pattern].add(w)
                word2pattern[w] = pattern
    #print(pattern2words)
    #print(word2pattern)
    res = (-1,)
    for length in reversed(lengths):
        sq_mn = base ** (length - 1)
        sq_mx = sq_mn * base
        if length & 1:
            sqrt_mx = isqrt(sq_mx)
            sqrt_mx += (sqrt_mx ** 2 != sq_mx)
            sqrt_mn = base ** ((length - 1) >> 1)
        else:
            sqrt_mn = isqrt(sq_mn)
            sqrt_mn += (sqrt_mn ** 2 != sq_mn)
            sqrt_mx = base ** (length >> 1)
        #sq_dig_patterns = {}
        word2sq_mappings = {}
        for i in range(sqrt_mn, sqrt_mx):
            num = i ** 2
            k = digitPattern(num, base=base)
            #if k not in pattern2words.keys(): continue
            for w in pattern2words.get(k, set()):
                mapping = num2wordMapping(num, w, base=base)
                word2sq_mappings.setdefault(w, set())
                word2sq_mappings[w].add(mapping)
            #sq_dig_patterns.setdefault(k, set())
            #sq_dig_patterns[k].add(num)
        #print(word2sq_mappings)
        #print(sq_dig_patterns)
    
        anagram_sets = anagrams[length]
        for anagram_set in anagram_sets:
            anagram_lst = list(anagram_set)
            for i2 in range(1, len(anagram_lst)):
                w2 = anagram_lst[i2]
                if w2 not in word2sq_mappings.keys(): continue
                mappings2 = word2sq_mappings[w2]
                for i1 in range(i2):
                    w1 = anagram_lst[i1]
                    map_intersect = word2sq_mappings.get(w1,\
                            set()).intersection(mappings2)
                    for mapping in map_intersect:
                        map_dict = {l: d for d, l in mapping}
                        #print(w1, w2)
                        for w in (w1, w2):
                            ans = 0
                            for l in w:
                                ans = ans * base + map_dict[l]
                            #print(ans)
                            res = max(res, (ans, w1, w2))
        if res[0] != -1: break
    else: return 0
    #print(f"Best anagram pair was ({res[1]}, {res[2]}), with "
    #        f"corresponding larger square {res[0]}")
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res[0]

# Problem 99
def loadExponentPairsFromFile(
    doc: str,
    rel_package_src: bool=True,
) -> List[Tuple[int]]:
    """
    Loads a list of base exponent pairs from a .txt file at
    relative or absolute location doc, where the pairs are
    separated from each other by line breaks ('\\n') and the
    base and exponent of each pair are separated by a comma,
    with the base before the comma and exponent after the
    comma.
    
    Args:
        Required positional:
        doc (str): The relative or absolute path to the .txt
                file containing the base exponent pairs

        Optional named:
        rel_package_src (bool): Whether a relative path given by
                doc is relative to the current directory (False)
                or the package directory (True).
            Default: False
    
    Returns:
    List of 2-tuples of ints, with each 2-tuple containing
    the base and exponent of the corresponding base exponent
    pair in doc at index 0 and 1 respectively. The base exponent
    pairs are in the list in the same order they appear in
    doc.
    """
    txt = loadTextFromFile(doc, rel_package_src=rel_package_src)
    return [tuple(int(y) for y in x.split(",")) for x in txt.split("\n")]

def largestLogExponential(exp_pairs: List[Tuple[int]]) -> int:
    """
    For a list of base exponent pairs, finds index in the list of
    the pair such that the base to the power of the exponent is largest
    and the natural logarithm of the base to the power of the exponent
    for that pair.
    
    Args:
        Required positional:
        exp_pairs (List of 2-tuples of ints): The base exponent
                pairs.
    
    Returns:
    2-tuple whose index 0 contains the natural logarithm of the
    largest base to the power of the exponent of all base exponent
    pairs in exp_pairs, and whose index 1 contains the index in
    exp_pairs giving rise to this value (if there are several that
    give the same value for that calculation, chooses the smallest
    index).
    
    Uses the fact that the log function is a strictly increasing
    function for positive values, so for two positive numbers
    a and b, if log(a) > log(b) then a > b
    """
    res = (-float("inf"),)
    for i, pair in enumerate(exp_pairs):
        res = max(res, (pair[1] * math.log(pair[0]), i))
    return res

def largestExponential(
    doc: str="project_euler_problem_data_files/0099_base_exp.txt",
    rel_package_src: bool=True,
) -> int:
    """
    Solution to Project Euler #99

    For a .txt file at relative or absolute location doc
    containing base exponent pairs (where the pairs are
    separated from each other by line breaks- '\\n'- and the
    base and exponent of each pair are separated by a comma,
    with the base before the comma and exponent after the
    comma), finds the line number (with the first pair
    being on line 1) such that base to the power of exponent
    of that pair is largest (with lowest of the relevant
    line numbers being chosen in the case of a tie).
    
    Args:
        Optional named:
        doc (str): The relative or absolute path to the .txt
                file containing the base exponent pairs
            Default: "project_euler_problem_data_files/0099_base_exp.txt"
        rel_package_src (bool): Whether a relative path given by
                doc is relative to the current directory (False)
                or the package directory (True).
            Default: True
    
    Returns:
    The line number of the base exponent pairs in the .txt
    file at location doc for which the base to the power of
    the exponent is largest, with the lowest of the relevant
    line numbers being chosen in the case of a tie.
    """
    #since = time.time()
    exp_pairs = loadExponentPairsFromFile(doc, rel_package_src=rel_package_src)
    res = largestLogExponential(exp_pairs)[1] + 1
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 100
def arrangedProbability(min_tot: int=10 ** 12 + 1) -> int:
    """
    Solution to Project Euler #100

    Consider the set of boxes containing an integer number
    of blue discs and an integer number of red discs such
    that there are at least max(mn, 2) discs in each box and
    for each box, if two discs are randomly selected (without
    replacement) there is exactly a 1 / 2 probability that
    both discs selected are blue. This function gives the
    number of blue discs in the box in this set containing
    the smallest total number of discs.
    
     Args:
        Optional named:
        min_tot (int): The smallest number of total discs in the
                boxes that are considered.
            Default: 10 ** 12 + 1
    
    Returns:
    The number of blue discs in the box with the smallest
    number of discs such that the total number of discs in
    that box is at least mn and the probability that a random
    selection of two discs from that box (without replacement)
    results in two blue discs is exactly 1 / 2.
    
    Outline of proof and rationale of approach:
    If we let x represent the total number of discs and
    y represents the number of blue discs, the problem
    can be restated as finding the smallest strictly
    positive integers x, y such that x > mn and:
        (x / y) * ((x - 1) / (y - 1)) = 2
    By rearranging and making the substitutions
    x' = (2 * x - 1), y' = (2 * y - 1)
    we can get:
        (2 * x) * (2 * x - 2) = 2 * (2 * y) * (2 * y - 2)
        (x' + 1) * (x' - 1) = 2 * (y' + 1) * (y' - 1)
        x' ** 2 - 1 = 2 * y' ** 2 - 2
        x' ** 2 - 2 * y' ** 2 = -1
    This is the negative Pell's equation. We are therefore
    searching for the smallest solution to the negative
    Pell's equation such that x' and y' are both strictly
    positive and odd and x' > (2 * mn - 1). The corresponding
    value of y (i.e. (y' + 1) / 2) then gives the solution.
    We can further simplify the calculation by noting that
    any integer solution to Pell's negative equation as
    expressed above in terms of x' and y' for D = 2, taking
    the equation modulo 2 implies that x' ** 2 is odd and
    so x' is odd, and then taking modulo 4, any square is
    0 or 1 modulo 4 and since x' must be odd, x' ** 2 must
    be 1 modulo 4. Therefore, 2 * y' ** 2 must be 2 modulo
    4 so y' ** 2 must be 1 modulo 4, implying that y' is odd.
    Therefore, any solution to Pell's negative equation
    must have both x' and y' as both odd, so the condition
    that x' and y' are both odd does not need to be separately
    checked.
    """
    #since = time.time()
    target_x_ = 2 * min_tot - 1
    for x_, y_ in pellSolutionGenerator(D=2, negative=True):
        if x_ > target_x_: break
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return (y_ + 1) >> 1

##############
project_euler_num_range = (51, 100)

def evaluateProjectEulerSolutions1to50(eval_nums: Optional[Set[int]]=None) -> None:
    if not eval_nums:
        eval_nums = set(range(project_euler_num_range[0], project_euler_num_range[1] + 1))
    
    if 51 in eval_nums:
        since = time.time()
        res = smallestPrimeDigitReplacementsPrime(
            family_min_n_primes=8,
            base=10,
            n_dig_max=8,
            ans_in_family=True,
        )
        print(f"Solution to Project Euler #51 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 52 in eval_nums:
        since = time.time()
        res = permutedMultiples(n_permutes=6, n_dig_mx=10, base=10)
        print(f"Solution to Project Euler #52 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 53 in eval_nums:
        since = time.time()
        res = combinatoricSelections(n_mn=1, n_mx=100, cutoff=10 ** 6)
        print(f"Solution to Project Euler #53 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 54 in eval_nums:
        since = time.time()
        res = numberOfPokerHandsWon(
            hand_file="project_euler_problem_data_files/p054_poker.txt",
            rel_package_src=True,
        )
        print(f"Solution to Project Euler #54 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 55 in eval_nums:
        since = time.time()
        res = countLychrelNumbers(n_max=10 ** 4, iter_cap=50, base=10)
        print(f"Solution to Project Euler #55 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 56 in eval_nums:
        since = time.time()
        res = powerfulDigitSum(a_mx=99, b_mx=99, base=10)
        print(f"Solution to Project Euler #56 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 57 in eval_nums:
        since = time.time()
        res = squareRootTwoConvergents(n_expansions=1000, base=10)
        print(f"Solution to Project Euler #57 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 58 in eval_nums:
        since = time.time()
        res = spiralPrimes(target_ratio=10)
        print(f"Solution to Project Euler #58 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 59 in eval_nums:
        since = time.time()
        res = xorDecryption(
            doc="project_euler_problem_data_files/p059_cipher.txt",
            key_len=3,
            auto_select=True,
            n_candidates=3,
            rel_package_src=True,
        )
        print(f"Solution to Project Euler #59 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 60 in eval_nums:
        since = time.time()
        res = minimumPrimePairSetsSum(n_pair=5, base=10)
        print(f"Solution to Project Euler #60 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 61 in eval_nums:
        since = time.time()
        res = cyclicalFigurateNumbersSum(n_dig=4, k_min=3, k_max=8, base=10)
        print(f"Solution to Project Euler #61 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 62 in eval_nums:
        since = time.time()
        res = smallestWithMNthPowerPermutations(m=5, n=3, base=10)
        print(f"Solution to Project Euler #62 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 63 in eval_nums:
        since = time.time()
        res = powerfulDigits(base=10)
        print(f"Solution to Project Euler #63 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 64 in eval_nums:
        since = time.time()
        res = sqrtCFCycleLengthOddTotal(mx=10 ** 4)
        print(f"Solution to Project Euler #64 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 65 in eval_nums:
        since = time.time()
        res = convergentENumeratorDigitSum(n=100, base=10)
        print(f"Solution to Project Euler #65 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 66 in eval_nums:
        since = time.time()
        res = pellLargestFundamentalSolution(D_max=1000)
        print(f"Solution to Project Euler #66 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 67 in eval_nums:
        since = time.time()
        res = triangleMaxSumFromFile(
            triangle_doc="project_euler_problem_data_files/p067_triangle.txt",
            rel_package_src=True,
        )
        print(f"Solution to Project Euler #67 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 68 in eval_nums:
        since = time.time()
        res = magic5gonRing()
        print(f"Solution to Project Euler #68 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 69 in eval_nums:
        since = time.time()
        res = totientMaximum(n_max=10 ** 6)
        print(f"Solution to Project Euler #69 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 70 in eval_nums:
        since = time.time()
        res = totientPermutation(num_mn=2, num_mx=10 ** 7 - 1, base=10)
        print(f"Solution to Project Euler #70 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 71 in eval_nums:
        since = time.time()
        res = orderedFractions(frac=(3, 7), max_denom=10 ** 6)
        print(f"Solution to Project Euler #71 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 72 in eval_nums:
        since = time.time()
        res = countingFractions(max_denom=10 ** 6)
        print(f"Solution to Project Euler #72 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 73 in eval_nums:
        since = time.time()
        res = countingFractionsRange(
            lower_frac=(1, 3),
            upper_frac=(1, 2),
            max_denom=12 * 10 ** 3,
        )
        print(f"Solution to Project Euler #73 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 74 in eval_nums:
        since = time.time()
        res = countDigitFactorialChains(chain_len=60, n_max=10 ** 6 - 1, base=10)
        print(f"Solution to Project Euler #74 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 75 in eval_nums:
        since = time.time()
        res = countUniquePythagoreanTripleSums(n_max=15 * 10 ** 5)
        print(f"Solution to Project Euler #75 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 76 in eval_nums:
        since = time.time()
        res = partitionFunctionNontrivial(n=100)
        print(f"Solution to Project Euler #76 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 77 in eval_nums:
        since = time.time()
        res = primeSummations(target_count=5000, batch_size=100)
        print(f"Solution to Project Euler #77 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 78 in eval_nums:
        since = time.time()
        res = coinPartitions(div=10 ** 6)
        print(f"Solution to Project Euler #78 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 79 in eval_nums:
        since = time.time()
        res = passcodeDerivation(
            doc="project_euler_problem_data_files/0079_keylog.txt",
            rel_package_src=True,
        )
        print(f"Solution to Project Euler #79 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 80 in eval_nums:
        since = time.time()
        res = squareRootDigitalExpansionSum(num_max=100, n_dig=100, base=10)
        print(f"Solution to Project Euler #80 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 81 in eval_nums:
        since = time.time()
        res = gridPathTwoWayFromFile(
            doc="project_euler_problem_data_files/0081_matrix.txt",
            rel_package_src=True,
        )
        print(f"Solution to Project Euler #81 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 82 in eval_nums:
        since = time.time()
        res = gridPathThreeWayFromFile(
            doc="project_euler_problem_data_files/0082_matrix.txt",
            alg="findShortestPath",
            bidirectional=True,
            rel_package_src=True,
        )
        print(f"Solution to Project Euler #82 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 83 in eval_nums:
        since = time.time()
        res = gridPathFourWayFromFile(
            doc="project_euler_problem_data_files/0083_matrix.txt",
            alg="findShortestPath",
            bidirectional=True,
            rel_package_src=True,
        )
        print(f"Solution to Project Euler #83 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 84 in eval_nums:
        since = time.time()
        res = monopolyOddsMostVisited(
            n_dice_faces=4,
            n_dice=2,
            n_double_jail=3,
            jail_resets_doubles=False,
            n_return=3,
        )
        print(f"Solution to Project Euler #84 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 85 in eval_nums:
        since = time.time()
        res = countingRectangles(target_count=2 * 10 ** 6)
        print(f"Solution to Project Euler #85 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 86 in eval_nums:
        since = time.time()
        res = integerMinCuboidRoute(
            target_count=10 ** 6,
            increment=500,
        )
        print(f"Solution to Project Euler #86 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 87 in eval_nums:
        since = time.time()
        res = countPrimePowerNTuples(
            mx_sum=5 * 10 ** 7 - 1,
            mn_pow=2,
            mx_pow=4,
        )
        print(f"Solution to Project Euler #87 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 88 in eval_nums:
        since = time.time()
        res = productSumNumbers(k_mn=2, k_mx=12000)
        print(f"Solution to Project Euler #88 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 89 in eval_nums:
        since = time.time()
        res = romanNumeralsSimplificationScoreFromFile(
            doc="project_euler_problem_data_files/0089_roman.txt",
            rel_package_src=True,
        )
        print(f"Solution to Project Euler #89 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 90 in eval_nums:
        since = time.time()
        res = polyhedraNumberedFacesFormingSquareNumbersCount(
            n_polyhedra=2,
            polyhedron_n_faces=6,
            base=10,
            interchangeable=({6, 9},),
        )
        print(f"Solution to Project Euler #90 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 91 in eval_nums:
        since = time.time()
        res = countRightTrianglesWithIntegerCoordinates(x_mx=50, y_mx=50)
        print(f"Solution to Project Euler #91 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 92 in eval_nums:
        since = time.time()
        res = squareDigitChains(num_mx=10 ** 7 - 1)
        print(f"Solution to Project Euler #92 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 93 in eval_nums:
        since = time.time()
        res = arithmeticExpressions(n_num=4, num_mn=1, num_mx=9)
        print(f"Solution to Project Euler #93 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 94 in eval_nums:
        since = time.time()
        res = almostEquilateralTriangles(perimeter_max=10 ** 9)
        print(f"Solution to Project Euler #94 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 95 in eval_nums:
        since = time.time()
        res = amicableChains(num_mx=10 ** 6)
        print(f"Solution to Project Euler #95 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 96 in eval_nums:
        since = time.time()
        res = sudokusSolutionUpperLeftSumFromFile(
            sudoku_doc="project_euler_problem_data_files/p096_sudoku.txt",
            rel_package_src=True,
        )
        print(f"Solution to Project Euler #96 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 97 in eval_nums:
        since = time.time()
        res = largeNonMersennePrimeLastDigits(
            mult=28433,
            a=2,
            b=7830457,
            add=1,
            n_tail_dig=10,
            base=10,
        )
        print(f"Solution to Project Euler #97 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 98 in eval_nums:
        since = time.time()
        res = anagramicSquares(
            doc="project_euler_problem_data_files/0098_words.txt",
            base=10,
            rel_package_src=True,
        )
        print(f"Solution to Project Euler #98 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 99 in eval_nums:
        since = time.time()
        res = largestExponential(
            doc="project_euler_problem_data_files/0099_base_exp.txt",
            rel_package_src=True,
        )
        print(f"Solution to Project Euler #99 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 100 in eval_nums:
        since = time.time()
        res = arrangedProbability(min_tot=10 ** 12 + 1)
        print(f"Solution to Project Euler #100 = {res}, calculated in {time.time() - since:.4f} seconds")


if __name__ == "__main__":
    eval_nums = {90}
    evaluateProjectEulerSolutions1to50(eval_nums)