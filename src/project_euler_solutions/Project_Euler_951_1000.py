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
import random
import time

import numpy as np

from collections import deque, defaultdict
from sortedcontainers import SortedDict, SortedList, SortedSet

from data_structures.fractions import CustomFraction
from data_structures.prime_sieves import PrimeSPFsieve, SimplePrimeSieve
from data_structures.fenwick_tree import FenwickTree

from algorithms.number_theory_algorithms import gcd, lcm, isqrt, integerNthRoot, PrimeModuloCalculator
from algorithms.pseudorandom_number_generators import blumBlumShubPseudoRandomGenerator
from algorithms.string_searching_algorithms import KnuthMorrisPratt
#from algorithms.geometry_algorithms import determinant
from algorithms.continued_fractions_and_Pell_equations import generalisedPellSolutionGenerator
from algorithms.farey_sequences import fareySequence

# Problem 951

def gameOfChanceNumberOfFairConfigurationsBruteForce(n: int) -> int:

    def configGenerator(n: int) -> Generator[int, None, None]:

        def recur(curr: int, z_remain: int, o_remain: int) -> Generator[int, None, None]:
            if not o_remain:
                yield curr << z_remain
                return
            elif not z_remain:
                yield ((curr + 1) << o_remain) - 1
                return
            yield from recur(curr << 1, z_remain - 1, o_remain)
            yield from recur((curr << 1) + 1, z_remain, o_remain - 1)
            return
        yield from recur(0, n, n)
    """
    def isFairRun(length: int) -> bool:
        res = 0
        for n_pair in range((length >> 1) + 1):
            n_single = length - (n_pair << 1)
            term = math.comb(n_pair + n_single, n_pair)
            res += term if n_pair & 1 else -term
        return not res

    fair_run_lens = []
    fair_run_lens_set = set()
    for run_len in range(2, n + 1, 3):
        if run_len > n: break
        if (not fair_run_lens or fair_run_lens[-1] != run_len) and isFairRun(run_len):
            fair_run_lens.append(run_len)
            fair_run_lens_set.add(run_len)
    """

    res = 0
    hlf = CustomFraction(1, 2)
    for num in configGenerator(n):
        curr = [[CustomFraction(1, 1), CustomFraction(0, 1)], [CustomFraction(0, 1), CustomFraction(0, 1)], [CustomFraction(0, 1), CustomFraction(0, 1)]]
        num2 = num
        d = num2 & 1
        num2 >>= 1
        #print(format(num, "b"))
        #print(curr)
        for _ in range(1, n << 1):
            nxt_d = num2 & 1
            num2 >>= 1
            if nxt_d == d:
                curr[1][0] += hlf * curr[0][1]
                curr[1][1] += hlf * curr[0][0]
                curr[2][0] += hlf * curr[0][1]
                curr[2][1] += hlf * curr[0][0]
                #print(curr)
            else:
                curr[1][0] += curr[0][1]
                curr[1][1] += curr[0][0]
            d = nxt_d
            curr = [*curr[1:], [CustomFraction(0, 1), CustomFraction(0, 1)]]
            #print(curr)
        curr[1][0] += curr[0][1]
        curr[1][1] += curr[0][0]
        #print(curr)
        res += (curr[1][0] == curr[1][1])
        """
        if curr[1][0] != curr[1][1]:
            #print(format(num, "b").zfill(n << 1))
            d0 = None
            run_len = 0
            num2 = num
            for _ in range(n << 1):
                d = num2 & 1
                num2 >>= 1
                if d == d0:
                    run_len += 1
                    continue
                if run_len in fair_run_lens_set:
                    break
                d0 = d
                run_len = 1
            else: 
                if (run_len not in fair_run_lens_set):
                    continue
            print(format(num, "b").zfill(n << 1))
            print(curr)
        """
    return res




def gameOfChanceNumberOfFairConfigurations(n: int=26) -> int:

    #def evenFibonacciGenerator() -> Generator[Tuple[int, int], None, None]:
    #    arr = [1, 1]
    #    idx = 2
    #    while True:
    #        arr = [arr[1], sum(arr)]
    #        if not arr[1] & 1:
    #            yield (idx, arr[1])
    #        idx += 1
    #    return
    
    def isFairRun(length: int) -> bool:
        res = -1
        for n_pair in range(1, (length >> 1) + 1):
            n_single = length - (n_pair << 1)
            # Last is a pair
            term = math.comb(n_pair + n_single - 1, n_pair - 1) << (n_pair - 1)
            # Last is single
            if n_single:
                term += math.comb(n_pair + n_single - 1, n_pair) << (n_pair)
            res += term if n_pair & 1 else -term
        print(length, res)
        return not res
    
    fair_run_lens = []
    fair_run_lens_set = set()
    for run_len in range(1, n + 1):
        if run_len > n: break
        if (not fair_run_lens or fair_run_lens[-1] != run_len) and isFairRun(run_len):
            fair_run_lens.append(run_len)
            fair_run_lens_set.add(run_len)
    print(fair_run_lens)
    
    memo = {}
    def unfairCount(n_first_color: int, n_other_color: int) -> int:

        def recur(curr_remain: int, other_remain: int) -> int:
            if not curr_remain:
                return int(not other_remain)
            if not other_remain:
                return int(curr_remain not in fair_run_lens_set)
            args = (curr_remain, other_remain)
            if args in memo.keys(): return memo[args]
            res = 0
            for run_len in range(1, curr_remain + 1):
                if run_len in fair_run_lens_set: continue
                #if j < len(fair_run_lens) and fair_run_lens[j] == run_len:
                #    j += 1
                #    continue
                res += recur(other_remain, curr_remain - run_len)
            memo[args] = res
            return res

        res = recur(n_first_color, n_other_color)
        #print(f"n_first_color = {n_first_color}, n_other_color = {n_other_color}, res = {res}")
        return res

    res = 0
    for run_len in fair_run_lens:
        #print(f"run length = {run_len}")
        n_red = n
        n_black = n - run_len
        res += unfairCount(n_red, n_black)
        n_red -= 1
        for tail_len in range((n_red + n_black) + 1):
            ans = 0
            for n_black_tail in range(min(n_black, tail_len) + 1):
                n_black_head = n_black - n_black_tail
                n_red_tail = tail_len - n_black_tail
                n_red_head = n_red - n_red_tail
                if min(n_red_tail, n_red_head) < 0: continue
                term = math.comb(n_black_head + n_red_head, n_red_head) * unfairCount(n_red_tail, n_black_tail)
                ans += term
                #print(f"n_black_head = {n_black_head}, n_red_head = {n_red_head}, n_black_tail = {n_black_tail}, n_red_tail = {n_red_tail}, term = {term}")
            #mult = 1 + (((tail_len) << 1) < (n_red + n_black))
            res += ans
        #print(f"current result = {res << 1}")
    return res << 1

# Problem 952
def factorialPrimeFactorisation(num: int, ps: Optional[SimplePrimeSieve]=None) -> Dict[int, int]:
    if ps is None:
        ps = SimplePrimeSieve(num)
    else:
        ps.extendSieve(num)
    res = {}
    for p in ps.p_lst:
        if p > num: break
        ans = 0
        num2 = num
        while num2:
            num2 //= p
            ans += num2
        res[p] = ans
    return res

def moduloFactorialMultiplicativeOrder(p: int=10 ** 9 + 7, n: int=10 ** 7, res_md: Optional[int]=10 ** 9 + 7) -> int:
    multMod = (lambda x, y: x * y) if res_md is None else (lambda x, y: (x * y) % res_md)

    fact_pf = factorialPrimeFactorisation(n, ps=None)
    print(2, fact_pf[2])
    print(len(fact_pf.keys()), max(fact_pf.keys()))
    res = 1
    carmichael_pf = {}
    ps2 = PrimeSPFsieve(max(fact_pf.keys()) >> 1)
    print(f"finished creating prime sieve")
    for p2, f in fact_pf.items():
        #phi = p2 ** (f - 1) * (p2 - 1)
        
        if p2 == 2:
            f2 = f - 1 - (f >= 3)
            
            carmichael_pf[p2] = max(carmichael_pf.get(p2, 0), f2)
            continue
        if f > 1: carmichael_pf[p2] = max(carmichael_pf.get(p2, 0), f - 1)
        pf2 = ps2.primeFactorisation(p2 >> 1)
        for p3, f2 in pf2.items():
            if p3 == 2:
                #print(f"p3 = {p3}, f2 = {f2}")
                carmichael_pf[2] = max(carmichael_pf.get(2, 0), f2 + 1)
            else: carmichael_pf[p3] = max(carmichael_pf.get(p3, 0), f2)
        #print(f"pow2 = {pow2}")
        #carmichael_pf[2] = lcm(carmichael_pf.get(2, 1), pow2)
        #print(p2, phi)
        #res = lcm(res, phi)
    #print(len(carmichael_pf), carmichael_pf)
    over_nums = {2: 1, 3: 1, 5: 1, 43: 1, 109: 1}
    res = 1
    for p2, f in carmichael_pf.items():
        if p2 == 2:
            f2 = max(0, f - (over_nums.get(p2, 0) if n >= 8 else 0))
        elif p2 == 5:
            f2 = max(0, f - (over_nums.get(p2, 0) if (p2 * 2 == n or p2 * 3 <= n) else 0))
        else: f2 = max(0, f - (over_nums.get(p2, 0) if p2 * 2 <= n else 0))
        #print(p2, f2)
        res = (res * pow(p2, f2, res_md)) % res_md
    return res

    res0 = 1
    n_fact = math.factorial(n)
    for p2, f in carmichael_pf.items():
        res0 *= p2 ** f
    
    print(f"carmichael function value = {res0}, pf = {carmichael_pf}")
    div = 1
    div_pf = {}
    res = res0
    res_pf = dict(carmichael_pf)
    #print(f"factorial pf = {pf}")
    for p2, f in carmichael_pf.items():
        print(f"p2 = {p2}")
        if p2 not in fact_pf.keys(): continue # ? is this possible
        p_pow_md = p2 ** fact_pf[p2]
        #print(f"p_pow_md = {p_pow_md}")
        #md_phi = (p2 - 1) * p2 ** (pf[p2] - 1)
        #print(f"md_phi = {md_phi}")
        #exp = 1
        #for p3, f2 in res_pf.items():
        #    exp = (exp * pow(p3, f2, md_phi)) % md_phi
        #p2_inv = pow(p2, md_phi - 2, md_phi)
        #p_pow = p2 ** f
        #print(f"exp = {exp}, p2_inv = {p2_inv}")
        exp = res
        for _ in reversed(range(f)):
            #exp = (exp * p2_inv) % md_phi
            exp //= p2
            print(f"p = {p}, exp = {exp}, p_pow_md = {p_pow_md}")
            if pow(p, exp, n_fact) != 1:
                break
            res_pf[p2] -= 1
            res //= p2
            div *= p2
            div_pf[p2] = div_pf.get(p2, 0) + 1
            print(f"update to overestimate factor:")
            print(div_pf)
        """
        for _ in range(f):
            if pow(p, res // p2, n_fact) != 1:
                break
            res //= p2
            div *= p2
            div_pf[p2] = div_pf.get(p2, 0) + 1
            print(f"update to overestimate factor:")
            print(div_pf)
        """
    print(res_pf)
    print(f"carmichael function overestimate by factor of {div} with prime factorisation {div_pf}")
    #for p2, f in res_pf.items():
    #    res = multMod(res, pow(p2, f, res_md))
    #for f2 in reversed(range(f)):
    #    if pow(p, p2 ** f2, n_fact) != 1:
    #        break
    #else: continue
    #print(p2, f2 + 1, f)
    #res = multMod(res, pow(p2, f2 + 1, res_md))
    return res % res_md

# Problem 953
def factorisationNimPlayerOneLoses(n_max: int=10 ** 14, res_md: Optional[int]=10 ** 9 + 7) -> int:
    """
    Solution to Project Euler #953
    """
    addMod = (lambda x, y: x + y) if res_md is None else (lambda x, y: (x + y) % res_md)
    n_max_rt = isqrt(n_max)
    ps = SimplePrimeSieve(n_max_rt)

    def squareSum(n: int) -> int:
        return (n * (n + 1) * (2 * n + 1)) // 6
    
    def primeCheck(num: int) -> bool:
        return ps.millerRabinPrimalityTestWithKnownBounds(num, max_n_additional_trials_if_above_max=10)[0]
    
    def recur(n_p_remain: int, curr: int, curr_xor: int, prev_p_idx: int=0) -> Generator[int, None, None]:
        if n_p_remain == 1:
            for p, num in ((curr_xor, curr), (curr_xor ^ 2, curr << 1)):
                num2 = num * p
                if p > ps.p_lst[prev_p_idx] and num2 < n_max and primeCheck(p):
                    yield num2
            #if curr << 1 <= n_max and curr_xor > ps.p_lst[prev_p_idx] and primeCheck(curr_xor ^ 2):
            #    yield curr * (curr_xor ^ 2) << 1
            return
        p_mx = integerNthRoot((n_max - 1) // curr, n_p_remain)
        for p_idx in range(prev_p_idx + 1, len(ps.p_lst)):
            p = ps.p_lst[p_idx]
            
            if p > p_mx: break
            if not prev_p_idx:
                print(f"p1 = {p}")
            yield from recur(n_p_remain - 1, curr * p, curr_xor ^ p, prev_p_idx=p_idx)
        return
    curr = 1
    for i in range(1, len(ps.p_lst)):
        curr *= ps.p_lst[i]
        if curr > n_max: break
    res = squareSum(n_max_rt) % res_md
    print(res)
    for n_p in range(2, i, 2):
        print(f"n_p = {n_p}, max n_p = {((i - 1) >> 1) << 1}")
        for num in recur(n_p, 1, 0, prev_p_idx=0):
            #print(num)
            res = addMod(res, num * squareSum(isqrt(n_max // num)))
    return res

    """   
    print(f"stage 1")
    xor_seen = {0: SortedList([1])}
    res = 0 #squareSum(n_max_rt)
    #print(f"initial res = {res}")
    for p in ps.p_lst[1:]:
        #print(f"p = {p}")
        add_dict = {}
        for num, num3_lst in xor_seen.items():
            num2 = num ^ p
            #if num2 > p and primeCheck(num2):
            #    print(f"prime num2 = {num2}")
            #    for num3 in xor_seen[num]:
            #        res = addMod(res, (num3 * num) * squareSum(isqrt(n_max // (num3 * p))))
            add_dict.setdefault(num2, [])
            for num3 in num3_lst:
                num4 = p * num3
                if num4 > n_max: break
                add_dict[num2].append(num4)
        for num, num3_lst in add_dict.items():
            xor_seen.setdefault(num, SortedList())
            for num3 in num3_lst:
                xor_seen[num].add(num3)
        #print(xor_seen)
        #print(res)
    #print(xor_seen)
    print("stage 2")
    print("zero bitwise xor")
    print(xor_seen[0])
    for num3 in xor_seen[0]:
        res = addMod(res, num3 * squareSum(isqrt(n_max // num3)))
    print(res)
    print("2 bitwise xor")
    for num3 in xor_seen.get(2, []):
        if num3 * 2 > n_max: break
        res = addMod(res, (num3 << 1) * squareSum(isqrt(n_max // (num3 << 1))))
    print(res)
    print(f"prime bitwise xor greater than {n_max_rt}")
    for p in reversed(sorted(xor_seen.keys())):
        if p <= n_max_rt: break
        if not primeCheck(p): continue
        #print(f"p = {p}")
        #print(xor_seen[p])
        for num3 in xor_seen[p]:
            num4 = p * num3
            if num4 > n_max: break
            res = addMod(res, num4 * squareSum(isqrt(n_max // num4)))
        for num3 in xor_seen[p ^ 2]:
            num4 = p * num3 << 1
            if num4 > n_max: break
            res = addMod(res, num4 * squareSum(isqrt(n_max // num4)))
    return res
    """


# Problem 955
def isSquare(num: int) -> bool:
    return isqrt(num) ** 2 == num

def isTriangle(num: int) -> bool:
    return isSquare(8 * num + 1)

def getTriangleIndex(num: int) -> int:
    discr = 8 * num + 1
    rt = isqrt(discr)
    if rt ** 2 != discr:
        return -1
    return (rt - 1) >> 1

def findIndexOfTriangleNumberInRecurrenceBruteForce(n_triangle: int=70) -> int:

    curr = [0, 3]
    triangle_idx = 0
    for idx in itertools.count(0):
        if not isTriangle(curr[-1]):
            curr = [curr[-1], 2 * curr[-1] - curr[-2] + 1]
            continue
        print(f"triangle {triangle_idx} with value {curr[-1]} at position {idx}")
        triangle_idx += 1
        if triangle_idx == n_triangle:
            break
        curr = [curr[-1], curr[-1] + 1]
    
    return curr[-1]


def findIndexOfTriangleNumberInRecurrence(n_triangle: int=70) -> int:
    t_idx = 2
    
    def calculateFactorBeforeTarget(num: int, target: int) -> int:
        for d in reversed(range(2, target)):
            if not num % d:
                return d
        return 1
    res = 0
    print(f"triangle 1 (t_idx {t_idx}, value {(t_idx * (t_idx + 1)) >> 1}) at position = {res}")
    for i in range(2, n_triangle + 1):
        
        n_pow2 = 0
        m1 = t_idx
        while not m1 & 1:
            m1 >>= 1
            n_pow2 += 1
        m2 = t_idx + 1
        while not m2 & 1:
            m2 >>= 1
            n_pow2 += 1
        
        opt1 = calculateFactorBeforeTarget(m1 * m2, (1 << (n_pow2 >> 1)) * isqrt((1 << (n_pow2 & 1)) * m1 * m2 - 1)) * 2
        #print(f"opt2 target = {isqrt(m1 * m2 - 1) // (1 << (n_pow2  >> 1))}, m1 * m2 = {m1 * m2}")
        opt2 = calculateFactorBeforeTarget(m1 * m2, isqrt(m1 * m2 - 1) // (1 << ((n_pow2 + 1) >> 1))) * (1 << (n_pow2 + 1))
        #print(m1, m2, n_pow2, opt1, opt2)
        if opt1 > opt2 or (4 * t_idx * (t_idx + 1) // opt2) - opt2 <= 2:
            d1 = opt1
        else: d1 = opt2
        #opt2_comp = 4 * t_idx * (t_idx + 1) // opt2
        #d1 = max(opt1, opt2) if opt2_comp - opt2 > 2 else opt1
        d2 = 4 * t_idx * (t_idx + 1) // d1
        #print(4 * t_idx * (t_idx + 1), d1, d2)
        n_ = (d1 + d2) >> 1
        m_ = (d2 - d1) >> 1
        n = n_ >> 1
        m = m_ >> 1
        #print((n_, m_), (n, m))
        res += m
        #print(t_idx, m, n, ((m * (m + 1)) >> 1) + ((t_idx * (t_idx + 1)) >> 1), ((n * (n + 1)) >> 1) )
        t_idx = n
        print(f"triangle {i} (t_idx {t_idx}, value {(t_idx * (t_idx + 1)) >> 1}) at position = {res}")
        
        """
        d1 = 2
        d2 = 2 * t_idx * (t_idx + 1)
        print(d1, d2)
        n_ = (d1 + d2) >> 1
        m_ = (d2 - d1) >> 1
        n = n_ >> 1
        m = m_ >> 1
        print((n_, m_), (n, m))
        res += m
        t_idx = n
        """
    return res

def findIndexOfTriangleNumberInRecurrence2(n_triangle: int=70) -> int:
    t_idx = 2
    t = 3
    res = 0
    print(f"triangle 1 (t_idx {t_idx}, value {t}) at position = {res}")
    for i in range(2, n_triangle + 1):
        t2 = t
        for t_idx2 in itertools.count(t_idx + 1):
            t2 += t_idx2
            t3 = t2 - t
            #print(t_idx2, t2, t3)
            t_idx3 = getTriangleIndex(t3)
            if t_idx3 >= 0: break
        t_idx = t_idx2
        t = t2
        res += t_idx3
        print(f"triangle {i} (t_idx {t_idx}, value {(t_idx * (t_idx + 1)) >> 1}) at position = {res}")
    return res

        

# Problem 958
def gcdNumSteps(a: int, b: int) -> int:
    n_step = 0
    a, b = sorted([a, b])
    while a > 1:
        a, b = sorted([a, b - a])
        #print(a, b)
        n_step += 1
    return (a, n_step + (b - a))

# Problem 959
def integerPartitionGenerator(
    n: int,
    min_n_part: int=0,
    max_n_part: Optional[int]=None,
    min_part_size: int=1,
    max_part_size: Optional[int]=None,
) -> Generator[Dict[int, int], None, None]:

    if max_n_part is None: max_n_part = float("inf")
    if max_part_size is None: max_part_size = float("inf")

    curr = {}
    def recur(n_remain: int, n_parts: int, part_sz: int) -> Generator[Dict[int, int], None, None]:
        if not n_remain:
            if n_parts >= min_n_part:
                yield dict(curr)
            return
        if part_sz > min(n_remain, max_part_size):
            return
        n_p_mx = min(max_n_part - n_parts, n_remain // part_sz)
        #print(n_p_mx, curr)
        if n_p_mx < 0: return
        
        if part_sz == n_remain:
            if n_parts + 1 < min_n_part or n_parts >= max_n_part:
                return
            curr[part_sz] = 1
            yield dict(curr)
            curr.pop(part_sz)
            return
        yield from recur(n_remain, n_parts, part_sz + 1)
        curr[part_sz] = 0
        rmn = n_remain
        prt = n_parts
        for n_p in range(1, n_p_mx + 1):
            #print(part_sz, n_p)
            curr[part_sz] += 1
            rmn -= part_sz
            prt += 1
            yield from recur(rmn, prt, part_sz + 1)
        curr.pop(part_sz)
        return
    yield from recur(n, 0, min_part_size)
    return

def numberOfFirstTimeReturningPathsBruteForce(n_steps: int, l_step_len: int, r_step_len: int) -> int:
    g = gcd(l_step_len, r_step_len)
    l = l_step_len * r_step_len // g
    n0_l = l // l_step_len
    n0_r = l // r_step_len
    n0 = n0_l + n0_r
    mx, r = divmod(n_steps, n0)
    if r: return 0
    unit = (n0, n0_r)
    res = 0
    for part in integerPartitionGenerator(mx):
        term = 1
        tot = 0
        for num, f in part.items():
            tot += f
            term *= math.comb(tot, f) * math.comb(unit[0] * num, unit[1] * num) ** f
        res += term if tot & 1 else -term
    return res

def determinant(
    mat: List[List[Union[int, float]]]
) -> Union[int, float]:
    """
    Calculates the determinant of a square matrix mat.

    Args:
        Required positional:
        mat (list of lists of real numeric values): The matrix
                whose determinant is to be calculated, expressed
                as a list of the rows of the matrix. This is
                required be a square matrix (i.e. all of the
                lists within mat are the same length as each
                other and mat itself).
    
    Returns:
    Real numeric value (int or float) giving the determinant of
    the matrix mat.
    """
    n = len(mat)
    cols = SortedList(range(n))
    def recur(start_row_idx: int) -> Union[int, float]:
        if len(cols) == 1:
            return mat[start_row_idx][cols[0]]
        mult = 1
        res = 0
        for i in range(len(cols)):
            col_idx = cols.pop(i)
            #mult2 = ((not col_idx & 1) << 1) - 1
            mult2 = mult * mat[start_row_idx][col_idx]
            if mult2:
                res += mult * mat[start_row_idx][col_idx] *\
                        recur(start_row_idx + 1)
            cols.add(col_idx)
            mult *= -1
        return res
    return recur(0)

def numberOfFirstTimeReturningPaths(n_steps: int, l_step_len: int, r_step_len: int) -> int:
    # Using Lemma 10.7.2 from https://www.mat.univie.ac.at/~kratt/artikel/encylatt.pdf
    # Using Proposition 2.5 of https://ece.iisc.ac.in/~parimal/2020/spqt/lecture-27.pdf
    if n_steps <= 0: return 0
    def pathCount(n_l_step: int, n_r_step: int) -> int:
        return math.comb(n_l_step + n_r_step, n_l_step)

    g = gcd(l_step_len, r_step_len)
    l = l_step_len * r_step_len // g
    n0_l = l // l_step_len
    n0_r = l // r_step_len
    n0 = n0_l + n0_r
    mx, r = divmod(n_steps, n0)
    if r: return 0
    #unit = (n0, n0_r)
    #res = 0
    mat = [[pathCount(n0_l, n0_r), 1] if mx > 1 else [pathCount(n0_l, n0_r)]]
    mat[0].extend([0] * (mx - 2))
    #mat = []
    for i in range(1, mx):
        mat.append([pathCount((i + 1) * n0_l, (i + 1) * n0_r)])
        for j in range(1, mx):
            mat[-1].append(mat[-2][j - 1])
        #mat.append([])
        #for j in range(i + 1):
        #    mat[-1].append(pathCount((i - j) * n0_l, (i - j) * n0_r))
        #for j in range(i + 1, mx):
        #    mat[-1].append(0)
    #mat.append([])
    #for j in range(mx):
    #    mat[-1].append(pathCount((mx - j) * n0_l, (mx - j) * n0_r))
    #print(mat)
    #mat2 = np.array(mat, dtype=int)
    #res = np.linalg.det(mat2)
    res = determinant(mat)
    #print(res)
    if not len(mat) & 1: res = -res
    return res

def randomWalkNumberOfUniquePointsOverPathLengthInfLimit(l_step_len: int=89, r_step_len: int=97, eps: float=10 ** -10) -> float:
    """
    Solution to Project Euler #159
    """
    # Using Lemma 10.7.2 from https://www.mat.univie.ac.at/~kratt/artikel/encylatt.pdf
    # Using Proposition 2.5 of https://ece.iisc.ac.in/~parimal/2020/spqt/lecture-27.pdf
    
    #l = lcm(l_step_len, r_step_len)
    #n1 = l // l_step_len
    #n2 = l // r_step_len
    #n = n1 + n2
    #res = 0
    inv_eps = math.ceil(1 / eps)
    n_bin_dig0 = inv_eps.bit_length()

    memo = {}
    def pathCount(n_l_step: int, n_r_step: int) -> int:
        args = (n_l_step, n_r_step)
        if args in memo.keys(): return memo[args]
        res = math.comb(n_l_step + n_r_step, n_l_step)
        memo[args] = res
        return res
    
    g = gcd(l_step_len, r_step_len)
    l = l_step_len * r_step_len // g
    n0_l = l // l_step_len
    n0_r = l // r_step_len
    n0 = n0_l + n0_r


    # Using https://people.rit.edu/ndcsma/pubs/CMJ_May_2002.pdf
    det_lst = [1, pathCount(n0_l, n0_r)]
    def getDeterminant(idx: int) -> int:
        for i in range(len(det_lst), idx + 1):
            res = pathCount(n0_l, n0_r) * det_lst[-1]
            for r in range(1, i):
                curr = (-1) ** (i - r) * pathCount((i - r + 1) * n0_l, (i - r + 1) * n0_r) * det_lst[r - 1]
                res += curr
            det_lst.append(res)
        return -det_lst[idx] * (-1) ** idx

    res = 1
    for i in itertools.count(1):
        n_step = i * n0
        num = getDeterminant(i)
        #num = numberOfFirstTimeReturningPathsBruteForce(n_steps=n_step, l_step_len=l_step_len, r_step_len=r_step_len)
        n_bin_dig = num.bit_length()
        if n_bin_dig > n_bin_dig0:
            pow2 = n_bin_dig - n_bin_dig0
            num = num >> pow2
        else: pow2 = 0
        ans = num * 2 ** (pow2 - n_step)
        
        res -= ans
        print(n_step, num, pow2, ans, res)
        if ans < eps: break
    return res

def randomWalksUniquePointSimulation(n_steps: int, l_step_len: int, r_step_len: int) -> int:
    curr = 0
    seen = {0}
    for _ in range(n_steps):
        step = random.choice([-l_step_len, r_step_len])
        curr += step
        seen.add(curr)
    print(len(seen) / n_steps)
    return len(seen)

# Problem 960
def stoneGameSolitaireScoresSumBruteForce(n: int=100, res_md: Optional[int]=10 ** 9 + 7) -> int:

    modAdd = (lambda x, y: x + y) if res_md is None else (lambda x, y: (x + y) % res_md)

    #curr = SortedDict({n - 1: n})
    memo = {}
    def recur(curr: List[int], nonzero_set: Set[int]) -> Tuple[int, int]:
        #print(curr, nonzero_set)
        #if not curr:
        #    return (1, 0)
        if len(nonzero_set) == 1:
            i1 = next(iter(nonzero_set))
            f1 = curr[i1]
            if f1 == 2:
                return (1, i1) if 2 * i1 == n else (0, 0)
        elif len(nonzero_set) == 2:
            i1, i2 = list(nonzero_set)
            f1 = curr[i1]
            if f1 == 1:
                f2 = curr[i2]
                if f2 == 1:
                    return (1, min(i1, i2)) if i1 + i2 == n else (0, 0)

        args = tuple(curr)
        if args in memo.keys():
            return memo[args]
        nonzero_lst = sorted(nonzero_set)
        #print(f"nonzero_lst = {nonzero_lst}")
        
        j2 = len(nonzero_lst) - 1
        i2 = nonzero_lst[j2]
        r2 = curr[i2]
        for j1 in range(len(nonzero_lst)):
            i1 = nonzero_lst[j1]
            if i1 * 2 >= n: break
            r1 = curr[i1]
            if i1 + i2 < n:
                #print("impossible 3")
                #print(i1, i2)
                res = (0, 0)
                memo[args] = res
                return res
            sub = min(r1, r2)
            r2 -= sub
            r1 -= sub
            if not r2:
                for j2 in reversed(range(j2)):
                    i2 = nonzero_lst[j2]
                    r2 = curr[i2]
                    if not r1: break
                    elif i1 + i2 < n:
                        #print("impossible 4")
                        res = (0, 0)
                        memo[args] = res
                        return res
                    sub = min(r1, r2)
                    r2 -= sub
                    r1 -= sub
                    if r2: break
        
        #nonzero_set2 = set(nonzero_set)
        res = [0, 0]
        for j1 in range(len(nonzero_lst)):
            i1 = nonzero_lst[j1]
            f1 = curr[i1]
            curr[i1] -= 1
            if i1 and not curr[i1]: nonzero_set.remove(i1)
            for j2 in reversed(range(j1 + 1, len(nonzero_lst))):
                i2 = nonzero_lst[j2]
                if i1 + i2 < n: break
                f2 = curr[i2]
                f0 = f1 * f2
                curr[i2] -= 1
                if i2 and not curr[i2]: nonzero_set.remove(i2)
                rm1_rng = (max(1, n - i2), min(n, i1 + 1))
                for rm1 in range(*rm1_rng):
                    rm2 = n - rm1
                    m1, m2 = i1 - rm1, i2 - rm2
                    if m1 and not curr[m1]: nonzero_set.add(m1)
                    curr[m1] += 1
                    if m2 and not curr[m2]: nonzero_set.add(m2)
                    curr[m2] += 1
                    #print(f"calling recur() 1 with args {args}, i1 = {i1}, i2 = {i2}, rm1 = {rm1}, rm2 = {rm2}, curr = {curr}")
                    f, s = recur(curr, nonzero_set)
                    #print(f"post recur() 1 with args {args}, curr = {curr}")
                    res[0] = modAdd(res[0], f0 * f)
                    res[1] = modAdd(res[1], f0 * (s + min(rm1, rm2) * f))
                    curr[m1] -= 1
                    if m1 and not curr[m1]: nonzero_set.remove(m1)
                    curr[m2] -= 1
                    if m2 and not curr[m2]: nonzero_set.remove(m2)
                if not curr[i2]: nonzero_set.add(i2)
                curr[i2] += 1
                
            if not curr[i1] or 2 * i1 < n:
                if i1 and not curr[i1]: nonzero_set.add(i1)
                curr[i1] += 1
                #print(f"end of iteration 1, j1 = {j1} with args = {args}. curr = {curr}")
                continue
            f0 = f1 * curr[i1]
            curr[i1] -= 1
            if i1 and not curr[i1]: nonzero_set.remove(i1)
            rm1_rng = (max(1, n - i1), (n + 1) >> 1)
            
            for rm1 in range(*rm1_rng):
                rm2 = n - rm1
                m1, m2 = i1 - rm1, i1 - rm2
                if m1 and not curr[m1]: nonzero_set.add(m1)
                curr[m1] += 1
                if m2 and not curr[m2]: nonzero_set.add(m2)
                curr[m2] += 1
                #print(f"calling recur() 2 with args {args}, i1 = {i1}, i2 = {i1}, rm1 = {rm1}, rm2 = {rm2}, curr = {curr}")
                f, s = recur(curr, nonzero_set)
                #print(f"post recur() 2 with args {args}, curr = {curr}")
                res[0] = modAdd(res[0], f0 * f)
                res[1] = modAdd(res[1], f0 * (s + min(rm1, rm2) * f))
                curr[m1] -= 1
                if m1 and not curr[m1]: nonzero_set.remove(m1)
                curr[m2] -= 1
                if m2 and not curr[m2]: nonzero_set.remove(m2)
            
            if not n & 1:
                f0 >>= 1
                rm = n >> 1
                m = i1 - rm
                if m and not curr[m]: nonzero_set.add(m)
                curr[m] += 2
                #print(f"calling recur() 3 with args {args}, i1 = {i1}, i2 = {i1}, rm1 = {rm}, rm2 = {rm}, curr = {curr}")
                f, s = recur(curr, nonzero_set)
                #print(f"post recur() 3 with args {args}, curr = {curr}")
                res[0] = modAdd(res[0], f0 * f)
                res[1] = modAdd(res[1], f0 * (s + rm * f))
                curr[m] -= 2
                if m and not curr[m]: nonzero_set.remove(m)
            if i1 and not curr[i1]: nonzero_set.add(i1)
            curr[i1] += 2
            #print(f"end of iteration2, j1 = {j1} with args = {args}. curr = {curr}")

        #print(f"returning from args = {args} with curr = {curr}")
        res = tuple(res)
        memo[args] = res
        return res
        """
        #num1, f1 = curr.peekitem(-1)
        if f1 > 1:
            if (num1 << 1) < n:
                return (0, 0)
            if len(curr) == 1 and f1 == 2:
                return (1, num1) if (num1 << 1) == n else (0, 0)
        else:
            if len(curr) == 1: return (0, 0)
            num2, f2 = curr.peekitem(-2)
            if num1 + num2 < n:
                return (0, 0)
            if len(curr) == 2 and f1 == 1 and f2 == 1:
                return (1, num2) if num1 + num2 == n else (0, 0)
        
        args = tuple((k, v) for k, v in curr.items())
        if args in memo.keys():
            return memo[args]
        i2 = 0
        res = [0, 0]
        for i1 in range(len(curr)):
            num1, f1 = curr.peekitem(i1)
            for i2 in range(max(i1 + (f1 == 1), i2), len(curr)):
                num2, f2 = curr.peekitem(i2)
                if num1 + num2 >= n: break
            else: continue
            f0 = ((f1 * (f1 - 1)) >> 1) if i1 == i2 else f1 * f2
            #print(f"num1 = {num1}, num2 = {num2}, f0 = {f0}")
            rm1_rng = (max(1, n - num2), min(n, num1 + 1))
            #print(f"rm1_rng = {rm1_rng}")
            for rm1 in range(*rm1_rng):
                rm2 = n - rm1
                m1, m2 = num1 - rm1, num2 - rm2
                for num, m in ((num1, m1), (num2, m2)):
                    curr[num] -= 1
                    if not curr[num]: curr.pop(num)
                    if m:
                        curr[m] = curr.get(m, 0) + 1
                f, s = recur()
                #print(f"curr = {curr}, f = {f}, score = {s}")
                #f0_2 = f0 if num1 != num2 or rm1 != rm2 else f0 >> 1
                res[0] = modAdd(res[0], f0 * f)
                res[1] = modAdd(res[1], f0 * (s + min(rm1, rm2) * f))
                for num, m in ((num1, m1), (num2, m2)):
                    if m:
                        curr[m] -= 1
                        if not curr[m]: curr.pop(m)
                    curr[num] = curr.get(num, 0) + 1
                #print(num1, num2, rm1, rm2, m1, m2, res)
        res = tuple(res)
        memo[args] = res
        return res
        """
    res = recur([0] * (n - 1) + [n], {n - 1})
    #print(res)
    print(len(memo))
    return res[1]

def stoneGameSolitaireScoresSum(n: int=100, res_md: Optional[int]=10 ** 9 + 7) -> int:

    modAdd = (lambda x, y: x + y) if res_md is None else (lambda x, y: (x + y) % res_md)

    def op(tup1: Tuple[int, int], tup2: Tuple[int, int]) -> Tuple[int]:
        return (tup1[0] + tup2[0], -(max(tup1[0], 0) + max(tup2[0], 0)))
    
    net_debt_bit = FenwickTree(n, (op, (0, 0)))
    net_debt_bit.update(n - 1, (n, 0))

    curr = [0] * (n - 2) + [n]
    curr_pos = {n - 1}
    curr_neg = set()

    def incrementCurrent(i: int, amount: int=1) -> None:
        val = curr[i] + amount
        if val * curr[i] <= 0:
            if curr[i] < 0: curr_neg.remove(i)
            if val > 0: curr_pos.add(i)
        curr[i] = i
        net_debt_bit.update(i, (amount, 0))
        return
    
    def decrementCurrent(i: int, amount: int=1) -> None:
        val = curr[i] - amount
        if val * curr[i] <= 0:
            if curr[i] > 0: curr_pos.remove(i)
            if val < 0: curr_neg.add(i)
        curr[i] = i
        net_debt_bit.update(i, (-amount, 0))
    
    def contributeToResult(res: List[int, int], f0: int, f: int, s: int, s2: int) -> None:
        res[0] = modAdd(res[0], f0 * f)
        res[1] = modAdd(res[1], f0 * (s + s2 * f))
        return

    def removeBatches(idx2: int, n_rm2: int, idx1: int, batch_size_max: int) -> Tuple[int, int]:
        if n - n_rm2 > idx1: return (0, 0)
        if idx1 == idx2:
            pass
    
    memo = {}
    def recur(turns_remain: int, prev: Tuple[int, int, int]=(n, 0, 0)) -> Tuple[int, int]:
        if not turns_remain:
            return (0, 0) if curr_neg else (1, 0)
        elif turns_remain == 1:
            return (0, 0) if curr_neg else (1, min(curr_pos))
        for i2 in reversed(range(min(n, prev[0] + 1))):
            if curr[i2] > 0: break
            elif curr[i2] < 0: return (0, 0)
            prev = (n, 0, 0)
        smaller_debt = net_debt_bit.query(i2 - 1)[1]
        if smaller_debt > curr[i2]:
            return (0, 0)

        args = (tuple(curr), prev)
        if args in memo.keys():
            return memo[args]

        res = [0, 0]
        rm2_mx = i2 if prev[0] > i1 else min(i2, prev[1] - 1)
        #i1_mx = i2 - (curr[i2] < 2) if i2 < prev[0] else min(i2 - 1, prev[1])
        f2 = curr[i2]
        decrementCurrent(i2)
        #if smaller_debt == curr[i2]:
        for rm2 in range(1, rm2_mx + 1):
            i1_mx = i2 - (curr[i2] < 2)
            if i2 == prev[0] and rm2_mx == prev[1]:
                i1_mx = min(i1_mx, prev[2] - 1)
            rm1 = n - rm2
            for i1 in reversed(range(max(1, rm1), i1_mx + 1)):
                f, s = removeBatches(i2, rm2, i1, batch_size_max=min(i2 // rm2, i1 // rm1)) # review- try to get a tighter constraint for batch_size_max based on the debt values
                contributeToResult(res, 1, f, s, min(rm1, rm2))
            
        """
        for m2 in reversed(range(i2)):
            if curr[m2] > 0: break
            rm2 = i2 - m2
            rm1 = n - rm2
            incrementCurrent(m2)
            for i1 in reversed(range(rm1, i1_mx + 1)):
                if i2 == prev[0] and i1 == prev[1] and rm2 > prev[2]:
                    continue
                f0 = (f2 * curr[i1])
                if i1 == i2:
                    if rm1 == rm2:
                        f0 >>= 1
                    elif rm1 > rm2: break

                m1 = i1 - rm1
                decrementCurrent(i1)
                incrementCurrent(m1)
                f, s = recur(turns_remain - 1, prev=(i2, i1, rm2))
                contributeToResult(res, f0, f, s, min(rm1, rm2))
                decrementCurrent(m1)
                incrementCurrent(i1)
            decrementCurrent(m2)
        """
        #incrementCurrent(i2)
        res = tuple(res)
        memo[args] = res
        return res

# Problem 961
def removingDigitsGamePlayerOneWinsCount(n_dig_max: int=18, base: int=10) -> int:
    """
    Solution to Project Euler #961
    """
    def numbersRepresentedByBinary(num_bin: int) -> int:
        return (base - 1) ** num_bin.bit_count()

    memo = {}
    def canWin(num_bin: int) -> bool:
        if not num_bin: return False
        args = num_bin
        if args in memo.keys(): return memo[args]
        l = num_bin.bit_length()
        #print(format(num_bin, "b"), format(num_bin >> 1, "b"), format(num_bin ^ (1 << (l - 1)), "b"))
        res = False
        bm = 0
        prev_tail = 0
        for _ in range(l):
            bm = (bm << 1) + 1
            tail = num_bin & bm
            head = num_bin ^ tail
            num_bin2 = (head >> 1) ^ prev_tail
            if not canWin(num_bin2):
                res = True
                break
            prev_tail = tail
        #res = not canWin(num_bin >> 1) or not canWin(num_bin ^ (1 << (l - 1)))
        memo[args] = res
        return res

    res = 0
    for num_bin in range(1, 1 << n_dig_max):
        if canWin(num_bin):
            term = numbersRepresentedByBinary(num_bin)
            res += term
            #print(format(num_bin, "b"), term, res)
    return res
        
# Problem 965
def expectedMinimalFractionalValueFraction(N: int) -> CustomFraction:
    
    # Review- look into the solutions in the forum that simplify the
    # expression

    #thresholds = [(CustomFraction(0, 1), 1, 0)]
    #for n in range(2, N + 1):
    #    for int_part in range(1, n):
    #        if gcd(int_part, n) != 1: continue
    #        thresholds.append((CustomFraction(int_part, n), n, int_part))
    #thresholds.sort()
    #thresholds.append((CustomFraction(1, 1),))
    #print(f"number of thresholds = {len(thresholds)}")
    #print(thresholds)
    res = CustomFraction(0, 1)
    it = iter(fareySequence(N, mn=(0, 1), mx=(1, 1)))
    curr = CustomFraction(*next(it))
    #print(curr, res)
    curr_sq = curr * curr
    sub = 0
    mult = 1
    for tup in it:
        if tup[0] == 1 or tup[1] == tup[0] + 1: print(tup)
        nxt = CustomFraction(*tup)
        nxt_sq = nxt * nxt
        #cbs = [x * sq for x, sq in zip(rng, sqs)]
        res += (CustomFraction(mult, 2) * (nxt_sq - curr_sq)) -\
                sub * (nxt - curr)
        #print(nxt, res)
        curr = nxt
        curr_sq = nxt_sq
        mult = tup[1]
        sub = tup[0]
    return res

def expectedMinimalFractionalValueFloat(N: int=10 ** 4) -> float:
    res = 0
    it = iter(fareySequence(N, mn=(0, 1), mx=(1, 1)))
    curr = CustomFraction(*next(it))
    #print(curr, res)
    curr_sq = curr * curr
    sub = 0
    mult = 1
    for tup in it:
        if tup[0] == 1 or tup[1] == tup[0] + 1: print(tup)
        nxt = CustomFraction(*tup)
        nxt_sq = nxt * nxt
        #cbs = [x * sq for x, sq in zip(rng, sqs)]
        ans = (CustomFraction(mult, 2) * (nxt_sq - curr_sq)) -\
                sub * (nxt - curr)
        res += ans.numerator / ans.denominator
        #print(nxt, res)
        curr = nxt
        curr_sq = nxt_sq
        mult = tup[1]
        sub = tup[0]
    return res

    

def expectedMinimalFractionalValue(N: int=10 ** 4, use_float: bool=True) -> float:
    """
    Solution to Project Euler #965
    """
    if use_float:
        return expectedMinimalFractionalValueFloat(N)
    res = expectedMinimalFractionalValueFraction(N)
    print(res)
    return res.numerator / res.denominator

# Problem 978
def randomWalkDistributionBruteForce(n_steps: int) -> Tuple[Dict[int, int], int]:
    if n_steps < 0: return ({}, 0)
    if not n_steps: return ({0: 1}, 1)
    if n_steps <= 2: return ({1: 1}, 1)

    curr = {(1, 1): 1}
    for i in range(3, n_steps + 1):
        print(f"i = {i}, curr size = {len(curr)}")
        prev = curr
        curr = {}
        for p, f in prev.items():
            if not p[0]:
                p2 = (abs(p[1]), p[1])
                curr[p2] = curr.get(p2, 0) + 2 * f
                continue
            p2 = (abs(p[1]), p[1] - p[0])
            curr[p2] = curr.get(p2, 0) + f
            p2 = (abs(p[1]), p[1] + p[0])
            curr[p2] = curr.get(p2, 0) + f
    res = {}
    for p, f in curr.items():
        res[p[1]] = res.get(p[1], 0) + f
    return (res, 1 << (n_steps - 2))

def randomWalkSkewnessBruteForce(n_steps: int=50) -> float:
    
    if n_steps <= 2: return 0

    curr = {(1, 1): 1}
    for i in range(3, n_steps + 1):
        print(f"i = {i}, curr size = {len(curr)}")
        prev = curr
        curr = {}
        distrib = {}
        for p, f in prev.items():
            if not p[0]:
                p2 = (abs(p[1]), p[1])
                curr[p2] = curr.get(p2, 0) + 2 * f
                distrib[p2[1]] = distrib.get(p2[1], 0) + 2 * f
                continue
            p2 = (abs(p[1]), p[1] - p[0])
            curr[p2] = curr.get(p2, 0) + f
            distrib[p2[1]] = distrib.get(p2[1], 0) + f
            p2 = (abs(p[1]), p[1] + p[0])
            curr[p2] = curr.get(p2, 0) + f
            distrib[p2[1]] = distrib.get(p2[1], 0) + f
        norm = 1 << (i - 2)
        mean = sum(x * f for x, f in distrib.items()) / norm
        var = sum(x ** 2 * f for x, f in distrib.items()) / norm - mean ** 2
        stdev = math.sqrt(var)
        cubed_expectation = sum(x ** 3 * f for x, f in distrib.items()) / norm
        skew = (cubed_expectation - 3 * mean * var - mean ** 3) / stdev ** 3
        print(f"after {i} steps, mean = {mean}, variance = {var}, skew = {skew}, cubed expectation = {cubed_expectation}")

    return skew

def randomWalkSkewness(n_steps: int=50) -> float:
    """
    Solution to Project Euler #978
    """
    # Using OEIS A006130
    if n_steps <= 2: return 0

    mean = 1
    fib1 = [1, 1]
    fib2 = [1, 1]
    for _ in range(3, n_steps + 1):
        fib1 = [fib1[1], sum(fib1)]
        fib2 = [fib2[1], 3 * fib2[0] + fib2[1]]
    var = fib1[1] - 1
    cubed_expectation = fib2[1]
    stdev = math.sqrt(var)
    skew = (cubed_expectation - 3 * mean * var - mean ** 3) / stdev ** 3
    print(f"after {n_steps} steps, mean = {mean}, variance = {var}, skew = {skew}, cubed expectation = {cubed_expectation}")
    return skew

# Problem 979
def countPolygonalTilingPaths(
    polygon_n_sides: int=7,
    n_steps: int=20,
) -> int:
    """
    Solution to Project Euler #979
    """
    # Review- try to generalise to any number of polygons meeting at each
    # vertex no less than 3, not just (as present) 3.
    adj = [{1: polygon_n_sides}, {0: 1, 1: 2}]
    counts = [1, polygon_n_sides]
    min_n_steps = [0, 1]
    mxmn_n_steps = n_steps >> 1
    remain_dict = {1: [polygon_n_sides - 3, 1, 1]}
    for idx in itertools.count(start=1):
        #print(f"idx = {idx}, adj = {adj}, remain_dict = {remain_dict}")
        #if not remain_dict: break
        if idx >= len(adj): break
        if min_n_steps[idx] >= mxmn_n_steps: continue
        if idx not in remain_dict.keys(): continue
        remain_lst = remain_dict.pop(idx)
        #if not remain_lst[0]: continue
        cnt = counts[idx]
        if remain_lst[1] == remain_lst[2]:
            for _ in range(remain_lst[0] >> 1):
                idx2 = remain_lst[1]
                idx3 = len(adj)
                adj.append({})
                min_n_steps.append(min(min_n_steps[idx], min_n_steps[idx2]) + 1)
                remain_lst[1] = idx3
                remain_dict[idx3] = [polygon_n_sides - 1, idx, idx2]
                if idx == idx2:
                    counts.append(cnt)
                    adj[idx][idx3] = 2
                    adj[idx3][idx] = 2
                    remain_dict[idx3][0] -= 1
                    continue
                counts.append(cnt << 1)
                adj[idx][idx3] = 2
                adj[idx3][idx] = 1
                
                #adj[idx3].setdefault(idx2, 0)
                #adj[idx3][idx2] += 1
                
                #remain_dict[idx3][0] -= 1
                #remain_dict[idx2][0] -= 1
                for j in range(1, 3):
                    #j = 2 - (remain_dict[idx2][1] == idx)
                    #print(f"idx = {idx}, idx2 = {idx2}, idx3 = {idx3}")
                    if remain_dict[idx2][j] == idx:
                        remain_dict[idx2][j] = idx3
                        adj[idx2][idx3] = adj[idx2].get(idx3, 0) + 1
                        remain_dict[idx2][0] -= 1
                    if remain_dict[idx3][j] == idx:
                        adj[idx3][idx2] = adj[idx3].get(idx2, 0) + 1
                        #print("subtracting")
                        remain_dict[idx3][0] -= 1
                        
                        

            if remain_lst[0] & 1:
                idx2 = remain_lst[1]
                idx3 = len(adj)
                #print(f"adding index {idx3}")
                adj.append({})
                min_n_steps.append(min(min_n_steps[idx], min_n_steps[idx2]) + 1)
                remain_dict[idx3] = [polygon_n_sides - 1, idx2, idx2]
                if idx == idx2:
                    # Closing a platonic solid
                    counts.append(1)
                    adj[idx][idx3] = 1
                    adj[idx3][idx] = polygon_n_sides
                    remain_dict.pop(idx3)
                    continue
                counts.append(cnt)
                adj[idx][idx3] = 1
                adj[idx3][idx] = 1
                #adj[idx2][idx3] = 2
                #adj[idx3][idx2] = 1 + (remain_dict[idx2][0] > 1)
                
                #adj[idx3].setdefault(idx2, 0)
                #adj[idx3][idx2] += 1
                
                
                adj[idx3][idx2] = adj[idx3].get(idx2, 0) + 2
                remain_dict[idx3][0] -= 2

                #remain_dict[idx3][0] -= 1
                #remain_dict[idx2][0] -= 2
                #remain_dict[idx3][0] -= adj[idx3][idx2] + 1
                #adj[idx2][idx3] = 1
                #adj[idx3][idx] = 0
                #adj[idx3][idx2] = 2
                for j in range(1, 3):
                    #j = 2 - (remain_dict[idx2][1] == idx)
                    if remain_dict[idx2][j] == idx:
                        remain_dict[idx2][j] = idx3
                        adj[idx2][idx3] = adj[idx2].get(idx3, 0) + 1
                        remain_dict[idx2][0] -= 1
                    #if remain_dict[idx3][j] == idx:
                    #    remain_dict[idx3][j] = idx3
                    #    adj[idx3][idx2] = adj[idx3].get(idx2, 0) + 1
                    #    remain_dict[idx3][0] -= 1
            else:
                idx2 = remain_lst[1]
                
                if idx2 == idx:
                    adj[idx2][idx2] = adj[idx2].get(idx2, 0) + 1
                    continue
                for j in range(1, 3):
                    #j = 2 - (remain_dict[idx2][1] == idx)
                    if remain_dict[idx2][j] == idx:
                        remain_dict[idx2][j] = idx2
                        adj[idx2][idx2] = adj[idx2].get(idx2, 0) + 1
                        remain_dict[idx2][0] -= 1
            continue
        for _ in range(remain_lst[0] >> 1):
            for i in range(1, 3):
                idx2 = remain_lst[i]
                idx3 = len(adj)
                adj.append({})
                min_n_steps.append(min(min_n_steps[idx], min_n_steps[idx2]) + 1)
                remain_lst[i] = idx3
                remain_dict[idx3] = [polygon_n_sides - 1, idx, idx2]
                if idx == idx2:
                    counts.append(cnt >> 1)
                    adj[idx][idx3] = 1
                    adj[idx3][idx] = 2
                    remain_dict[idx3][0] -= 1
                    continue
                counts.append(cnt)
                #adj[idx][idx3] = 1
                #adj[idx2][idx3] = 1
                #adj[idx3][idx] = 0
                #adj[idx3][idx2] = 1
                adj[idx][idx3] = 1
                adj[idx3][idx] = 1
                
                #adj[idx3][idx2] = 1
                #adj[idx2][idx3] = 1

                #remain_dict[idx2][0] -= 1
                #remain_dict[idx3][0] -= 2
                for j in range(1, 3):
                    #j = 2 - (remain_dict[idx2][1] == idx)
                    #print(f"idx = {idx}, idx2 = {idx2}, idx3 = {idx3}")
                    if remain_dict[idx2][j] == idx:
                        remain_dict[idx2][j] = idx3
                        adj[idx2][idx3] = adj[idx2].get(idx3, 0) + 1
                        remain_dict[idx2][0] -= 1
                    if remain_dict[idx3][j] == idx:
                        adj[idx3][idx2] = adj[idx3].get(idx2, 0) + 1
                        remain_dict[idx3][0] -= 1
                #if idx3 == 7:
                #    print(f"idx2 = {idx2}, idx3 = {idx3}, adj[idx2][idx3] = {adj[idx2][idx3]}, adj[idx3][idx2] = {adj[idx3][idx2]}")
        #print(remain_lst)
        if remain_lst[0] & 1:
            idx3 = len(adj)
            adj.append({})
            min_n_steps.append(min(min_n_steps[idx], min_n_steps[remain_lst[1]], min_n_steps[remain_lst[2]]) + 1)
            remain_dict[idx3] = [polygon_n_sides - 3, remain_lst[1], remain_lst[2]]
            counts.append(cnt)
            adj[idx][idx3] = 1
            adj[idx3][idx] = 1
            for i in range(1, 3):
                idx2 = remain_lst[i]
                if idx == idx2:
                    counts.append(cnt >> 1)
                    adj[idx][idx3] = 1
                    adj[idx3][idx] = 2
                    adj[idx3][idx3] = adj[idx3].get(idx3, 0) + 2
                    remain_dict[idx3][0] -= 1
                    remain_dict[idx3][i] = idx3
                    continue
                #adj[idx2][idx3] = 1
                #adj[idx3][idx2] = 1
                #remain_dict[idx2][0] -= 1
                adj[idx3][idx2] = adj[idx3].get(idx2, 0) + 1
                for j in range(1, 3):
                    #j = 2 - (remain_dict[idx2][1] == idx)
                    #print(f"idx = {idx}, idx2 = {idx2}, idx3 = {idx3}")
                    if remain_dict[idx2][j] == idx:
                        remain_dict[idx2][j] = idx3
                        adj[idx2][idx3] = adj[idx2].get(idx3, 0) + 1
                        remain_dict[idx2][0] -= 1
            continue
        else:
            idx2 = remain_lst[1]
            idx3 = remain_lst[2]
            #adj[idx2][idx3] = 1
            #adj[idx3][idx2] = 1
            
            #remain_dict[idx2][0] -= 1
            #remain_dict[idx3][0] -= 1
            for j in range(1, 3):
                #j = 2 - (remain_dict[idx2][1] == idx)
                if remain_dict[idx2][j] == idx:
                    remain_dict[idx2][j] = idx3
                    adj[idx2][idx3] = adj[idx2].get(idx3, 0) + 1
                    remain_dict[idx2][0] -= 1
                if remain_dict[idx3][j] == idx:
                    remain_dict[idx3][j] = idx2
                    adj[idx3][idx2] = adj[idx3].get(idx2, 0) + 1
                    remain_dict[idx3][0] -= 1

    print(counts)
    print(min_n_steps)
    print(adj)
    print(f"number of distinct tile types = {len(counts)}")
    print(f"total number of tiles = {sum(counts)}")

    curr = {0: 1}
    for i in range((n_steps) >> 1):
        prev = curr
        curr = {}
        for idx1, f1 in prev.items():
            for idx2, wt in adj[idx1].items():
                curr[idx2] = curr.get(idx2, 0) + f1 * wt
        print(f"after {i + 1} steps, the number of ways to get to each tile type are: {curr}")
        for idx, f in curr.items():
            if f % counts[idx]:
                print(f"for tile type {idx}, the number of such tiles {counts[idx]} does not exactly divide the number of paths of length {i} to that tile type {f}")

    res = 0
    if not n_steps & 1:
        for idx, f in curr.items():
            res += (f ** 2) // counts[idx]
        return res
    for idx1, f1 in curr.items():
        for idx2, wt in adj[idx1].items():
            res += (f1 * curr.get(idx2, 0) * wt) // counts[idx2]
    return res

# Problem 980
def stringIsNeutral(s: str) -> bool:

    counts = [0, 0, 0]
    idx_map = "xyz"
    idx_inv_map = {l: i for i, l in enumerate(idx_map)}
    for l in s:
        counts[idx_inv_map[l]] += 1
    print(counts)
    n_odd = sum(x & 1 for x in counts)
    if n_odd % 3:
        #print(f"odd count found: {counts}")
        return False
    curr_cnts = [0, 0, 0]
    n_swap = 0
    for l in s:
        j = idx_inv_map[l]
        n_swap += sum(curr_cnts[j2] for j2 in range(j + 1, 3))
        curr_cnts[j] += 1

    res = (sum(((x >> 1) & 1) for x in counts) + n_swap) & 1
    print(n_swap)
    return not bool(res)

def countGeneratedSequencesNeutralStringsBruteForce(n_max: int=10 ** 6 - 1) -> int:

    idx_map = "xyz"
    #idx_inv_map = {l: i for i, l in enumerate(idx_map)}

    memo0 = {}
    def getSequenceTerm(i: int) -> int:
        if i < 0: return 0
        if not i: return 88_888_888
        if i in memo0.keys(): return memo0[i]
        res = (8888 * getSequenceTerm(i - 1)) % 888_888_883
        memo0[i] = res
        return res
    

    memo = {}
    def generateIthString(i: int) -> str:
        if i in memo.keys(): return memo[i]
        res = []
        for j in range(50):
            res.append(idx_map[getSequenceTerm(50 * i + j) % 3])
        return "".join(res)

    res = 0
    for i in range(n_max + 1):
        s1 = generateIthString(i)
        #print(s1)
        for j in range(n_max + 1):
            s = "".join([s1, generateIthString(j)])
            b = stringIsNeutral(s)
            print(s, b)
            res += b
    return res

def countGeneratedSequencesNeutralStrings(n_max: int=10 ** 6 - 1) -> int:
    """
    Solution to Project Euler #980
    """
    idx_map = "xyz"
    idx_inv_map = {l: i for i, l in enumerate(idx_map)}

    memo0 = {}
    def getSequenceTerm(i: int) -> int:
        if i < 0: return 0
        if not i: return 88_888_888
        if i in memo0.keys(): return memo0[i]
        res = (8888 * getSequenceTerm(i - 1)) % 888_888_883
        memo0[i] = res
        return res
    

    def generateIthString(i: int) -> str:
        res = []
        for j in range(50):
            res.append(idx_map[getSequenceTerm(50 * i + j) % 3])
        return "".join(res)

    def strEncoding(s: str) -> int:
        counts_mod4 = [0, 0, 0]
        swaps_parity = False
        for l in s:
            idx = idx_inv_map[l]
            swaps_parity = (swaps_parity != (bool(sum(counts_mod4[idx + 1:]) & 1)))
            counts_mod4[idx] = (counts_mod4[idx] + 1) & 3
        res = 0
        for num in counts_mod4:
            res <<= 2
            res ^= num
        
        return (res << 1) ^ int(swaps_parity)
    
    def numberListEncoding(lst: List[int]) -> int:
        counts_mod4 = [0, 0, 0]
        swaps_parity = False
        for idx in lst:
            swaps_parity = (swaps_parity != (bool(sum(counts_mod4[idx + 1:]) & 1)))
            counts_mod4[idx] = (counts_mod4[idx] + 1) & 3
        res = 0
        for num in counts_mod4:
            res <<= 2
            res ^= num
        
        return (res << 1) ^ int(swaps_parity)
    
    f_dict = {}
    curr = 88_888_888
    for i in range(n_max + 1):
        if not i % 10000: print(i)
        num_lst = []
        for _ in range(50):
            num_lst.append(curr % 3)
            curr = (8888 * curr) % 888_888_883
        #num = strEncoding(generateIthString(i))
        #print(num_lst)
        e = numberListEncoding(num_lst) 
        f_dict[e] = f_dict.get(e, 0) + 1
    print(f_dict)
    res = 0
    for e2, f2 in f_dict.items():
        print(format(e2, "b"))
        for e1, f1 in f_dict.items():
            parity = (e1 & 1) != (e2 & 1)
            counts1_mod4 = [(e1 >> 5), (e1 >> 3) & 3, (e1 >> 1) & 3]
            counts2_mod4 = [(e2 >> 5), (e2 >> 3) & 3, (e2 >> 1) & 3]
            if len({(x + y) & 1 for x, y in zip(counts1_mod4, counts2_mod4)}) > 1: continue
            if sum(((x + y) >> 1) & 1 for x, y in zip(counts1_mod4, counts2_mod4)) & 1:
                parity = not parity
            if counts2_mod4[0] & 1 and ((counts1_mod4[1] & 1) != (counts1_mod4[2] & 1)):
                parity = not parity
            if counts2_mod4[1] & 1 and (counts1_mod4[2] & 1):
                parity = not parity
            if parity: continue
            print((format(e1, "b"), f1), (format(e2, "b"), f2))
            res += f1 * f2
    return res

# Problem 981
def countNeutralStringsWithGivenCharacterCountsBruteForce(n_x: int, n_y: int, n_z: int) -> int:
    counts = sorted([n_x, n_y, n_z])
    n_odd = sum(x & 1 for x in counts)
    if n_odd % 3: return 0
    seen = set()
    res = 0
    swap_counts = []
    for s_tup in itertools.permutations("".join(["x" * n_x, "y" * n_y, "z" * n_z])):
        s = "".join(s_tup)
        if s in seen: continue
        seen.add(s)
        counts = [0, 0, 0]
        idx_map = "xyz"
        idx_inv_map = {l: i for i, l in enumerate(idx_map)}
        for l in s:
            counts[idx_inv_map[l]] += 1
        #print(counts)
        n_odd = sum(x & 1 for x in counts)
        if n_odd % 3:
            #print(f"odd count found: {counts}")
            return False
        curr_cnts = [0, 0, 0]
        n_swap = 0
        for l in s:
            j = idx_inv_map[l]
            n_swap += sum(curr_cnts[j2] for j2 in range(j + 1, 3))
            curr_cnts[j] += 1
        swap_counts += [0] * (n_swap - len(swap_counts) + 1)
        swap_counts[n_swap] += 1
        b = (sum(((x >> 1) & 1) for x in counts) + n_swap) & 1

        res += not b
    print(f"total number of strings seen = {len(seen)}")
    print(f"swap counts: {swap_counts}")
    return res
    

def countNeutralStringsWithGivenCharacterCounts(n_x: int, n_y: int, n_z: int, res_md: Optional[int]=None) -> int:
    counts = sorted([n_x, n_y, n_z])
    if counts[0] < 0: return 0
    if counts[-1] == 0: return 1
    n_odd = sum(x & 1 for x in counts)
    if n_odd % 3: return 0
    if res_md is None: 
        if n_odd == 3:
            return (math.comb(n_x + n_y + n_z, n_z) * math.comb(n_x + n_z, n_y)) >> 1
        n_x2, n_y2, n_z2 = n_x >> 1, n_y >> 1, n_z >> 1
        if (n_x2 + n_y2 + n_z2) & 1:
            return (math.comb(n_x + n_y + n_z, n_z) * math.comb(n_x + n_y, n_y) - math.comb(n_x2 + n_y2 + n_z2, n_z2) * math.comb(n_x2 + n_y2, n_y2)) >> 1
        return (math.comb(n_x + n_y + n_z, n_z) * math.comb(n_x + n_y, n_y) + math.comb(n_x2 + n_y2 + n_z2, n_z2) * math.comb(n_x2 + n_y2, n_y2)) >> 1
        #print(math.comb(n_x + n_y + n_z, n_z) * math.comb(n_x + n_y, n_y))
    two_inv = pow(2, res_md - 2, res_md)
    #print(2, two_inv, (2 * two_inv) % res_md)
    res = ((math.comb(n_x + n_y + n_z, n_z) % res_md) * (math.comb(n_x + n_y, n_y) % res_md)) % res_md
    if n_odd == 3:
        return (res * two_inv) % res_md
    n_x2, n_y2, n_z2 = n_x >> 1, n_y >> 1, n_z >> 1
    res2 = ((math.comb(n_x2 + n_y2 + n_z2, n_z2) % res_md) * (math.comb(n_x2 + n_y2, n_y2) % res_md)) % res_md
    if (n_x2 + n_y2 + n_z2) & 1:
        return ((res - res2) * two_inv) % res_md
    return ((res + res2) * two_inv) % res_md

def neutralStringsWithCubeCharacterCountsSum(cube_max: int=87, res_md: Optional[int]=888_888_883) -> int:
    """
    Solution to Project Euler #981
    """
    if res_md is not None:
        pmc = PrimeModuloCalculator(res_md)

    memo1 = {}
    def calculateBinomial(n: int, k: int) -> int:
        args = (n, min(k, n - k))
        if args in memo1.keys(): return memo1[args]
        res = math.comb(n, k)
        memo1[args] = res
        return res

    memo2 = {}
    def calculateBinomialMod(n: int, k: int, pmc: PrimeModuloCalculator) -> int:
        args = (n, min(k, n - k))
        if args in memo2.keys(): return memo2[args]
        res = pmc.binomial(*args)#math.comb(n, k) % pmc.p#
        memo2[args] = res
        return res

    def findCount(n_x: int, n_y: int, n_z: int) -> int:
        counts = sorted([n_x, n_y, n_z])
        if counts[0] < 0: return 0
        if counts[-1] == 0: return 1
        n_odd = sum(x & 1 for x in counts)
        if n_odd % 3: return 0
        if n_odd == 3:
            return (calculateBinomial(n_x + n_y + n_z, n_z) * calculateBinomial(n_x + n_y, n_y)) >> 1
        n_x2, n_y2, n_z2 = n_x >> 1, n_y >> 1, n_z >> 1
        if (n_x2 + n_y2 + n_z2) & 1:
            return (calculateBinomial(n_x + n_y + n_z, n_z) * calculateBinomial(n_x + n_y, n_y) - calculateBinomial(n_x2 + n_y2 + n_z2, n_z2) * calculateBinomial(n_x2 + n_y2, n_y2)) >> 1
        return (calculateBinomial(n_x + n_y + n_z, n_z) * calculateBinomial(n_x + n_y, n_y) + calculateBinomial(n_x2 + n_y2 + n_z2, n_z2) * calculateBinomial(n_x2 + n_y2, n_y2)) >> 1

    def findDoubleCountMod(n_x: int, n_y: int, n_z: int, pmc: PrimeModuloCalculator) -> int:
        res_md = pmc.p
        counts = sorted([n_x, n_y, n_z])
        if counts[0] < 0: return 0
        if counts[-1] == 0: return 2
        n_odd = sum(x & 1 for x in counts)
        if n_odd % 3: return 0
        res = (calculateBinomialMod(n_x + n_y + n_z, n_z, pmc=pmc) * calculateBinomialMod(n_x + n_y, n_y, pmc=pmc)) % res_md
        if n_odd == 3:
            return res
        n_x2, n_y2, n_z2 = n_x >> 1, n_y >> 1, n_z >> 1
        res2 = (calculateBinomialMod(n_x2 + n_y2 + n_z2, n_z2, pmc=pmc) * calculateBinomialMod(n_x2 + n_y2, n_y2, pmc=pmc)) % res_md
        if (n_x2 + n_y2 + n_z2) & 1:
            return (res - res2) % res_md
        return (res + res2) % res_md

    res = 0
    #res_md2 = res_md
    #res_md = None
    addMod = (lambda x, y: x + y) if res_md is None else (lambda x, y: (x + y) % res_md)
    getCount = lambda n_x, n_y, n_z: findCount(n_x, n_y, n_z)
    if res_md is not None:
        pmc = PrimeModuloCalculator(res_md)
        getCount = lambda n_x, n_y, n_z: findDoubleCountMod(n_x, n_y, n_z, pmc=pmc)
    #two_inv = pow(2, res_md - 2, res_md)
    # For non-zero count, values must be either all odd or all even (noting that the
    # parity of a number and its cube are always the same)
    for i in range(cube_max + 1):
        #print(f"i = {i}")
        i_cub = i ** 3
        print(f"i = {i}, i ** 3 = {i_cub}")
        
        for j in range(i & 1, i, 2):
            j_cub = j ** 3
            #print(f"j = {j}, j ** 3 = {j_cub}")
            for k in range(j & 1, j, 2):
                res = addMod(res, 6 * getCount(i_cub, j_cub, k ** 3))
            res = addMod(res, 3 * getCount(i_cub, j_cub, j_cub))
        #print(f"j = {i}, j ** 3 = {i_cub}")
        for k in range(i & 1, i, 2):
            res = addMod(res, 3 * getCount(i_cub, i_cub, k ** 3))
        res = addMod(res, getCount(i_cub, i_cub, i_cub))
        
    #print(res)
    return res if res_md is None else (res * pow(2, res_md - 2, res_md)) % res_md

##############
project_euler_num_range = (951, 1000)

def evaluateProjectEulerSolutions951to1000(eval_nums: Optional[Set[int]]=None) -> None:
    if not eval_nums:
        eval_nums = set(range(project_euler_num_range[0], project_euler_num_range[1] + 1))

    if 951 in eval_nums:
        since = time.time()
        res = gameOfChanceNumberOfFairConfigurations(n=26)
        print(f"Solution to Project Euler #951 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 952 in eval_nums:
        since = time.time()
        res = moduloFactorialMultiplicativeOrder(p=10 ** 9 + 7, n=10 ** 7, res_md=10 ** 9 + 7)
        print(f"Solution to Project Euler #952 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 953 in eval_nums:
        since = time.time()
        res = factorisationNimPlayerOneLoses(n_max=10 ** 14, res_md=10 ** 9 + 7)
        print(f"Solution to Project Euler #953 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 955 in eval_nums:
        since = time.time()
        res = findIndexOfTriangleNumberInRecurrence(n_triangle=70)
        print(f"Solution to Project Euler #955 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 959 in eval_nums:
        since = time.time()
        res = randomWalkNumberOfUniquePointsOverPathLengthInfLimit(l_step_len=89, r_step_len=97, eps=10 ** -12)
        print(f"Solution to Project Euler #959 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 960 in eval_nums:
        since = time.time()
        res = stoneGameSolitaireScoresSumBruteForce(n=12, res_md=None)
        print(f"Solution to Project Euler #960 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 961 in eval_nums:
        since = time.time()
        res = removingDigitsGamePlayerOneWinsCount(n_dig_max=18, base=10)
        print(f"Solution to Project Euler #961 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 965 in eval_nums:
        since = time.time()
        res = expectedMinimalFractionalValue(N=10 ** 1, use_float=True)
        print(f"Solution to Project Euler #965 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 978 in eval_nums:
        since = time.time()
        res = randomWalkSkewness(n_steps=50)
        print(f"Solution to Project Euler #978 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 979 in eval_nums:
        since = time.time()
        res = countPolygonalTilingPaths(polygon_n_sides=7, n_steps=20)
        print(f"Solution to Project Euler #979 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 980 in eval_nums:
        since = time.time()
        res = countGeneratedSequencesNeutralStrings(n_max=10 ** 6 - 1)
        print(f"Solution to Project Euler #980 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 981 in eval_nums:
        since = time.time()
        res = neutralStringsWithCubeCharacterCountsSum(cube_max=87, res_md=888_888_883)
        print(f"Solution to Project Euler #981 = {res}, calculated in {time.time() - since:.4f} seconds")

if __name__ == "__main__":
    eval_nums = {981}
    evaluateProjectEulerSolutions951to1000(eval_nums)


#randomWalksUniquePointSimulation(n_steps=10 ** 7, l_step_len=1, r_step_len=3)
"""
tot = 0
for part in integerPartitionGenerator(
    n=20,
    min_n_part=0,
    max_n_part=None,
    min_part_size=1,
    max_part_size=None,
):
    tot += 1
    #print(part)
print(f"total = {tot}")
"""
"""
l_step_len = 1
r_step_len = 2
l = lcm(l_step_len, r_step_len)
n1 = l // l_step_len
n2 = l // r_step_len
n = n1 + n2
for n_step in range(n, n * 20 + 1, n):
    num = numberOfFirstTimeReturningPaths(n_steps=n_step, l_step_len=l_step_len, r_step_len=r_step_len)
    num2 = numberOfFirstTimeReturningPathsBruteForce(n_steps=n_step, l_step_len=l_step_len, r_step_len=r_step_len)
    print(n_step, num, num2)

def A(n: int) -> int:
    return math.comb(3 * n, n) * 2 // (3 * n - 1)
"""
"""
m = 100
res = 0
for n in range(1, m + 1):
    res += A(n) * 2 ** (-3 * n)
    print(n, res)
print(1 - res)
"""
"""
phi = (math.sqrt(5) + 1) / 2
#print(phi)
for num in range(2, 500):
    res = (float("inf"), None)
    for a in range(1, num):
        g, n_step = gcdNumSteps(num, a)
        if n_step < res[0] and g == 1:
            res = (n_step, a)
    
    print(num, res, num / res[1], num / (res[1] * phi ** 2))
"""

#for args in [(1, 1, 1), (1, 1, 3), (1, 3, 3), (3, 3, 3), (0, 0, 0), (0, 0, 2), (0, 2, 0), (2, 0, 0), (0, 2, 2), (2, 0, 2), (2, 2, 0), (2, 2, 2), (2, 2, 4), (2, 4, 2), (4, 2, 2), (2, 4, 4), (4, 2, 4), (4, 4, 2), (4, 4, 4)]:
#    print(f"N{args} = {countNeutralStringsWithGivenCharacterCounts(*args, res_md=888_888_883)}, {countNeutralStringsWithGivenCharacterCountsBruteForce(*args)}")
#for args in [(0, 2, 4), (2, 0, 4), (1, 3, 5), (3, 1, 5)]:
#    print(f"N{args} = {countNeutralStringsWithGivenCharacterCounts(*args, res_md=888_888_883)}, {countNeutralStringsWithGivenCharacterCountsBruteForce(*args)}")


#print(randomWalkDistributionBruteForce(n_steps=35))