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

from algorithms.number_theory_algorithms import gcd, lcm, isqrt, integerNthRoot, extendedEuclideanAlgorithm
from algorithms.pseudorandom_number_generators import blumBlumShubPseudoRandomGenerator
from algorithms.string_searching_algorithms import KnuthMorrisPratt
#from algorithms.geometry_algorithms import determinant
from algorithms.continued_fractions_and_Pell_equations import generalisedPellSolutionGenerator

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

if __name__ == "__main__":
    eval_nums = {953}
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