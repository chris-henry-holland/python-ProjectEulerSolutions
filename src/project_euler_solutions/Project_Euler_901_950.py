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

from collections import deque, defaultdict
from sortedcontainers import SortedDict, SortedList, SortedSet

from data_structures.fractions import CustomFraction
from data_structures.prime_sieves import PrimeSPFsieve, SimplePrimeSieve

from algorithms.number_theory_algorithms import gcd, lcm, isqrt, integerNthRoot
from algorithms.pseudorandom_number_generators import blumBlumShubPseudoRandomGenerator
from algorithms.string_searching_algorithms import KnuthMorrisPratt


# Problem 932
def splitSumSquareNumbersSum(n_dig_max: int=16, base: int=10) -> int:

    def squareIsSplitSquare(num: int, num_sqrt: int, n_d: int, base: int=10) -> bool:
        i0 = n_d >> 1
        i_set = {i0, n_d - i0}
        for i in i_set:
            base_pow = base ** i
            num1, num2 = divmod(num, base_pow)
            #num1 = 0
            #for j in reversed(range(i)):
            #    num1 = num1 * base + digs[j]
            #num2 = 0
            #for j in reversed(range(i, n_d)):
            #    num2 = num2 * base + digs[j]
            if (num2 * base) // base_pow and (num1 + num2) ** 2 == num:
                return True
        return False
    
    res = 0
    next_base_pow = base ** 2
    n_dig = 2
    for num_sqrt in range(isqrt(base - 1) + 1, isqrt(base ** n_dig_max - 1) + 1):
        num = num_sqrt * num_sqrt
        while num >= next_base_pow:
            next_base_pow *= base
            n_dig += 1
            #print(next_base_pow, n_dig)
        if squareIsSplitSquare(num, num_sqrt, n_dig, base=base):
            num = num_sqrt * num_sqrt
            res += num
            print(num)
    return res


# Problem 933
def paperCuttingWinningMoveSumBasic(width_min: int=2, width_max: int=123, height_min: int=2, height_max: int=1234567) -> int:
    # Using Sprague-Grundy theorem

    memo = {}
    def grundy(w: int, h: int) -> int:
        if w == 1 or h == 1:
            return 0
        if w > h: w, h = h, w
        args = (w, h)
        if args in memo.keys(): return memo[args]
        nums = SortedSet()
        for i1 in range(1, w):
            for i2 in range(1, h):
                num = grundy(i1, i2) ^ grundy(w - i1, i2) ^ grundy(i1, h - i2) ^ grundy(w - i1, h - i2)
                nums.add(num)
        lft, rgt = 0, len(nums)
        while lft < rgt:
            mid = lft + ((rgt - lft) >> 1)
            if nums[mid] == mid:
                lft = mid + 1
            else: rgt = mid
        res = lft
        memo[args] = res
        return res
    """
    for w in range(19, 24):
        for h in range(250, 400):
            ans = 0
            for w1 in range(1, w):
                w2 = w - w1
                for h1 in range(1, h):
                    h2 = h - h1
                    ans += not (grundy(w1, h1) ^ grundy(w1, h2) ^ grundy(w2, h1) ^ grundy(w2, h2))
            print(f"C({w}, {h}) = {ans}")
    """
    res = 0
    for w in range(width_min, width_max + 1):
        print(f"w = {w} of {width_max}")
        for w1 in range(1, w):
            seen_lsts = {}
            w2 = w - w1
            for h in range(max(1, height_min >> 1), (height_max >> 1) + 1):
                g = grundy(w1, h) ^ grundy(w2, h)
                seen_lsts.setdefault(g, [])
                res += (len(seen_lsts[g]) << 1) + 1
                seen_lsts[g].append(h)
            for h in range(max(1, height_min >> 1, (height_max >> 1) + 1), height_max):
                g = grundy(w1, h) ^ grundy(w2, h)
                seen_lsts.setdefault(g, [])
                res += bisect.bisect_right(seen_lsts[g], height_max - h) << 1
                seen_lsts[g].append(h)
    return res
    
def paperCuttingWinningMoveSum(width_min: int=2, width_max: int=123, height_min: int=2, height_max: int=1234567) -> int:

    # Using C(w, h + 1) = C(w, h) + w - 1 and C(w, 1) = 0 for all w and sufficiently large h
    if height_max < width_max:
        height_max, width_max = width_max, height_max
        height_min, width_min = width_min, height_min

    grundy_arr = []

    def extendGrundy(num: int) -> int:
        for h in range(len(grundy_arr), num + 1):
            print(f"extending Grundy array to h = {h}")
            grundy_arr.append([])
            for w in range(min(h, width_max - 1) + 1):
                nums = set()
                curr = 0
                for w1 in range(1, w):
                    w2 = w - w1
                    for h1 in range(1, h):
                        h2 = h - h1
                        #print(sorted([w1, h1]))
                        num = 0
                        for i1, i2 in [[w1, h1], [w1, h2], [w2, h1], [w2, h2]]:
                            if i1 < i2: i1, i2 = i2, i1
                            num ^= grundy_arr[i1][i2]
                        if num == curr:
                            for curr in itertools.count(num + 1):
                                if curr not in nums: break
                        nums.add(num)
                grundy_arr[-1].append(curr)
        return

    def getGrundy(w: int, h: int) -> int:
        if w > h: w, h = h, w
        extendGrundy(h)
        return grundy_arr[h][w]
    """
    memo = {}
    def grundy(w: int, h: int) -> int:
        if w == 1 or h == 1:
            return 0
        if w > h: w, h = h, w
        args = (w, h)
        if args in memo.keys(): return memo[args]
        nums = SortedSet()
        for i1 in range(1, w):
            for i2 in range(1, h):
                num = grundy(i1, i2) ^ grundy(w - i1, i2) ^ grundy(i1, h - i2) ^ grundy(w - i1, h - i2)
                nums.add(num)
        lft, rgt = 0, len(nums)
        while lft < rgt:
            mid = lft + ((rgt - lft) >> 1)
            if nums[mid] == mid:
                lft = mid + 1
            else: rgt = mid
        res = lft
        memo[args] = res
        return res
    """
    h_min = max(height_min, 2)
    w_min = max(width_min, 2)
    res = 0
    #for w in range(w_min, width_max + 1):
    #    res += (w - 1) * ((height_max - 1) * height_max - (h_min) * (h_min - 1))
    
    req_run_len = 150
    for w in range(w_min, width_max + 1):
        print(f"w = {w} of {width_max}")
        since = time.time()
        ans = -float("inf")
        d_run = 0
        for h in range(h_min, height_max + 1):
            prev = ans
            ans = 0
            for w1 in range(1, w):
                w2 = w - w1
                for h1 in range(1, h):
                    h2 = h - h1
                    ans += not (getGrundy(w1, h1) ^ getGrundy(w1, h2) ^ getGrundy(w2, h1) ^ getGrundy(w2, h2))
            #print(f"C({w}, {h}) = {ans}")
            res += ans
            diff = ans - prev
            if diff == w - 1:
                d_run += 1
                if d_run >= req_run_len: break
            else: d_run = 0
        else: continue
        print(f"pattern starts at h = {h - req_run_len}")
        #print(w, h, ans, (height_max - h) * ans + ((w - 1) * ((height_max - h + 1) * (height_max - h)) >> 1))
        res += (height_max - h) * ans + ((w - 1) * ((height_max - h + 1) * (height_max - h)) >> 1)
        print(f"iteration took {time.time() - since:.4f} seconds")
    return res

def paperCuttingWinningMoveSum2(width_min: int=2, width_max: int=123, height_min: int=2, height_max: int=1234567) -> int:

    # Using C(w, h + 1) = C(w, h) + w - 1 and C(w, 1) = 0 for all w and sufficiently large h
    if height_max < width_max:
        height_max, width_max = width_max, height_max
        height_min, width_min = width_min, height_min

    grundy_arr = []
    since = [time.time()]
    def extendGrundy(num: int) -> int:
        for h in range(len(grundy_arr), num + 1):
            if not h % 10:
                print(f"extending Grundy array to h = {h}, time since last print = {time.time() - since[0]:.3f} seconds")
                since[0] = time.time()
            grundy_arr.append([])
            for w in range(min(h, width_max - 1) + 1):
                nums = set()
                curr = 0
                for w1 in range(1, w):
                    w2 = w - w1
                    for h1 in range(1, h):
                        h2 = h - h1
                        #print(sorted([w1, h1]))
                        num = 0
                        for i1, i2 in [[w1, h1], [w1, h2], [w2, h1], [w2, h2]]:
                            if i1 < i2: i1, i2 = i2, i1
                            num ^= grundy_arr[i1][i2]
                        if num == curr:
                            for curr in itertools.count(num + 1):
                                if curr not in nums: break
                        nums.add(num)
                grundy_arr[-1].append(curr)
        return

    def getGrundy(w: int, h: int) -> int:
        if w > h: w, h = h, w
        extendGrundy(h)
        return grundy_arr[h][w]

        
    h_min = max(height_min, 2)
    w_min = max(width_min, 2)
    #res = 0
    #for w in range(w_min, width_max + 1):
    #    res += (w - 1) * ((height_max - 1) * height_max - (h_min) * (h_min - 1))
    
    def calculateTotal(h_max: int) -> int:
        #print(f"calculating total for h_max = {h_max}")
        res = 0
        for w in range(w_min, width_max + 1):
            #print(f"w = {w} of {width_max}")
            for w1 in range(1, w):
                seen_lsts = {}
                w2 = w - w1
                for h in range(max(1, h_min >> 1), (h_max >> 1) + 1):
                    g = getGrundy(w1, h) ^ getGrundy(w2, h)
                    seen_lsts.setdefault(g, [])
                    res += (len(seen_lsts[g]) << 1) + 1
                    seen_lsts[g].append(h)
                for h in range(max(1, h_min >> 1, (h_max >> 1) + 1), h_max):
                    g = getGrundy(w1, h) ^ getGrundy(w2, h)
                    seen_lsts.setdefault(g, [])
                    res += bisect.bisect_right(seen_lsts[g], h_max - h) << 1
                    seen_lsts[g].append(h)
        return res

    #req_run_len = 150
    #tot = 0
    target_diff = (width_max * (width_max - 1) - (w_min - 1) * (w_min - 2)) >> 1
    h_mx = 2
    #print(f"height_max = {height_max}")
    linear_transition = 100
    h_mx_linear_diff = 100
    while h_mx < height_max:
        h1 = h_mx - 1
        h2 = h_mx
        h3 = h_mx + 1
        h4 = h_mx + 2
        tot1 = calculateTotal(h1)
        tot2 = calculateTotal(h2)
        tot3 = calculateTotal(h3)
        d1 = tot3 - 2 * tot2 + tot1
        if d1 == target_diff:
            tot4 = calculateTotal(h4)
            d2 = tot4 - 2 * tot3 + tot2
            print(f"differences = {d1}, {d2}, target difference = {target_diff}")
            if d2 == target_diff:
                print("target difference found")
                return tot1 + (tot2 - tot1) * (height_max - h1) + target_diff * (((height_max - h1) * (height_max - h1 - 1)) >> 1)
        else:
            print(f"difference = {d1}, target difference = {target_diff}")
        
        if h_mx > linear_transition:
            h_mx += h_mx_linear_diff
        else:
            h_mx <<= 1
        print(f"new h_mx = {h_mx}")
    #print(f"height_max = {height_max}")
    return calculateTotal(height_max)

def paperCuttingWinningMoveSum3(width_min: int=2, width_max: int=123, height_min: int=2, height_max: int=1234567) -> int:

    # Using C(w, h + 1) = C(w, h) + w - 1 and C(w, 1) = 0 for all w and sufficiently large h
    if height_max < width_max:
        height_max, width_max = width_max, height_max
        height_min, width_min = width_min, height_min
    
    h_min = max(2, height_min)
    w_min = max(2, width_min)
    if w_min > width_max or h_min > height_max:
        return 0

    grundy_pairs = [[[0] * min(w1, width_max - w1) for w1 in range(1, width_max)]]

    def getGrundyPair(h: int, w1: int, w2: int) -> int:
        #if w2 > w1: w1, w2 = w2, w1
        #print(h, w1, w2)
        #print(grundy_pairs)
        return grundy_pairs[h - 1][w1 - 1][w2 - 1]
    
    def calculateTotal(h_max: int) -> int:
        #print(f"calculating total for h_max = {h_max}")
        #print(grundy_pairs)
        res = 0
        for w in range(w_min, width_max + 1):
            #print(f"w = {w} of {width_max}")
            for w1 in range((w + 1) >> 1, w):
                seen_lsts = {}
                w2 = w - w1
                mult = 1 + (w1 != w2)
                ans = 0
                for h in range(max(1, h_min >> 1), (h_max >> 1) + 1):
                    g = getGrundyPair(h, w1, w2)#grundy_pairs[h][w1][w2]
                    seen_lsts.setdefault(g, [])
                    ans += ((len(seen_lsts[g]) << 1) + 1) * mult
                    seen_lsts[g].append(h)
                for h in range(max(1, h_min >> 1, (h_max >> 1) + 1), h_max):
                    g = getGrundyPair(h, w1, w2)#grundy_pairs[h - 1][w1 - 1][w2 - 1]
                    seen_lsts.setdefault(g, [])
                    ans += (bisect.bisect_right(seen_lsts[g], h_max - h) << 1) * mult
                    #print(h_max - h)
                    seen_lsts[g].append(h)
                #print(f"w1 = {w1}, w2 = {w2}, ans = {ans}")
                #print(seen_lsts)
                res += ans
        return res

    
    target_diffdiff = (width_max * (width_max - 1) - (w_min - 1) * (w_min - 2)) >> 1
    req_run_len = 50
    curr_run_len = 0
    tots_prev = [0, 0]

    def calculateSecondDifference(tot1: int, tot2: int, tot3: int) -> int:
        return tot3 - 2 * tot2 + tot1

    since = time.time()
    for h in range(2, height_max):
        #grundy_pairs.append([[0] * (w + 1) for w in range(width_max)])
        tot = calculateTotal(h)
        dd = calculateSecondDifference(*tots_prev, tot)
        if not h % 10:
            print(f"h = {h}, diffdiff = {dd} (target = {target_diffdiff}), time since last print = {time.time() - since:.3f} seconds")
            since = time.time()
        if dd == target_diffdiff:
            curr_run_len += 1
            if curr_run_len >= req_run_len:
                print(f"diffdiff run found starting at h = {h - req_run_len}")
                break
        else: curr_run_len = 0
        tots_prev = [tots_prev[1], tot]

        curr_grundy = [0, 0]
        for w in range(2, width_max):
            seen = set()
            curr = 0
            for w1 in range((w + 1) >> 1, w):
                w2 = w - w1
                for h1 in range((h + 1) >> 1, h):
                    h2 = h - h1
                    grundy = getGrundyPair(h1, w1, w2) ^ getGrundyPair(h2, w1, w2) #grundy_pairs[h2 - 1][w - 1][w - w2 - 1] ^ grundy_pairs[h - h2 - 1][w - 1][w - w2 - 1]
                    if curr == grundy:
                        curr += 1
                        while curr in seen:
                            curr += 1
                    seen.add(grundy)
            
            curr_grundy.append(curr)
            #print(f"h = {h}, w = {w}, seen = {seen}, curr = {curr}")
        grundy_pairs.append([])
        for w1 in range(1, width_max):
            grundy_pairs[-1].append([])
            for w2 in range(1, min(w1, width_max - w1) + 1):
                grundy_pairs[-1][-1].append(curr_grundy[w1] ^ curr_grundy[w2])
        if h < h_min: continue
        
    else:
        return calculateTotal(height_max)
    tot0 = tots_prev[-1]
    return tot + (tot - tot0) * (height_max - h) + target_diffdiff * (((height_max - h + 1) * (height_max - h)) >> 1)

# Problem 934
def unluckyPrimeCalculatorBruteForce(n: int, p_md: int, ps: Optional["SimplePrimeSieve"]=None) -> int:
    if ps is None: ps = SimplePrimeSieve()
    for p in ps.endlessPrimeGenerator():
        if (n % p) % p_md: return p
    return -1

def unluckyPrimeSumBruteForce(n_max: int, p_md: int=7, ps: Optional["SimplePrimeSieve"]=None) -> int:
    if ps is None: ps = SimplePrimeSieve()
    res = 0
    for n in range(1, n_max + 1):
        p = unluckyPrimeCalculatorBruteForce(n, p_md=p_md, ps=ps)
        #if p > 11: print(n, p)
        res += p
    return res

def unluckyPrimeSum(n_max: int=10 ** 17, p_md: int=7, ps: Optional["SimplePrimeSieve"]=None) -> int:
    """
    Solution to Project Euler #934
    """
    # Look into the residue class solutions to get faster solution
    if ps is None: ps = SimplePrimeSieve()
    res = 0
    remain = n_max
    mult = 1
    add_starts = []
    p_it = ps.endlessPrimeGenerator()
    remain_transition = integerNthRoot(n_max, 3)#isqrt(n_max)
    for p in p_it:
        print(f"p = {p}, remain = {remain}, len(add_starts) = {len(add_starts)}")
        mult_md = mult % p
        mult_md_inv = pow(mult_md, p - 2, p)
        add_starts2 = [(num + mult * ((((-num) % p) * mult_md_inv) % p)) for num in add_starts]
        for r in range(p_md, p, p_md):
            add_starts2.append(mult * ((r * mult_md_inv) % p))
            for num in add_starts:
                add_starts2.append(num + mult * ((((r - num) % p) * mult_md_inv) % p))

        add_starts = sorted(add_starts2)
        #print(f"p = {p}, add_starts = {add_starts}, mult_md_inv = {mult_md_inv}")
        mult *= p
        cnt0, r = divmod(n_max, mult)
        cnt = cnt0 * (1 + len(add_starts)) + bisect.bisect_right(add_starts, r)
        res += (remain - cnt) * p
        remain = cnt
        if remain < remain_transition: break
        #add_starts2 = []
        #for p2 in range(p_md, p, p_md):
        #    add_starts2
        #add_starts2.sort()
    print("transitioning to phase 2")
    remain_set = set(range(mult, n_max + 1, mult))
    for add_start in add_starts:
        for num in range(add_start, n_max + 1, mult):
            remain_set.add(num)
    print(len(remain_set), remain)
    for p in p_it:
        print(f"p = {p}, remain = {len(remain_set)}")
        remain_set2 = set()
        for num in remain_set:
            if not (num % p) % p_md:
                remain_set2.add(num)
            else: res += p
        remain_set = remain_set2
        if not remain_set:
            break

    #print(f"largest p = {p}")
    #print(unluckyPrimeSumBruteForce(n_max, p_md=p_md, ps=None))
    return res

# Problem 935
def rollingRegularPolygonReturnAfterAtMostNRollsCountBasic(n_roll_max: int, n_sides: int=4) -> int:
    res = 0
    arr = [[] for _ in range(n_roll_max + 1)]
    
    for corner in range(1, n_roll_max + 1):
        ans = 0
        n_rpt = n_sides // (gcd(corner, n_sides))
        mn = corner
        mx = n_roll_max // n_rpt
        intvl = 1
        arr[corner].extend([0] * (mx - len(arr[corner]) + 1))
        for n_roll in range(mn, mx + 1, intvl):
            arr[corner][n_roll] += 1
            ans += arr[corner][n_roll]
        for i, corner2 in enumerate(range(corner * 2, n_roll_max + 1, corner), start=2):
            n_rpt2 = n_sides // (gcd(corner2, n_sides))
            mx2 = n_roll_max // n_rpt2
            for n_roll in range(mn, mx + 1, intvl):
                if not arr[corner][n_roll]: continue
                n_roll2 = (n_roll + 1) * i - 1
                if n_roll2 > mx2: break
                arr[corner2].extend([0] * (n_roll2 - len(arr[corner2]) + 1))
                arr[corner2][n_roll2] -= arr[corner][n_roll]
        res += ans
    return res

def rollingRegularPolygonReturnAfterAtMostNRollsCount(n_roll_max: int, n_sides: int=4) -> int:
    """
    Solution to Project Euler #935
    """
    # Note- requires n_sides > 3 (otherwise cannot transition from a
    # shape being in the corner as described without part of the
    # shape being outside the boundaries, effectively locking the
    # shape in the corner)

    # Review- Try to make faster

    if n_sides <= 3:
        return 0

    """
    rpt_func = lambda corner: n_sides // (gcd(corner, n_sides))
   
    res = 0
    for corner in range(1, n_roll_max + 1):
        ans = max(0, (n_roll_max // rpt_func(corner)) - corner + 1)
        #print(ans)
        for a in range(n_sides):
            rpt2 = rpt_func(a * corner)
            term = 0
            for r in range(corner, (n_roll_max >> 1) + 1):
                term2 = (n_roll_max + rpt2) // ((r + 1) * rpt2)
                term += max((term2 - a) // n_sides, 0)
            ans -= term
        print(corner, ans)
        res += ans
    return res
    """
    #ps = PrimeSPFsieve(n_roll_max)
    ps = SimplePrimeSieve(isqrt(n_roll_max))
    print("finished creating prime sieve")

    def calculatePrimeFactors(num: int) -> List[int]:
        num2 = num
        res = []
        for p in ps.p_lst:
            if p ** 2 > num2: break
            num3, r = divmod(num2, p)
            if r: continue
            res.append(p)
            num2 = num3
            num3, r = divmod(num2, p)
            while not r:
                num2 = num3
                num3, r = divmod(num2, p)
        if num2 > 1: res.append(num2)
        return res
    res = 0
    #corner_md_rpts = [n_sides // (gcd(corner, n_sides)) for corner in range(n_sides)]
    for corner_md in range(n_sides):
        #print(f"corner mod {n_sides} = {corner_md}")
        n_rpt = n_sides // (gcd(corner_md, n_sides))
        start = corner_md + (n_sides if not corner_md else 0)
        end = n_roll_max // n_rpt
        roll_mx = n_roll_max // n_rpt
        for corner in range(start, end + 1, n_sides):
            if not ((corner - start) % (n_sides * 10 ** 5)):
                print(f"corner mod {n_sides} = {corner_md}, corner = {corner} (max {end})")
            #pf = ps.primeFactorisation(corner)
            #p_lst = sorted(pf.keys())
            p_lst = calculatePrimeFactors(corner)
            #print(corner, p_lst)
            n_p = len(p_lst)
            r_plus_one_rng = (corner + 1, roll_mx + 1)
            ans = r_plus_one_rng[1] - r_plus_one_rng[0] + 1
            for bm in range(1, 1 << n_p):
                neg = False
                num = 1
                for i in range(n_p):
                    if bm & 1:
                        neg = not neg
                        num *= p_lst[i]
                        if bm == 1: break
                    bm >>= 1
                cnt = (r_plus_one_rng[1] // num) - ((r_plus_one_rng[0] - 1) // num)
                ans += -cnt if neg else cnt
            res += ans
    #print(rollingRegularPolygonReturnAfterAtMostNRollsCountBasic(n_roll_max=n_roll_max, n_sides=n_sides))
    return res
    
    """
    ps = PrimeSPFsieve(n_roll_max)

    def cornerRollCount(corner: int, mx: int) -> int:
        p_fact = ps.primeFactorisation(corner)
        phi = 1
        for p, f in enumerate(p_fact):
            phi *= (p - 1) * p ** (f - 1)
        p_lst = list(p_fact.keys())
        f_lst = [p_fact[p] for p in p_lst]
        print(p_lst, f_lst)
        n_p = len(p_lst)
        n_fact = sum(f_lst)
        
        def recur(idx: int, curr: int=1) -> Generator[int, None, None]:
            if idx == n_p:
                yield curr
                return
            curr2 = curr
            for f in range(f_lst[idx] + 1):
                yield from recur(idx + 1, curr=curr2)
                curr2 *= -p_lst[idx]
            return
        
        #res = phi
        #for num in range(1, corner):
        #    if gcd(num, corner)
        #if gcd(res)

        #return max(0, mx - corner + 1)

        res = 0
        for fact in recur(0, curr=(1 - ((n_fact & 1) << 1))):
            fact2 = corner // abs(fact)
            mx2 = ((mx - fact2 + 1) // fact2)
            #num = max(0, ((mx * abs(fact)) // corner) - abs(fact) + 1)# // corner
            num = max(0, mx2 - abs(fact) + 1)
            #num = max(0, mx2 - abs(fact) + 1 + (abs(fact) != 1))
            print(f"fact = {fact}, mx2 = {mx2}, contrib = {num}")
            res += num if fact > 0 else -num
            #res += fact
        return res
    """
    """
    res = 0
    #ps = PrimeSPFsieve()
    cnt_arr = [[0] for _ in range(n_roll_max + 1)]
    for corner in range(1, n_roll_max + 1):
        n_rpt = n_sides // (gcd(corner, n_sides))#lcm(corner, n_sides) // corner
        m = n_roll_max // n_rpt
        #if m < corner: continue
        #mult = ((m - 1) // (corner + 1))
        ans0 = m - corner + 1
        if ans0 > 0:
            cnt_arr[corner] += ans0
        res += cnt_arr[corner]
        for num in range(2 * corner, n_roll_max + 1, corner):
            cnt_arr[num] -= cnt_arr[corner]
        print(cnt_arr)
    return res
    """
    """
    res = 0
    arr = [[] for _ in range(n_roll_max + 1)]
    
    for corner in range(1, n_roll_max + 1):
        ans = 0
        n_rpt = n_sides // (gcd(corner, n_sides))
        mn = corner
        mx = n_roll_max // n_rpt
        intvl = 1#2 - (corner & 1)
        arr[corner].extend([0] * (mx - len(arr[corner]) + 1))
        for n_roll in range(mn, mx + 1, intvl):
            arr[corner][n_roll] += 1
            ans += arr[corner][n_roll]
        for i, corner2 in enumerate(range(corner * 2, n_roll_max + 1, corner), start=2):
            n_rpt2 = n_sides // (gcd(corner2, n_sides))
            mx2 = n_roll_max // n_rpt2
            #mn = corner2 + i - 1
            #mx = n_roll_max // n_rpt
            #if mn > mx: break
            #intvl = i#2 - (corner & 1)
            for n_roll in range(mn, mx + 1, intvl):
                if not arr[corner][n_roll]: continue
                n_roll2 = (n_roll + 1) * i - 1
                if n_roll2 > mx2: break
                arr[corner2].extend([0] * (n_roll2 - len(arr[corner2]) + 1))
                arr[corner2][n_roll2] -= arr[corner][n_roll]
        res += ans
        #print(corner, ans, res)#, arr)
        #if ans0 > 0:
        #    cnt_arr[corner] += ans0
        #res += cnt_arr[corner]
        #for num in range(2 * corner, n_roll_max + 1, corner):
        #    cnt_arr[num] -= cnt_arr[corner]
        #print(cnt_arr)
    #print(arr)
    #for i, lst in enumerate(arr):
    #    print(i, sum(lst))
    
    return res
    """
    """
    res = 0
    ps = PrimeSPFsieve(n_roll_max)
    print("finished creating prime sieve")
    for corner in range(1, n_roll_max + 1):
        
        ans = 0
        for i, num in enumerate(range(corner, n_roll_max + 1, corner), start=1):
            # number of rolls = (n_rolls0 + 1) * i - 1
            # min number of rolls = corner
            # max number of rolls = n_roll_max
            
            # (max number of rolls + 1) // i - 1
            mn_abs = (num + i - 1)
            #if mn_abs & 1 and not num & 1: continue

            n_rpt = n_sides // (gcd(num, n_sides))#lcm(corner, n_sides) // corner
            #print(num, n_rpt)
            #mx = ((n_roll_max + 1) // i - 1) // n_rpt
            #mx = n_roll_max // (i * n_rpt)
            #mn = max(0, num + i - 2)
            #if not num & 1:
            #    mx >>= 1
            #    mn >>= 1
            
            
            #if not num & 1 and mn_abs & 1: mn_abs += 1
            mn_abs *= n_rpt
            mx_abs = n_roll_max
            intvl = n_rpt * i# * (2 - (num & 1))
            if mx_abs < mn_abs: continue
            #mn, mx = (mn_abs - 1) // intvl, mx_abs // intvl
            #if m < num: continue
            #if m < corner: continue
            #mult = ((m - 1) // (corner + 1))
            #ans0 = m - (corner - 1) // n_rpt#(m // (i)) - (num - 1) // (i)
            ans0 = (mx_abs - mn_abs) // intvl + 1
            
            if ans0 <= 0: continue
            #print(corner, num, i, (mn_abs, mx_abs), intvl, ans0)
            
            pf = ps.primeFactorisation(i)
            if any(x & 1 for x in pf.values()) or not any(x % 4 for x in pf.values()):
                #print(sum(pf.values()))
                ans += -ans0 if (sum(pf.values()) & 1) else ans0
            #ans += -ans0 if (sum(pf.values()) & 1) else ans0
            #n_factors_pa = 1
            #for exp in pf.values():
            #    n_factors *= (exp + 1)
            #n_factors_odd = (isqrt(i) ** 2 == i)
            #ans += ans0 if (n_factors_odd) else -ans0
            
        
        res += ans
        #print(corner, ans, res)
        #if ans0 > 0:
        #    cnt_arr[corner] += ans0
        #res += cnt_arr[corner]
        #for num in range(2 * corner, n_roll_max + 1, corner):
        #    cnt_arr[num] -= cnt_arr[corner]
        #print(cnt_arr)
    #print(rollingRegularPolygonReturnAfterAtMostNRollsCountBasic(n_roll_max=n_roll_max, n_sides=n_sides))
    return res
    """
    """
    ps = PrimeSPFsieve(n_roll_max)
    res = n_roll_max // n_sides
    print(f"res = {res}")
    for corner in range(1, (n_roll_max >> 1) + 1):
        # lcm = a * b // gcd
        # lcm // a = b // gcd
        n_rpt = n_sides // (gcd(2 * corner, n_sides))#lcm(corner, n_sides) // corner
        
        m = (n_roll_max) // (2 * n_rpt)
        #print(f"corner = {corner}, n_rpt = {n_rpt}, m = {m}")
        if m - corner + 1 <= 0: continue
        ans = m - corner + 1# ((not corner & 1) or corner == 1)#(corner <= 24)#cornerRollCount(corner, m)
        corner2 = corner
        while not corner2 & 1: corner2 >>= 1
        pf = ps.primeFactors(corner2)
        nums = set()
        for p in pf:
            if p == 2: continue
            num0 = p >> 1
            nums |= set(range(num0, corner, p))
        nums = sorted(nums)
        q, r = divmod(m, corner)
        ans -= len(nums) * (q - 1) + bisect.bisect_right(nums, r)

        #if corner > 1 and corner & 1:
        #if corner == 3:
        #    n_roll0 = (corner >> 1) * 3 + 1
        #    sub = max(0, (m - n_roll0) // corner) + 1
        #    print(f"corner = {corner}, n_roll0 = {n_roll0}, sub = {sub}")
        #    ans -= sub
        print(corner, n_rpt, m, ans)
        res += ans
    return res
    """

# Problem 936
def peerlessTreesWithVertexCountInRange(n_vertex_min: int=3, n_vertex_max: int=50) -> int:

    # dim 0: degree of root (including incoming edge)
    # dim 1: max number of vertices in the rooted trees
    # dim 2: max depth of the rooted trees
    arr = [[], [[0], [1]]]

    d_mx = (n_vertex_max - 1) >> 1
    v_mx = n_vertex_max - d_mx - 1

    def recur(idx: int, curr: int, v_remain: int, prev_depth: int, prev_n_v: int, root_deg: int, n_rpt: int=1) -> int:
        if not curr: return 0
        if idx == root_deg - 2:
            #print("hi")
            #print(idx, curr, v_remain, prev_depth, prev_n_v, root_deg, n_rpt)
            depth_max = min(v_remain, prev_depth - (v_remain > prev_n_v))# + (prev_n_v >= v_remain))
            res = getRootedCountRootDegreeAndDepthCumu(v_remain + 1, n_v=v_remain, depth_max=depth_max - 1)
            #print(f"res = {res}")
            if v_remain + 1 >= root_deg:
                res -= getRootedCountRootDegreeAndDepthCumu(root_deg, n_v=v_remain, depth_max=depth_max - 1) - getRootedCountRootDegreeAndDepthCumu(root_deg - 1, n_v=v_remain, depth_max=depth_max - 1) 
            #print(res)
            res *= curr
            if v_remain == prev_n_v and v_remain > prev_depth:
                div = n_rpt + 1
                res -= (curr * (div - 1) * getRootedCount(prev_depth, n_v=prev_n_v, depth=prev_depth)) // div
            #print(f"res = {res}")
            return res
        res = 0
        #print(f"depth range = [1, {min(prev_depth, v_remain - (root_deg - idx - 2)) + 1})")
        for depth in range(1, min(prev_depth - 1, v_remain - (root_deg - idx - 2)) + 1):
            #print(f"v_remain = {v_remain}, root_deg = {root_deg}, idx = {idx}, n_v range = [{depth}, {v_remain - (root_deg - idx - 2) + 1})")
            for n_v in range(depth, v_remain - (root_deg - idx - 2) + 1):
                #print(f"n_v = {n_v}, depth = {depth}")
                curr2 = getRootedCountRootDegreeCumu(n_v, n_v, depth - 1)
                if n_v >= root_deg:
                    curr2 -= getRootedCount(root_deg, n_v, depth - 1)
                #print(f"curr2 = {curr2}")
                if not curr2: continue
                res += recur(idx + 1, curr * curr2, v_remain - n_v, depth, n_v, root_deg, n_rpt=1)
        #print(f"main loop res = {res}")
        if prev_depth <= v_remain:
            #print("hi2")
            #print(prev_depth, v_remain, root_deg, idx, prev_n_v + 1, v_remain - (root_deg - idx - 2))
            for n_v in range(prev_depth, min(prev_n_v - 1, v_remain - (root_deg - idx - 2)) + 1):
                #print(f"n_v = {n_v}, prev_depth = {prev_depth}")
                curr2 = getRootedCountRootDegreeCumu(n_v, n_v, prev_depth - 1)
                if n_v >= root_deg:
                    curr2 -= getRootedCount(root_deg, n_v, prev_depth - 1)
                #print(f"curr2 2 = {curr2}")
                if not curr2: continue
                res += recur(idx + 1, curr * curr2, v_remain - n_v, prev_depth, n_v, root_deg, n_rpt=1)
            #print(f"post depth match res = {res}")
            if v_remain - (root_deg - idx - 2) >= prev_n_v:
                n_v = prev_n_v
                n_rpt2 = n_rpt + 1
                curr2 = (getRootedCountRootDegreeCumu(n_v, n_v, prev_depth - 1) + n_rpt)
                if n_v >= root_deg:
                    curr2 -= getRootedCount(root_deg, n_v, prev_depth - 1)
                #print(f"curr2 3 = {curr2}")
                if curr2:
                    #print("repeat found")
                    res += recur(idx + 1, curr * curr2, v_remain - n_v, prev_depth, n_v, root_deg, n_rpt=n_rpt2) // n_rpt2
            #print(f"post duplicates res = {res}")
        #print(f"overall res = {res}")
        return res

    memo1 = {}
    def getRootedCount(root_deg: int, n_v: int, depth: int) -> int:
        
        if not depth:
            #print("hi1")
            return int(n_v == 1 and root_deg == 1)
        elif depth == 1:
            return int(n_v == root_deg)
        if root_deg < 1: return 0
        elif n_v < 1 or depth < 0: return 0
        elif depth > n_v - root_deg + 1: return 0
        elif root_deg == 1:
            #print("hi2")
            return int(n_v == 1 and depth == 0)
        args = (root_deg, n_v, depth)
        if args in memo1.keys():
            return memo1[args]
        ref = None#(4, 4, 1)
        if (root_deg, n_v, depth) == ref:
            print("***************************")
        if root_deg == 2:
            # Exactly one sub-tree
            
            res = getRootedCountRootDegreeCumu(n_v, n_v - 1, depth - 1)
            if n_v - 1 >= root_deg:
                res -= getRootedCount(root_deg, n_v - 1, depth - 1)
            if args == ref:
                print(res)
            memo1[args] = res
            return res
        #v_remain = n_v
        #curr = (depth, n_v + 1)
        #seen = {}
        
        res = 0
        if args == ref:
            print(f"n_v2 range: [{depth}, {n_v - root_deg + 2})")
        for n_v2 in range(depth, n_v - root_deg + 2):
            
            curr = getRootedCountRootDegreeCumu(n_v2, n_v2, depth - 1)
            if n_v2 >= root_deg:
                curr -= getRootedCount(root_deg, n_v2, depth - 1)
            #print(f"n_v2 = {n_v2}, curr = {curr}")
            if not curr: continue
            ans = recur(1, curr, n_v - n_v2 - 1, depth, n_v2, root_deg, n_rpt=1)
            if args == ref:
                print(f"for n_v2 = {n_v2}, ans = {ans}")

            res += ans
            #if depth == 1:
            #    print(f"root_deg = {root_deg}, n_v = {n_v}, depth = {depth}, n_v2 = {n_v2}, curr = {curr}, ans = {ans}")

        memo1[args] = res
        if (root_deg, n_v, depth) == ref:
            print("***************************")
            print(f"res = {res}")
        return res 

    memo2 = {}
    def getRootedCountCumu(root_deg: int, n_v_max: int, depth_max: int) -> int:
        if root_deg < 1: return 0
        elif n_v_max < 1 or depth_max < 0: return 0
        args = (root_deg, n_v_max, depth_max)
        if args in memo2.keys(): return memo2[args]
        res = getRootedCount(root_deg, n_v_max, depth_max) + getRootedCountCumu(root_deg, n_v_max - 1, depth_max) +\
            getRootedCountCumu(root_deg, n_v_max, depth_max - 1) - getRootedCountCumu(root_deg, n_v_max - 1, depth_max - 1)
        memo2[args] = res
        return res

    memo3 = {}
    def getRootedCountRootDegreeCumu(root_deg_max: int, n_v: int, depth: int) -> int:
        if root_deg_max < 1: return 0
        elif n_v < 1 or depth < 0: return 0
        args = (root_deg_max, n_v, depth)
        if args in memo3.keys(): return memo3[args]
        res = getRootedCount(root_deg_max, n_v, depth) + getRootedCountRootDegreeCumu(root_deg_max - 1, n_v, depth)
        memo3[args] = res
        return res
    
    memo4 = {}
    def getRootedCountRootDegreeAndDepthCumu(root_deg_max: int, n_v: int, depth_max: int) -> int:
        if root_deg_max < 1: return 0
        elif n_v < 1 or depth_max < 0: return 0
        args = (root_deg_max, n_v, depth_max)
        if args in memo4.keys(): return memo4[args]
        res = getRootedCount(root_deg_max, n_v, depth_max) + getRootedCountRootDegreeAndDepthCumu(root_deg_max - 1, n_v, depth_max) +\
            getRootedCountRootDegreeAndDepthCumu(root_deg_max, n_v, depth_max - 1) - getRootedCountRootDegreeAndDepthCumu(root_deg_max - 1, n_v, depth_max - 1)
        memo4[args] = res
        return res

    memo5 = {}
    def getRootedCountVertexCountCumu(root_deg: int, n_v_max: int, depth: int) -> int:
        if root_deg < 1: return 0
        elif n_v_max < 1 or depth < 0: return 0
        args = (root_deg, n_v_max, depth)
        if args in memo5.keys(): return memo5[args]
        res = getRootedCount(root_deg, n_v_max, depth) + getRootedCountVertexCountCumu(root_deg, n_v_max - 1, depth)
        memo5[args] = res
        return res

    memo6 = {}
    def getRootedCountRootDegreeAndVertexCountCumu(root_deg_max: int, n_v_max: int, depth: int) -> int:
        if root_deg_max < 1: return 0
        elif n_v_max < 1 or depth < 0: return 0
        args = (root_deg_max, n_v_max, depth)
        if args in memo6.keys(): return memo6[args]
        res = getRootedCount(root_deg_max, n_v_max, depth) + getRootedCountRootDegreeAndVertexCountCumu(root_deg_max, n_v_max - 1, depth) +\
            getRootedCountRootDegreeAndVertexCountCumu(root_deg_max - 1, n_v_max, depth) - getRootedCountRootDegreeAndVertexCountCumu(root_deg_max - 1, n_v_max - 1, depth)
        memo6[args] = res
        return res

    def peerlessTreesWithMaxVertexCount(n_vertex_max: int):
        if n_vertex_max < 1: return 0
        elif n_vertex_max <= 2: return 1
        #elif n_vertex_max == 3: return 2
        res = 0
        # Tree centre is two adjacent nodes
        for depth in range(1, d_mx + 1):
            for n_v1 in range(depth + 1, (n_vertex_max >> 1) + 1):
                for degree1 in range(2, n_v1 + 1):
                    mult = getRootedCount(degree1, n_v1, depth)
                    #mult = getRootedCount(degree1, n_v1, depth)
                    n_v2_min = n_v1 + 1
                    n_v2_max = n_vertex_max - n_v1
                    cumu1 = getRootedCountRootDegreeAndVertexCountCumu(n_v2_max, n_v2_max, depth)
                    cumu2 = getRootedCountRootDegreeAndVertexCountCumu(n_v2_max, n_v2_min - 1, depth)
                    #print(f"n_v1 = {n_v1}, degree1 = {degree1}, degree_max = {n_v2_max}, depth = {depth}, n_v2 range = [{n_v2_min}, {n_v2_max}], lower cumu = {cumu2}, upper_cumu = {cumu1}")
                    #print(n_v2_min, n_v2_max + 1)
                    #for n_v2 in range(n_v2_min, n_v2_max + 1):
                    #    for degree in range(1, n_v2_max + 1):
                    #        print(degree, n_v2, depth, getRootedCount(degree, n_v2, depth))
                    ans = cumu1 - cumu2
                    #print(f"ans0 = {ans}")
                    if n_v2_max >= degree1:
                        # Ensure the two roots have different degrees
                        
                        sub = getRootedCountVertexCountCumu(degree1, n_v2_max, depth) - getRootedCountVertexCountCumu(degree1, n_v2_min - 1, depth)
                        #print(f"subtracting {sub}")
                        ans -= sub
                    #print(f"ans1 = {ans}")
                    #for degree2 in range(2, n_v2_max + 1):
                    #    if degree2 == degree1: continue
                    #    ans += getRootedCountVertexCountCumu(degree2, n_v2_max, depth) - getRootedCountRootVertexCountCumu(degree2, n_v1, depth)
                    #ans *= mult
                    #ans2 = 0
                    # Avoiding double counting when the two nodes central nodes have the same number of vertices
                    # by only counting those for which degree2 is less than degree1 (recall they cannot be equal)
                    ans += getRootedCountRootDegreeCumu(degree1 - 1, n_v1, depth)
                    
                    ans2 = ans * mult
                    #print(f"depth = {depth}, n_v1 = {n_v1}, n_v2 range = [{n_v2_min - 1}, {n_v2_max}], degree1 = {degree1}, mult = {mult}, count = {ans2}")
                    res += ans2
        print(f"\ndouble centre: {res}\n")
        res2 = n_vertex_max - 1 # Every vertex other than the centre is a leaf, plus the single node graph
        # Trees with a single centre
        # Centre degree 2
        centre_degree = 2
        #print(f"\ncentre_degree = {centre_degree}")
        for depth in range(1, d_mx + 1):
            #print(f"depth = {depth}")
            for tot_n_v in range((depth + 1) * 2 + 1 + (centre_degree - 2), n_vertex_max + 1):
                #print(f"tot_n_v = {tot_n_v}")
                for n_v1 in range(max(depth + 1, (tot_n_v >> 1)), tot_n_v - depth - 1):
                    #print(f"n_v1 = {n_v1}")
                    curr2 = getRootedCountRootDegreeCumu(n_v1, n_v1, depth)
                    if n_v1 >= centre_degree:
                        curr2 -= getRootedCount(centre_degree, n_v1, depth)
                    n_v2 = tot_n_v - n_v1 - 1
                    #print(f"n_v2 = {n_v2}")
                    n_rpt = 1
                    if n_v2 == n_v1:
                        curr3 = curr2 + 1
                        n_rpt += 1
                    else:
                        curr3 = getRootedCountRootDegreeCumu(n_v2, n_v2, depth)
                        if n_v2 >= centre_degree:
                            curr3 -= getRootedCount(centre_degree, n_v2, depth)
                    ans0 = curr2 * curr3
                    ans = ans0 // n_rpt
                    #print(f"centre_degree = {centre_degree}, depth = {depth}, tot_n_v = {tot_n_v}, n_v1 = {n_v1}, n_v2 = {n_v2}, curr2 = {curr2}, curr3 = {curr3}, count0 = {ans0}, count = {ans}")
                    res2 += ans

        # Centre degree 3 or more
        for centre_degree in range(3, n_vertex_max):
            #print(f"\ncentre_degree = {centre_degree}")
            #res2 += 1 # Every vertex other than centre is a leaf
            for depth in range(1, d_mx + 1):
                #print(f"depth = {depth}")
                for tot_n_v in range((depth + 1) * 2 + 1 + (centre_degree - 2), n_vertex_max + 1):
                    #print(f"tot_n_v = {tot_n_v}")
                    for n_v1 in range(depth + 1, tot_n_v - 1 - (depth + 1) - (centre_degree - 2) + 1):
                        #print(f"n_v1 = {n_v1}")
                        curr2 = getRootedCountRootDegreeCumu(n_v1, n_v1, depth)
                        if n_v1 >= centre_degree:
                            curr2 -= getRootedCount(centre_degree, n_v1, depth)
                        v_remain = tot_n_v - n_v1 - 1
                        #print(f"v_remain = {v_remain}, n_v2 range = [{depth + 1}, {min(n_v1, v_remain - centre_degree + 2) + 1})")
                        #n_v2_mn = max(depth + 1, )
                        for n_v2 in range(depth + 1, min(n_v1, v_remain - centre_degree + 2) + 1):
                            #print(f"n_v2 = {n_v2}")
                            n_rpt = 1
                            if n_v2 == n_v1:
                                curr3 = curr2 + 1
                                n_rpt += 1
                            else:
                                curr3 = getRootedCountRootDegreeCumu(n_v2, n_v2, depth)
                                if n_v2 >= centre_degree:
                                    curr3 -= getRootedCount(centre_degree, n_v2, depth)
                            #print(f"calling recur() with curr = {curr2 * curr3}, v_remain = {v_remain - n_v2}, prev_depth = {depth + 1}, n_v2 = {n_v2}, root_degree = {centre_degree}")
                            ans0 = recur(1, curr2 * curr3, v_remain - n_v2, depth + 1, n_v2, centre_degree, n_rpt=n_rpt)
                            ans = ans0 // n_rpt
                            #print(f"centre_degree = {centre_degree}, depth = {depth}, tot_n_v = {tot_n_v}, n_v1 = {n_v1}, n_v2 = {n_v2}, curr2 = {curr2}, curr3 = {curr3}, count0 = {ans0}, count = {ans}")
                            res2 += ans
        print(f"\nsingle centre: {res2}\n")
        return res + res2

    return peerlessTreesWithMaxVertexCount(n_vertex_max) - peerlessTreesWithMaxVertexCount(n_vertex_min - 1)

    """
    for v in range(2, v_mx + 1):
        for d in range(1, min(d_mx, v) + 1):
            for deg0 in range(2, v - d + 1):
                if len(arr) <= deg0:
                    arr.append([])
                while len(arr[deg0]) <= v:
                    arr[deg0].append([])
                while len(arr[deg0][v] <= d):
                    arr[deg0][v].append(0)
    """


    #memo1 = {}
    #def rootedTreeWithMaxDepthUpToMaxVertices(max_depth: int, max_n_vertices: int, req_max_depth: bool=True) -> int:

def peerlessTreesWithVertexCountInRange2(n_vertex_min: int=3, n_vertex_max: int=50) -> int:
    """
    Solution to Project Euler #936
    """

    def recur(branch_remain: int, curr: int, v_remain: int, prev_n_v: int, root_deg: int, n_rpt: int=1) -> int:
        #print(branch_remain, curr, v_remain, prev_n_v, root_deg, n_rpt)
        if not curr: return 0
        if branch_remain == 1:
            #print("hello")
            
            if v_remain > prev_n_v: return 0
            res = getRootedCountRootDegreeCumu(root_deg_max=v_remain + 1, n_v=v_remain)
            #print(f"res = {res}")
            if v_remain + 1 >= root_deg:
                res -= getRootedCount(root_deg=root_deg, n_v=v_remain)
            #print(res)
            if v_remain == prev_n_v:
                #div = n_rpt + 1
                res = (curr * (res + n_rpt)) // (n_rpt + 1)
            else: res *= curr
            #print(f"res = {res}")
            return res
        res = 0
        min_n_v = (v_remain - 1) // branch_remain + 1
        max_n_v = min(prev_n_v, v_remain - (branch_remain - 1))
        for n_v in range(min_n_v, max_n_v - (max_n_v == prev_n_v) + 1):
            curr2 = getRootedCountRootDegreeCumu(root_deg_max=n_v, n_v=n_v)
            if n_v >= root_deg:
                curr2 -= getRootedCount(root_deg, n_v)
            if not curr2: continue
            res += recur(branch_remain - 1, curr * curr2, v_remain - n_v, n_v, root_deg, n_rpt=1)
        if max_n_v == prev_n_v:
            n_v = prev_n_v
            n_rpt2 = n_rpt + 1
            curr2 = getRootedCountRootDegreeCumu(root_deg_max=n_v, n_v=n_v) + n_rpt
            if n_v >= root_deg:
                curr2 -= getRootedCount(root_deg, n_v)
            if curr2:
                res += recur(branch_remain - 1, curr * curr2, v_remain - n_v, n_v, root_deg, n_rpt=n_rpt2) // n_rpt2
        return res

    memo1 = {}
    def getRootedCount(root_deg: int, n_v: int) -> int:
        if root_deg < 1 or n_v < 1: return 0
        elif root_deg == 1:
            #print("hi2")
            return int(n_v == 1)
        args = (root_deg, n_v)
        if args in memo1.keys():
            return memo1[args]
        ref = None#(4, 4, 1)
        if (root_deg, n_v) == ref:
            print("***************************")
        if root_deg == 2:
            # Exactly one sub-tree
            
            res = getRootedCountRootDegreeCumu(n_v - 1, n_v - 1)
            if n_v - 1 >= root_deg:
                res -= getRootedCount(root_deg, n_v - 1)
            if args == ref:
                print(res)
            memo1[args] = res
            return res
        
        res = 0
        if args == ref:
            print(f"n_v2 range: [1, {n_v - root_deg + 2})")

        res = recur(root_deg - 1, 1, n_v - 1, n_v, root_deg, n_rpt=1)
        #for n_v2 in range(1, n_v - root_deg + 2):
        #    
        #    curr = getRootedCountRootDegreeCumu(n_v2, n_v2)
        #    if n_v2 >= root_deg:
        #        curr -= getRootedCount(root_deg, n_v2)
        #    #print(f"n_v2 = {n_v2}, curr = {curr}")
        #    if not curr: continue
        #    ans = recur(root_deg - 2, curr, n_v - n_v2 - 1, n_v2, root_deg, n_rpt=1)
        #    if args == ref:
        #        print(f"for n_v2 = {n_v2}, ans = {ans}")
        #
        #    res += ans

        memo1[args] = res
        if args == ref:
            print("***************************")
            print(f"res = {res}")
        return res 

    memo2 = {}
    def getRootedCountCumu(root_deg_max: int, n_v_max: int) -> int:
        if root_deg_max < 1 or n_v_max < 1: return 0
        args = (root_deg_max, n_v_max)
        if args in memo2.keys(): return memo2[args]
        res = getRootedCount(root_deg_max, n_v_max) + getRootedCountCumu(root_deg_max, n_v_max - 1) +\
            getRootedCountCumu(root_deg_max - 1, n_v_max) - getRootedCountCumu(root_deg_max - 1, n_v_max - 1)
        memo2[args] = res
        return res

    memo3 = {}
    def getRootedCountRootDegreeCumu(root_deg_max: int, n_v: int) -> int:
        if root_deg_max < 1 or n_v < 1: return 0
        args = (root_deg_max, n_v)
        if args in memo3.keys(): return memo3[args]
        res = getRootedCount(root_deg_max, n_v) + getRootedCountRootDegreeCumu(root_deg_max - 1, n_v)
        memo3[args] = res
        return res

    memo4 = {}
    def getRootedCountVertexCountCumu(root_deg: int, n_v_max: int) -> int:
        if root_deg < 1 or n_v_max < 1: return 0
        args = (root_deg, n_v_max)
        if args in memo4.keys(): return memo4[args]
        res = getRootedCount(root_deg, n_v_max) + getRootedCountVertexCountCumu(root_deg, n_v_max - 1)
        memo4[args] = res
        return res

    def peerlessTreesWithNumberOfVerticesCount(n_v: int) -> int:
        if n_v < 1 or n_v == 2: return 0
        elif n_v == 1: return 1

        n_v_hlf = n_v >> 1

        # Tree centroid is two adjacent nodes
        res1 = 0
        if not n_v & 1:
            for deg1 in range(3, n_v_hlf + 1):
                mult = getRootedCount(root_deg=deg1, n_v=n_v_hlf)
                ans = mult * getRootedCountRootDegreeCumu(root_deg_max=deg1 - 1, n_v=n_v_hlf)
                #print(f"n_v = {n_v}, centroid pair, deg1 = {deg1}, total = {ans}")
                res1 += ans
            #print(f"centroid pair count = {res1}")
        
        # Tree centroid is a single node
        res2 = 0
        for root_deg in range(2, n_v):
            ans = recur(root_deg, 1, n_v - 1, n_v_hlf - (not n_v & 1), root_deg, n_rpt=0)
            #print(f"n_v = {n_v}, single centroid, root_deg = {root_deg}, total = {ans}")
            res2 += ans
        return res1 + res2
    
    res = 0
    for n_v in range(n_vertex_min, n_vertex_max + 1):
        ans = peerlessTreesWithNumberOfVerticesCount(n_v)
        res += ans
        print(n_v, ans, res)
    return res

# Problem 937
def calculateFactorialsInEquiproductPartitionWithUnit(n_max: int) -> List[int]:
    ps = PrimeSPFsieve()
    def factorsPrimeFactorisationPowersGenerator(p_pows: List[int]) -> Generator[List[int], None, None]:
        n_p = len(p_pows)
        curr = [0] * n_p
        def recur(idx: int, non_mid_seen: bool=False) -> Generator[List[int], None, None]:
            if idx == n_p:
                yield list(curr)
                return
            if non_mid_seen:
                for i in range(p_pows[idx] + 1):
                    curr[idx] = i
                    yield from recur(idx + 1, non_mid_seen=True)
                return
            for i in range((p_pows[idx] + 1) >> 1):
                curr[idx] = i
                yield from recur(idx + 1, non_mid_seen=True)
            if not p_pows[idx] & 1:
                curr[idx] = p_pows[idx] >> 1
                yield from recur(idx + 1, non_mid_seen=False)
            return
        
        yield from recur(0, non_mid_seen=False)
        return

    memo = {}
    def isInA(p_pows: List[int]) -> bool:
        p_pows2 = sorted(p_pows, reverse=True)
        #while p_pows2 and not p_pows2[-1]:
        #    p_pows2.pop()
        if not p_pows2: return True
        elif len(p_pows2) == 1: return False
        args = tuple(p_pows2)
        if args in memo.keys(): return memo[args]
        tot = 0
        it = iter(factorsPrimeFactorisationPowersGenerator(p_pows2))
        next(it)
        for f1_pows in it:
            f2_pows = [x - y for x, y in zip(p_pows, f1_pows)]
            if f1_pows >= f2_pows: continue
            #print(p_pows, f1_pows, f2_pows)
            b1, b2 = isInA(f1_pows), isInA(f2_pows)
            if b1 != b2: continue
            tot += 2 * b1 - 1
        res = (tot == -1)

        memo[args] = res
        return res
    
    memo2 = {}
    def isPSplit(p: int) -> bool:
        if p in memo2.keys(): return memo2[p]
        res = False
        for b in range(isqrt(p >> 1) + 1):
            
            a_sq = p - 2 * b ** 2
            #print(f"p = {p}, b = {b}, a_sq = {a_sq}")
            if isqrt(a_sq) ** 2 == a_sq:
                res = True
                break
        memo2[p] = res
        return res
    #print(isPSplit(2), isPSplit(3), isPSplit(5))
    """
    for num in range(1, 100):
        pf = ps.primeFactorisation(num)
        curr = {}
        for p, f in pf.items():
            curr[p] = curr.get(p, 0) + f
        lst = []
        for p, f in curr.items():
            if p == 2: lst.append(2 * f)
            elif isPSplit(p): lst.extend([f, f])
            else: lst.append(f)
        print(num, isInA(lst))
    return [1]
    """
    res = []
    curr = {}
    if isInA(list(curr.values())):
        print(1, math.factorial(1))
        res.append(1)
    for num in range(2, n_max + 1):
        print(f"num = {num}")
        pf = ps.primeFactorisation(num)
        for p, f in pf.items():
            curr[p] = curr.get(p, 0) + f
        lst = []
        for p, f in curr.items():
            if p == 2: lst.append(2 * f)
            elif isPSplit(p): lst.extend([f, f])
            else: lst.append(f)
        #print(num, curr, lst)
        if isInA(lst):
            print(f"solution found: {num}! = {math.factorial(num)}")
            res.append(num)
    #print(memo)
    return res

def calculateFactorialsInEquiproductPartitionWithUnitSum(n_max: int=10 ** 8, md: Optional[int]=10 ** 9 + 7) -> int:

    nums = calculateFactorialsInEquiproductPartitionWithUnit(n_max)
    print(nums)
    addMod = (lambda x, y: x + y) if md is None else (lambda x, y: (x + y) % md)
    multMod = (lambda x, y: x * y) if md is None else (lambda x, y: (x * y) % md)

    curr = 1
    res = 0
    i = 0
    for j in range(1, nums[-1] + 1):
        curr = multMod(curr, j)
        #print(nums[i], j)
        if nums[i] == j:
            res = addMod(res, curr)
            i += 1

    return res

# Problem 938
def redBlackCardGameLastCardBlackProbabilityFraction(n_red_init: int, n_black_init: int) -> "CustomFraction":

    prev = [0] * (n_red_init + 1)
    row = [0] * (n_red_init + 1)
    row[0] = 1
    row[1] = 0
    for n_red in range(2, n_red_init + 1):
        tot = n_red + 1
        denom = (n_red + 1) * n_red
        #p_2red = CustomFraction(n_red - 1, n_red + 1)
        #p_redblack = CustomFraction(2, n_red + 1)
        row[n_red] = CustomFraction(n_red - 1, n_red + 1) * row[n_red - 2]# + CustomFraction(2, n_red + 1) * prev[n_red]


    for n_black in range(2, n_black_init + 1):
        if not n_black % 100:
            print(f"n_black = {n_black} of {n_black_init}")
        prev = row
        row = [0] * (n_red_init + 1)
        row[0] = 1
        m = 1 - CustomFraction((n_black - 1), n_black + 1)
        k = CustomFraction(2, n_black + 1) * prev[1]
        row[1] = k / m
        #denom = n_black + 1
        for n_red in range(2, n_red_init + 1):
            tot = n_red + n_black
            denom = tot * (tot - 1)
            p_2red = CustomFraction(n_red * (n_red - 1), denom)
            p_2black = CustomFraction(n_black * (n_black - 1), denom)
            p_redblack = CustomFraction(2 * n_red * n_black, denom)
            m = 1 - p_2black
            k = p_2red * row[n_red - 2] + p_redblack * prev[n_red]
            row[n_red] = k / m
    return row[-1]


    """
    memo = {}
    def recur(n_red: int, n_black: int) -> CustomFraction:
        if not n_red: return CustomFraction(1, 1)
        elif not n_black: return CustomFraction(0, 1)
        args = (n_red, n_black)
        if args in memo.keys(): return memo[args]
        tot = n_red + n_black
        denom = tot * (tot - 1)
        p_2red = CustomFraction(n_red * (n_red - 1), denom)
        p_2black = CustomFraction(n_black * (n_black - 1), denom)
        p_redblack = CustomFraction(2 * n_red * n_black, denom)
        #print(f"total probability = {p_2red + p_2black + p_redblack}")
        slf_mult = CustomFraction(1, 1)
        res = CustomFraction(0, 1)
        if p_2red != 0:
            res += p_2red * recur(n_red - 2, n_black)
        if p_2black != 0:
            slf_mult -= p_2black
        if p_redblack != 0:
            res += p_redblack * recur(n_red, n_black - 1)
        res /= slf_mult
        memo[args] = res
        return res
    res = recur(n_red_init, n_black_init)
    #print(memo)
    return res
    """
    
def redBlackCardGameLastCardBlackProbabilityFloat(n_red_init: int=24690, n_black_init: int=12345) -> float:
    """
    Solution to Project Euler #938
    """
    # Look into closed form solution
    prev = [0] * (n_red_init + 1)
    row = [0] * (n_red_init + 1)
    row[0] = 1
    row[1] = 0
    for n_red in range(2, n_red_init + 1):
        tot = n_red + 1
        denom = (n_red + 1) * n_red
        #p_2red = CustomFraction(n_red - 1, n_red + 1)
        #p_redblack = CustomFraction(2, n_red + 1)
        row[n_red] = ((n_red - 1) / (n_red + 1)) * row[n_red - 2]# + CustomFraction(2, n_red + 1) * prev[n_red]


    for n_black in range(2, n_black_init + 1):
        if not n_black % 10:
            print(f"n_black = {n_black} of {n_black_init}")
        prev = row
        row = [0] * (n_red_init + 1)
        row[0] = 1
        m = 1 - ((n_black - 1) / (n_black + 1))
        k = (2 / (n_black + 1)) * prev[1]
        row[1] = k / m
        #denom = n_black + 1
        for n_red in range(2, n_red_init + 1):
            tot = n_red + n_black
            denom = tot * (tot - 1)
            p_2red = (n_red * (n_red - 1) / denom)
            p_2black = (n_black * (n_black - 1) / denom)
            p_redblack = (2 * n_red * n_black / denom)
            m = 1 - p_2black
            k = p_2red * row[n_red - 2] + p_redblack * prev[n_red]
            row[n_red] = k / m
    return row[-1]
    """
    memo = {}
    def recur(n_red: int, n_black: int) -> float:
        if not n_red: return 1
        elif not n_black: return 0
        args = (n_red, n_black)
        if args in memo.keys(): return memo[args]
        tot = n_red + n_black
        denom = tot * (tot - 1)
        p_2red = n_red * (n_red - 1) / denom
        p_2black = n_black * (n_black - 1) / denom
        p_redblack = 2 * n_red * n_black / denom
        #print(f"total probability = {p_2red + p_2black + p_redblack}")
        slf_mult = 1
        res = 0
        if p_2red != 0:
            res += p_2red * recur(n_red - 2, n_black)
        if p_2black != 0:
            slf_mult -= p_2black
        if p_redblack != 0:
            res += p_redblack * recur(n_red, n_black - 1)
        res /= slf_mult
        memo[args] = res
        return res
    res = recur(n_red_init, n_black_init)
    #print(memo)
    return res
    """
    #res = redBlackCardGameLastCardBlackProbabilityFraction(n_red_init, n_black_init)
    #print(res)
    #return res.numerator / res.denominator

# Problem 939
def partisanNimNumberOfWinningPositionsBasic(max_n_stones: int) -> int:

    # Review- try to make fully iterative
    # Also look into the game theoretic solutions on the forum

    memo = {}
    def turn(player1_piles: Dict[int, int], player2_piles: Dict[int, int]) -> bool:
        if not player1_piles and not player2_piles:
            return False
        args = tuple(tuple((x, pile[x]) for x in sorted(pile.keys())) for pile in (player1_piles, player2_piles))
        if args in memo.keys(): return memo[args]
        res = False
        for num in list(player1_piles.keys()):
            player1_piles[num] -= 1
            if not player1_piles[num]: player1_piles.pop(num)
            if not turn(player2_piles, player1_piles):
                res = True
            player1_piles[num] = player1_piles.get(num, 0) + 1
        if res:
            memo[args] = res
            return res
        for num in list(player2_piles.keys()):
            player2_piles[num] -= 1
            if not player2_piles[num]: player2_piles.pop(num)
            if num > 1:
                player2_piles[num - 1] = player2_piles.get(num - 1, 0) + 1
            if not turn(player2_piles, player1_piles):
                res = True
            if num > 1:
                player2_piles[num - 1] -= 1
                if not player2_piles[num - 1]: player2_piles.pop(num - 1)
            player2_piles[num] = player2_piles.get(num, 0) + 1
            if res: break
        memo[args] = res
        return res

    res = []
    p1_dict = {}
    p2_dict = {}
    def recur(remain: int, sz: Optional[int]=None, p1: bool=True) -> None:
        if not remain: return
        if sz is None:
            sz = remain
        if not sz:
            if not p1: return
            p1 = False
            sz = remain
        p_dict = p1_dict if p1 else p2_dict
        recur(remain, sz - 1, p1)
        if remain < sz: return
        for f in range(1, (remain // sz) + 1):
            p_dict[sz] = f
            if turn(p1_dict, p2_dict) and not turn(p2_dict, p1_dict):
                res.append((dict(p1_dict), dict(p2_dict)))
            recur(remain - sz * f, sz - 1, p1)
        p_dict.pop(sz)
        return

    recur(max_n_stones, sz=max_n_stones, p1=True)
    #print(res)
    return len(res)

def partisanNimNumberOfWinningPositions(max_n_stones: int=5000, md: int=1234567891) -> int:

    sys.setrecursionlimit(10 ** 6)

    part_arr = [[1]]
    for i in range(1, max_n_stones + 1):
        part_arr.append([0])
        i_hlf = i >> 1
        #print(i, i_hlf)
        for j in range(1, i_hlf + 1):
            #print(j, i - j)
            #print(part_arr)
            part_arr[-1].append(part_arr[i - j][j] + part_arr[i - 1][j - 1])
        for j in range(i_hlf + 1, i + 1):
            part_arr[-1].append(part_arr[i - 1][j - 1])


    #memo = {}
    def nObjectsIntoKNonemptyBoxes(n: int, k: int) -> int:
        if n < 0 or k < 0 or n < k: return 0
        #print(n, k)
        return part_arr[n][k]
        """
        if not n and not k:
            return 1
        if n <= 0 or k <= 0:# or k > n:
            return 0
        args = (n, k)
        if args in memo.keys(): return memo[args]
        res = nObjectsIntoKNonemptyBoxes(n - k, k) + nObjectsIntoKNonemptyBoxes(n - 1, k - 1)
        if md is not None: res %= md
        memo[args] = res
        return res
        """

    memo2 = {}
    def countPartitionsWithUpToNObjectsAndUpToMMoreObjectsThanNonemptyBoxes(n: int, m: int) -> int:
        if n < 0 or m < 0: return 0
        m = min(m, n)
        args = (n, m)
        if args in memo2.keys(): return memo2[args]
        res = nObjectsIntoKNonemptyBoxes(n, n - m) +\
            countPartitionsWithUpToNObjectsAndUpToMMoreObjectsThanNonemptyBoxes(n - 1, m) +\
            countPartitionsWithUpToNObjectsAndUpToMMoreObjectsThanNonemptyBoxes(n, m - 1) -\
            countPartitionsWithUpToNObjectsAndUpToMMoreObjectsThanNonemptyBoxes(n - 1, m - 1)
        if md is not None: res %= md
        memo2[args] = res
        return res
    
    res = 0
    for n_extra1 in range(1, max_n_stones):
        border_cumu = [nObjectsIntoKNonemptyBoxes(i, i - n_extra1 + 1) for i in range(2)]
        for n_stones2 in range(2, max_n_stones - n_extra1):
            border_cumu.append(border_cumu[-2] + nObjectsIntoKNonemptyBoxes(n_stones2, n_stones2 - n_extra1 + 1))
            if md is not None: border_cumu[-1] %= md
        for n_stones1 in range(n_extra1 + 1, max_n_stones + 1):
            num = nObjectsIntoKNonemptyBoxes(n_stones1, n_stones1 - n_extra1)
            num2 = countPartitionsWithUpToNObjectsAndUpToMMoreObjectsThanNonemptyBoxes(max_n_stones - n_stones1, n_extra1 - 2)
            #print(num2)
            num2 += border_cumu[max_n_stones - n_stones1 - (max_n_stones & 1)]
            if md is not None: num2 %= md
            #for n_stones2 in range(n_stones1 & 1, max_n_stones - n_stones1 + 1, 2):
            #    #print(n_stones2, n_piles1 + 1 + (n_stones2 - n_stones1))
            #    num2 += nObjectsIntoKNonemptyBoxes(n_stones2, n_stones2 - n_extra1 + 1)
            #    if md is not None: num2 %= md
            #print(f"n_stones1 = {n_stones1}, n_piles1 = {n_piles1}, mult = {num}, n_player2_configs = {num2}, ans = {num * num2}")
            res += num * num2
            if md is not None: res %= md
    #print(partisanNimNumberOfWinningPositionsBasic(max_n_stones))
    return res

# Problem 940
def twoDimensionalRecurrenceFibonacciSum(k_min: int=2, k_max: int=50, md: Optional[int]=1123581313) -> int:

    addMod = (lambda a, b: a + b) if md is None else (lambda a, b: (a + b) % md)
    

    # Assumes, if given, that md is prime (uses FLT)
    inv13 = 1 / 13 if md is None else pow(13, md - 2, md)

    fib_arr = [0, 1]
    for _ in range(2, k_max + 1):
        fib_arr.append(fib_arr[-2] + fib_arr[-1])
    print(f"maximum argument = {fib_arr[-1]}")
    
    a1 = (3 + math.sqrt(13)) / 2
    a2 = (3 - math.sqrt(13)) / 2
    b1 = (1 + math.sqrt(13)) / 2
    b2 = (1 - math.sqrt(13)) / 2
    print(f"a1 = {a1}, a2 = {a2}, b1 = {b1}, b2 = {b2}")

    def floatPowerMod(base: float, exp: int, md: int, mult: float=1) -> float:
        if abs(base) <= 1: return (mult * base ** exp) % md
        base_neg = base < 0
        base = abs(base)
        mult_neg = mult < 0
        mult = abs(mult)
        res_neg = mult_neg
        if exp & 1 and base_neg: res_neg = not res_neg
        #print(base, exp, md)
        res = mult % md
        print(f"mult = {mult}, res = {res}")
        p = base
        exp2 = exp
        while exp2:
            if exp2 & 1:
                print(res, p)
                res = (res * p) % md
                #print(res, p)
                if exp2 == 1:
                    break
            p = (p * p) % md
            exp2 >>= 1
        print(base, exp, md, mult, res)
        print(f"res_neg = {res_neg}")
        return (md - res) if res_neg else res
    #print(floatPowerMod(2, 13, md))
    #print(f"inv13 = {inv13}, multiplied by 13 modulo md = {(inv13 * 13) % md}")

    def Apure(m: int, n: int) -> int:
        res = ((a1 - 1) * a1 ** m - (a2 - 1) * a2 ** m) * (b1 ** n - b2 ** n)
        #print(res)
        res += 3 * (a1 ** m - a2 ** m) * (b1 ** (n - 1) - b2 ** (n - 1))
        #print(res)
        res = round(res / 13)
        return res % md

    def inverseMod(num: int, md: int) -> int:
        # Using FLT
        return pow(num, md - 2, md)
    
    divideMod = (lambda a, b: a // b) if md is None else (lambda a, b: (a * inverseMod(b, md)) % md)

    def binomialOddCoefficientsGeneratorMod(n: int, md: int) -> Generator[Tuple[int, int], None, None]:
        if n < 1: return
        num = n % md
        i = -1
        for i in range(1, n - 1, 2):
            yield (i, num)
            num = (num * inverseMod((i + 1) * (i + 2), md)) % md
            num = (num * (n - i) * (n - i - 1)) % md
        yield (i + 2, num)
        return

    #inv3 = inverseMod(3, md)

    def addRoots(rt1: Tuple[int, int], rt2: Tuple[int, int], md: Optional[int]=None) -> Tuple[int, int]:
        res = (rt1[0] + rt2[0], rt1[1] + rt2[1])
        if md is not None:
            res = tuple(x % md for x in res)
        return res

    def subtractRoots(rt1: Tuple[int, int], rt2: Tuple[int, int], md: Optional[int]=None) -> Tuple[int, int]:
        res = (rt1[0] - rt2[0], rt1[1] - rt2[1])
        if md is not None:
            res = tuple(x % md for x in res)
        return res

    def multiplyRoots(rt1: Tuple[int, int], rt2: Tuple[int, int], m: int, md: Optional[int]=None) -> Tuple[int, int]:
        res = (rt1[0] * rt2[0] + m * rt1[1] * rt2[1], rt1[0] * rt2[1] + rt1[1] * rt2[0])
        if md is not None:
            res = tuple(x % md for x in res)
        return res

    def rootPower(rt: Tuple[int, int], m: int, exp: int, md: Optional[int]=None) -> int:
        res = (1, 0)
        p = rt
        exp2 = exp
        while exp2:
            if exp2 & 1:
                res = multiplyRoots(res, p, m, md=md)
                if exp2 == 1: break
            p = multiplyRoots(p, p, m, md=md)
            exp2 >>= 1
        return res

    a_div = 2
    b_div = 2
    a = (3, 1)
    b = (1, 1)
    ab_m = 13
    a1_pows = {}
    a2_pows = {}
    a_div_pows = {}
    b1_pows = {}
    b2_pows = {}
    b_div_pows = {}
    for idx in range(k_min, k_max + 1):
        exp = fib_arr[idx]
        if exp in a1_pows.keys(): continue
        a1_pows[exp] = rootPower(rt=a, m=ab_m, exp=exp, md=md)
        a2_pows[exp] = rootPower(rt=(a[0], -a[1]), m=ab_m, exp=exp, md=md)
        a_div_pows[exp] = pow(a_div, exp, mod=md)
        b1_pows[exp] = rootPower(rt=b, m=ab_m, exp=exp, md=md)
        b2_pows[exp] = rootPower(rt=(b[0], -b[1]), m=ab_m, exp=exp, md=md)
        b_div_pows[exp] = pow(b_div, exp, mod=md)
    
    res = 0
    for idx1 in range(k_min, k_max + 1):
        exp1 = fib_arr[idx1]
        for idx2 in range(k_min, k_max + 1):
            exp2 = fib_arr[idx2]
            t1 = multiplyRoots(a1_pows[exp1], b1_pows[exp2], ab_m, md=md)
            t2 = multiplyRoots(a2_pows[exp1], b2_pows[exp2], ab_m, md=md)
            rt_pow = subtractRoots(t1, t2, md=md)
            ans = rt_pow[1]
            div = b_div_pows[exp1] * b_div_pows[exp2]
            res = addMod(res, divideMod(ans, div))
    return res
    """
    if md is not None:
        m_term1_dict = {}
        m_term2_dict = {}
        n_term1_dict = {}
        n_term2_dict = {}
        inv9 = inverseMod(9, md)
        print("Precalculating m terms")
        for idx in range(k_min, k_max + 1):
            m = fib_arr[idx]
            print(f"m = {m}, Fibonacci number {idx} of {k_max}")
            if m in n_term1_dict.keys(): continue
            m_mult = (inv9 * 13) % md
            
            n_mult = 13 % md
            m_term1 = 0
            inv_mod_pow2_m = inverseMod(pow(2, m, md), md)
            curr = (pow(3, m, md) * inv_mod_pow2_m) % md
            m_term1_dict[m] = 0
            for i, (j, coef) in enumerate(binomialOddCoefficientsGeneratorMod(m + 1, md)):
                m_term1_dict[m] = (m_term1_dict[m] + coef * curr) % md
                curr = (curr * m_mult) % md
            curr = (pow(3, m - 1, md) * inv_mod_pow2_m * 2) % md
            m_term2_dict[m] = 0
            for i, (j, coef) in enumerate(binomialOddCoefficientsGeneratorMod(m, md)):
                m_term2_dict[m] = (m_term2_dict[m] + coef * curr) % md
                curr = (curr * m_mult) % md
        print("Precalculating n terms")
        for idx in range(k_min - 1, k_max + 1):
            n = fib_arr[idx]
            print(f"n = {n}, Fibonacci number {idx} of {k_max}")
            if n in n_term1_dict.keys(): continue
            inv_mod_pow2_n = inverseMod(pow(2, n - 1, md), md)
            curr = inv_mod_pow2_n
            n_term1_dict[n] = 0
            for i, (j, coef) in enumerate(binomialOddCoefficientsGeneratorMod(n, md)):
                n_term1_dict[n] = (n_term1_dict[n] + coef * curr) % md
                curr = (curr * n_mult) % md
            
            curr = (inv_mod_pow2_n * 6) % md
            n_term2_dict[n] = 0
            for i, (j, coef) in enumerate(binomialOddCoefficientsGeneratorMod(n - 1, md)):
                n_term2_dict[n] = (n_term2_dict[n] + coef * curr) % md
                curr = (curr * n_mult) % md
            
        #print(f"m_term1_dict = {m_term1_dict}")
        #print(f"m_term2_dict = {m_term2_dict}")
        #print(f"n_term1_lst = {n_term1_dict}")
        #print(f"n_term2_lst = {n_term2_dict}")
    """
    #def Amod(m: int, n: int, md: int) -> int:
        #ans1 = ((m_term1_dict[m] - m_term2_dict[m]) * n_term1_dict[n]) % md
        #ans2 = (m_term2_dict[m] * n_term2_dict[n]) % md
        #return (ans1 + ans2) % md
    """
        inv9 = inverseMod(9, md)
        m_mult = (inv9 * 13) % md
        n_mult = 13
        m_term1 = 0
        curr = pow(3, m, md)
        for i, (j, coef) in enumerate(binomialOddCoefficientsGeneratorMod(m + 1, md)):
            m_term1 = (m_term1 + coef * curr) % md
            curr = (curr * m_mult) % md
        m_term2 = 0
        curr = pow(3, m - 1, md)
        for i, (j, coef) in enumerate(binomialOddCoefficientsGeneratorMod(m, md)):
            m_term2 = (m_term2 + coef * curr) % md
            curr = (curr * m_mult) % md
        n_term1 = 0
        curr = 1
        for i, (j, coef) in enumerate(binomialOddCoefficientsGeneratorMod(n, md)):
            n_term1 = (n_term1 + coef * curr) % md
            curr = (curr * n_mult) % md
        n_term2 = 0
        curr = 1
        for i, (j, coef) in enumerate(binomialOddCoefficientsGeneratorMod(n - 1, md)):
            n_term2 = (n_term2 + coef * curr) % md
            curr = (curr * n_mult) % md
        inv_pow2 = inverseMod(pow(2, m + n - 1, md), md)
        ans1 = ((m_term1 - 2 * m_term2) * n_term1 * inv_pow2) % md
        ans2 = (3 * m_term2 * n_term2 * inv_pow2 * 4) % md
        return (ans1 + ans2) % md
    """
        
    """
        #print(m, n)
        #print(a1, m, md, a1 - 1, floatPowerMod(a1, m, md, mult=a1 - 1))
        #print(a2, m, md, a2 - 1, floatPowerMod(a2, m, md, mult=a2 - 1))
        print(m, n)
        res = (floatPowerMod(a1, m, md, mult=(a1 - 1) / 13) - (((a2 - 1) * a2 ** m) / 13)) % md#floatPowerMod(a2, m, md, mult=(a2 - 1) / 13)) % md
        print(res)
        res = (floatPowerMod(b1, n, md, mult=res) - (res * b2 ** n)) % md# - floatPowerMod(b2, n, md, mult=res)) % md
        print(res)
        res2 = (floatPowerMod(a1, m, md, mult=3 / 13) - ((3 * a2 ** m) / 13)) % md# - floatPowerMod(a2, m, md, mult=3 / 13)) % md
        res2 = (floatPowerMod(b1, (n - 1), md, mult=res2) - (res2 * b2 ** (n - 1))) % md# - floatPowerMod(b2, (n - 1), md, mult=res2)) % md
        print(res2)
        #res = (round((res + res2) % md) * inv13) % md
        res = round((res + res2) % md)
        return res
    """

    #A = Apure if md is None else lambda m, n: Amod(m, n, md=md)
    #A = Apure
    """
    res = 0
    res2 = 0
    for i in range(k_min, k_max + 1):
        print(f"i = {i} of {k_max}")
        f1 = fib_arr[i]
        for j in range(k_min, k_max + 1):
            f2 = fib_arr[j]
            ans = A(f1, f2)
            #print(f"A({f1}, {f2}) = {ans}")
            res += ans
            #ans2 = Apure(f1, f2)
            #print(f"Apure({f1}, {f2}) = {ans2}")
            #res2 = addMod(res2, ans2)
    #print(res, res2)
    return res
    """
    """
    appendMod = (lambda lst, val: lst.append(val)) if md is None else (lambda lst, val: lst.append(val % md))
    addMod = (lambda a, b: a + b) if md is None else (lambda a, b: (a + b) % md)

    fib_arr = [0, 1]
    for i in range(2, k_max + 1):
        appendMod(fib_arr, fib_arr[-2] + fib_arr[-1])
    print(fib_arr[-1])
    res = 0
    n_max = fib_arr[-1]
    row = [0, 1]
    nxt_idx = k_min
    for _ in range(2, n_max + 1):
        appendMod(row, 3 * row[-2] + row[-1])
        if row[-1] == 1 and row[-2] == 0: print("here")
    while nxt_idx < len(fib_arr) and fib_arr[nxt_idx] < 0:
        nxt_idx += 1
    if nxt_idx < len(fib_arr) and fib_arr[nxt_idx] == 0:
        lst = []
        for k_ in range(k_min, k_max + 1):
            res = addMod(res, row[fib_arr[k_]])
            lst.append(row[fib_arr[k_]])
        #print(0, lst)
    print(0, row)
    #print(nxt_idx, row)
    for m in range(1, n_max + 1):
        prev = row
        row = []
        for i in range(n_max):
            appendMod(row, prev[i + 1] + prev[i])
        appendMod(row, 2 * row[-1] + prev[-2])
        while nxt_idx < len(fib_arr) and fib_arr[nxt_idx] < m:
            nxt_idx += 1
        if nxt_idx < len(fib_arr) and fib_arr[nxt_idx] == m:
            lst = []
            for k_ in range(k_min, k_max + 1):
                res = addMod(res, row[fib_arr[k_]])
                lst.append(row[fib_arr[k_]])
            #print(m, lst)
        print(m, row)
    return res
    """

# Problem 941
def isStringPrimitive(s: Iterable) -> bool:
    kmp = KnuthMorrisPratt(s)
    l = len(s)
    lps = kmp.lps
    return lps[-1] < (l >> 1)

def calculateStringPrimitiveRoot(s: Iterable) -> Iterable:
    #print(s)
    kmp = KnuthMorrisPratt(s)
    l = len(s)
    lps = kmp.lps
    if lps[-1] < (l >> 1): return s
    return s[:(l - lps[-1])]

def calculateNumberLyndonWordIndexInLexicographicallySmallestDeBruijnSequence(num: int, w_len: int, base: int=10) -> Tuple[Tuple[int, int], int]:
    # Using https://arxiv.org/pdf/1510.02637

    # Review- try to implement finding the index in the De Bruijn sequence
    # from the above paper
    # Review- try to make faster
    mult = base ** (w_len - 1)

    def cycleWord(num: int, num_len: int) -> int:
        num2, d = divmod(num, base)
        return d * base ** (num_len - 1) + num2

    def getSelfMinimalWord(num: int, num_len: int) -> Tuple[int, int]:
        res = (num, 0)
        num2 = num
        #print(f"num = {format(num, 'b')}, num_len = {num_len}")
        for i in range(1, num_len):
            num2 = cycleWord(num2, num_len)
            #print(f"num2 = {format(num2, 'b')}")
            res = min(res, (num2, i))
        return res
    
    def isPrimitive(num: int, length: int) -> bool:
        s = []
        num2 = num
        for _ in range(length):
            num2, d = divmod(num2, base)
            s.append(d)
        return isStringPrimitive(s)

    def calculatePrimitiveRoot(num: int, length: int) -> Tuple[int, int]:
        s = []
        num2 = num
        #print(f"length = {length}")
        for _ in range(length):
            num2, d = divmod(num2, base)
            s.append(d)
        digs = calculateStringPrimitiveRoot(s[::-1])
        res = 0
        for d in digs:
            res = res * base + d
        return (res, len(digs))

    def calculateNextLyndon(lynd: int, lynd_len: int, n: int) -> Tuple[int, int]:
        # Using Duval (1988)
        #mult2 = base ** lynd_len
        num = lynd
        curr_len = lynd_len
        while curr_len < n:
            num = (base ** curr_len + 1) * num
            curr_len <<= 1
        num //= base ** (curr_len - n)
        curr_len = n
        while num % base == base - 1:
            num //= base
            curr_len -= 1
        return num + 1, curr_len

    def calculateNextLyndonInDeBruijn(lynd: int, lynd_len: int, n: int) -> Tuple[int, int]:
        num = lynd
        num_len = lynd_len
        while True:
            num, num_len = calculateNextLyndon(num, num_len, n)
            if not n % num_len: break
        return num, num_len
    
    # Try to find a more efficient method
    def calculatePreviousLyndon(lynd: int, lynd_len: int, n: int) -> Tuple[int, int]:
        #print("Using calculatePreviousLyndon()")
        #print(lynd, lynd_len, n, format(lynd, "b"))
        # Brute force
        #num = lynd
        #curr_len = lynd_len
        num = lynd * base ** (n - lynd_len) - 1
        curr_len = n
        #while curr_len < n:
        #    num = (base ** curr_len + 1) * num
        #    curr_len <<= 1
        #print(num)
        d0 = num // (base ** (curr_len))
        #num -= 1
        #print(format(num, "b"), curr_len)
        seen = set()
        while True:
            while num in seen: num -= 1
            seen.add(num)
            #print(num)
            #print(num, format(num, "b"),)
            while curr_len > 1 and num % base <= d0:
                num //= base
                curr_len -= 1
            #print(num, format(num, "b"), curr_len)
            if not n % curr_len and numberIsSelfMinimal(num, curr_len):
                return num, curr_len
            while curr_len < n:
                num = (base ** curr_len + 1) * num
                curr_len <<= 1
            num //= base ** (curr_len - n)
            curr_len = n
            num -= 1
        """
        # Using Duval (1988)
        mult2 = base ** lynd_len
        suff = 0
        num = lynd - 1
        curr_len = lynd_len
        while curr_len < n:
            num = (base ** curr_len + 1) * num
            curr_len <<= 1
        num //= base ** (curr_len - n)
        curr_len = n
        while not num % base:
            num //= base
            curr_len -= 1
        return num, curr_len
        """

    def calculatePreviousLyndonInDeBruijn(lynd: int, lynd_len: int, n: int) -> Tuple[int, int]:
        num = lynd
        num_len = lynd_len
        while True:
            num, num_len = calculatePreviousLyndon(num, num_len, n)
            if not n % num_len: break
        return num, num_len

    def calculateSubstringPosition(num: int, num_len: int, substring_num: int, substring_num_len: int) -> int:
        num_lst = []
        substring_num_lst = []
        num2 = substring_num
        for _ in range(substring_num_len):
            num2, d = divmod(num2, base)
            substring_num_lst.append(d)
        kmp = KnuthMorrisPratt(substring_num_lst[::-1])
        num2 = num
        for _ in range(num_len):
            num2, d = divmod(num2, base)
            num_lst.append(d)
        for i in kmp.matchStartGenerator(num_lst[::-1]):
            return i
        return -1

    def numberIsSelfMinimal(num: int, num_len: int) -> bool:
        #print("Using numberIsSelfMinimal()")
        #print(num, num_len, format(num, "b"))
        if num_len == 1: return True
        num_lst = []
        num2 = num
        for _ in range(num_len):
            num2, d = divmod(num2, base)
            num_lst.append(d)
        num_lst = num_lst[::-1]
        if num_lst[0] >= num_lst[-1]:
            #print("false")
            return False
        kmp = KnuthMorrisPratt(num_lst)
        if kmp.lps[-1] != 0:
            #print("false")
            return False
        #print(num_lst)
        #print("lps:")
        #print(kmp.lps)
        for i in range(1, num_len - 1):
            j = kmp.lps[i]
            #print(j, i + 1)
            if num_lst[j] > num_lst[i + 1]:
                #print("false")
                return False
        #print("true")
        return True

    def numberIsLyndon(num: int, num_len: int) -> bool:
        if not isPrimitive(num, num_len): return False
        return numberIsSelfMinimal(num, num_len)

    def numberIsLyndonInDeBruijn(num: int, num_len: int, n: int) -> bool:
        
        if n % num_len: return False
        return numberIsLyndon(num, num_len)

    def findLargestSmallerSelfMinimalWordOfGivenLength(num: int, num_len: int, target_len: int) -> Tuple[int, int]:
        #print(num, num_len)
        res = -1
        mult2 = 1
        num2 = num
        num2_len = num_len
        while num2_len < target_len:
            num2 = num2 * (base ** num2_len + 1)
            num2_len <<= 1
        #print(format(num2, "b"), num2_len)
        num2 //= base ** (num2_len - target_len)
        num2_len = target_len
        #print(f"rescaled num = {format(num2, 'b')}, {num2_len}")
        for _ in reversed(range(num2_len)):
            #print(f"num2 = {format(num2, 'b')}, mult2 = {mult2}")
            d = num2 % base
            if d:
                num2 -= 1
                num3 = (num2 + 1) * mult2 - 1
                #print(f"num3 = {num3}, {format(num3, 'b')}")
                if num3 <= res: continue
                if numberIsSelfMinimal(num3, target_len):
                    res = num3
            mult2 *= base
            num2 //= base
        return (res, target_len) if res >= 0 else (-1, 0)

    def lexMax(num1: int, num1_len: int, num2: int, num2_len: int) -> Tuple[int, int]:
        pair1, pair2 = (num1, num1_len), (num2, num2_len)
        if num1_len < num2_len: pair1, pair2 = pair2, pair1
        m1 = pair1[0]
        m2 = pair2[0] * base ** (pair1[1] - pair2[1])
        if m1 >= m2: return pair1
        return pair2

    #def calculateDeBruijnPrefixLengthUpToLyndonWord(lynd_w: int, lynd_w_len: int) -> int:
    #    lynd_lst = []
    #    num2 = lynd_w
    #    for _ in range(lynd_w_len):
    #        num2, d = divmod(num2, base)
    #        lynd_lst.append(d)
    #    lynd_lst = lynd_lst[::-1]
    #    w_lst = lynd_lst * (w_len // lynd_w_len)
    num_lst = []
    num2 = num
    for _ in range(w_len):
        num2, d = divmod(num2, base)
        num_lst.append(d)
    num_lst = num_lst[::-1]
    #print(f"\ndigits: {num_lst}")
    
    if not num: return ((0, 1), 0)
    num2 = num
    i = w_len
    while not num2 % base:
        num2 //= base
        i -= 1
    if num2 == base ** i - 1:
        # Case a
        #print("case a")
        #print(format(num, "b"), w_len)
        if i == 1: return ((1, 1), 0)
        lynd = calculatePreviousLyndonInDeBruijn(1, 1, w_len)
        #print(f"lynd = {lynd}, {format(lynd[0], 'b')}")
        return (lynd, lynd[1] - i + 1)
    prim, prim_len = calculatePrimitiveRoot(num, w_len)
    #print(format(prim, "b"), prim_len)
    lynd_k, beta_len = getSelfMinimalWord(prim, prim_len)
    #print(format(lynd_k, "b"), beta_len)
    alpha_len = prim_len - beta_len
    alpha, beta = divmod(prim, base ** beta_len)
    #print(f"alpha = {format(alpha, 'b')}, {alpha_len}, beta = {format(beta, 'b')}, {beta_len}")
    
    if alpha != base ** alpha_len - 1:
        # Case b
        #print("case b")
        lynd1, lynd1_len = lynd_k, prim_len
        lynd2, lynd2_len = calculateNextLyndonInDeBruijn(lynd1, lynd1_len, w_len)
        num2 = base ** lynd2_len * lynd1 + lynd2
        num2_len = lynd1_len + lynd2_len
    else:
        d = w_len // prim_len
        if d > 1:
            # Case c
            #print("case c")
            lynd2, lynd2_len = lynd_k, prim_len
        else:
            # Case d
            #print("case d")
            #print(f"beta = {format(beta, 'b')}, beta_len = {beta_len}")
            
            num2, num2_len = findLargestSmallerSelfMinimalWordOfGivenLength(beta, beta_len, w_len)
            #print(f"num2 = {format(num2, 'b')}, num2_len = {num2_len}")
            ans = calculatePrimitiveRoot(num2, num2_len) if num2_len > 0 else (-1, 0)
            beta2 = beta
            for beta2_len in reversed(range(1, beta_len)):
                beta2 //= base
                if numberIsLyndonInDeBruijn(beta2, beta2_len, w_len):
                    ans = lexMax(*ans, beta2, beta2_len)
                    break
            #print(ans)
            lynd2, lynd2_len = ans
        lynd1, lynd1_len = calculatePreviousLyndonInDeBruijn(lynd2, lynd2_len, w_len)
        lynd3, lynd3_len = calculateNextLyndonInDeBruijn(lynd2, lynd2_len, w_len)
        #print(f"lynd1 = {(lynd1, lynd1_len)}, lynd2 = {(lynd2, lynd2_len)}, lynd3 = {(lynd3, lynd3_len)}")
        #print(format(lynd1, "b"), format(lynd2, "b"), format(lynd3, "b"))
        num2 = base ** lynd2_len * lynd1 + lynd2
        num2 = base ** lynd3_len * num2 + lynd3
        num2_len = lynd1_len + lynd2_len + lynd3_len
        #print(num2, num2_len)
    m2 = num2
    num_lst = []
    for _ in range(num2_len):
        m2, d = divmod(m2, base)
        num_lst.append(d)
    #print(format(num, "b"), w_len)
    #print(f"lyndon concatenation digits: {num_lst[::-1]}")
    i = calculateSubstringPosition(num2, num2_len, num, w_len)
    if i < 0: raise ValueError("Unexpected error occurred- num was not found in the expected location")
    if i < lynd1_len: return ((lynd1, lynd1_len), i)
    elif i < lynd1_len + lynd2_len: return ((lynd2, lynd2_len), i - lynd1_len)
    return ((lynd3, lynd3_len), i - lynd1_len - lynd2_len)

def calculateOrderOfNumbersInLexicographicallySmallestDeBruijnSequence(nums: Iterable[int], w_len: int, base: int=10) -> List[int]:
    lst = []
    for num in nums:
        num2 = num
        num_lst = []
        for _ in range(w_len):
            num2, d = divmod(num2, base)
            num_lst.append(d)
        if num2: print("too many digits")
        #print(num_lst[::-1])
        lynd, idx = calculateNumberLyndonWordIndexInLexicographicallySmallestDeBruijnSequence(num, w_len, base=base)
        #print(num, lynd)
        num2 = lynd[0]
        num_lst = []
        for _ in range(lynd[1]):
            num2, d = divmod(num2, base)
            num_lst.append(d)
        #print(f"lyndon digits: {num_lst[::-1]}, start index = {idx}")
        num2 = 0
        mult2 = (base ** lynd[1])
        for _ in range(w_len // lynd[1]):
            num2 = num2 * mult2 + lynd[0]
        lst.append((num2, idx, num))
    lst.sort()
    return [x[2] for x in lst]

def calculateOrderSumOfNumbersInLexicographicallySmallestDeBruijnSequence(nums: Iterable[int], w_len: int, base: int=10, md: Optional[int]=None) -> List[int]:
    res = 0
    for i, num in enumerate(calculateOrderOfNumbersInLexicographicallySmallestDeBruijnSequence(nums, w_len, base=base), start=1):
        res += i * num
        if md is not None: res %= md
    return res

def linearRecurrencePseudoRandomNumberGenerator(
    a0: int=0,
    a_min: int=0,
    a_max: int=10 ** 12 - 1,
    k: int=920461,
    m: int=800217387569,
) -> Generator[int, None, None]:
    a = a0
    md = a_max - a_min + 1
    while True:
        a = (k * a + m) % md
        yield a + a_min
    return

def calculateOrderSumOfLinearRecurrencePseudoRandomNumbersInLexicographicallySmallestDeBruijnSequence(
    n_nums: int=10 ** 7,
    w_len: int=12,
    a0: int=0,
    k: int=920461,
    m: int=800217387569,
    base: int=10,
    md: Optional[int]=1234567891,
)-> List[int]:
    """
    Solution to Project Euler #941
    """
    a_min = 0
    a_max = base ** w_len - 1
    
    def numberGenerator() -> Generator[int, None, None]:
        it = iter(linearRecurrencePseudoRandomNumberGenerator(
            a0=a0,
            a_min=a_min,
            a_max=a_max,
            k=k,
            m=m,
        ))
        for _ in range(n_nums):
            yield next(it)
        return
    
    return calculateOrderSumOfNumbersInLexicographicallySmallestDeBruijnSequence(iter(numberGenerator()), w_len=w_len, base=base, md=md)

# Problem 944
def sumOfSubsetElevisorsBruteForce(n_max: int) -> int:
    res = 0
    for bm in range(1, 1 << n_max):
        ss = set()
        for i in reversed(range(1, n_max + 1)):
            if bm & 1:
                for num in range(i << 1, n_max + 1, i):
                    if num in ss:
                        res += i
                        break
                ss.add(i)
            bm >>= 1
    return res

def sumOfSubsetElevisorsBruteForce2(n_max: int) -> int:
    res = 0
    for num in range(1, n_max + 1):
        if not num % 10 ** 5:
            print(f"phase 1, num = {num} of {n_max - 1}")
        n_mults = n_max // num
        res = res + num * (pow(2, n_max - 1) - pow(2, n_max - n_mults))
    return res

def sumOfSubsetElevisors(n_max: int=10 ** 14, md: Optional[int]=1234567891) -> int:
    """
    Solution to Project Euler #944
    """
    rt = isqrt(n_max)
    res = 0
    mx = rt + 1
    if rt * rt == n_max:
        res = res + rt * (pow(2, n_max - 1, mod=md) - pow(2, n_max - rt, mod=md))
        if md is not None: res %= md
        mx -= 1
    for num in range(1, mx):
        if not num % 10 ** 5:
            print(f"phase 1, num = {num} of {mx - 1}")
        n_mults = n_max // num
        res = res + num * (pow(2, n_max - 1, mod=md) - pow(2, n_max - n_mults, mod=md))
        if md is not None: res %= md
    #print(res)
    for num in range(2, mx):
        
        if not num % 10 ** 5:
            print(f"phase 2, num = {num} of {mx - 1}")
        rgt = (n_max) // num
        lft = max((n_max) // (num + 1), rt) + 1
        #print(num, lft, rgt)
        if rgt < lft: break
        mult = (rgt * (rgt + 1) - lft * (lft - 1)) >> 1
        if md is not None: mult %= md
        ans = mult * (pow(2, n_max - 1, mod=md) - pow(2, n_max - num, mod=md))
        res = res + ans
        #print(num, mn, mx, mult, res)
        if md is not None: res %= md
    return res

# Problem 945
def xorMultiply(num1: int, num2: int) -> int:
    if num1.bit_count() < num2.bit_count(): num1, num2 = num2, num1
    res = 0
    while num2:
        if num2 & 1:
            res ^= num1
        num2 >>= 1
        num1 <<= 1
    return res

def xorEquationNontrivialPrimitiveSolutionsGenerator(a_b_max: int) -> Generator[Tuple[int, int, int], None, None]:

    sq_lst = [0]

    sq_nxt = 1
    for b in range(1, a_b_max + 1):
        if b >= len(sq_lst):
            sq_lst.append(xorMultiply(sq_nxt, sq_nxt))
            sq_nxt += 1
        b_sq = sq_lst[b]
        for a in range(1 + (b & 1), b + 1, 2):
            #if gcd(a, b) > 1: continue
            a_sq = sq_lst[a]
            num = a_sq ^ xorMultiply(2, xorMultiply(a, b)) ^ b_sq
            while num > sq_lst[-1]:
                sq_lst.append(xorMultiply(sq_nxt, sq_nxt))
                sq_nxt += 1
            c = bisect.bisect_left(sq_lst, num)
            if c < len(sq_lst) and sq_lst[c] == num:
                yield (a, b, c)
    print(sq_lst)
    return

def xorEquationSolutionsCount(a_b_max: int=10 ** 7) -> int:
    def solutionCheck(a, b, c) -> bool:
        return xorMultiply(a, a) ^ xorMultiply(2, xorMultiply(a, b)) ^ xorMultiply(b, b) == xorMultiply(c, c)

    res = a_b_max + 1
    ab_pairs = {}
    for triple in xorEquationNontrivialPrimitiveSolutionsGenerator(a_b_max):
        #print(triple)
        a, b = triple[0], triple[1]
        mult = a_b_max // b
        res += mult.bit_length()
        #while mult:
        #    res += 1
        #    mult >>= 1
        ab_pairs.setdefault(a, [])
        ab_pairs[a].append(b)
        ab_pairs.setdefault(b, [])
        ab_pairs[b].append(a)
        #res += a_b_max // b
        #for mult in range(2, (a_b_max // b) + 1):
        #    triple2 = tuple(x * mult for x in triple)
        #    if not solutionCheck(*triple2):
        #        print(f"Multiple of solution is not a solution: {triple2} = {mult} * {triple}")
    print("Pairs:")
    for odd in sorted(ab_pairs.keys()):
        if not odd & 1: continue
        print(f"{format(odd, 'b')}: {[format(x, 'b') for x in sorted(ab_pairs[odd])]}")
    return res

# Problem 946
def continuedFractionRationalExpression(cf: Iterable[int], a: int, b: int, c: int, d: int) -> Generator[int, None, None]:

    # Using algorithm from https://perl.plover.com/classes/cftalk/TALK

    #print("hi")
    it = iter(cf)

    def outputLoop(curr: Tuple[int, int, int, int]) -> Tuple[List[int], Tuple[int, int, int, int]]:
        res = []
        while curr[2] and curr[3]:
            q1 = curr[0] // curr[2]
            q2 = curr[1] // curr[3]
            if q1 != q2: break
            res.append(q1)
            #print("output", q1, curr)
            curr = (curr[2], curr[3], curr[0] - curr[2] * q1, curr[1] - curr[3] * q1)
        return (res, curr)


    curr = (a, b, c, d)
    for p in it:
        curr = (curr[1], curr[0] + curr[1] * p, curr[3], curr[2] + curr[3] * p)
        #print("input", p, curr)
        lst, curr = outputLoop(curr)
        for num in lst: yield num
        if not curr[2] and not curr[3]:
            break
    else:
        curr = (curr[1], curr[1], curr[3], curr[3])
        lst, curr = outputLoop(curr)
        for num in lst: yield num
    return

def continuedFractionAlphaTermsGenerator() -> Generator[int, None, None]:
    ps = SimplePrimeSieve()
    for p in ps.endlessPrimeGenerator():
        yield 2
        for _ in range(p):
            yield 1
    return

def continuedFractionAlphaRationalExpressionInitalTermsSum(n_init_terms: int=10 ** 8, a: int=3, b: int=2, c: int=2, d: int=3) -> int:
    cf = iter(continuedFractionAlphaTermsGenerator())
    res = sum(num for _, num in zip(range(n_init_terms), continuedFractionRationalExpression(cf, a, b, c, d)))
    return res

# Problem 948
memo_lft_glob = {}
memo_rgt_glob = {}
def leftVsRightOptimalPlayIsWinnerPlayerOne(s_bm: int, s_len: int, player_one_left: bool) -> bool:
    def leftTurn(bm: int, length: int) -> bool:
        if length == 1:
            return bool(bm)
        elif bm & 1: return True
        args = (bm, length)
        if args in memo_lft_glob.keys():
            return memo_lft_glob[args]
        mask = 0
        res = False
        for length2 in range(1, length):
            mask = (mask << 1) | 1
            if not rightTurn(bm & mask, length2):
                res = True
                break
        memo_lft_glob[args] = res
        return res
    
    def rightTurn(bm: int, length: int) -> bool:
        if length == 1:
            return not bool(bm)
        elif bm < (1 << (length - 1)): return True
        args = (bm, length)
        if args in memo_rgt_glob.keys():
            return memo_rgt_glob[args]
        res = False
        bm2 = bm
        for length2 in reversed(range(1, length)):
            bm2 >>= 1
            if not leftTurn(bm2, length2):
                res = True
                break
        
        memo_rgt_glob[args] = res
        return res
    
    return leftTurn(s_bm, s_len) if player_one_left else rightTurn(s_bm, s_len)


def leftVsRightPlayerOneWinsCountBruteForce(n: int=60) -> int:
    """
    memo_lft = {}
    def leftTurn(bm: int, length: int) -> bool:
        if length == 1:
            return bool(bm)
        elif bm & 1: return True
        args = (bm, length)
        if args in memo_lft.keys():
            return memo_lft[args]
        mask = 0
        res = False
        for length2 in range(1, length):
            mask = (mask << 1) | 1
            if not rightTurn(bm & mask, length2):
                res = True
                break
        memo_lft[args] = res
        return res

    memo_rgt = {}
    def rightTurn(bm: int, length: int) -> bool:
        if length == 1:
            return not bool(bm)
        elif bm < (1 << (length - 1)): return True
        args = (bm, length)
        if args in memo_rgt.keys():
            return memo_rgt[args]
        res = False
        bm2 = bm
        for length2 in reversed(range(1, length)):
            bm2 >>= 1
            if not leftTurn(bm2, length2):
                res = True
                break
        
        memo_rgt[args] = res
        return res
    """
    res = 0
    for bm in range(1 << n):
        #print(bm)
        res += leftVsRightOptimalPlayIsWinnerPlayerOne(bm, n, player_one_left=True) and\
                leftVsRightOptimalPlayIsWinnerPlayerOne(bm, n, player_one_left=False)#(leftTurn(bm, n) and rightTurn(bm, n))
        #print(res)
    return res

def leftVsRightPlayerOneWinsCount(n: int=60) -> int:
    """
    Solution to Project Euler #948
    """
    if n <= 0: return 0
    res = (1 << n) - 2 * math.comb(n - 1, (n - 1) >> 1)
    if n & 1:
        return res
    return res - math.comb(n - 2, (n >> 1) - 1) + (0 if n < 4 else math.comb(n - 2, (n >> 1) - 2))

# Problem 949
def leftVsRightMultipleWordsPlayerTwoWinsBruteForce(s_lens: int=20, n_words: int=7) -> int:
    
    n_s = 1 << s_lens
    n_s_winner_same_player = math.comb(s_lens - 1, (s_lens - 1) >> 1)
    n_s_winner_second = 0 if s_lens & 1 else (math.comb(s_lens - 2, (s_lens >> 1) - 1) - (0 if s_lens < 4 else math.comb(s_lens - 2, (s_lens >> 1) - 2)))
    n_s_winner_first = n_s - 2 * n_s_winner_same_player - n_s_winner_second
    print(n_s, n_s_winner_same_player, n_s_winner_second, n_s_winner_first)

    memo_lft = {}
    def leftTurn(bm_lst: List[Tuple[int, int]], bal: int) -> bool:
        bm_lst2 = []
        for bm, l in bm_lst:
            if l == 1:
                bal += 2 * bool(bm) - 1
            else: bm_lst2.append((bm, l))
        if abs(bal) > len(bm_lst2):
            return bal > 0
        bm_lst2 = tuple(sorted(bm_lst2))
        args = (bm_lst2, bal)
        if args in memo_lft.keys():
            return memo_lft[args]

        n_bm = len(bm_lst2)
        bm_lst3 = list(bm_lst2)
        def recur(idx: int, nonzero_seen: bool=False) -> bool:
            bm_lst3[idx] = bm_lst2[idx]
            if idx == n_bm - 1:
                if nonzero_seen:
                    if not rightTurn(bm_lst3, -bal): return True
                mask = 0
                for l2 in range(1, bm_lst2[idx][1]):
                    mask = (mask << 1) | 1
                    bm2 = bm_lst2[idx][0] & mask
                    bm_lst3[idx] = (bm2, l2)
                    if not rightTurn(bm_lst3, -bal):
                        return True
                return False
            if recur(idx + 1, nonzero_seen=nonzero_seen):
                return True
            mask = 0
            for l2 in range(1, bm_lst2[idx][1]):
                mask = (mask << 1) | 1
                bm2 = bm_lst2[idx][0] & mask
                bm_lst3[idx] = (bm2, l2)
                if recur(idx + 1, nonzero_seen=True):
                    return True
            return False
        res = recur(0, nonzero_seen=False)
        memo_lft[args] = res
        return res

    memo_rgt = {}
    def rightTurn(bm_lst: List[Tuple[int, int]], bal: int) -> bool:
        bm_lst2 = []
        for bm, l in bm_lst:
            if l == 1:
                bal += 1 - 2 * bool(bm)
            else: bm_lst2.append((bm, l))
        if abs(bal) > len(bm_lst2):
            return bal > 0
        bm_lst2 = tuple(sorted(bm_lst2))
        args = (bm_lst2, bal)
        if args in memo_rgt.keys():
            return memo_rgt[args]

        n_bm = len(bm_lst2)
        bm_lst3 = list(bm_lst2)
        def recur(idx: int, nonzero_seen: bool=False) -> bool:
            bm_lst3[idx] = bm_lst2[idx]
            if idx == n_bm - 1:
                if nonzero_seen:
                    if not leftTurn(bm_lst3, -bal): return True
                bm2 = bm_lst2[idx][0]
                for l2 in reversed(range(1, bm_lst2[idx][1])):
                    bm2 >>= 1
                    bm_lst3[idx] = (bm2, l2)
                    if not leftTurn(bm_lst3, -bal):
                        return True
                return False
            if recur(idx + 1, nonzero_seen=nonzero_seen):
                return True
            bm2 = bm_lst2[idx][0]
            for l2 in reversed(range(1, bm_lst2[idx][1])):
                bm2 >>= 1
                bm_lst3[idx] = (bm2, l2)
                if recur(idx + 1, nonzero_seen=True):
                    return True
            return False
        res = recur(0, nonzero_seen=False)
        memo_rgt[args] = res
        return res
    
    bm_lst = [(0, s_lens) for _ in range(n_words)]
    seen_category_cnts = {}
    def recur(idx: int) -> int:
        if idx == n_words:
            res = 1 - leftTurn(bm_lst, 0)
            if res:
                cat_lst = []
                for bm, l in bm_lst:
                    cat_lst.append((leftVsRightOptimalPlayIsWinnerPlayerOne(bm, l, True), not leftVsRightOptimalPlayIsWinnerPlayerOne(bm, l, False)))
                cat_lst = tuple(sorted(cat_lst))
                seen_category_cnts[cat_lst] = seen_category_cnts.get(cat_lst, 0) + res
            return res
        res = 0
        for bm in range(1 << s_lens):
            if not idx:
                print(f"first bm = {bm} of {(1 << s_lens) - 1}")
            bm_lst[idx] = (bm, s_lens)
            res += recur(idx + 1)
        return res

    res = recur(0)
    #print(memo_lft)
    #print(memo_rgt)
    #print(seen_category_cnts)
    full_cnts = {}
    for lst, f in seen_category_cnts.items():
        cnt = math.factorial(n_words)
        f_dict = {}
        for l1, l2 in lst:
            f_dict[(l1, l2)] = f_dict.get((l1, l2), 0) + 1
            cnt //= f_dict[(l1, l2)]
            if l1 == l2: cnt *= n_s_winner_same_player
            elif l1: cnt *= n_s_winner_first
            else: cnt *= n_s_winner_second
        full_cnts[lst] = (f, cnt)
    print(full_cnts)
    return res

def leftVsRightMultipleWordsPlayerTwoWins(s_lens: int=20, n_words: int=7) -> int:

    n_s = 1 << s_lens
    n_s_winner_same_player = math.comb(s_lens - 1, (s_lens - 1) >> 1)
    n_s_winner_second = 0 if s_lens & 1 else (math.comb(s_lens - 2, (s_lens >> 1) - 1) - (0 if s_lens < 4 else math.comb(s_lens - 2, (s_lens >> 1) - 2)))
    n_s_winner_first = n_s - 2 * n_s_winner_same_player - n_s_winner_second
    print(n_s, n_s_winner_same_player, n_s_winner_second, n_s_winner_first)

    def player2Winner(n_player1_wins: int, n_player2_wins: int, n_first_wins: int, n_second_wins: int) -> bool:
        #return True
        if not n_player1_wins and not n_first_wins and not n_player2_wins:
            return True
        bal0 = n_player1_wins + n_first_wins - n_player2_wins 
        bal = bal0 + ((n_second_wins) & 1)
        return bal < 0
        #if n_player2_wins >= n_player1_wins + (n_second_wins & 1):
        #    return True
    res = 0
    for n_p1 in range(n_words + 1):
        n_comb1 = math.comb(n_words, n_p1) * n_s_winner_same_player ** n_p1
        for n_p2 in range(n_words - n_p1 + 1):
            n_comb2 = n_comb1 * math.comb(n_words - n_p1, n_p2) * n_s_winner_same_player ** n_p2
            for n_w2 in range(n_words - n_p1 - n_p2 + 1):
                n_w1 = n_words - n_p1 - n_p2 - n_w2
                if not player2Winner(n_p1, n_p2, n_w1, n_w2): continue
                #player2_winner = not player2Winner(n_p2, n_p1, n_w2 + n_w1) if n_w1 & 1 else player2Winner(n_p1, n_p2, n_w2 + n_w1)
                #if not player2_winner: continue
                cnt = n_comb2 * math.comb(n_words - n_p1 - n_p2, n_w2) * n_s_winner_second ** n_w2 * n_s_winner_first ** n_w1
                print(f"n_p1 = {n_p1}, n_p2 = {n_p2}, n_w2 = {n_w2}, n_w1 = {n_w1}, cnt = {cnt}")
                res += cnt
    return res

if __name__ == "__main__":
    to_evaluate = {937}
    since0 = time.time()

    if not to_evaluate or 932 in to_evaluate:
        since = time.time()
        res = splitSumSquareNumbersSum(n_dig_max=16, base=10)
        print(f"Solution to Project Euler #932 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 933 in to_evaluate:
        since = time.time()
        res = paperCuttingWinningMoveSum2(width_min=2, width_max=123, height_min=2, height_max=1234567)
        print(f"Solution to Project Euler #933 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 934 in to_evaluate:
        since = time.time()
        res = unluckyPrimeSum(n_max=10 ** 17, p_md=7, ps=None)
        print(f"Solution to Project Euler #934 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 935 in to_evaluate:
        since = time.time()
        res = rollingRegularPolygonReturnAfterAtMostNRollsCount(n_roll_max=10 ** 8, n_sides=4)
        print(f"Solution to Project Euler #935 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 936 in to_evaluate:
        since = time.time()
        res = peerlessTreesWithVertexCountInRange2(n_vertex_min=3, n_vertex_max=50)
        print(f"Solution to Project Euler #936 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 937 in to_evaluate:
        since = time.time()
        res = calculateFactorialsInEquiproductPartitionWithUnitSum(n_max=100, md=10 ** 9 + 7)
        print(f"Solution to Project Euler #937 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 938 in to_evaluate:
        since = time.time()
        res = redBlackCardGameLastCardBlackProbabilityFloat(n_red_init=24690, n_black_init=12345)
        print(f"Solution to Project Euler #938 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 939 in to_evaluate:
        since = time.time()
        res = partisanNimNumberOfWinningPositions(max_n_stones=5000, md=1234567891)
        print(f"Solution to Project Euler #939 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 940 in to_evaluate:
        since = time.time()
        res = twoDimensionalRecurrenceFibonacciSum(k_min=2, k_max=50, md=1123581313)
        print(f"Solution to Project Euler #940 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 941 in to_evaluate:
        since = time.time()
        res = calculateOrderSumOfLinearRecurrencePseudoRandomNumbersInLexicographicallySmallestDeBruijnSequence(
            n_nums=10 ** 7,
            w_len=12,
            a0=0,
            k=920461,
            m=800217387569,
            base=10,
            md=1234567891,
        )
        print(f"Solution to Project Euler #941 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 944 in to_evaluate:
        since = time.time()
        res = sumOfSubsetElevisors(n_max=10 ** 14, md=1234567891)
        print(f"Solution to Project Euler #944 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 945 in to_evaluate:
        since = time.time()
        res = xorEquationSolutionsCount(a_b_max=10 ** 3)
        print(f"Solution to Project Euler #945 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 946 in to_evaluate:
        since = time.time()
        res = continuedFractionAlphaRationalExpressionInitalTermsSum(n_init_terms=10 ** 8, a=3, b=2, c=2, d=3)
        print(f"Solution to Project Euler #946 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 948 in to_evaluate:
        since = time.time()
        res = leftVsRightPlayerOneWinsCount(n=60)
        print(f"Solution to Project Euler #948 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 949 in to_evaluate:
        since = time.time()
        res = leftVsRightMultipleWordsPlayerTwoWinsBruteForce(s_lens=4, n_words=5)
        print(f"Solution to Project Euler #949 = {res}, calculated in {time.time() - since:.4f} seconds")

    print(f"Total time taken = {time.time() - since0:.4f} seconds")

        

"""
for num in range(1, 17):
    ans1 = sumOfSubsetElevisorsBruteForce(num)
    ans2 = sumOfSubsetElevisorsBruteForce2(num)
    ans = sumOfSubsetElevisors(n_max=num, md=None)
    print(f"num = {num}, brute force 1 = {ans1}, brute force 2 = {ans2}, func = {ans}")
"""
#a, b, c = 1, 8, 13
#print(xorMultiply(a, a) ^ xorMultiply(2, xorMultiply(a, b)) ^ xorMultiply(b, b), xorMultiply(c, c))
#for num in range(21):
#    print(num, xorMultiply(num, num))

#for s in ("aaaa", "ababab", "aabaab", "abcd"):
#    print(s, calculateStringPrimitiveRoot(s))

"""
x1_inv = .5
x2_inv = 2 - math.sqrt(2)
x3_inv = 2 + math.sqrt(2) - math.sqrt(2 + 4 * math.sqrt(2))
x4_inv = 8 - 5 * math.sqrt(2) + 4 * math.sqrt(3) - 3 * math.sqrt(6)
print(x1_inv, x2_inv, x3_inv, x4_inv)
print(1 / x1_inv, 1 / x2_inv, 1 / x3_inv, 1 / x4_inv)
"""
"""
a = 1 / (8 - 5 * math.sqrt(2) + 4 * math.sqrt(3) - 3 * math.sqrt(6))#(2 + math.sqrt(2) + math.sqrt(2 + 4 * math.sqrt(2))) / 4
x = a - 1
for i in range(100):
    print(i, x)
    x = a - math.sqrt(1 - x ** 2)
    x -= math.floor(x)
    #if x > math.sqrt(2) / 2: break
"""
"""
n = 4
res = 0
for k in range(1, n):
    res += math.floor(math.sqrt(n ** 2 - (k + .5) ** 2) - .5)
res = res * 4 + 4 * n - 3
print(n, res)
"""
"""
def calculateEndState(n_roll: int, a: float, verbose: bool=False) -> tuple[int, float]:
    x = a - 1#math.sqrt(2) / 2
    if verbose: print(f"a = {a}, n_roll = {n_roll}, x0 = {x}")
    remain = n_roll
    side = 1
    while remain > 0:
        x_int = min(math.floor(x), remain)
        x -= x_int
        remain -= x_int
        if verbose: print(x, remain)
        if not remain:
            break
        x = a - math.sqrt(1 - x ** 2)
        side += 1
        remain -= 1
    #print(side, x)
    return (side, -x)

corner = 16
eps = 1e-12
a = math.sqrt(2)
target_func1 = lambda corner_, a_: (corner_, -math.sqrt(2) / 2)
target_func2 = lambda corner_, a_: (corner_, -a_ * .5)
target_func3 = lambda corner_, a_: (corner_, -(a_ + 1) * .5)

target_func = target_func3

for n_roll in range(corner, (corner << 1) - 1):
    print(f"corner = {corner}, n_roll = {n_roll}")
    lft, rgt = a, 2
    while (rgt - lft) > eps:
        mid = lft + (rgt - lft) * .5
        end = calculateEndState(n_roll - 1, mid, verbose=False)
        t = target_func(corner, mid)
        #print(f"mid = {mid}, end_state = {end}, target = {t}")
        if end > t:#(corner, -math.sqrt(2) / 2):
            lft = mid
        else: rgt = mid
    a = lft + (rgt - lft) * .5
    side, x = calculateEndState(n_roll - 1, a, verbose=True)
    #if side > corner:
    #    side, x = side - 1, x + a
    print(a, side, x, abs(x - target_func(corner, a)[1]))
"""
"""
row_max = 20
col_max = 30
res = [[False for _ in range(col_max + 1)] for _ in range(row_max + 1)]
res[0][0] = True
for i1 in range(1, row_max + 1):
    for i2 in range(1, col_max + 1):
        tot = 0
        for j1 in range((i1 + 1) >> 1):
            for j2 in range(i2 + 1):
                if res[j1][j2] == res[i1 - j1][i2 - j2]:
                    if (j1 == i1 - j1 and j2 == i2 - j2):
                        print(f"repeat 1: ({j1, j2})")
                    tot += 2 * res[j1][j2] - 1
        if not i1 & 1:
            j1 = i1 >> 1
            for j2 in range(((i2 + 1) >> 1)):
                if res[j1][j2] == res[i1 - j1][i2 - j2]:
                    if (j1 == i1 - j1 and j2 == i2 - j2):
                        print(f"repeat 2: ({j1, j2})")
                    tot += 2 * res[j1][j2] - 1
        #print(tot)
        res[i1][i2] = (tot == -1)
for i, row in enumerate(res):
    print(i, ["A" if b else "B" for b in row])
"""
"""
cnt = 0
for term in continuedFractionRationalExpression([1, 5, 2], 1, 2, 2, 0):
    print(term)
    if cnt > 10: break
    cnt += 1
"""
"""
a = 5
b = 9
print(xorMultiply(a ^ b, a ^ b))
print(xorMultiply(a, a) ^ xorMultiply(b, b))
"""