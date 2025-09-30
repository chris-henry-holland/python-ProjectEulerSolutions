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
import numpy as np
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

# Problem 251
def cardanoTripletGeneratorBySum(sum_max: Optional[int]=None) -> Generator[Tuple[int, Tuple[int, int, int]], None, None]:
    ps = PrimeSPFsieve()
    h = []
    for k in itertools.count(1):
        #rt1 = isqrt(8 * k - 3)
        #rt2 = 2 * isqrt(2 * k - 1) + 1
        
        a = 3 * k - 1
        b_lb = integerNthRoot(2 * k ** 2 * (8 * k - 3), 3)#k * (3 + min(rt1, rt2))
        sm_lb = a + b_lb + k ** 2 * (8 * k - 3) // b_lb ** 2
        if sum_max is not None and sm_lb > sum_max: break
        #print(a, lb)
        while h and h[0][0] <= sm_lb:
            yield heapq.heappop(h)
        #b_sq_c = 27 * k ** 2
        
        pf = ps.primeFactorisation(k)
        c0 = 1

        non_sq = (8 * k - 3)
        num = non_sq
        for p in ps.endlessPrimeGenerator():
            p_sq = p * p
            if p_sq > num: break
            num2, r = divmod(num, p_sq)
            while not r:
                num = num2
                pf[p] = pf.get(p, 0) + 1
                num2, r = divmod(num, p_sq)
            num2, r = divmod(num, p)
            if not r:
                num = num2
                c0 *= p
        c0 *= num
        p_lst = sorted(pf.keys())
        n_p = len(p_lst)
        f_lst = [pf[p] for p in p_lst]
        c_mx = sum_max - a - 1 if sum_max is not None else float("inf")
        #print(k, non_sq, pf, c0)

        def recur(idx: int, b_curr: int=1, c_curr: int=c0) -> Generator[Tuple[int, int], None, None]:
            if idx == n_p:
                if a + b_curr + c_curr > sum_max: return
                yield (b_curr, c_curr)
                return
            c = c_curr
            b = b_curr * p_lst[idx] ** f_lst[idx]
            for i in range(f_lst[idx] + 1):
                yield from recur(idx + 1, b_curr=b, c_curr=c)
                c *= p_lst[idx] ** 2
                if c > c_mx: break
                b //= p_lst[idx]
            return

        for b, c in recur(0, b_curr=1, c_curr=c0):
            sm = a + b + c
            heapq.heappush(h, (sm, (a, b, c)))


    while h: yield heapq.heappop(h)
    return


def cardanoTripletCount(sum_max: int=11 * 10 ** 7) -> int:
    """
    Solution to Project Euler #251
    """
    # Review- Try to make faster
    ps = PrimeSPFsieve()
    h = []
    res = 0
    for k in itertools.count(1):
        #rt1 = isqrt(8 * k - 3)
        #rt2 = 2 * isqrt(2 * k - 1) + 1
        a = 3 * k - 1
        b_lb = integerNthRoot(2 * k ** 2 * (8 * k - 3), 3)#k * (3 + min(rt1, rt2))
        sm_lb = a + b_lb + k ** 2 * (8 * k - 3) // b_lb ** 2
        if sum_max is not None and sm_lb > sum_max: break
        
        if not k % 10 ** 4: print(f"a = {a}")
        
        pf = ps.primeFactorisation(k)
        c0 = 1

        non_sq = (8 * k - 3)
        num = non_sq
        for p in ps.endlessPrimeGenerator():
            p_sq = p * p
            if p_sq > num: break
            num2, r = divmod(num, p_sq)
            while not r:
                num = num2
                pf[p] = pf.get(p, 0) + 1
                num2, r = divmod(num, p_sq)
            num2, r = divmod(num, p)
            if not r:
                num = num2
                c0 *= p
        c0 *= num
        p_lst = sorted(pf.keys())
        n_p = len(p_lst)
        f_lst = [pf[p] for p in p_lst]
        c_mx = sum_max - a - 1 if sum_max is not None else float("inf")
        if a + max(k * isqrt((8 * k - 3) // c0) + c0, 1 + (k ** 2 * (8 * k - 3))) <= sum_max:
            ans = 1
            for f in f_lst:
                ans *= (f + 1)
            res += ans
            continue
        #print(k, non_sq, pf, c0)

        def recur(idx: int, b_curr: int=1, c_curr: int=c0) -> int:
            if idx == n_p:
                return a + b_curr + c_curr <= sum_max
            c = c_curr
            b = b_curr * p_lst[idx] ** f_lst[idx]
            res = 0
            for i in range(f_lst[idx] + 1):
                res += recur(idx + 1, b_curr=b, c_curr=c)
                c *= p_lst[idx] ** 2
                if c > c_mx: break
                b //= p_lst[idx]
            return res
        res += recur(0, b_curr=1, c_curr=c0)

    return res

# Problem 252
def triangleDoubleArea(p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int]) -> int:
    res = abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
    #print(p1, p2, p3, res)
    return res

def calculateLargestEmptyConvexPolygonDoubleArea(points: List[Tuple[int, int]]) -> int:
    # Review- try to make faster
    n = len(points)
    points.sort()
    #print(points)

    def calculateFullSlope(idx1: int, idx2: int) -> Tuple[bool, CustomFraction]:
        pt1, pt2 = points[idx1], points[idx2]
        diff = tuple(x - y for x, y in zip(pt2, pt1))
        return (diff[0] < 0, CustomFraction(diff[1], diff[0]))

    def largestDoubleAreaWithMinimumPoint(min_pt_idx: int) -> int:
        
        seen_order = [min_pt_idx]
        area_stks = [[((False, CustomFraction(-1, 0)), 0)]]
        ref_pt = points[min_pt_idx]
        #print(min_pt_idx, ref_pt)
        sweep_pts = []
        for idx in range(min_pt_idx + 1, n):
            pt = points[idx]
            diff = tuple(x - y for x, y in zip(pt, ref_pt))
            slope = CustomFraction(diff[1], diff[0])
            sweep_pts.append((slope, diff, idx))
        sweep_pts.sort()
        res = 0
        for _, _, idx in sweep_pts:
            src_pts = []
            for i in reversed(range(len(seen_order))):
                idx0 = seen_order[i]
                slope = calculateFullSlope(idx0, idx)
                if src_pts and slope >= src_pts[-1][0]:
                    continue
                src_pts.append((slope, i))
            #src_pts = src_pts[::-1]
            #print(points[idx])
            #print([(points[seen_order[tup[1]]], tup[0]) for tup in reversed(src_pts)])
            #print(src_pts)
            #print(area_stks)
            stk = []
            #print(idx, points[idx], src_pts)
            for slope, i2 in reversed(src_pts):
                j = bisect.bisect_right(area_stks[i2], (slope, float("inf"))) - 1
                if j < 0: continue
                idx2 = seen_order[i2]
                #print(area_stks[j])
                area0 = area_stks[i2][j][1]
                
                area = area0 + triangleDoubleArea(ref_pt, points[idx], points[idx2])
                #print(i2, slope, area_stks[i2], j, area0, area, ref_pt, points[idx], points[idx2])
                #print(stk)
                if stk and area <= stk[-1][1]: continue
                res = max(res, area)
                stk.append((slope, area))
            seen_order.append(idx)
            area_stks.append(stk)
        #print(seen_order)
        #print([points[idx] for idx in seen_order])
        #print(area_stks)
        return res
    
    res = 0
    for idx in range(n):
        if not idx % 10:
            print(f"{idx} of {n} vertices processed")
        ans = largestDoubleAreaWithMinimumPoint(idx)
        if ans > res:
            #print(points[idx], ans)
            res = max(res, ans)
        #break
    return res
    """
    points = [tuple(x) for x in points]
    if len(points) < 3: return sorted(set(points))

    comp = (lambda x, y: x <= y) if include_border_points else (lambda x, y: x < y)
    
    ref_pt = min(points)
    sorted_pts = []
    min_x_pts = []
    for pos in points:
        if pos[0] == ref_pt[0]:
            min_x_pts.append(pos)
            continue
        diff = tuple(x - y for x, y in zip(pos, ref_pt))
        slope = diff[1] / diff[0]
        sorted_pts.append((slope, diff, pos))
    sorted_pts.sort()
    
    if len(min_x_pts) > 1:
        if include_border_points:
            tail = sorted(min_x_pts)
            pos = tail.pop()
            sorted_pts.append((None, tuple(x - y for x, y in zip(pos, ref_pt)), pos))
            tail = tail[::-1]
        else:
            pos = max(min_x_pts)
            sorted_pts.append((None, tuple(x - y for x, y in zip(pos, ref_pt)), pos))
            tail = [ref_pt]
    else:
        tail = []
        tup0 = sorted_pts.pop()
        diff0 = tup0[1]
        while sorted_pts and diff0[0] * sorted_pts[-1][1][1] == diff0[1] * sorted_pts[-1][1][0]:
            tail.append(sorted_pts.pop()[2])
        tail = tail[::-1] if include_border_points else []
        tail.append(ref_pt)
        sorted_pts.append(tup0)
    stk = [(sorted_pts[0][2], tuple(x - y for x, y in zip(sorted_pts[0][2], ref_pt)))]
    order = [x[0] for x in stk]
    for i in range(1, len(sorted_pts)):
        pos = sorted_pts[i][2]
        while stk:
            diff = tuple(x - y for x, y in zip(pos, stk[-1][0]))
            cross_prod = stk[-1][1][0] * diff[1] -\
                    stk[-1][1][1] * diff[0]
            if comp(0, cross_prod): break
            stk.pop()
        
        stk.append((pos, tuple(x - y for x, y in zip(pos, (stk[-1][0] if stk else ref_pt)))))
    res = [x[0] for x in stk] + tail

    return [res[-1], *res[:-1]]
    """

def calculateLargestEmptyConvexPolygonAreaFloat(points: List[Tuple[int, int]]) -> float:
    return 0.5 * calculateLargestEmptyConvexPolygonDoubleArea(points)

def blumBlueShubPseudoRandomPointGenerator(
        n_dim: int=2,
        s_0: int=290797,
        s_mod: int=50515093,
        t_min: int=-1000,
        t_max: int=999,
) -> Generator[Tuple[Tuple[int, int], Tuple[int, int]], None, None]:
    """
    Generator yielding pseudo-random pints in n_dim-dimensional
    space whose end points have integer Cartesian coordinates based
    on the terms in a Blum Blum Shub sequence for a given seed value,
    modulus and value range.

    Each group of consecutive (n_dim) values in the specified Blum Blum
    Shub sequence produces the (in order) the Cartesian coordinates of
    a point.

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
                points are to be generated.
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
            Default: -1000
        t_max (int): Integer giving the smallest value possible
                for terms in the Blum Blum Shub sequence, and so the
                largest value possible for any coordinate for the
                end points of the line segments. Must be no smaller
                than t_min.
            Default: 999
    
    Yields:
    n_dim-tuple of ints, with each integer value being between min_value
    and max_value inclusive, representing the Cartesian coordinates of
    a point based on the given Blum Blum Shub sequence as specified above.
    """
    it = iter(blumBlumShubPseudoRandomGenerator(s_0=s_0, s_mod=s_mod, t_min=t_min, t_max=t_max))
    while True:
        yield tuple(next(it) for _ in range(n_dim))
    return

def blumBlumShubPseudoRandomTwoDimensionalPointsLargestEmptyConvexHoleArea(
        n_points: int=500,
        blumblumshub_s_0: int=290797,
        blumblumshub_s_mod: int=50515093,
        coord_min: int=-1000,
        coord_max: int=999,
) -> int:
    """
    Solution to Project Euler #252

    TODO


    For details regarding the point generation, see the documentation
    of blumBlueShubPseudoRandomPointGenerator(), from which (for the
    given Blum Blum Shub parameters) the first n_points values are
    taken as the points to be analysed.
    
    Args:
        Optional named:
        n_points (int): The number of points to be generated for which the
                largest area of any convex hole is to be calculated.
            Default=500
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
            Default=-1000
        coord_max (int): Integer giving the smalles value possible for either
                x or y coordinate for the endpoints of the line segments.
                Note that changes to this value will in general completely
                change the line segments generated. Must be no smaller than
                coord_min.
            Default=999
    
    Returns:
    Float (float) giving the largest area of any convex hole formed by the
    n_points points generated by blumBlueShubPseudoRandomPointGenerator().
    """
    it = iter(blumBlueShubPseudoRandomPointGenerator(
        n_dim=2,
        s_0=blumblumshub_s_0,
        s_mod=blumblumshub_s_mod,
        t_min=coord_min,
        t_max=coord_max,
    ))
    points = [next(it) for _ in range(n_points)]
    #print(points)
    res = calculateLargestEmptyConvexPolygonAreaFloat(points)
    return res

# Problem 253
def constructingLinearPuzzleMaxSegmentCountDistribution0(n_pieces: int) -> List[int]:
    
    
    if n_pieces == 1: return [0, 1]

    def addDistribution(distr: List[int], distr_add: List[int], mn_val: int=0) -> None:
        for _ in range(max(mn_val + 1, len(distr_add)) - len(distr)):
            distr.append(distr[-1])
        for j in range(mn_val, len(distr)):
            #print(j)
            distr[j] += distr_add[min(j, len(distr_add) - 1)]
        while len(distr) > 1 and distr[-1] == distr[-2]:
            distr.pop()
        return

    def distributionProduct(distr1: List[int], distr2: List[int], mn_val: int=0) -> None:
        if mn_val >= len(distr1) + len(distr2) - 2:
            res = [0] * (mn_val + 1)
            res[-1] = distr1[-1] * distr2[-1]
            return res
        res = [0] * (len(distr1) + len(distr2) - 1)
        for i1 in range(len(distr1)):
            if not distr1[i1]: continue
            for i2 in range(max(0, mn_val - i1), len(distr2)):
                res[i1 + i2] += distr1[i1] * distr2[i2]
        return res
    """
    memo = {}
    def recur(size: int, first: bool, last: bool) -> List[int]:
        if size == 1:
            return [0, 1] if not first and not last else [1]
        if first < last: first, last = last, first
        args = (size, first, last)
        if args in memo.keys(): return memo[args]
        res = recur(size - 1, True, last) if first else [0, *recur(size - 1, True, last)]
        addDistribution(res, recur(size - 1, first, True) if last else [0, *recur(size - 1, first, True)])
        mn_val = first + last + 1
        for i in range(1, size - 1):
            distr1 = recur(i, first, True)
            distr2 = recur(size - i - 1, True, last)
            mult = math.comb(size - 1, i)
            print(distr1, distr2)
            print(f"mult = {mult}")
            prod = [x * mult for x in distributionProduct(distr1, distr2, mn_val=mn_val)]
            print(f"prod = {prod}")
            addDistribution(res, prod)
        memo[args] = res
        return res


    """
    memo = {}
    def recur(gaps: Dict[int, int], ends: List[int]) -> List[int]:
        #print(gaps, ends)
        if gaps and min(gaps.keys()) < 0: return []
        if not gaps and not max(ends):
            return [0, 1]
        #elif max(gaps) < 3 and (first or gaps[0] < 2) and (last or gaps[-1] < 2):
        #    m = len(gaps) + first + last - 1
        #    tot = sum(gaps)
        #    res = [0] * (m + 1)
        #    res[m] = math.factorial(tot)
        #    return res
        #gaps, first, last = min((gaps, first, last), (gaps[::-1], last, first))
        gaps2 = tuple(sorted((k, v) for k, v in gaps.items()))
        args = (gaps2, tuple(sorted(ends)))
        if args in memo.keys(): return memo[args]
        n_pieces = (sum(gaps.values()) if gaps else 0) + 1
        res = [0] * (n_pieces + 1)
        
        for j in range(2):
            end0 = ends[j]
            if not end0: continue
            elif end0 == 1:
                ends[j] = 0
                distr = recur(gaps, ends)
                addDistribution(res, distr, mn_val=n_pieces)
                ends[j] = end0
                continue
            for i in range(end0 - 1):
                i2 = end0 - i - 1
                ends[j] = i
                gaps[i2] = gaps.get(i2, 0) + 1
                distr = recur(gaps, ends)
                addDistribution(res, distr, mn_val=n_pieces)
                gaps[i2] -= 1
                if not gaps[i2]: gaps.pop(i2)
            ends[j] = end0 - 1
            distr = recur(gaps, ends)
            addDistribution(res, distr, mn_val=n_pieces)
            ends[j] = end0
        
        for gap in list(gaps.keys()):
            f = gaps[gap]
            gaps[gap] -= 1
            if not gaps[gap]: gaps.pop(gap)
            if gap == 1:
                distr = [f * x for x in recur(gaps, ends)]
                addDistribution(res, distr, mn_val=n_pieces)
                gaps[gap] = gaps.get(gap, 0) + 1
                continue
            gaps[gap - 1] = gaps.get(gap - 1, 0) + 1
            distr = [2 * f * x for x in recur(gaps, ends)]
            addDistribution(res, distr, mn_val=n_pieces)
            gaps[gap - 1] -= 1
            if not gaps[gap - 1]: gaps.pop(gap - 1)
            for i in range(1, (gap >> 1)):
                i2 = gap - i - 1
                gaps[i] = gaps.get(i, 0) + 1
                gaps[i2] = gaps.get(i2, 0) + 1
                distr = [2 * f * x for x in recur(gaps, ends)]
                addDistribution(res, distr, mn_val=n_pieces)
                gaps[i] -= 1
                if not gaps[i]: gaps.pop(i)
                gaps[i2] -= 1
                if not gaps[i2]: gaps.pop(i2)
            if gap & 1:
                i = gap >> 1
                gaps[i] = gaps.get(i, 0) + 2
                distr = [f * x for x in recur(gaps, ends)]
                addDistribution(res, distr, mn_val=n_pieces)
                gaps[i] -= 2
                if not gaps[i]: gaps.pop(i)
            
            gaps[gap] = gaps.get(gap, 0) + 1
        memo[args] = res
        return res
        """
        for i in range(len(gaps)):
            num = gaps[i]
            if num == 1:
                addDistribution(res, recur([*gaps[:i], *gaps[i + 1:]], first or not i, last or (i == len(gaps) - 1)), mn_val=n_pieces)
                continue
            gaps[i] -= 1
            addDistribution(res, recur(gaps, first or not i, last), mn_val=n_pieces)
            addDistribution(res, recur(gaps, first, last or (i == len(gaps) - 1)), mn_val=n_pieces)
            gaps[i] += 1
            if num <= 2: continue
            gaps2 = [*gaps[:i], 1, num - 2, *gaps[i + 1:]]
            for j in range(1, num - 2):
                addDistribution(res, recur(gaps2, first, last), mn_val=n_pieces)
                gaps2[i] += 1
                gaps2[i + 1] -= 1
            addDistribution(res, recur(gaps2, first, last), mn_val=n_pieces)
        
        memo[args] = res
        return res
        """
        """
        for i in range(n_pieces):
            #print(f"i = {i}")
            bm2 = 1 << i
            if bm2 & bm: continue
            n_segs2 = n_segs + 1
            if i > 0 and bm & (1 << (i - 1)):
                n_segs2 -= 1
            if i < n_pieces - 1 and bm & (1 << (i + 1)):
                n_segs2 -= 1
            ans = recur(bm | bm2, n_segs2)
            for _ in range(len(ans) - len(res)):
                res.append(res[-1])
            for j in range(n_segs, len(res)):
                #print(j)
                res[j] += ans[min(j, len(ans) - 1)]
        while len(res) > 1 and res[-1] == res[-2]:
            res.pop()
        memo[args] = res
        return res
        """

    res_cumu = recur({}, [0, n_pieces - 1])
    for i in range(1, n_pieces >> 1):
        distr = recur({}, [i, n_pieces - i - 1])
        addDistribution(res_cumu, distr)
    res_cumu = [x << 1 for x in res_cumu]
    if n_pieces & 1:
        x = n_pieces >> 1
        addDistribution(res_cumu, recur({}, [x, x]))
    #res_cumu = recur([n_pieces], False, False)
    print(f"len(memo) = {len(memo)}")
    #print(f"res_cumu = {res_cumu}")
    #print(memo)
    res = [res_cumu[0]]
    for i in range(1, len(res_cumu)):
        res.append(res_cumu[i] - res_cumu[i - 1])
    return res

def constructingLinearPuzzleMaxSegmentCountDistribution(n_pieces: int) -> List[int]:
    
    # Review- Look into the binary tree solution of Lucy_Hedgehog
    if n_pieces == 1: return [0, 1]

    def addDistribution(distr: List[int], distr_add: List[int], mn_val: int=0) -> None:
        for _ in range(max(mn_val + 1, len(distr_add)) - len(distr)):
            distr.append(distr[-1])
        for j in range(mn_val, len(distr)):
            #print(j)
            distr[j] += distr_add[min(j, len(distr_add) - 1)]
        while len(distr) > 1 and distr[-1] == distr[-2]:
            distr.pop()
        return

    def distributionProduct(distr1: List[int], distr2: List[int], mn_val: int=0) -> None:
        if mn_val >= len(distr1) + len(distr2) - 2:
            res = [0] * (mn_val + 1)
            res[-1] = distr1[-1] * distr2[-1]
            return res
        res = [0] * (len(distr1) + len(distr2) - 1)
        for i1 in range(len(distr1)):
            if not distr1[i1]: continue
            for i2 in range(max(0, mn_val - i1), len(distr2)):
                res[i1 + i2] += distr1[i1] * distr2[i2]
        return res
    
    memo = {}
    def recur(seg_lens: Dict[int, int]) -> List[int]:
        if not seg_lens:
            return [1]
        seg_lens2 = tuple(sorted((k, v) for k, v in seg_lens.items()))
        args = seg_lens2
        if args in memo.keys(): return memo[args]
        n_pieces = (sum(seg_lens.values()) if seg_lens else 0)
        res = [0] * (n_pieces + 1)
        
        for seg_len in list(seg_lens.keys()):
            f = seg_lens[seg_len]
            seg_lens[seg_len] -= 1
            if not seg_lens[seg_len]: seg_lens.pop(seg_len)
            if seg_len == 1:
                distr = [f * x for x in recur(seg_lens)]
                addDistribution(res, distr, mn_val=n_pieces)
                seg_lens[seg_len] = seg_lens.get(seg_len, 0) + 1
                continue
            seg_lens[seg_len - 1] = seg_lens.get(seg_len - 1, 0) + 1
            distr = [2 * f * x for x in recur(seg_lens)]
            addDistribution(res, distr, mn_val=n_pieces)
            seg_lens[seg_len - 1] -= 1
            if not seg_lens[seg_len - 1]: seg_lens.pop(seg_len - 1)
            for i in range(1, (seg_len >> 1)):
                i2 = seg_len - i - 1
                seg_lens[i] = seg_lens.get(i, 0) + 1
                seg_lens[i2] = seg_lens.get(i2, 0) + 1
                distr = [2 * f * x for x in recur(seg_lens)]
                addDistribution(res, distr, mn_val=n_pieces)
                seg_lens[i] -= 1
                if not seg_lens[i]: seg_lens.pop(i)
                seg_lens[i2] -= 1
                if not seg_lens[i2]: seg_lens.pop(i2)
            if seg_len & 1:
                i = seg_len >> 1
                seg_lens[i] = seg_lens.get(i, 0) + 2
                distr = [f * x for x in recur(seg_lens)]
                addDistribution(res, distr, mn_val=n_pieces)
                seg_lens[i] -= 2
                if not seg_lens[i]: seg_lens.pop(i)
            
            seg_lens[seg_len] = seg_lens.get(seg_len, 0) + 1
        memo[args] = res
        return res

    res_cumu = recur({n_pieces: 1})
    #res_cumu = recur([n_pieces], False, False)
    print(f"len(memo) = {len(memo)}")
    #print(f"res_cumu = {res_cumu}")
    #print(memo)
    res = [res_cumu[0]]
    for i in range(1, len(res_cumu)):
        res.append(res_cumu[i] - res_cumu[i - 1])
    return res

def constructingLinearPuzzleMaxSegmentCountMeanFraction(n_pieces: int) -> CustomFraction:
    denom = 0
    numer = 0
    distr = constructingLinearPuzzleMaxSegmentCountDistribution(n_pieces)
    print(distr)
    for i, num in enumerate(distr):
        numer += i * num
        denom += num
    return CustomFraction(numer, denom)

def constructingLinearPuzzleMaxSegmentCountMeanFloat(n_pieces: int=40) -> float:
    """
    Solution to Project Euler #253
    """
    frac = constructingLinearPuzzleMaxSegmentCountMeanFraction(n_pieces)
    print(frac)
    return frac.numerator / frac.denominator

# Problem 254
def calculateSmallestNumberDigitFrequenciesWithSumOfFactorialSumDigitsEqualToN(n: int, base: int=10) -> List[int]:
    
    mx_non_max_dig_tot = math.factorial(base - 1) - 1
    mx_non_max_dig_n_dig = 0
    num = mx_non_max_dig_tot
    while num:
        num //= base
        mx_non_max_dig_n_dig += 1
    mask = ~(1 << mx_non_max_dig_n_dig)
    #tail = base ** mx_non_max_dig_n_dig
    
    factorials = [math.factorial(i) for i in range(base)]
    
    def upperBoundDigitSum(max_dig_count: int) -> int:
        #num_max = factorials[-1] * (max_dig_count + 1) - 1
        #n_dig = 0
        #num2 = num_max
        #while num2:
        #    num2 //= base
        #    n_dig += 1
        #n_dig = max(n_dig, mx_non_max_dig_n_dig)
        #return n_dig * (base - 1)
        num = factorials[-1] * max_dig_count
        num //= base ** mx_non_max_dig_n_dig
        res = (base - 1) * mx_non_max_dig_n_dig
        #print(res, num)
        while num:
            num, d = divmod(num, base)
            res += d
        return res
    
    # Review- why does this give such a close lower bound on the
    # number of max digits required for a given factorial digit
    # sum? (for values above 60 it is exact)
    def upperBoundDigitSumCoarse(max_dig_count: int) -> int:
        num_max = factorials[-1] * (max_dig_count + 1) - 1
        n_dig = 0
        num2 = num_max
        while num2 >= base:
            num2 //= base
            n_dig += 1
        #n_dig = max(n_dig, mx_non_max_dig_n_dig)
        return n_dig * (base - 1) + num2
        
    
    #for num in range(100):
    #    print(num, upperBoundDigitSum(num))

    # Review- Justify why this is > and not >= n
    
    if upperBoundDigitSumCoarse(0) > n:
        lb = 0
    elif upperBoundDigitSumCoarse(1) > n:
        lb = 1
    else:
        lb = 1
        while True:
            lb2 = lb << 1
            if upperBoundDigitSumCoarse(lb2) > n:
                break
            lb = lb2
        lft, rgt = lb, lb2
        while lft < rgt:
            mid = lft + ((rgt - lft) >> 1)
            if upperBoundDigitSumCoarse(mid) > n:
                rgt = mid
            else: lft = mid + 1
        lb = lft
    
    curr = [0] * (base - 1)
    def recur(dig_remain: int, target_dig_sum: int, dig_fact_sum: int, dig: int) -> None:
        #print(f"Using recur() with dig_remain = {dig_remain}, target_dig_sum = {target_dig_sum}, dig_fact_sum = {dig_fact_sum}, dig = {dig}")
        if not dig_remain:
            g = 0
            #print(num, dig_fact_sum)
            while dig_fact_sum:
                dig_fact_sum, r = divmod(dig_fact_sum, base)
                g += r
                if g > target_dig_sum: break
            else:
                return g == target_dig_sum
            return False
        elif dig >= base - 1: return
        
        min_n_add = max(0, dig_remain - (((base - 1) * (base - 2) - dig * (dig - 1)) >> 1))
        max_n_add = min(dig_remain, dig) #if dig < base - 1 else dig_remain
        #print(min_n_add, max_n_add)
        #for _ in range(max_n_add):
        #    num2 = num2 * base + dig
        dig_fact_sum2 = dig_fact_sum + max_n_add * factorials[dig]
        dig_remain2 = dig_remain - max_n_add
        curr[dig] = max_n_add#max_n_add
        for _ in reversed(range(min_n_add, max_n_add + 1)):
            if recur(dig_remain2, target_dig_sum=target_dig_sum, dig_fact_sum=dig_fact_sum2, dig=dig + 1):
                return True
            dig_remain2 += 1
            curr[dig] -= 1
            dig_fact_sum2 -= factorials[dig]
        curr[dig] = 0
        return False

    print(f"max digit count coarse lower bound = {lb}")
    #print(upperBoundDigitSumCoarse(lb))
    res = (-float("inf"), [])
    for n_max_dig in itertools.count(lb):
        
        #print(f"n_max_dig = {n_max_dig}")
        if n_max_dig >= -res[0]: break
        if upperBoundDigitSum(n_max_dig) < n:
            #print(f"skipping n_max_dig = {n_max_dig}")
            continue
        max_dig_val = factorials[-1] * n_max_dig
        head = max_dig_val // (base ** (mx_non_max_dig_n_dig))
        #nonmax_dig_contrib_mx_n_dig = 
        tail_len = mx_non_max_dig_n_dig + 1
        while head % base == base - 1:
            head //= base
            tail_len += 1
        head //= base
        target_dig_sum = n
        head2 = head
        while head2:
            head2, d = divmod(head2, base)
            target_dig_sum -= d
        if target_dig_sum < 0: break
        tail_init = max_dig_val % (base ** tail_len)
        for n_nonmax_dig in range(min(((base - 2) * (base - 1)) >> 1, -res[0] - n_max_dig) + 1):
            #print(f"n_max_dig = {n_max_dig}, n_nonmax_dig = {n_nonmax_dig}, target_dig_sum = {target_dig_sum}, tail_init = {tail_init}")
            curr = [0] * (base - 1)
            b = recur(n_nonmax_dig, target_dig_sum, dig_fact_sum=tail_init, dig=1)
            if not b: continue
            
            ans = curr + ([n_max_dig])
            print(f"possible solution: {ans}")
            tot_n_dig = sum(ans)
            res = max(res, (-tot_n_dig, ans))
            break
    return res[1]

def calculateSmallestNumberDigitFrequenciesWithTheFirstNSumOfDigitFactorials(n_max: int, base: int=10) -> List[int]:

    res = [[]]
    for num in range(1, n_max + 1):
        dig_freqs = calculateSmallestNumberDigitFrequenciesWithSumOfFactorialSumDigitsEqualToN(num, base=base)
        print(num, dig_freqs)
        res.append(dig_freqs)
    return res

def calculateSmallestNumberWithTheFirstNSumOfDigitFactorials0(n_max: int, base: int=10) -> List[int]:

    factorials = [math.factorial(i) for i in range(base)]

    res = [-1] * (n_max + 1)
    n_seen = [0]
    memo = set()
    def recur(dig_remain: int, num: int=0, dig_fact_sum: int=0, dig: int=2) -> None:
        if not dig_remain:
            g = 0
            #print(num, dig_fact_sum)
            while dig_fact_sum:
                dig_fact_sum, r = divmod(dig_fact_sum, base)
                g += r
                if g > n_max: break
            else:
                if res[g] >= 0: return
                res[g] = num
                n_seen[0] += 1
                print(f"{g}: {num}, {dig_fact_sum}")
            return
        args = (dig_remain, dig_fact_sum, dig)
        if args in memo: return
        if dig == base - 1:
            for _ in range(dig_remain):
                num = (num + 1) * base - 1
            return recur(0, num=num, dig_fact_sum=(dig_fact_sum + factorials[-1] * dig_remain), dig=base)
        num2 = num
        
        max_n_add = min(dig_remain, dig) #if dig < base - 1 else dig_remain
        for _ in range(max_n_add):
            num2 = num2 * base + dig
        dig_fact_sum2 = dig_fact_sum + max_n_add * factorials[dig]
        dig_remain2 = dig_remain - max_n_add
        for _ in reversed(range(max_n_add + 1)):
            recur(dig_remain2, num=num2, dig_fact_sum=dig_fact_sum2, dig=dig + 1)
            if n_seen[0] == n_max: break
            dig_remain2 += 1
            num2 //= base
            dig_fact_sum2 -= factorials[dig]
        memo.add(args)
        return

    for n_dig in itertools.count(1):
        since = time.time()
        recur(n_dig - 1, num=1, dig_fact_sum=1, dig=2)
        recur(n_dig, num=0, dig_fact_sum=0, dig=2)
        print(f"n_dig = {n_dig}, n_seen = {n_seen[0]} of {n_max}")
        print(f"iteration time = {time.time() - since} seconds")
        #print(res)
        if n_seen[0] == n_max: break
        #if n_dig > 10: break
        
    """
    for n_dig in itertools.count(1):
        
        head = base ** (n_dig - 1)
        dig_fact_sum = factorials[1] + factorials[0] * (n_dig - 1)
        dig_remain = 0
        for head_len in reversed(range(1, n_dig + 1)):
            recur(n_dig - head_len, head, dig_fact_sum=dig_fact_sum, dig=2)
            dig_remain += 1
            head //= base
            dig_fact_sum -= factorials[0]
        recur(n_dig, num=0, dig_fact_sum=0, dig=2)
        if n_seen[0] == n_max: break
        print(f"n_dig = {n_dig}, n_seen = {n_seen[0]} of {n_max}")
    """
    return res

def calculateSmallestNumberWithTheFirstNSumOfDigitFactorialsDigitSumTotal(n_max: int=150, base: int=10) -> int:
    """
    Solution to Project Euler #254
    """
    freq_lsts = calculateSmallestNumberDigitFrequenciesWithTheFirstNSumOfDigitFactorials(n_max, base=base)
    print(freq_lsts)
    res = 0
    for freq_lst in freq_lsts[1:]:
        for d, f in enumerate(freq_lst):
            res += d * f
    return res
    """
    sg_lst = calculateSmallestNumberWithTheFirstNSumOfDigitFactorials(n_max, base=base)
    print(sg_lst)
    fact_lst = [(-1, [])]
    for num in sg_lst[1:]:
        if num < 0:
            fact_lst.append((-1, []))
            continue
        num2 = 0
        num2_comps = []
        while num:
            num, d = divmod(num, base)
            d2 = math.factorial(d)
            num2 += d2
            num2_comps.append(d2)
        fact_lst.append((num2, num2_comps))
    #print(fact_lst)
    res = 0
    for i in range(1, len(sg_lst)):
        num = sg_lst[i]
        while num:
            num, d = divmod(num, base)
            res += d
    """
    return res

# Problem 255
def calculateNumberOfIterationsOfHeronsMethodForIntegers(n: int, base: int=10) -> int:

    n_dig = 0
    num2 = n
    while num2:
        num2 //= base
        n_dig += 1
    x0 = None
    x = 2 * base ** (n_dig >> 1) if n_dig & 1 else 7 * base ** ((n_dig - 2) >> 1)
    res = 0
    while x != x0:
        res += 1
        x0 = x
        x = (x0 + ((n - 1) // x0) + 1) >> 1
    return res

def meanNumberOfIterationsOfHeronsMethodForIntegersFraction(
    n_min: int,
    n_max: int,
    base: int=10,
) -> CustomFraction:


    def calculateTransition(rng: Tuple[int], lower: int) -> int:

        lft, rgt = rng
        while lft < rgt:
            mid = rgt - ((rgt - lft) >> 1)
            #print(lft, rgt, mid, calculateNumberOfIterationsOfHeronsMethodForIntegers(mid, base=base), lower)
            if calculateNumberOfIterationsOfHeronsMethodForIntegers(mid, base=base) > lower:
                rgt = mid - 1
            else: lft = mid
        return lft

    rt_mn = isqrt(n_min)
    rt_mx = isqrt(n_max)
    #print(rt_mn, rt_mx)
    frac1 = CustomFraction(2 * rt_mn + 1, 2) 
    #print(frac1, frac1 * frac1)
    if frac1 * frac1 < n_min:
        rt_mn += 1
    frac2 = CustomFraction(2 * rt_mx + 1, 2)
    if frac2 * frac2 < n_max:
        rt_mx += 1
    #print(rt_mn, rt_mx)
    #print(f"rt_mn = {rt_mn}, rt_mn ** 2 = {rt_mn ** 2}, rt_mx = {rt_mx}, rt_mx ** 2 = {rt_mx ** 2}")
    res = 0
    for rt in range(rt_mn, rt_mx + 1):
        #print(f"root = {rt} (iterating between {rt_mn} and {rt_mx})")
        frac1 = CustomFraction(2 * rt - 1, 2)
        frac1_sq = frac1 * frac1
        sq_mn = max(n_min, ((frac1_sq.numerator - 1) // frac1_sq.denominator) + 1)
        frac2 = CustomFraction(2 * rt + 1, 2)
        frac2_sq = frac2 * frac2
        sq_mx = min(n_max, ((frac2_sq.numerator - 1) // frac2_sq.denominator))
        #print(sq_mn, sq_mx, n_max)
        if sq_mn > sq_mx: break
        n_dig_sq_mn = 0
        num = sq_mn
        while num:
            num //= base
            n_dig_sq_mn += 1
        n_dig_sq_mx = n_dig_sq_mn
        num = sq_mx // (base ** n_dig_sq_mn)
        while num:
            num //= base
            n_dig_sq_mx += 1
        if n_dig_sq_mx == n_dig_sq_mn:
            rngs = [((sq_mn, sq_mx), n_dig_sq_mn)]
        else:
            rngs = [((sq_mn, base ** (n_dig_sq_mn) - 1), n_dig_sq_mn)]
            for n_dig in range(n_dig_sq_mn + 1, n_dig_sq_mx):
                rngs.append(((base ** (n_dig - 1), base ** n_dig - 1), n_dig))
            rngs.append(((base ** (n_dig_sq_mx - 1), sq_mx), n_dig_sq_mx))
        #print(rt, rngs)
        for sub_rng in rngs:
            cnt_rng = (
                calculateNumberOfIterationsOfHeronsMethodForIntegers(sub_rng[0][0], base=base),
                calculateNumberOfIterationsOfHeronsMethodForIntegers(sub_rng[0][1], base=base),
            )
            #print(f"sub_rng = {sub_rng}, cnt_rng = {cnt_rng}")
            sub_rng2 = list(sub_rng[0])
            for iter_cnt in range(*cnt_rng):
                #print(f"iter_cnt = {iter_cnt}")
                mid = calculateTransition(sub_rng2, iter_cnt)
                res += iter_cnt * (mid - sub_rng2[0] + 1)
                #print(f"answer for sub range [{sub_rng2[0]}, {mid}] = {iter_cnt}")
                #for num in range(sub_rng2[0], mid + 1):
                #    ans = calculateNumberOfIterationsOfHeronsMethodForIntegers(num, base=base)
                #    if ans != iter_cnt:
                #        print(f"{num}: {calculateNumberOfIterationsOfHeronsMethodForIntegers(num, base=base)} (expected {iter_cnt})")
                sub_rng2[0] = mid + 1
            iter_cnt = cnt_rng[1]
            res += iter_cnt * (sub_rng2[1] - sub_rng2[0] + 1)
            #print(f"answer for sub range [{sub_rng2[0]}, {sub_rng2[1]}] = {iter_cnt}")
            #for num in range(sub_rng2[0], sub_rng2[1] + 1):
            #    ans = calculateNumberOfIterationsOfHeronsMethodForIntegers(num, base=base)
            #    if ans != iter_cnt:
            #        print(f"{num}: {calculateNumberOfIterationsOfHeronsMethodForIntegers(num, base=base)} (expected {iter_cnt})")
    print(res, n_max - n_min + 1)
    res = CustomFraction(res, n_max - n_min + 1)
    #print(res)
    return res
    
def meanNumberOfIterationsOfHeronsMethodForIntegersFloat(
    n_min: int,
    n_max: int,
    base: int=10,
) -> float:
    """
    Solution to Project Euler #255
    """
    res = meanNumberOfIterationsOfHeronsMethodForIntegersFraction(n_min, n_max, base=base)
    print(res)
    return res.numerator / res.denominator

# Problem 257
def angularBisectorTrianglePartitionIntegerRatioCountBruteForce(perimeter_max: int) -> int:

    res = 0
    m_primitive_cnts = {}
    for perim in range(3, perimeter_max + 1):
        mult = perimeter_max // perim
        for a in range(1, (perim) // 3 + 1):
            prod = perim * a
            b_plus_c = perim - a
            c_max = min((perim - 1) >> 1, perim - 2 * a)
            c_min = max((b_plus_c + 1) >> 1, a)
            #print(perim, a, b_plus_c, c_min, c_max)
            for c in range(c_min, c_max + 1):
                b = b_plus_c - c
                #print(a, b, c)
                g = gcd(b, c)
                if g > 1: continue
                if prod % (b * c): continue
                m, r = divmod((a + b) * (a + c), (b * c))
                print((a, b, c), m, r)
                #m_set.add(m)
                m_primitive_cnts[m] = m_primitive_cnts.get(m, 0) + 1
                res += mult
    #print(m_set)
    print(m_primitive_cnts)
    return res

def angularBisectorTrianglePartitionIntegerRatioCount(perimeter_max: int=10 ** 8) -> int:
    """
    Solution to Project Euler #257
    """
    
    res = perimeter_max // 3 # equilateral triangles (ratio = 4)
    if perimeter_max < 4: return res

    # Ratio = 2
    print("counting solutions for ratio 2")
    M = 2
    discr = (M - 1) ** 2 + 4 * perimeter_max
    m_max = (isqrt(discr) - (M + 1)) // 2
    cnts = {}
    for m in range(1, m_max + 1):
        #discr = 9 * m ** 2 - 4 * M * perimeter_max
        n_min = (m - 1) // M + 1
        n_max = min((isqrt((M - 1) ** 2 * m ** 2 + 4 * M * perimeter_max) - (M + 1) * m) // 4, m)
        #print(f"m = {m}, n_max = {n_max}")
        for n in range(n_min, n_max + 1):
            a, b, c = sorted([(M - 1) * m * n, m * n + M * n ** 2, m * n + m ** 2])
            if c >= a + b: continue
            if gcd(b, c) > 1: continue
            #print(f"solution for ratio 2: {(a, b, c)}, m = {m}, n = {n}")
            cnts[2] = cnts.get(2, 0) + 1
            res += perimeter_max // (a + b + c)
    
    # Ratio = 3
    M = 3

    # n has different parity from m
    print("counting solutions for ratio 3, different parity")
    discr = (M - 1) ** 2 + 4 * perimeter_max
    m_max = (isqrt(discr) - (M + 1)) // 2
    for m in range(1, m_max + 1):
        #discr = 9 * m ** 2 - 4 * M * perimeter_max
        n_min = (m - 1) // M + 1
        n_min += not (n_min + m) & 1
        n_max = min((isqrt((M - 1) ** 2 * m ** 2 + 4 * M * perimeter_max) - (M + 1) * m) // 4, m)
        #print(f"m = {m}, n_max = {n_max}")
        for n in range(n_min, n_max + 1, 2):
            a, b, c = sorted([(M - 1) * m * n, m * n + M * n ** 2, m * n + m ** 2])
            if c >= a + b: continue
            if gcd(b, c) > 1: continue
            #print(f"solution for ratio 3, different parity: {(a, b, c)}, m = {m}, n = {n}")
            cnts[3] = cnts.get(3, 0) + 1
            res += perimeter_max // (a + b + c)


    # n has the same parity as m
    print("counting solutions for ratio 3, same parity")
    discr = (2 * perimeter_max + 1)
    m_max = isqrt(discr) - 2
    for m in range(1, m_max + 1):
        n_min = (m - 1) // M + 1
        n_min += (m + n_min) & 1
        n_max = min((isqrt(m ** 2 + 6 * perimeter_max) - 2 * m) // 3, m)
        for n in range(2 - (m & 1), n_max + 1, 2):
            a, b, c = sorted([m * n, (m * n + M * n ** 2) // 2, (m * n + m ** 2) // 2])
            #print(f"second type: {a, b, c}, m = {m}, n = {n}")
            if c >= a + b: continue
            if gcd(b, c) > 1: continue
            #print(f"solution for ratio 3, same parity: {(a, b, c)}, m = {m}, n = {n}")
            cnts[3] = cnts.get(3, 0) + 1
            res += perimeter_max // (a + b + c)
    print(cnts)
    """
    for M in (2, 3):
        discr = perimeter_max - 2 * M + 2
        if discr < 0: break
        m_max = isqrt(discr) - 1
        for m in range(1, m_max + 1):
            #discr = 9 * m ** 2 - 4 * M * perimeter_max
            n_max = (isqrt((M - 1) ** 2 * m ** 2 + 4 * M * perimeter_max) - (M + 1) * m) // (2 * M)
            for n in range(1, n_max + 1):
                a, b, c = sorted([(M - 1) * m * n, m * n + M * n ** 2, m * n + m ** 2])
                if c >= a + b: continue
                if gcd(b, c) > 1: continue
                print(a, b, c)
                res += perimeter_max // (a + b + c)
    """
    return res

# Problem 258
def calculateLaggedFibonacciTerm(
    term_idx: int=10 ** 18,
    initial_terms: List[int]=[1] * 2000,
    prev_terms_to_sum: List[int]=[1999, 2000],
    res_md: Optional[int]=20092010,
) -> int:
    """
    Solution to Project Euler #258
    """
    # Review- look into method using Cayley-Hamilton theory, from which
    # we can get (for the default values) that the transition matrix M satisfies:
    #  M ** 2000 = M + I
    n = len(initial_terms)
    if term_idx < n: return initial_terms[n]

    modAdd0 = (lambda x, y: x + y) if res_md is None else (lambda x, y: (x + y) % res_md)
    modMultiply0 = (lambda x, y: x * y) if res_md is None else (lambda x, y: (x * y) % res_md)

    n_iter = term_idx - n + 1
    transition_matrix = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        transition_matrix[i][i + 1] = 1
    for j in prev_terms_to_sum:
        transition_matrix[-1][n - j] = 1
    
    #trans = np.matrix(transition_matrix, dtype=int)

    def matrixMultiply(
        M1: np.matrix,
        M2: np.matrix,
        md: np.matrix,
    ) -> List[List[int]]:
        #print(M1.shape, M2.shape)
        res = np.linalg.matmul(M1, M2)
        res %= md
        return res
    
    n_iter2 = n_iter
    p = np.matrix(transition_matrix, dtype=int)
    if n_iter2 & 1:
        curr = np.matrix(transition_matrix, dtype=int)
    else:
        curr = np.identity(n, dtype=int)
    n_iter2 >>= 1
    while n_iter2:
        print(n_iter2)
        p = matrixMultiply(p, p, md=res_md)
        print("finished multiply")
        #print(p.shape)
        if n_iter2 & 1:
            curr = matrixMultiply(p, curr, md=res_md)
        n_iter2 >>= 1
    res = 0
    for i in range(n):
        res = modAdd0(res, modMultiply0(curr[n - 1, i], initial_terms[i]))
    return res
    
    """
    def matrixMultiply(
        M1: List[List[int]],
        M2: List[List[int]],
        md: Optional[int]=None,
    ) -> List[List[int]]:
        modAdd = (lambda x, y: x + y) if md is None else (lambda x, y: (x + y) % md)
        modMultiply = (lambda x, y: x * y) if md is None else (lambda x, y: (x * y) % md)
        n1 = len(M1)
        n2 = len(M2[0])
        m = len(M2)
        if len(M1[0]) != m: 
            raise ValueError("The number of columns in M1 must equal "
                             "the number of rows in M2")
        res = [[0] * n2 for _ in range(n1)]
        for i1 in range(n1):
            print(f"i1 = {i1}")
            for i2 in range(n2):
                for j in range(m):
                    res[i1][i2] = modAdd(res[i1][i2], modMultiply(M1[i1][j], M2[j][i2]))
        return res
    
    n_iter2 = n_iter
    p = transition_matrix
    if n_iter2 & 1:
        curr = p
    else:
        curr = [[0] * n for _ in range(n)]
        for i in range(n): curr[i][i] = 1
    n_iter2 >>= 1
    while n_iter2:
        print(n_iter2)
        p = matrixMultiply(p, p, md=res_md)
        print("finished multiply")
        if n_iter2 & 1:
            curr = matrixMultiply(p, curr, md=res_md)
        n_iter2 >>= 1
    res = 0
    for i in range(n):
        res = modAdd0(res, modMultiply0(curr[-1][i], initial_terms[i]))
    return res
    """

# Problem 259
def calculateReachableIntegers(dig_min: int=1, dig_max: int=9, base: int=10) -> List[int]:

    def concatenationsGenerator(
        digs: List[int],
        base: int,
    ) -> Generator[int, None, None]:
        #print(digs)
        curr = [digs[0]]
        def recur(idx: int) -> Generator[int, None, None]:
            #print(idx)
            if idx == len(digs):
                yield curr
                return
            orig = curr[-1]
            curr.append(digs[idx])
            yield from recur(idx + 1)
            curr.pop()
            curr[-1] = curr[-1] * base + digs[idx]
            yield from recur(idx + 1)
            curr[-1] = orig
            return

        yield from recur(1)
        return
    
    def operationsGenerator(
        num1: CustomFraction,
        num2: CustomFraction,
    ) -> Generator[CustomFraction, None, None]:
        #print(num1, num2)
        yield num1 + num2
        yield num1 - num2
        yield num1 * num2
        if num2 != 0:
            yield num1 / num2
        return
    #cnt = 0
    #for lst in concatenationsGenerator(list(range(1, base)), base):
    #    print(lst)
    #    cnt += 1
    #print (f"count = {cnt}")
    #return []
    
    #curr_incl = SortedList()
    curr = SortedDict({i: CustomFraction(i, 1) for i in range(1, base)})

    res = set()

    seen = set()
    def recur(start: int, prev_changed: List[int], curr_changed: List[int]) -> None:
        #print(curr, prev_changed, start)
        if len(curr) == 1:
            #print(curr)
            frac = curr.peekitem(0)[1]
            if frac.denominator == 1 and frac.numerator > 0 and frac.numerator not in res:
                #print(frac.numerator)
                res.add(frac.numerator)
            return
        if not start:
            args = tuple(curr[i] for i in curr)
            if args in seen: return
            seen.add(args)
        if curr_changed:
            recur(0, curr_changed, [])
        if start > prev_changed[-1]:
            return
        for i in range(len(prev_changed)):
            if prev_changed[i] >= start: break
        for i in range(i, len(prev_changed)):
            idx = prev_changed[i]
            if idx not in curr.keys(): continue
            j = curr.bisect_left(idx)
            inds = []
            if idx > start and j > 0:
                idx0 = curr.peekitem(j - 1)[0]
                if not i or idx0 != prev_changed[i - 1]:
                    inds.append((idx0, idx))
            if j < len(curr) - 1:
                idx2 = curr.peekitem(j + 1)[0]
                inds.append((idx, idx2))
            #print(inds, curr)
            for pair in inds:
                curr_changed.append(pair[0])
                num1, num2 = [curr[idx] for idx in pair]
                curr.pop(pair[1])
                for num in operationsGenerator(num1, num2):
                    curr[pair[0]] = num
                    recur(start + 2, prev_changed, curr_changed)
                curr[pair[0]] = num1
                curr[pair[1]] = num2
                curr_changed.pop()
        return

    for nums in concatenationsGenerator(list(range(dig_min, min(dig_max + 1, base))), base):
        print(f"concatenation {nums}")
        curr = SortedDict({i: CustomFraction(num, 1) for i, num in enumerate(nums)})
        recur(0, list(range(len(curr))), [])
    return sorted(res)

def calculateReachableNumbersSum(dig_min: int=1, dig_max: int=9, base: int=10) -> int:
    """
    Solution to Project Euler #259
    """
    res = sum(calculateReachableIntegers(dig_min=dig_min, dig_max=dig_max, base=base))
    return res

# Problem 260
def stoneGamePlayerTwoWinningConfigurationsGenerator(n_piles: int, pile_size_max: int) -> Generator[Tuple[int], None, None]:
    # Using Sprague-Grundy
    if pile_size_max < 0: return
    #state0 = tuple([0] * n_piles)
    #yield state0
    """
    seen_non_winning = set()
    def winning(state: List[int]) -> bool:
        state = tuple(sorted(state))
        print(f"state = {state}")
        length = len(state)
        for idx, num in enumerate(state):
            length -= 1
            if not num or idx and num == state[idx - 1]: continue
            sub = state[idx]
            for bm in range(1 << length):
                state2 = list(state)
                
                state2[idx] = 0
                for i in range(length):
                    if bm & 1:
                        state2[idx + i + 1] -= sub
                        if bm == 1: break
                    bm >>= 1
                print(f"state2 = {state2}")
                if tuple(sorted(state2)) in seen_non_winning:
                    print(f"non-winning state 2 = {state2}")
                    return True
        seen_non_winning.add(state)
        return False
                
    """
    seen_non_winning = set()#{state0}
    def winning(state: List[int]) -> bool:
        state = tuple(sorted(state))
        print(f"state = {state}")
        #if state[-1] == 0: return False
        #args = state
        #if args in memo.keys(): return memo[args]
        #res = 0
        #seen = SortedSet()
        for bm in range(1, 1 << n_piles):
            idx_lst = []
            mx = float("inf")
            for i in range(n_piles):
                if bm & 1:
                    mx = min(mx, state[i])
                    idx_lst.append(i)
                    if bm == 1: break
                bm >>= 1
            state2 = list(state)
            for sub in range(1, mx + 1):
                for idx in idx_lst:
                    state2[idx] -= 1
                if tuple(sorted(state2)) in seen_non_winning:
                    print(f"non-winning state 2 = {state2}")
                    return True
        seen_non_winning.add(state)
        return False
    
    curr = []
    def recur(idx: int) -> Generator[Tuple[int], None, None]:
        """
        if idx == n_piles - 1:
            mn = curr[-1] if curr else 0
            curr.append(mn)
            for _ in range(mn, pile_size_max + 1):
                if not winning(curr):
                    yield tuple(curr)
                    break
                curr[-1] += 1
            curr.pop()
            return
        """
        if idx == n_piles - 2:
            
            #print(curr)
            mn = curr[-1] if curr else 0
            skipped_gaps = SortedSet()
            nxt_nonskipped_gap = 0
            curr.extend([0, 0])
            seen = set()
            for i1 in range(mn, pile_size_max + 1):
                if i1 in seen: continue
                #print(i1, nxt_nonskipped_gap, skipped_gaps, seen)
                curr[-2] = i1
                for j in reversed(range(len(skipped_gaps))):
                    i2 = i1 + skipped_gaps[~j]
                    if i2 > pile_size_max: break
                    curr[-1] = i2
                    if not winning(curr):
                        yield tuple(curr)
                        seen.add(i2)
                        skipped_gaps.pop(~j)
                        break
                if i1 in seen: continue
                gap = pile_size_max - i1
                for gap in range(nxt_nonskipped_gap, pile_size_max - i1 + 1):
                    i2 = i1 + gap
                    if i2 in seen: continue
                    
                    curr[-1] = i2
                    if not winning(curr):
                        yield tuple(curr)
                        seen.add(i2)
                        break
                    skipped_gaps.add(gap)
                nxt_nonskipped_gap = gap + 1
                seen.add(i1)
            curr.pop()
            curr.pop()
            return
        
            
            
        mn = curr[-1] if curr else 0
        curr.append(mn)
        for _ in range(mn, pile_size_max + 1):
            yield from recur(idx + 1)
            curr[-1] += 1
        curr.pop()
        return
    
    yield from recur(0)

def stoneGamePlayerTwoWinningConfigurationsGenerator2(pile_size_max: int) -> Generator[Tuple[int], None, None]:
    # Based on https://euler.stephan-brumme.com/260/
    # Review- generalise to any number of piles
    def enc(j1: int, j2: int) -> int:
        #if a > b: a, b = b, a
        return j1 * (pile_size_max + 1) + j2

    one = [False] * ((pile_size_max + 1) ** 2)
    two = [False] * ((pile_size_max + 1) ** 2)
    three = [False] * ((pile_size_max + 1) ** 2)

    for i1 in range(pile_size_max + 1):
        for i2 in range(i1, pile_size_max + 1):
            if one[enc(i1, i2)]: continue
            for i3 in range(i2, pile_size_max + 1):
                if (one[enc(i1, i3)] or one[enc(i2, i3)] or two[enc(i2 - i1, i3)] or two[enc(i3 - i1, i2)] or two[enc(i3 - i2, i1)] or three[enc(i2 - i1, i3 - i1)]):
                    continue
                yield (i1, i2, i3)
                one[enc(i1, i2)] = True
                one[enc(i1, i3)] = True
                one[enc(i2, i3)] = True
                two[enc(i2 - i1, i3)] = True
                two[enc(i3 - i1, i2)] = True
                two[enc(i3 - i2, i1)] = True
                three[enc(i2 - i1, i3 - i1)] = True
                break
    return

def stoneGamePlayerTwoWinningConfigurationsSum(n_piles: int=3, pile_size_max: int=1000) -> int:
    res = 0
    cnt = 0
    counts = []
    for state in stoneGamePlayerTwoWinningConfigurationsGenerator2(pile_size_max=pile_size_max): #stoneGamePlayerTwoWinningConfigurationsGenerator(n_piles=n_piles, pile_size_max=pile_size_max):
        #print(state)
        counts +=[0] * (state[-1] - len(counts) + 1)
        for num in state:
            counts[num] += 1
        cnt += 1
        res += sum(state)
    print(f"number of states where player 2 is winning = {cnt}")
    print(counts)
    return res

# Problem 265
def findAllBinaryCircles(n: int) -> List[int]:
    if n == 1: return [1]
    s_len = 1 << n
    Trie = lambda: defaultdict(Trie)
    init_lst = [1] + ([0] * n) + [1]

    full_trie = Trie()
    trie_lst = []
    for i0 in range(len(init_lst)):
        t = full_trie
        t["tot"] = t.get("tot", 0) + 1
        length = min(n, len(init_lst) - i0)
        for j in range(length):
            t = t[init_lst[i0 + j]]
            t["tot"] = t.get("tot", 0) + 1
        if length < n:
            #print(i0, init_lst[i0:i0 + length], t)
            trie_lst.append(t)
    
    """
    for num in init_lst:
        t["tot"] = 1
        t = t[num]
    
    t = full_trie
    t["tot"] = t.get("tot", 0) + n + 1
    for i0 in range(n):
        t2 = t[1]
        t2["tot"] = 1
        trie_lst.append(t2)
        t = t[0]
        t["tot"] = n - i0
    trie_lst = trie_lst[::-1]
    """
    dig_lst = list(init_lst)
    #print(full_trie)
    #print(trie_lst)
    res = []

    def removeSubs(trie: Any) -> None:

        def recur2(t: Any, num: int, t0: Optional[Any]) -> None:
            if t0 is not None and "sub" not in t.keys(): return
            for num2 in range(2):
                if not num2 in t.keys(): continue
                recur2(t[num2], num2, t)
            sub = t.pop("sub", 0)
            if not sub: return
            t["tot"] -= sub
            if t["tot"] or t0 is None: return
            #print(t0, t, num)
            t0.pop(num)
            return
        recur2(trie, -1, None)
        return

    def recur(idx: int, trie_lst: List[Any]) -> None:
        if idx == s_len:
            #print("hi")
            n_exp0 = 1
            #print(full_trie)
            for i, t in enumerate(trie_lst):
                n_exp = n_exp0
                for j in range(i + 1):
                    num = dig_lst[j]
                    t = t[num]
                    t["sub"] = t.get("sub", 0) + 1
                    t["tot"] = t.get("tot", 0) + 1
                    if t["tot"] > n_exp: break
                    n_exp >>= 1
                else:
                    n_exp0 <<= 1
                    continue
                break
            else:
                ans = 0
                for d in dig_lst[1:]:
                    ans = (ans << 1) + d
                ans <<= 1
                if dig_lst[0]: ans += 1
                res.append(ans)
                #print(ans, format(ans, "b"), len(trie_lst))
                #print(full_trie)
            for t in trie_lst:
                removeSubs(t)
            return
        
        dig_lst.append(0)
        full_trie["tot"] = full_trie.get("tot", 0) + 1
        for num in range(2):
            n_exp = 1
            to_stop = False
            for t in trie_lst:
                t2 = t[num]
                if t2.get("tot", 0) >= n_exp:
                    to_stop = True
                    break
                n_exp <<= 1
            else:
                t = full_trie[num]
                if t.get("tot", 0) >= n_exp:
                    to_stop = True
            if to_stop: continue
            dig_lst[-1] = num
            trie_lst2 = []
            t2 = trie_lst[0][num]
            t2["tot"] = t2.get("num", 0) + 1
            for t in trie_lst[1:]:
                t2 = t[num]
                t2["tot"] = t2.get("tot", 0) + 1
                trie_lst2.append(t2)
            
            t2 = full_trie[num]
            t2["tot"] = t2.get("tot", 0) + 1
            trie_lst2.append(t2)
            recur(idx + 1, trie_lst2)
            for t2 in trie_lst2:
                t2["tot"] -= 1
            t2 = trie_lst[0][num]
            t2["tot"] -= 1
        full_trie["tot"] -= 1
        dig_lst.pop()
    
    recur(n + 2, trie_lst)

    return res

def allBinaryCirclesSum(n: int=5) -> List[int]:
    """
    Solution to Project Euler #265
    """
    return sum(findAllBinaryCircles(n))

##############
project_euler_num_range = (51, 100)

def evaluateProjectEulerSolutions51to100(eval_nums: Optional[Set[int]]=None) -> None:
    if not eval_nums:
        eval_nums = set(range(project_euler_num_range[0], project_euler_num_range[1] + 1))

    since0 = time.time()

    if 251 in eval_nums:
        since = time.time()
        res = cardanoTripletCount(sum_max=11 * 10 ** 7)
        print(f"Solution to Project Euler #251 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 252 in eval_nums:
        since = time.time()
        res = blumBlumShubPseudoRandomTwoDimensionalPointsLargestEmptyConvexHoleArea(
            n_points=500,
            blumblumshub_s_0=290797,
            blumblumshub_s_mod=50515093,
            coord_min=-1000,
            coord_max=999,
        )
        print(f"Solution to Project Euler #252 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 253 in eval_nums:
        since = time.time()
        res = constructingLinearPuzzleMaxSegmentCountMeanFloat(n_pieces=40)
        print(f"Solution to Project Euler #253 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 254 in eval_nums:
        since = time.time()
        res = calculateSmallestNumberWithTheFirstNSumOfDigitFactorialsDigitSumTotal(n_max=150, base=10)
        print(f"Solution to Project Euler #254 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 255 in eval_nums:
        since = time.time()
        res = meanNumberOfIterationsOfHeronsMethodForIntegersFloat(
            n_min=10 ** 13,
            n_max=10 ** 14 - 1,
            base=10,
        )
        print(f"Solution to Project Euler #255 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 257 in eval_nums:
        since = time.time()
        res = angularBisectorTrianglePartitionIntegerRatioCount(perimeter_max=10 ** 8)
        print(f"Solution to Project Euler #257 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 258 in eval_nums:
        since = time.time()
        res = calculateLaggedFibonacciTerm(
            term_idx=10 ** 18,
            initial_terms=[1] * 2000,
            prev_terms_to_sum=[1999, 2000],
            res_md=20092010,
        )
        print(f"Solution to Project Euler #258 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 259 in eval_nums:
        since = time.time()
        res = calculateReachableNumbersSum(dig_min=1, dig_max=9, base=10)
        print(f"Solution to Project Euler #259 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 260 in eval_nums:
        since = time.time()
        res = stoneGamePlayerTwoWinningConfigurationsSum(n_piles=3, pile_size_max=1000)
        print(f"Solution to Project Euler #260 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 265 in eval_nums:
        since = time.time()
        res = allBinaryCirclesSum(n=5)
        print(f"Solution to Project Euler #265 = {res}, calculated in {time.time() - since:.4f} seconds")

    print(f"Total time taken = {time.time() - since0:.4f} seconds")

if __name__ == "__main__":
    eval_nums = {258}
    evaluateProjectEulerSolutions51to100(eval_nums)

"""
n_max = 1000
for n in range(1, n_max + 1):
    res = func(n)
    res2 = func2(n)
    if res != res2:
        print(n, res, res2)
"""
"""
for k in range(1, 101):
    #num = 8 * a ** 3 + 15 * a ** 2 + 6 * a - 1
    #if not num % 27:
    #    print(a, num // 27)
    num = 8 * k - 3
    print(k, 3 * k - 1, num, k ** 2 * num)
"""
"""
def upperBoundDigitSumCoarse(max_dig_count: int, base: int=10) -> Tuple[int, int]:
    num_max = math.factorial(base - 1) * (max_dig_count + 1) - 1
    n_dig = 0
    num2 = num_max
    while num2 >= base:
        num2 //= base
        n_dig += 1
    #n_dig = max(n_dig, mx_non_max_dig_n_dig)
    return num_max, n_dig * (base - 1) + num2

prev = -1
for i in range(1, 10 ** 9):
    n_dig = upperBoundDigitSumCoarse(i, base=10)[1]
    if n_dig > prev:
        print(i, n_dig)
        prev = n_dig
"""
"""
print(calculateNumberOfIterationsOfHeronsMethodForIntegers(4321))
for n_dig in range(1, 9):
    f_dict = {}
    for num in range(2 ** (n_dig - 1), 2 ** n_dig):
        n_iter = calculateNumberOfIterationsOfHeronsMethodForIntegers(num, base=10)
        #print(num, n_iter)
        #if n_iter == 1: print(num)
        f_dict[n_iter] = f_dict.get(n_iter, 0) + 1
    print(n_dig, f_dict)
    tot, cnt = 0, 0
    for num, f in f_dict.items():
        tot += f * num
        cnt += f
    print(tot, cnt, tot / cnt)
"""
"""
mx_n_dig = 5
n_iter_prev = calculateNumberOfIterationsOfHeronsMethodForIntegers(1, base=10)
curr_rng_start = 1
for num in range(2, 10 ** mx_n_dig):
    n_iter = calculateNumberOfIterationsOfHeronsMethodForIntegers(num, base=10)
    if n_iter == n_iter_prev: continue
    print((curr_rng_start, num - 1), num - curr_rng_start, n_iter_prev)
    curr_rng_start = num
    n_iter_prev = n_iter
    #if n_iter == 1: print(num)
"""
#print(meanNumberOfIterationsOfHeronsMethodForIntegersFraction(7, 2606, base=10))