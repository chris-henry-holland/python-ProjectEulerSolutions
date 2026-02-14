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

from data_structures.fractions import CustomFraction
from data_structures.prime_sieves import PrimeSPFsieve, SimplePrimeSieve

from algorithms.number_theory_algorithms import gcd, lcm, isqrt, integerNthRoot, solveLinearCongruence
from algorithms.pseudorandom_number_generators import blumBlumShubPseudoRandomGenerator
from algorithms.continued_fractions_and_Pell_equations import pellSolutionGenerator, generalisedPellSolutionGenerator, pellFundamentalSolution

# Problem 251
def cardanoTripletGeneratorBySum(
    sum_max: Optional[int]=None,
) -> Generator[Tuple[int, Tuple[int, int, int]], None, None]:
    """
    Generator iterating over the Cardano Triplets in increasing order
    of the sum of the triplet integers, with, if sum_max given, sum no
    greater than sum_max.

    An Cardano Triplet is an ordered triple of strictly positive integers
    (a, b, c) that satisfies:
        cuberoot(a + b * sqrt(c)) + cuberoot(a - b * sqrt(c)) = 1
    
    Note that if sum_max is not specified, the generator never
    of itself terminates and thus any in such a case for loops over
    this generator must include provision to terminate (e.g. a
    break or return statement inside a conditional, or be zipped
    with a generator that does terminate), otherwise it would result
    in an infinite loop.

    Args:
        Optional named:
        sum_max (int or None): If given as an integer, is the
                inclusive upper bound on the sum of Cardano Triplets
                yielded, otherwise there is no such upper bound. 
                Note that if this is not given or given as None,
                the generator does not of itself terminate.
            Default: None
    
    Yields:
    3-tuple of integers representing a Cardano Triplet. These are
    yielded in strictly increasing order of the sum of the three
    numbers, with different triples with the same sum ordered
    in increasing lexicographic order. A Cardano Triplet is
    yielded only after all existing Cardano Triplets that precede
    it by this ordering are yielded (i.e. no Cardano Triplets are
    skipped) and if sum_max is given as an integer, all Cardano
    Triplets whose sum is no greater than sum_max are yielded
    before terminating.
    Note that if sum_max is not given or given as None, the generator
    does not of itself terminate so in any in such a case for loops
    over this generator must include provision to terminate (e.g. a
    break or return statement inside a conditional, or be zipped
    with a generator that does terminate), otherwise it would result
    in an infinite loop.

    Outline of rationale:
    TODO
    """
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

    Calculates the number of Cardano Triplets for which the sum
    of the three integers in the triplet is no greater than
    sum_max.

    An Cardano Triplet is an ordered triple of strictly positive
    integers (a, b, c) that satisfies:
        cuberoot(a + b * sqrt(c)) + cuberoot(a - b * sqrt(c)) = 1

    Args:
        Optional named:
        sum_max (int): The inclusive upper bound on the sum of
                Cardano Triplets included in the count.
            Default: 11 * 10 ** 7
    
    Yields:
    Integer (int) giving the number of Cardano Triplets for which
    the sum of the three integers in the triplet is no greater than
    sum_max.

    Outline of rationale:
    TODO
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
def triangleDoubleArea(
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    p3: Tuple[int, int],
) -> int:
    """
    For a triangle in the Cartesian plane whose vertices
    p1, p2 and p3 have integer Cartesian coordinates,
    calculates double the area of the triangle.

    Args:
        Required positional:
        p1 (2-tuple of ints): The integer Cartesian coordinates
                giving the position of the first of the
                triangle vertices.
        p2 (2-tuple of ints): The integer Cartesian coordinates
                giving the position of the second of the
                triangle vertices.
        p3 (2-tuple of ints): The integer Cartesian coordinates
                giving the position of the third of the
                triangle vertices.

    Returns:
    Integer (int) giving twice the area of the triangle in
    the Cartesian plane whose vertices are at the positions
    with Cartesian coordinates p1, p2 and p3 respectively.
    """
    res = abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
    #print(p1, p2, p3, res)
    return res

def calculateLargestEmptyConvexPolygonDoubleArea(
    points: List[Tuple[int, int]],
) -> int:
    """
    For the points in the Cartesian plane with integer Cartesian
    coordinates given by the list points, calculates the largest
    integer such that there exists a convex hole whose area is
    equal to half that number.

    For the points in the Cartesian plane with integer Cartesian
    coordinates given by the list points, calculates the largest
    integer such that there exists a convex polygon for which
    twice its area is equal to that integer, whose vertices are
    all in points and there are no other elements of points
    that are in the interior of the polygon (with the perimeter
    of the polygon not being considered to be in the interior of
    the polygon).

    A polygon is convex if and only if for any two points in the
    interior of the polygon, every point on the straight line
    between the two points is also in the interior of the polygon.

    Args:
        Required positional:
        points (list of 2-tuples of ints): The Cartesian coordinates
                of the points in the plane for which the double
                area of the largest polygon satisfying the given
                constraints is to be found.
    
    Returns:
    Integer (int) giving the largest number such that there exists
    a convex hole for the points in the Cartesian plane in the list
    points whose area is half that integer.

    Outline of rationale:
    TODO
    """
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

def calculateLargestEmptyConvexPolygonAreaFloat(
    points: List[Tuple[int, int]],
) -> float:
    """
    For the points in the Cartesian plane with integer Cartesian
    coordinates given by the list points, calculates the largest
    number such that there exists a convex hole whose area is
    equal to that number.
     
    For a set of points in the Cartesian plane a convex hole is
    a convex polygon whose vertices are a subset of that set of
    points and there are no other elements of points that are in
    the interior of the polygon (with the perimeter of the polygon
    not being considered to be in the interior of the polygon).

    A polygon is convex if and only if for any two points in the
    interior of the polygon, every point on the straight line
    between the two points is also in the interior of the polygon.

    Args:
        Required positional:
        points (list of 2-tuples of ints): The Cartesian coordinates
                of the points in the plane for which the area of the
                largest polygon satisfying the given constraints is
                to be found.
    
    Returns:
    Float giving the largest real number such that there exists
    a convex hole for the points in the Cartesian plane in the list
    points whose area is equal to that number.

    Outline of rationale:
    See outline of rationale section of the function
    calculateLargestEmptyConvexPolygonDoubleArea().
    """
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

    For the first n_points points in the Cartesian plane with integer
    Cartesian coordinates generated by the Blum Blum Shub pseudo-random
    number generator blumBlueShubPseudoRandomPointGenerator() with the
    given parameters, calculates the largest number such that there
    exists a convex hole whose area is equal to that number.

    For a set of points in the Cartesian plane a convex hole is
    a convex polygon whose vertices are a subset of that set of
    points and there are no other elements of points that are in
    the interior of the polygon (with the perimeter of the polygon
    not being considered to be in the interior of the polygon).

    A polygon is convex if and only if for any two points in the
    interior of the polygon, every point on the straight line
    between the two points is also in the interior of the polygon.

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
    Float (float) giving the largest real number such that there exists a
    convex hole in the first n_points points generated by
    blumBlueShubPseudoRandomPointGenerator() whose area is equal to that
    number.

    Outline of rationale:
    See outline of rationale section of the function
    calculateLargestEmptyConvexPolygonDoubleArea().
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
def constructingLinearPuzzleMaxSegmentCountDistribution0(
    n_pieces: int,
) -> List[int]:
    """
    For a permutation of the first n_pieces integers, the
    numbers are selected in the order they appear in the
    permutation and placed in another list of length n_pieces
    in the position corresponding to the integer until the
    list is filled. After any given step in this process,
    the number of segments in the list is the number of
    contiguous blocks of filled entries that are in the
    list (where a contiguous block of filled entries is
    a number of consecutive entries in the list, all of
    which are filled, where the first element is either the
    first entry of the list or is preceded by an unfilled
    entry and the last element is either the last entry of
    the list or is succeeded by an unfilled entry). This
    function calculates the counts of the maximum number of
    segments that occur at any part of the process over
    each possible distinct permutation of the n_pieces integers.

    Args:
        Required positional:
        n_pieces (int): The number of integers used during the
                described process.
    
    Returns:
    List of integers (int), where the 0-indexed i:th element
    of the list corresponds to the number of permutations of the
    n_pieces integers which, when the described process is
    applied has a maximum number of segments seen equal to i.

    Outline of rationale:
    TODO
    """
    # Review- look into rewording the documentation for clarity
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
    """
    For a permutation of the first n_pieces integers, the
    numbers are selected in the order they appear in the
    permutation and placed in another list of length n_pieces
    in the position corresponding to the integer until the
    list is filled. After any given step in this process,
    the number of segments in the list is the number of
    contiguous blocks of filled entries that are in the
    list (where a contiguous block of filled entries is
    a number of consecutive entries in the list, all of
    which are filled, where the first element is either the
    first entry of the list or is preceded by an unfilled
    entry and the last element is either the last entry of
    the list or is succeeded by an unfilled entry). This
    function calculates the counts of the maximum number of
    segments that occur at any part of the process over
    each possible distinct permutation of the n_pieces integers.

    Args:
        Required positional:
        n_pieces (int): The number of integers used during the
                described process.
    
    Returns:
    List of integers (int), where the 0-indexed i:th element
    of the list corresponds to the number of permutations of the
    n_pieces integers which, when the described process is
    applied has a maximum number of segments seen equal to i.

    Outline of rationale:
    TODO
    """
    # Review- look into rewording the documentation for clarity
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

def constructingLinearPuzzleMaxSegmentCountMeanFraction(
    n_pieces: int,
) -> CustomFraction:
    """
    For a permutation of the first n_pieces integers, the
    numbers are selected in the order they appear in the
    permutation and placed in another list of length n_pieces
    in the position corresponding to the integer until the
    list is filled. After any given step in this process,
    the number of segments in the list is the number of
    contiguous blocks of filled entries that are in the
    list (where a contiguous block of filled entries is
    a number of consecutive entries in the list, all of
    which are filled, where the first element is either the
    first entry of the list or is preceded by an unfilled
    entry and the last element is either the last entry of
    the list or is succeeded by an unfilled entry). This
    function calculates the mean of the maximum number of
    segments that occur at any part of the process over
    each possible distinct permutation of the n_pieces integers.

    Args:
        Required positional:
        n_pieces (int): The number of integers used during the
                described process.
    
    Returns:
    CustomFraction object representing the rational number equal
    to the mean maximum number of segments that occur at any
    part of the described process over each possible distinct
    permutation of the n_pieces integers.

    Outline of rationale:
    See outline of rationale section of the function
    constructingLinearPuzzleMaxSegmentCountDistribution().
    """
    # Review- look into rewording the documentation for clarity
    denom = 0
    numer = 0
    distr = constructingLinearPuzzleMaxSegmentCountDistribution(n_pieces)
    print(distr)
    for i, num in enumerate(distr):
        numer += i * num
        denom += num
    return CustomFraction(numer, denom)

def constructingLinearPuzzleMaxSegmentCountMeanFloat(
    n_pieces: int=40,
) -> float:
    """
    Solution to Project Euler #253

    For a permutation of the first n_pieces integers, the
    numbers are selected in the order they appear in the
    permutation and placed in another list of length n_pieces
    in the position corresponding to the integer until the
    list is filled. After any given step in this process,
    the number of segments in the list is the number of
    contiguous blocks of filled entries that are in the
    list (where a contiguous block of filled entries is
    a number of consecutive entries in the list, all of
    which are filled, where the first element is either the
    first entry of the list or is preceded by an unfilled
    entry and the last element is either the last entry of
    the list or is succeeded by an unfilled entry). This
    function calculates the mean of the maximum number of
    segments that occur at any part of the process over
    each possible distinct permutation of the n_pieces integers.

    Args:
        Required positional:
        n_pieces (int): The number of integers used during the
                described process.
    
    Returns:
    Float representing the real number equal to the mean
    maximum number of segments that occur at any part of the
    described process over each possible distinct permutation
    of the n_pieces integers.

    Outline of rationale:
    See outline of rationale section of the function
    constructingLinearPuzzleMaxSegmentCountDistribution().
    """
    frac = constructingLinearPuzzleMaxSegmentCountMeanFraction(n_pieces)
    print(frac)
    return frac.numerator / frac.denominator

# Problem 254
def calculateSmallestNumberDigitFrequenciesWithSumOfFactorialSumDigitsEqualToN(
    n: int,
    base: int=10,
) -> List[int]:
    """
    Calculates the frequencies of the digits of the representation
    of the number in the chosen base that is the smallest strictly
    positive integer for which the sum of the factorial over each
    of its digits when represented in that base is equal to n.

    Args:
        Required positional:
        n (int): Strictly positive integer giving the number to which
                the number with the digit frequencies returned
                should have the factorials of its digits sum
                (when represented in the chosen base).

        Optional named:
        base (int): Integer strictly greater than 1 giving the
                base in which the numbers should be represented
                when assessing their digits.
            Default: 10

    Returns:
    List of integers (int) with length base, where the 0-indexed
    i:th element is equal to the frequency of the digit i in the
    representation in the chosen base of the smallest strictly
    positive integer for which the sum of the factorial over each
    of its digits when represented in that base is equal to n.

    Outline of rationale:
    TODO
    """
    
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

def calculateSmallestNumberDigitFrequenciesWithTheFirstNSumOfDigitFactorials(
    n_max: int,
    base: int=10,
) -> List[int]:
    """
    Calculates the total frequencies of the digits of the
    representations of the numbers in the chosen base that are
    the smallest strictly positive integers for which the sum
    of the factorial over each of their digits when represented
    in that base is equal to the integers between 1 and n_max
    inclusive.

    Args:
        Required positional:
        n_max (int): Strictly positive integer giving the inclusive
                upper bound on the values to which the numbers whose
                digit frequencies are included in the returned total
                frequencies should have the factorials of its digits
                sum (when represented in the chosen base).

        Optional named:
        base (int): Integer strictly greater than 1 giving the
                base in which the numbers should be represented
                when assessing their digits.
            Default: 10

    Returns:
    List of integers (int) with length base, where the 0-indexed
    i:th element is equal to the total frequency of the digit i in
    the representations in the chosen base of the smallest strictly
    positive integer for which the sum of the factorial over each
    of its digits when represented in that base is equal to each
    of the integers between 1 and n_max inclusive.

    Outline of rationale:
    See outline of rationale section in the documentation of function
    calculateSmallestNumberDigitFrequenciesWithSumOfFactorialSumDigitsEqualToN().
    """
    
    res = [[]]
    for num in range(1, n_max + 1):
        dig_freqs = calculateSmallestNumberDigitFrequenciesWithSumOfFactorialSumDigitsEqualToN(num, base=base)
        print(num, dig_freqs)
        res.append(dig_freqs)
    return res

def calculateSmallestNumberWithTheFirstNSumOfDigitFactorials0(
    n_max: int,
    base: int=10,
) -> List[int]:
    """
    Calculates the total frequencies of the digits of the
    representations of the numbers in the chosen base that are
    the smallest strictly positive integers for which the sum
    of the factorial over each of their digits when represented
    in that base is equal to the integers between 1 and n_max
    inclusive.

    Args:
        Required positional:
        n_max (int): Strictly positive integer giving the inclusive
                upper bound on the values to which the numbers whose
                digit frequencies are included in the returned total
                frequencies should have the factorials of its digits
                sum (when represented in the chosen base).

        Optional named:
        base (int): Integer strictly greater than 1 giving the
                base in which the numbers should be represented
                when assessing their digits.
            Default: 10

    Returns:
    List of integers (int) with length base, where the 0-indexed
    i:th element is equal to the total frequency of the digit i in
    the representations in the chosen base of the smallest strictly
    positive integer for which the sum of the factorial over each
    of its digits when represented in that base is equal to each
    of the integers between 1 and n_max inclusive.

    Outline of rationale:
    See outline of rationale section in the documentation of function
    calculateSmallestNumberDigitFrequenciesWithSumOfFactorialSumDigitsEqualToN().
    """
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

    Calculates the sum of the digits in the representations of
    the numbers in the chosen base that are the smallest strictly
    positive integers for which the sum of the factorial over each
    of their digits when represented in that base is equal to the
    integers between 1 and n_max inclusive.

    Args:
        Required positional:
        n_max (int): Strictly positive integer giving the inclusive
                upper bound on the values to which the numbers whose
                digit sums are included in the returned total sum
                should have the factorials of its digits sum (when
                represented in the chosen base).

        Optional named:
        base (int): Integer strictly greater than 1 giving the
                base in which the numbers should be represented
                when assessing their digits.
            Default: 10

    Returns:
    Integer (int) giving the sum of the digits in the representations
    of the numbers in the chosen base that are the smallest strictly
    positive integers for which the sum of the factorial over each
    of their digits when represented in that base is equal to the
    integers between 1 and n_max inclusive.

    Outline of rationale:
    See outline of rationale section in the documentation of function
    calculateSmallestNumberDigitFrequenciesWithSumOfFactorialSumDigitsEqualToN().
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
def calculateNumberOfIterationsOfHeronsMethodForIntegers(
    n: int,
) -> int:
    """
    Calculates the number of iterations required by an adaptation
    of Heron's method for integer arithmetic to calculate the
    rounded square root of the non-negative integer n.

    The adaptation of Heron's method for integer arithmetic used
    is an iterative procedure whereby for integer k >= 0, the
    term x_(k + 1) is calculated using:
        x_(k + 1) = floor((x_k + ceil(n / x_k)) / 2)
    where floor() is the floor function (equal to the largest
    integer no greater than the input) and ceil() is the ceiling
    function (equal to the smallest integer no less than the input).
    The procedure starts with x_0, where, for d equal to the number
    of digits in the representation of the integer in base 10:
        x_0 = 2 * 10 ** ((d - 1) / 2) if d is odd
        x_0 = 7 * 10 ** ((d - 2) / 2) if d is even
    The iteration stops for the smallest k such that x_(k + 1) = x_k,
    and for this k the number of iterations is (k + 1).
    For this process, it is guaranteed that there exists a
    non-negative integer k for which  x_(k + 1) = x_k, and therefore
    the adaptation of Heron's method for integer arithmetic is
    guaranteed to terminate in a finite number of steps.

    Args:
        n (int): The non-negative integer n for which the number
                of steps in the adaptation of Heron's method for
                integer arithmetic described to find the rounded
                square root.

    Returns:
    Integer (int) giving the number of iterations required by the
    adaptation of Heron's method for integer arithmetic described
    to calculate the rounded square root of the non-negative integer
    n.
    """
    base = 10
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
) -> CustomFraction:
    """
    Calculates the mean number of iterations required by an
    adaptation of Heron's method for integer arithmetic to
    calculate the rounded square root of each of the non-negative
    integers between n_min and n_max inclusive as a fraction.

    The adaptation of Heron's method for integer arithmetic used
    is an iterative procedure whereby for integer k >= 0, the
    term x_(k + 1) is calculated using:
        x_(k + 1) = floor((x_k + ceil(n / x_k)) / 2)
    where floor() is the floor function (equal to the largest
    integer no greater than the input) and ceil() is the ceiling
    function (equal to the smallest integer no less than the input).
    The procedure starts with x_0, where, for d equal to the number
    of digits in the representation of the integer in base 10:
        x_0 = 2 * 10 ** ((d - 1) / 2) if d is odd
        x_0 = 7 * 10 ** ((d - 2) / 2) if d is even
    The iteration stops for the smallest k such that x_(k + 1) = x_k,
    and for this k the number of iterations is (k + 1).
    For this process, it is guaranteed that there exists a
    non-negative integer k for which  x_(k + 1) = x_k, and therefore
    the adaptation of Heron's method for integer arithmetic is
    guaranteed to terminate in a finite number of steps.

    Args:
        n_min (int): The inclusive lower bound on the non-negative
                integers for which the number of steps in the
                adaptation of Heron's method for integer arithmetic
                described to find the rounded square root is to
                be included in the calculation of the mean.
        n_max (int): The inclusive upper bound on the non-negative
                integers for which the number of steps in the
                adaptation of Heron's method for integer arithmetic
                described to find the rounded square root is to
                be included in the calculation of the mean.

    Returns:
    CustomFraction object giving the rational number representing
    the mean number of iterations required by the adaptation of
    Heron's method for integer arithmetic described to calculate
    the rounded square root of the non-negative integers between
    n_min and n_max inclusive.
    """
    base = 10
    def calculateTransition(rng: Tuple[int], lower: int) -> int:

        lft, rgt = rng
        while lft < rgt:
            mid = rgt - ((rgt - lft) >> 1)
            #print(lft, rgt, mid, calculateNumberOfIterationsOfHeronsMethodForIntegers(mid, base=base), lower)
            if calculateNumberOfIterationsOfHeronsMethodForIntegers(mid) > lower:
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
                calculateNumberOfIterationsOfHeronsMethodForIntegers(sub_rng[0][0]),
                calculateNumberOfIterationsOfHeronsMethodForIntegers(sub_rng[0][1]),
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
) -> float:
    """
    Solution to Project Euler #255

    Calculates the mean number of iterations required by an
    adaptation of Heron's method for integer arithmetic to
    calculate the rounded square root of each of the non-negative
    integers between n_min and n_max inclusive as a float.

    The adaptation of Heron's method for integer arithmetic used
    is an iterative procedure whereby for integer k >= 0, the
    term x_(k + 1) is calculated using:
        x_(k + 1) = floor((x_k + ceil(n / x_k)) / 2)
    where floor() is the floor function (equal to the largest
    integer no greater than the input) and ceil() is the ceiling
    function (equal to the smallest integer no less than the input).
    The procedure starts with x_0, where, for d equal to the number
    of digits in the representation of the integer in base 10:
        x_0 = 2 * 10 ** ((d - 1) / 2) if d is odd
        x_0 = 7 * 10 ** ((d - 2) / 2) if d is even
    The iteration stops for the smallest k such that x_(k + 1) = x_k,
    and for this k the number of iterations is (k + 1).
    For this process, it is guaranteed that there exists a
    non-negative integer k for which  x_(k + 1) = x_k, and therefore
    the adaptation of Heron's method for integer arithmetic is
    guaranteed to terminate in a finite number of steps.

    Args:
        n_min (int): The inclusive lower bound on the non-negative
                integers for which the number of steps in the
                adaptation of Heron's method for integer arithmetic
                described to find the rounded square root is to
                be included in the calculation of the mean.
        n_max (int): The inclusive upper bound on the non-negative
                integers for which the number of steps in the
                adaptation of Heron's method for integer arithmetic
                described to find the rounded square root is to
                be included in the calculation of the mean.

    Returns:
    Float object giving the real number representing the mean number
    of iterations required by the adaptation of Heron's method for
    integer arithmetic described to calculate the rounded square root
    of the non-negative integers between n_min and n_max inclusive.
    """
    res = meanNumberOfIterationsOfHeronsMethodForIntegersFraction(n_min, n_max)
    print(res)
    return res.numerator / res.denominator


# Problem 256
def isTatamiFreeBruteForce(
    w: int,
    h: int,
) -> bool:
    """
    Calculates whether a rectangular grid of equally sized squares
    with dimensions w x h is Tatami-free.

    For a rectangular grid of equally sized squares, consider an
    arrangement of 1 x 2 rectangles that collectively completely
    fill the rectangular grid with no overlap between the rectangles.
    This arrangement is Tatami if and only if there is no point
    in the rectangular grid at which corners of four different
    rectangles meet.

    A rectangular grid with given dimensions is Tatami-free if
    and only if there does not exist any such arrangement that
    is Tatami.

    Args:
        Required positional:
        w (int): Strictly positive integer giving the width of the
                rectangular grid in terms of the side lengths of
                the equally sized squares of which it is comprised
        h (int): Strictly positive integer giving the height of the
                rectangular grid in terms of the side lengths of
                the equally sized squares of which it is comprised
    
    Returns:
    Boolean (bool) specifying whether the rectangular grid of
    squares with dimensions w x h is Tatami-free.

    Outline of rationale:
    TODO
    """
    if (w & 1) and (h & 1):
        # All odd area rectangular grids are Tatami-free given
        # that it is not possible to fill such a rectangular grid
        # with 1 x 2 rectangles without overlap.
        return True
    if w > h: w, h = h, w
    #if known_tatami_free is None: known_tatami_free = set()
    known_tatami_free = set()
    def transitionGenerator(
        h_bm: int,
        corner_bm: int,
    ) -> Generator[Tuple[int, int], None, None]:
        
        def backtrack(
            idx: int,
            h_bm2: int,
            corner_bm2: int,
            prev_lvl: int,
        ) -> Generator[Tuple[int, int], None, None]:
            
            if idx == w:
                if prev_lvl: yield (0, 0)
                return
            #print(idx, format(h_bm2, "b"), format(corner_bm2, "b"), prev_lvl)
            #if idx == w - 1:
            #    if not prev_lvl:
            #        if not h_bm2 & 1: yield (1, 0)
            #        return
            #    yield (0, 0)
            #    return

            # Space already filled
            if h_bm2 & 1:
                if not prev_lvl: return
                #print("skipping filled space")
                for (h_bm_head, corner_bm_head) in backtrack(idx + 1, h_bm2 >> 1, corner_bm2 >> 1, 1):
                    yield ((h_bm_head << 1), (corner_bm_head << 1) | (prev_lvl == 1))
                return
            # Adding horizontal
            if prev_lvl:
                #print("leaving space for horizontal")
                for (h_bm_head, corner_bm_head) in backtrack(idx + 1, h_bm2 >> 1, corner_bm2 >> 1, 0):
                    yield ((h_bm_head << 1), (corner_bm_head << 1) | (prev_lvl == 1))
            if corner_bm2 & 1: return
            if not prev_lvl:
                #print(f"adding horizontal to this and previous space")
                for (h_bm_head, corner_bm_head) in backtrack(idx + 1, h_bm2 >> 1, corner_bm2 >> 1, 1):
                    yield ((h_bm_head << 1), (corner_bm_head << 1))
                return

            # Adding vertical
            for (h_bm_head, corner_bm_head) in backtrack(idx + 1, h_bm2 >> 1, corner_bm2 >> 1, 2):
                #print("adding vertical")
                yield ((h_bm_head << 1) | 1, (corner_bm_head << 1) | (prev_lvl == 2))
            
            return

        if (h_bm & 1):
            for (h_bm_head, corner_bm_head) in backtrack(1, h_bm >> 1, corner_bm >> 1, 1):
                yield (h_bm_head << 1, corner_bm_head)
            return

        # Adding horizontal
        for (h_bm_head, corner_bm_head) in backtrack(1, h_bm >> 1, corner_bm >> 1, 0):
            yield (h_bm_head << 1, corner_bm_head)
        # Adding vertical
        if not corner_bm & 1:
            for (h_bm_head, corner_bm_head) in backtrack(1, h_bm >> 1, corner_bm >> 1, 2):
                yield ((h_bm_head << 1) | 1, corner_bm_head)
        return

    # True siginifies that a Tatami room has been found
    #seen = set()
    def recur(h_remain: int, h_bm: int, corner_bm: int) -> bool:
        args = (w, h_remain, h_bm, corner_bm)
        if args in known_tatami_free: return False
        #print(f"h_remain = {h_remain}")
        if h_remain == 1:
            prev_filled = True
            for _ in range(w - 1):
                if h_bm & 1:
                    if not prev_filled:
                        known_tatami_free.add(args)
                        return False
                    prev_filled = True
                elif prev_filled:
                    prev_filled = False
                else:
                    if corner_bm & 1:
                        known_tatami_free.add(args)
                        return False
                    prev_filled = True
                h_bm >>= 1
                corner_bm >>= 1
            return True

        for (h_bm2, corner_bm2) in transitionGenerator(h_bm, corner_bm):
            if recur(h_remain - 1, h_bm2, corner_bm2):
                #print(h_remain - 1, format(h_bm2, "b"), format(corner_bm2, "b"))
                return True
        known_tatami_free.add(args)
        return False

    res = recur(h, 0, 0)
    #print(seen)
    return not res

def isTatamiFree(
    w: int,
    h: int,
) -> int:
    """
    Calculates whether a rectangular grid of equally sized squares
    with dimensions w x h is Tatami-free.

    For a rectangular grid of equally sized squares, consider an
    arrangement of 1 x 2 rectangles that collectively completely
    fill the rectangular grid with no overlap between the rectangles.
    This arrangement is Tatami if and only if there is no point
    in the rectangular grid at which corners of four different
    rectangles meet.

    A rectangular grid with given dimensions is Tatami-free if
    and only if there does not exist any such arrangement that
    is Tatami.

    Args:
        Required positional:
        w (int): Strictly positive integer giving the width of the
                rectangular grid in terms of the side lengths of
                the equally sized squares of which it is comprised
        h (int): Strictly positive integer giving the height of the
                rectangular grid in terms of the side lengths of
                the equally sized squares of which it is comprised
    
    Returns:
    Boolean (bool) specifying whether the rectangular grid of
    squares with dimensions w x h is Tatami-free.

    Outline of rationale:
    TODO
    """
    if (w & 1) and (h & 1):
        # All odd area rectangular grids are Tatami-free given
        # that it is not possible to fill such a rectangular grid
        # with 1 x 2 rectangles without overlap.
        return True
    if w > h: w, h = h, w
    m = (h - 2) // (w + 1)
    if m < 1: return False
    return h <= (m + 1) * (w - 1) - 2

def integersWithAtLeastNFactorsPrimeFactorisationsGenerator(
    n_factors: int,
) -> Generator[Tuple[int, Dict[int, int]], None, None]:
    """
    Generator iterating over the strictly positive integers with
    at least n_factors distinct factors in strictly increasing
    order and their prime factorisations.

    Args:
        Required positional:
        n_factors (int): Non-negative integer giving the inclusive
                lower bound on the number of distinct factors the
                values yielded must have.

    Yields:
    2-tuple whose index 0 contains a strictly positive integer
    with at least n_factors distinct factors, and whose index
    1 contains a dictionary representing the prime factorisation
    of the integer at index 0, whose keys are integers giving a
    prime that divides the integer with corresponding value being
    the power of that prime in the prime factorisation of the
    integer. These are yielded in increasing order of size of the
    integer.
                
    Outline of rationale:
    TODO
    """
    # Review- try to make faster.
    ps = SimplePrimeSieve()

    def numbersWithExactlyNPrimeFactorsWithLargestPrimeGenerator(
        largest_prime_idx: int,
        n_prime_factors: int,
    ) -> Generator[Tuple[int, Dict[int, int]], None, None]:
        p_lst = []
        for p in ps.endlessPrimeGenerator():
            p_lst.append(p)
            if len(p_lst) == largest_prime_idx + 1: break
        #f_lst = [0] * m_primes
        
        def recur(idx: int, p_remain: int, min_n_facts_reqd: int, f_lst: List[int]) -> Generator[Tuple[int, Dict[int, int]], None, None]:
            #print(f"recur() with idx = {idx}, p_remain = {p_remain}, min_n_facts_reqd = {min_n_facts_reqd}, f_lst = {f_lst}")
            if not idx:
                if min_n_facts_reqd > p_remain + 1:
                    return
                num = 1#p_lst[0] ** p_remain
                f_lst[0] = p_remain
                num_facts = {}
                for i in range(largest_prime_idx + 1):
                    if not f_lst[i]: continue
                    num *= p_lst[i] ** f_lst[i]
                    num_facts[p_lst[i]] = f_lst[i]
                yield (num, num_facts)
                return
            h1 = []
            #f_mx = min(p_remain, )
            f_mn, f_mx = None, None
            f_mn = 0
            f_mx = p_remain
            if idx == largest_prime_idx: f_mn = max(f_mn, 1)
            for f in range(f_mn, f_mx + 1):
                min_n_facts_reqd2 = ((min_n_facts_reqd - 1) // (f + 1)) + 1
                q, r = divmod(p_remain, idx)
                if q ** (idx - r) * (q + 1) ** r < min_n_facts_reqd2: continue
                f_lst[idx] = f
                it = iter(recur(idx - 1, p_remain - 1, min_n_facts_reqd2, list(f_lst)))
                try:
                    ans = next(it)
                    heapq.heappush(h1, (ans, it))
                except StopIteration:
                    pass
            f_lst[idx] = 0
            while h1:
                ans, it = h1[0]
                yield ans
                try:
                    ans = next(it)
                    heapq.heappushpop(h1, (ans, it))
                except StopIteration:
                    heapq.heappop(h1)
            return

        yield from recur(largest_prime_idx, n_prime_factors, n_factors, [0] * (largest_prime_idx + 1))
        return

    def numbersWithLargestPrimeFactor(
        largest_prime_idx: int,
    ) -> Generator[Tuple[int, Dict[int, int]], None, None]:
        
        def getNextNumberWithExactlyNPrimeFactorsWithLargestPrimeGenerator(
            n_prime_factors: int,
        ) -> Tuple[int, Tuple[int, Dict[int, int]], Iterable[Tuple[int, Dict[int, int]]]]:
            for n_p_facts in itertools.count(n_prime_factors):
                it = iter(numbersWithExactlyNPrimeFactorsWithLargestPrimeGenerator(largest_prime_idx, n_p_facts))
                try:
                    ans = next(it)
                except StopIteration:
                    continue
                break
            return (n_p_facts, ans, it)

        n_p_facts, ans, it = getNextNumberWithExactlyNPrimeFactorsWithLargestPrimeGenerator(1)
        h2 = [(ans, it)]
        nxt = getNextNumberWithExactlyNPrimeFactorsWithLargestPrimeGenerator(n_p_facts + 1)

        while h2 or nxt is not None:
            while not h2 or (nxt is not None and nxt[1] < h2[0][0]):
                idx, ans, it2 = nxt
                heapq.heappush(h2, (ans, it2))
                nxt = getNextNumberWithExactlyNPrimeFactorsWithLargestPrimeGenerator(idx + 1)
            ans, it2 = h2[0]
            yield ans
            try:
                ans = next(it2)
                heapq.heappushpop(h2, (ans, it2))
            except StopIteration:
                heapq.heappop(h2)
            """
            if h2 and (nxt is None or nxt[1] >= h2[0][0]):
                ans, it2 = h2[0]
                yield ans
                try:
                    ans = next(it2)
                    heapq.heappushpop(h2, (ans, it2))
                except StopIteration:
                    heapq.heappop(h2)
                continue

            ans, it2 = (nxt[1], nxt[2])
            yield ans
            try:
                ans = next(it2)
                heapq.heappush(h2, (ans, it2))
            except StopIteration:
                pass
            idx = nxt[0] + 1
            nxt = getNextNumberWithExactlyNPrimeFactorsWithLargestPrimeGenerator(idx)
            """
        return
    
    def getNextLargestPrimeFactorGenerator(
        largest_prime_idx: int,
    ) -> Tuple[int, Tuple[int, Dict[int, int]], Iterable[Tuple[int, Dict[int, int]]]]:
        for l_p_idx in itertools.count(largest_prime_idx):
            it = iter(numbersWithLargestPrimeFactor(l_p_idx))
            try:
                ans = next(it)
            except StopIteration:
                continue
            break
        return (l_p_idx, ans, it)

    idx, ans, it = getNextLargestPrimeFactorGenerator(0)
    h0 = [(ans, it)]
    nxt = getNextLargestPrimeFactorGenerator(idx + 1)
    #print(h0, nxt)

    while h0 or nxt is not None:
        
        while not h0 or (nxt is not None and nxt[1] < h0[0][0]):
            #print(h0[0], nxt)
            idx, ans, it2 = nxt
            heapq.heappush(h0, (ans, it2))
            nxt = getNextLargestPrimeFactorGenerator(idx + 1)
        ans, it2 = h0[0]
        yield ans
        try:
            ans = next(it2)
            heapq.heappushpop(h0, (ans, it2))
        except StopIteration:
            heapq.heappop(h0)

        """
        if h0 and (nxt is None or nxt[1] >= h0[0][0]):
            ans, it2 = h0[0]
            yield ans
            try:
                ans = next(it2)
                heapq.heappushpop(h0, (ans, it2))
            except StopIteration:
                heapq.heappop(h0)
            continue

        ans, it2 = (nxt[1], nxt[2])
        yield ans
        try:
            ans = next(it2)
            heapq.heappush(h0, (ans, it2))
        except StopIteration:
            pass
        idx = nxt[0] + 1
        nxt = getNextLargestPrimeFactorGenerator(idx)
        """
        
    return

def smallestRoomSizeWithExactlyNTatamiFreeConfigurations(
    n_config: int=200,
) -> int:
    """
    Solution to Project Euler #256

    Calculates the smallest strictly positive integer for which
    there exists exactly n_config distinct rectangular grids of
    equally sized squares containing that number of squares that
    are Tatami-free and for which the width is no greater than
    its height.

    For a rectangular grid of equally sized squares, consider an
    arrangement of 1 x 2 rectangles that collectively completely
    fill the rectangular grid with no overlap between the rectangles.
    This arrangement is Tatami if and only if there is no point
    in the rectangular grid at which corners of four different
    rectangles meet.

    A rectangular grid with given dimensions is Tatami-free if
    and only if there does not exist any such arrangement that
    is Tatami.

    Args:
        Optional named:
        n_config (int): Non-negative integer giving the exact
                number of Tatami-free rectangular grids whose width
                is no greater than its height that should exist
                containing the number of squares returned.
            Default: 200
        
    Returns:
    Integer (int) giving the smallest strictly positive integer for
    which there exists exactly n_config distinct rectangular grids of
    equally sized squares containing that number of squares that
    are Tatami-free and for which the width is no greater than
    its height.

    Outline of rationale:
    TODO
    """
    ps = PrimeSPFsieve()
    #known_non_tatami = set()
    # note that no perfect squares are Tatami-free
    mn_factor_cnt = (n_config << 1)
    for sz in itertools.count(2, step=2):
        #print(sz)
        if (not sz % 10 ** 5): print(f"size = {sz}")
        fc = ps.factorCount(sz) 
        if fc < mn_factor_cnt: continue
        
        # For rooms of width less than 5 and height such that
        # both height and width are not both odd there is always
        # a Tatami-free configuration
        #for f in range(7, rt + 1):
        #    f2, r = divmod(sz, f)
        #    if not r: fact_pairs.append((f, f2))
        #if len(fact_pairs) < n_config: continue
        
        #fact_pairs = []
        rt = isqrt(sz)
        remain = fc >> 1
        cnt = 0
        for w in sorted(ps.factors(sz)):
            if w > rt: break
            h = sz // w
            remain -= 1
            if isTatamiFree(w, h):
                cnt += 1
                if cnt > n_config: break
            elif cnt + remain < n_config: break
            """
            rng = (w + 1, ((w ** 2 - 4 * w - 1) >> 1) if w & 1 else ((w * (w - 5)) >> 1))
            if rng[0] > rng[1]: continue
            h = sz // w
            if h < rng[0] or h > rng[1]: continue
            fact_pairs.append((w, h))
            """
            #if w < 7: continue
            #fact_pairs.append((w, sz // w))
        if cnt == n_config: break
        """
        if len(fact_pairs) < n_config: continue
        print(sz, fact_pairs)
        #mx = isqrt(sz)
        cnt = 0
        n_fact_pairs = len(fact_pairs)
        for idx in range(n_fact_pairs):
            w, h = fact_pairs[idx]
            #print(w, h)
            #if w > mx: break
            h = sz // w
            if isTatamiFree(w, h):
                cnt += 1
                
                if cnt > n_config: break
            elif cnt + (n_fact_pairs - idx - 1) < n_config:
                break
            print((w, h), cnt)
        print(cnt)
        if cnt == n_config: break
        """
    return sz


# Problem 257
def angularBisectorTrianglePartitionIntegerRatioCountBruteForce(
    perimeter_max: int,
) -> int:
    """
    For triangles ABC with integer side lengths BC <= AC <= AB and
    the points where the angular bisectors intersect with the opposite
    edge E (for the bisector of angle C), F (for the bisector of angle
    A) and G (for the bisector of angle B), calculates the number of
    such triangles ABC whose perimeter is no greater than perimeter_max
    and whose area is an integer multiple of the area of the triangle
    AEG.

    Args:
        Required positional:
        perimeter_max (int): Integer giving the inclusive upper bound
                on the perimeters of the triangles ABC considered for
                inclusion in the count.

    Returns:
    Integer (int) giving the number of triangles ABC as described whose
    perimeter is no greater than perimeter_max and whose area is an
    integer multiple of the area of the triangle AEG.

    Outline of rationale:
    TODO
    """
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

def angularBisectorTrianglePartitionIntegerRatioCount(
    perimeter_max: int=10 ** 8,
) -> int:
    """
    Solution to Project Euler #257

    For triangles ABC with integer side lengths BC <= AC <= AB and
    the points where the angular bisectors intersect with the opposite
    edge E (for the bisector of angle C), F (for the bisector of angle
    A) and G (for the bisector of angle B), calculates the number of
    such triangles ABC whose perimeter is no greater than perimeter_max
    and whose area is an integer multiple of the area of the triangle
    AEG.

    Args:
        Optional named:
        perimeter_max (int): Integer giving the inclusive upper bound
                on the perimeters of the triangles ABC considered for
                inclusion in the count.
            Default: 10 ** 8

    Returns:
    Integer (int) giving the number of triangles ABC as described whose
    perimeter is no greater than perimeter_max and whose area is an
    integer multiple of the area of the triangle AEG.

    Outline of rationale:
    TODO
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
def calculateGeneralisedLaggedFibonacciTerm(
    term_idx: int=10 ** 18,
    initial_terms: List[int]=[1] * 2000,
    prev_terms_to_sum: List[int]=[1999, 2000],
    res_md: Optional[int]=20092010,
) -> int:
    """
    Solution to Project Euler #258

    Calculates value of term term_idx of the generalisation of
    the lagged Fibonacci sequence with initial terms initial_terms
    and previous terms to sum prev_terms_to_sum. If res_md
    is given as a strictly positive integer, this value is
    given modulo res_md.
    
    The generalisation of the lagged Fibonacci sequence for
    given initial terms in the list initial_terms and given
    previous terms to sum in the list prev_terms_to_sum is the
    sequence such that for integer i >= 0, the i:th term in
    the sequence is:
        t_i = initial_terms[i] if i < len(initial_terms)
              ((sum j from 0 to len(prev_terms_to_sum) - 1) (t_(i - prev_terms_to_sum[i])))
                otherwise

    Args:
        Optional named:
        term_idx (int): The term of the generalisation of the
                lagged Fibonacci sequence to be returned, given
                modulo res_md if res_md is a strictly positive
                integer.
            Default: 10 ** 18,
        initial_terms (list of ints): List of integers giving the
                initial terms of the generalisation of the lagged
                Fibonacci sequence, with t_i = initial_terms[i]
                for integers 0 <= i < len(initial_terms). The
                length of this sequence should be no less than the
                largest value in prev_terms_to_sum.
            Default: [1] * 2000 (a list of length 2000 of 1s)
        prev_terms_to_sum (list of ints): List of strictly positive
                integers specifying, for the terms in the sequence
                after the initial terms, how many terms back in the
                sequence from which each the terms summed to produce
                the current term should be taken relative to the
                current term's position in the sequence. Each of the
                integers contained should not exceed the length of
                initial_terms.
            Default: [1999, 2000]
        res_md (int or None): If given as a strictly positive integer,
                the modulus to which the value of term term_idx should
                be taken when returned, otherwise the term value itself
                is returned.
            Default: 20092010
        
    Returns:
    Integer (int) giving the value of term term_idx of the generalisation
    of the lagged Fibonacci sequence described, with initial terms
    initial_terms and previous terms to sum prev_terms_to_sum. If res_md
    is given as a strictly positive integer, this value is given modulo
    res_md, otherwise the value of the term itself is returned.

    Outline of rationale:
    TODO
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
def calculateReachableIntegers(
    dig_min: int=1,
    dig_max: int=9,
    base: int=10,
) -> List[int]:
    """
    Calculates all reachable integers for the chosen base with
    minimum and maximum digit dig_min and dig_max.

    A reachable integer for a given base with minimum and maximum digit
    is a strictly positive integer that is the result of an arithmetic
    expression containing only the digits between dig_min and dig_max
    exactly once each in increasing order of size from left to right,
    which can be combined using concatenation in the chosen base,
    addition, subtraction, multiplication and division, without unary
    minus and with order of operations (other than concatenation, which
    is always performed first) specified freely using brackets.

    Args:
        Optional named:
        dig_min (int): Non-negative integer strictly less than base giving
                the smallest and leftmost digit used in any arithmetic
                expression equal to one of the returned integers.
            Default: 1
        dig_max (int): Non-negative integer no less than dig_max and
                strictly less than base giving the largest and rightmost
                digit used in any arithmetic expression equal to one of
                the returned integers.
            Default: 9
        base (int): Integer strictly greater than 1 giving the base in
                which concatenation of digits is to be performed.
            Default: 10
    
    Returns:
    List of integers (int) giving all reachable integers for the chosen base
    with minimum and maximum digit dig_min and dig_max in strictly increasing
    order.

    Outline of rationale:
    TODO
    """
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

    Calculates the sum of all reachable integers for the chosen base with
    minimum and maximum digit dig_min and dig_max.

    A reachable integer for a given base with minimum and maximum digit
    is a strictly positive integer that is the result of an arithmetic
    expression containing only the digits between dig_min and dig_max
    exactly once each in increasing order of size from left to right,
    which can be combined using concatenation in the chosen base,
    addition, subtraction, multiplication and division, without unary
    minus and with order of operations (other than concatenation, which
    is always performed first) specified freely using brackets.

    Args:
        Optional named:
        dig_min (int): Non-negative integer strictly less than base giving
                the smallest and leftmost digit used in any arithmetic
                expression equal to one of the returned integers.
            Default: 1
        dig_max (int): Non-negative integer no less than dig_max and
                strictly less than base giving the largest and rightmost
                digit used in any arithmetic expression equal to one of
                the returned integers.
            Default: 9
        base (int): Integer strictly greater than 1 giving the base in
                which concatenation of digits is to be performed.
            Default: 10
    
    Returns:
    List of integers (int) giving the sum of all reachable integers for the
    chosen base with minimum and maximum digit dig_min and dig_max.

    Outline of rationale:
    See outline of rationale section in the documentation of the function
    calculateReachableIntegers().
    """
    res = sum(calculateReachableIntegers(dig_min=dig_min, dig_max=dig_max, base=base))
    return res

# Problem 260
def stoneGamePlayerTwoWinningConfigurationsGenerator(
    n_piles: int,
    pile_size_max: int,
) -> Generator[List[int], None, None]:
    """
    Generator iterating over the initial configurations of the
    following two player stone game with n_piles piles of stones,
    each with maximum initial size of pile_size_max for which the
    player whose turn is second (referred to as player 2) can
    guarantee a win with perfect play.

    The stone game starts with a given number of (possibly empty)
    piles of stones, where the number of stones in each pile at this
    stage in non-decreasing order is referred to as the initial
    configuration.
    The two players alternately take turns, with player 1 taking the
    first turn and player 2 taking the second turn. On each turn,
    a player selects a strictly positive integer N and a non-empty
    subset of the piles of stones, each of which must contain no less
    than N stones. A player loses the game if they are unable to make
    a valid move (i.e. if there are no non-empty piles remaining at
    the start of their turn) with the other player consequently winning.

    Args:
        Required positional:
        n_piles (int): Non-negative integer giving the number of
                stone piles in the stone game described.
        pile_size_max (int): Non-negative integer giving the maximum
                number of stones in each pile for the initial
                configurations considered.

    Yields:
    List of integers (int) with length n_piles giving an initial
    configuration for which player 2 can with perfect play guarantee
    a win as a list of the initial pile sizes in non-decreasing order,
    with the generator yielding all such initial configurations for
    which no pile contains more than pile_size_max stones.
    These configurations are yielded in no particular order.

    Outline of rationale:
    TODO
    """
    
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
    def recur(idx: int) -> Generator[List[int], None, None]:
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
                        yield list(curr)
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
                        yield list(curr)
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

def stoneGamePlayerTwoWinningConfigurationsGenerator2(
    pile_size_max: int,
) -> Generator[Tuple[int, int, int], None, None]:
    """
    Generator iterating over the initial configurations of the
    following two player stone game with three piles of stones,
    each with maximum initial size of pile_size_max for which the
    player whose turn is second (referred to as player 2) can
    guarantee a win with perfect play.

    The stone game starts with three (possibly empty) piles of stones,
    where the number of stones in each pile at this stage in
    non-decreasing order is referred to as the initial configuration.
    The two players alternately take turns, with player 1 taking the
    first turn and player 2 taking the second turn. On each turn,
    a player selects a strictly positive integer N and a non-empty
    subset of the piles of stones, each of which must contain no less
    than N stones. A player loses the game if they are unable to make
    a valid move (i.e. if there are no non-empty piles remaining at
    the start of their turn) with the other player consequently winning.

    Args:
        Required positional:
        pile_size_max (int): Non-negative integer giving the maximum
                number of stones in each pile for the initial
                configurations considered.

    Yields:
    3-tuple of integers (int) giving an initial configuration for
    which player 2 can with perfect play guarantee a win as a list
    of the initial pile sizes in non-decreasing order, with the
    generator yielding all such initial configurations for
    which no pile contains more than pile_size_max stones.
    These configurations are yielded in no particular order.

    Outline of rationale:
    TODO
    """
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

def stoneGamePlayerTwoWinningConfigurationsSum(
    pile_size_max: int=1000,
) -> int:
    """
    Calculates the total number of stones over all of the distinct
    initial configurations of the following two player stone game
    with three piles of stones, each with maximum initial size of
    pile_size_max for which the player whose turn is second (referred
    to as player 2) can guarantee a win with perfect play.

    The stone game starts with three (possibly empty) piles of stones,
    where the number of stones in each pile at this stage in
    non-decreasing order is referred to as the initial configuration.
    The two players alternately take turns, with player 1 taking the
    first turn and player 2 taking the second turn. On each turn,
    a player selects a strictly positive integer N and a non-empty
    subset of the piles of stones, each of which must contain no less
    than N stones. A player loses the game if they are unable to make
    a valid move (i.e. if there are no non-empty piles remaining at
    the start of their turn) with the other player consequently winning.

    Args:
        Required positional:
        pile_size_max (int): Non-negative integer giving the maximum
                number of stones in each pile for the initial
                configurations considered.

    Yields:
    Integer (int) giving the total number of stones over all of the
    distinct initial configurations of the stone game described for
    which player 2 can with perfect play guarantee a win over all
    such initial configurations for which no pile contains more than
    pile_size_max stones.

    Outline of rationale:
    See outline of rationale section in the documentation for function
    stoneGamePlayerTwoWinningConfigurationsGenerator2().
    """
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

# Problem 261
def distinctSquarePivots(k_max: int) -> List[int]:
    """
    Finds all the strictly positive integers no greater than
    k_max that are square-pivots.

    A strictly positive integer k is a square pivot if and only
    if there exists an ordered pair of integer (m, n) for which
    m is strictly positive and n is no less than k such that
    the sum of the squares of the (m + 1) consecutive integers
    between (k - m) and k inclusive is equal to the sum of the
    squares of the m consecutive integers between (n + 1) and
    (n + m) inclusive.

    Args:
        Required positional:
        k_max (int): Integer giving the inclusive upper bound
                on the value of integers that are square-pivots
                to be included in the result.
    
    Returns:
    List of integers (int) giving every distinct integer no greater
    than k_max that is a square-pivot in strictly increasing order.
    """
    res = set()
    m_max = (isqrt(1 + 2 * k_max) - 1) >> 1
    ps = PrimeSPFsieve(m_max + 1)
    k2_max = k_max << 1
    for m in range(1, m_max + 1):
        pf1 = ps.primeFactorisation(m)
        a, s = 1, 1
        for p, f in pf1.items():
            if f & 1: a *= p
            s *= p ** (f >> 1)
        pf2 = ps.primeFactorisation(m + 1)
        b, r = 1, 1
        for p, f in pf2.items():
            if f & 1: b *= p
            r *= p ** (f >> 1)
        for alpha, gamma in pellSolutionGenerator(a * b, negative=False):
            beta = gamma * a * s * b * r
            k2 = m * (alpha + 1) + beta
            if k2 > k2_max: break
            if k2 & 1: continue
            n2 = (m + 1) * (alpha - 1) + beta
            if n2 & 1: continue
            k = k2 >> 1
            n = n2 >> 1
            if n < k: continue
            #print(f"solution with m = {m}, k = {k}, n = {n}")
            res.add(k)
    return sorted(res)
    """
    res = set()
    m_max = (isqrt(1 + 2 * k_max) - 1) >> 1

    def isValidSolution(k: CustomFraction, m: int) -> bool:
        
        if k.denominator != 1 or k < m: return False
        k = k.numerator
        rhs = (m + 1) * (m + (2 * k - m) ** 2)
        q, r = divmod(rhs, m)
        
        if r: return False
        
        q_rt = isqrt(q)
        if q_rt * q_rt != q: return False
        #print(k, m, q_rt, m)
        if q_rt & 1 == m & 1: return False
        n = (q_rt - m - 1) >> 1
        #print(f"n = {n}")
        if n < k: return False
        #print(f"solution m = {m}, k = {k}, n = {n} is valid")
        return True

    def kGenerator(m: int) -> Generator[int, None, None]:
        add = CustomFraction(m, 2)
        
        curr1 = (CustomFraction(m, 1), CustomFraction(m, 1))
        curr2 = (CustomFraction(m, 1), -CustomFraction(m, 1))
        k = (curr1[0] + curr2[0]) / 4 + add
        print(f"k = {k}")
        if k > k_max: return
        if isValidSolution(k, m):
            yield k.numerator
        q = CustomFraction(m + 1, m)
        mult1 = (m * (q + 1), CustomFraction(2 * m, 1))
        mult2 = (m * (q + 1), -CustomFraction(2 * m, 1))
        while True:
            curr1 = (mult1[0] * curr1[0] + q * mult1[1] * curr1[1], mult1[0] * curr1[1] + mult1[1] * curr1[0])
            curr2 = (mult2[0] * curr2[0] + q * mult2[1] * curr2[1], mult2[0] * curr2[1] + mult2[1] * curr2[0])
            #print(curr1, curr2)
            k = (curr1[0] + curr2[0]) / 4 + add
            print(f"k = {k}")
            if k > k_max: return
            if isValidSolution(k, m):
                yield k.numerator
        return

    cnt = 0
    cnt2 = 0
    for m in range(m_max + 1):
        print(f"m = {m}")
        for k in kGenerator(m):
            print(f"m = {m}, k = {k}")
            cnt += 1
            cnt2 += k not in res
            if k in res: print(f"repeated pivot {k}")
            res.add(k)
    print(f"total count = {cnt}, unique count = {cnt2}")
    return sorted(res)
    """
    """
    res = set()
    m_max = (isqrt(1 + 2 * k_max) - 1) >> 1

    for sol in generalisedPellSolutionGenerator(2, -1):
        
        m = 1
        mx = 2 * k_max - 1
        k2, n2 = sol
        if k2 > mx: break
        if not k2 & 1: continue
        k = (k2 + 1) >> 1
        n = n2 - m
        #print(m, k, n)
        if n < k: continue
        print(f"m = {m}, k = {k}, n = {n}")
        res.add(k)
    
    for sol in generalisedPellSolutionGenerator(6, 3):
        m = 2
        mx = k_max - 1
        n2, k2 = sol
        if k2 > mx: break
        k = k2 + 1
        n = ((n2 + 1) >> 1) - m
        if not n2 & 1: continue
        #print(m, k, n)
        if n < k: continue
        print(f"m = {m}, k = {k}, n = {n}")
        res.add(k)
    
    for sol in generalisedPellSolutionGenerator(3, -3):
        m = 3
        mx = 2 * k_max - 3
        #print(m, sol)
        k2, n2 = sol
        if k2 > mx: break
        if not k2 & 1: continue
        k = (k2 + 3) >> 1
        n = (n2 + 1) - m
        #print(m, k, n)
        if n < k: continue
        print(f"m = {m}, k = {k}, n = {n}")
        res.add(k)
    
    for sol in generalisedPellSolutionGenerator(5, 5):
        m = 4
        mx = k_max - 2
        #print(m, sol)
        n2, k2 = sol
        if k2 > mx: break
        if not n2 & 1: continue
        k = k2 + 2
        n = ((n2 + 3) >> 1) - m
        #print(m, k, n)
        if n < k: continue
        print(f"m = {m}, k = {k}, n = {n}")
        res.add(k)
    
    return sorted(res)
    """

def distinctSquarePivotsSum(k_max: int=10 ** 10) -> int:
    """
    Solution to Project Euler #261

    Finds the sum of all the strictly positive integers no greater
    than k_max that are square-pivots.

    A strictly positive integer k is a square pivot if and only
    if there exists an ordered pair of integer (m, n) for which
    m is strictly positive and n is no less than k such that
    the sum of the squares of the (m + 1) consecutive integers
    between (k - m) and k inclusive is equal to the sum of the
    squares of the m consecutive integers between (n + 1) and
    (n + m) inclusive.

    Args:
        Optional named:
        k_max (int): Integer giving the inclusive upper bound
                on the value of integers that are square-pivots
                to be included in the sum.
            Default: 10 ** 10
    
    Returns:
    Integer (int) giving the sum of every distinct integer no greater
    than k_max that is a square-pivot.
    """
    res = sum(distinctSquarePivots(k_max))
    return res

# Problem 262
def mountainRangeDistance(res_eps: float=1e-4) -> float:

    # Review- try to make more general (starting with arbitrary start and end points)
    def h(x: float, y: float):
        return (5000 - 0.005 * (x * x + y * y + x * y) + 12.5 * (x + y) ) * math.exp(-abs(0.000001 * (x * x + y * y) - 0.0015 * (x + y) + 0.7))
    
    def h_y_deriv(x: float, y: float):
        exp_num = 0.000001 * (x * x + y * y) - 0.0015 * (x + y) + 0.7
        exp_num_deriv = 0.000002 * y - 0.0015
        mult = 5000 - 0.005 * (x * x + y * y + x * y) + 12.5 * (x + y)
        mult_deriv = -0.005 * (2 * y + x) + 12.5
        e = math.exp(-exp_num)
        if exp_num >= 0:
            return (mult_deriv - mult * exp_num_deriv) * e
        return (mult_deriv + mult * exp_num_deriv) * e

    def h_plus_const_over_h_y_deriv(x: float, y: float, h0: float=0):
        #print((x, y), h0)
        res0 = (h(x, y) + h0) / h_y_deriv(x, y)
        #print(h(x, y) / h_y_deriv(x, y), h0 / h_y_deriv(x, y))
        #return (h(x, y) + h0) / h_y_deriv(x, y)
        
        exp_num = 0.000001 * (x * x + y * y) - 0.0015 * (x + y) + 0.7
        exp_num_deriv = 0.000002 * y - 0.0015
        mult = 5000 - 0.005 * (x * x + y * y + x * y) + 12.5 * (x + y)
        mult_deriv = -0.005 * (2 * y + x) + 12.5
        #if exp_num >= 0:
        #    res = (mult_deriv / mult - exp_num_deriv)
        #res = (mult_deriv / mult + exp_num_deriv)
        #res = 1 / res
        res = mult / (mult_deriv - mult * exp_num_deriv) if exp_num >= 0 else mult / (mult_deriv + mult * exp_num_deriv)
        #print(f"res0 = {res0}, res without const = {res}")
        if not h0: return res
        e = math.exp(-abs(exp_num))
        #print(mult_deriv, mult, exp_num_deriv, exp_num, e)
        add_term = h0 / ((mult_deriv - mult * exp_num_deriv) * e) if exp_num >= 0 else h0 / ((mult_deriv + mult * exp_num_deriv) * e)
        
        #print(f"res0 = {res0}, res without add term = {res}, add term = {add_term}")
        res += add_term
        return res
        

    def h_y_deriv_over_h_y_deriv2(x: float, y: float):
        exp_num = 0.000001 * (x * x + y * y) - 0.0015 * (x + y) + 0.7
        exp_num_deriv = 0.000002 * y - 0.0015
        exp_num_deriv2 = 0.000002
        mult = 5000 - 0.005 * (x * x + y * y + x * y) + 12.5 * (x + y)
        mult_deriv = -0.005 * (2 * y + x) + 12.5
        mult_deriv2 = -0.01
        if exp_num >= 0:
            #(mult_deriv - mult * exp_num_deriv)
            #(-2 * mult_deriv * exp_num_deriv + mult * exp_num_deriv ** 2 + mult_deriv2 - mult * exp_num_deriv2)
            res = (-2 * mult_deriv * exp_num_deriv + mult * exp_num_deriv ** 2 + mult_deriv2 - mult * exp_num_deriv2) / (mult_deriv - mult * exp_num_deriv)
        else: res = (2 * mult_deriv * exp_num_deriv + mult * exp_num_deriv ** 2 + mult_deriv2 + mult * exp_num_deriv2) / (mult_deriv + mult * exp_num_deriv)
        return 1 / res

    def newtonRaphsonY(f_over_f_y_deriv: Callable[[float, float], float], x: float, y0: float, eps: float=res_eps) -> float:
        y = y0
        y0 = float("inf")
        while 2 * abs(y - y0) > eps:
            #print(f"y = {y}, f_over_f_y_deriv = {f_over_f_y_deriv(x, y)}")
            y0 = y
            y = y - f_over_f_y_deriv(x, y)
        return y
    
    y_h_max_for_zero_x = newtonRaphsonY(h_y_deriv_over_h_y_deriv2, x=0, y0=800, eps=res_eps)
    #print(f"max h for x = 0 is at y = {y_h_max_for_zero_x}")
    f_min = h(0, y_h_max_for_zero_x)

    print((0, y_h_max_for_zero_x), f_min)

    h2_plus_const_over_h2_y_deriv = functools.partial(h_plus_const_over_h_y_deriv, h0=-f_min)
    #print(h2_plus_const_over_h2_y_deriv)
    #print("hi", h2_plus_const_over_h2_y_deriv(0, 0))
    print(h_plus_const_over_h_y_deriv(0, y_h_max_for_zero_x, h0=f_min))
    #print(h0, h1, h2)
    #print(h(100, 100))
    res = 0
    target = [200, 200]
    x_step = res_eps
    x_step_sq = x_step * x_step
    y0 = y_h_max_for_zero_x - 1
    x0 = 0
    x = x0 + x_step
    y = newtonRaphsonY(h2_plus_const_over_h2_y_deriv, x, y0, eps=res_eps)
    y0 = y_h_max_for_zero_x
    while True:
        grad = (y - y0) / x_step
        if math.floor(x) != math.floor(x0): print(x, y, h(x, y), y0 + (target[0] - x0) * grad, res)
        
        if y0 + (target[0] - x0) * grad >= target[1]:
            res += math.sqrt((target[0] - x0) ** 2 + (target[1] - y0) ** 2)
            break
        res += math.sqrt(x_step_sq + (y - y0) ** 2)
        x0 = x
        x += x_step
        y0 = y
        y = newtonRaphsonY(h2_plus_const_over_h2_y_deriv, x, y0, eps=res_eps)
    print(f"branch 1 has length {res}")
    res2 = 0
    target = [1400, 1400]
    x_step = res_eps
    x_step_sq = x_step * x_step
    y0 = y_h_max_for_zero_x + 1
    x0 = 0
    x = x0 + x_step
    y = newtonRaphsonY(h2_plus_const_over_h2_y_deriv, x, y0, eps=res_eps)
    y0 = y_h_max_for_zero_x
    while True:
        grad = (y - y0) / x_step
        if math.floor(x) != math.floor(x0): print(x, y, h(x, y), y0 + (target[0] - x0) * grad, res2)
        
        if y0 + (target[0] - x0) * grad <= target[1]:
            res2 += math.sqrt((target[0] - x0) ** 2 + (target[1] - y0) ** 2)
            break
        res2 += math.sqrt(x_step_sq + (y - y0) ** 2)
        x0 = x
        x += x_step
        y0 = y
        y = newtonRaphsonY(h2_plus_const_over_h2_y_deriv, x, y0, eps=res_eps)
    print(f"branch 2 has length {res2}")

    return res + res2


# Problem 263
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

def calculateFactors(
    num: int,
    ps: Optional[PrimeSPFsieve]=None,
) -> Set[int]:
    """
    Gives the complete set of positive factors of a strictly positive
    integer.
    
    Args:
        Required positional:
        n (int): The strictly positive integer whose factors are to
                be found.
    
    Returns:
    Set of ints, representing the complete set of positive factors
    of n.
    """
    p_fact = calculatePrimeFactorisation(num, ps=ps)
    p_lst = list(p_fact.keys())
    iter_lst = [[p ** x for x in range(p_fact[p] + 1)] for p in p_lst]
    res = set()
    for p_pow_tup in itertools.product(*iter_lst):
        res.add(functools.reduce(lambda x, y: x * y, p_pow_tup))
    return res

def isPractical(num: int, ps: Optional[PrimeSPFsieve]=None) -> bool:

    # Using that a number greater than 1 is practical if and only
    # if it is even and every other prime in its prime factorisation,
    # is no greater than one plus the divisor function of the
    # product of all smaller primes to the power in the prime
    # factorisation of the number
    if num == 1: return True
    if num & 1: return False
    pf = calculatePrimeFactorisation(num, ps=ps)
    sigma = (1 << (pf[2] + 1)) - 1
    p_lst = sorted(pf.keys())
    for p in p_lst[1:]:
        if p > 1 + sigma: return False
        sigma *= (p ** (pf[p] + 1) - 1) // (p - 1)
    return True

def primeGenerator(ps: Optional[Union[SimplePrimeSieve, PrimeSPFsieve]]=None, n_p_mults_ignore: int=7) -> Generator[int, None, None]:
    if ps is not None:
        yield from ps.endlessPrimeGenerator()
        return
    # Consider numbers that are not a multiple of any of
    # the first n_p_mults_ignore primes
    ps2 = PrimeSPFsieve()
    ps3 = SimplePrimeSieve()
    add_lst = [1]
    p_prod = 1
    p_gen = iter(ps2.endlessPrimeGenerator())
    for i, p in enumerate(p_gen):
        yield p
        p_prod *= p
        if i == n_p_mults_ignore - 1: break
    ps2.extendSieve(p_prod - 1)
    for num in range(p + 2, p_prod, 2):
        #if ps2.getSmallestPrimeFactor(num) > init_p_max:
        if ps2.sieve[num][0] <= p: continue
        add_lst.append(num)
        if ps2.sieve[num][0] == num:
            yield num
        #for p in init_p_lst[1:]:
        #    if not num % p: break
        #else: add_lst.append(num)
    #for p in p_gen:
    #    if p >= p_prod: break
    #    add_lst.append(p)
    print(add_lst)
    print(len(add_lst), p_prod)
    for num0 in itertools.count(p_prod, step=p_prod):
        for add in add_lst:
            num = num0 + add
            if ps3.millerRabinPrimalityTestWithKnownBounds(num, max_n_additional_trials_if_above_max=10)[0]:
                #print(f"{num} is prime")
                yield num
    return


def engineersParidiseGenerator(ps: Optional[PrimeSPFsieve]=None) -> Generator[int, None, None]:
    #if ps is None: ps = PrimeSPFsieve()
    p_qu = deque()
    p_set = set()
    cnt = 0
    practical_checked_dict = SortedDict()

    for i, p in enumerate(primeGenerator(ps=None, n_p_mults_ignore=3)):
        if (not i % 10 ** 5): print(f"next prime {p}, triple-pair seen count = {cnt}")
        while p_qu and p_qu[0] < p - 18:
            p_set.remove(p_qu[0])
            p_qu.popleft()
        p_qu.append(p)
        p_set.add(p)
        if len(p_set) != 4 or (p - 6) not in p_set or (p - 12) not in p_set or (p - 18) not in p_set:
            continue
        cnt += 1
        #print(p - 18, p - 12, p - 6, p - 18, cnt)
        n = p - 9
        while practical_checked_dict and practical_checked_dict.peekitem(0)[0] < n - 8:
            practical_checked_dict.popitem(0)
        #print(len(practical_checked_dict))
        for add in (-8, -4, 0, 4, 8):
            num = n + add
            if num in practical_checked_dict.keys():
                if practical_checked_dict[num]: continue
                break
            practical = isPractical(num, ps)
            practical_checked_dict[num] = practical
            if not practical: break
        else: yield n
    #print(f"cnt = {cnt}")
    return

def engineersParadiseSum(n_incl: int=4, ps: Optional[PrimeSPFsieve]=None) -> int:
    """
    Solution to Project Euler #263
    """
    # Review- try to make faster using a more effective strategy
    # for sieving out candidates using the properties of arithmetic
    # progressions of primes and practical numbers modulo different
    # small numbers
    #if ps is None: ps = PrimeSPFsieve()
    res = 0
    if n_incl < 1: return 0
    for i, num in enumerate(engineersParidiseGenerator(ps=ps)):
        print(num)
        res += num
        if i >= n_incl - 1: break
    return res

# Problem 264
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
    

def sumOfTwoSquaresSolutionGenerator(target: int, ps: Optional[PrimeSPFsieve]=None) -> Generator[Tuple[int, int], None, None]:
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
                print(f"repeat seen: {ans}")
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
    return

def trianglesWithLatticePointVerticesAndFixedCircumcentreAndOrthocentreByPerimeterGenerator(
    orthocentre_x: int,
    perimeter_max: Optional[int]=None,
) -> Generator[Tuple[float, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]], None, None]:
    if perimeter_max is None:
        perimeter_max = float("inf")
    x0 = orthocentre_x
    
    # Right-angled triangles
    # These correspond to the integer points (a, b) where b > 0 and
    # a ** 2 + b ** 2 = orthocentre_x ** 2, with the triangle corresponding
    # to such a point (a, b) being the triangle with vertices at (orthocentre_x, 0),
    # (a, b) and (-a, -b).
    print(f"orthocentre_x = {orthocentre_x}")
    #ps = SimplePrimeSieve()
    h = []
    for a_, b_ in sumOfTwoSquaresSolutionGenerator(x0 ** 2, ps=None):
        
        #if not a_ or not b_: continue
        print(a_, b_)
        # Should not be possible for a_ and b_ to be equal as sqrt(2) is irrational
        pair_lst1 = [(a_, b_)] if not a_ else [(a_, b_), (b_, a_)]
        for (a0, b0) in pair_lst1:
            perim = math.sqrt(4 * (a0 ** 2 + b0 ** 2)) + math.sqrt((x0 - a0) ** 2 + b0 ** 2) + math.sqrt((x0 + a0) ** 2 + b0 ** 2)
            if perim > perimeter_max: continue
            pair_lst2 = [(a0, b0), (-a0, b0)] if a0 else [(a0, b0)]
            for (a, b) in pair_lst2:
                heapq.heappush(h, (perim, ((x0, 0), (a, b), (-a, -b))))
    

    # Acute-angled triangles
    xA_maximiser = (x0 + math.sqrt(x0 ** 2 + 4)) / 2
    k_sq_max = math.floor(4 * ((xA_maximiser ** 2 + 1) / ((xA_maximiser - x0) ** 2 + 1))) - 1
    k_max = isqrt(k_sq_max)

    for k in range(1, k_max + 1):
        div = (k ** 2 - 3)
        sub = (div + 4) * x0
        num = 4 * (k ** 2 + 1) * x0
        for a_, b_ in sumOfTwoSquaresSolutionGenerator(num, ps=None):
            for a, b in [(a_, b_), (b_, a_)]:
                xA, r = divmod(a - sub, div)
                if r: continue
                yA, r = divmod(b, div)
                if r: continue
                t_sq = (4 * (xA ** 2 + yA ** 2) // ((xA - x0) ** 2 + yA ** 2)) - 1
                t = isqrt(t_sq)
                if t * t != t_sq: continue
                xB, r = divmod(x0 - xA + t * yA, 2)
                if r: continue
                yB, r = divmod(-yA + t * (x0 - xA), 2)
                if r: continue
                xC, r = divmod(x0 - xA - t * yA, 2)
                if r: continue
                yC, r = divmod(-yA - t * (x0 - xA), 2)
                if r: continue
                perim = math.sqrt((xA - xB) ** 2 + (yA - yB) ** 2) + math.sqrt((xB - xC) ** 2 + (yB - yC) ** 2) + math.sqrt((xC - xA) ** 2 + (yC - yA) ** 2)
                if perim > perimeter_max: continue
                heapq.heappush(h, (perim, ((xA, yA), (xB, yB), (xC, yC))))
                heapq.heappush(h, (perim, ((xA, -yA), (xB, -yB), (xC, -yC))))
    
    while h:
        yield heapq.heappop(h)
    return

def trianglesWithLatticePointVerticesAndFixedCircumcentreAndOrthocentrePerimeterSum(
    orthocentre_x: int=5,
    perimeter_max: int=10 ** 5,
) -> float:
    """
    Solution to Project Euler #264
    """
    res = 0
    for tup in trianglesWithLatticePointVerticesAndFixedCircumcentreAndOrthocentreByPerimeterGenerator(
        orthocentre_x,
        perimeter_max=perimeter_max,
    ):
        res += tup[0]
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

# Problem 266
def calculateLargestFactorNotExceedingValue(
    num_pf: Dict[int, int],
    factor_max: int,
) -> int:
    n_end_primes = 20
    
    p_lst = sorted(num_pf.keys(), reverse=True)
    n_p = len(p_lst)
    f_lst = [num_pf[p] for p in p_lst]
    remain_lst = [1] * n_p
    curr = 1
    for i in reversed(range(n_p)):
        curr *= p_lst[i] ** f_lst[i]
        remain_lst[i] = curr
    
    transition_idx = max(0, n_p - n_end_primes)
    tail_nums = [1]
    for idx in reversed(range(transition_idx, n_p)):
        prev = list(tail_nums)
        p = p_lst[idx]
        f = f_lst[idx]
        mult = 1
        for f in range(1, f_lst[idx] + 1):
            mult *= p
            for num in prev:
                num2 = mult * num
                if num2 > factor_max: break
                elif num2 == factor_max: return factor_max
                tail_nums.append(num2)
        tail_nums.sort()
    print(f"number of tail numbers = {len(tail_nums)}")

    res = [0]
    print(num_pf, factor_max)
    print(p_lst, f_lst, remain_lst)
    
    def recur(idx: int=0, curr: int=1) -> None:
        #print(f"idx = {idx}, curr = {curr}, res = {res}")
        if idx < 5:
            print(f"idx = {idx}, curr = {curr}, current best = {res[0]}, target = {factor_max}")
        mx_poss = curr * remain_lst[idx]
        #print(f"idx = {idx}, curr = {curr}, mx_poss = {mx_poss}")
        if mx_poss <= factor_max:
            res[0] = max(res[0], mx_poss)
            return
        if idx == transition_idx:
            t = factor_max // curr
            j = bisect.bisect_right(tail_nums, t) - 1
            if j < 0: return # should not happen
            res[0] = max(res[0], tail_nums[j] * curr)
            return
        #if idx == n_p - 1:
        #    #print("hi")
        #    for _ in range(1, f_lst[idx] + 1):
        #        nxt = curr * p_lst[idx]
        #        if nxt > factor_max: break
        #        curr = nxt
        #    res[0] = max(res[0], curr)
        #    return
        for f in reversed(range(f_lst[idx] + 1)):
            curr2 = curr * p_lst[idx] ** f
            
            if curr2 > factor_max: continue
            if curr2 * remain_lst[idx + 1] <= res[0]:
                break
            #print(f"f = {f}, curr = {curr}, curr2 = {curr2}")
            recur(idx=idx + 1, curr=curr2)
            if res[0] == factor_max: break
        return
    
    recur(idx=0, curr=1)
    return res[0]

def pseudoSquareRootFromPrimeFactorisation(
    num_pf: Dict[int, int],
    res_md: Optional[int]=None,
) -> int:
    num = 1
    for p, f in num_pf.items():
        num *= p ** f
    target = isqrt(num)
    res = calculateLargestFactorNotExceedingValue(
        num_pf,
        target,
    )
    return res if res_md is None else res % res_md

def pseudoSquareRootOfProductOfInitialPrimes(
    p_max: int=190,
    res_md: Optional[int]=10 ** 16
) -> int:
    num_pf = {}
    ps = SimplePrimeSieve(p_max)
    for p in ps.p_lst:
        if p > p_max: break
        num_pf[p] = 1
    return pseudoSquareRootFromPrimeFactorisation(
        num_pf,
        res_md=res_md,
    )

# Problem 267
def newtonRaphson(x0: float, f: Callable[[float], float], f_derivative: Callable[[float], float], eps: float) -> float:
    x = x0
    while True:
        #print(f"x = {x}")
        x1 = x - f(x) / f_derivative(x)
        if abs(x1 - x) <= eps:
            break
        x = x1
    return x1

def maximiseProbabilityOfGivenProfitInCoinTossGameFraction(
    n_tosses: int=10 ** 3,
    target_multiplier: float=10 ** 9,
    eps: float=1e-13
) -> CustomFraction:

    log_target = math.log(target_multiplier)
    g = lambda f: (1 + 2 * f) * math.log(1 + 2 * f) + 2 * (1 - f) * math.log(1 - f) - 3 * log_target / n_tosses
    g_deriv = lambda f: 2 * (math.log(1 + 2 * f) - math.log(1 - f))
    f = newtonRaphson(0.5, g, g_deriv, eps)
    print(f"f = {f}")

    nw_min = math.ceil(math.log(target_multiplier / (1 - f) ** n_tosses) / math.log((1 + 2 * f) / (1 - f)))
    #print(f"nw_min = {nw_min}")
    res = 0
    term_func = lambda nw: math.comb(n_tosses, nw)#lambda nw: (1 + 2 * f) ** nw * (1 - f) ** (n_tosses - nw) * math.comb(nw, n_tosses)
    if nw_min > (n_tosses >> 1):
        for nw in range(nw_min, n_tosses + 1):
            term = term_func(nw)
            res += term
    else:
        for nw in range(0, nw_min):
            term = term_func(nw)
            #print(nw, n_tosses, term)
            res += term
        res = (1 << n_tosses) - res
    return CustomFraction(res, 1 << n_tosses)

def maximiseProbabilityOfGivenProfitInCoinTossGameFloat(
    n_tosses: int=10 ** 3,
    target_multiplier: float=10 ** 9,
    eps: float=1e-13
) -> float:
    """
    Solution to Project Euler #267
    """
    res = maximiseProbabilityOfGivenProfitInCoinTossGameFraction(n_tosses=n_tosses, target_multiplier=target_multiplier, eps=eps)
    return res.numerator / res.denominator

# Problem 268
def numbersDivisibleByAtLeastNOfInitialPrimesCountBruteForce(
    num_max: int,
    p_max: int,
    min_n_p_divide: int,
) -> int:
    ps = PrimeSPFsieve(num_max)
    res = 0
    for num in range(1, num_max + 1):
        facts = ps.primeFactors(num)
        n_fact = 0
        for fact in facts:
            n_fact += (fact <= p_max)
        res += (n_fact >= min_n_p_divide)
    return res

def numbersDivisibleByAtLeastNOfInitialPrimesCount(
    num_max: int=10 ** 16 - 1,
    p_max: int=99,
    min_n_p_divide: int=4,
) -> int:
    
    # TODO- justify the factor of ((n_p_incl - 1) choose (min_n_p_divide - 1))
    # in the inclusion-exclusion- probably related to the basic binomial coefficient
    # recurrence relation
    ps = SimplePrimeSieve(p_max)
    p_lst = []
    for p in ps.p_lst:
        if p > p_max: break
        p_lst.append(p)
    p_lst = p_lst[::-1]
    
    n_p = len(p_lst)
    
    if n_p < min_n_p_divide: return 0
    tail_p_prod_lst = [1]
    for i in reversed(range(n_p - min_n_p_divide, n_p)):
        tail_p_prod_lst.append(p_lst[i] * tail_p_prod_lst[-1])
    print(p_lst)
    print(tail_p_prod_lst)
    def recur(idx: int, curr: int, n_p_incl: int) -> int:
        #print(idx, curr, n_p_incl)
        if curr * (tail_p_prod_lst[max(0, min_n_p_divide - n_p_incl)]) > num_max:
            return 0
        elif idx == n_p:
            ans = (num_max // curr) * math.comb(n_p_incl - 1, min_n_p_divide - 1)
            #if n_p_incl > min_n_p_divide:
            #    print(n_p_incl, curr, ans)
            #    return 0
            return -ans if (n_p_incl - min_n_p_divide) & 1 else ans
        elif idx + (min_n_p_divide - n_p_incl) == n_p:
            #print(min_n_p_divide, curr * tail_p_prod_lst[n_p - idx])
            return num_max // (curr * tail_p_prod_lst[n_p - idx])
        
        return recur(idx + 1, curr, n_p_incl) + recur(idx + 1, curr * p_lst[idx], n_p_incl + 1)
    
    res = recur(0, 1, 0)
    return res

# Problem 269
def countNonnegativeCoefficientPolynomialsWithIntegerZero(
    polynomial_num_max: int=10 ** 16,
    base: int=10,
) -> int:
    """
    Solution to Project Euler #269
    """
    # Using digit dynamic programming
    # Also using the fact that for any polynomial with integer
    # coefficients, if a polynomial factor exists with integer
    # coefficients, its complementary factor (i.e. the polynomial
    # that multiplies with the factor polynomial to give the
    # original polynomial) must also have integer coefficients.
    # Consequently, any integer root must be a factor of the
    # constant term, and thus for any polynomial whose constant
    # term is non-zero, any integer root can be no larger than
    # the constant term.
    # Furthermore, for any polynomial with no negative terms,
    # any real (and so integer) root must be non-positive, with
    # zero being a root if and only if the constant coefficient
    # is zero


    # All polynomials that do not have zero as a root (corresponding
    # to the polynomials with non-zero constant coefficient)
    num_mx_digs = []
    num2 = polynomial_num_max
    while num2:
        num2, d = divmod(num2, base)
        num_mx_digs.append(d)
    n_dig_max = len(num_mx_digs)
    # Consider reversing
    remain_mx_lsts = []
    remain_mn_lsts = []
    for d in range(1, base):
        remain_mx_lsts.append([0, base - 1])
        # in the list below, 1 is used rather than 0 as
        # we are considering the polynomials with non-zero constant
        # coefficient
        remain_mn_lsts.append([0, 1])
        mult = d
        for i in range(2, n_dig_max, 2):
            mult *= d
            remain_mx_lsts[d - 1].append((base - 1) * mult)
            remain_mn_lsts[d - 1].append(0)
            if i == n_dig_max - 1: continue
            mult *= d
            remain_mx_lsts[d - 1].append(0)
            remain_mn_lsts[d - 1].append(-(base - 1) * mult)
        #remain_mx_lsts.append(0)
        #remain_mn_lsts.append(0)
        for i in range(1, n_dig_max):
            remain_mx_lsts[d - 1][i] += remain_mx_lsts[d - 1][i - 1]
            remain_mn_lsts[d - 1][i] += remain_mn_lsts[d - 1][i - 1]
    print(remain_mn_lsts)
    print(remain_mx_lsts)
    def positiveDistinctIntegerCombinationsGenerator(
        prod_mx: int,
    ) -> Generator[List[int], None, None]:
        
        curr_incl = []
        def recur(num: int, curr_prod: int=1) -> Generator[List[int], None, None]:
            if not num:
                if curr_incl: yield curr_incl[::-1]
                return
            yield from recur(num - 1, curr_prod=curr_prod)
            curr_prod *= num
            if curr_prod > prod_mx: return
            curr_incl.append(num)
            yield from recur(num - 1, curr_prod=curr_prod)
            curr_incl.pop()
            return

        yield from recur(prod_mx, curr_prod=1)
    
    ref = [1]
    def countPolynomialsWithNegativeRoots(
        poly_coeffs: List[int],
        neg_roots: List[int],
    ) -> int:
        

        n_roots = len(neg_roots)

        curr_bals = [0] * n_roots
        memo = {}
        def recur(idx: int, tight: bool=True) -> int:
            if not idx:
                st = set(curr_bals)
                # All balances should be guaranteed to be between 1 and base - 1 inclusive
                # at this point
                return int(len(st) == 1 and (not tight or curr_bals[0] <= poly_coeffs[0]))
            args = (idx, tuple(curr_bals), tight)
            if args in memo.keys(): return memo[args]
            res = 0
            # Consider splitting into even idx and odd idx paths
            d_mn = 0 # review based on the next remain
            d_mx = poly_coeffs[idx] if tight else base - 1
            mults = [(-rt) ** idx for rt in neg_roots]
            if idx & 1:
                # Odd powers (subtracted)
                for i in range(n_roots):
                    # mults[i] * d + curr_bals[i] >= -remain_mx_lsts[neg_roots[i] - 1][idx]- note mults[i] negative
                    d_mx2 = ((remain_mx_lsts[neg_roots[i] - 1][idx] + curr_bals[i]) // (-mults[i]))
                    d_mx = min(d_mx, d_mx2)
                    # mults[i] * d + curr_bals[i] <= -remain_mn_lsts[neg_roots[i] - 1][idx]- note mults[i] negative
                    d_mn2 = -((-(remain_mn_lsts[neg_roots[i] - 1][idx] + curr_bals[i])) // (-mults[i]))
                    d_mn = max(d_mn, d_mn2)
                    if neg_roots == ref:
                        print(f"neg_roots = {neg_roots}, idx = {idx}, i = {i}, balance = {curr_bals[i]}, d_mn2 = {d_mn2}, d_mx2 = {d_mx2}, d_mn = {d_mn}, d_mx = {d_mx}")

            else:
                # Even powers (added)
                for i in range(n_roots):
                    # mults[i] * d + curr_bals[i] >= -remain_mx_lsts[neg_roots[i] - 1][idx]
                    d_mn2 = -((remain_mx_lsts[neg_roots[i] - 1][idx] + curr_bals[i]) // mults[i])
                    d_mn = max(d_mn, d_mn2)
                    # mults[i] * d + curr_bals[i] <= -remain_mn_lsts[neg_roots[i] - 1][idx]
                    d_mx2 = (-(remain_mn_lsts[neg_roots[i] - 1][idx] + curr_bals[i])) // mults[i]
                    d_mx = min(d_mx, d_mx2)
                    if neg_roots == ref:
                        print(f"neg_roots = {neg_roots}, idx = {idx}, i = {i}, balance = {curr_bals[i]}, d_mn2 = {d_mn2}, d_mx2 = {d_mx2}, d_mn = {d_mn}, d_mx = {d_mx}")
            tight2 = (tight and d_mx == poly_coeffs[idx])
            curr_bals0 = list(curr_bals)
            for d in range(d_mn, d_mx - tight2 + 1):
                for i in range(n_roots):
                    curr_bals[i] = curr_bals0[i] + d * mults[i]
                res += recur(idx - 1, tight=False)
            if tight2:
                for i in range(n_roots):
                    curr_bals[i] = curr_bals0[i] + poly_coeffs[idx] * mults[i]
                res += recur(idx - 1, tight=True)
            for i in range(n_roots):
                curr_bals[i] = curr_bals0[i]
            
            memo[args] = res
            return res
        
        res = recur(n_dig_max - 1, tight=True)
        if neg_roots == ref:
            print(memo)
        return res

    res = 0
    for root_comb in positiveDistinctIntegerCombinationsGenerator(base - 1):
        
        cnt = countPolynomialsWithNegativeRoots(num_mx_digs, root_comb)
        print(f"for root combination {root_comb}, count = {cnt}")
        # Using inclusion/exclusion
        res += cnt if len(root_comb) & 1 else -cnt

    # Number of polynomials that have 0 as a root
    # This is equal to he total count of numbers that end in
    # zero when represented in the chosen base no greater
    # than polynomial_num_max
    res += polynomial_num_max // base
    return res

# Problem 270
def countPolygonCuts(
    side_lengths: List[int]=[30] * 4,
    res_md: Optional[int]=10 ** 8,
) -> int:
    """
    Solution to Project Euler #270
    """
    # Review- look into the solutions on the Project Euler forum, which generally appear to
    # take a very different approach- especially those utilising Catalan numbers
    print(f"side lengths = {side_lengths}")
    modAdd = (lambda x, y: x + y) if res_md is None else (lambda x, y: (x + y) % res_md)
    modMult = (lambda x, y: x * y) if res_md is None else (lambda x, y: (x * y) % res_md)

    n_sides = len(side_lengths)

    #def reflectPosition(pos: Tuple[int, int]) -> Tuple[int, int]:
    #    return (3 - )
    getSideLength = lambda idx, rev: side_lengths[~idx] if rev else side_lengths[idx]

    ref = None#((1, 0), ((1, 0), (2, 2)))

    memo1 = {}
    def recur1(start: Tuple[int, int], end_rng: Tuple[Tuple[int, int],Tuple[int, int]]) -> int:
        #print(start, end_rng)
        if end_rng[0] == end_rng[1]:
            return 1
        if start[0] == end_rng[1][0] and start[1] <= end_rng[1][1]:# or (end[0] == (start[0] + 1) % n_sides and not end[1]) or (start[0] == (end[0] + 1) % n_sides and not start[1]):
            #print("hi1")
            return int(end_rng[1][1] - start[1] <= 1)
        elif end_rng[1][1] == 0 and (start[0] + 1) % n_sides == end_rng[1][0]:
            return int(start[1] == getSideLength(start[0], False) - 1)
        if end_rng[1][0] == end_rng[0][0] or (end_rng[1][1] == 0 and (end_rng[0][0] + 1) % n_sides == end_rng[1][0]):
            return 1
        #if end_rng[1][0] == end_rng[0][0]:
        #    if start[0] != end_rng[1][0]: return 1
        #    return int(start == end_rng[0] and end_rng[1][1] <= start[1] + 1)
        #elif (end_rng[1][1] == 0 and (end_rng[0][0] + 1) % n_sides == end_rng[1][0]):
        #    if start[0] != end_rng[0][0]: return 1
        #    return int(start == end_rng[0] and start[1] == getSideLength(start[0], False) - 1)

        side_len0 = getSideLength(start[0], False)
        if end_rng[0] == start:
            start_nxt = (start[0], start[1] + 1) if start[1] < side_len0 - 1 else ((start[0] + 1) % n_sides, 0)
        else: start_nxt = end_rng[0]
        #if not start_nxt[1] and (start_nxt[0] == end[0] or (not end[1] and (start_nxt[0] + 1) % n_sides == end[0])):
        #    #print("hi2")
        #    return 1
        args = (start, end_rng)
        
        if args in memo1.keys():
            return memo1[args]
        #print(args)
        #ref = None#((0, 0), ((2, 1), (3, 1)))#((0, 0), ((0, 0), (3, 0)))#((0, 0), ((2, 0), (3, 1)))
        is_ref = (args == ref)

        if is_ref:
            print(f"calculating {ref}")

        res = 0
        if (end_rng[0][0] == start[0] and end_rng[0][1] >= start[1]) or (end_rng[0][1] == 0 and end_rng[0][0] == (start[0] + 1) % n_sides):
            side_idx0 = (start[0] + 1) % n_sides
            begin_idx = 1
        else:
            side_idx0 = end_rng[0][0]
            begin_idx = end_rng[0][1] + 1
            if begin_idx == getSideLength(side_idx0, False):
                side_idx0 = (side_idx0 + 1) % n_sides
                begin_idx = 0
        
        for side_idx in range(side_idx0, end_rng[1][0]):
            side_len = getSideLength(side_idx, False)
            end_idx = side_len
            if side_idx == start[0]: end_idx = min(end_idx, 2)
            if side_idx == (start[0] + 1) % n_sides: begin_idx = max(begin_idx, 1)
            #print(f"side_dx = {side_idx}, idx_rng = [{begin_idx}, {end_idx})")
            for idx in range(begin_idx, end_idx):
                #print(f"hi1.1 from {args}")
                term1 = recur1(start_nxt, (start_nxt, (side_idx, idx)))
                #print(f"hi1.2 from {args}")
                # (3, 0), ((3, 0), (3, 1))
                term2 = recur1(start, ((side_idx, idx), end_rng[1]))
                term = modMult(term1, term2)
                if is_ref:
                    print(f"for start = {args[0]}, end range = {args[1]}, term type 1 ({side_idx}, {idx}) equals {term}")
                res = modAdd(res, term)
            begin_idx = 0

        side_idx = end_rng[1][0]
        end_idx = end_rng[1][1]
        side_len = getSideLength(side_idx, False)
        if (side_idx == start[0] and end_idx < start[1]) or (start[1] == 0 and (side_idx + 1) % n_sides == start[0]):
            begin_idx = end_idx
            end_idx = end_idx
            #if (start[1] if start[1] else side_len) - end_idx == 1:
            #    begin_idx = end_idx
            #    end_idx = end_idx + 1

        #if side_idx == start[0]:
        #    end_idx = min(end_idx, 2)
        #if side_idx == (start[0] + 1) % n_sides: begin_idx = max(begin_idx, 1)
        #print(f"side_dx = {side_idx}, idx_rng = [{begin_idx}, {end_idx})")
        for idx in range(begin_idx, end_idx):
            #print(f"hi2.1 from {args}")
            term1 = recur1(start_nxt, (start_nxt, (side_idx, idx)))
            #print(f"hi2.2 from {args}")
            term2 = recur1(start, ((side_idx, idx), end_rng[1]))
            term = modMult(term1, term2)
            if is_ref:
                print(f"for start = {args[0]}, end range = {args[1]}, term type 2 ({side_idx}, {idx}) equals {term}")
            res = modAdd(res, term)
        #begin_idx = 0
        #print(f"hi3 from {args}")
        term = recur1(start_nxt, (start_nxt, end_rng[1]))
        if is_ref:
            print(f"for start = {args[0]}, end range = {args[1]}, term type 3 ({start_nxt[0]}, {start_nxt[1]}) equals {term}")
        res = modAdd(res, term)
        memo1[args] = res
        return res
    #print("hi0")
    res = recur1((0, 0), ((0, 0), (n_sides - 1, side_lengths[-1] - 1)))
    #print(memo1)
    return res
    """
    memo2 = {}
    def recur2(start: Tuple[int, int], end: Tuple[int, int]) -> int:
        args = (start, end)
        
        if args in memo2.keys():
            return memo2[args]
        res = 0


        memo2[args] = res
        return res
    """
    """
    res = 0
    start1 = (0, 1) if getSideLength(0, False) > 1 else (1, 0)
    start2 = (0, 1) if getSideLength(0, True) > 1 else (1, 0)
    # TODO- account for the first or last edge (or both) having length 1
    for end_side1_idx in range(1, n_sides - 1):
        side1_begin_idx = 1 if end_side1_idx == 1 else 0
        side_len1 = getSideLength(end_side1_idx, rev=False)
        end_side2_idx = 1
        side_len2 = getSideLength(end_side2_idx, rev=True)
        for end_side2_idx in range(1, n_sides - end_side1_idx - 1):
            side_len2 = getSideLength(end_side2_idx, rev=True)
            side2_begin_idx = 1 if end_side2_idx == 1 else 0
            for idx1 in range(side1_begin_idx, side_len1):
                for idx2 in range(side2_begin_idx, side_len2):
                    cnt = modMult(modMult(recur1(start1, (end_side1_idx, idx1), rev=False), recur1(start2, (end_side2_idx, idx2), rev=True)), recur1((end_side1_idx, idx1), (), rev=False))
                    res = modAdd(res, cnt)
                    print(f"count for end1 = ({end_side1_idx}, {idx1}), end2 = ({end_side2_idx}, {idx2}) equals {cnt}")
        end_side2_idx = n_sides - end_side1_idx - 1
        side2_begin_idx = 1 if end_side2_idx == 1 else 0
        for idx1 in range(side1_begin_idx, side_len1):
            print(f"idx1 = {idx1}")
            for idx2 in range(side2_begin_idx, min(side_len1 - idx1 + 1, side_len1)):
                print(f"idx2 = {idx2}")
                cnt = modMult(recur1(start1, (end_side1_idx, idx1), rev=False), recur1(start2, (end_side2_idx, idx2), rev=True))
                res = modAdd(res, cnt)
                print(f"count for end1 = ({end_side1_idx}, {idx1}), end2 = ({end_side2_idx}, {idx2}) equals {cnt}")
            if not idx1:
                cnt = modMult(recur1(start1, (end_side1_idx, idx1), rev=False), recur1(start2, (end_side2_idx + 1, 0), rev=True))
                res = modAdd(res, cnt)
                print(f"count for end1 = ({end_side1_idx}, {idx1}), end2 = ({end_side2_idx + 1}, 0) equals {cnt}")
    
    end_side1_idx = n_sides - 1
    side1_begin_idx = 1 if end_side1_idx == 1 else 0
    side_len1 = getSideLength(end_side1_idx, rev=False)
    idx1 = side_len1 - 1
    cnt = recur1(start1, (end_side1_idx, idx1), rev=False)
    res = modAdd(res, cnt)
    print(f"total count for end1 = ({end_side1_idx}, {idx1}) equals {cnt}")
    print(memo1)
    return res % res_md
    """

# Problem 271
def carmichaelLambdaFunctionPrimeFactorisation(
    n_pf: Dict[int, int],
    ps: Optional[PrimeSPFsieve]=None
) -> Dict[int, int]:
    carmichael_pf = {}
    print(f"finished creating prime sieve")
    for p2, f in n_pf.items():
        if p2 == 2:
            f2 = f - 1 - (f >= 3)
            carmichael_pf[p2] = max(carmichael_pf.get(p2, 0), f2)
            continue
        if f > 1: carmichael_pf[p2] = max(carmichael_pf.get(p2, 0), f - 1)
        #pf2 = ps2.primeFactorisation(p2 >> 1)
        for p3, f2 in calculatePrimeFactorisation(p2 >> 1, ps=ps):#pf2.items():
            if p3 == 2:
                #print(f"p3 = {p3}, f2 = {f2}")
                carmichael_pf[2] = max(carmichael_pf.get(2, 0), f2 + 1)
            else: carmichael_pf[p3] = max(carmichael_pf.get(p3, 0), f2)
    return carmichael_pf

"""
def solveSimultaneousLinearCongruence(a1: int, b1: int, a2: int, b2: int) -> Tuple[int, int]:
    # Assumes b1 and b2 are coprime
    if gcd(b1, b2) != 1:
        raise ValueError("b1 and b2 must be coprime")

    if b1 < b2: (a1, b1), (a2, b2) = (a2, b2), (a1, b1)
    b3 = b2
    k3, a3 = b1 % b3, (a2 - a1) % b3
"""

def cubicRootsOfUnityModuloNGenerator(
    n: int,
    ps: Optional[PrimeSPFsieve]=None,
) -> Generator[int, None, None]:

    # Using Chinese remainder theorem

    # Note- the values are not yielded in order of size.

    pf = calculatePrimeFactorisation(n, ps=ps)
    n_p = len(pf)
    p_pow_lst = [p ** pf[p] for p in sorted(pf.keys())]
    p_pow_lst.sort(reverse=True)
    p_pow_bs = [[] for _ in range(n_p)]
    md_lst = [0] * n_p
    curr_md = 1
    for idx, num in enumerate(p_pow_lst):
        curr_md *= num
        md_lst[idx] = curr_md
        # 1 is always a cubic root of unity
        p_pow_bs[idx].append(1)
        for x in range(2, num):
            if pow(x, 3, num) == 1:
                p_pow_bs[idx].append(x)
    #print(p_pow_lst)
    #print(p_pow_mds)
    #print(md_lst)
    md_prods_cumu = [1]
    for num in p_pow_lst[:-1]:
        md_prods_cumu.append(md_prods_cumu[-1] * num)
    md_prods_cumu_rev = [1]
    for num in reversed(p_pow_lst[1:]):
        md_prods_cumu_rev.append(md_prods_cumu_rev[-1] * num)
    md_prods_cumu_rev = md_prods_cumu_rev[::-1]
    #print(md_prods_cumu, md_prods_cumu_rev)
    mults = [1] * n_p
    for idx, num in enumerate(p_pow_lst):
        N = md_prods_cumu[idx] * md_prods_cumu_rev[idx]
        #print(num, N)
        N_inv = solveLinearCongruence(N, 1, num)
        #print(num, n // num, N, N_inv)
        mults[idx] = (N * N_inv) % n

    def recur(idx: int, curr: int) -> Generator[int, None, None]:
        if idx == n_p:
            #print(curr)
            #return 0 if curr == 1 else curr
            yield curr
            return
        for b in p_pow_bs[idx]:
            yield from recur(idx + 1, curr=(curr + mults[idx] * b) % n)
        return

    yield from recur(0, 0)
    return

def sumOfNontrivialCubicRootsOfUnityModuloN(
    n: int=13082761331670030,
    ps: Optional[PrimeSPFsieve]=None,
) -> int:
    """
    Solution to Project Euler #271
    """
    res = sum(cubicRootsOfUnityModuloNGenerator(n, ps=ps)) - 1
    return res

# Problem 272
def integersNForWhichThereAreGivenNumberOfNontrivialCubicRootsOfUnityModuloNSum(
    n_max: int,
    nontrivial_root_count: int,
) -> int:
    """
    Solution to Project Euler #272
    """
    m2 = nontrivial_root_count + 1
    # Solution counts are always a power of three
    exp = 0
    while True:
        m2, r = divmod(m2, 3)
        if r: break
        exp += 1
    print(f"m2 = {m2}, r = {r}, exp = {exp}")
    if m2 or r != 1: return []
    #mn_p_1mod3_prod = 1
    mn_p_1mod3_lst = []
    ps = SimplePrimeSieve()
    if exp:
        for p in ps.endlessPrimeGenerator():
            if p % 3 != 1: continue
            #mn_p_1mod3_prod *= p
            mn_p_1mod3_lst.append(p)
            if len(mn_p_1mod3_lst) == max(2, exp): break
    mn_p_1mod3_prod0 = 1
    for p in mn_p_1mod3_lst[:-2]:
        mn_p_1mod3_prod0 *= p
    mn_p_1mod3_prod1 = mn_p_1mod3_prod0 * mn_p_1mod3_lst[-2]
    mn_p_1mod3_prod2 = mn_p_1mod3_prod1 * mn_p_1mod3_lst[-1]
    print(f"minimum p that are 1 modulo 3 product of {exp - 1} terms = {mn_p_1mod3_prod1}, of {exp} terms = {mn_p_1mod3_prod2}")
    print(n_max // (9 * mn_p_1mod3_prod1), n_max // mn_p_1mod3_prod2)
    # The integers up to n_max // min(9 * mn_p_1mod3_prod1, mn_p_1mod3_prod2) who have no prime factors that are not of the form (3k + 2)
    p_2mod3_factors_lst = [1]
    p_2mod3_mx1 = n_max // min(9 * mn_p_1mod3_prod1, mn_p_1mod3_prod2)
    #print(p_2mod3_mx1)
    prev_p = -1
    for p in ps.endlessPrimeGenerator():
        if p > p_2mod3_mx1: break
        elif p % 3 != 2:
            continue
        if (prev_p // 10000) != (p // 10000):
            print(f"p = {p}, p max = {p_2mod3_mx1}")
        prev_p = p
        #print(f"p = {p}, p_2mod3_mx1 = {p_2mod3_mx1}, count = {len(p_2mod3_factors_lst)}")
        #p_2mod3_factors_lst.append(p)
        for i in itertools.count(0):
            if i >= len(p_2mod3_factors_lst): break
            num = p * p_2mod3_factors_lst[i]
            if num > p_2mod3_mx1: continue
            p_2mod3_factors_lst.append(num)
    p_2mod3_factors_lst.sort()
    p_2mod3_factors_lst_cumu = [0]
    for num in p_2mod3_factors_lst:
        p_2mod3_factors_lst_cumu.append(p_2mod3_factors_lst_cumu[-1] + num)
    #print(len(p_2mod3_factors_lst), p_2mod3_factors_lst[-1])
    #print(p_2mod3_factors_lst, p_2mod3_factors_lst_cumu)

    p_1mod3_lst = []
    p_1mod3_mx1 = n_max // min(mn_p_1mod3_prod1, mn_p_1mod3_prod0 * 9)
    for p in ps.endlessPrimeGenerator():
        if p > p_1mod3_mx1: break
        elif p % 3 != 1: continue
        p_1mod3_lst.append(p)
    #print(p_1mod3_lst)


    def calculateSum(n_p_1mod3: int, mx: int) -> int:
        if n_p_1mod3 < 0: return 0
        def recur(p_1mod3_idx: int, remain_p_1mod3: int, mx: int) -> int:
            if p_1mod3_idx + remain_p_1mod3 > len(p_1mod3_lst): return 0
            if not remain_p_1mod3:
                i = bisect.bisect_right(p_2mod3_factors_lst, mx)
                return p_2mod3_factors_lst_cumu[i]
            res = 0
            
            for idx in range(p_1mod3_idx, len(p_1mod3_lst)):
                p = p_1mod3_lst[idx]
                if p ** remain_p_1mod3 > mx: break
                mx2 = mx // p
                mult = p
                while mx2 > 0:
                    res += recur(idx + 1, remain_p_1mod3 - 1, mx2) * mult
                    mx2 //= p
                    mult *= p
            return res
        return recur(0, n_p_1mod3, mx)

    res = 0
    n_mx2 = n_max
    mult = 1
    for _ in range(2):
        mult2 = mult * 3
        print(f"calculating for numbers that are multiples of {mult} but not {mult2}")
        #print(exp, n_mx2)
        tot = calculateSum(exp, n_mx2) * mult
        print(f"sum for these numbers = {tot}")
        res += tot
        n_mx2 //= 3
        mult = mult2

    # Numbers that are multiples of 3 but not 9
    #print("calculating for numbers that are multiples of 3 but not 9")
    #res += calculateSum(exp, n_max // 3)

    # Numbers that are multiples of 9
    #print("calculating for numbers that are multiples of 9")
    #n_mx2 = n_max // 9
    #mult = 9
    while n_mx2 > 0:
        mult2 = mult * 3
        print(f"calculating for numbers that are multiples of {mult} but not {mult2}")
        print(exp - 1, n_mx2)
        tot = calculateSum(exp - 1, n_mx2) * mult
        print(f"sum for these numbers = {tot}")
        res += tot
        n_mx2 //= 3
        mult = mult2

    return res

# Problem 273
def findSumOfSquaresEqualToSquareFreeProductBruteForce(p_max: int) -> List[List[Tuple[int, List[Tuple[int, int]]]]]:

    ps = SimplePrimeSieve()
    p_lst = []
    for num in range(5, p_max + 1, 4):
        if ps.isPrime(num):
            p_lst.append(num)
    n_p = len(p_lst)
    print(f"the primes are (total of {n_p}): {p_lst}")
    res = [[] for _ in range(n_p + 1)]
    res[0].append((1, [(0, 1)]))
    for bm in range(1, 1 << n_p):
        p_cnt = bm.bit_count()
        num = 1
        bm2 = bm
        for i, p in enumerate(p_lst):
            if bm2 & 1:
                num *= p
                if bm2 == 1: break
            bm2 >>= 1
        res[p_cnt].append((num, []))
        for a in range(isqrt(num >> 1) + 1):
            num2 = num - a ** 2
            b = isqrt(num2)
            if b ** 2 == num2:
                res[p_cnt][-1][1].append((a, b))
        if not res[p_cnt][-1][1]:
            res[p_cnt].pop()
    for p_cnt in range(n_p):
        print(f"products of {p_cnt} primes of the form 4k + 1:")
        for num, ab_lst in res[p_cnt]:
            print(f"{num}: {ab_lst}")
    return res

def sumOfSquaresSmallerSquareSum(p_max: int=149) -> int:
    """
    Solution to Project Euler #273
    """
    # Using Gaussian integers
    ps = SimplePrimeSieve()
    p_lst = []
    for num in range(5, p_max + 1, 4):
        if ps.isPrime(num):
            p_lst.append(num)
    n_p = len(p_lst)
    print(f"the primes are (total of {n_p}): {p_lst}")
    #mx = 1
    #for p in p_lst: mx *= p
    #print(f"max number = {mx}")
    curr = []
    res = 0
    
    for p in p_lst[:-1]:
        print(f"p = {p}")
        # Primes of the form p = 4k + 1 are guaranteed to have exactly one
        # ordered pair of stricltly poitive integers (a, b) for which
        # a <= b and a ** 2 + b ** 2 = p
        for a in range(isqrt(p >> 1) + 1):
            b_sq = p - a ** 2
            b = isqrt(b_sq)
            if b ** 2 == b_sq:
                pair = (a, b)
                break
        else:
            # Should not happen
            continue
        n_pair0 = len(curr)
        #print(p, pair)
        #print(f"n_pair0 = {n_pair0}")
        res += pair[0]
        curr.append(pair)
        for i in range(n_pair0):
            for pair2 in [curr[i], (curr[i][1], curr[i][0])]:
                pair3 = tuple(sorted((abs(pair[0] * pair2[0] - pair[1] * pair2[1]), pair[0] * pair2[1] + pair[1] * pair2[0])))
                #print(p, pair, pair2, pair3)
                res += pair3[0]
                curr.append(pair3)
    p = p_lst[-1]
    print(f"p = {p}")
    pair = None
    for a in range(isqrt(p >> 1) + 1):
        b_sq = p - a ** 2
        b = isqrt(b_sq)
        if b ** 2 == b_sq:
            pair = (a, b)
            break
    if pair is not None:
        #print(p, pair)
        n_pair0 = len(curr)
        #print(f"n_pair0 = {n_pair0}")
        res += pair[0]
        for i in range(n_pair0):
            for pair2 in [curr[i], (curr[i][1], curr[i][0])]:
                pair3 = tuple(sorted((abs(pair[0] * pair2[0] - pair[1] * pair2[1]), pair[0] * pair2[1] + pair[1] * pair2[0])))
                #print(p, pair, pair2, pair3)
                res += pair3[0]
    return res

# Problem 274
def calculateOsculator(num: int, base: int=10) -> int:
    k = solveLinearCongruence(num, base - 1, base)
    if k < 0: return -1
    return (num * k + 1) // base

def calculateCoprimePrimeOsculatorSum(p_max: int=10 ** 7 - 1, base: int=10) -> int:
    """
    Solution to Project Euler #274
    """
    res = 0
    ps = SimplePrimeSieve(p_max)
    for p in ps.p_lst:
        if not base % p: continue
        elif p > p_max: break
        res += calculateOsculator(p, base=base)
    return res

# Problem 277
def modifiedCollatzSequenceSmallestStartWithSequence(n_min: int=10 ** 15 + 1, seq: str="UDDDUdddDDUDDddDdDddDDUDDdUUDd") -> int:

    # Review- Look into solutions on the Project Euler forum which calculate directly
    def reversedStep(num: CustomFraction, l: int) -> CustomFraction:
        if l == "D":
            return num * 3
        elif l == "U":
            return (num * 3 - 2) / 4
        elif l == "d":
            return (num * 3 + 1) / 2

    curr = CustomFraction(1, 1)
    for l in reversed(seq):
        curr = reversedStep(curr, l)
    
    lo, hi = 1, 1
    while True:
        lo = hi
        hi <<= 1
        curr = hi
        for l in reversed(seq):
            curr = reversedStep(curr, l)
        if curr >= n_min: break
    
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        curr = mid
        for l in reversed(seq):
            curr = reversedStep(curr, l)
        if curr >= n_min: hi = mid
        else: lo = mid + 1

    #r = n_min / curr
    #print(curr, r, r.numerator / r.denominator)
    #curr = r
    #lb = ((r.numerator - 1) // r.denominator) + 1
    lb = lo
    print(f"lower bound finish = {lb}")
    curr = CustomFraction(lb, 1)
    for l in reversed(seq):
        curr = reversedStep(curr, l)
    print(curr, curr.numerator / curr.denominator)

    for num in itertools.count(lb):
        #print(f"trying {num}")
        curr = CustomFraction(num, 1)
        for l in reversed(seq):
            curr = reversedStep(curr, l)
            if curr.denominator != 1: break
        else: break
    return curr.numerator

# Problem 278
def calculateDistinctPrimeCombinationsFrobeniusNumber(p_lst: List[int]) -> int:
    res = 1
    n = len(p_lst)
    tail_prod = [1] * (n + 1)
    for i in reversed(range(n)):
        tail_prod[i] = p_lst[i] * tail_prod[i + 1]
    res = (n - 1) * tail_prod[0]
    curr = 1
    #print(tail_prod)
    for i in range(n):
        #print(i, curr, tail_prod[i + 1])
        res -= curr * tail_prod[i + 1]
        curr *= p_lst[i]
    return res

def calculateDistinctPrimeCombinationsFrobeniusNumberSum(n_p: int=3, p_max: int=4999) -> int:
    """
    Solution to Project Euler #278
    """
    # Review- Try to find an equation for the number of occurrences
    # of the different prime combinations
    def orderedPrimeListGenerator(
        n_p: int,
        p_max: int,
        p_distinct: bool=True,
        ps: Optional[SimplePrimeSieve]=None,
    ) -> Generator[List[int], None, None]:
        if ps is None:
            ps = SimplePrimeSieve()
        ps.extendSieve(p_max)

        curr = [0] * n_p
        def recur(idx: int, p_max: int):
            if idx < 0:
                yield list(curr)
                return
            for p in ps.endlessPrimeGenerator():
                if p > p_max: break
                curr[idx] = p
                yield from recur(idx - 1, p - p_distinct)
            return
        yield from recur(n_p - 1, p_max)
        return
    res = 0
    largest_p = 0
    for p_lst in orderedPrimeListGenerator(n_p, p_max, p_distinct=True, ps=None):
        if p_lst[-1] != largest_p:
            largest_p = p_lst[-1]
            print(f"adding terms whose largest prime is {largest_p}")
        res += calculateDistinctPrimeCombinationsFrobeniusNumber(p_lst)
    return res

##############
project_euler_num_range = (251, 300)

def evaluateProjectEulerSolutions251to300(eval_nums: Optional[Set[int]]=None) -> None:
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
        )
        print(f"Solution to Project Euler #255 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 256 in eval_nums:
        since = time.time()
        res = smallestRoomSizeWithExactlyNTatamiFreeConfigurations(n_config=200)
        print(f"Solution to Project Euler #256 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 257 in eval_nums:
        since = time.time()
        res = angularBisectorTrianglePartitionIntegerRatioCount(perimeter_max=10 ** 8)
        print(f"Solution to Project Euler #257 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 258 in eval_nums:
        since = time.time()
        res = calculateGeneralisedLaggedFibonacciTerm(
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
        res = stoneGamePlayerTwoWinningConfigurationsSum(pile_size_max=1000)
        print(f"Solution to Project Euler #260 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 261 in eval_nums:
        since = time.time()
        res = distinctSquarePivotsSum(k_max=10 ** 10)
        print(f"Solution to Project Euler #261 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 262 in eval_nums:
        since = time.time()
        res = mountainRangeDistance(res_eps=1e-4)
        print(f"Solution to Project Euler #263 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 263 in eval_nums:
        since = time.time()
        res = engineersParadiseSum(n_incl=3, ps=None)
        print(f"Solution to Project Euler #263 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 264 in eval_nums:
        since = time.time()
        res = trianglesWithLatticePointVerticesAndFixedCircumcentreAndOrthocentrePerimeterSum(
            orthocentre_x=5,
            perimeter_max=50,
        )
        print(f"Solution to Project Euler #264 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 265 in eval_nums:
        since = time.time()
        res = allBinaryCirclesSum(n=5)
        print(f"Solution to Project Euler #265 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 266 in eval_nums:
        since = time.time()
        res = pseudoSquareRootOfProductOfInitialPrimes(
            p_max=190,
            res_md=10 ** 16,
        )
        print(f"Solution to Project Euler #266 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 267 in eval_nums:
        since = time.time()
        res = maximiseProbabilityOfGivenProfitInCoinTossGameFloat(n_tosses=10 ** 3, target_multiplier=10 ** 9, eps=1e-13)
        print(f"Solution to Project Euler #267 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 268 in eval_nums:
        since = time.time()
        res = numbersDivisibleByAtLeastNOfInitialPrimesCount(
            num_max=10 ** 16 - 1,
            p_max=99,
            min_n_p_divide=4,
        )
        print(f"Solution to Project Euler #268 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 269 in eval_nums:
        since = time.time()
        res = countNonnegativeCoefficientPolynomialsWithIntegerZero(
            polynomial_num_max=10 ** 5,
            base=10,
        )
        print(f"Solution to Project Euler #269 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 270 in eval_nums:
        since = time.time()
        res = countPolygonCuts(side_lengths=[30, 30, 30, 30], res_md=10 ** 8)
        print(f"Solution to Project Euler #270 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 271 in eval_nums:
        since = time.time()
        res = sumOfNontrivialCubicRootsOfUnityModuloN(
            n=13082761331670030,
            ps=None,
        )
        print(f"Solution to Project Euler #271 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 272 in eval_nums:
        since = time.time()
        res = integersNForWhichThereAreGivenNumberOfNontrivialCubicRootsOfUnityModuloNSum(
            n_max=10 ** 11,
            nontrivial_root_count=242,
        )
        print(f"Solution to Project Euler #272 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 273 in eval_nums:
        since = time.time()
        res = sumOfSquaresSmallerSquareSum(p_max=149)
        print(f"Solution to Project Euler #273 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 274 in eval_nums:
        since = time.time()
        res = calculateCoprimePrimeOsculatorSum(p_max=10 ** 7 - 1, base=10)
        print(f"Solution to Project Euler #274 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 277 in eval_nums:
        since = time.time()
        res = modifiedCollatzSequenceSmallestStartWithSequence(n_min=10 ** 15 + 1, seq="UDDDUdddDDUDDddDdDddDDUDDdUUDd")
        print(f"Solution to Project Euler #277 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 278 in eval_nums:
        since = time.time()
        res = calculateDistinctPrimeCombinationsFrobeniusNumberSum(n_p=3, p_max=4999)
        print(f"Solution to Project Euler #278 = {res}, calculated in {time.time() - since:.4f} seconds")

    print(f"Total time taken = {time.time() - since0:.4f} seconds")

if __name__ == "__main__":
    eval_nums = {269}
    evaluateProjectEulerSolutions251to300(eval_nums)

"""
for triangle_pts in trianglesWithLatticePointVerticesAndFixedCircumcentreAndOrthocentreByPerimeterGenerator(
    orthocentre_x=5,
    perimeter_max=50,
):
    print(triangle_pts)
"""
#print(calculateDistinctPrimeCombinationsFrobeniusNumber([5, 7]))
#print(calculateDistinctPrimeCombinationsFrobeniusNumber([2, 3, 5]))
#print(calculateDistinctPrimeCombinationsFrobeniusNumber([2, 7, 11]))