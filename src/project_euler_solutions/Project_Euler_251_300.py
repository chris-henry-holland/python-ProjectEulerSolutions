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

def trianglesWithLatticePointVerticesAndFixedCircumcentreAndOrthocentreByPerimeterGenerator(
    orthocentre: Tuple[int, int],
    perimeter_max: Optional[int]=None,
    ps: Optional[PrimeSPFsieve]=None,
) -> Generator[Tuple[float, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]], None, None]:

    # Review- look into the possibility mentioned on the Project Euler
    # forum that if one vertex, the circumcentre and the orthocentre
    # are known then the two other vertices can be found.

    if perimeter_max is None:
        perimeter_max = float("inf")
    #x0 = orthocentre_x
    
    orthocentre_dist_sq = orthocentre[0] ** 2 + orthocentre[1] ** 2
    orthocentre_dist = math.sqrt(orthocentre_dist_sq)

    def calculatePerimeter(pt1: Tuple[int, int], pt2: Tuple[int, int], pt3: Tuple[int, int]) -> float:
        return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) +\
                math.sqrt((pt1[0] - pt3[0]) ** 2 + (pt1[1] - pt3[1]) ** 2) +\
                math.sqrt((pt2[0] - pt3[0]) ** 2 + (pt2[1] - pt3[1]) ** 2)

    def calculatePerimeterLowerBound(r_sq: int) -> float:
        r = math.sqrt(r_sq)
        d = orthocentre_dist
        return math.sqrt(max(0, 3 * r ** 2 + 2 * d * r - d ** 2)) + 2 * math.sqrt(max(0, 3 * r ** 2 - d * r))
    
    


    ub = None #float("inf")
    #ps = PrimeSPFsieve()
    if perimeter_max is not None:
        lo, hi = 1, 1
        while calculatePerimeterLowerBound(hi) <= perimeter_max:
            lo = hi + 1
            hi <<= 1
        while lo < hi:
            mid = hi - ((hi - lo) >> 1)
            if calculatePerimeterLowerBound(mid) <= perimeter_max:
                lo = mid
            else: hi = mid - 1
        ub = lo
        #print(f"radius squared upper bound = {lo}")
    #else:
    #    rad_sq_iter = itertools.count(1)
    rad_sq_step = 1
    
    if not orthocentre[1]:
        if not orthocentre[0]:
            # The orthocentre and circumcentre only coincide when the
            # triangle is equilateral- there are no equilateral triangles
            # whose vertices are all on lattice points in a square lattice
            return
        rad_sq_step = orthocentre[0]
    elif not orthocentre[0]:
        rad_sq_step = orthocentre[1]
    
    x_axis_sym = not orthocentre[1]
    y_axis_sym = not orthocentre[0]
    
    rad_sq_iter = itertools.count(rad_sq_step, step=rad_sq_step) if ub is None else range(rad_sq_step, ub + 1, rad_sq_step)
    if ub is not None:
        print(f"radius squared upper bound = {ub}")
        if ps is not None: ps.extendSieve(ub)
    
    #perim_chk_rad_sq_cutoff = ((perimeter_max ** 2 - 1) // 27) + 1 if perimeter_max is not None else 0
    #x_axis_sym = False
    #y_axis_sym = False
    ref = None

    h = []
    for rad_sq in rad_sq_iter:
        if not rad_sq % 10 ** 6:
            if h:
                perim_lb = calculatePerimeterLowerBound(rad_sq)
                while h and h[0][0] <= perim_lb:
                    yield heapq.heappop(h)
            suff_str = "" if ub is None else f" of {ub}"
            print(f"rad_sq = {rad_sq}{suff_str}")
        seen_lst = []
        seen_dict = {}
        for sq_pair1 in sumOfTwoSquaresSolutionGenerator(rad_sq, ps=ps):
            #if rad_sq == ref:
            #    print(sq_pair1)
            #x_axis_reflect = False
            #y_axis_reflect = False
            if x_axis_sym:# or not sq_pair1[1]:
                #x_axis_reflect = bool(sq_pair1[1])
                pt1_lst = {sq_pair1, (-sq_pair1[0], sq_pair1[1]), (sq_pair1[1], sq_pair1[0]), (-sq_pair1[1], sq_pair1[0])}
            elif y_axis_sym:# or not sq_pair1[1]:
                #y_axis_reflect = bool(sq_pair1[0])
                pt1_lst = {sq_pair1, (sq_pair1[0], -sq_pair1[1]), (sq_pair1[1], sq_pair1[0]), (sq_pair1[1], -sq_pair1[0])}
            else:
                pt1_lst = {sq_pair1, (sq_pair1[0], -sq_pair1[1]), (-sq_pair1[0], sq_pair1[1]), (-sq_pair1[0], -sq_pair1[1]),\
                            (sq_pair1[1], sq_pair1[0]), (sq_pair1[1], -sq_pair1[0]), (-sq_pair1[1], sq_pair1[0]), (-sq_pair1[1], -sq_pair1[0])}
            pt1_lst = sorted(pt1_lst)
            #if rad_sq == ref:
            #    print(pt1_lst, x_axis_reflect, y_axis_reflect)
            #print(pt1_lst)
            for i, sq_pair2 in enumerate(seen_lst):
                for pt1 in pt1_lst:
                    pt2_lst = sorted({sq_pair2, (sq_pair2[0], -sq_pair2[1]), (-sq_pair2[0], sq_pair2[1]), (-sq_pair2[0], -sq_pair2[1]),\
                                (sq_pair2[1], sq_pair2[0]), (sq_pair2[1], -sq_pair2[0]), (-sq_pair2[1], sq_pair2[0]), (-sq_pair2[1], -sq_pair2[0])})
                    #if rad_sq == ref:
                    #    print(pt1, pt2_lst)
                    for pt2 in pt2_lst:
                        pt3 = (orthocentre[0] - pt1[0] - pt2[0], orthocentre[1] - pt1[1] - pt2[1])
                        #print(pt1, pt2, pt3)
                        j = seen_dict.get(tuple(sorted([abs(x) for x in pt3])), float("inf"))
                        #if rad_sq == ref: print(pt1, pt2, pt3, i, j, sq_pair1)
                        if j > i or (j == i and pt3 >= pt2): continue
                        perim = calculatePerimeter(pt1, pt2, pt3)
                        if perim > perimeter_max: continue
                        heapq.heappush(h, (perim, (pt1, pt2, pt3)))
                        if x_axis_sym and pt1[1] and pt2[1] and pt3[1]:#and (pt2[0], -pt2[1]) != pt1 and (pt3[0], -pt3[1]) not in {pt1, pt2}:
                            heapq.heappush(h, (perim, ((pt1[0], -pt1[1]), (pt2[0], -pt2[1]), (pt3[0], -pt3[1]))))
                        if y_axis_sym and pt1[0] and pt2[0] and pt3[0]:# and (-pt2[0], pt2[1]) != pt1 and (-pt3[0], pt3[1]) not in {pt1, pt2}:
                            heapq.heappush(h, (perim, ((-pt1[0], pt1[1]), (-pt2[0], pt2[1]), (-pt3[0], pt3[1]))))
            tup = tuple(sorted([abs(x) for x in sq_pair1]))
            i = len(seen_lst)
            seen_dict[tup] = i
            seen_lst.append(tup)
            #pt2_lst = pt1_lst if not x_axis_sym and not y_axis_sym else sorted({sq_pair1, (sq_pair1[0], -sq_pair1[1]), (-sq_pair1[0], sq_pair1[1]), (-sq_pair1[0], -sq_pair1[1]),\
            #                (sq_pair1[1], sq_pair1[0]), (sq_pair1[1], -sq_pair1[0]), (-sq_pair1[1], sq_pair1[0]), (-sq_pair1[1], -sq_pair1[0])})
            #print(pt2_lst)
            for idx1 in range(1, len(pt1_lst)):
                pt1 = pt1_lst[idx1]
                for idx2 in range(idx1):
                    
                    pt2_0 = pt1_lst[idx2]
                    if x_axis_sym: pt2_set = {pt2_0, (pt2_0[0], -pt2_0[1])}
                    elif y_axis_sym: pt2_set = {pt2_0, (-pt2_0[0], pt2_0[1])}
                    else: pt2_set = {pt2_0}
                    #if rad_sq == ref:
                    #    print(pt1, pt2_set)
                    for pt2 in pt2_set:
                        pt3 = (orthocentre[0] - pt1[0] - pt2[0], orthocentre[1] - pt1[1] - pt2[1])
                        j = seen_dict.get(tuple(sorted([abs(pt3[0]), abs(pt3[1])])), float("inf"))
                        #print(pt1, pt2, pt3, j)
                        if j > i or (j == i and pt3 >= pt2): continue
                        perim = calculatePerimeter(pt1, pt2, pt3)
                        if perim > perimeter_max: continue
                        heapq.heappush(h, (perim, (pt1, pt2, pt3)))
                        if x_axis_sym and pt1[1] and pt2[1] and pt3[1]:# and (pt2[0], -pt2[1]) != pt1 and (pt3[0], -pt3[1]) not in {pt1, pt2}:
                            heapq.heappush(h, (perim, ((pt1[0], -pt1[1]), (pt2[0], -pt2[1]), (pt3[0], -pt3[1]))))
                        if y_axis_sym and pt1[0] and pt2[0] and pt3[0]:# and (-pt2[0], pt2[1]) != pt1 and (-pt3[0], pt3[1]) not in {pt1, pt2}:
                            heapq.heappush(h, (perim, ((-pt1[0], pt1[1]), (-pt2[0], pt2[1]), (-pt3[0], pt3[1]))))
                        
    while h:
        yield heapq.heappop(h)
    return

    """
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
    """
    while h:
        yield heapq.heappop(h)
    return

def trianglesWithLatticePointVerticesAndFixedCircumcentreAndOrthocentrePerimeterSum(
    orthocentre: Tuple[int, int]=(5, 0),
    perimeter_max: int=10 ** 5,
    ps: Optional[PrimeSPFsieve]=None,
) -> float:
    """
    Solution to Project Euler #264
    """
    res = 0
    for tup in trianglesWithLatticePointVerticesAndFixedCircumcentreAndOrthocentreByPerimeterGenerator(
        orthocentre,
        perimeter_max=perimeter_max,
        ps=ps,
    ):
        
        res += tup[0]
        print(f"solution found: {tup}, current total = {res}")
    return res

def trianglesWithLatticePointVerticesAndFixedCircumcentreAndOrthocentrePerimeterSum2(
    orthocentre: Tuple[int, int]=(5, 0),
    perimeter_max: int=10 ** 5,
) -> float:
    # Rotate around origin a multiple of pi / 2 radians and/or reflect through the x- or y-axis
    # the orthocentre to be in the first quadrant with an angle no greater than pi / 4 with the x axis
    # These transformations only alter the orientation of the possible triangles, leaving the
    # total perimeter unchanged
    orthocentre = tuple(sorted([abs(x) for x in orthocentre], reverse=True))
    centroid = tuple(CustomFraction(x, 3) for x in orthocentre)

    if not orthocentre[0]:
        # Any triangle whose orthocentre and circumcentre coincide is an equilateral, and
        # it is not possible for the vertices of an equilateral triangle to all lie on
        # lattice points for a square lattice
        return 0

    # Checking whether the Euler line is on the x-axis, in which case this symmetry
    # may be used to reduce the number of calculations
    x_sym = not (orthocentre[1])

    def calculateOtherVerticesIfLatticePoints(
        v1: Tuple[int, int],
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        r_sq = sum(x * x for x in v1)
        vec = tuple((x - y) / 2 for x, y in zip(centroid, v1))
        bisector = tuple(x + y for x, y in zip(centroid, vec))
        if bisector[0].denominator > 2 or bisector[1].denominator > 2:
            return None
        d_sq = sum(x * x for x in bisector)
        a_hlf_sq = r_sq - d_sq
        if a_hlf_sq <= 0: return None
        sin_sq = vec[1] * vec[1] / sum(x * x for x in vec)
        a_hlf_x_sq = a_hlf_sq * sin_sq
        #print(v1, vec, bisector, r_sq, d_sq, a_hlf_x_sq, sin_sq)
        a_hlf_x = CustomFraction(isqrt(a_hlf_x_sq.numerator), isqrt(a_hlf_x_sq.denominator))
        if a_hlf_x * a_hlf_x != a_hlf_x_sq or a_hlf_x.denominator != bisector[0].denominator:
            return None
        a_hlf_y_sq = a_hlf_sq - a_hlf_x_sq
        a_hlf_y = CustomFraction(isqrt(a_hlf_y_sq.numerator), isqrt(a_hlf_y_sq.denominator))
        if a_hlf_y * a_hlf_y != a_hlf_y_sq or a_hlf_y.denominator != bisector[1].denominator:
            return None
        if (vec[0] >= 0) == (vec[1] >= 0):
            v2 = (bisector[0] + a_hlf_x, bisector[1] - a_hlf_y)
            #if v2[0].denominator != 1 or v2[1].denominator != 1:
            #    return None
            v3 = (bisector[0] - a_hlf_x, bisector[1] + a_hlf_y)
        else:
            v2 = (bisector[0] + a_hlf_x, bisector[1] + a_hlf_y)
            v3 = (bisector[0] - a_hlf_x, bisector[1] - a_hlf_y)
        return (tuple(x.numerator for x in v2), tuple(x.numerator for x in v3))

    for i in range(10):
        for j in range(10):
            v1 = (i, j)
            ans = calculateOtherVerticesIfLatticePoints(v1)
            if ans is None: continue
            v2, v3 = ans
            print(v1, v2, v3)
    return 0

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
        mult = 1
        for i in range(2, n_dig_max, 2):
            mult *= d
            remain_mx_lsts[d - 1].append(0)
            remain_mn_lsts[d - 1].append(-(base - 1) * mult)
            if i == n_dig_max - 1: continue
            mult *= d
            remain_mx_lsts[d - 1].append((base - 1) * mult)
            remain_mn_lsts[d - 1].append(0)
        #remain_mx_lsts.append(0)
        #remain_mn_lsts.append(0)
        for i in range(1, n_dig_max):
            remain_mx_lsts[d - 1][i] += remain_mx_lsts[d - 1][i - 1]
            remain_mn_lsts[d - 1][i] += remain_mn_lsts[d - 1][i - 1]
    #print(remain_mn_lsts)
    #print(remain_mx_lsts)
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
    
    ref = None#[1]
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
                # All balances that are not tight should be guaranteed to be between 1 and base - 1 inclusive
                # at this point
                res = int(len(st) == 1 and (not tight or 1 <= -curr_bals[0] <= poly_coeffs[0]))
                if neg_roots == ref:
                    print(f"solution for ({idx}, {tuple(curr_bals)}, {tight}) = {res}")
                return res
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
                #print(f"tight2 is True")
                #print(f"mults = {mults}")
                for i in range(n_roots):
                    curr_bals[i] = curr_bals0[i] + poly_coeffs[idx] * mults[i]
                ans = recur(idx - 1, tight=True)
                res += ans
                #print(f"tight2 contributed {ans}")
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

# Problem 275
def countBalancedPolyominoSculptures(
    n_tiles: int=18,
) -> int:
    """
    Solution to Project Euler #275
    """
    # Review- try to make faster
    def centreOfMassXCanBeZero(x_rng: Tuple[int, int], curr_com_x: int, tiles_remain: int) -> bool:
        #print(f"using centreOfMassXCanBeZero() with state = {state}, curr_com_x = {curr_com_x}, tiles_remain = {tiles_remain}")
        if not curr_com_x: return True
        if curr_com_x < 0:
            mx = curr_com_x + x_rng[1] * tiles_remain + ((tiles_remain * (tiles_remain + 1)) >> 1)
            return mx >= 0
        mn = curr_com_x + x_rng[0] * tiles_remain - ((tiles_remain * (tiles_remain + 1)) >> 1)
        return mn <= 0

    # Using Redelmeier's algorithm

    #untried = {(0, 0)}
    curr_incl_or_neighbour = {(0, 0)}
    tot = [0]
    tot_sym = [0]

    def recur(tiles_remain: int, curr_com_x: int, x_rng: Tuple[int, int], n_zero: int, untried: Set[int]) -> None:
        #print(tiles_remain, curr_com_x, x_rng, n_zero, untried, curr_incl_or_neighbour)
        if not x_rng[0] and n_tiles == 2 * tiles_remain + n_zero:
            #print("found symmetric")
            tot_sym[0] += 1
        if not tiles_remain:
            #if not curr_com_x: print("found")
            tot[0] += not curr_com_x
            return
        elif not centreOfMassXCanBeZero(x_rng, curr_com_x, tiles_remain): return
        untried2 = set(untried)
        for p in untried:
            untried2.remove(p)
            new_neighbours = set()
            for p2 in [(p[0] - 1, p[1]), (p[0], p[1] - 1), (p[0] + 1, p[1]), (p[0], p[1] + 1)]:
                if p2[1] < 0 or p2 in curr_incl_or_neighbour: continue
                new_neighbours.add(p2)
                curr_incl_or_neighbour.add(p2)
                untried2.add(p2)
            #print(f"adding {p}")
            if n_tiles - tiles_remain < 5:
                print(f"placing tile {n_tiles - tiles_remain + 1} at {p}")
            recur(tiles_remain - 1, curr_com_x + p[0], (min(x_rng[0], p[0]), max(x_rng[1], p[0])), n_zero + (not p[0]), untried2)
            for p2 in new_neighbours:
                curr_incl_or_neighbour.remove(p2)
                untried2.remove(p2)
            #print(f"removing {p}")
        return
        

    recur(n_tiles, 0, (0, 0), 0, {(0, 0)})
    print(tot[0], tot_sym[0])
    return (tot[0] + tot_sym[0]) >> 1
"""
def countBalancedPolyominoSculptures(
    n_tiles: int=18,
) -> int:

    ref_state = ((1, 1), (3, 3), (0, 2))

    curr = {0: {((1, 1),)}}

    def centreOfMassXCanBeZero(state: Iterable[Tuple[int, int]], curr_com_x: int, tiles_remain: int) -> bool:
        #print(f"using centreOfMassXCanBeZero() with state = {state}, curr_com_x = {curr_com_x}, tiles_remain = {tiles_remain}")
        if not curr_com_x: return True
        idx = (curr_com_x > 0)
        curr_max_extent = max(x[idx].bit_length() - 1 for x in state)
        reach = curr_max_extent * tiles_remain + ((tiles_remain * (tiles_remain + 1)) >> 1)
        #print(f"reach = {reach}")
        return reach >= abs(curr_com_x)
    
    def getStandardisedState(state: Iterable[Tuple[int, int]], curr_com_x: int) -> Tuple[Iterable[Tuple[int, int]], int]:
        if curr_com_x > 0: return (state, curr_com_x)
        state2 = [tuple(x[::-1]) for x in state]
        if curr_com_x < 0: return (state2, -curr_com_x)
        return (min(state, state2), 0)

    def calculateNextMoves(bm_lst: Iterable[int]) -> Dict[int, List[int]]:
        incl_locs = {}
        for i1, bm in enumerate(bm_lst):
            if not bm: continue
            incl_locs.setdefault(i1, set())
            bm2 = bm
            for i2 in range(bm.bit_length()):
                if bm2 & 1:
                    incl_locs[i1].add(i2)
                    if bm2 == 1: break
                bm2 >>= 1
        #print(incl_locs)
        seen_locs = {}
        for i1, i2_set in incl_locs.items():
            for i2 in i2_set:
                for j, p in enumerate([(i1 + 1, i2), (i1, i2 + 1), (i1 - 1, i2), (i1, i2 - 1)]):
                    if p[0] >= 0 and p[1] >= 0:
                        #seen_locs.setdefault(i1 - 1, {})
                        if p[1] in incl_locs.get(p[0], set()): continue
                        seen_locs.setdefault(p[0], {})
                        seen_locs[p[0]].setdefault(p[1], [])
                        seen_locs[p[0]][p[1]].append(j)
        #if state == ref_state: print(seen_locs)
        res = {}
        for i1, i2_dict in seen_locs.items():
            res[i1] = set()
            for i2, src_lst in i2_dict.items():
                #if len(src_lst) > 1 and src_lst != [0, 1]:
                #    continue
                res[i1].add(i2)
            if not res[i1]:
                res.pop(i1)
                continue
            #print(res)
            res[i1] = sorted(res[i1])
        return res

    
    n_tiles_remain = n_tiles - 1
    for n_tiles in range(2, n_tiles + 1):
        print(f"calculating for {n_tiles} tiles, number of starting states = {sum(len(x) for x in curr.values())}")
        n_tiles_remain -= 1
        prev = curr
        curr = {}
        for com_x0, state_set in prev.items():
            #print(state_set)
            for state0 in state_set:
                #states = [state0] if com_x0 else [state0, [tuple(x[::-1]) for x in state0]]
                for state, com_x in [(state0, com_x0), ([tuple(x[::-1]) for x in state0], -com_x0)]:
                    #if state == ref_state: print(f"state = {state}")
                    for i1, i2_lst in calculateNextMoves([x[0] for x in state]).items():
                        #print(i1, i2_lst)
                        for i2 in i2_lst:
                            com_x2 = com_x - i2
                            state2 = list(state)
                            if i1 == len(state2): state2.append((0, 0))
                            bm2 = (1 << i2)
                            state2[i1] = (state2[i1][0] | bm2, state2[i1][1] | (0 if i2 else bm2))
                            #if state == ref_state: print(f"possible new state = {state2}")
                            #if not centreOfMassXCanBeZero(state2, com_x2, n_tiles_remain):
                            #    #if state == ref_state: print(f"centre of mass cannot be zero")
                            #    continue#break
                            state2, com_x2 = getStandardisedState(state2, com_x2)
                            curr.setdefault(com_x2, set())
                            curr[com_x2].add(tuple(state2))
        #print(n_tiles, n_tiles_remain, curr)
        #print(max(curr.keys()))
        print(f"solution for {n_tiles} tiles = {len(curr[0])}")
    return len(curr[0])
"""
    
# Problem 276
def countPrimitiveTriangles(perim_max: int=10 ** 7) -> int:
    """
    Solution to Project Euler #276
    """
    # Using Alcuin's sequence

    ps = PrimeSPFsieve(perim_max)
    print("finished creating prime sieve")
    tot = 0
    res = 0
    nxt_fact = perim_max
    for i in range(3, perim_max + 1):
        mn_fact = ((perim_max) // i) + 1
        if tot:
            for fact in reversed(range(mn_fact, nxt_fact + 1)):
                pf = ps.primeFactorisation(fact)
                if max(pf.values()) > 1: continue
                neg = len(pf) & 1
                #for f in pf.values():
                #    if f & 1: neg = not neg
                #print(fact, i, neg)
                #print(i, fact, -tot if neg else tot)
                res += -tot if neg else tot
        nxt_fact = min(nxt_fact, mn_fact - 1)
        # Alcuin's sequence term
        tot += round(i ** 2 / 12) - (i >> 2) * ((i + 2) >> 2)
    print(f"tot = {tot}")
    return tot + res


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

# Problem 279
def integerSidedTrianglesWithIntegerAngleCount(
    max_perimeter: int=10 ** 8,
    n_degrees_in_circle: int=360,                  
) -> int:
    """
    Solution to Project Euler #279
    """
    # Using Niven's theorem
    # Note there is no triangle with integer sides whose angles are
    # all rational multiples of 2 * pi (the only candidate is the
    # 60, 90, 120 triangle)

    res = 0
    if not n_degrees_in_circle % 3:
        # 2 * pi / 3 angle
        print("counting triangles with 120 degree angle")
        m_max = (-3 + isqrt(1 + 8 * max_perimeter)) >> 2
        ans = 0
        for m in range(2, m_max + 1):
            n_max = min(m - 1, (-3 * m + isqrt(m ** 2 + 4 * max_perimeter)) >> 1)
            n_rng = range(1, n_max + 1) if m & 1 else range(1, n_max + 1, 2)
            for n in n_rng:
                if not (m - n) % 3 or gcd(m, n) > 1: continue
                primitive_perim = 2 * m ** 2 + 3 * m * n + n ** 2
                a, b, c = m ** 2 + m * n + n ** 2, 2 * m * n + n ** 2, m ** 2 - n ** 2
               
                if n == 1 and primitive_perim > max_perimeter:
                    break
                
                ans += max_perimeter // primitive_perim
                #print((a, b, c), primitive_perim, max_perimeter // primitive_perim, res)
            else: continue
            break
        print(f"there are {ans} triangles with integer side length and a 120 degree angle whose perimeter does not exceed {max_perimeter}.")
        res += ans
    if not n_degrees_in_circle & 3:
        # pi / 2 angle
        # Pythagorean triples
        print("counting triangles with 90 degree angle (Pythagorean triples)")
        ans = 0
        m_max = (-2 + isqrt(4 + 8 * max_perimeter)) >> 2
        for m in range(2, m_max + 1):
            n_max = min(m - 1, (max_perimeter // (2 * m)) - m)
            #n_max = m - 1
            for n in range(1 + (m & 1), n_max + 1, 2):
                if gcd(m, n) > 1: continue
                primitive_perim = 2 * m ** 2 + 2 * m * n
                #a, b, c = m ** 2 - n ** 2, 2 * m * n, m ** 2 + n ** 2
                
                if n == (1 + (m & 1)) and primitive_perim > max_perimeter:
                    break
                
                ans += max_perimeter // primitive_perim
                #print((a, b, c), primitive_perim, max_perimeter // primitive_perim, res)
            else: continue
            break
        print(f"there are {ans} triangles with integer side length and a 90 degree angle whose perimeter does not exceed {max_perimeter}.")
        res += ans
    if not n_degrees_in_circle % 6:
        # pi / 3 angle
        # Eisenstein triples
        print("counting triangles with 60 degree angle (Eisenstein triples)")
        ans = 0
        m_max = 3 * isqrt((4 * max_perimeter) // 9)#(-1 + isqrt(9 + 8 * max_perimeter)) >> 2
        #for m in range(2, m_max + 1):
        for m in range(2, m_max + 1):
            #discr = 9 * m ** 2 - 4 * max_perimeter
            n_max = m >> 1
            #if discr >= 0: n_max = min(n_max, (m + isqrt(discr)) >> 1)
            n_rng = range(1, n_max + 1) if m & 1 else range(1, n_max + 1, 2)
            for n in n_rng:
                if gcd(m, n) > 1: continue
                div = 1 if (m + n) % 3 else 3
                primitive_perim = (2 * m ** 2 + m * n - n ** 2) // div
                #a, b, c = (m ** 2 - m * n + n ** 2) // div, (2 * m * n - n ** 2) // div, (m ** 2 - n ** 2) // div
                
                if primitive_perim > max_perimeter:
                    continue
                ans += max_perimeter // primitive_perim
                #print((m, n), div, (a, b, c), primitive_perim, max_perimeter // primitive_perim, res)
            else: continue
            break
        print(f"there are {ans} triangles with integer side length and at least one 60 degree angle whose perimeter does not exceed {max_perimeter}.")
        res += ans
    return res
# Problem 280
def antRandomWalkSimulation(
    n_rows: int,
    n_cols: int,
    start: Tuple[int, int],
) -> int:
    
    res = 0
    pos = start
    seed_bm = (1 << n_cols) - 1
    target = ((1 << n_cols) - 1) << n_cols
    has_seed = False
    while True:
        #print(pos, has_seed, format(seed_bm, "b"))
        pos_opts = []
        if pos[0] > 0:
            pos_opts.append((pos[0] - 1, pos[1]))
        if pos[0] < n_rows - 1:
            pos_opts.append((pos[0] + 1, pos[1]))
        if pos[1] > 0:
            pos_opts.append((pos[0], pos[1] - 1))
        if pos[1] < n_cols - 1:
            pos_opts.append((pos[0], pos[1] + 1))
        pos = random.choice(pos_opts)
        res += 1
        bm2 = (1 << pos[1])
        if not has_seed:
            if pos[0] == 0 and seed_bm & bm2:
                has_seed = True
                seed_bm ^= bm2
        elif pos[0] == n_rows - 1:
            bm2 <<= n_cols
            #print("hi")
            if not seed_bm & bm2:
                #print("hi2")
                has_seed = False
                seed_bm ^= bm2
                if seed_bm == target:
                    break
    return res

def antRandomWalkExpectedNumberOfStepsSimulation(
    n_rows: int,
    n_cols: int,
    start: Tuple[int, int],
    n_sim: int,
) -> Tuple[float, float]:
    mean = 0
    var = 0
    for i in range(n_sim):
        #print(f"simulation {i}")
        num = antRandomWalkSimulation(n_rows, n_cols, start)
        mean += num
        var += num ** 2
    mean /= n_sim
    var /= n_sim
    var -= mean ** 2
    var *= (n_sim - 1) / n_sim
    return (mean, math.sqrt(var * (n_sim - 1)) / n_sim)

def antRandomWalkExpectedNumberOfStepsFraction(
    n_rows: int,
    n_cols: int,
    start: Tuple[int, int],
) -> CustomFraction:
    
    def seedPositionBitmaskGenerator(n_seeds: int) -> Generator[int, None, None]:

        def nextBitmask(bm: int) -> int:
            if not bm: return 0
            lowest_bit = bm & (-bm)
            res = bm + lowest_bit
            lowest_bit2 = res & (-res)
            while lowest_bit:
                lowest_bit >>= 1
                lowest_bit2 >>= 1
            trail_ones = lowest_bit2 - 1
            return res | trail_ones

        curr = (1 << n_seeds) - 1
        ub = 1 << (n_cols << 1)
        print(curr, ub)
        while curr < ub:
            yield curr
            curr = nextBitmask(curr)
        return
    
    n_seed_states = math.comb(n_cols << 1, n_cols - 1) + math.comb(n_cols << 1, n_cols) - 1
    #print(f"n_seed_states = {n_seed_states}")

    m = (n_cols * n_rows) * n_seed_states
    #print(f"m = {m}")

    seed_bm_lst = []
    seed_bm_dict = {}
    #print("hi1")
    for i, s_bm in enumerate(seedPositionBitmaskGenerator(n_cols - 1)):
        seed_bm_lst.append(s_bm)
        seed_bm_dict[s_bm] = i
    #print("hi2")
    for i, (_, s_bm) in enumerate(zip(range(math.comb(n_cols << 1, n_cols) - 1), seedPositionBitmaskGenerator(n_cols)), start=len(seed_bm_lst)):
        seed_bm_lst.append(s_bm)
        seed_bm_dict[s_bm] = i
    #print("hi3")
    def stateEncoding(pos: Tuple[int, int], s_bm: int) -> int:
        return (pos[0] + n_cols * pos[1]) * n_seed_states + seed_bm_dict[s_bm]

    mat = []
    vec = []
    for i in range(m):
        #print(f"i = {i}")
        mat.append([CustomFraction(0, 1) for _ in range(m)])
        vec.append(CustomFraction(0, 1))
    #mat = [[CustomFraction(0, 1) for _ in range(m)] for _ in range(m)]
    #vec = [CustomFraction(0, 1) for _ in range(m)]

    #print("hi4")
    # Lowest row
    idx1 = 0
    #print(f"idx1 = {idx1}")
    idx2 = 0
    bm2 = 1 << idx2
    if n_cols == 1:
        for s_bm in seed_bm_lst:
            i1 = stateEncoding((idx1, idx2), s_bm)
            mat[i1][i1] = CustomFraction(1, 1)
            
            if s_bm.bit_count() == n_cols and s_bm & bm2:
                # Forced to pick up the seed
                i2 = stateEncoding((idx1, idx2), s_bm ^ bm2)
                mat[i1][i2] = CustomFraction(-1, 1)
            else:
                vec[i1] = CustomFraction(1, 1)
                neg_p = CustomFraction(-1, 1)
                for pos2 in [(idx1 + 1, idx2)]:
                    i2 = stateEncoding(pos2, s_bm)
                    mat[i1][i2] = neg_p
    else:
        for s_bm in seed_bm_lst:
            i1 = stateEncoding((idx1, idx2), s_bm)
            mat[i1][i1] = CustomFraction(-1, 1)
            
            if s_bm.bit_count() == n_cols and s_bm & bm2:
                # Forced to pick up the seed
                i2 = stateEncoding((idx1, idx2), s_bm ^ bm2)
                mat[i1][i2] = CustomFraction(-1, 1)
            else:
                vec[i1] = CustomFraction(1, 1)
                neg_p = CustomFraction(-1, 2)
                for pos2 in [(idx1, idx2 + 1), (idx1 + 1, idx2)]:
                    i2 = stateEncoding(pos2, s_bm)
                    mat[i1][i2] = neg_p
        for idx2 in range(1, n_cols - 1):
            bm2 = 1 << idx2
            for s_bm in seed_bm_lst:
                i1 = stateEncoding((idx1, idx2), s_bm)
                mat[i1][i1] = CustomFraction(1, 1)
                if s_bm.bit_count() == n_cols and s_bm & bm2:
                    # Forced to pick up the seed
                    i2 = stateEncoding((idx1, idx2), s_bm ^ bm2)
                    mat[i1][i2] = CustomFraction(-1, 1)
                else:
                    vec[i1] = CustomFraction(1, 1)
                    neg_p = CustomFraction(-1, 3)
                    for pos2 in [(idx1, idx2 + 1), (idx1 + 1, idx2), (idx1, idx2 - 1)]:
                        i2 = stateEncoding(pos2, s_bm)
                        mat[i1][i2] = neg_p
        idx2 = n_cols - 1
        bm2 = 1 << idx2
        for s_bm in seed_bm_lst:
            i1 = stateEncoding((idx1, idx2), s_bm)
            mat[i1][i1] = CustomFraction(1, 1)
            if s_bm.bit_count() == n_cols and s_bm & bm2:
                # Forced to pick up the seed
                i2 = stateEncoding((idx1, idx2), s_bm ^ bm2)
                mat[i1][i2] = CustomFraction(-1, 1)
            else:
                vec[i1] = CustomFraction(1, 1)
                neg_p = CustomFraction(-1, 2)
                for pos2 in [(idx1, idx2 - 1), (idx1 + 1, idx2)]:
                    i2 = stateEncoding(pos2, s_bm)
                    mat[i1][i2] = neg_p

    # Rows that are not the lowest or highest
    for idx1 in range(1, n_rows - 1):
        #print(f"idx1 = {idx1}")
        idx2 = 0
        if n_cols == 1:
            for s_bm in seed_bm_lst:
                i1 = stateEncoding((idx1, idx2), s_bm)
                mat[i1][i1] = CustomFraction(1, 1)
                vec[i1] = CustomFraction(1, 1)
                neg_p = CustomFraction(-1, 2)
                for pos2 in [(idx1 - 1, idx2), (idx1 + 1, idx2)]:
                    i2 = stateEncoding(pos2, s_bm)
                    mat[i1][i2] = neg_p
        else:
            for s_bm in seed_bm_lst:
                i1 = stateEncoding((idx1, idx2), s_bm)
                mat[i1][i1] = CustomFraction(1, 1)
                vec[i1] = CustomFraction(1, 1)
                neg_p = CustomFraction(-1, 3)
                for pos2 in [(idx1 - 1, idx2), (idx1, idx2 + 1), (idx1 + 1, idx2)]:
                    i2 = stateEncoding(pos2, s_bm)
                    mat[i1][i2] = neg_p
            for idx2 in range(1, n_cols - 1):
                for s_bm in seed_bm_lst:
                    i1 = stateEncoding((idx1, idx2), s_bm)
                    mat[i1][i1] = CustomFraction(1, 1)
                    vec[i1] = CustomFraction(1, 1)
                    neg_p = CustomFraction(-1, 4)
                    for pos2 in [(idx1 - 1, idx2), (idx1, idx2 + 1), (idx1 + 1, idx2), (idx1, idx2 - 1)]:
                        i2 = stateEncoding(pos2, s_bm)
                        mat[i1][i2] = neg_p
            idx2 = n_cols - 1
            for s_bm in seed_bm_lst:
                i1 = stateEncoding((idx1, idx2), s_bm)
                mat[i1][i1] = CustomFraction(1, 1)
                vec[i1] = CustomFraction(1, 1)
                neg_p = CustomFraction(-1, 3)
                for pos2 in [(idx1 - 1, idx2), (idx1, idx2 - 1), (idx1 + 1, idx2)]:
                    i2 = stateEncoding(pos2, s_bm)
                    mat[i1][i2] = neg_p

    # Highest row
    target_s_bm = ((1 << n_cols) - 1) << n_cols
    #print(f"target_s_bm = {target_s_bm}")
    idx1 = n_rows - 1
    #print(f"idx1 = {idx1}")
    idx2 = 0
    bm2 = 1 << (idx2 + n_cols)
    if n_cols == 1:
        for s_bm in seed_bm_lst:
            i1 = stateEncoding((idx1, idx2), s_bm)
            mat[i1][i1] = CustomFraction(1, 1)
            if s_bm.bit_count() == n_cols - 1 and not s_bm & bm2:
                # Forced to drop off seed
                if s_bm ^ bm2 != target_s_bm:
                    i2 = stateEncoding((idx1, idx2), s_bm ^ bm2)
                    mat[i1][i2] = CustomFraction(-1, 1)
            else:
                vec[i1] = CustomFraction(1, 1)
                neg_p = CustomFraction(-1, 1)
                for pos2 in [(idx1 - 1, idx2)]:
                    i2 = stateEncoding(pos2, s_bm)
                    mat[i1][i2] = neg_p
    else:
        for s_bm in seed_bm_lst:
            i1 = stateEncoding((idx1, idx2), s_bm)
            mat[i1][i1] = CustomFraction(1, 1)
            if s_bm.bit_count() == n_cols - 1 and not s_bm & bm2:
                # Forced to drop off seed
                if s_bm ^ bm2 != target_s_bm:
                    i2 = stateEncoding((idx1, idx2), s_bm ^ bm2)
                    mat[i1][i2] = CustomFraction(-1, 1)
            else:
                vec[i1] = CustomFraction(1, 1)
                neg_p = CustomFraction(-1, 2)
                for pos2 in [(idx1 - 1, idx2), (idx1, idx2 + 1)]:
                    i2 = stateEncoding(pos2, s_bm)
                    mat[i1][i2] = neg_p
        for idx2 in range(1, n_cols - 1):
            bm2 = 1 << (idx2 + n_cols)
            for s_bm in seed_bm_lst:
                i1 = stateEncoding((idx1, idx2), s_bm)
                mat[i1][i1] = CustomFraction(1, 1)
                if s_bm.bit_count() == n_cols - 1 and not s_bm & bm2:
                    # Forced to drop off seed
                    if s_bm ^ bm2 != target_s_bm:
                        i2 = stateEncoding((idx1, idx2), s_bm ^ bm2)
                        mat[i1][i2] = CustomFraction(-1, 1)
                else:
                    vec[i1] = CustomFraction(1, 1)
                    neg_p = CustomFraction(-1, 3)
                    for pos2 in [(idx1, idx2 + 1), (idx1 - 1, idx2), (idx1, idx2 - 1)]:
                        i2 = stateEncoding(pos2, s_bm)
                        mat[i1][i2] = neg_p
        idx2 = n_cols - 1
        bm2 = 1 << (idx2 + n_cols)
        for s_bm in seed_bm_lst:
            i1 = stateEncoding((idx1, idx2), s_bm)
            mat[i1][i1] = CustomFraction(1, 1)
            #print(f"idx1 = {idx1}, idx2 = {idx2}, s_bm = {format(s_bm, 'b')}, bm2 = {format(bm2, 'b')}, s_bm & bm2 = {s_bm & bm2}")
            if s_bm.bit_count() == n_cols - 1 and not s_bm & bm2:
                
                # Forced to drop off seed
                if s_bm ^ bm2 != target_s_bm:
                    i2 = stateEncoding((idx1, idx2), s_bm ^ bm2)
                    mat[i1][i2] = CustomFraction(-1, 1)
            else:
                vec[i1] = CustomFraction(1, 1)
                neg_p = CustomFraction(-1, 2)
                for pos2 in [(idx1 - 1, idx2), (idx1, idx2 - 1)]:
                    i2 = stateEncoding(pos2, s_bm)
                    mat[i1][i2] = neg_p
    # TODO
    return CustomFraction(0, 1)


def antRandomWalkExpectedNumberOfStepsFloat(
    n_rows: int=5,
    n_cols: int=5,
    start: Tuple[int, int]=(2, 2),
) -> float:
    res = antRandomWalkExpectedNumberOfStepsFraction(
        n_rows,
        n_cols,
        start,
    )
    return res.numerator / res.denominator

def antRandomWalkExpectedNumberOfStepsFloatDirect(
    n_rows: int,
    n_cols: int,
    start: Tuple[int, int],
) -> CustomFraction:
    """
    Solution to Project Euler #280
    """
    # Review- look into the solutions given on the Project Euler forum
    # splitting the matrix based on the positions of the seeds
    def seedPositionBitmaskGenerator(n_seeds: int) -> Generator[int, None, None]:
        if not n_seeds:
            yield 0
            return

        def nextBitmask(bm: int) -> int:
            lowest_bit = bm & (-bm)
            res = bm + lowest_bit
            lowest_bit2 = res & (-res)
            while lowest_bit:
                lowest_bit >>= 1
                lowest_bit2 >>= 1
            trail_ones = lowest_bit2 - 1
            return res | trail_ones

        curr = (1 << n_seeds) - 1
        ub = 1 << (n_cols << 1)
        #print(curr, ub)
        while curr < ub:
            yield curr
            curr = nextBitmask(curr)
        return
    
    n_seed_states = math.comb(n_cols << 1, n_cols - 1) + math.comb(n_cols << 1, n_cols) - 1
    #print(f"n_seed_states = {n_seed_states}")

    m = (n_cols * n_rows) * n_seed_states
    #print(f"m = {m}")

    seed_bm_lst = []
    seed_bm_dict = {}
    #print("hi1")
    for i, s_bm in enumerate(seedPositionBitmaskGenerator(n_cols - 1)):
        seed_bm_lst.append(s_bm)
        seed_bm_dict[s_bm] = i
    #print("hi2")
    for i, (_, s_bm) in enumerate(zip(range(math.comb(n_cols << 1, n_cols) - 1), seedPositionBitmaskGenerator(n_cols)), start=len(seed_bm_lst)):
        seed_bm_lst.append(s_bm)
        seed_bm_dict[s_bm] = i
    #print("hi3")
    #print(seed_bm_lst)
    #print(seed_bm_dict)

    def stateEncoding(pos: Tuple[int, int], s_bm: int) -> int:
        #print(pos, s_bm, format(s_bm, "b"))
        return (pos[0] + n_rows * pos[1]) * n_seed_states + seed_bm_dict[s_bm]

    #mat = []
    #vec = []
    #for i in range(m):
    #    print(f"i = {i}")
    #    mat.append([CustomFraction(0, 1) for _ in range(m)])
    #    vec.append(CustomFraction(0, 1))
    mat = np.zeros((m, m), dtype=float)
    vec = np.zeros((m,), dtype=float)
    #mat = [[CustomFraction(0, 1) for _ in range(m)] for _ in range(m)]
    #vec = [CustomFraction(0, 1) for _ in range(m)]

    #print("hi4")
    # Lowest row
    idx1 = 0
    #print(f"idx1 = {idx1}")
    idx2 = 0
    bm2 = 1 << idx2
    if n_cols == 1:
        for s_bm in seed_bm_lst:
            i1 = stateEncoding((idx1, idx2), s_bm)
            mat[i1][i1] = 1
            
            if s_bm.bit_count() == n_cols and s_bm & bm2:
                # Forced to pick up the seed
                i2 = stateEncoding((idx1, idx2), s_bm ^ bm2)
                mat[i1][i2] = -1
            else:
                vec[i1] = 1
                neg_p = -1
                for pos2 in [(idx1 + 1, idx2)]:
                    i2 = stateEncoding(pos2, s_bm)
                    mat[i1][i2] = neg_p
    else:
        for s_bm in seed_bm_lst:
            i1 = stateEncoding((idx1, idx2), s_bm)
            mat[i1][i1] = 1
            
            if s_bm.bit_count() == n_cols and s_bm & bm2:
                # Forced to pick up the seed
                i2 = stateEncoding((idx1, idx2), s_bm ^ bm2)
                mat[i1][i2] = -1
            else:
                vec[i1] = 1
                neg_p = -1 / 2
                for pos2 in [(idx1, idx2 + 1), (idx1 + 1, idx2)]:
                    i2 = stateEncoding(pos2, s_bm)
                    mat[i1][i2] = neg_p
        for idx2 in range(1, n_cols - 1):
            bm2 = 1 << idx2
            for s_bm in seed_bm_lst:
                i1 = stateEncoding((idx1, idx2), s_bm)
                mat[i1][i1] = 1
                if s_bm.bit_count() == n_cols and s_bm & bm2:
                    # Forced to pick up the seed
                    i2 = stateEncoding((idx1, idx2), s_bm ^ bm2)
                    mat[i1][i2] = -1
                else:
                    vec[i1] = 1
                    neg_p = -1 / 3
                    for pos2 in [(idx1, idx2 + 1), (idx1 + 1, idx2), (idx1, idx2 - 1)]:
                        i2 = stateEncoding(pos2, s_bm)
                        mat[i1][i2] = neg_p
        idx2 = n_cols - 1
        bm2 = 1 << idx2
        for s_bm in seed_bm_lst:
            i1 = stateEncoding((idx1, idx2), s_bm)
            mat[i1][i1] = 1
            if s_bm.bit_count() == n_cols and s_bm & bm2:
                # Forced to pick up the seed
                i2 = stateEncoding((idx1, idx2), s_bm ^ bm2)
                mat[i1][i2] = -1
            else:
                vec[i1] = 1
                neg_p = -1 / 2
                for pos2 in [(idx1, idx2 - 1), (idx1 + 1, idx2)]:
                    i2 = stateEncoding(pos2, s_bm)
                    mat[i1][i2] = neg_p

    # Rows that are not the lowest or highest
    for idx1 in range(1, n_rows - 1):
        #print(f"idx1 = {idx1}")
        idx2 = 0
        if n_cols == 1:
            for s_bm in seed_bm_lst:
                i1 = stateEncoding((idx1, idx2), s_bm)
                mat[i1][i1] = 1
                vec[i1] = 1
                neg_p = -1 / 2
                for pos2 in [(idx1 - 1, idx2), (idx1 + 1, idx2)]:
                    i2 = stateEncoding(pos2, s_bm)
                    mat[i1][i2] = neg_p
        else:
            for s_bm in seed_bm_lst:
                i1 = stateEncoding((idx1, idx2), s_bm)
                mat[i1][i1] = 1
                vec[i1] = 1
                neg_p = -1 / 3
                for pos2 in [(idx1 - 1, idx2), (idx1, idx2 + 1), (idx1 + 1, idx2)]:
                    i2 = stateEncoding(pos2, s_bm)
                    mat[i1][i2] = neg_p
            for idx2 in range(1, n_cols - 1):
                for s_bm in seed_bm_lst:
                    i1 = stateEncoding((idx1, idx2), s_bm)
                    mat[i1][i1] = 1
                    vec[i1] = 1
                    neg_p = -1 / 4
                    for pos2 in [(idx1 - 1, idx2), (idx1, idx2 + 1), (idx1 + 1, idx2), (idx1, idx2 - 1)]:
                        i2 = stateEncoding(pos2, s_bm)
                        mat[i1][i2] = neg_p
            idx2 = n_cols - 1
            for s_bm in seed_bm_lst:
                i1 = stateEncoding((idx1, idx2), s_bm)
                mat[i1][i1] = 1
                vec[i1] = 1
                neg_p = -1 / 3
                for pos2 in [(idx1 - 1, idx2), (idx1, idx2 - 1), (idx1 + 1, idx2)]:
                    i2 = stateEncoding(pos2, s_bm)
                    mat[i1][i2] = neg_p

    # Highest row
    target_s_bm = ((1 << n_cols) - 1) << n_cols
    #print(f"target_s_bm = {target_s_bm}")
    idx1 = n_rows - 1
    #print(f"idx1 = {idx1}")
    idx2 = 0
    bm2 = 1 << (idx2 + n_cols)
    if n_cols == 1:
        for s_bm in seed_bm_lst:
            i1 = stateEncoding((idx1, idx2), s_bm)
            mat[i1][i1] = 1
            if s_bm.bit_count() == n_cols - 1 and not s_bm & bm2:
                # Forced to drop off seed
                if s_bm ^ bm2 != target_s_bm:
                    i2 = stateEncoding((idx1, idx2), s_bm ^ bm2)
                    mat[i1][i2] = -1
            else:
                vec[i1] = 1
                neg_p = -1
                for pos2 in [(idx1 - 1, idx2)]:
                    i2 = stateEncoding(pos2, s_bm)
                    mat[i1][i2] = neg_p
    else:
        for s_bm in seed_bm_lst:
            i1 = stateEncoding((idx1, idx2), s_bm)
            mat[i1][i1] = 1
            if s_bm.bit_count() == n_cols - 1 and not s_bm & bm2:
                # Forced to drop off seed
                if s_bm ^ bm2 != target_s_bm:
                    i2 = stateEncoding((idx1, idx2), s_bm ^ bm2)
                    mat[i1][i2] = -1
            else:
                vec[i1] = 1
                neg_p = -1 / 2
                for pos2 in [(idx1 - 1, idx2), (idx1, idx2 + 1)]:
                    i2 = stateEncoding(pos2, s_bm)
                    mat[i1][i2] = neg_p
        for idx2 in range(1, n_cols - 1):
            bm2 = 1 << (idx2 + n_cols)
            for s_bm in seed_bm_lst:
                i1 = stateEncoding((idx1, idx2), s_bm)
                mat[i1][i1] = 1
                if s_bm.bit_count() == n_cols - 1 and not s_bm & bm2:
                    # Forced to drop off seed
                    if s_bm ^ bm2 != target_s_bm:
                        i2 = stateEncoding((idx1, idx2), s_bm ^ bm2)
                        mat[i1][i2] = -1
                else:
                    vec[i1] = 1
                    neg_p = -1 / 3
                    for pos2 in [(idx1, idx2 + 1), (idx1 - 1, idx2), (idx1, idx2 - 1)]:
                        i2 = stateEncoding(pos2, s_bm)
                        mat[i1][i2] = neg_p
        idx2 = n_cols - 1
        bm2 = 1 << (idx2 + n_cols)
        for s_bm in seed_bm_lst:
            i1 = stateEncoding((idx1, idx2), s_bm)
            mat[i1][i1] = 1
            #print(f"idx1 = {idx1}, idx2 = {idx2}, s_bm = {format(s_bm, 'b')}, bm2 = {format(bm2, 'b')}, s_bm & bm2 = {s_bm & bm2}")
            if s_bm.bit_count() == n_cols - 1 and not s_bm & bm2:
                
                # Forced to drop off seed
                if s_bm ^ bm2 != target_s_bm:
                    i2 = stateEncoding((idx1, idx2), s_bm ^ bm2)
                    mat[i1][i2] = -1
            else:
                vec[i1] = 1
                neg_p = -1 / 2
                for pos2 in [(idx1 - 1, idx2), (idx1, idx2 - 1)]:
                    i2 = stateEncoding(pos2, s_bm)
                    mat[i1][i2] = neg_p
    #print("hi5")
    
    #print(f"cnt = {cnt}")
    #print("matrix:")
    #for row, val in zip(mat, vec): print([float(x) for x in row], float(sum(row)), float(val))
    #print("vector:")
    #print([float(x) for x in vec])
    print(f"finished creating matrix and vector. Solving matrix equation.")
    sol = np.linalg.solve(mat, vec)
    #print("solution:")
    #print([float(x) for x in sol])
    #mat_inv = np.linalg.inv(mat)

    state_lst2 = [None for _ in range(m)]
    cnt = 0
    for idx1 in range(n_rows):
        for idx2 in range(n_cols):
            for s_bm in seed_bm_lst:
                i = stateEncoding((idx1, idx2), s_bm)
                state_lst2[i] = ((idx1, idx2), format(s_bm, "b"))
                cnt += 1
    print(state_lst2)
    print("solution:")
    print([float(x) for x in sol])
    
    start_state_idx = stateEncoding(start, (1 << n_cols) - 1)
    print(max(sol))
    cnts1 = [0, 0, 0]
    cnts2 = [0, 0, 0]
    for num in vec:
        if abs(num) <= 1e-5:
            cnts1[0] += 1
        elif abs(num - 1) <= 1e-5:
            cnts1[1] += 1
        else: cnts1[2] += 1
    for num in np.matmul(mat, sol):
        if abs(num) <= 1e-5:
            cnts2[0] += 1
        elif abs(num - 1) <= 1e-5:
            cnts2[1] += 1
        else: cnts2[2] += 1
    
    
    #print(set(np.matmul(mat, sol)))
    #print(set(vec))
    print(cnts1)
    print(cnts2)
    #print(sol)
    print(all(abs(x - y) <= 1e-5 for x, y in zip(vec, np.matmul(mat, sol))))
    #print("inverse matrix:")
    #for row in mat_inv: print([float(x) for x in row])
    #print(np.linalg.det(mat))

    #print(np.linalg.matmul(mat, sol))

    #print(sol)
    #print(np.linalg.matmul(mat_inv, vec))
    return sol[start_state_idx]

# Problem 281
def countPizzaToppings(n_topping_types: int, n_slices_per_topping: int) -> int:
    # Using Burnside's Lemma
    gcd_cnts = {}
    for n in range(1, n_slices_per_topping + 1):
        g = gcd(n, n_slices_per_topping)
        gcd_cnts[g] = gcd_cnts.get(g, 0) + 1
    
    def multinomialEqual(m: int, k: int) -> int:
        n = m * k
        
        res = math.comb(n, k)
        for n2 in reversed(range(k, n, k)):
            res *= math.comb(n2, k)
        return res

    res = 0
    for g, f in gcd_cnts.items():
        res += multinomialEqual(n_topping_types, g) * f
    return res // (n_topping_types * n_slices_per_topping)

def pizzaToppingsSum(max_count: int=10 ** 15) -> int:
    """
    Solution to Project Euler #281
    """
    res = 0
    for m in itertools.count(2):
        cnt = countPizzaToppings(m, 1)
        if cnt > max_count: break
        res += cnt
        for n in itertools.count(2):
            cnt = countPizzaToppings(m, n)
            if cnt > max_count:
                print(f"for m = {m}, n < {n}")
                break
            res += cnt
    return res

# Problem 282
def ackermannFunctionModuloSum(n_max: int=6, md: int=14 ** 8) -> int:
    """
    Solution to Project Euler #282
    """
    res = 0
    if n_max < 0: return res
    res = (res + 1) % md
    if n_max < 1: return res
    res = (res + 3) % md
    if n_max < 2: return res
    res = (res + 7) % md
    if n_max < 3: return res
    res = (res + (1 << 6) - 3) % md
    if n_max < 4: return res
    print(res)

    

    pow2_md, odd_md = 0, md
    while not odd_md & 1:
        odd_md >>= 1
        pow2_md += 1
    
    pf = calculatePrimeFactorisation(odd_md)
    euler_tot = 1
    for p, f in pf.items():
        euler_tot *= (p - 1) * p ** (f - 1)

    (g, (m1, m2)) = extendedEuclideanAlgorithm(pow2_md, odd_md)

    ack_md = [(pow(2, 4, md) - 3) % md]
    ack_pow2_md = 1 << 16 if pow2_md > 16 else 0
    ack_odd_md = pow(2, 4, euler_tot)
    
    for i in itertools.count(1):
        print(ack_pow2_md, ack_odd_md)
        a1, a2 = ack_pow2_md, pow(2, ack_odd_md, odd_md)
        nxt = ((a1 * m2 * odd_md + a2 * m1 * pow2_md) - 3) % md
        print(nxt)
        if nxt == ack_md[-1]: break
        ack_md.append(nxt)
        if (n_max == 4 and i == 4): break
        if not ack_pow2_md or ack_pow2_md >= pow2_md: ack_pow2_md = 0
        else: ack_pow2_md = 1 << ack_pow2_md
        ack_odd_md = pow(2, ack_odd_md, euler_tot)
    print(res)
    #md2 = len(ack) - i0
    #res = (res + ack[4 if len(ack) > 4 else i0 + (4 - len(ack)) % md2]) % md
    res = (res + ack_md[min(4, len(ack_md) - 1)])
    if n_max < 5: return res
    print(4, len(ack_md), ack_md)
    if (len(ack_md) + 2).bit_length() < 17:
        # Almost certainly the case
        res = (res + (n_max - 4) * ack_md[-1]) % md
    else:
        #TODO
        pass
    return res

# Problem 283
def brahmaguptaHeronianTriangleGenerator(m_max: int) -> Generator[Tuple[Tuple[int, int, int], Tuple[int, int, int], int, CustomFraction], None, None]:
    seen = set()
    for m in range(1, m_max + 1):
        for n in range(1, m + 1):
            g1 = gcd(m, n)
            k_rng = (isqrt((m ** 2 * n - 1) // (2 * m + n)) + 1, isqrt(m * n - 1) + 1)
            for k in range(*k_rng):
                if gcd(g1, k) > 1: continue
                a, b, c = n * (m ** 2 + k ** 2), m * (n ** 2 + k ** 2), (m + n) * (m * n - k ** 2)
                g = gcd(a, gcd(b, c))
                a, b, c = a // g, b // g, c // g
                tup = tuple(sorted([a, b, c]))
                if tup in seen: continue
                seen.add(tup)
                r = CustomFraction(k * (m * n - k * k), 2 * g)
                yield ((m, n, k), (a, b, c), g, r)

def heronianTrianglesWithGivenAreaPerimeterRatio(
    area_perimeter_ratio: int,
) -> List[Tuple[int, int, int]]:
    # Review- try to make faster (look into Markov paper "Heronian triangles
    # whose areas are integer multiples of their perimeters")
    r_sq = area_perimeter_ratio ** 2
    res = []
    for x in range(1, 12 * r_sq + 1):
        for y in range(max(x, ((4 * r_sq) // x) + 1), ((12 * r_sq) // x) + 1):
            numer = 4 * (x + y) * r_sq
            denom = x * y - 4 * r_sq
            z, rem = divmod(numer, denom)
            if z < y: break
            if rem: continue
            #print(x, y, z)
            res.append(tuple(sorted([y + z, x + z, x + y])))
    return res

def heronianTrianglesWithIntegerAreaPerimeterRatioPerimeterSum(
    area_perimeter_ratio_max: int=1000,
) -> int:
    """
    Solution to Project Euler #283
    """
    res = 0
    for ratio in range(1, area_perimeter_ratio_max + 1):
        triangles = heronianTrianglesWithGivenAreaPerimeterRatio(ratio)
        print(f"ratio = {ratio}, number of triangles = {len(triangles)}")
        res += sum(sum(x) for x in triangles)
    return res
    """
    cnts = {}
    for r in range(1, area_perimeter_ratio_max + 1):
        if not r % 10: print(f"r = {r}")
        cnts[r] = 0
        r_sq = r * r
        for m in range(1, isqrt(12 * r_sq) + 1):
            for n in range(max((4 * r_sq) // m + 1, m), (4 * r_sq + isqrt((4 * r_sq) * (4 * r_sq + m * m))) // m + 1):
                if not (4 * r_sq * (m + n)) % (m * n - 4 * r_sq):
                    cnts[r] += 1
                    print((r, m, n), (4 * r_sq * (m + n)), (m * n - 4 * r_sq))
    print(cnts)
    return 0
    """
    """
    # Using Brahmagupta's parametric equation
    res = 0
    m_max = (isqrt(area_perimeter_ratio_max) + 1) ** 2 - 1
    print(f"max m = {m_max}")
    for m in range(2, m_max + 1):
        #ratio_min = (11 * m + 13 - (6 * m + 14) * isqrt(m + 1)) >> 1
        print(f"m = {m}")#, min ratio for this m = {ratio_min}")
        #if ratio_min > area_perimeter_ratio_max:
        #    break
        for n in range(1, m + 1):
            g1 = gcd(m, n)
            k_max = isqrt(m * n - 1)
            k_iter = range(2, k_max + 1, 2) if not m & 1 or not n & 1 else range(1, k_max + 1)
            for k in k_iter:
                if gcd(k, g1) > 1: continue
                print((m, n, k), k * (m * n - k ** 2) / 2)
                a = n * (m ** 2 + k ** 2)
                b = m * (n ** 2 + k ** 2)
                c = (m + n) * (m * n - k ** 2)
                ratio = CustomFraction(k * (m * n - k ** 2), 2)
                g = gcd(a, gcd(b, c))
                ratio /= g
                if ratio.numerator > area_perimeter_ratio_max:
                    continue
                
                perim = (2 * (m * n * (m + n))) // g
                area = (m * n * k * (m + n) * (m * n - k ** 2)) // g ** 2
                print(area, perim, ratio)
                mult = ratio.denominator
                perim *= mult
                area *= mult ** 2
                ratio = ratio.numerator
                #mult = 1
                #if not ratio & 1:
                #    ratio >>= 1
                #    perim <<= 1
                #    mult = 2
                print(mult)
                print(f"found primitive Heronian triangle with integer ratio, a = {(a * mult) // g}, b = {(b * mult) // g}, c = {(c * mult) // g}, perimeter = {perim}, area = {area}, ratio = {ratio}")
                #sq_max = isqrt(area_perimeter_ratio_max // perim)
                #res += perim * ((sq_max * (sq_max + 1) * (2 * sq_max + 1)) // 6)
                scale_mx = area_perimeter_ratio_max // ratio
                res += perim * ((scale_mx * (scale_mx + 1)) >> 1)
    return res
    """

# Problem 284
def steadySquaresDigitSum(max_n_digs: int, base: int=10) -> int:

    curr = [(0, 0, 0)]
    res = 1 # The only steady square ending in 1 is 1 itself (review- prove this)
    prev = curr
    curr = []
    num0, num0_sq, dig_sum0 = 0, 0, 0
    for d in range(2, base):
        num = d
        num_sq = d ** 2
        #print(m, num, num_sq)
        if num_sq % base != num:
            continue
        curr.append((num, num_sq, d))
        #print(num, num_sq)
        res += dig_sum0 + d
    mult1 = base
    md = base ** 2
    mult2 = base ** 2
    for m in range(1, max_n_digs):
        if not m % 100: print(f"m = {m} of {max_n_digs}, number of options = {len(curr)}")
        prev = curr
        curr = []
        for num0, num0_sq, dig_sum0 in prev:
            if num0_sq % md == num0:
                curr.append((num0, num0_sq, dig_sum0))
            for d in range(1, base):
                num = d * mult1 + num0
                num_sq_tail = ((((2 * d * (num0 % base)) % base) * mult1) + (num0_sq % md)) % md
                #print(num0, num0_sq, d, num, num_sq_tail)
                if num_sq_tail != num: continue
                num_sq = d ** 2 * mult2 + 2 * d * num0 * mult1 + num0_sq
                #print(m, num, num_sq)
                #if num_sq % (mult1 * base) != num:
                #    continue
                curr.append((num, num_sq, dig_sum0 + d))
                #print(num, num_sq)
                res += dig_sum0 + d
        #print(m, curr)
        mult1 *= base
        mult2 *= base ** 2
        md *= base
    return res

def steadySquaresDigitSumBaseRepr(max_n_digs: int=10 ** 4, base: int=14) -> str:
    """
    Solution to Project Euler #284
    """
    num = steadySquaresDigitSum(max_n_digs=max_n_digs, base=base)
    print(num)
    if not num: return "0"
    res = []
    ord_a = ord("a")
    while num:
        num, d = divmod(num, base)
        if d < 10: res.append(str(d))
        else: res.append(chr(ord_a + d - 10))
    return "".join(res[::-1])

# Problem 985
def calculatePythagoreanOddsGameExpectedValue(k: int) -> float:
    #print(f"k = {k}")
    k2 = k + .5
    sin_theta2 = 1 / k2
    theta2 = math.asin(sin_theta2)
    seg2_area = k2 * k2 * (math.pi / 4 - theta2)
    cos_theta2 = math.sqrt(1 - sin_theta2 * sin_theta2)
    #d2 = (1 / cos_theta2)
    if k == 1:
        #print(f"seg2_area = {seg2_area}, double triangle area = {(k2 * cos_theta2 - 1)}")
        res = seg2_area - (k2 * cos_theta2 - 1)
        return res / k

    k1 = k - .5
    sin_theta1 = 1 / k1
    theta1 = math.asin(sin_theta1)
    seg1_area = k1 * k1 * (math.pi / 4 - theta1)
    cos_theta1 = math.sqrt(1 - sin_theta1 * sin_theta1)
    res = seg2_area - seg1_area - (k2 * cos_theta2 - k1 * cos_theta1)
    return res / k
    """
    n_mx = (k + 1) ** 2 + 1
    res = 2
    #for n in range(1, min(n_mx, 2) + 1):
    #    rt1 = math.sqrt(n - .5)
    #    res += math.asin((k + 1) / rt1) - math.asin(1 / rt1) - 1 / k
    for n in range(3, (k + 1) ** 2 + 1):
        rt1 = math.sqrt(n - .5)
        rt2 = math.sqrt(n - 1.5)
        ans = math.pi / 4 - math.asin(rt2 / rt1) - rt2 / (n - .5) + (n - 1.5 - rt1 * rt2) / k ** 2 + 1 - rt1 / k + 1 / k
        print(f"k = {k}, n = {n}, P(>= n) = {ans}")
        res += ans#math.asin((k + 1) / rt1) - math.asin(rt2 / rt1) - rt2 / k + (n - 1.5) / k ** 2
    num1 = (k + 1) ** 2 + .5
    rt1 = math.sqrt(num1)
    num2 = num1 - 1
    rt2 = math.sqrt(num2)
    ans = .5 * (math.asin((k + 1) / rt1) + (k + 1) / (math.sqrt(2) * rt1) - math.asin(rt2 / rt1) - rt2 / num1)
    print(f"k = {k}, n = {(k + 1) ** 2 + 1}, P(>= n) = {ans}")
    res += ans
    return res
    """

def calculatePythagoreanOddsRangeGameExpectedValue(k_min: int=1, k_max: int=10 ** 5) -> float:
    """
    Solution to Project Euler #285
    """
    res = 0
    for k in range(k_min, k_max + 1):
        if not k % 1000:
            print(f"k = {k} of {k_max}")
        exp = calculatePythagoreanOddsGameExpectedValue(k)
        #print(f"k = {k}, expected value = {exp}")
        res += exp
    return res

# Problem 286
def exactBasketballScoreProbability(
    p: float=0.02,
    d_min: int=1,
    d_max: int=50,
    total_score: int=20,
    eps: float=1e-12,
) -> float:
    """
    Solution to Project Euler #286
    """
    if total_score > d_max - d_min + 1 or total_score < 0: return 0

    def calculateScoreProbability(q: float, score: int) -> float:

        row = [d_min / q, 1 - d_min / q]
        #print(row)
        for d in range(d_min + 1, d_max + 1):
            prev = row
            mult1 = d / q
            mult2 = 1 - mult1
            row = [prev[0] * mult1]
            #print(f"mult1 = {mult1}, mult2 = {mult2}")
            #print(min(len(prev), score + 1))
            for j in range(1, min(len(prev), score + 1)):
                #print(j, row)
                row.append(prev[j] * mult1 + prev[j - 1] * mult2)
            #print(row)
            if len(prev) <= score:
                row.append(prev[-1] * mult2)
            #print(row)
        #print(score, len(row))
        return row[score]

    lo, hi = d_max, d_max
    p_hi = calculateScoreProbability(hi, total_score)
    #print(hi, p_hi)
    lo, hi = hi, hi << 1
    if p_hi == p: return p_hi
    elif p_hi < p:
        while calculateScoreProbability(hi, total_score) < p:
            lo, hi = hi, hi << 1
        while 2 * (hi - lo) > eps:
            mid = lo + (hi - lo) * .5
            if calculateScoreProbability(mid, total_score) > p:
                hi = mid
            else: lo = mid
        return lo + (hi - lo) * .5
    while calculateScoreProbability(hi, total_score) > p:
        lo, hi = hi, hi << 1
    while 2 * (hi - lo) > eps:
        mid = lo + (hi - lo) * .5
        if calculateScoreProbability(mid, total_score) < p:
            hi = mid
        else: lo = mid
    return lo + (hi - lo) * .5

    """
    memo = {}
    def recur(d: int, pts_remain: int) -> List[int]:
        if d == d_max + 1:
            return [1]
        args = (d, pts_remain)
        if args in memo.keys(): return memo[args]
        res = []
        if pts_remain < d_max - d + 1:
            res = [0] + [x * d for x in recur(d + 1, pts_remain)]
        if pts_remain:
            ans = recur(d + 1, pts_remain - 1)
            res += [0] * (len(ans) + 1 - len(res))
            res[0] += ans[0]
            for i in range(1, len(ans)):
                res[i] += ans[i] - ans[i - 1] * d
            res[len(ans)] -= ans[-1] * d

        memo[args] = res
        return res
    
    poly_coeffs = recur(d_min, total_score)
    diff_poly_coeffs = [0, 0] + [-i * poly_coeffs[i] for i in range(1, len(poly_coeffs))]
    print(f"poly_coeffs = {poly_coeffs}")
    print(f"diff_poly_coeffs = {diff_poly_coeffs}")

    def calculateReciprocalPolynomial(q: Union[CustomFraction, float, mpfr], poly_coeffs: List[int]) -> Union[CustomFraction, float, mpfr]:
        res = mpfr(0, precision=10000)
        inv_q = 1 / mpfr(q, precision=10000)
        for num in reversed(poly_coeffs):
            res = res * inv_q + mpfr(num, precision=10000)
        return res
        #return sum(x / q ** i for i, x in enumerate(poly_coeffs))

    #print(memo)

    q0 = mpfr(d_max, precision=-2 * math.ceil(math.log(eps, 10)))
    while True:
        # Newton-Raphson method
        print(f"q0 = {q0}")
        q1 = -float("inf")
        f1 = -float("inf")
        q2 = q0
        f2 = calculateReciprocalPolynomial(q2, poly_coeffs) - p #sum(x / q2 ** i for i, x in enumerate(poly_coeffs)) - p
        mx_n_iter = 50
        i = 0
        while 2 * abs(q2 - q1) > eps or ((f1 and f2) and (f1 > 0) != (f2 < 0)):
            i += 1
            if i > mx_n_iter and (not (f1 and f2) or (f1 > 0) == (f2 < 0)): break
            print(f"q2 = {q2}")
            q1 = q2
            f1 = f2
            f_diff = calculateReciprocalPolynomial(q1, diff_poly_coeffs)
            print(f1, f_diff, f1 / f_diff)
            q2 = q1 - f1 / f_diff
            f2 = calculateReciprocalPolynomial(q2, poly_coeffs) - p #sum(x / q2 ** i for i, x in enumerate(poly_coeffs)) - p
        else:
            if q2 > d_max: break
            q0 <<= 1
            continue
        (lo, f1), (hi, f2) = sorted([(q1, f1), (q2, f2)])
        print(f"lo = {lo} with value {f1 + p}, hi = {hi} with value {f2 + p}")
        if f2 > f1:
            while hi - lo > eps:
                mid = lo + (hi - lo) * .5
                ans =  calculateReciprocalPolynomial(mid, poly_coeffs) #sum(x / mid ** i for i, x in enumerate(poly_coeffs))
                print(mid, ans)
                if ans > p:
                    hi = mid
                else: lo = mid
        else:
            while hi - lo > eps:
                mid = lo + (hi - lo) * .5
                ans = calculateReciprocalPolynomial(mid, poly_coeffs) #sum(x / mid ** i for i, x in enumerate(poly_coeffs))
                print(mid, ans)
                if ans > p:
                    lo = mid
                else: hi = mid
        q2 = lo + (hi - lo) *.5
        break

    #for q in [52.6494571948, 52.6494571949, 52.6494571950, 52.6494571951, 52.6494571952, 52.6494571953, 52.6494571954]:
    for num in range(470, 550):
        q = mpfr(52.64945719, precision=10000) + mpfr(num * 10 ** -11, precision=10000)
        p2 = calculateReciprocalPolynomial(q, poly_coeffs)
        print(f"q = {float(q)}, prob = {p2}, ge = {p2 >= p} diff = {float(calculateReciprocalPolynomial(q, diff_poly_coeffs))}")
    print(sum(x / q2 ** i for i, x in enumerate(poly_coeffs)), calculateReciprocalPolynomial(q2, poly_coeffs))
    return float(q2)
    """

# Problem 287
def singleCircleQuadTreeEncodingMinimumLength(
    image_size_pow2: int=24,
    circle_centre: Tuple[int, int]=(1 << 23, 1 << 23),
    circle_radius_sq: int=1 << 46,
) -> int:
    """
    Solution to Project Euler #287
    """
    # Review- try to make faster. Consider optimising for
    # the specific problem case (where the circle is central and
    # there is mirror symmetry about the line x = y)
    side_len_tots = [0] * (image_size_pow2 + 1)
    circle_rad_floor = isqrt(circle_radius_sq)
    def encMinLen(topleft: Tuple[int, int], side_len_pow2: int) -> int:
        if not side_len_pow2:
            side_len_tots[0] += 1
            if not side_len_tots[0] % 10 ** 5:
                n_pixels_encoded = sum((1 << (2 * i)) * f for i, f in enumerate(side_len_tots))
                print(f"encoding size counts = {side_len_tots}, pixels encoded = {n_pixels_encoded} of {1 << (2 * image_size_pow2)} (proportion = {n_pixels_encoded / (1 << (2 * image_size_pow2))})")
            return 2
        side_len = 1 << side_len_pow2
        rngs = [(x, x + side_len - 1) for x in topleft]
        #print(topleft, side_len, rngs, circle_centre)
        vals = set()
        for i1 in rngs[0]:
            for i2 in rngs[1]:
                b = (i1 - circle_centre[0]) ** 2 + (i2 - circle_centre[1]) ** 2 <= circle_radius_sq
                vals.add(b)
                if len(vals) > 1: break
            else: continue
            break
        else:
            #while len(side_len_tots) <= side_len_pow2:
            #    side_len_tots.append(0)
            if vals == {True}:
                # Whole square is black
                side_len_tots[side_len_pow2] += 1
                return 2
            #if not rngs[0][0] and not rngs[1][0]:
            #    print(rngs, circle_centre, rngs[0][1] <= circle_centre[0], rngs[1][1] >= circle_centre[1], rngs[1][1] <= circle_centre[1], (rngs[0][1] - circle_centre[0]) ** 2 > circle_radius_sq)
            if (rngs[0][0] >= circle_centre[0] and (rngs[1][0] >= circle_centre[1] or rngs[1][1] <= circle_centre[1] or (rngs[0][0] - circle_centre[0]) > circle_rad_floor)) or\
                    (rngs[0][1] <= circle_centre[0] and (rngs[1][0] >= circle_centre[1] or rngs[1][1] <= circle_centre[1] or (circle_centre[0] - rngs[0][1]) > circle_rad_floor)) or\
                    (rngs[1][0] >= circle_centre[1] and (rngs[1][0] - circle_centre[1]) > circle_rad_floor) or\
                    (rngs[1][1] <= circle_centre[1] and (circle_centre[1] - rngs[1][1]) > circle_rad_floor):
                # Whole square is white
                side_len_tots[side_len_pow2] += 1
                return 2

        #if ((rngs[0][0] <= circle_centre[0]) == (rngs[0][1] <= circle_centre[0])) and\
        #        ((rngs[1][0] <= circle_centre[1]) == (rngs[1][1] <= circle_centre[1])):
            
        side_len2_pow2 = side_len_pow2 - 1
        side_len2 = side_len >> 1
        coord_starts = [(x, x + side_len2) for x in topleft]
        res = 1
        for i1 in coord_starts[0]:
            for i2 in coord_starts[1]:
                res += encMinLen((i1, i2), side_len2_pow2)
        return res

    res = encMinLen((0, 0), image_size_pow2)
    n_pixels_encoded = sum((1 << (2 * i)) * f for i, f in enumerate(side_len_tots))
    print(f"encoding size counts = {side_len_tots}, pixels encoded = {n_pixels_encoded} of {1 << (2 * image_size_pow2)} (proportion = {n_pixels_encoded / (1 << (2 * image_size_pow2))})")
    return res


# Problem 288
def factorialPrimeFactorCount(n: int, p: int) -> int:
    n2 = n // p
    res = 0
    while n2:
        res += n2
        n2 //= p
    return res

def squareModSeriesFactorialPrimeFactorCount(
    p: int=61,
    s0: int=290797,
    s_md: int=50515093,
    series_max_pow: int=10 ** 7,
    res_md_pow: Optional[int]=10,
) -> int:
    """
    Solution to Project Euler #288
    """
    res_md = None if res_md_pow is None else p ** res_md_pow
    addMod = (lambda x, y: x + y) if res_md is None else (lambda x, y: (x + y) % res_md)

    s = s0
    res = 0
    if res_md_pow is None: res_md_pow = float("inf")
    
    iter_mx = series_max_pow#min(res_md_pow, series_max_pow) if res_md_pow is not None else series_max_pow
    mult = 1
    for i in range(1, iter_mx + 1):
        s = (s * s) % s_md
        t = s % p
        #print(i, t, mult, mult * t, (mult * t) % res_md, res)
        res = addMod(res, mult * t)
        if i < res_md_pow:
            mult = addMod(mult, p ** (i))
        
        
    return res

# Problem 289
def circleArrayEulerianNonCrossingCycleCountBruteForce(
    n_rows: int=6,
    n_cols: int=10,
    res_md: Optional[int]=10 ** 10,
) -> int:
    if n_rows == 1 and n_cols == 1: return 1
    connect_combs_interior = [
        (1, 0, 3, 2, 5, 4, 7, 6),
        (1, 0, 3, 2, 7, 6, 5, 4),
        (1, 0, 5, 4, 3, 2, 7, 6),
        (1, 0, 7, 4, 3, 6, 5, 2),
        (1, 0, 7, 6, 5, 4, 3, 2),
        (3, 2, 1, 0, 5, 4, 7, 6),
        (3, 2, 1, 0, 7, 6, 5, 4),
        (5, 2, 1, 4, 3, 0, 7, 6),
        (5, 4, 3, 2, 1, 0, 7, 6),
        (7, 2, 1, 4, 3, 6, 5, 0),
        (7, 2, 1, 6, 5, 4, 3, 0),
        (7, 4, 3, 2, 1, 6, 5, 0),
        (7, 6, 3, 2, 5, 4, 1, 0),
        (7, 6, 5, 4, 3, 2, 1, 0),
    ]
    connect_combs_left = [
        (None, 4, None, None, 1, None, 7, 6),
        (None, 7, None, None, 6, None, 4, 1),
    ]
    connect_combs_upper = [
        (None, None, 4, None, 2, 7, None, 5),
        (None, None, 7, None, 5, 4, None, 2),
    ]
    connect_combs_right = [
        (2, None, 0, 5, None, 3, None, None),
        (5, None, 3, 2, None, 0, None, None),
    ]
    connect_combs_bottom = [
        (1, 0, None, 6, None, None, 3, None),
        (6, 3, None, 1, None, None, 0, None),
    ]
    connect_combs_ul = [(None, None, None, None, 7, None, None, 4)]
    connect_combs_ur = [(None, None, 5, None, None, 2, None, None)]
    connect_combs_bl = [(None, 6, None, None, None, None, 1, None)]
    connect_combs_br = [(3, None, None, 0, None, None, None, None)]
    links = [[(-1, 0), 5], [(-1, 0), 4], [(0, -1), 7], [(0, -1), 6], [(1, 0), 1], [(1, 0), 0], [(0, 1), 3], [(0, 1), 2]]

    n_circles = n_rows * n_cols
    n_edges = n_circles * 4

    def isNonCrossingEulerianCircuit(arr: List[List[int]]) -> bool:
        

        cnt = 0
        start = [(0, 0), 7]
        nextEdge = lambda curr: [tuple(x + y for x, y in zip(curr[0], links[curr[1]][0])), links[curr[1]][1]]

        curr = start
        cnt = 0
        nxt_lst = []
        #seen = set()
        first = True
        while first or curr != start:
            first = False
            #if tuple(curr) in seen:
            #    break
            #seen.add(tuple(curr))
            curr = nextEdge(curr)
            if not curr[0][0]:
                if not curr[0][1]:
                    nxt_lst = connect_combs_ul
                elif curr[0][1] == n_rows:
                    nxt_lst = connect_combs_ur
                else: nxt_lst = connect_combs_upper
            elif curr[0][0] == n_cols:
                if not curr[0][1]:
                    nxt_lst = connect_combs_bl
                elif curr[0][1] == n_rows:
                    nxt_lst = connect_combs_br
                else: nxt_lst = connect_combs_bottom
            else:
                if not curr[0][1]:
                    nxt_lst = connect_combs_left
                elif curr[0][1] == n_rows:
                    nxt_lst = connect_combs_right
                else: nxt_lst = connect_combs_interior
            #print(curr, nxt_lst)
            #print(arr[curr[0][0]][curr[0][1]])
            curr[1] = nxt_lst[arr[curr[0][0]][curr[0][1]]][curr[1]]
            cnt += 1
        #print(arr, cnt, n_edges)
        return cnt == n_edges
        

    curr = [[0] * (n_rows + 1) for _ in range(n_cols + 1)]
    res = [0]

    def recur(i1: int, i2: int) -> None:
        if i2 == n_rows + 1:
            if i1 == n_cols:
                if isNonCrossingEulerianCircuit(curr):
                    res[0] += 1
                    if res_md is not None: res[0] %= res_md
                    print(f"found solution, current count = {res[0]}")
                return
            i1 += 1
            i2 = 0
        #print(i1, i2)
        is_at_vertical_edge = (i1 == 0 or i1 == n_cols)
        is_at_horizontal_edge = (i2 == 0 or i2 == n_rows)
        #n_opts = 0
        #print(f"({i1}, {i2}), at vertical edge = {is_at_vertical_edge}, at horizontal edge = {is_at_horizontal_edge}")
        #if is_at_horizontal_edge or is_at_vertical_edge:
        n_opts = 3 - (is_at_horizontal_edge + is_at_vertical_edge)
        if n_opts == 3: n_opts = len(connect_combs_interior)
        for j in range(n_opts):
            curr[i1][i2] = j
            recur(i1, i2 + 1)
        return


    recur(0, 0)
    return res[0]
        

class UnionFind:
    def __init__(self, n: int):
        self.n = n
        self.root = list(range(n))
        self.rank = [1] * n
    
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
        return
    
    def connected(self, i1: int, i2: int) -> bool:
        return self.find(i1) == self.find(i2)


def circleArrayEulerianNonCrossingCycleCount(
    n_rows: int=6,
    n_cols: int=10,
    res_md: Optional[int]=10 ** 10,
) -> int:
    """
    Solution to Project Euler #289
    """
    # Adapted from solution to Project Euler #237
    # Review- Consider sharing on Project Euler forum

    # Known:
    # (1, m): 1 << (m - 1)
    # (2, m): 2, 37, 672, 12182, 220792,
    # (3, m): 4, 672, 104290

    if n_rows > n_cols:
        n_rows, n_cols = n_cols, n_rows

    

    states = []
    states_dict = {}

    state_memo = {}
    def getStateIndex(state_raw: Iterable[int]) -> int:
        #idx_dict = {}
        args = tuple(state_raw)
        if args in state_memo.keys():
            return state_memo[args]
        state1, state2 = [], []
        for stt, it in ((state1, state_raw), (state2, reversed(state_raw))):
            idx_dict = {0: 0}
            nxt = 1
            for num in it:
                if num not in idx_dict.keys():
                    idx_dict[num] = nxt
                    nxt += 1
                stt.append(idx_dict[num])
        std_state = tuple(min(state1, state2))
        if std_state in states_dict.keys():
            idx = states_dict[std_state]
        else:
            idx = len(states)
            states_dict[std_state] = idx
            states.append(std_state)
        state_memo[args] = idx
        return idx

    # The standardised states that can exist at the beginning
    # and end and their corresponding frequencies
    poss_end_states_std = {}
    end_connects = []
    for bm in range(1 << (n_rows - 1)):
        state = [1]
        connects = [-1] * (2 * n_rows)
        curr = 1
        for i in range(2, n_rows + 1):
            j = bm & 1
            bm >>= 1
            if j:
                state.extend([i, i])
                connects[2 * i - 3] = 2 * i - 2
                connects[2 * i - 2] = 2 * i - 3
            else:
                state.extend([curr, i])
                connects[2 * curr - 2] = 2 * i - 3
                connects[2 * i - 3] = 2 * curr - 2
                curr = i
        
        state.append(curr)
        connects[2 * curr - 2] = 2 * n_rows - 1
        connects[2 * n_rows - 1] = 2 * curr - 2
        idx = getStateIndex(state)
        poss_end_states_std[idx] = poss_end_states_std.get(idx, 0) + 1
        end_connects.append(connects)

    incoming_branch_connections = []
    incoming_branch_connections_dict = {}
    n_branches = 2 * n_rows
    print(f"end connects = {end_connects}")

    def getIncomingBranchConnectionsIndex(connects_raw: List[Tuple[int, int]]) -> int:
        pairs1 = sorted(tuple(sorted(x)) for x in connects_raw)
        pairs2 = sorted(tuple(sorted([n_branches - x[0] - 1, n_branches - x[1] - 1])) for x in connects_raw)
        pairs_tup = tuple(min(pairs1, pairs2))
        if pairs_tup in incoming_branch_connections_dict.keys():
            return incoming_branch_connections_dict[pairs_tup]
        res = len(incoming_branch_connections)
        incoming_branch_connections_dict[pairs_tup] = res
        incoming_branch_connections.append(pairs_tup)
        return res

    def calculateLayerConnections(n_rows: int) -> Dict[tuple, Dict[tuple, int]]:
        res = {}
        #nxt_mid = [0]
        #nxt_r = [0]

        curr = [[], [None] * (n_rows * 2)]

        res = {}
        def recur(idx: int, curr_l: Optional[Tuple[bool, int]], curr_r: Optional[Tuple[bool, int]], mult: int=1) -> None:
            #print(f"args = {(idx, curr_l, curr_r, mult)}, curr = {curr}")
            j2 = (idx << 1)
            j1 = j2 - 1
            curr[1][j1] = None
            if idx == n_rows:

                # The left and right descending branches connect to each other (either
                # directly in the case that curr_l is not None or through the descending
                # branches in the case that curr_l is None)
                pop_count = 0
                if curr_l is not None:
                    if curr_l[0]:
                        curr[1][curr_l[1]] = curr_r
                    if curr_r[0]:
                        curr[1][curr_r[1]] = curr_l
                    if not curr_r[0] and not curr_l[0]:
                        curr[0].append((curr_l[1], curr_r[1]))
                        pop_count += 1

                curr[1][j1] = (False, j1)

                #branch_connects_l_idx = getIncomingBranchConnectionsIndex(curr[0])
                #state_r_idx = getStateIndex(curr[1])
                #res.setdefault(branch_connects_l_idx, {})
                #res[branch_connects_l_idx][state_r_idx] = res[branch_connects_l_idx].get(state_r_idx, 0) + mult

                #branch_connects_l_idx = getIncomingBranchConnectionsIndex(curr[0])
                branch_connects_l_tup = tuple(sorted([tuple(sorted(x)) for x in curr[0]]))
                #state_r_idx = getStateIndex(curr[1])
                #print(curr[1])
                connects_l_tup = tuple(tuple(x) for x in curr[1])
                #print(branch_connects_l_tup, connects_l_tup, mult)
                res.setdefault(branch_connects_l_tup, {})
                res[branch_connects_l_tup][connects_l_tup] = res[branch_connects_l_tup].get(connects_l_tup, 0) + mult

                if curr_l is not None:
                    if curr_l[0]: curr[1][curr_l[1]] = None
                    if curr_r[0]: curr[1][curr_r[1]] = None
                curr[1][j1] = None
                    
                for _ in range(pop_count): curr[0].pop()
                
                # Note that if curr_l[0] is None the final incoming and outgoing
                # branches must connect via the descending branches, so no further
                # configurations are possible
                if curr_l is None: return

                # The final incoming and outgoing branches connect to the left and
                # right descending branches respectively
                pop_count = 0
                if curr_l[0]:
                    curr[1][curr_l[1]] = (False, j1)
                else:
                    curr[0].append((curr_l[1], j1))
                    pop_count += 1
                curr[1][j1] = curr_r
                if curr_r[0]: curr[1][curr_r[1]] = (True, j1)

                if len(curr[0]) < n_rows:
                    # Require the layers to be connected, so at least one connection
                    # must exist between the incoming and outgoing branches (in fact,
                    # as the number of such connections must be even, this means there
                    # must be least 2 such connections, which is consistent with an
                    # Eulerian circuit)
                    #branch_connects_l_idx = getIncomingBranchConnectionsIndex(curr[0])
                    branch_connects_l_tup = tuple(sorted([tuple(sorted(x)) for x in curr[0]]))
                    #state_r_idx = getStateIndex(curr[1])
                    #print(curr[1])
                    connects_l_tup = tuple(tuple(x) for x in curr[1])
                    #print(branch_connects_l_tup, connects_l_tup, mult)
                    res.setdefault(branch_connects_l_tup, {})
                    res[branch_connects_l_tup][connects_l_tup] = res[branch_connects_l_tup].get(connects_l_tup, 0) + mult
                curr[1][j1] = None
                if curr_l[0]: curr[1][curr_l[1]] = None
                if curr_r[0]: curr[1][curr_r[1]] = None
                for _ in range(pop_count): curr[0].pop()
                
                return


            curr[1][j2] = None

            if curr_l is None:
                # The descending branches are already connected to each other and not yet any
                # incoming or outgoing branches

                # The left descending branch connects to the upper incoming branch

                # The right descending branch connects to the upper outgoing branch
                curr[1][j1] = (False, j1)
                recur(idx + 1, (False, j2), (True, j2), mult=mult)
                curr[1][j2] = (False, j2)
                recur(idx + 1, None, None, mult=mult)
                
                #curr[1][j1] = None
                # The right descending branch continues as the right descending branch
                curr[1][j1] = (True, j2)
                curr[1][j2] = (True, j1)
                #print("right descending continues")
                recur(idx + 1, (False, j2), (False, j1), mult=mult)
                curr[1][j1] = None
                curr[1][j2] = None
                # The two cases:
                #  1) The right descending branch connects to the lower incoming branch,
                #      the upper outgoing branch connects to the new left descending branch
                #  2) The left descending branch continues as the left descending branch and
                #      the right descending branch connects to the upper outgoing branch
                curr[0].append((j1, j2))
                recur(idx + 1, (True, j1), (True, j2), mult=2 * mult)
                #curr[0].pop()

                # The three cases:
                #  1) the both descending branches continue
                #  2) the descending branches connect to the incoming branches with the outgoing branches connecting to each other
                #  3) the descending branches connect to the outgoing branches with the incoming branches connecting to each other
                curr[1][j1] = (True, j2)
                curr[1][j2] = (True, j1)
                recur(idx + 1, None, None, mult=mult * 3)
                curr[1][j1] = None
                curr[1][j2] = None

                # The descending branches connect to the incoming branches and the outgoing branches connect to the new descending branches
                #recur(idx + 1, (True, j1), (True, j2), mult=mult)
                curr[0].pop()
                
                # The descending branches connect to the outgoing branches and the incoming branches connect to the new descending branches
                curr[1][j1] = (True, j2)
                curr[1][j2] = (True, j1)
                recur(idx + 1, (False, j2), (False, j1), mult=mult)
                curr[1][j1] = None
                curr[1][j2] = None
                

                return
            
            # The descending branches arise from incoming or outgoing branches
            # and are not yet connected to each other

            # The descending branches become connected

            pop_count = 0
            if curr_l[0]:
                curr[1][curr_l[1]] = curr_r
            if curr_r[0]:
                curr[1][curr_r[1]] = curr_l
            if not curr_r[0] and not curr_l[0]:
                curr[0].append((curr_l[1], curr_r[1]))
                pop_count += 1


            #if curr_l[0] == curr_r[0]:
            #    if curr_l[0]:
            #        curr[1][curr_r[1]] = curr_l#curr[1][curr_l[1]]
            #        curr[1][curr_l[1]] = curr_r
            #    else:
            #        curr[0].append((curr_l[1], curr_r[1]))
            #        pop_count += 1

            # The upper incoming branch connects to the upper outgoing branch
            curr[1][j1] = (False, j1)
            curr[1][j2] = (False, j2)
            recur(idx + 1, None, None, mult=mult)
            curr[1][j2] = None
            recur(idx + 1, (False, j2), (True, j2), mult=mult)
            # The upper incoming branch connects to the new right descending branch
            curr[1][j1] = (True, j2)
            curr[1][j2] = (True, j1)
            recur(idx + 1, (False, j2), (False, j1), mult=mult)
            # The upper incoming branch connects to the lower incoming branch
            curr[0].append((j1, j2))
            pop_count += 1
            recur(idx + 1, None, None, mult=mult)
            curr[1][j1] = None
            curr[1][j2] = None
            recur(idx + 1, (True, j1), (True, j2), mult=mult)

            #curr[1][curr_r[1]] = curr_r[1] + 1
            if curr_l[0]: curr[1][curr_l[1]] = None
            if curr_r[0]: curr[1][curr_r[1]] = None
            for _ in range(pop_count): curr[0].pop()

            # The left descending branch connects to the upper incoming branch

            pop_count = 0
            if curr_l[0]:
                curr[1][curr_l[1]] = (False, j1)
            else:
                curr[0].append((curr_l[1], j1))
                pop_count += 1

            # The right descending branch connects to the upper outgoing branch
            #if curr_r[0]:
            #    curr[1][j1] = curr[1][curr_r[1]]
            if curr_r[0]:
                curr[1][curr_r[1]] = (True, j1)
            curr[1][j1] = curr_r
            curr[1][j2] = (False, j2)
            recur(idx + 1, None, None, mult=mult)
            curr[1][j2] = None
            recur(idx + 1, (False, j2), (True, j2), mult=mult)
            #curr[1][j1] = None
            # The right descending branch continues as the new right descending branch
            if curr_r[0]: curr[1][curr_r[1]] = None
            curr[1][j1] = (True, j2)
            curr[1][j2] = (True, j1)
            recur(idx + 1, (False, j2), curr_r, mult=mult)
            
            # The right descending branch connects to the lower incoming branch
            if curr_r[0]:
                curr[1][curr_r[1]] = (False, j2)
            else:
                curr[0].append((curr_r[1], j2))
                pop_count += 1
            recur(idx + 1, None, None, mult=mult)
            curr[1][j1] = None
            curr[1][j2] = None
            recur(idx + 1, (True, j1), (True, j2), mult=mult)

            if curr_l[0]: curr[1][curr_l[1]] = None
            if curr_r[0]: curr[1][curr_r[1]] = None
            for _ in range(pop_count): curr[0].pop()

            # The left descending branch continues as the new left descending branch
            
            curr[0].append((j1, j2))
            pop_count = 1

            # The right descending branch connects to the upper outgoing branch
            #if curr_r[0]:
            #    curr[1][j1] = curr[1][curr_r[1]]
            if curr_r[0]:
                curr[1][curr_r[1]] = (True, j1)
            curr[1][j1] = curr_r
            recur(idx + 1, curr_l, (True, j2), mult=mult)
            curr[1][j1] = None
            if curr_r[0]: curr[1][curr_r[1]] = None
            # The right descending branch continues as the new right descending branch
            curr[1][j1] = (True, j2)
            curr[1][j2] = (True, j1)
            recur(idx + 1, curr_l, curr_r, mult=mult)
            curr[1][j1] = None
            curr[1][j2] = None

            for _ in range(pop_count): curr[0].pop()

            # The left descending branch connects to the lower outgoing branch

            pop_count = 0
            #if curr_l[0]:
            #    curr[1][j2] = curr_l[1] + 1
            curr[1][j2] = curr_l
            if curr_l[0]:
                curr[1][curr_l[1]] = (True, j2)

            # The right descending branch connects to the upper outgoing branch
            #if curr_r[0]:
            #    curr[1][j1] = curr[1][curr_r[1]]
            if curr_r[0]:
                curr[1][curr_r[1]] = (True, j1)
            curr[1][j1] = curr_r
            recur(idx + 1, (False, j2), (False, j1), mult=mult)
            curr[0].append((j1, j2))
            pop_count += 1
            recur(idx + 1, None, None, mult=mult)

            if curr_l[0]: curr[1][curr_l[1]] = None
            if curr_r[0]: curr[1][curr_r[1]] = None
            curr[1][j1] = None
            curr[1][j2] = None
            for _ in range(pop_count): curr[0].pop()

            return

        recur(1, (False, 0), (True, 0), mult=1)
        #print(incoming_branch_connections)
        #print(res)
        #curr[0][0] = [True, 0]
        #curr[1][0] = [False, 0]
        #nxt_mid[0] += 1
        curr[1][0] = (False, 0)
        recur(1, None, None, mult=1)
        #print(incoming_branch_connections)
        #print(res)
        #nxt_mid[0] -= 1
        return res

    
    layer_connections = calculateLayerConnections(n_rows)
    print(f"number of pairings of the incoming branches = {len(layer_connections)}")
    #print(list(layer_connections.keys()))

    #print(incoming_branch_connections)
    #print(states)
    #for k, v in layer_connections.items():
    #    print(f"{k}: {v}")

    print("initial states:")
    print(states)
    print("frequencies:")
    print(poss_end_states_std)
    nxt_print = [10]

    def getTransferOutEdges(state_idx: int) -> Dict[int, int]:
        state0 = states[state_idx]
        state0_mx = max(state0)
        res = {}
        for connect_pairs in layer_connections.keys():
            #print(connect_pairs)
            #cp1, cp2 = connect_pairs, tuple(sorted(tuple(sorted(n_rows * 2 - x[0], n_rows * 2 - x[1])) for x in connect_pairs))
            #it = [cp1] if cp1 == cp2 else [cp1, cp2]
            cnt = 0
            for cp in [connect_pairs]:
                uf = UnionFind(2 * n_rows)
                for pair in cp:
                    num1, num2 = state0[pair[0]] - 1, state0[pair[1]] - 1
                    if uf.connected(num1, num2):
                        break
                    uf.union(num1, num2)
                else: cnt += 1
            if not cnt: continue
            #incoming_branch_map = {}

            #to_break = False
            #for idx1, idx2 in connect_pairs:
            #    num1, num2 = state0[idx1] - 1, state0[idx2] - 1
            #    if uf.connected(num1, num2):
            #        to_break = True
            #        break
            #    uf.union(num1, num2)
            #if to_break: break
            
            for out_branch_connects, freq in layer_connections[connect_pairs].items():
                state = []
                
                for idx0, (b, idx) in enumerate(out_branch_connects):
                    if not b:
                        state.append(uf.find(state0[idx] - 1) + 1)
                        continue
                    if idx < idx0: state.append(state[idx])
                    else: state.append(idx0 + state0_mx + 1)
                    continue
                state_idx = getStateIndex(tuple(state))
                #print(f"state0 = {state0}, incoming connect_pairs = {connect_pairs}, out_branch_connects = {out_branch_connects}, state = {state}")
                res[state_idx] = res.get(state_idx, 0) + freq * cnt
        #print(f"for state_idx = {state_idx}, out edges = {res}")
        return res

    
    def getTransferOutEdges0(state_idx: int) -> Dict[int, int]:
        state0 = states[state_idx]
        m = (len(state0)) >> 1
        t_dict = {}

        curr = []
        l_set = set(state)
        #r_dict = {}
        nxt = [max(l_set) + 1 if l_set else 1]
        curr_connects = {}
        
        
        def recur(idx: int, above_l: int, above_r: int) -> None:
            #if len(states) >= nxt_print[0]:
            #    print(f"number of states seen = {len(states)}, latest state = {states[-1]}")
            #    nxt_print[0] += 10
            jl = above_l
            while jl != curr_connects.get(jl, jl):
                jl = curr_connects[jl]
            jr = above_r
            while jr != curr_connects.get(jr, jr):
                jr = curr_connects[jr]
            if idx == m:
                j0 = state0[-1]
                while j0 != curr_connects.get(j0, j0):
                    j0 = curr_connects[j0]
                curr.append(-1)
                if jl != j0:
                    curr_connects[jl] = j0
                    curr[-1] = jr
                    ans = []
                    for idx in curr:
                        stt = idx
                        while stt != curr_connects.get(stt, stt):
                            stt = curr_connects[stt]
                        ans.append(stt)
                    idx2 = getStateIndex(tuple(ans))
                    t_dict[idx2] = t_dict.get(idx2, 0) + 1
                    curr_connects.pop(jl)
                if jl != jr:
                    curr_connects[jr] = jl
                    curr[-1] = j0
                    ans = []
                    for idx in curr:
                        stt = idx
                        while stt != curr_connects.get(stt, stt):
                            stt = curr_connects[stt]
                        ans.append(stt)
                    idx2 = getStateIndex(tuple(ans))
                    t_dict[idx2] = t_dict.get(idx2, 0) + 1
                    curr_connects.pop(jr)
                curr.pop()
                return
            j0 = state0[2 * idx - 1]
            while j0 != curr_connects.get(j0, j0):
                j0 = curr_connects[j0]
            #print(idx, 2 * idx, state0)
            j1 = state0[2 * idx]
            while j1 != curr_connects.get(j1, j1):
                j1 = curr_connects[j1]
            
            curr.extend([0, 0])
            j2 = nxt[0]
            nxt[0] += 1
            if j0 != j1:
                # Connecting the incoming left branches
                curr_connects[j1] = j0
                
                if jl != jr and ({jl, jr} != {j0, j1}):
                    # Connecting the incoming top branches
                    i1, i2 = (jl, jr) if j1 != jl else (jr, jl)
                    curr_connects[i1] = i2
                    j3 = nxt[0]
                    nxt[0] += 1
                    curr[-2] = j3
                    curr[-1] = j3
                    recur(idx + 1, j2, j2)
                    curr[-2] = j2
                    recur(idx + 1, j2, j3)
                    nxt[0] -= 1
                    curr_connects.pop(i1)
                
                curr[-2] = jr
                curr[-1] = jl
                recur(idx + 1, j2, j2)
                curr[-1] = j2
                recur(idx + 1, jl, j2)
                curr[-2] = j2
                recur(idx + 1, jl, jr)
                #print(j0, j1, jl, jr)
                curr_connects.pop(j1)
            
            if j0 != jl:
                # Connecting the incoming left upper branch to the top
                # leftmost branch
                curr_connects[jl] = j0
                #j1_2 = j1
                #while j1_2 != curr_connects.get(j1_2, j1_2):
                #    j1_2 = curr_connects[j1_2]
                if j1 != jr and ({j0, jl} != {j1, jr}):
                    # Connecting the incoming left lower branch to the top
                    # rightmost branch
                    i1, i2 = (j1, jr) if j1 != jl else (jr, j1)
                    curr_connects[i1] = i2
                    j3 = nxt[0]
                    nxt[0] += 1
                    curr[-2] = j3
                    curr[-1] = j3
                    recur(idx + 1, j2, j2)
                    curr[-2] = j2
                    recur(idx + 1, j2, j3)
                    nxt[0] -= 1
                    curr_connects.pop(i1)
                
                curr[-2] = jr
                curr[-1] = j1
                recur(idx + 1, j2, j2)
                curr[-1] = j2
                recur(idx + 1, j1, j2)
                curr[-2] = j2
                recur(idx + 1, j1, jr)
                #print(j1, j2, jl, jr)
                curr_connects.pop(jl)

            if jl != jr:
                # Connecting the incoming top branches (note that
                # the case where the incoming left branches are
                # also connected has already been handled)
                curr_connects[jr] = jl

                curr[-2] = j0
                curr[-1] = j1
                recur(idx + 1, j2, j2)
                curr[-1] = j2
                recur(idx + 1, j1, j2)
                curr[-2] = j2
                recur(idx + 1, j1, j0)

                curr_connects.pop(jr)
            nxt[0] -= 1
            # None of the incoming left or top branches connect to
            # each other
            curr[-2] = jr
            curr[-1] = jl
            recur(idx + 1, j1, j0)

            curr.pop()
            curr.pop()
            return
        
        j = nxt[0]
        nxt[0] += 1
        curr = [state0[0]]
        recur(1, j, j)
        curr[0] = j
        recur(1, state0[0], j)
        
        return t_dict

        """
        curr = []
        l_set = set(state) - {0}
        r_dict = {}
        #incl_dict = {}
        #map_dict = {}
        nxt = [max(l_set) + 1 if l_set else 1]
        curr_map = list(range(len(state)))
        def recur(idx: int=0, above: int=0, non_zero_seen: bool=False) -> None:
            #if state_idx == 3 and len(curr) >= 2 and curr[0] == 1 and curr[1] in {1, 2}:
            #    print(idx, above, non_zero_seen, curr)
            if idx == m:
                if above or not non_zero_seen: return
                ans = []
                for idx in curr:
                    stt = idx
                    while stt != curr_map[stt]:
                        stt = curr_map[stt]
                    ans.append(stt)
                #if state_idx == 3:
                ##    print(ans)
                idx2 = getStateIndex(tuple(ans))
                t_dict[idx2] = t_dict.get(idx2, 0) + 1
                return
            curr.append(0)
            if not state[idx]:
                if above:
                    recur(idx=idx + 1, above=above, non_zero_seen=non_zero_seen)
                    curr[idx] = above
                    recur(idx=idx + 1, above=0, non_zero_seen=True)
                    curr[idx] = 0
                else:
                    above2 = nxt[0]
                    curr[idx] = above2
                    nxt[0] += 1
                    r_dict[above2] = idx
                    recur(idx=idx + 1, above=above2, non_zero_seen=True)
                    r_dict.pop(above2)
                    curr[idx] = 0
                    nxt[0] -= 1
            else:
                stt = state[idx]
                while stt != curr_map[stt]:
                    stt = curr_map[stt]
                if not above:
                    recur(idx=idx + 1, above=stt, non_zero_seen=non_zero_seen)
                    curr[idx] = stt
                    recur(idx=idx + 1, above=0, non_zero_seen=True)
                    curr[idx] = 0
                elif above != stt:
                    curr_map[stt] = above
                    recur(idx=idx + 1, above=0, non_zero_seen=True)
                    curr_map[stt] = stt
            curr.pop()
            return
        recur(idx=0, above=0)
        
        return t_dict
        """
    
    def createTransferAdj0(start_state_inds: List[int]) -> List[Dict[int, int]]:
        
        seen = set()
        qu = deque()
        adj = []
        for idx in start_state_inds:
            if idx in seen: continue
            seen.add(idx)
            qu.append(idx)
        while qu:
            idx = qu.popleft()
            print(f"creating out edges for index {idx}, state {states[idx]}. Total number of states seen = {len(states)}")
            adj += [{} for _ in range(idx - len(adj) + 1)]
            #print(f"creating out edges for index {idx}, state {states[idx]}")
            adj[idx] = getTransferOutEdges0(idx)
            for idx2 in adj[idx].keys() - seen:
                qu.append(idx2)
                seen.add(idx2)
            #print([(states[idx2], f) for idx2, f in adj[idx].items()])
        return adj
    
    def createTransferAdj(start_state_inds: List[int]) -> List[Dict[int, int]]:
        
        seen = set()
        qu = deque()
        adj = []
        for idx in start_state_inds:
            if idx in seen: continue
            seen.add(idx)
            qu.append(idx)
        while qu:
            idx = qu.popleft()
            print(f"creating out edges for index {idx}, state {states[idx]}. Total number of states seen = {len(states)}")
            adj += [{} for _ in range(idx - len(adj) + 1)]
            #print(f"creating out edges for index {idx}, state {states[idx]}")
            adj[idx] = getTransferOutEdges(idx)
            for idx2 in adj[idx].keys() - seen:
                qu.append(idx2)
                seen.add(idx2)
            #print([(states[idx2], f) for idx2, f in adj[idx].items()])
        return adj
    
    state_adj = createTransferAdj(list(poss_end_states_std.keys()))
    #state_adj0 = createTransferAdj0(list(poss_end_states_std.keys()))
    n_states = len(states)
    print(f"finished creating state adjacency table")
    
    #print(states)
    print(f"number of distinct reachable states = {n_states}")
    #print(states)
    #print(state_adj)
    #print(state_adj0)
    def multiplyStateAdj(state_adj1: List[Dict[int, int]], state_adj2: List[Dict[int, int]]) -> List[Dict[int, int]]:
        res = [{} for _ in range(n_states)]
        for idx1 in range(n_states):
            for idx2, f2 in state_adj1[idx1].items():
                for idx3, f3 in state_adj2[idx2].items():
                    res[idx1][idx3] = (res[idx1].get(idx3, 0) + f2 * f3)
                    if res_md is not None: res[idx1][idx3] %= res_md
        return res
    
    def applyStateAdj(state_adj: List[Dict[int, int]], state_f_dict: Dict[int, int]) -> Dict[int, int]:
        res = {}
        for idx, f in state_f_dict.items():
            for idx2, f2 in state_adj[idx].items():
                res[idx2] = (res.get(idx2, 0) + f * f2)
                if res_md is not None: res[idx2] %= res_md
        return res

    
    # binary lift
    state_adj_bin = state_adj
    curr = dict(poss_end_states_std)
    #print(f"start state = {curr}")
    m = n_cols - 1
    while True:
        if m & 1:
            curr = applyStateAdj(state_adj_bin, curr)
        m >>= 1
        if not m: break
        state_adj_bin = multiplyStateAdj(state_adj_bin, state_adj_bin)
    
    """
    # basic repeated application of the state transfer
    curr = dict(poss_end_states_std)
    for _ in range(n_cols - 1):
        curr = applyStateAdj(state_adj, curr)
    """
    # Closing the paths so that the path forms a single cycle
    res = 0
    #print(curr)
    for state_idx, f in curr.items():
        state = states[state_idx]
        connects0 = [-1] * (2 * n_rows)
        seen = {}
        for i, num in enumerate(state):
            if num in seen.keys():
                connects0[i] = seen[num]
                connects0[seen[num]] = i
            else: seen[num] = i
        for connects in end_connects:
            n_seen = 1
            curr = 0
            while True:
                curr = connects0[curr]
                if not curr: break
                n_seen += 1
                curr = connects[curr]
                if not curr: break
                n_seen += 1
            if n_seen == 2 * n_rows:
                res += f
                if res_md is not None: res %= res_md
    return res



    #for idx in poss_end_states_std.keys():
    #    res += curr.get(idx, 0)
    #    if res_md is not None: res %= res_md
    #return res
    """
    for idx, f in curr.items():
        #if state == (1, 0, 0, 1):
        #    res += state_dict[state]
        pair_adj = {}
        prev = None
        state = states[idx]
        for num in state:
            if not num:
                if prev is None: break
                continue
            if prev is None:
                prev = num
                continue
            pair_adj.setdefault(prev, set())
            pair_adj.setdefault(num, set())
            pair_adj[prev].add(num)
            pair_adj[num].add(prev)
            prev = None
        else:
            if prev is not None: continue
            n_nums = len(pair_adj)
            
            num1 = next(iter(pair_adj.keys()))
            if len(pair_adj[num1]) == 1:
                cycle_len = 1 + (next(iter(pair_adj[num1])) != num1)
            else:
                num2 = next(iter(pair_adj[num1]))
                cycle = [num1]
                while num2 != cycle[0]:
                    cycle.append(num2)
                    num1, num2 = num2, next(iter(pair_adj[num2] - {num1}))
                cycle_len = len(cycle)
            if cycle_len != n_nums: continue
            #print(state)
            res += f
            if res_md is not None: res %= res_md
    return res
    """
    

# Problem 290
def digitSumEqualsMultipleDigitSumCount(
    n_dig_max: int=18,
    mult: int=137,
    base: int=10,
) -> int:
    """
    Solution to Project Euler #290
    """
    def digitSum(num: int) -> int:
        res = 0
        while num:
            num, d = divmod(num, base)
            res += d
        return res

    memo = {}
    def recur(n_dig_remain: int, carry: int=0, diff: int=0) -> int:
        if not n_dig_remain:
            return int(digitSum(carry) == -diff)
        elif diff + bool(carry) > (base - 1) * n_dig_remain:
            #print("hi")
            return 0
        args = (n_dig_remain, carry, diff)
        if args in memo.keys(): return memo[args]
        res = 0
        for d in range(base):
            carry2, d2 = divmod(carry + mult * d, base)
            res += recur(n_dig_remain - 1, carry=carry2, diff=diff + d2 - d)
        memo[args] = res
        return res
    
    res = recur(n_dig_max, carry=0, diff=0)
    return res

# Problem 291
def panaitopolPrimesBruteForce(p_max: int) -> List[int]:
    ps = SimplePrimeSieve()
    def primeCheck(num: int) -> int:
        return ps.millerRabinPrimalityTestWithKnownBounds(num, max_n_additional_trials_if_above_max=10)[0]
    
    res = []
    for x in range(2, 20 * p_max + 1):
        y = x - 1
        if (x ** 4 - y ** 4 - 1) // (x ** 3 + y ** 3) >= p_max: break
        for y in reversed(range(1, x)):
            p, r = divmod(x ** 4 - y ** 4, x ** 3 + y ** 3)
            if p > p_max: break
            if r or not primeCheck(p): continue
            ratio = (x ** 2 - x * y + y ** 2) // (x - y)
            print(p, x, y, (x ** 2 - x * y + y ** 2), (x - y), ratio)
            res.append(p)
    
    return sorted(res)

def panaitopolPrimesCount(p_max: int=5 * 10 ** 15 - 1) -> int:
    """
    Solution to Project Euler #291
    """
    # Review- prove that p must be of the form n ** 2 + (n + 1) ** 2
    # and that all primes of that form are panaitolo primes.
    ps = SimplePrimeSieve()
    def primeCheck(num: int) -> int:
        return ps.millerRabinPrimalityTestWithKnownBounds(num, max_n_additional_trials_if_above_max=10)[0]
    n_max = (isqrt(2 * p_max - 1) - 1) >> 1
    print(f"n_max = {n_max}")
    res = 0
    for n in range(1, n_max + 1):
        if not n % 10000:
            print(f"n = {n} of {n_max}")
        p = 2 * n ** 2 + 2 * n + 1
        is_prime = primeCheck(p)
        #if is_prime: print(p)
        res += is_prime
    return res

# Problem 292
def pythagoreanPolygonCountInitialSolution(perim_max: int) -> int:
    
    # Review- try to make faster- possibly using double-ended
    # search (keeping one branch of edges with total length
    # no greater than that of the other to avoid double counting).
    side_len_max = (perim_max - 1) >> 1
    pythag_triple_lst = [(0, 1, 1)]
    poss_sides_cnt = 2 * side_len_max
    for tup in pythagoreanTripleGeneratorByHypotenuse(primitive_only=True, max_hypotenuse=side_len_max):
        pythag_triple_lst.append(tup[0])
        poss_sides_cnt += 4 * (side_len_max // tup[0][2])
    print(pythag_triple_lst)
    print(f"maximum number of possible sides out of each vertex = {poss_sides_cnt}")

    pythag_grads_sorted = sorted([(CustomFraction(x[0], x[1]), (x[1], x[0], x[2])) for x in pythag_triple_lst])
    print(pythag_grads_sorted)
    # Review- try to account for this in recur() to save memory
    pythag_grads_sorted2 = [(CustomFraction(-y[0], y[1]), (y[1], -y[0], y[2])) for x, y in pythag_grads_sorted]
    pythag_grads_sorted2.extend([(CustomFraction(-y[1], y[0]), (y[0], -y[1], y[2])) for x, y in reversed(pythag_grads_sorted)])
    pythag_grads_sorted2.pop()
    pythag_grads_sorted2.extend(pythag_grads_sorted)
    pythag_grads_sorted2.extend([(CustomFraction(y[0], y[1]), (y[1], y[0], y[2])) for x, y in reversed(pythag_grads_sorted)])
    pythag_grads_sorted2.pop()
    print(pythag_grads_sorted2)
    print(len(pythag_grads_sorted2))
    n_pythag_grads = len(pythag_grads_sorted2)

    curr_first_edge_idx = [-1]
    memo = {}
    def recur(idx: int, neg: bool, perim_remain: int, pt: Tuple[int, int], first_edge_idx: int=-1, latest_incl: Tuple[bool, int]=(True, -1)) -> int:
        
        #d_sq = sum(x * x for x in pt)
        #perim_remain_sq = perim_remain * perim_remain
        #if d_sq > perim_remain_sq: return 0
        #if first_edge_idx >= 0 and pt[0] <= 0 and CustomFraction(pt[1], pt[0]) >= pythag_grads_sorted2[first_edge_idx][0]: return 0
        if first_edge_idx != curr_first_edge_idx[0]:
            # Optimisation to save memory by removing results that are no longer needed
            memo.clear()
            curr_first_edge_idx[0] = first_edge_idx
        #args = (idx, neg, perim_remain, pt, first_edge_idx, latest_incl)
        args = (idx, neg, perim_remain, pt, latest_incl)
        #print("start", args)
        ref = None#(0, True, 2, (1, -1), 0, (False, 1))
        ref_first_edge_idx = None#0
        
        is_ref = (ref == args)
        idx_nxt = idx + 1
        neg_nxt = neg

        is_ref_first_edge_idx = (first_edge_idx == ref_first_edge_idx)

        if neg and not latest_incl[0] and idx >= latest_incl[1]:
            return 0
        last = False
        if idx_nxt == n_pythag_grads:
            # if neg:
            #     d_sq = sum(x * x for x in pt)
            #     d = isqrt(d_sq)
            #     if d * d == d_sq and pt[0] > 0:
            #         if is_ref_first_edge_idx:
            #             print(f"integer distance from origin: {pt}")
            #         return 1
            #     return 0
            if first_edge_idx < 0: return 0
            elif neg: last = True

            idx_nxt = 0
            neg_nxt = not neg
        
        vec = tuple(pythag_grads_sorted2[idx][1][:2])

        if last:
            mult, r = divmod(pt[0], vec[0]) if vec[0] else divmod(pt[1], vec[1])
            if r or tuple(x * mult for x in vec) != pt: return 0
            return 1
        if first_edge_idx >= 0 and args in memo.keys(): return memo[args]
        res0 = recur(idx_nxt, neg_nxt, perim_remain, pt, first_edge_idx=first_edge_idx, latest_incl=latest_incl)
        res = res0

        first_edge = False
        first_edge_nxt = first_edge_idx
        if first_edge_idx < 0:
            first_edge = True
            first_edge_idx = idx
            is_ref_first_edge_idx = (idx == ref_first_edge_idx)
        
        perim_remain2 = perim_remain
        
        if neg: vec = (-vec[0], -vec[1])
        #print(f"vec = {vec}, length = {pythag_grads_sorted2[idx][1][2]}")
        for mult in range(1, (perim_remain // pythag_grads_sorted2[idx][1][2]) + 1):
            pt2 = tuple(x + mult * y for x, y in zip(pt, vec))
            perim_remain2 -= pythag_grads_sorted2[idx][1][2]
            if is_ref_first_edge_idx:
                print(f"args = {args}, unit vec = {vec}, mult = {mult}, edge vec = {tuple(mult * x for x in vec)}, pt2 = {pt2}, perim_remain2 = {perim_remain2}")
            
            if pt2 == (0, 0):
                if is_ref_first_edge_idx: print("polygon closed")
                #print(pt, tuple(mult * y for y in vec))
                res += 1
                break
            elif neg and pt2[0] <= 0:
                if is_ref_first_edge_idx: print("x-coordinate is non-positive")
                break
            elif sum(x * x for x in pt2) > perim_remain2 * perim_remain2:
                if is_ref_first_edge_idx: print("distance from origin too large")
                break
            #elif first_edge_idx != idx and pt2[0] <= 0 and CustomFraction(pt2[1], pt2[0]) >= pythag_grads_sorted2[first_edge_idx][0]:
            #    if is_ref: print("gradient to origin too large")
            #    break
            #if first_edge:
            #    print(f"first edge, unit vec = {vec}, mult = {mult}, edge vec = {tuple(mult * x for x in vec)}, pt2 = {pt2}, perim_remain2 = {perim_remain2}")
            res += recur(idx_nxt, neg_nxt, perim_remain2, pt2, first_edge_idx=first_edge_idx, latest_incl=(neg, idx))
        if first_edge:
            print(f"end for first edge unit {vec}", args, vec, res - res0)
        if first_edge >= 0:
            memo[args] = res
        return res
    
    res = 0
    res = recur(0, False, perim_max, (0, 0), first_edge_idx=-1, latest_incl=(True, -1))
    #for i, (grad, triple) in enumerate(pythag_grads_sorted2):
    #    mult_max = side_len_max // triple[2]
    #    for mult in range(1, mult_max + 1):
    #        res += recur(perim_max - mult * triple[2], (triple[0] * mult, triple[1] * mult), i, i, False)
    return res

def pythagoreanPolygonCount(perim_max: int) -> int:
    """
    Solution to Project Euler #292
    """
    side_len_max = (perim_max - 1) >> 1
    pythag_triple_lst = []
    for tup in pythagoreanTripleGeneratorByHypotenuse(primitive_only=True, max_hypotenuse=side_len_max):
        pythag_triple_lst.append(tup[0])
    print(pythag_triple_lst)
    curr = {(0, 0): SortedDict({0: 1})}
    
    for triple in pythag_triple_lst:
        vec = (triple[0], triple[1])
        mult_mx = side_len_max // triple[2]
        d = triple[2]
        prev = dict(curr)
        curr = {}
        for pos, len_dict in prev.items():
            for l, f in len_dict.items():
                pos2 = pos
                remain = perim_max - l
                l2 = l
                curr.setdefault(pos2, SortedDict())
                curr[pos2][l2] = curr[pos2].get(l2, 0) + f
                
                for mult in range(1, mult_mx + 1):
                    pos2 = tuple(x + y for x, y in zip(pos2, vec))
                    remain -= d
                    if sum(x * x for x in pos2) > remain * remain:
                        break
                    l2 += d
                    curr.setdefault(pos2, SortedDict())
                    curr[pos2][l2] = curr[pos2].get(l2, 0) + f
    
    def combinePositionLengthFrequencies(
        pos_len_f_dict1: Dict[Tuple[int, int], SortedDict[int, int]],
        pos_len_f_dict2: Dict[Tuple[int, int], SortedDict[int, int]],
        pos2_transform: Callable[[Tuple[int, int]], Tuple[int, int]]=(lambda pos: pos),
    ) -> Dict[Tuple[int, int], SortedDict[int, int]]:
        res = {}
        for pos1, len_dict1 in pos_len_f_dict1.items():
            #cumu_lst1 = []
            #tot = 0
            #for l1, f1 in len_dict1.items():
            #    tot += f1
            #    cumu_lst1.append((l1, tot))
            len_lst1 = []
            for l1, f1 in len_dict1.items():
                len_lst1.append((l1, f1))
            for pos2_prov, len_dict2 in pos_len_f_dict2.items():
                #if pos2 < pos1: continue
                mult = 1# + (pos2 != pos1)
                pos2 = pos2_transform(pos2_prov)
                pos3 = tuple(x + y for x, y in zip(pos1, pos2))
                d_sq = sum(x * x for x in pos3)
                #remain_mn = isqrt(d_sq - 1) + 1
                
                l_sm_mx = perim_max - isqrt(d_sq - 1) - 1 if d_sq > 0 else perim_max
                
                if len_lst1[0][0] + len_dict2.peekitem(0)[0] > l_sm_mx:
                    continue
                res.setdefault(pos3, SortedDict())
                i1_0 = len(len_lst1) - 1
                #tot2 = 0
                #if pos1 == (12, 9) and pos2 == (0, 0):
                #    print(f"pos1 = {pos1}, pos2 = {pos2}")
                #    print(f"len_lst1 = {len_lst1}, len_dict2 = {len_dict2}")
                for l2, f2 in len_dict2.items():
                    l1_mx = l_sm_mx - l2
                    for i1_0 in reversed(range(i1_0 + 1)):
                        if len_lst1[i1_0][0] <= l1_mx: break
                    else: break
                    for i1 in reversed(range(i1_0 + 1)):
                        l3 = len_lst1[i1][0] + l2
                        #if pos1 == (12, 9) and pos2 == (0, 0): print(l3)
                        #tot2 += f2
                        res[pos3][l3] = res[pos3].get(l3, 0) + mult * len_lst1[i1][1] * f2
        return res

    # Add to reflection in the line x = y
    curr = combinePositionLengthFrequencies(curr, curr, pos2_transform=(lambda pos: (pos[1], pos[0])))
    #if (12, 9) in curr.keys():
    #    print(curr[(12, 9)])
    # Add the multiples of (0, 1)
    curr = combinePositionLengthFrequencies(curr, {(0, i): SortedDict({i: 1}) for i in range(side_len_max + 1)}, pos2_transform=(lambda pos: pos))
    #if (12, 9) in curr.keys():
    #    print(curr[(12, 9)])
    #if (0, 0) in curr.keys():
    #    print(curr[(0, 0)])
    # Clockwise rotation of pi / 2 about origin
    curr = combinePositionLengthFrequencies(curr, curr, pos2_transform=(lambda pos: (pos[1], -pos[0])))
    #if (12, 9) in curr.keys():
    #    print(curr[(12, 9)])
    #print(curr)
    excl_cnt1 = 0
    res = 0
    curr.pop((0, 0))
    for pos1, len_dict1 in curr.items():
        cumu_lst1 = []
        tot = 0
        for l1, f1 in len_dict1.items():
            tot += f1
            cumu_lst1.append((l1, tot))
        i1 = len(cumu_lst1) - 1
        for l2, f2 in len_dict1.items():
            l1_mx = perim_max - l2
            for i1 in reversed(range(i1 + 1)):
                if cumu_lst1[i1][0] <= l1_mx: break
            else: break
            #tot2 += f2
            res += cumu_lst1[i1][1] * f2
        
        # Excluding the paths with exactly two edges (which are not
        # polygons)
        #if cumu_lst1[0][0] * cumu_lst1[0][0] == sum(x * x for x in pos1):
        #    print(f"pos1 = {pos1}, excluded count = {cumu_lst1[0][1] * cumu_lst1[0][1]}")
        #    excl_cnt1 += cumu_lst1[0][1] * cumu_lst1[0][1]
    # Excluding the paths with exactly two edges (which are therefore
    # not polygons)
    #print("alternative count:")
    excl_cnt2 = 2 * side_len_max
    #print(f"triple (0, 1, 1) excluded count = {2 * side_len_max}")
    for triple in pythag_triple_lst:
        excl_cnt2 += 4 * (side_len_max // triple[2])
        #print(f"triple {triple} excluded count = {4 * (side_len_max // triple[2])}")
    #print(excl_cnt1, excl_cnt2)
    return res - excl_cnt2

# Problem 293
def pseudoFortunateNumberSum(n_max: int=10 ** 9 - 1) -> int:

    ps = SimplePrimeSieve()
    res = set()
    curr = SortedList()
    nxt = SortedList([1])
    #sieve = []
    #p_prod = 1
    cnt = 0
    for p in ps.endlessPrimeGenerator():
        if p * nxt[0] > n_max: break
        print(f"p = {p}")
        #if p < 13:
        #    sieve_len = len(sieve)
        #    for i in range(sieve_len + )
        curr = nxt
        nxt = SortedList()
        for i in itertools.count(0):
            if i >= len(curr): break
            num = curr[i]
            num2 = num * p
            if num2 > n_max: break
            cnt += 1
            curr.add(num2)
            nxt.add(num2)
            for num3 in itertools.count(num2 + 3, step=2):
                if ps.millerRabinPrimalityTestWithKnownBounds(num3, max_n_additional_trials_if_above_max=10)[0]:
                    #print(num2, num3, num3 - num2)
                    res.add(num3 - num2)
                    break
    print(f"total admissible numbers count = {cnt}")
    print(res)
    return sum(res)

# Problem 294
def isMultipleOfAndHasDigitSumEqualToNCountDigitDP(
    n: int=23,
    max_n_dig: int=11 ** 12,
    base: int=10,
    res_md: Optional[int]=10 ** 9,
) -> int:
    md_mapping = [-1] * n
    for num in range(n):
        num2 = (num * base) % n
        md_mapping[num] = num2
    
    md_mapping_binary_lift = [md_mapping]
    m2 = max_n_dig
    while m2 > 1:
        m2 >>= 1
        md_mapping_binary_lift.append([md_mapping_binary_lift[-1][x] for x in md_mapping_binary_lift[-1]])
    
    def getDigitMultipliedByBasePowerMod(d: int, exp: int) -> int:
        d %= n
        if not exp: return d
        exp2 = exp
        res = d
        for i in itertools.count(0):
            if exp2 & 1:
                res = md_mapping_binary_lift[i][res]
                if exp2 == 1: break
            exp2 >>= 1
        return res

    addMod = (lambda a, b: a + b) if res_md is None else (lambda a, b: (a + b) % res_md)
    
    memo = {}
    def recur(idx: int, remain: int, curr_md: int) -> int:
        if not remain:
            return not curr_md
        elif idx == max_n_dig - 1:
            if remain >= base: return 0
            d = remain
            d_contrib = getDigitMultipliedByBasePowerMod(d, idx)
            return not ((curr_md + d_contrib) % n)
        
        args = (idx, remain, curr_md)
        if args in memo.keys(): return memo[args]
        res = 0
        for d in range(min(base, remain + 1)):
            d_contrib = getDigitMultipliedByBasePowerMod(d, idx)
            res = addMod(res, recur(idx + 1, remain - d, ((curr_md + d_contrib) % n)))
        memo[args] = res
        return res
    
    res = recur(0, n, 0)
    return res

def isMultipleOfAndHasDigitSumEqualToNCount(
    n: int=23,
    max_n_dig: int=11 ** 12,
    base: int=10,
    res_md: Optional[int]=10 ** 9,
) -> int:
    """
    Solution to Project Euler #294
    """
    md_mapping = [-1] * n
    md_mapping_sources = set(range(n))
    for num in range(n):
        num2 = (num * base) % n
        md_mapping[num] = num2
        md_mapping_sources.discard(num2)
    #print(md_mapping_sources)
    md_num_locations = {}
    remain = set(range(n))
    md_paths = []
    for num0 in md_mapping_sources:
        num = num0
        md_path = []
        while num not in md_num_locations.keys():
            md_num_locations[num] = (len(md_paths), len(md_path))
            remain.remove(num)
            md_path.append(num)
            num = (num * base) % n
        md_paths.append((md_path, md_num_locations[num]))
    while remain:
        num0 = next(iter(remain))
        num = num0
        md_path = []
        while num not in md_num_locations.keys():
            md_num_locations[num] = (len(md_paths), len(md_path))
            remain.remove(num)
            md_path.append(num)
            num = (num * base) % n
        md_paths.append((md_path, md_num_locations[num]))
    #print(md_paths)
    cycle_starts = set()
    cycle_len = 1
    for i, (chain, insertion) in enumerate(md_paths):
        if insertion[0] != i: continue
        cycle_starts.add(chain[insertion[1]])
        length = len(chain) - insertion[1]
        cycle_len = lcm(cycle_len, length)
    start_chain_len = 0
    for i, (chain, _) in enumerate(md_paths):
        num = chain[0]
        for j in itertools.count(0):
            if num in cycle_starts: break
            num = md_mapping[num]
        start_chain_len = max(start_chain_len, j)
    print(f"start_chain_len = {start_chain_len}, cycle_len = {cycle_len}")

    addMod = (lambda a, b: a + b) if res_md is None else (lambda a, b: (a + b) % res_md)
    multMod = (lambda a, b: a * b) if res_md is None else (lambda a, b: (a * b) % res_md)
    sumMod = (lambda lst: sum(lst)) if res_md is None else (lambda lst: sum(lst) % res_md)

    curr_states = [{} for _ in range(n + 1)]
    curr_states[n] = {0: {0: 1}}
    curr_mults = list(range(min(n, base)))
    #print(curr_mults)
    for i in range(min(start_chain_len, max_n_dig)):
        #print(i)
        for dig_sm_rem in range(n + 1):
            if not curr_states[dig_sm_rem]: continue
            for d in range(1, min(base, dig_sm_rem + 1)):
                d2 = d % n
                mul = curr_mults[d2]
                dig_sm_rem2 = dig_sm_rem - d
                for r, occ_dict in curr_states[dig_sm_rem].items():
                    r2 = (r + mul) % n
                    curr_states[dig_sm_rem2].setdefault(r2, {})
                    #if dig_sm_rem2 == 8 and r2 == 16:
                    #    print(f"intermediary: {d}, {r}")
                    #if not dig_sm_rem2 and not r2 and occ_dict.get(0, 0):
                    #    print(f"solution found: {d}, {r}")
                    curr_states[dig_sm_rem2][r2][0] = addMod(curr_states[dig_sm_rem2][r2].get(0, 0), occ_dict.get(0, 0))
        #for j in range(n + 1):
        #    curr_states[j] = {x: {0: sumMod(y.values())} for x, y in curr_states[j].items()}
        #print(curr_states)
        curr_mults = [md_mapping[x] for x in curr_mults]
        #print(curr_mults)
    
    cnt_sm = 0
    n_opts_lst = []
    for i0 in range(start_chain_len, min(cycle_len + start_chain_len, max_n_dig)):
        n_opts = (max_n_dig - i0 - 1) // cycle_len + 1
        i = i0 % cycle_len
        #print(i0, i, n_opts)
        n_opts_lst.append(n_opts)
        cnt_sm += n_opts
        #if res_md is not None: n_opts %= res_md
        for d in range(1, base):
            d2 = d % n
            mul = curr_mults[d2]
            for dig_sm_rem in range(d, n + 1):
                f_mx = min(dig_sm_rem // d, n_opts)
                for f in range(1, f_mx + 1):
                    dig_sm_rem2 = dig_sm_rem - f * d
                    for r, occ_dict in curr_states[dig_sm_rem].items():
                        r2 = (r + mul * f) % n
                        for n_occ, cnt in occ_dict.items():
                            n_occ2 = n_occ + f
                            if n_occ2 > n_opts: continue
                            curr_states[dig_sm_rem2].setdefault(r2, {})
                            curr_states[dig_sm_rem2][r2][n_occ2] = addMod(curr_states[dig_sm_rem2][r2].get(n_occ2, 0), multMod(cnt, math.comb(n_opts - n_occ, f)))
        for j in range(n + 1):
            curr_states[j] = {x: {0: sumMod(y.values())} for x, y in curr_states[j].items()}
        #if i < 5:
        #    print(curr_states)

        curr_mults = [md_mapping[x] for x in curr_mults]
    #print(f"max_n_dig = {max_n_dig}, count sum = {cnt_sm}")
    #print(n_opts_lst)
    res = curr_states[0].get(0, {}).get(0, 0)
    if res_md is not None: res %= res_md
    return res

def isMultipleOfAndHasDigitSumEqualTo23CountDigitDP(
    max_n_dig: int=11 ** 12,
    res_md: Optional[int]=10 ** 9,
) -> int:
    n = 23
    base = 10
    md_mapping = [-1] * n
    for num in range(n):
        num2 = (num * base) % n
        md_mapping[num] = num2
    
    md_mapping_binary_lift = [md_mapping]
    m2 = max_n_dig
    while m2 > 1:
        m2 >>= 1
        md_mapping_binary_lift.append([md_mapping_binary_lift[-1][x] for x in md_mapping_binary_lift[-1]])
    
    def getDigitMultipliedByBasePowerMod(d: int, exp: int) -> int:
        d %= n
        if not exp: return d
        exp2 = exp
        res = d
        for i in itertools.count(0):
            if exp2 & 1:
                res = md_mapping_binary_lift[i][res]
                if exp2 == 1: break
            exp2 >>= 1
        return res

    addMod = (lambda a, b: a + b) if res_md is None else (lambda a, b: (a + b) % res_md)
    
    memo = {}
    def recur(idx: int, remain: int, curr_md: int) -> int:
        if not remain:
            return not curr_md
        elif idx == max_n_dig - 1:
            if remain >= base: return 0
            d = remain
            d_contrib = getDigitMultipliedByBasePowerMod(d, idx)
            return not ((curr_md + d_contrib) % n)
        
        args = (idx, remain, curr_md)
        if args in memo.keys(): return memo[args]
        res = 0
        for d in range(min(base, remain + 1)):
            d_contrib = getDigitMultipliedByBasePowerMod(d, idx)
            res = addMod(res, recur(idx + 1, remain - d, ((curr_md + d_contrib) % n)))
        memo[args] = res
        return res
    
    res = recur(0, n, 0)
    return res

def isMultipleOfAndHasDigitSumEqualTo23Count(
    max_n_dig: int=11 ** 12,
    res_md: Optional[int]=10 ** 9,
) -> int:
    n = 23
    base = 10
    md_mapping = [-1] * n
    for num in range(n):
        num2 = (num * base) % n
        #print(n, num)
        md_mapping[num] = num2
    #print(md_mapping)

    #nonzero_cycle_len = 22

    addMod = (lambda a, b: a + b) if res_md is None else (lambda a, b: (a + b) % res_md)
    multMod = (lambda a, b: a * b) if res_md is None else (lambda a, b: (a * b) % res_md)
    sumMod = (lambda lst: sum(lst)) if res_md is None else (lambda lst: sum(lst) % res_md)

    curr_states = [{} for _ in range(n + 1)]
    curr_states[n] = {0: {0: 1}}
    curr_mults = list(range(min(n, base)))
    cnt_sm = 0
    n_opts_lst = []
    for i in range(min(n - 1, max_n_dig)):
        n_opts = (max_n_dig - i - 1) // (n - 1) + 1
        #print(i, n_opts)
        n_opts_lst.append(n_opts)
        cnt_sm += n_opts
        #if res_md is not None: n_opts %= res_md
        for d in range(1, base):
            d2 = d % n
            mul = curr_mults[d2]
            for dig_sm_rem in range(d, n + 1):
                f_mx = min(dig_sm_rem // d, n_opts)
                for f in range(1, f_mx + 1):
                    dig_sm_rem2 = dig_sm_rem - f * d
                    for r, occ_dict in curr_states[dig_sm_rem].items():
                        r2 = (r + mul * f) % n
                        for n_occ, cnt in occ_dict.items():
                            n_occ2 = n_occ + f
                            if n_occ2 > n_opts: continue
                            curr_states[dig_sm_rem2].setdefault(r2, {})
                            curr_states[dig_sm_rem2][r2][n_occ2] = addMod(curr_states[dig_sm_rem2][r2].get(n_occ2, 0), multMod(cnt, math.comb(n_opts - n_occ, f)))

                #dig_sm_rem2 = dig_sm_rem - d
                #for r, f in curr_states[dig_sm_rem].items():
                #    r2 = (r + mul) % n
                #    curr_states[dig_sm_rem2][r2] = addMod(curr_states[dig_sm_rem2].get(r2, 0), multMod(cnt, f))
        for j in range(n + 1):
            curr_states[j] = {x: {0: sumMod(y.values())} for x, y in curr_states[j].items()}
        #if i < 5:
        #    print(curr_states)

        curr_mults = [md_mapping[x] for x in curr_mults]
    #print(f"max_n_dig = {max_n_dig}, count sum = {cnt_sm}")
    #print(n_opts_lst)
    res = curr_states[0].get(0, {}).get(0, 0)
    if res_md is not None: res %= res_md
    return res

# Problem 295
def calculateCircularSegmentsWithEndsAtLatticePointsContainingNoLatticePointsDisplacements(
    rad_sq: int,
    ps: Optional[PrimeSPFsieve]=None,
) -> List[Tuple[int, int]]:
    
    def segmentContainsALatticePoint(p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        # Note that points along the straight edge of the segment (excluding
        # the two end points) are consdered to be inside but points on the
        # curved edge (including the two end points) are considered to be
        # outside
        
        if abs(p1[0] - p2[0]) > abs(p1[1] - p2[1]):
            p1 = (p1[1], p1[0])
            p2 = (p2[1], p2[0])
        p1, p2 = sorted([p1, p2])
        v = [x - y for x, y in zip(p2, p1)]
        #print(v, gcd(*v))
        if abs(gcd(*v)) > 1:
            return True
        #print(p1, p2)
        for x in range(p1[0] + 1, p2[0]):
            y_mn = p1[1] + (((x - p1[0]) * v[1] - 1) // v[0]) + 1
            y_mx = isqrt(rad_sq - x * x)
            #print(x, (y_mn, y_mx))
            if y_mn <= y_mx: return True
        #print("does not contain a lattice point")
        return False

    pair_lst = sorted([tuple(sorted(x)) for x in sumOfTwoSquaresSolutionGenerator(rad_sq, ps=ps)])
    if not pair_lst: return []
    m = len(pair_lst)
    #print(pair_lst)
    res = []
    i1 = 0
    for i2 in range(1, m):
        p2 = pair_lst[i2]
        for i1 in range(i1, i2):
            if not segmentContainsALatticePoint(pair_lst[i1], p2): break
        else: i1 = i2
        #print(i1, i2)
        for i1_2 in range(i1, i2):
            #print(1, p2, pair_lst[i1_2])
            res.append(tuple(sorted([abs(x - y) for x, y in zip(p2, pair_lst[i1_2])])))
        #if i1 < i2:
        #    print(p2, pair_lst[i2 - 1])
        #    res.append(tuple(sorted([abs(x - y) for x, y in zip(p2, pair_lst[i2 - 1])])))
    m2 = m - (pair_lst[-1][0] == pair_lst[-1][1])
    for i2 in reversed(range(m2)):
        p2 = (pair_lst[i2][1], pair_lst[i2][0])
        for i1 in range(i1, m):
            if not segmentContainsALatticePoint(pair_lst[i1], p2): break
        else: break
        #print(i1, i2)
        #if i2 == m - 1:
        #    print(p2, pair_lst[i2])
        #    res.append(tuple(sorted([abs(x - y) for x, y in zip(p2, pair_lst[i2])])))
        #else:
        #    print(p2, (pair_lst[i2 + 1][1], pair_lst[i2 + 1][0]))
        #    res.append(tuple(sorted([abs(x - y) for x, y in zip(p2, (pair_lst[i2 + 1][1], pair_lst[i2 + 1][0]))])))
        for i1_2 in range(i1, m):
            #print(2, p2, pair_lst[i1_2])
            res.append(tuple(sorted([abs(x - y) for x, y in zip(p2, pair_lst[i1_2])])))
        for i1_2 in reversed(range(i2 + 1, m2)):
            #print(3, p2, (pair_lst[i1_2][1], pair_lst[i1_2][0]))
            res.append(tuple(sorted([abs(x - y) for x, y in zip(p2, (pair_lst[i1_2][1], pair_lst[i1_2][0]))])))
    return sorted(res)
    # sumOfTwoSquaresSolutionGenerator(
    #    target: int,
    #    ps: Optional[PrimeSPFsieve]=None,
    #)

def calculateNumberOfRadiusCombinationsCanMakeLenticularHoleBruteForce(
    rad_max: int=10 ** 5,
    ps: Optional[PrimeSPFsieve]=None,
) -> int:
    vec_sets = []
    for rad_sq in range(1, rad_max ** 2 + 1):
        vecs = calculateCircularSegmentsWithEndsAtLatticePointsContainingNoLatticePointsDisplacements(
            rad_sq,
            ps=None,
        )
        vec_sets.append(set(vecs))
    res = 0
    for i1 in range(len(vec_sets) - 1):
        for i2 in range(i1, len(vec_sets)):
            res += not vec_sets[i1].isdisjoint(vec_sets[i2])
    return res

def calculateNumberOfRadiusCombinationsCanMakeLenticularHole(
    rad_max: int=10 ** 5,
    ps: Optional[PrimeSPFsieve]=None,
) -> int:
    """
    Solution to Project Euler #295
    """

    """
    rad_sq_mx = rad_max * rad_max
    res = 0
    seen_cnts = {}
    for rad_sq in range(1, rad_sq_mx + 1):
        if not rad_sq % 100000:
            print(f"rad_sq = {rad_sq} of {rad_sq_mx}")
        v_lst = calculateCircularSegmentsWithEndsAtLatticePointsContainingNoLatticePointsDisplacements(
            rad_sq,
            ps=ps,
        )
        
        if not v_lst: continue
        print(rad_sq, v_lst)
        res += 1
        for v in v_lst:
            seen_cnts.setdefault(v, 0)
            res += seen_cnts[v]
            seen_cnts[v] += 1
        #print(rad_sq, v_lst, seen_cnts, res)
    return res

    """
    rad_sq_mx = rad_max * rad_max
    res = 0
    seen_cnts = {}
    vecs = []
    vec_dict = {}
    for rad_sq in range(1, rad_sq_mx + 1):
        if not rad_sq % 100000:
            print(f"rad_sq = {rad_sq} of {rad_sq_mx}")
        v_lst = calculateCircularSegmentsWithEndsAtLatticePointsContainingNoLatticePointsDisplacements(
            rad_sq,
            ps=ps,
        )
        if not v_lst: continue
        res += 1
        bm = 0
        for v in v_lst:
            if v in vec_dict.keys():
                i = vec_dict[v]
            else:
                i = len(vecs)
                vecs.append(v)
                vec_dict[v] = i
            bm |= 1 << i
        bm2 = bm
        while bm2:
            seen_cnts.setdefault(bm2, 0)
            res += seen_cnts[bm2] if (bm2.bit_count() & 1) else -seen_cnts[bm2]
            seen_cnts[bm2] += 1
            bm2 = (bm2 - 1) & bm
        #print(rad_sq, v_lst, seen_cnts, res)
    #print(vecs)
    #print(seen_cnts)
    return res

# Problem 297
def zeckendorfRepresentationTermCount(n_max: int=10 ** 17 - 1) -> int:
    """
    Solution to Project Euler #297
    """
    cnt_cumu = [0, 1, 2]
    fib = [1, 2]
    while True:
        num = fib[-1] + fib[-2]
        if num > n_max: break
        fib.append(num)
        cnt_cumu.append(cnt_cumu[-1] + cnt_cumu[-2] + fib[-2])
    #print(cnt_cumu)
    #print(fib)
    n_ub_repr = []
    num = n_max + 1
    i = len(fib) - 1
    while num:
        #print(f"i = {i}, num = {num}")
        if num >= fib[i]:
            n_ub_repr.extend([1, 0])
            num -= fib[i]
            i -= 2
            continue
        n_ub_repr.append(0)
        i -= 1
    if i == -2: n_ub_repr.pop()
    elif i >= 0:
        n_ub_repr.extend([0] * (i + 1))
    n_ub_repr = n_ub_repr[::-1]
    res = 0
    d_cnt = 0
    #print(n_ub_repr)
    #print(len(cnt_cumu), len(fib), len(n_ub_repr))
    for i in reversed(range(len(n_ub_repr))):
        if not n_ub_repr[i]: continue
        #print(i, len(fib))
        res += d_cnt * fib[i] + cnt_cumu[i]
        d_cnt += 1
    
    return res

# Problem 298
def memoryGameStrategyExpectedAbsoluteDifferenceSimulation(
    n: int,
    mem_size: int,
    n_turns: int,
    n_sim: int,
) -> Tuple[float, float]:
    mean = 0
    var = 0
    for _ in range(n_sim):
        curr = 0
        lst1 = SortedList()
        incl1 = {}
        lst2 = SortedList()
        incl2 = {}
        for i in range(n_turns):
            num = random.randrange(1, n + 1)
            if num not in incl1.keys():
                lst1.add((i, num))
                incl1[num] = i
            else: curr += 1
            if num in incl2.keys():
                lst2.remove((incl2[num], num))
                curr -= 1
            lst2.add((i, num))
            incl2[num] = i
            if len(incl1) == mem_size: break
        for i in range(i + 1, n_turns):
            num = random.randrange(1, n + 1)
            if num not in incl1.keys():
                i2, num2 = lst1.pop(0)
                incl1.pop(num2)
                lst1.add((i, num))
                incl1[num] = i
            else: curr += 1
            if num in incl2.keys():
                lst2.remove((incl2[num], num))
                curr -= 1
            else:
                i2, num2 = lst2.pop(0)
                incl2.pop(num2)
            lst2.add((i, num))
            incl2[num] = i
        
        mean += abs(curr)
        var += curr ** 2
    mean /= n_sim
    var /= n_sim
    var -= mean ** 2
    var *= (n_sim - 1) / n_sim
    return (mean, math.sqrt(var * (n_sim - 1)) / n_sim)


def memoryGameStrategyExpectedAbsoluteDifferenceFraction(
    n: int,
    mem_size: int,
    n_turns: int,
) -> CustomFraction:

    memo = {}
    def transferFunction(state: Tuple) -> Dict[Tuple, Dict[int, CustomFraction]]:
        args = state
        if args in memo.keys(): return memo[args]
        res = {}
        state_lst = list(state)

        # An integer present in Larry's memory is chosen
        
        #state_tup = tuple(state)
        #res.setdefault(state_tup, {})
        #res[state_tup][0] = res[state_tup].get(0, CustomFraction(0, 1)) + CustomFraction(1, n)
        #z_pos_lst = []
        missing_set = set(range(1, mem_size + 1))
        for i in reversed(range(len(state_lst))):
            if state_lst[i]: missing_set.remove(state_lst[i])
            state_lst[i], state_lst[-1] = state_lst[-1], state_lst[i]
            # Note that if 0 encountered, state must be length n.
            net_score = int(not state_lst[-1])
            state_tup = tuple(state_lst) if state_lst[-1] else tuple([max(0, x - 1) for x in state_lst[:-1]] + [len(state_lst)])
            #if not state[-1]:
            #    z_pos_lst.append(i)
            #    continue
            res.setdefault(state_tup, {})
            res[state_tup][net_score] = res[state_tup].get(net_score, CustomFraction(0, 1)) + CustomFraction(1, n)
        if len(state) > 1:
            # putting state_lst back into its original configuration
            state_lst = list(state)
        #if state == (0, 2):
        #    print(f"for state {state} missing_set = {missing_set}, current transfer function = {res}")
        if len(state_lst) < mem_size:
            # The number of distinct integers encountered after this
            # turn is no greater than the memoery size, meaning that the
            # set of integers in memory is the same for both after this
            # turn and neither player needs to remove an integer from
            # memory
            net_score = 0
            state_tup = tuple(state_lst + [len(state_lst) + 1])
            res.setdefault(state_tup, {})
            res[state_tup][net_score] = res[state_tup].get(net_score, CustomFraction(0, 1)) + CustomFraction(n - len(state_lst), n)
            memo[args] = res
            return res
        
        # Both memories are full

        # An integer absent from both memories is chosen
        net_score = 0
        state_tup = tuple([max(0, x - 1) for x in state_lst[1:]] + [mem_size])
        res.setdefault(state_tup, {})
        n_choice = n - len(state_lst) - len(missing_set)
        res[state_tup][net_score] = res[state_tup].get(net_score, CustomFraction(0, 1)) + CustomFraction(n_choice, n)
        #if state == (0, 2):
        #    print(f"for state {state} missing_set = {missing_set}, current transfer function = {res}")
        
        # An integer present in Robin's memory but not
        # Larry's memory is chosen
        #missing_lst = sorted(missing_set)
        net_score = -1
        for num in missing_set:
            #state_tup = tuple([max(0, x - (x > num)) for x in state_lst[1:]] + [mem_size])
            state_tup = tuple(state_lst[1:] + [num])
            res.setdefault(state_tup, {})
            res[state_tup][net_score] = res[state_tup].get(net_score, CustomFraction(0, 1)) + CustomFraction(1, n)
        #if state == (0, 2):
        #    print(f"for state {state} missing_set = {missing_set}, current transfer function = {res}")
        memo[args] = res
        return res
    
    seen_states = set()
    curr_states = {(): {0: CustomFraction(1, 1)}}
    for i in range(n_turns):
        #print(curr_states)
        res = CustomFraction(0, 1)
        for state, exp_dict in curr_states.items():
            for exp, p in exp_dict.items():
                res += abs(exp) * p
        print(f"probability sum = {sum(sum(exp_dict.values()) for exp_dict in curr_states.values())}, number of states = {len(curr_states)}, expected value after {i} turns = {res} ({res.numerator / res.denominator})")
        print(f"turn {i + 1}")
        
        prev_states = curr_states
        curr_states = {}
        for state, exp_dict in prev_states.items():
            seen_states.add(state)
            for exp, p in exp_dict.items():
                transf_dict = transferFunction(state)
                p_sm = sum(sum(exp_dict2.values()) for exp_dict2 in transf_dict.values())
                if p_sm != 1:
                    print(f"transfer function for state {state} does not sum to 1 but rather {p_sm}")
                for state2, exp_dict2 in transf_dict.items():
                    
                    #if state == (1, 2, 3):
                    #    print(state2)
                    for exp2, p2 in exp_dict2.items():
                        exp3 = exp + exp2
                        p3 = p * p2
                        curr_states.setdefault(state2, {})
                        curr_states[state2][exp3] = curr_states[state2].get(exp3, 0) + p3
    #print(curr_states)
    print(f"probability sum = {sum(sum(exp_dict.values()) for exp_dict in curr_states.values())}, number of states = {len(curr_states)}")

    #print("transfer function values:")
    #for state in seen_states:
    #    print(f"{state}: {transferFunction(state)}")
    res = CustomFraction(0, 1)
    for state, exp_dict in curr_states.items():
        for exp, p in exp_dict.items():
            res += abs(exp) * p
    return res

def memoryGameStrategyExpectedAbsoluteDifferenceFloat(
    n: int=10,
    mem_size: int=5,
    n_turns: int=50,
) -> float:
    """
    Solution to Project Euler #298
    """

    res = memoryGameStrategyExpectedAbsoluteDifferenceFraction(
        n=n,
        mem_size=mem_size,
        n_turns=n_turns,
    )
    print(res)
    return res.numerator / res.denominator

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
            orthocentre=(5, 0),
            perimeter_max=10 ** 3,
            ps=None,
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
            polynomial_num_max=10 ** 16,
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

    if 275 in eval_nums:
        since = time.time()
        res = countBalancedPolyominoSculptures(n_tiles=18)
        print(f"Solution to Project Euler #275 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 276 in eval_nums:
        since = time.time()
        res = countPrimitiveTriangles(perim_max=10 ** 7)
        print(f"Solution to Project Euler #276 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 277 in eval_nums:
        since = time.time()
        res = modifiedCollatzSequenceSmallestStartWithSequence(n_min=10 ** 15 + 1, seq="UDDDUdddDDUDDddDdDddDDUDDdUUDd")
        print(f"Solution to Project Euler #277 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 278 in eval_nums:
        since = time.time()
        res = calculateDistinctPrimeCombinationsFrobeniusNumberSum(n_p=3, p_max=4999)
        print(f"Solution to Project Euler #278 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 279 in eval_nums:
        since = time.time()
        res = integerSidedTrianglesWithIntegerAngleCount(
            max_perimeter=10 ** 8,
            n_degrees_in_circle=360,                  
        )
        print(f"Solution to Project Euler #279 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 280 in eval_nums:
        since = time.time()
        res = antRandomWalkExpectedNumberOfStepsFloatDirect(
            n_rows=5,
            n_cols=5,
            start=(2, 2),
        )
        print(f"Solution to Project Euler #280 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 281 in eval_nums:
        since = time.time()
        res = pizzaToppingsSum(max_count=10 ** 15)
        print(f"Solution to Project Euler #281 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 282 in eval_nums:
        since = time.time()
        res = ackermannFunctionModuloSum(n_max=6, md=14 ** 8)
        print(f"Solution to Project Euler #282 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 283 in eval_nums:
        since = time.time()
        res = heronianTrianglesWithIntegerAreaPerimeterRatioPerimeterSum(
            area_perimeter_ratio_max=1000,
        )
        print(f"Solution to Project Euler #283 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 284 in eval_nums:
        since = time.time()
        res = steadySquaresDigitSumBaseRepr(max_n_digs=10 ** 4, base=14)
        print(f"Solution to Project Euler #284 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 285 in eval_nums:
        since = time.time()
        res = calculatePythagoreanOddsRangeGameExpectedValue(k_min=1, k_max=10 ** 5)
        print(f"Solution to Project Euler #285 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 286 in eval_nums:
        since = time.time()
        res = exactBasketballScoreProbability(
            p=0.02,
            d_min=1,
            d_max=50,
            total_score=20,
            eps=1e-12,
        )
        print(f"Solution to Project Euler #286 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 287 in eval_nums:
        since = time.time()
        res = singleCircleQuadTreeEncodingMinimumLength(
            image_size_pow2=24,
            circle_centre=(1 << 23, 1 << 23),
            circle_radius_sq=1 << 46,
        )
        print(f"Solution to Project Euler #287 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 288 in eval_nums:
        since = time.time()
        res = squareModSeriesFactorialPrimeFactorCount(
            p=61,
            s0=290797,
            s_md=50515093,
            series_max_pow=10 ** 7,
            res_md_pow=10,
        )
        print(f"Solution to Project Euler #288 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 289 in eval_nums:
        since = time.time()
        res = circleArrayEulerianNonCrossingCycleCount(
            n_rows=7,
            n_cols=10,
            res_md=10 ** 10,
        )
        print(f"Solution to Project Euler #289 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 290 in eval_nums:
        since = time.time()
        res = digitSumEqualsMultipleDigitSumCount(
            n_dig_max=18,
            mult=137,
            base=10,
        )
        print(f"Solution to Project Euler #290 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 291 in eval_nums:
        since = time.time()
        res = panaitopolPrimesCount(p_max=5 * 10 ** 15 - 1)
        print(f"Solution to Project Euler #291 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 292 in eval_nums:
        since = time.time()
        res = pythagoreanPolygonCount(perim_max=120)
        print(f"Solution to Project Euler #292 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 293 in eval_nums:
        since = time.time()
        res = pseudoFortunateNumberSum(n_max=10 ** 9 - 1)
        print(f"Solution to Project Euler #293 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 294 in eval_nums:
        since = time.time()
        res = isMultipleOfAndHasDigitSumEqualToNCount(
            n=23,
            max_n_dig=11 ** 12,
            base=10,
            res_md=10 ** 9,
        )
        print(f"Solution to Project Euler #294 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 295 in eval_nums:
        since = time.time()
        res = calculateNumberOfRadiusCombinationsCanMakeLenticularHole(
            rad_max=10 ** 5,
            ps=None,
        )
        print(f"Solution to Project Euler #294 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 297 in eval_nums:
        since = time.time()
        res = zeckendorfRepresentationTermCount(n_max=10 ** 17 - 1)
        print(f"Solution to Project Euler #297 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 298 in eval_nums:
        since = time.time()
        res = memoryGameStrategyExpectedAbsoluteDifferenceFloat(
            n=10,
            mem_size=5,
            n_turns=50,
        )
        print(f"Solution to Project Euler #298 = {res}, calculated in {time.time() - since:.4f} seconds")

    print(f"Total time taken = {time.time() - since0:.4f} seconds")

if __name__ == "__main__":
    eval_nums = {295}
    evaluateProjectEulerSolutions251to300(eval_nums)

"""
n = 24
base = 10
res_md = None
for max_n_dig in range(1, 51):
    ans1 = isMultipleOfAndHasDigitSumEqualToNCount(n=n, max_n_dig=max_n_dig, base=base, res_md=res_md)
    ans2 = isMultipleOfAndHasDigitSumEqualToNCountDigitDP(n=n, max_n_dig=max_n_dig, base=base, res_md=res_md)
    print(max_n_dig, ans1, ans2, ans1 == ans2)
"""

#print(calculateCircularSegmentsWithEndsAtLatticePointsContainingNoLatticePointsDisplacements(
#    rad_sq=25,
#    ps=None,
#))