#!/usr/bin/env python

from typing import (
    Dict,
    List,
    Tuple,
    Set,
    Union,
    Generator,
    Optional,
    Any,
)

import bisect
import heapq
import itertools
import math
import numpy as np
import time

from collections import deque, defaultdict
from sortedcontainers import SortedList, SortedSet

from data_structures.fractions import CustomFraction
from data_structures.prime_sieves import PrimeSPFsieve, SimplePrimeSieve

from algorithms.number_theory_algorithms import (
    gcd,
    lcm,
    isqrt,
    integerNthRoot,
)
from algorithms.pseudorandom_number_generators import (
    generalisedLaggedFibonacciGenerator,
    blumBlumShubPseudoRandomGenerator,
)
from algorithms.geometry_algorithms import grahamScan

# Problem 201
def subsetsWithUniqueSumTotal(nums: Set[int], k: int) -> int:
    """
    Given a set of integers nums, finds the sum of all integers for
    which there is exactly one subset of nums with size k whose sum
    is equal to that integer.

    Args:
        Required positional:
        nums (set of ints): The set of integers of interest. Note
                that this cannot contain repeated elements.
        k (int): The size of the subsets of nums for which an integer
                included in the sum can be the sum of exactly one
                of these subsets.
    
    Returns:
    Integer (int) giving the sum of all integers for which there is
    exactly one subset of nums with size k whose sum is equal to that
    integer.

    Outline of rationale:
    This is solved using bottom up dynamic programming, for integer m
    increasing from 0 to n finding the subset sums including the first m
    elements of the set, by adding the m:th element of nums to the
    possible subset sums including the first (m - 1) elements (i.e.
    the result of the previous step), keeping track of the number of
    included elements (capping this at k) and whether that sum for that
    number of included elements can be achieved in more than one way.
    This is optimised for k > len(nums) / 2 by noting that if an integer
    a is a unique sum among subsets of size k, then as its complement,
    (sum(nums) - a) is a unique sum among subsets of size (len(nums) - k).
    We can additionally save space by noting that sums of sets including
    more elements cannot by inserting further elements affect the sums
    of sets including fewer elements. This allows us to maintain a single
    structure with the results, taking care when adding a new element
    to iterate over the structure in decreasing order of number of
    included elements.
    Furthermore, for the final elements added, we need only focus on
    those subsets that have a number of elements that stand a chance
    of reaching the required number of elements (so those for which,
    even if all the rest of the elements to be inserted are included
    would still result in fewer than the required number of elements
    can be ignored).
    """
    n = len(nums)
    rev = False
    if (k << 1) > n:
        k = n - k
        rev = True
    if k < 0: return 0
    elif not k: return 1
    nums2 = sorted(nums)
    curr = [{0: True}]
    for j, num in enumerate(nums2):
        if len(curr) <= k: curr.append({})
        for i in reversed(range(max(0, k - (n - j)), len(curr) - 1)):
            for num2, b in curr[i].items():
                num3 = num + num2
                curr[i + 1][num3] = b and num3 not in curr[i + 1].keys()
    if len(curr) <= k: return 0
    res = sum(x for x, y in curr[k].items() if y)
    return sum(nums) * sum(curr[k].values()) - res if rev else res

def subsetsOfSquaresWithUniqueSumTotal(n_max: int=100, k: int=50) -> int:
    """
    Solution to Project Euler #201

    Given a set of all perfect squares from 1 up to n_max ** 2 inclusive,
    finds the sum of all integers for which there is exactly one subset
    of nums with size k whose sum is equal to that integer.

    Args:
        Optional named:
        n_max (int): Integer whose square is the largest perfect
                square to be included in the set.
            Defualt: 100
        k (int): The size of the subsets of the set of perfect squares
                from 1 up to n_max ** 2 inclusve for which an integer
                included in the sum can be the sum of exactly one
                of these subsets.
            Default: 50
    
    Returns:
    Integer (int) giving the sum of all integers for which there is
    exactly one subset of the set of perfect squares from 1 up to
    n_max ** 2 inclusive with size k whose sum is equal to that
    integer.

    Outline of rationale:
    See outline of rationale for subsetsWithUniqueSumTotal().
    """
    nums = {x ** 2 for x in range(1, n_max + 1)}
    res = subsetsWithUniqueSumTotal(nums, k)
    return res

# Problem 202
def equilateralTriangleReflectionCountNumberOfWays(
    n_reflect: int=12017639147,
) -> int:
    """
    Solution to Project Euler #202

    Calculates the number of ways that for a reflective
    equilateral triangle boundary, a beam can be emitted
    from one of its vertices to the interior of the triangle,
    reflect exactly n_reflect times without encountering a
    vertex of the triangle before returning to that same vertex.

    Reflections of the beam from the boundary are occur in the
    standard fashion in which the angle of incidence is equal
    to the angle of reflection (i.e. when reflecting, the
    component parallel to the boundary is unchanged but the
    component normal to the boundary is reversed, becoming
    its additive inverse).

    Args:
        Optional named:
        n_reflect (int): Strictly positive integer giving the
                exact number of reflections from the boundary
                that should occur without encountering any
                vertex before the beam returns to the original
                vertex for beam paths considered in the count.
            Default: 12017639147
    
    Returns:
    Integer (int) giving the number of ways that for a reflective
    equilateral triangle boundary, a beam can be emitted
    from one of its vertices to the interior of the triangle,
    reflect exactly n_reflect times without encountering a
    vertex of the triangle before returning to that same vertex.

    Outline of rationale:
    Given that equilateral triangles tile the plane, this problem
    can be shown to be equivalent to counting the number of
    lines on a plane tiled by equilateral triangles that for
    one of the triangles and one of its vertices passes through
    the vertex and the opposite edge of the triangle and
    in that direction intersects exactly n_reflect edges before
    next intersecting with a vertex that can be reached by
    repeated reflections of the original vertex in the lines
    defined by the edges of the triangles, without intersecting
    with any other vertices first.

    TODO
    """
    # Review documentation for clarity

    if not n_reflect & 1: return 0
    md3 = n_reflect % 3
    if not md3: return 0
    if n_reflect == 1: return 1

    def primeFactors(num: int) -> Set[int]:
        res = set()
        num2 = num
        if not num2 & 1:
            res.add(2)
            num2 >>= 1
            while not num2 & 1:
                num2 >>= 1
        for p in range(3, num + 1, 2):
            if p ** 2 > num2: break
            num3, r = divmod(num2, p)
            if r: continue
            res.add(p)
            num2 = num3
            num3, r = divmod(num2, p)
            while not r:
                num2 = num3
                num3, r = divmod(num2, p)
        if num2 != 1: res.add(num2)
        return res

    res = 0
    target = (n_reflect + 3) >> 1

    pf = sorted(primeFactors(target))
    print("factored")
    print(pf)
    pf_md3 = {p: p % 3 for p in pf}

    res = 0
    for bm in range(1 << len(pf)):
        neg = False
        num = 1
        bm2 = bm
        md3_2 = 1
        for i, p in enumerate(pf):
            if not bm2: break
            if bm2 & 1:
                neg = not neg
                num *= p
                md3_2 = (md3_2 * pf_md3[p]) % 3
            bm2 >>= 1
        ans = target // num
        md3_2 = (md3_2 * md3) % 3
        ans = (ans - md3_2) // 3

        res += -ans if neg else ans
    return res
    """
    cnt = 0
    for m in range(md3, target >> 1, 3):
        cnt += 1
        if not cnt % 1000000: print(m, target >> 1)
        for p in pf:
            if not m % p: break
        else: res += 1
    return res << 1
    """

def distinctSquareFreeBinomialCoefficientSum(n_max: int=51) -> int:
    """
    Solution to Project Euler #203

    Calculates the sum of distinct square free integers that are equal
    to a binomial coefficients (n choose k) for which n does not exceed
    n_max and 0 <= k <= n.

    An integer is square free if and only if it is strictly positive
    and is not divisible by the square of any prime number.

    Args:
        Optional named:
        n_max (int): The largest value of n for which square free integers
                equal to a binomial coefficient (n choose k) is included
                in the sum.
            Default: 51
    
    Returns:
    Integer (int) giving the sum of distinct square free integers that are
    equal to a binomial coefficients (n choose k) for which n does not
    exceed n_max and 0 <= k <= n.

    Outline of rationale:
    We simply iterate over the binomial coefficients in question, checking
    whether it is square free, utilising the following optimisations:
     1) Given that (n choose k) = n! / (k! * (n - k)!), and n! is not
        divisible by any prime greater than n, such a binomial coefficient
        cannot be divisible by any prime greater than n, and so cannot
        be divisible by the square of any prime number greater than n.
        Therefore, to check if a binomial coefficient is square free
        we only need to check if it is divisible by the square of the
        primes not exceeding n.
     2) Given that (n choose (n - k)) = (n choose k), for each n we
        need only check the values of k not exceeding n / 2.
    """
    if n_max < 0: return 0
    ps = SimplePrimeSieve(n_max - 1)
    p_lst = ps.p_lst
    res = {1}
    curr = [1]
    for n in range(1, n_max):
        
        prev = curr
        curr = [1]
        i_mx = bisect.bisect_right(p_lst, n)
        for k in range(1, len(prev)):
            num = prev[k - 1] + prev[k]
            curr.append(num)
            if num in res: continue
            for i in range(i_mx):
                p = p_lst[i]
                if not num % p ** 2: break
            else: res.add(num)
        #print(curr)
        if n & 1: curr.append(curr[-1])
    return sum(res)

# Problem 204
def generalisedHammingNumberCount(typ: int=100, n_max: int=10 ** 9) -> int:
    """
    Solution to Project Euler #204

    Calculates the number of generalised Hamming numbers of type typ not
    exceeding n_max.

    A generalised Hamming number of type n (where n is a strictly positive
    integer) is a strictly positive integer with no prime factor greater than
    n.

    Args:
        Optional named:
        typ (int): The type of generalised Hamming number to be counted.
            Default: 100
        n_max (int): Integer giving the upper bound on the generalised Hamming
                numbers of type typ to be included in the count.
            Default: 10 ** 9
    
    Returns:
    Integer (int) giving the number of generalised Hamming numbers of type
    typ not exceeding n_max.
    """
    if n_max <= 0: return 0
    ps = SimplePrimeSieve(typ)
    p_lst = ps.p_lst
    if not p_lst: return 1

    n_p = len(p_lst)
    print(p_lst)

    memo = {}
    def recur(idx: int, mx: int) -> int:
        if idx == n_p: return 1
        args = (idx, mx)
        if args in memo.keys(): return memo[args]
        p = p_lst[idx]
        mx2 = mx
        res = 0
        while mx2 > 0:
            res += recur(idx + 1, mx2)
            mx2 //= p
        memo[args] = res
        return res
    
    res = recur(0, n_max)
    #print(memo)
    return res

# Problem 205
def probabilityDieOneSumWinsFraction(
    die1_face_values: List[int],
    n_die1: int,
    die2_face_values: List[int],
    n_die2: int
) -> CustomFraction:
    """
    For two players, player 1 with n_die1 dice each with face values
    die1_face_values and player 2 with n_die2 dice each face values
    die2_face_values, where for each die the probability of each
    face value occurring on a given roll is equal, each player rolls
    all of their dice and adds the total. The player with the highest
    total wins, with equal totals resulting in a draw. This function
    calculates the probability on a given round that player 1 will
    win as a rational number.

    Args:
        Required positional:
        die1_face_values (list of ints): List of integers giving the
                face values of the dice rolled by player 1, whose
                elements each have an equal probability of being
                rolled. Note that repeated values are allowed.
        n_die1 (int): Strictly positive integer giving the number
                of dice that player 1 rolls.
        die2_face_values (list of ints): List of integers giving the
                face values of the dice rolled by player 2, whose
                elements each have an equal probability of being
                rolled. Note that repeated values are allowed.
        n_die2 (int): Strictly positive integer giving the number
                of dice that player 2 rolls.
    
    Returns:
    CustomFraction object giving the probability that for a given
    roll player 1 wins, represented as a fraction.
    """
    d1 = {}
    for num in die1_face_values:
        d1[num] = d1.get(num, 0) + 1
    d2 = {}
    for num in die2_face_values:
        d2[num] = d2.get(num, 0) + 1
    n1, n2 = len(die1_face_values), len(die2_face_values)

    def dieSumValues(d: Dict[int, int], n_d: int) -> Dict[int, int]:
        #print(d, n_d)
        res = {0: 1}
        curr_bin = d
        n_d2 = n_d
        while n_d2:
            if n_d2 & 1:
                prev = res
                res = {}
                for num1, f1 in prev.items():
                    for num2, f2 in curr_bin.items():
                        sm = num1 + num2
                        res[sm] = res.get(sm, 0) + f1 * f2
            prev_bin = curr_bin
            curr_bin = {}
            vals = sorted(prev_bin)
            for i1, num1 in enumerate(vals):
                f1 = prev_bin[num1]
                sm = num1 * 2
                curr_bin[sm] = curr_bin.get(sm, 0) + f1 ** 2
                for i2 in range(i1):
                    num2 = vals[i2]
                    f2 = prev_bin[num2]
                    sm = num1 + num2
                    curr_bin[sm] = curr_bin.get(sm, 0) + f1 * f2 * 2
            n_d2 >>= 1
        return res

    sms1 = dieSumValues(d1, n_die1)
    sms2 = dieSumValues(d2, n_die2)
    #print(sms1)
    #print(sms2)
    sm_vals1 = sorted(sms1.keys())
    sm_vals2 = sorted(sms2.keys())
    n_v2 = len(sm_vals2)
    i2 = 0
    res = 0
    cnt2 = 0
    for num1 in sm_vals1:
        f1 = sms1[num1]
        for i2 in range(i2, n_v2):
            if sm_vals2[i2] >= num1: break
            cnt2 += sms2[sm_vals2[i2]]
        res += cnt2 * f1
    return CustomFraction(res, n1 ** n_die1 * n2 ** n_die2)

def probabilityDieOneSumWinsFloat(
    die1_face_values: List[int]=[1, 2, 3, 4],
    n_die1: int=9,
    die2_face_values: List[int]=[1, 2, 3, 4, 5, 6],
    n_die2: int=6
) -> float:
    """
    Solution to Project Euler #205
    
    For two players, player 1 with n_die1 dice each with face values
    die1_face_values and player 2 with n_die2 dice each face values
    die2_face_values, where for each die the probability of each
    face value occurring on a given roll is equal, each player rolls
    all of their dice and adds the total. The player with the highest
    total wins, with equal totals resulting in a draw. This function
    calculates the probability on a given round that player 1 will
    win as a float.

    Args:
        Optional name:
        die1_face_values (list of ints): List of integers giving the
                face values of the dice rolled by player 1, whose
                elements each have an equal probability of being
                rolled. Note that repeated values are allowed.
            Default: [1, 2, 3, 4]
        n_die1 (int): Strictly positive integer giving the number
                of dice that player 1 rolls.
            Default: 9
        die2_face_values (list of ints): List of integers giving the
                face values of the dice rolled by player 2, whose
                elements each have an equal probability of being
                rolled. Note that repeated values are allowed.
            Default: [1, 2, 3, 4, 5, 6]
        n_die2 (int): Strictly positive integer giving the number
                of dice that player 2 rolls.
            Default: 6
    
    Returns:
    Float giving the probability that for a given roll player 1 wins,
    represented as a float.
    """
    res = probabilityDieOneSumWinsFraction(
        die1_face_values,
        n_die1,
        die2_face_values,
        n_die2,
    )
    #print(res)
    return res.numerator / res.denominator

# Problem 206
def concealedSquare(pattern: List[Optional[int]]=[1, None, 2, None, 3, None, 4, None, 5, None, 6, None, 7, None, 8, None, 9, None, 0], base: int=10) -> List[int]:
    """
    Solution to Project Euler #206

    Calculates all the possible strictly positive integers whose squares,
    when expressed in the chosen base, are consistent with pattern, where
    pattern gives the digit values when read from left to right, with
    an integer representing the digit value that must go at the corresponding
    location in the representation of the square and None representing that
    in that position any digit is allowed.

    Args:
        Optional named:
        pattern (list of ints/None): The pattern that the square of any
                returned value must be consistent with, as outlined above.
            Default: [1, None, 2, None, 3, None, 4, None, 5, None, 6, None, 7, None, 8, None, 9, None, 0]
        base (int): Integer strictly greater than 1 giving the base in which
                the squares of integers should be expressed when assessing
                whether they are consistent with pattern.
            Defualt: 10
    
    Returns:
    List of integers (int) giving all the strictly positive integers whose
    squares, when expressed in the chosen base, are consistent with pattern
    (as outlined above) in strictly increasing order of size.
    """
    mn = 0
    mx = 0
    for i, d in enumerate(pattern):
        mn *= base
        mx *= base
        if d is None:
            mx += base - 1
            if not i: mn += 1
        else:
            mn += d
            mx += d
    sqrt_mn = isqrt(mn)
    sqrt_mx = isqrt(mx)
    print(sqrt_mn, sqrt_mx)
    
    def isMatch(num: int) -> bool:
        for i in reversed(range(len(pattern))):
            if pattern[i] is None:
                num //= base
                continue
            num, d = divmod(num, base)
            if d != pattern[i]: return False
        if num: return False
        return True
    
    def isPartialMatch(num: int, n_digs: int) -> bool:
        for i in range(n_digs):
            if pattern[~i] is None:
                num //= base
                continue
            num, d = divmod(num, base)
            if d != pattern[~i]: return False
        return True

    poss_tails = []
    n_tail_digs = len(pattern) >> 2
    for num_sqrt in range(base ** n_tail_digs):
        num = num_sqrt ** 2
        if isPartialMatch(num, n_tail_digs):
            poss_tails.append(num_sqrt)
    #print(f"n possible tails = {len(poss_tails)}")
    #print(poss_tails)
    res = []
    div = base ** n_tail_digs
    for num_sqrt_head in range(sqrt_mn // div, (sqrt_mx // div) + 1):
        for num_sqrt_tail in poss_tails:
            num_sqrt = num_sqrt_head * div + num_sqrt_tail
            num = num_sqrt ** 2
            if isMatch(num):
                print(num_sqrt, num)
                res.append(num_sqrt)
    return res
    """
    for num_sqrt in range(sqrt_mn, sqrt_mx + 1):
        #print(num_sqrt)
        num = num_sqrt ** 2
        if isMatch(num):
            print(num)
            res.append(num)
    return res
    """

# Problem 207
def findSmallestPartitionBelowGivenProportion(
    proportion: CustomFraction=CustomFraction(1, 12345),
) -> int:
    """
    Solution to Project Euler #207

    Consider the equation:
        4 ** t = 2 ** t + k
    where k is a strictly positive integer and t is a real
    number. This function calculates the smallest strictly
    positive integer m for which the proportion of solutions
    to this equation with k <= m for which t is an integer
    is strictly less than the number proportion.

    Args:
        Optional named:
        proportion (CustomFraction object): Rational number
                represented as a fraction giving the exclusive
                upper bound for the proportion of solutions
                to the equation with k no greater than the
                returned value for which t is an integer.
            Default: CustomFraction(1, 12345) (1 / 12345)
    
    Returns:
    Integer (int) giving the smallest strictly positive integer
    m for which the proportion of solutions to the equation
    above with k <= m for which t is an integer is strictly
    less than the number proportion.

    Outline of rationale:
    TODO
    """
    # Review- try to find a better name for the function
    def findExponent(proportion: CustomFraction) -> int:
        comp = lambda m: m * proportion.denominator < 2 * (2 ** m - 1) * proportion.numerator
        if comp(0): return 0
        n = 1
        while True:
            if comp(n):
                break
            n <<= 1
        lft, rgt = n >> 1, n
        #print(lft, rgt)
        while lft < rgt:
            mid = lft + ((rgt - lft) >> 1)
            if comp(mid): rgt = mid
            else: lft = mid + 1
        #print(lft)
        return lft
    
    n = findExponent(proportion)
    l = ((n * proportion.denominator) // proportion.numerator) + 1
    #print(n, l)
    return ((2 * l + 1) ** 2 - 1) >> 2

# Problem 208
def robotWalks(reciprocal: int=5, n_steps: int=70) -> int:
    """
    Solution to Project Euler #208

    Calculates the number of closed paths from a given point
    and initial direction in the 2D plane consisting of circular
    arcs of 1 / reciprocal of a unit radius circle containing
    exactly n_steps such arcs where there are no sharp corners
    between arcs, where reciprocal is prime.

    Args:
        Optional named:
        reciprocal (int): Strictly positive integer giving the
                size of the arc in terms of the
                number of arcs that collectively would make up
                a unit circle.
                This must be prime.
            Default: 5
        n_steps (int): Strictly positive integer giving the number
                of circular arcs that comprise the closed paths
                counted.
            Default: 70
    
    Returns:
    Integer (int) giving the number of closed paths satisfying the
    constraints outlined above.
    
    Outline of rationale:
    TODO
    """
    # Review- attempt more efficient solution with either binary lifting
    # or double ended approach
    # Review- does reciprocal need to be prime for this to work?
    if n_steps % reciprocal: return 0
    curr = {}
    start = tuple([1] + [0] * (reciprocal - 2))
    curr[(0, start)] = 1
    target = n_steps // reciprocal
    for step in range(n_steps - 1):
        prev = curr
        curr = {}
        for (i, cnts), f in prev.items():
            for j in ((i + 1) % reciprocal, (i - 1) % reciprocal):
                if j == reciprocal - 1:
                    cnt = step - sum(cnts)
                    if cnt > target: continue
                    k = (j, cnts)
                else:
                    cnts2 = list(cnts)
                    cnts2[j] += 1
                    if cnts2[j] > target: continue
                    k = (j, tuple(cnts2))
                curr[k] = curr.get(k, 0) + f
        #print(step, curr)
    target_cnts = tuple([target] * (reciprocal - 1))
    #print(curr)
    return curr.get((1, target_cnts), 0) + curr.get(((reciprocal - 1), target_cnts), 0)

# Problem 209
def countZeroMappings(n_inputs: int=6) -> int:
    """
    Solution to Project Euler #209

    Calculates the number of binary truth tables T on a list of length
    n_inputs of boolean variables x for which:
        T(x[0], x[1], ..., x[n_inputs - 1]) & T(x[1], x[2], ..., x[n_inputs - 1], x[0] ^ (x[1] & x[2]))
    evaluates as false for all x, where & and ^ represent logical and
    and xor respectively.

    Args:
        Optional named:
        n_inputs (int): The length of the boolean variable lists used
                by the binary truth tables in question.
            Default: 6
    
    Returns:
    Integer (int) giving the number of binary truth tables satisfying
    the conditions outlined above.

    Outline of rationale:
    TODO
    """
    def bitmaskFunction(bm: int) -> int:
        res = (bm & ((1 << (n_inputs - 1)) - 1)) << 1
        bm2 = bm >> (n_inputs - 3)
        res |= ((bm2 & 4) >> 2) ^ (((bm2 & 2) >> 1) & (bm2 & 1))
        return res
    
    bm_mapping = [bitmaskFunction(bm) for bm in range(1 << n_inputs)]
    #print(bm_mapping)
    #print(len(bm_mapping))

    seen = set()
    cycle_lens = {}
    for bm in range(1 << n_inputs):
        if bm in seen: continue
        l = 0
        bm2 = bm
        while bm2 not in seen:
            seen.add(bm2)
            l += 1
            bm2 = bm_mapping[bm2]
        cycle_lens[l] = cycle_lens.get(l, 0) + 1
    mx_cycle_len = max(cycle_lens.keys())
    n_opts = [0, 1, 3]
    for _ in range(3, mx_cycle_len + 1):
        n_opts.append(n_opts[-1] + n_opts[-2])
    res = 1
    for l, f in cycle_lens.items():
        res *= n_opts[l] ** f
    return res

# Problem 210
def countObtuseTriangles(
    r: Union[int, float]=10 ** 9,
    div: Union[int, float]=4,
) -> int:
    """
    Solution to Project Euler #210

    Calculates the number of distinct points in the Cartesian
    plane with integer coordinates and Manhattan distance of
    at most r from the origin for which the triangle whose
    vertices are that point, the origin and the point with
    Cartesian coordinates (r / div, r / div) is an obtuse
    triangle.

    An obtuse triangle is a triangle where one of the angles
    is strictly greater than pi / 2 (90 degrees).

    In the Cartesian plane, the Manhattan distance between two
    points is the sum of the absolute difference of their
    x-coordinates and their y-coordinates.

    Args:
        Optional named:
        r (real numeric): A strictly positive real number
                specifying the value of r, the inclusive
                upper bound on the Manhattan distance from
                the origin of the points considered for the
                count.
            Default: 10 ** 9
        div (real numeric): Strictly positive real number
                specifying the value of div, which for a
                given value of r determines the position of
                the final triangle vertex.
            Default: 4
    
    Returns:
    Integer (int) giving the number of distinct points in the
    Cartesian plane that satisfy the described constraints.

    Outline of rationale:
    TODO
    """
    r2 = math.floor(r)
    d = math.floor(2 * r / div)
    tot = ((r2 * (r2 + 1)) << 1) + 1
    diag1_cnt = ((r2 >> 1) << 1) + 1
    diag2_cnt = (((r2 + 1) >> 1) << 1)
    strip_n_diag1 = (d >> 1) + 1
    strip_n_diag2 = (d + 1) >> 1
    strip_cnt_wo_diag = (diag1_cnt - 1) * strip_n_diag1 + diag2_cnt * strip_n_diag2
    print(f"tot = {tot}, strip_cnt_wo_diag = {strip_cnt_wo_diag}, diag1_cnt = {diag1_cnt}")
    res = tot - strip_cnt_wo_diag - diag1_cnt
    
    
    # small triangles
    d2 = r / div
    c = d2 / 2
    small_cnt = 0
    x_rng = [math.floor((1 - math.sqrt(2)) * c) + 1, math.ceil((1 + math.sqrt(2)) * c) - 1]
    print(f"x range = {x_rng}")
    for x in range(x_rng[0], x_rng[1] + 1):
        if not x % 10 ** 6: print(f"x = {x}, x range = {x_rng}")
        discr = c ** 2 - x ** 2 + 2 * c * x
        discr_sqrt = math.sqrt(discr)
        y_mn = math.floor(c - discr_sqrt) + 1
        y_mx = math.ceil(c + discr_sqrt) - 1
        small_cnt += y_mx - y_mn + 1
    small_cnt -= math.ceil(d2) - 1
    print(f"small count = {small_cnt}")
    
    return res + small_cnt

# Problem 211
def divisorSquareSumIsSquareTotal(n_max: int=64 * 10 ** 6 - 1) -> int:
    """
    Solution to Project Euler #211

    Calculates the number of strictly positive integers no greater
    than n_max for which the sum of the squares of its divisors
    is a perfect square.

    Args:
        Optional named:
        n_max (int): Integer giving the inclusive upper bound for
                the numbers considered for inclusion in the count.
            Default: 64 * 10 ** 6 - 1 (63,999,999)
    
    Returns:
    Integer (int) giving the number of strictly positive integers
    no greatr than n_max for which the sum of the squares of its
    divisors is a perfect square.

    Brief outline of rationale:
    A sieve approach is used, storing the sum of squares of
    divisors for each integer up to n_max in an array. This is
    populated by iterating over the strictly positive integers,
    adding the square of that integer to the indices of the
    array that are multiples of the integer. By iterating in
    increasing order (given that divisors are no greater than
    the number they divide), this allows for the assessment of
    whether the sum of squares of the divisors is itself a
    square alongside this process, adding the number to the
    total if it is indeed a square.
    """
    # Review- try to speed up
    def isSquare(num: int) -> bool:
        num_sqrt = isqrt(num)
        return num_sqrt ** 2 == num

    arr = [1] * (n_max + 1)
    res = 1
    for num in range(2, n_max + 1):
        #print(num)
        if not num % 500: print(num)
        num_sq = num ** 2
        for num2 in range(num, n_max + 1, num):
            arr[num2] += num_sq
        if isSquare(arr[num]):
            print(num, arr[num])
            res += num
    return res

# Problem 212
def cuboidUnionVolume(cuboids: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]) -> int:

    # Review- try to speed up
    # TODO- investigate why repeats occur for triple or more intersections
    # where the two cuboids with the largest x0-value have the same x0-value.

    def cuboidVolume(cuboid: Tuple[Tuple[int, int, int], Tuple[int, int, int]]) -> int:
        res = 1
        for l in cuboid[1]:
            res *= l
        return res

    def cuboidIntersection(cuboid1: Tuple[Tuple[int, int, int], Tuple[int, int, int]], cuboid2: Tuple[Tuple[int, int, int], Tuple[int, int, int]]) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        res = [[], []]
        for i in range(len(cuboid1[0])):
            x1 = max(cuboid1[0][i], cuboid2[0][i])
            x2 = min(cuboid1[0][i] + cuboid1[1][i], cuboid2[0][i] + cuboid2[1][i])
            if x1 >= x2: return None
            res[0].append(x1)
            res[1].append(x2 - x1)
        return tuple(tuple(x) for x in res)

    cuboids2 = SortedList([(x, {i}) for i, x in enumerate(cuboids)])

    res = 0
    i1 = 0
    seen = set()
    cnt = 0
    while cuboids2:
        cuboid, inds = cuboids2.pop(0)
        cnt += 1
        if not cnt % 1000:
            print(f"seen {cnt}, x0 = {cuboid[0][0]}, list length currently {len(cuboids2)}")
        inds_tup = tuple(sorted(inds))
        if inds_tup in seen:
            print(f"repeated index combination: {inds_tup}")
            for idx in inds_tup:
                print(cuboids[idx])
            continue
        seen.add(inds_tup)
        vol = cuboidVolume(cuboid)
        # Inclusion-exclusion
        res += vol if (len(inds) & 1) else -vol
        add_lst = []
        mx_x = cuboid[0][0] + cuboid[1][0]
        if len(inds) > 1: continue
        for i2 in range(len(cuboids2)):
            cuboid2, inds2 = cuboids2[i2]
            if cuboid2[0][0] >= mx_x: break
            elif not inds.isdisjoint(inds2): continue
            intersect = cuboidIntersection(cuboid, cuboid2)
            if intersect is None: continue
            add_lst.append((intersect, inds.union(inds2)))
        for tup in add_lst: cuboids2.add(tup)
        i1 += 1
    print(f"total seen = {cnt}, unique seen = {len(seen)}")
    return res

def laggedFibonacciNDimensionalHyperCuboidGenerator(
    hypercuboid_smallest_coord_ranges: Tuple[Tuple[int, int]]=((0, 9999), (0, 9999), (0, 9999)),
    hypercuboid_dim_ranges: Tuple[Tuple[int, int]]=((1, 399), (1, 399), (1, 399)),
    n_hypercuboids: Optional[int]=None,
    l_fib_modulus: int=10 ** 6,
    l_fib_poly_coeffs: Tuple[int]=(100003, -200003, 0, 300007),
    l_fib_lags: Tuple[int]=(24, 55),
) -> Generator[Tuple[Tuple[int, int, int], Tuple[int, int, int]], None, None]:
    """
    Generator yielding TODO

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

    Note that if n_hypercuboids is not specified, the generator never terminates and
    thus any iterator over this generator must include provision to terminate
    (e.g. a break or return statement), otherwise it would result in an infinite
    loop.

    Args:
        Optional named:
        TODO
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
    TODO
    If n_hypercuboids is specified as a non-negative integer then (unless
    externally terminated first) exactly n_edges such values are yielded,
    otherwise the generator never of itself terminates.
    """
    n_dim = len(hypercuboid_smallest_coord_ranges)
    it = generalisedLaggedFibonacciGenerator(poly_coeffs=l_fib_poly_coeffs, lags=l_fib_lags, min_val=0, max_val=l_fib_modulus - 1)
    it2 = range(n_hypercuboids) if isinstance(n_hypercuboids, int) and n_hypercuboids >= 0 else itertools.count(0)
    for _ in it2:
        cuboid = [[], []]
        for rng in hypercuboid_smallest_coord_ranges:
            cuboid[0].append(rng[0] + (next(it) % (rng[1] - rng[0] + 1)))
        for rng in hypercuboid_dim_ranges:
            cuboid[1].append(rng[0] + (next(it) % (rng[1] - rng[0] + 1)))
        yield tuple(tuple(x) for x in cuboid)
    return 

def laggedFibonacciCuboidUnionVolume(
    n_cuboids: int=50000,
    cuboid_smallest_coord_ranges: Tuple[Tuple[int, int]]=((0, 9999), (0, 9999), (0, 9999)),
    cuboid_dim_ranges: Tuple[Tuple[int, int]]=((1, 399), (1, 399), (1, 399)),
    l_fib_modulus: int=10 ** 6,
    l_fib_poly_coeffs: Tuple[int]=(100003, -200003, 0, 300007),
    l_fib_lags: Tuple[int]=(24, 55),
) -> int:
    """
    Solution to Project Euler #212
    """
    cuboids = [c for c in laggedFibonacciNDimensionalHyperCuboidGenerator(
        hypercuboid_smallest_coord_ranges=cuboid_smallest_coord_ranges,
        hypercuboid_dim_ranges=cuboid_dim_ranges,
        n_hypercuboids=n_cuboids,
        l_fib_modulus=l_fib_modulus,
        l_fib_poly_coeffs=l_fib_poly_coeffs,
        l_fib_lags=l_fib_lags,
    )]
    res = cuboidUnionVolume(cuboids)
    return res

# Problem 213
def fleaCircusExpectedNumberOfUnoccupiedSquaresFraction(dims: Tuple[int, int], n_steps: int) -> CustomFraction:

    def transferFunction(pos: Tuple[int, int]) -> Tuple[Set[Tuple[int, int]], CustomFraction]:
        denom = 4
        denom -= (pos[0] == 0) + (pos[0] == (dims[0] - 1)) + (pos[1] == 0) + (pos[1] == (dims[1] - 1))
        p = CustomFraction(1, denom)
        res = set()
        if pos[0] > 0: res.add((pos[0] - 1, pos[1]))
        if pos[0] < dims[0] - 1: res.add((pos[0] + 1, pos[1]))
        if pos[1] > 0: res.add((pos[0], pos[1] - 1))
        if pos[1] < dims[1] - 1: res.add((pos[0], pos[1] + 1))
        return (res, p)

    is_square = (dims[0] == dims[1])

    p_arr = [[CustomFraction(1, 1)] * dims[1] for _ in range(dims[0])]
    sym_funcs = []
    sym_funcs.append(lambda pos: pos)
    sym_funcs.append(lambda pos: (dims[0] - 1 - pos[0], pos[1]))
    sym_funcs.append(lambda pos: (pos[0], dims[1] - 1 - pos[1]))
    sym_funcs.append(lambda pos: (dims[0] - 1 - pos[0], dims[1] - 1 - pos[1]))
    if is_square:
        sym_funcs.append(lambda pos: (pos[1], pos[0]))
        sym_funcs.append(lambda pos: (dims[0] - 1 - pos[1], pos[0]))
        sym_funcs.append(lambda pos: (pos[1], dims[1] - 1 - pos[0]))
        sym_funcs.append(lambda pos: (dims[0] - 1 - pos[1], dims[1] - 1 - pos[0]))

    for i1 in range((dims[0] + 1) >> 1):
        for i2 in range(i1 if is_square else 0, (dims[1] + 1) >> 1):
            arr = [[0] * dims[1] for _ in range(dims[0])]
            arr[i1][i2] = 1
            for m in range(n_steps):
                arr0 = arr
                arr = [[0] * dims[1] for _ in range(dims[0])]
                for j1 in range(dims[0]):
                    odd = (i1 + i2 + j1 + m) & 1
                    for j2 in range(odd, dims[1], 2):
                        if arr0[j1][j2] == 0: continue
                        t_set, p = transferFunction((j1, j2))
                        for pos in t_set:
                            arr[pos[0]][pos[1]] += arr0[j1][j2] * p
            seen = set()
            sym_funcs2 = []
            for sym_func in sym_funcs:
                pos = sym_func((i1, i2))
                if pos in seen: continue
                seen.add(pos)
                sym_funcs2.append(sym_func)
            
            for j1 in range(dims[0]):
                odd = (i1 + i2 + j1 + n_steps) & 1
                for j2 in range(odd, dims[1], 2):
                    if arr[j1][j2] == 0: continue
                    for sym_func in sym_funcs2:
                        pos2 = sym_func((j1, j2))
                        p_arr[pos2[0]][pos2[1]] *= 1 - arr[j1][j2]
    res = sum(sum(row) for row in p_arr)
    return res

def fleaCircusExpectedNumberOfUnoccupiedSquaresFloatDirect(dims: Tuple[int, int], n_steps: int) -> float:
    """
    Solution to Project Euler #213
    """

    def transferFunction(pos: Tuple[int, int]) -> Tuple[Set[Tuple[int, int]], float]:
        denom = 4
        denom -= (pos[0] == 0) + (pos[0] == (dims[0] - 1)) + (pos[1] == 0) + (pos[1] == (dims[1] - 1))
        p = 1 / denom
        res = set()
        if pos[0] > 0: res.add((pos[0] - 1, pos[1]))
        if pos[0] < dims[0] - 1: res.add((pos[0] + 1, pos[1]))
        if pos[1] > 0: res.add((pos[0], pos[1] - 1))
        if pos[1] < dims[1] - 1: res.add((pos[0], pos[1] + 1))
        return (res, p)

    is_square = (dims[0] == dims[1])

    p_arr = [[1] * dims[1] for _ in range(dims[0])]
    sym_funcs = []
    sym_funcs.append(lambda pos: pos)
    sym_funcs.append(lambda pos: (dims[0] - 1 - pos[0], pos[1]))
    sym_funcs.append(lambda pos: (pos[0], dims[1] - 1 - pos[1]))
    sym_funcs.append(lambda pos: (dims[0] - 1 - pos[0], dims[1] - 1 - pos[1]))
    if is_square:
        sym_funcs.append(lambda pos: (pos[1], pos[0]))
        sym_funcs.append(lambda pos: (dims[0] - 1 - pos[1], pos[0]))
        sym_funcs.append(lambda pos: (pos[1], dims[1] - 1 - pos[0]))
        sym_funcs.append(lambda pos: (dims[0] - 1 - pos[1], dims[1] - 1 - pos[0]))

    for i1 in range((dims[0] + 1) >> 1):
        for i2 in range(i1 if is_square else 0, (dims[1] + 1) >> 1):
            arr = [[0] * dims[1] for _ in range(dims[0])]
            arr[i1][i2] = 1
            for m in range(n_steps):
                arr0 = arr
                arr = [[0] * dims[1] for _ in range(dims[0])]
                for j1 in range(dims[0]):
                    odd = (i1 + i2 + j1 + m) & 1
                    for j2 in range(odd, dims[1], 2):
                        if arr0[j1][j2] == 0: continue
                        t_set, p = transferFunction((j1, j2))
                        for pos in t_set:
                            arr[pos[0]][pos[1]] += arr0[j1][j2] * p
            seen = set()
            sym_funcs2 = []
            for sym_func in sym_funcs:
                pos = sym_func((i1, i2))
                if pos in seen: continue
                seen.add(pos)
                sym_funcs2.append(sym_func)
            
            for j1 in range(dims[0]):
                odd = (i1 + i2 + j1 + n_steps) & 1
                for j2 in range(odd, dims[1], 2):
                    if arr[j1][j2] == 0: continue
                    for sym_func in sym_funcs2:
                        pos2 = sym_func((j1, j2))
                        p_arr[pos2[0]][pos2[1]] *= 1 - arr[j1][j2]
    res = sum(sum(row) for row in p_arr)
    return res

def fleaCircusExpectedNumberOfUnoccupiedSquaresFloatFromFraction(dims: Tuple[int, int]=(30, 30), n_steps: int=50) -> float:
    """
    Alternative (more precise) solution to Project Euler #213
    """
    res = fleaCircusExpectedNumberOfUnoccupiedSquaresFraction(dims, n_steps)
    print(res)
    return res.numerator / res.denominator

# Problem 214
def primesOfTotientChainLengthSum(p_max: int=4 * 10 ** 7 - 1, chain_len: int=25) -> int:
    """
    Solution to Project Euler #214
    """
    # It appear that for chain lengths len no less than 2, the last integer
    # with that chain length is 2 * 3 ** (len - 2). Can this be proved?
    ps = PrimeSPFsieve(p_max)
    print("calculated prime sieve")
    totient_vals = [0, 1, 2]
    totient_lens = [0, 1, 2]
    last_chain_lens = [0, 1, 2]
    last_prime_chain_lens = [-1, -1, 2]
    
    res = 0
    for num in range(3, p_max + 1):
        p, exp, num2 = ps.sieve[num]
        #print(num, (p, exp, num2), totient_vals)
        if p == num:
            # num is prime
            totient_vals.append(num - 1)
            totient_lens.append(totient_lens[totient_vals[-1]] + 1)
            if totient_lens[-1] == chain_len:
                #print(num)
                res += num
            last_prime_chain_lens += [-1] * max(0, totient_lens[-1] + 1 - len(last_prime_chain_lens))
            last_prime_chain_lens[totient_lens[-1]] = num
        else:
            totient_vals.append(totient_vals[num2] * (p - 1) * p ** (exp - 1))
            totient_lens.append(totient_lens[totient_vals[-1]] + 1)
        last_chain_lens += [-1] * max(0, totient_lens[-1] + 1 - len(last_chain_lens))
        last_chain_lens[totient_lens[-1]] = num
    #print(totient_vals)
    #print(totient_lens)
    print(last_chain_lens)
    return res

# Problem 215
def crackFreeWalls(n_rows: int=32, n_cols: int=10) -> int:
    """
    Solution to Project Euler #215
    """
    if not n_rows or not n_cols: return 1
    row_opts = []
    row_opts_dict = {}
    transfer = []

    step_opts = {2, 3}
    mn_step_opt = min(step_opts)

    #memo = {}
    def recur(mn_remain: int=n_cols, diff: int=0) -> Generator[Tuple[Tuple[int], Tuple[int]], None, None]:
        if not mn_remain:
            if not diff:
                yield ((), ())
                return
            elif diff in step_opts:
                ans = ((), (diff,))
                yield ans
                return
            return
        elif mn_remain < mn_step_opt: return
        #args = (mn_remain, diff)
        #if args in memo.keys():
        #    return memo[args]
        res = []
        for step in step_opts:
            diff2 = diff - step
            if not diff2: continue
            if diff2 > 0:
                for ans in recur(mn_remain=mn_remain, diff=diff2):
                    ans2 = (ans[0], tuple([step] + list(ans[1])))
                    #print(1, mn_remain, diff, ans2)
                    yield ans2
            else:
                mn_remain2 = mn_remain + diff2
                for ans in recur(mn_remain=mn_remain2, diff=-diff2):
                    ans2 = (ans[1], tuple([step] + list(ans[0])))
                    #print(2, mn_remain, diff, ans2)
                    yield ans2

        #res = tuple(res)
        #memo[args] = res
        #return res
        return

    transfer = []
    for pair in recur(mn_remain=n_cols, diff=0):
        #print(pair)
        for tup in pair:
            if tup in row_opts_dict.keys(): continue
            row_opts_dict[tup] = len(row_opts)
            row_opts.append(tup)
            transfer.append(set())
        idx1, idx2 = row_opts_dict[pair[0]], row_opts_dict[pair[1]]
        transfer[idx1].add(idx2)
    
    #print(row_opts)
    #print(transfer)
    n_opts = len(row_opts)
    #print(f"n_opts = {n_opts}")
    curr = [1] * n_opts
    for _ in range(n_rows - 1):
        prev = curr
        curr =  [0] * n_opts
        for i1 in range(n_opts):
            for i2 in transfer[i1]:
                curr[i2] += prev[i1]
    return sum(curr)

# Problem 216
def countPrimesOneLessThanTwiceASquare(n_max: int=5 * 10 ** 7) -> int:
    """
    Solution to Project Euler #216
    """
    # Review- look into the more efficient methods as outlined in
    # the PDF document accompanying the problem.
    ps = SimplePrimeSieve()
    def primeCheck(num: int) -> bool:
        res = ps.millerRabinPrimalityTestWithKnownBounds(num, max_n_additional_trials_if_above_max=10)
        return res[0]

    res = 0
    for num in range(2, n_max + 1):
        if not num % 10000: print(num)
        res += primeCheck(2 * num ** 2 - 1)
    return res

# Problem 217
def balancedNumberCount(max_n_dig: int=47, base: int=10, md: Optional[int]=3 ** 15) -> int:
    """
    Solution to Project Euler #217
    """
    # Review- see if it can be made more efficient using cumulative totals
    if max_n_dig < 1: return 0
    dig_sum = (base * (base - 1)) >> 1
    #if max_n_dig == 1:
    #    res = dig_sum
    #    return res if md is None else res % md
    #elif max_n_dig == 2:
    #    res = dig_sum * (base + 1)
    #    return res if md is None else res % md
    #elif max_n_dig == 3:
    #    res = dig_sum * (base ** 2 + 1) + base * 

    def calculateRunningTotalWithoutMod(i: int, res_init: int=0) -> int:
        res = res_init
        if (i << 1) + 1 > max_n_dig:
            for j in range(1, len(row_lft)):
                res += row_lft[j][1] * (base ** i) * row_rgt[j][0] + row_rgt[j][1] * row_lft[j][0]
            return res
        for j in range(1, len(row_lft)):
            res += (row_lft[j][1] * (base ** 2 + 1) + dig_sum * row_lft[j][0])* (base ** i) * row_rgt[j][0] + (base + 1) * row_rgt[j][1] * row_lft[j][0]
        return res
    
    def calculateRunningTotalWithMod(i: int, res_init: int=0) -> int:
        res = res_init
        if (i << 1) + 1 > max_n_dig:
            for j in range(1, len(row_lft)):
                res = (res + row_lft[j][1] * pow(base, i, md) * row_rgt[j][0] + row_rgt[j][1] * row_lft[j][0]) % md
            return res
        for j in range(1, len(row_lft)):
            res = (res + (row_lft[j][1] * (base ** 2 + 1) + dig_sum * row_lft[j][0])* pow(base, i, md) * row_rgt[j][0] + (base + 1) * row_rgt[j][1] * row_lft[j][0]) % md
        return res
    
    calculateRunningTotal = calculateRunningTotalWithoutMod if md is None else calculateRunningTotalWithMod
    
    m = max_n_dig >> 1
    #print(f"m = {m}")
    row_rgt = [[1, x] for x in range(base)]
    row_lft = [[1, x] for x in range(base)]
    row_lft[0] = [0, 0]
    
    res = dig_sum
    if max_n_dig == 1:
        return res
    res = calculateRunningTotal(1, res_init=res)
    #print(1)
    #print(row_lft)
    #print(row_rgt)
    #print(res)
    for i in range(2, m + 1):
        prev_rgt = row_rgt
        row_rgt = [[0, 0] for x in range(len(prev_rgt) + base - 1)]
        prev_lft = row_lft
        row_lft = [[0, 0] for x in range(len(prev_lft) + base - 1)]
        for row, prev, mn in [(row_lft, prev_lft, 1), (row_rgt, prev_rgt, 0)]:
            for j in range(0, len(prev)):
                for d in range(base):
                    j2 = j + d
                    row[j2][0] += prev[j][0]
                    row[j2][1] += (prev[j][1] * base) + d * prev[j][0]
        res = calculateRunningTotal(i, res_init=res)
        #print(f"i = {i}")
        #print(row_lft)
        #print(row_rgt)
        #print(res)
        
        """
        if md is None:
            if (i << 1) + 1 > max_n_dig:
                for j in range(1, len(row_lft)):
                    res += row_lft[j][1] * (base ** m) * row_rgt[j][0] + row_rgt[j][1] * row_lft[j][0]
                continue
            for j in range(1, len(row_lft)):
                res += (row_lft[j][1] * (base ** 2 + 1) + dig_sum * row_lft[j][0])* (base ** m) * row_rgt[j][0] + 2 * row_rgt[j][1] * row_lft[j][0]
            continue
        if (i << 1) + 1 > max_n_dig:
            for j in range(1, len(row_lft)):
                res = (res + row_lft[j][1] * pow(base, m, md) * row_rgt[j][0] + row_rgt[j][1] * row_lft[j][0]) % md
            continue
        for j in range(1, len(row_lft)):
            res = (res + (row_lft[j][1] * (base ** 2 + 1) + dig_sum * row_lft[j][0])* pow(base, m, md) * row_rgt[j][0] + 2 * row_rgt[j][1] * row_lft[j][0]) % md
        """
            
    return res


# Problem 218
def perfectRightAngledTriangleGenerator(max_hypotenuse: Optional[int]=None) -> Generator[Tuple[Tuple[int, int, int], bool], None, None]:

    #m = 1
    heap = []
    if max_hypotenuse is None: max_hypotenuse = float("inf")
    perfect_cnt = 0
    for m in itertools.count(1):
        m += 1
        m_odd = m & 1
        n_mn = 1 + m_odd
        m_sq = m ** 2

        #m2_mn
        min_hyp = (m_sq + n_mn ** 2) ** 2
        while heap and heap[0][0] < min_hyp:
            ans = heapq.heappop(heap)
            yield tuple(ans[::-1])
            perfect_cnt += 1
        if min_hyp > max_hypotenuse: break
        n_mx = min(m - 1, isqrt(isqrt(max_hypotenuse) - m_sq)) if max_hypotenuse != float("inf") else m - 1
        # Note that since m and n are coprime and not both can be odd,
        # m and n must have different parity (as if they were both
        # even then they would not be coprime)
        n = 1 + m_odd
        for n in range(n_mn, n_mx + 1, 2):
            if gcd(m, n) != 1:
                n += 1
                continue
            m2, n2 = m_sq - n ** 2, 2 * m * n
            if m2 ** 2 + n2 ** 2 > max_hypotenuse: break
            # Note that since m and n are of different parity and coprime, m2 and n2
            # are also guaranteed to be of different parit and coprime
            #if m2 & 1 == n2 & 1: continue
            if m2 < n2: m2, n2 = n2, m2
            m2_sq, n2_sq = m2 * m2, n2 * n2
            a, b, c = m2_sq - n2_sq, 2 * m2 * n2, m2_sq + n2_sq
            if b < a: a, b = b, a
            heapq.heappush(heap, (c, b, a))
            n += 1
        #for n in range(1 + m_odd, max_n + 1, 2):
        #    if gcd(m, n) != 1: continue
        #    a, b, c = m_sq - n ** 2, 2 * m * n, m_sq + n ** 2
        #    if b < a: a, b = b, a
        #    heapq.heappush(heap, ((c, b, a), (c, b, a), True))
    print(f"perfect count = {perfect_cnt}")
    return

    """
    def isSquare(num: int) -> bool:
        rt = isqrt(num)
        if rt * rt == num: return True
    tot_cnt = 0
    perfect_cnt = 0
    for tri in pythagoreanTripleGeneratorByHypotenuse(primitive_only=True, max_hypotenuse=max_hypotenuse):
        tot_cnt += 1
        if isSquare(tri[0][2]):
            print(tri)
            perfect_cnt += 1
            yield tri[0]
    print(f"perfect count = {perfect_cnt} of {tot_cnt}")
    return
    """

def nonSuperPerfectPerfectRightAngledTriangleCount(max_hypotenuse: int=10 ** 16) -> int:
    """
    Solution to Project Euler #218
    """
    # Note that it can be proved that there are no perfect right angled triangles
    # that are not also super-perfect, and therefore for any max_hypotenuse the
    # solution is 0
    mults = [6, 28]
    m = 1
    for num in mults:
        m = lcm(m, num)
    m <<= 1

    res = 0
    perfect_cnt = 0
    for tri in perfectRightAngledTriangleGenerator(max_hypotenuse=max_hypotenuse):
        #print(tri)
        res += ((tri[0] * tri[1]) % m)
    return res

# Problem 219
def prefixFreeCodeMinimumTotalSkewCost(n_words: int=10 ** 9, cost1: int=1, cost2: int=4) -> int:
    """
    Solution to Project Euler #219
    """
    # Using Huffman enconding
    if n_words < 1: return 0
    elif n_words == 1: return min(cost1, cost2)

    remain = n_words - 2
    cost_sm = cost1 + cost2
    g = gcd(cost1, cost2)
    curr = {cost1: 1}
    curr[cost2] = curr.get(cost2, 0) + 1
    res = cost_sm
    for c in itertools.count(min(cost1, cost2), step=g):
        f = curr.pop(c, 0)
        if not f: continue
        if remain < f:
            res += (cost_sm + c) * remain
            break
        res += (cost_sm + c) * f
        remain -= f
        for add in (cost1, cost2):
            c2 = c + add
            curr[c2] = curr.get(c2, 0) + f
    return res

# Problem 220
def heighwayDragon(order: int=50, n_steps: int=10 ** 12, init_pos: Tuple[int, int]=(0, 0), init_direct: Tuple[int, int]=(0, 1), initial_str: str="Fa", recursive_strs: Dict[str, str]={"a": "aRbFR", "b": "LFaLb"}) -> Optional[Tuple[int, int]]:
    """
    Solution to Project Euler #220
    """
    if n_steps <= 0: return init_pos
    direct_dict = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
    if init_direct not in direct_dict.keys(): raise ValueError("The initial direction given is not valid")
    state0 = (init_pos, direct_dict[init_direct])

    basic_effects = {"L": (((0, 0), 1), 0), "R": (((0, 0), -1), 0), "F": (((1, 0), 0), 1)}
    recursive_effects = [{l: (((0, 0), 0), 0) for l in recursive_strs.keys()}]

    def applyCharacter(curr_state: Tuple[Tuple[int, int], int], l: str, curr_order: int) -> Tuple[Tuple[int, int], int, int]:
        effect, n_steps = basic_effects[l] if l in basic_effects.keys() else recursive_effects[curr_order][l]
        #print(l, effect)
        new_direct = (curr_state[1] + effect[1]) % 4
        if effect[0] == (0, 0): return ((curr_state[0], new_direct), n_steps)
        if curr_state[1] == 0:
            pos_effect = effect[0]
        elif curr_state[1] == 1:
            pos_effect = (-effect[0][1], effect[0][0])
        elif curr_state[1] == 2:
            pos_effect = (-effect[0][0], -effect[0][1])
        elif curr_state[1] == 3:
            pos_effect = (effect[0][1], -effect[0][0])
        return ((tuple(x + y for x, y in zip(curr_state[0], pos_effect)), new_direct), n_steps)

    for ordr in range(1, order + 1):
        recursive_effects.append({})
        for l, s in recursive_strs.items():
            ordr2 = ordr - 1
            state = ((0, 0), 0)
            tot_steps = 0
            for l2 in s:
                state, add_steps = applyCharacter(state, l2, ordr2)
                tot_steps += add_steps
            recursive_effects[-1][l] = (state, tot_steps)
    #print(recursive_effects)
    remain_steps = n_steps
    ordr = order
    state = state0

    for l in initial_str:
        state2, add_steps = applyCharacter(state, l, ordr)
        if add_steps == remain_steps: return state2[0]
        elif add_steps > remain_steps: break
        state = state2
        remain_steps -= add_steps
    else: return None
    
    while ordr > 0:
        ordr -= 1
        for l2 in recursive_strs[l]:
            state2, add_steps = applyCharacter(state, l2, ordr)
            if add_steps == remain_steps: return state2[0]
            elif add_steps > remain_steps:
                l = l2
                break
            state = state2
            remain_steps -= add_steps
    return None

# Problem 121
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

def calculateFactorsInRange(num: int, fact_min: Optional[int]=None, fact_max: Optional[int]=None) -> Set[int]:
    if fact_min > num: return set()
    pf = calculatePrimeFactorisation(num)
    #print(num, pf)
    if fact_max is None: fact_max = num
    if fact_min is None: fact_min = 1
    curr = {1}
    res = set() if fact_min > 1 else {1}
    for p, f in pf.items():
        prev = set(curr)
        for m in prev:
            m2 = m
            for i in range(f + 1):
                if m2 > fact_max: break
                curr.add(m2)
                if m2 >= fact_min:
                    res.add(m2)
                m2 *= p
    return res


def alexandrianIntegerGenerator() -> Generator[int, None, None]:
    h = []
    cnt = 0
    for m in itertools.count(1):
        #print(f"m = {m}, count = {cnt}")
        m2 = m ** 2 + 1
        mn = 2 * m * m2
        while h and h[0] <= mn:
            cnt += 1
            yield heapq.heappop(h)
        #print(f"heap size = {len(h)}")
        d_set = calculateFactorsInRange(m2, fact_max=m)
        #print(m2, d_set)
        for d in d_set:
            heapq.heappush(h, m * (m + d) * (m + m2 // d))
        #for d in range(1, m + 1):
        #    d2, r = divmod(m2, d)
        #    if r: continue
        #    heapq.heappush(h, m * (m + d) * (m + d2))
    return

def nthAlexandrianInteger(n: int=15 * 10 ** 4) -> int:
    """
    Solution to Project Euler #221
    """
    it = iter(alexandrianIntegerGenerator())
    for _ in range(n):
        num = next(it)
    return num

# Problem 222
def shortestSpherePackingInTube(tube_radius: int=50, radii: List[int]=list(range(30, 51))) -> float:
    """
    Solution to Project Euler #222
    """
    n = len(radii)
    if not n: return 0.

    radii.sort()
    if radii[0] * 2 < tube_radius or (n > 1 and radii[1] * 2 <= tube_radius) or radii[-1] > tube_radius:
        raise ValueError("Radii must be no less than half of tube_diameter and "
            "no more than tube_diameter")
    d = tube_radius * 2
    
    if n == 1: return 2 * radii[0]

    def packingAddDistance(r1: int, r2: int) -> float:
        return math.sqrt(d * (2 * (r1 + r2) - d))

    res = packingAddDistance(radii[0], radii[1])

    for i in range(2, n):
        res += packingAddDistance(radii[i], radii[i - 2])
    return res + radii[-1] + radii[-2]

# Problem 223
def barelyAcuteIntegerSidedTrianglesAscendingPerimeterGenerator(max_perim: Optional[int]=None) -> Generator[Tuple[int, int, int], None, None]:

    if max_perim is None: max_perim = float("inf")
    if max_perim < 3: return

    A = [[1, -2, 2], [2, -1, 2], [2, -2, 3]]
    B = [[1, 2, 2], [2, 1, 2], [2, 2, 3]]
    C = [[-1, 2, 2], [-2, 1, 2], [-2, 2, 3]]

    def matrixMultiplyVector(M: List[List[int]], v: List[int]) -> List[int]:
        return [sum(x * y for x, y in zip(row, v)) for row in M]

    #seen = {(1, 0, 0), (1, 1, 1)}
    h = [[1, (1, 0, 0)], [3, (1, 1, 1)]]
    while h:
        triple = heapq.heappop(h)[1]
        #seen.remove(triple)
        if min(triple) > 0: yield triple
        seen = set()
        for M in (A, B, C):
            triple2 = tuple(sorted(matrixMultiplyVector(M, triple)))
            if triple2[0] <= 0:
                continue
            elif triple2 in seen:
                print(f"repeat: {triple2}")
                continue
            seen.add(triple2)
            perim = sum(triple2)
            if perim > max_perim: continue
            heapq.heappush(h, [perim, triple2])
            """
            triple2 = tuple(sorted(matrixMultiplyVector(M, triple)))
            triple3 = tuple(sorted(matrixMultiplyVector(M, [triple[1], triple[0], triple[2]])))
            for t in (triple2, triple3):
                if t[0] <= 0 or t in seen:
                    continue
                seen.add(t)
                heapq.heappush(h, [sum(t), t])
            """
    return

def countBarelyAcuteIntegerSidedTrianglesUpToMaxPerimeter(max_perimeter: int=25 * 10 ** 6) -> int:
    """
    Solution to Project Euler #223
    """
    # Review- give proof that this approach works

    # Review- look into method using factorisation (a - 1)(a + 1) = (c - b)(c + 1)
    """
    #it = barelyAcuteIntegerSidedTrianglesAscendingPerimeterGenerator(max_perim=max_perimeter)
    print_intvl = 10 ** 4
    nxt_perim = print_intvl
    for i, triple in enumerate(barelyAcuteIntegerSidedTrianglesAscendingPerimeterGenerator(max_perim=max_perimeter), start=1):
        perim = sum(triple)
        if perim > nxt_perim:
            print(triple, perim, i)
            nxt_perim += print_intvl
    return i
    """
    if max_perimeter < 3: return

    A = [[1, -2, 2], [2, -1, 2], [2, -2, 3]]
    B = [[1, 2, 2], [2, 1, 2], [2, 2, 3]]
    C = [[-1, 2, 2], [-2, 1, 2], [-2, 2, 3]]

    def matrixMultiplyVector(M: List[List[int]], v: List[int]) -> List[int]:
        return [sum(x * y for x, y in zip(row, v)) for row in M]

    #seen = {(1, 0, 0), (1, 1, 1)}
    #h = [[1, (1, 0, 0)], [3, (1, 1, 1)]]
    stk = [(1, 0, 0), (1, 1, 1)]
    res = 1
    while stk:
        triple = stk.pop()
        seen = set()
        for M in (A, B, C):
            triple2 = matrixMultiplyVector(M, triple)
            triple3 = tuple(sorted(triple2))
            if triple3[0] <= 0: continue
            elif triple3 in seen:
                print(f"repeat: {triple2}")
                continue
            seen.add(triple3)
            perim = sum(triple2)
            if perim > max_perimeter: continue
            stk.append(triple2)
            res += 1
    return res

# Problem 224
def barelyObtuseIntegerSidedTrianglesAscendingPerimeterGenerator(max_perim: Optional[int]=None) -> Generator[Tuple[int, int, int], None, None]:

    
    if max_perim is None: max_perim = float("inf")
    if max_perim < 3: return

    A = [[1, -2, 2], [2, -1, 2], [2, -2, 3]]
    B = [[1, 2, 2], [2, 1, 2], [2, 2, 3]]
    C = [[-1, 2, 2], [-2, 1, 2], [-2, 2, 3]]

    def matrixMultiplyVector(M: List[List[int]], v: List[int]) -> List[int]:
        return [sum(x * y for x, y in zip(row, v)) for row in M]

    #seen = {(1, 0, 0), (1, 1, 1)}
    h = [[1, (0, 0, 1)]]
    while h:
        triple = heapq.heappop(h)[1]
        #seen.remove(triple)
        if min(triple) > 0: yield triple
        seen = set()
        for M in (A, B, C):
            triple2 = tuple(sorted(matrixMultiplyVector(M, triple)))
            if triple2[0] <= 0:
                continue
            elif triple2 in seen:
                print(f"repeat: {triple2}")
                continue
            seen.add(triple2)
            perim = sum(triple2)
            if perim > max_perim: continue
            heapq.heappush(h, [perim, triple2])
            """
            triple2 = tuple(sorted(matrixMultiplyVector(M, triple)))
            triple3 = tuple(sorted(matrixMultiplyVector(M, [triple[1], triple[0], triple[2]])))
            for t in (triple2, triple3):
                if t[0] <= 0 or t in seen:
                    continue
                seen.add(t)
                heapq.heappush(h, [sum(t), t])
            """
    return

def countBarelyObtuseIntegerSidedTrianglesUpToMaxPerimeter(max_perimeter: int=75 * 10 ** 6) -> int:
    """
    Solution to Project Euler #224
    """
    # Review- give proof that this approach works including justification
    # of the initial values

    """
    print_intvl = 10 ** 4
    nxt_perim = print_intvl
    for i, triple in enumerate(barelyObtuseIntegerSidedTrianglesAscendingPerimeterGenerator(max_perim=max_perimeter), start=1):
        perim = sum(triple)
        if perim > nxt_perim:
            print(triple, perim, i)
            nxt_perim += print_intvl
    return i
    """
    if max_perimeter < 3: return

    A = [[1, -2, 2], [2, -1, 2], [2, -2, 3]]
    B = [[1, 2, 2], [2, 1, 2], [2, 2, 3]]
    C = [[-1, 2, 2], [-2, 1, 2], [-2, 2, 3]]

    def matrixMultiplyVector(M: List[List[int]], v: List[int]) -> List[int]:
        return [sum(x * y for x, y in zip(row, v)) for row in M]

    #seen = {(1, 0, 0), (1, 1, 1)}
    #h = [[1, (1, 0, 0)], [3, (1, 1, 1)]]
    stk = [(0, 0, 1)]
    res = 0
    while stk:
        triple = stk.pop()
        seen = set()
        for M in (A, B, C):
            triple2 = matrixMultiplyVector(M, triple)
            triple3 = tuple(sorted(triple2))
            if triple3[0] <= 0: continue
            elif triple3 in seen:
                print(f"repeat: {triple2}")
                continue
            seen.add(triple3)
            perim = sum(triple2)
            if perim > max_perimeter: continue
            stk.append(triple2)
            res += 1
    return res    

# Problem 225
def tribonacciOddNonDivisorGenerator(init_terms: Tuple[int, int, int]=(1, 1, 1)) -> Generator[int, None, None]:

    #Trie = lambda: defaultdict(Trie)
    ref = list(init_terms)
    for num in itertools.count(3, step=2):
        #seen_triples = Trie()
        curr = [x % num for x in init_terms]
        #t = seen_triples
        #for m in curr:
        #    t = t[m]
        #t[True] = True
        while True:
            curr = [curr[1], curr[2], sum(curr) % num]
            if not curr[-1]:
                break
            if curr == ref:
                yield num
                break
            #t = seen_triples
            #for m in curr:
            #    t = t[m]
            #if True in t.keys():
            #    yield num
            #    break
            #t[True] = True
    return

def nthSmallestTribonacciOddNonDivisors(odd_non_divisor_number: int=124, init_terms: Tuple[int, int, int]=(1, 1, 1)) -> int:
    """
    Solution to Project Euler #225
    """
    it = iter(tribonacciOddNonDivisorGenerator(init_terms=init_terms))
    num = -1
    for i in range(odd_non_divisor_number):
        num = next(it)
        #print(i + 1, num)
    return num

# Problem 226
def findBlacmangeValue(x: CustomFraction) -> CustomFraction:
    x -= x.numerator // x.denominator
    if x.denominator < 2 * x.numerator:
        x = 1 - x
    a0, b0 = 0, CustomFraction(1, 1)
    while x != 0 and not x.denominator & 1:
        a0 += b0 * x
        b0 /= 2
        x.denominator >>= 1
        if x.denominator < 2 * x.numerator:
            x = 1 - x
    if x == 0: return a0
    a, b = x, CustomFraction(1, 2)
    x2 = 2 * x
    if x2.denominator < 2 * x2.numerator:
        x2 = 1 - x2
    while x2 != x and x2 != 0:
        #print(x2, x)
        a += b * x2
        b /= 2
        x2 = 2 * x2
        if x2.denominator < 2 * x2.numerator:
            x2 = 1 - x2
    
    res = a if x2 == 0 else a / (1 - b)
    #print(a0, b0, res)
    return a0 + b0 * res
    #return res

def findBlacmangeIntegralValue(x: CustomFraction) -> CustomFraction:
    q = CustomFraction(x.numerator // x.denominator, 1)
    x -= q
    a0, b0 = 0, CustomFraction(1, 1)
    if x.denominator < 2 * x.numerator:
        x = 1 - x
        a0 += b0 / 2
        b0 = -b0
    a0, b0 = 0, CustomFraction(1, 1)
    while x != 0 and not x.denominator & 1:
        a0 += b0 * x * x / 2
        b0 /= 4
        x.denominator >>= 1
        if x.denominator < 2 * x.numerator:
            x = 1 - x
            a0 += b0 / 2
            b0 = -b0
    if x == 0: return a0 + q / 2
    a, b = x * x / 2, CustomFraction(1, 4)
    x2 = 2 * x
    if x2.denominator < 2 * x2.numerator:
        x2 = 1 - x2
        a += b / 2
        b = -b
    while x2 != x and x2 != 0:
        #print(x2, x)
        a += b * x2 * x2 / 2
        b /= 4
        x2 = 2 * x2
        if x2.denominator < 2 * x2.numerator:
            x2 = 1 - x2
            a += b / 2
            b = -b
    
    res = a if x2 == 0 else a / (1 - b)
    #print(a0, b0, res)
    return a0 + b0 * res + q / 2

def blacmangeCircleIntersectionArea(eps: float=10 ** -9) -> float:

    # Review- try to generalise to any circle

    # Rightmost intersection point is at (1 / 2, 1 / 2)
    # Find leftmost intersection point
    centre = (CustomFraction(1, 4), CustomFraction(1, 2))
    rad = CustomFraction(1, 4)
    rad_sq = rad * rad

    lft, rgt = CustomFraction(0, 1), CustomFraction(1, 4)
    while rgt - lft >= eps:
        x = lft + (rgt - lft) / 2
        y = findBlacmangeValue(x)
        v = (x - centre[0], y - centre[1])
        d_sq = v[0] * v[0] + v[1] * v[1]
        if d_sq == rad_sq:
            print("exact found")
            break
        elif d_sq > rad_sq:
            lft = x
        else: rgt = x
    else:
        x = lft + (rgt - lft) / 2
        y = findBlacmangeValue(x)
        v = (x - centre[0], y - centre[1])
        d_sq = v[0] * v[0] + v[1] * v[1]
        print(d_sq, d_sq.numerator / d_sq.denominator)
    #print(lft, rgt)
    #print((x, y), (x.numerator / x.denominator, y.numerator / y.denominator))
    b_area = findBlacmangeIntegralValue(CustomFraction(1, 2)) - findBlacmangeIntegralValue(x)
    x2 = CustomFraction(1, 2)
    y2 = findBlacmangeValue(x2)
    trap_area = (y + y2) * (x2 - x) / 2
    area1 = (b_area - trap_area)
    area1 = area1.numerator / area1.denominator
    angle = math.acos((v[0].numerator / v[0].denominator) * (rad.denominator / rad.numerator))
    #print(area1)
    #print(angle * 180 / math.pi)
    res = area1 + .5 * (angle - math.sin(angle)) * (rad_sq.numerator / rad_sq.denominator)
    return res


# Problem 227
def chaseGameExpectedNumberOfTurns(die_n_faces: int=6, n_opts_left: int=1, n_opts_right: int=1, n_players: int=100, separation_init: int=50) -> float:
    m = n_players >> 1
    n_opts_still = die_n_faces - n_opts_left - n_opts_right
    n_unchanged = (n_opts_still ** 2 + n_opts_left ** 2 + n_opts_right ** 2)
    n_shift1 = (n_opts_left + n_opts_right) * n_opts_still
    n_shift2 = n_opts_left * n_opts_right
    #print(n_unchanged, n_shift1, n_shift2, n_unchanged + 2 * (n_shift1 + n_shift2), die_n_faces ** 2)
    T = np.zeros([m, m])
    for i in range(m):
        T[i, i] = n_unchanged
    T[0, 0] += n_shift2
    for i in range(1, m):
        T[i, i - 1] += n_shift1
    for i in range(2, m):
        T[i, i - 2] += n_shift2
    for i in range(m):
        #print(i)
        j1 = min(i + 1, n_players - i - 3)
        j2 = min(i + 2, n_players - i - 4)
        #print(i, j1, j2)
        T[i, j1] += n_shift1
        T[i, j2] += n_shift2
    T /= die_n_faces ** 2
    #print(T)
    eig_vals, eig_vecs = np.linalg.eig(T)
    #print(eig_vals)
    #print(eig_vals)
    #print(eig_vecs)
    P = np.zeros([m, m])
    D = np.zeros([m, m])
    D2 = np.zeros([m, m])
    for i in range(m):
        D[i, i] = eig_vals[i]
        D2[i, i] = 1 / (1 - eig_vals[i])
        P[i] = eig_vecs[i]
    P = P.transpose()
    P_inv = np.linalg.inv(P)
    #print(P, P_inv, D)
    #print(np.matmul(P, P_inv))
    #print(np.dot(eig_vecs[0], eig_vecs[1]))
    #print(P)
    #print(D2)
    M = np.matmul(P_inv, np.matmul(D2, P))
    #print(T)
    #print(M)
    v = np.zeros([m])
    i = min(separation_init, n_players - separation_init) - 1
    #print(f"i = {i}")
    v[i] = 1
    #print(v)
    v2 = np.matmul(M, v)
    #print(v2)
    res = sum(v2)
    #print(res2)
    return res

# Problem 228
def convexPolygonsAroundOriginMinkowskiSum(poly1: List[Tuple[float, Tuple[float, float]]], poly2: List[Tuple[float, Tuple[float, float]]]) -> List[Tuple[float, float]]:
    poly1.sort()
    poly2.sort()
    if poly1[0][0] > poly2[0][0]: poly1, poly2 = poly2, poly1
    n1, n2 = len(poly1), len(poly2)
    i1 = 0
    v_lst = []
    for i2 in range(n2):
        for i1 in range(i1, n1 - 1):
            if poly1[i1 + 1][0] >= poly2[i2][0]:
                break
        else: break
        v_lst.append(tuple(x + y for x, y in zip(poly1[i1][1], poly2[i2][1])))
        v_lst.append(tuple(x + y for x, y in zip(poly1[i1 + 1][1], poly2[i2][1])))
    for i2 in range(i2, n2):
        v_lst.append(tuple(x + y for x, y in zip(poly1[-1][1], poly2[i2][1])))
        v_lst.append(tuple(x + y for x, y in zip(poly1[0][1], poly2[i2][1])))
    
    return grahamScan(v_lst, include_border_points=True)

def regularPolygonMinkowskiSum(vertex_counts: List[int]) -> List[Tuple[float, float]]:

    poly_lst = []
    for cnt in vertex_counts:
        lst = []
        for i in range(cnt):
            angle = (2 * i + 1) * math.pi / cnt
            lst.append((angle, (math.cos(angle), math.sin(angle))))
        poly_lst.append(lst)
    
    curr = poly_lst[0]
    for i in range(1, len(poly_lst)):
        print(i, len(curr))
        lst = convexPolygonsAroundOriginMinkowskiSum(curr, poly_lst[i])
        curr = []
        for v in lst:
            angle = math.atan2(v[1], v[0])
            if angle < 0: angle += 2 * math.pi
            curr.append((angle, v))
    return [x[1] for x in curr]

    """
    while len(poly_lst) > 1:
        print(len(poly_lst))
        print([len(x) for x in poly_lst])
        prev = poly_lst
        poly_lst = []
        for i in range(0, len(prev) - 1, 2):
            lst = convexPolygonsAroundOriginMinkowskiSum(prev[i], prev[i + 1])
            #print(lst)
            lst2 = []
            for v in lst:
                angle = math.atan2(v[1], v[0])
                if angle < 0: angle += 2 * math.pi
                lst2.append((angle, v))
            poly_lst.append(lst2)
        if len(prev) & 1:
            poly_lst.append(prev[-1])
    return [x[1] for x in poly_lst[0]]
    """

def regularPolygonMinkowskiSumSideCount(vertex_counts: List[int]=list(range(1864, 1910))) -> int:

    #res = regularPolygonMinkowskiSum(vertex_counts)
    #for v in res:
    #    print(v[0], v[1])
    #return res
    mn, mx = min(vertex_counts), max(vertex_counts)
    rng = mx - mn
    
    res = sum(vertex_counts) - (len(vertex_counts) - 1)
    for num in range(2, rng + 1):
        d_cnt = 0
        for v_cnt in vertex_counts:
            d_cnt += (not v_cnt % num)
        if d_cnt == 1: continue
        euler_tot = 1
        for num2 in range(2, num):
            euler_tot += (gcd(num2, num) == 1)
        res -= euler_tot * (d_cnt - 1)
    return res

# Problem 229
def fourRepresentationsUsingSquaresCount(mults: Tuple[int]=(1, 2, 3, 7), num_max: int=2 * 10 ** 9) -> int:
    part_size = 10 ** 8
    n_mults = len(mults)
    mults = sorted(set(mults))
    bm_target = (1 << n_mults) - 1
    res = 0
    md168_set = set()
    sq_cnt = 0
    for part_start in range(0, num_max + 1, part_size):
        print(part_start, res)
        part_end = min(part_start + part_size - 1, num_max)
        bm_sieve = [0] * (part_end - part_start + 1)
        part_end_sqrt = isqrt(part_end)
        for a in range(1, part_end_sqrt + 1):
            a_sq = a ** 2
            b_mn = max(1, isqrt((part_start - a_sq - 1) // mults[-1]) + 1 if a_sq < part_start else 0)
            b_mx = isqrt(num_max - a_sq)
            #if not a % 100: print(f"a = {a} of {part_end_sqrt}, b_max = {b_mx}")
            j_mn = len(mults) - 1
            j_mx = len(mults)
            for b in range(b_mn, b_mx + 1):
                b_sq = b ** 2
                #print(a, b)
                for j_mn in reversed(range(1, j_mn + 1)):
                    if a_sq + mults[j_mn - 1] * b_sq < part_start: break
                else: j_mn = 0
                
                for j in range(j_mn, j_mx):
                    m = mults[j]
                    num = a_sq + m * b_sq
                    if num > part_end:
                        j_mx = j
                        break
                    bm2 = 1 << j
                    num2 = num - part_start
                    #print(num, part_start, num2, len(bm_sieve))
                    if bm_sieve[num2] & bm2: continue
                    bm_sieve[num2] |= bm2
                    if (bm_sieve[num2] == bm_target):
                        res += 1
                        if isqrt(num) ** 2 == num:
                            #print(num)
                            sq_cnt += 1
                        #sq_cnt += (isqrt(num) ** 2 == num)
                        md168_set.add(num % 168)
    print(sorted(md168_set))
    print(f"square count = {sq_cnt}")
    return res

def fourSquaresRepresentationCountSpecialised(num_max: int=2 * 10 ** 9) -> int:

    # Try to generalise to arbitrary k using the Legendre/Jacobi symbol
    # and quadratic reciprocity. Possibly in first instance, restricting
    # to either prime or unit values of k allowed.

    ps = SimplePrimeSieve()
    def primeCheck(num: int) -> bool:
        return ps.millerRabinPrimalityTestWithKnownBounds(num, max_n_additional_trials_if_above_max=10)[0]

    p_prods = SortedSet()
    res = 0
    for num in range(0, num_max, 168):
        for r in (1, 25, 121):
            p = num + r
            b = primeCheck(p)
            #print(p, b)
            if not b: continue
            add_set = {p}
            res += isqrt(num_max // p)
            for p_prod in p_prods:
                num2 = p * p_prod
                if num2 > num_max: break
                res += isqrt(num_max // num2)
                add_set.add(num2)
            for num2 in add_set:
                p_prods.add(num2)
            #print(p, res)
    #print(res)
    for num in range(1, isqrt(num_max) + 1):
        pf = calculatePrimeFactorisation(num)
        remain_criterion = {0, 1, 2, 3}
        if 2 in pf.keys():
            remain_criterion.remove(2)
            if pf[2] >= 2: remain_criterion.remove(3)
        for p in pf.keys() - {2}:
            if 0 in remain_criterion and p % 4 == 1:
                remain_criterion.remove(0)
                if not remain_criterion:
                    break
            if 1 in remain_criterion and p % 8 in {1, 3}:
                remain_criterion.remove(1)
                if not remain_criterion: break
            if 2 in remain_criterion and p % 3 == 1:
                remain_criterion.remove(2)
                if not remain_criterion: break
            if 3 in remain_criterion and p % 7 in {1, 2, 4}:
                remain_criterion.remove(3)
                if not remain_criterion: break
        else:
            continue
        #print(num, num ** 2)
        res += 1
    return res

# Problem 230
def fibonacciWordsSum(
    A: int=1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679,
    B: int=8214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196,
    poly_coeffs: Tuple[int]=(127, 19),
    exp_base: int=7,
    n_max: int=17,
    base: int=10
) -> int:
    
    def evaluateTermNumber(n: int) -> int:
        res = 0
        for c in reversed(poly_coeffs):
            res = res * n + c
        return res * exp_base ** n

    mx_term = 0
    for n in range(0, n_max + 1):
        mx_term = max(mx_term, evaluateTermNumber(n))
    
    A_digs = []
    A2 = A
    while A2:
        A2, r = divmod(A2, base)
        A_digs.append(r)
    A_digs = A_digs[::-1]
    B_digs = []
    B2 = B
    while B2:
        B2, r = divmod(B2, base)
        B_digs.append(r)
    B_digs = B_digs[::-1]

    A_len = len(A_digs)
    B_len = len(B_digs)
    len_lst = [B_len, A_len + B_len]
    while len_lst[-1] < mx_term:
        len_lst.append(len_lst[-2] + len_lst[-1])
    
    #print(mx_term)
    #print(len_lst)
    
    def findDigit(term: int) -> int:
        #print(f"term = {term}")
        i = term - 1
        if i <= A_len:
            return A_digs[i]
        j = bisect.bisect_right(len_lst, i)
        while j > 1:
            #print(j, i, len_lst[j])
            if i >= len_lst[j - 2]:
                i -= len_lst[j - 2]
                j -= 1
            else: j -= 2
        if j == 1:
            #print(i)
            return B_digs[i - A_len] if i >= A_len else A_digs[i]
        return B_digs[i]

    res = 0
    for n in reversed(range(0, n_max + 1)):
        res = res * base + findDigit(evaluateTermNumber(n))
    return res


# Problem 231
def binomialCoefficientPrimeFactorisation(n: int, k: int) -> Dict[int, int]:

    ps = SimplePrimeSieve(n)
    k2 = n - k
    res = {}
    if n < 0 or k < 0 or k2 < 0:
        return res
    for p in ps.p_lst:
        cnt = 0
        n2 = n
        while n2:
            n2 //= p
            cnt += n2
        for num in (k, k2):
            while num:
                num //= p
                cnt -= num
        res[p] = cnt
    return res

def binomialCoefficientPrimeFactorisationSum(n: int=20 * 10 ** 6, k: int=15 * 10 ** 6) -> int:
    """
    Solution to Project Euler #231
    """

    res = sum(p * f for p, f in binomialCoefficientPrimeFactorisation(n, k).items())
    return res

# Problem 232
def probabilityPlayer2WinsFractionGivenSuccessProbabilitiesAndScores(
    points_required: int,
    player1_success_prob: CustomFraction=CustomFraction(1, 2),
    player1_success_points: int=1,
    player2_success_prob: CustomFraction=CustomFraction(1, 2),
    player2_success_points: int=1,
) -> CustomFraction:
    # Used in related problem where the value of T for Player 2 cannot be
    # changed
    memo = {}  
    def recur(player1_remain: int, player2_remain: int) -> CustomFraction:
        if player1_remain <= 0: return CustomFraction(0, 1)
        elif player2_remain <= 0: return CustomFraction(1, 1)
        
        args = (player1_remain, player2_remain)
        if args in memo.keys(): return memo[args]

        # At least one player succeeds on this turn
        #print("hi")
        res = player1_success_prob * (player2_success_prob * recur(player1_remain - player1_success_points, player2_remain - player2_success_points) +\
                                       (1 - player2_success_prob) * recur(player1_remain - player1_success_points, player2_remain)) +\
                (1 - player1_success_prob) *  player2_success_prob * recur(player1_remain, player2_remain - player2_success_points)

        unchanged_prob = (1 - player1_success_prob) * (1 - player2_success_prob)
        res /= (1 - unchanged_prob)

        memo[args] = res
        #print("hi2", memo)
        return res
    #print(player2_success_points, memo)
    res = recur(points_required, points_required)
    print(player2_success_points, memo)
    return res

def probabilityPlayer2WinsFraction(points_required: int=100) -> CustomFraction:
    res = 0
    player1_success_prob = CustomFraction(1, 2)
    player1_success_points = 1

    memo = {}  
    def recur(player1_remain: int, player2_remain: int) -> CustomFraction:
        if player2_remain <= 0: return CustomFraction(1, 1)
        if player1_remain <= 0: return CustomFraction(0, 1)
        
        args = (player1_remain, player2_remain)
        if args in memo.keys(): return memo[args]

        # At least one player succeeds on this turn
        #print("hi")
        res = 0
        for T in itertools.count(1):
            player2_success_prob = CustomFraction(1, 2 ** T)
            player2_success_points = 2 ** (T - 1)
            ans = player1_success_prob * (player2_success_prob * recur(player1_remain - player1_success_points, player2_remain - player2_success_points) +\
                                        (1 - player2_success_prob) * recur(player1_remain - player1_success_points, player2_remain)) +\
                    (1 - player1_success_prob) *  player2_success_prob * recur(player1_remain, player2_remain - player2_success_points)

            unchanged_prob = (1 - player1_success_prob) * (1 - player2_success_prob)
            ans /= (1 - unchanged_prob)
            res = max(res, ans)
            if player2_success_points >= player2_remain: break

        memo[args] = res
        #print("hi2", memo)
        return res
    #print(player2_success_points, memo)

    # Transpose problem into one where each turn player 2 goes first, so we
    # can avoid at each step accounting for player 2 responding to the outcome
    # of player 1's turn
    res = player1_success_prob * recur(points_required - 1, points_required) + (1 - player1_success_prob) * recur(points_required, points_required)
    #print(player2_success_points, memo)
    #print(memo)
    return res

    """
    for T in itertools.count(1):
        player2_success_prob = CustomFraction(1, 2 ** T)
        player2_success_points = 2 ** (T - 1)
        frac = probabilityPlayer2WinsFractionGivenSuccessProbabilitiesAndScores(
            points_required=points_required,
            player1_success_prob=player1_success_prob,
            player1_success_points=player1_success_points,
            player2_success_prob=player2_success_prob,
            player2_success_points=player2_success_points,
        )
        print(T, frac.numerator / frac.denominator)
        res = max(res, frac)
        if player2_success_points >= points_required: break
    return res
    """

def probabilityPlayer2WinsFloat(points_required: int=100) -> CustomFraction:
    """
    Solution to Project Euler #132
    """
    res = probabilityPlayer2WinsFraction(points_required=points_required)
    #print(res)
    return res.numerator / res.denominator


# Problem 233
def factorisationsGenerator(num: int) -> Generator[Dict[int, int], None, None]:
    pf = calculatePrimeFactorisation(num)
    yield from factorisationsFromPrimeFactorisationGenerator(pf)
    return

def factorisationsFromPrimeFactorisationGenerator(prime_factorisation: Dict[int, int]) -> Generator[Dict[int, int], None, None]:
    pf = prime_factorisation
    #print(pf)
    p_lst = sorted(pf.keys())
    n_p = len(p_lst)
    f_lst = [pf[p] for p in p_lst]
    z_cnt = [0]

    #print(p_lst, f_lst)

    curr = {}
    def recur(idx: int, num: int, prev: int=2) -> Generator[Dict[int, int], None, None]:
        #print(idx, num, prev, curr)
        if idx == n_p:
            #print("hi2")
            if num < prev: return
            #print(idx, num, prev, f_lst, curr, z_cnt[0])
            curr[num] = curr.get(num, 0) + 1
            if z_cnt[0] == n_p:
                yield dict(curr)
            else:
                yield from recur(0, 1, prev=num)
            curr[num] -= 1
            if not curr[num]: curr.pop(num)
            return
        f0 = f_lst[idx]
        num0 = num
        #print(f"idx = {idx}, f0 = {f0}, f_lst = {f_lst}, z_cnt = {z_cnt}, curr = {curr}")
        if not f0:
            yield from recur(idx + 1, num, prev=prev)
            return
        for i in range(f0):
            #print(f"i = {i}")
            yield from recur(idx + 1, num, prev=prev)
            #print(f"i = {i}, idx = {idx}, f_lst[idx] = {f_lst[idx]}")
            num *= p_lst[idx]
            f_lst[idx] -= 1
        #print("finished loop")
        z_cnt[0] += 1
        #print(idx, num, prev)
        yield from recur(idx + 1, num, prev=prev)
        num = num0
        z_cnt[0] -= 1
        f_lst[idx] = f0
        return
    
    yield from recur(0, 1, prev=2)
    return

def circleInscribedSquareSideLengthWithLatticePointCount(n_lattice_points: int=420, max_inscribed_square_side_length: int=10 ** 11) -> int:
    """
    Solution to Project Euler #233
    """
    q, r = divmod(n_lattice_points, 8)
    if r != 4: return 0
    
    target = (q << 1) + 1
    print(target)
    p_pow_opts = []
    mx_n_p = 0
    for fact in factorisationsGenerator(target):
        #print(fact)
        p_pows = {}
        for num, f in fact.items():
            if not num & 1: break
            num2 = num >> 1
            p_pows[num2] = f
        else:
            p_pow_opts.append(p_pows)
            mx_n_p = max(mx_n_p, len(p_pows))
    print(p_pow_opts)

    ps = SimplePrimeSieve()
    p_gen = iter(ps.endlessPrimeGenerator())
    
    p_r1_lst = []
    p_other_lst = []
    while True:
        p = next(p_gen)
        r = p & 3
        if r == 1:
            p_r1_lst.append(p)
            if len(p_r1_lst) >= mx_n_p: break
            continue
        p_other_lst.append(p)
    p_pow_opts2 = []
    print(p_r1_lst)
    mx_p_r1 = 0
    for opts in p_pow_opts:
        print(opts)
        i_p = 0#sum(opts.values()) - 1
        num = 1
        opts2 = sorted(opts.keys())
        for j in reversed(range(1, len(opts2))):
            m = opts2[j]
            f = opts[m]
            for _ in range(f):
                num *= p_r1_lst[i_p] ** m
                print(i_p, p_r1_lst[i_p], m, num)
                i_p += 1
                if num > max_inscribed_square_side_length: break
            else: continue
            break
        else:
            j = 0
            m = opts2[j]
            f = opts[m]
            for _ in range(f - 1):
                num *= p_r1_lst[i_p] ** m
                print(i_p, p_r1_lst[i_p], m, num)
                i_p += 1
                if num > max_inscribed_square_side_length: break
            else:
                mx_p_r1 = max(mx_p_r1, integerNthRoot(max_inscribed_square_side_length // num, m))
                num *= p_r1_lst[i_p] ** m
                print(i_p, p_r1_lst[i_p], m, num)
                i_p += 1
                if num > max_inscribed_square_side_length: continue
                p_pow_opts2.append((num, opts))
    
    p_pow_opts = sorted(p_pow_opts2)
    print(p_pow_opts)
    print(mx_p_r1)
    if not p_pow_opts: return 0
    mult_mx = max_inscribed_square_side_length // p_pow_opts[0][0]
    print(mult_mx)

    mults = SortedList([1])
    for p in p_other_lst:
        #mults.add(p)
        for i in itertools.count(0):
            num = mults[i] * p
            if num > mult_mx: break
            mults.add(num)
    while True:
        p = next(p_gen)
        if p & 3 == 1:
            if p <= mx_p_r1:
                p_r1_lst.append(p)
            continue
        if p > mult_mx: break
        #mults.add(p)
        for i in itertools.count(0):
            num = mults[i] * p
            if num > mult_mx: break
            mults.add(num)
    
    print(len(mults))
    print(mults[:50])
    mults_cumu = [0]
    for num in mults:
        mults_cumu.append(mults_cumu[-1] + num)

    while True:
        p = next(p_gen)
        if p > mx_p_r1: break
        if p & 3 == 1:
            p_r1_lst.append(p)
    print(len(p_r1_lst))
    #n_p_r1 = len(p_r1_lst)

    def primeProductGenerator(pow_opts: Dict[int, int], p_lst: List[int], mx: int) -> Generator[int, None, None]:
        pow_lst = []
        for num in reversed(sorted(pow_opts.keys())):
            f = pow_opts[num]
            for f_ in range(f - 1):
                pow_lst.append((num, True))
            pow_lst.append((num, False))
        n_pow = len(pow_lst)
        n_p = len(p_lst)
        print(pow_opts, pow_lst)

        curr_incl = set()
        def recur(idx: int=0, curr: int=1, mn_p_idx: int=0) -> Generator[int, None, None]:
            if idx == n_pow:
                yield curr
                return
            for p_idx in range(mn_p_idx, n_p):
                if p_idx in curr_incl: continue
                p = p_lst[p_idx]
                exp, b = pow_lst[idx]
                nxt = curr * p ** exp
                if nxt > mx: break
                curr_incl.add(p_idx)
                yield from recur(idx=idx + 1, curr=nxt, mn_p_idx=p_idx + 1 if b else 0)
                curr_incl.remove(p_idx)
            return

        yield from recur(idx=0, curr=1, mn_p_idx=0)
        return

    res = 0
    cnt = 0
    for opts in p_pow_opts:
        for num in primeProductGenerator(opts[1], p_r1_lst, max_inscribed_square_side_length):
            j = mults.bisect_right(max_inscribed_square_side_length // num)
            #print(num, max_inscribed_square_side_length // num, mults[j - 1])
            cnt += j
            res += num * mults_cumu[j]
    print(f"total count = {cnt}")
    return res

# Problem 234
def semiDivisibleNumberCount(n_max: int=999966663333) -> int:
    """
    Solution to Project Euler #234
    """
    if n_max < 4: return 0

    #sqrt_max = isqrt(n_max - 1) + 1
    #print(f"sqrt_max = {sqrt_max}")
    ps = SimplePrimeSieve()
    p_gen = iter(ps.endlessPrimeGenerator())


    def singleDivisibleCount(p1: int, p2: int, mn: int, mx: int) -> int:
        if mn > mx: return 0
        q = p1 * p2
        i1, i2 = (mn - 1) // p1, mx // p1
        j1, j2 = (mn - 1) // p2, mx // p2
        k1, k2 = (mn - 1) // q, mx // q
        ans1 = p1 * (i2 * (i2 + 1) - i1 * (i1 + 1)) >> 1
        ans2 = p2 * (j2 * (j2 + 1) - j1 * (j1 + 1)) >> 1
        ans3 = q * (k2 * (k2 + 1) - k1 * (k1 + 1))
        #print(p1, p2, ans1, ans2, ans3)
        return ans1 + ans2 - ans3

    res = 0
    p2 = next(p_gen)
    p2_sq = p2 ** 2
    while True:
        p1, p1_sq = p2, p2_sq
        p2 = next(p_gen)
        p2_sq = p2 ** 2
        if p2_sq > n_max:
            ans = singleDivisibleCount(p1, p2, p1_sq + 1, n_max)
            print(p1, p2, p1_sq + 1, n_max, ans)
            res += ans
            break
        ans = singleDivisibleCount(p1, p2, p1_sq + 1, p2_sq - 1)
        #print(p1, p2, p1_sq + 1, p2_sq - 1, ans)
        res += ans
    return res


# Problem 235
def arithmeticGeometricSeries(a: float=900, b: int=-3, n: int=5000, val: float=-6 * 10 ** 11, eps: float=10 ** -13) -> float:

    if b < 0:
        a, b, val = -a, -b, -val
    #r0 = -a / b
    print(a, b, val)

    def func(r: float, n: int) -> float:
        #print(r)
        if r == 1: return 900 * n - 1.5 * n * (n + 1)
        res = (a + b * n) * r ** (n + 1) - (a + b * (n + 1)) * r ** n - a * r + (a + b)
        return float(res) / (r - 1) ** 2

    print(f"values at 1 - 10 ** -10, 1 and 1 + 10 ** -10")
    print(func(1 - 10 ** -9, n), func(1, n), func(1 + 10 ** -11, n))

    r0 = 0
    r = 0
    diff = 1
    while True:
        r2 = r0 + diff
        y = func(r2, n)
        if y >= val: break
        r = r2
        print(r, y)
        diff *= 1.1
    print(r, y)
    diff0 = 0 if diff == 1 else diff / 2
    lft, rgt = r, r2
    while rgt - lft >= eps:
        mid = lft + (rgt - lft) * .5
        
        y = func(mid, n)
        print(mid, y)
        if y == val:
            return mid
        elif y < val: lft = mid
        else: rgt = mid
    return lft + (rgt - lft) * .5

# Problem 236
def luxuryHamperPossibleMValues(
    pairs: List[Tuple[int, int]]=[(5248, 640), (1312, 1888), (2624, 3776), (5760, 3776), (3936, 5664)],
) -> List[CustomFraction]:

    # Review- try to make faster
    n_pairs = len(pairs)
    pairs2 = [(x * y, (x, y)) for x, y in pairs]
    pairs2.sort()
    poss_m_vals = {}
    #print(pairs2)
    #print(f"j = 0")
    i2_0 = 1
    for i1 in range(1, pairs2[0][1][0] + 1):
        #print(i1, pairs2[0][1][0])
        for i2_0 in range(i2_0, pairs2[0][1][1] + 1):
            if i1 * pairs2[0][1][1] < i2_0 * pairs2[0][1][0]:
                break
        else: break
        #print(i1, i2_0)
        for i2 in range(i2_0, pairs2[0][1][1] + 1):
            if gcd(i1, i2) != 1: continue
            m = CustomFraction(i2 * pairs2[0][1][0], i1 * pairs2[0][1][1])
            poss_m_vals[m] = [(i1, i2)]
    for j in range(1, len(pairs2)):
        #print(len(poss_m_vals))
        #print(CustomFraction(1476, 1475) in poss_m_vals)
        #print(f"j = {j}")
        prev_m_vals = poss_m_vals
        poss_m_vals = {}
        for m in prev_m_vals.keys():
            mult1, mult2 = pairs2[j][1][1] * m.numerator, pairs2[j][1][0] * m.denominator
            l = lcm(mult1, mult2)
            i2 = l // mult2
            if i2 > pairs2[j][1][1]: continue
            i1 = l // mult1
            poss_m_vals[m] = prev_m_vals[m]
            poss_m_vals[m].append((i1, i2))
    tots = [sum(x[0] for x in pairs), sum(x[1] for x in pairs)]
    #print(tots)
    #prev_m_vals = poss_m_vals
    #poss_m_vals = {}

    def bruteForce(m: CustomFraction) -> bool:
        mults = poss_m_vals[m]
        mx_incl = [x[1][1] // y[1] for x, y in zip(pairs2, mults)]
        #print(pairs2, mults)
        #print(mx_incl)
        #sol = [0] * (n_pairs >> 1)
        seen = set()
        vals = set()
        terms = [mults[idx][0] * tots[1] * m.denominator - mults[idx][1] * tots[0] * m.numerator for idx in range(n_pairs)]
        def recur1(idx: int, net: int) -> None:
            if idx == (n_pairs >> 1):
                vals.add(net)
                #vals[net] = tuple(sol)
                #return not net#curr1 * tots[1] * m.denominator == curr2 * tots[0] * m.numerator
                return
            args = (idx, net)
            if args in seen: return False
            for num in range(1, mx_incl[idx] + 1):
                #sol[idx] = (num * mults[idx][0], num * mults[idx][1])
                recur1(idx + 1, net + num * terms[idx])
            seen.add(args)
            return
        
        #sol = [0] * n_pairs
        def recur2(idx: int, net: int) -> None:
            if idx == (n_pairs >> 1) - 1:
                if -net not in vals:
                    return False
                #sol2 = vals[-net]
                #for i in range(n_pairs >> 1):
                #    sol[i] = sol2[i]
                #print(m, sol)
                return True
            args = (idx, net)
            if args in seen: return False
            for num in range(1, mx_incl[idx] + 1):
                #sol[idx] = (num * mults[idx][0], num * mults[idx][1])
                if recur2(idx - 1, net + num * terms[idx]):
                    return True
            seen.add(args)
            return False

        recur1(0, 0)
        seen = set()
        res = recur2(n_pairs - 1, 0)

        #if res:
        #    print(sol)
        return res

    def mHasSolution(m: CustomFraction) -> bool:
        #print(m)
        mult1, mult2 = tots[1] * m.numerator, tots[0] * m.denominator
        l = lcm(mult1, mult2)
        #print(mult1, mult2, l)
        i1_incr = l // mult1
        i2_incr = l // mult2
        #print(i1_incr, i2_incr)
        #print(m, i1_incr, i2_incr, abs(i1_incr - i2_incr))
        #print(i1_incr, i2_incr)
        diff0 = i2_incr - i1_incr
        diffs = []
        for i1, i2 in poss_m_vals[m]:
            #print((i1, i2))
            diffs.append(i2 - i1 - diff0)
        #mults = []
        l = abs(diff0) if diff0 else 1
        for idx in set(range(n_pairs)):
            if not diffs[idx]: continue
            l = lcm(l, abs(diffs[idx]))
        #print(f"diff0 = {diff0}")
        #print(f"diffs = {diffs}")
        #print(l)
        #for idx in range(n_pairs):
        #    if not diffs[idx]:
        #        mults.append(1)
        #        continue
        #    l2 = l // abs(diffs[idx])
        #    if l2 > pairs2[idx][1][0]:
        #        return False
        #    mults.append(l2)
        
        #print(f"mults = {mults}")
        
        memo = {}
        sol = [0] * n_pairs
        def recur(idx: int, curr: int, mult_sm: int) -> bool:
            if idx == n_pairs:
                #print("hello")
                return not curr
            args = (idx, curr, mult_sm)
            if args in memo.keys(): return memo[args]
            #mult0 = mults[idx]
            #print(mult0, pairs[idx][0])
            res = False
            for mult in range(1, (pairs2[idx][1][0] // (poss_m_vals[m][idx][0] * i1_incr)) + 1):
                sol[idx] = mult
                if recur(idx + 1, curr + mult * diffs[idx], mult_sm + mult):
                    res = True
                    break
            memo[args] = res
            return res
        #print("hi")
        res = recur(0, 0, 0)
        if res:
            cnts1 = [x * y[0] for x, y in zip(sol, poss_m_vals[m])]
            #print(cnts1)
            cnts2 = [x * y[1] for x, y in zip(sol, poss_m_vals[m])]
            #print(cnts2)
            #print(sum(cnts1) * sum(x[1] for x in pairs) * m.denominator)
            #print(sum(cnts2) * sum(x[0] for x in pairs) * m.numerator)
        return res
        """
        print(m)
        mult1, mult2 = tots[1] * m.denominator, tots[0] * m.numerator
        l = lcm(mult1, mult2)
        print(mult1, mult2, l)
        i1_incr = l // mult1
        i2_incr = l // mult2
        #print(m, i1_incr, i2_incr, abs(i1_incr - i2_incr))
        print(i1_incr, i2_incr)
        diff0 = i2_incr - i1_incr
        diffs = []
        for i1, i2 in poss_m_vals[m]:
            print((i1, i2))
            diffs.append(i2 - i1)
        mults = []
        l = abs(diff0) if diff0 else 1
        for idx in set(range(n_pairs)):
            if not diffs[idx]: continue
            l = lcm(l, abs(diffs[idx]))
        print(f"diff0 = {diff0}")
        print(f"diffs = {diffs}")
        print(l)
        for idx in range(n_pairs):
            if not diffs[idx]:
                mults.append(1)
                continue
            l2 = l // abs(diffs[idx])
            if l2 > pairs2[idx][1][0]:
                return False
            mults.append(l2)
        
        print(f"mults = {mults}")
        
        memo = {}
        def recur(idx: int, curr: int) -> bool:
            if idx == n_pairs:
                #print("hello")
                return (curr * diff0 > 0) and not abs(curr) % abs(diff0)
            args = (idx, curr)
            if args in memo.keys(): return memo[args]
            mult0 = mults[idx]
            #print(mult0, pairs[idx][0])
            res = False
            for mult in range(mult0, pairs2[idx][1][0] + 1, mult0):
                if recur(idx + 1, curr + mult * diffs[idx]):
                    res = True
                    break
            memo[args] = res
            return res
        #print("hi")
        res = recur(0, 0)
        return res
        """
    
    #print(mHasSolution(CustomFraction(1476, 1475)))
    #print(bruteForce(CustomFraction(1476, 1475)))
    #return []
    
    

    res = []
    cnt = 0
    for m in poss_m_vals.keys():
        #print(cnt, m, len(res))
        cnt += 1
        
        if bruteForce(m):
            print(m)
            res.append(m)
        #if mHasSolution(m): res.append(m)

        """
        i2 = 0
        sub1 = sum(x[0] for x in poss_m_vals[m])
        sub2 = sum(x[1] for x in poss_m_vals[m])
        for i1 in range(i1_incr, tots[0] + 1, i1_incr):
            print(i1, tots[0])
            i2 += i2_incr
            if i1 < sub1 or i2 < sub2: continue
            if recur(m, 0, i1 - sub1, i2 - sub2): break
        else: continue
        res.append(m)
        print(f"found {m}")
        """
    """
    for j in range(1, len(pairs2)):
        i2_0 = 1
        print(len(poss_m_vals))
        print(CustomFraction(1476, 1475) in poss_m_vals)
        print(f"j = {j}")
        prev_m_vals = poss_m_vals
        poss_m_vals = {}
        for i1 in range(1, pairs2[j][1][0] + 1):
            for i2_0 in range(i2_0, pairs2[j][1][1] + 1):
                if i1 * pairs2[j][1][1] < i2_0 * pairs2[j][1][0]:
                    break
            else: break
            for i2 in range(i2_0, pairs2[j][1][1] + 1):
                if gcd(i1, i2) != 1: continue
                m = CustomFraction(i2 * pairs2[j][1][0], i1 * pairs2[j][1][1])
                if m not in prev_m_vals.keys(): continue
                poss_m_vals[m] = prev_m_vals[m]
                poss_m_vals[m].append((i1, i2))
    """
    #print(len(res))
    #print(CustomFraction(1476, 1475) in res)
    #print(res)
    #print(min(poss_m_vals.keys()), max(poss_m_vals.keys()))
    #print(poss_m_vals.keys())
    #print(poss_m_vals.get(CustomFraction(1476, 1475), None))
    print(f"number of solutions = {len(res)}")
    return res

def largestLuxuryHamperPossibleMValue(
    pairs: List[Tuple[int, int]]=[(5248, 640), (1312, 1888), (2624, 3776), (5760, 3776), (3936, 5664)],
) -> List[CustomFraction]:

    m_vals = luxuryHamperPossibleMValues(pairs)
    return max(m_vals)

# Problem 237
def playingBoardTourCount(n_rows: int=4, n_cols: int=10 ** 12, start_row: int=0, end_row: int=3, md: Optional[int]=10 ** 8) -> int:


    if start_row == end_row: raise ValueError("start_row and end_row must be different")
    init_state = [0] * n_rows
    init_state[start_row] = 1
    init_state[end_row] = 1
    init_state = tuple(init_state)

    states = []
    states_dict = {}

    state_memo = {}
    def getStateIndex(state_raw: Tuple[int]) -> int:
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

    #def addTransferEdge(state1: Tuple[int], state2: Tuple[int], qty: int=1) -> None:
    #    idx1, idx2 = map(getStateIndex, (state1, state2))
    #    while len(transfer_adj) <= idx1:
    #        transfer_adj.append({})
    #    transfer_adj[idx1][idx2] = transfer_adj[idx1].get(idx2, 0) + qty

    def getTransferOutEdges(state_idx: int) -> Dict[int, int]:
        state = states[state_idx]
        m = len(state)
        t_dict = {}

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
                """
                else:
                    idx2 = r_dict.pop(above)
                    curr[idx2] = stt
                    if stt not in incl_dict.keys(): incl_dict[stt] = idx2
                    recur(idx=idx + 1, above=0, non_zero_seen=True)
                    if incl_dict.get(stt, -1) == idx: curr[idx2] = stt
                    incl_dict.pop(stt)
                    r_dict[above] = idx2
                """
            curr.pop()
            return
        recur(idx=0, above=0)
        return t_dict
        """
        if state == (1, 1, 0, 0):
            t_dict[(1, 1, 2, 2)] = 1
            t_dict[(1, 0, 0, 1)] = 1
        elif state == (1, 1, 2, 2):
            t_dict[(1, 1, 2, 2)] = 1
            t_dict[(1, 0, 0, 1)] = 1
        elif state == (1, 0, 1, 0):
            t_dict[(1, 0, 1, 0)] = 1 # Reflected from (0, 1, 0, 1)
        elif state == (1, 0, 0, 1):
            t_dict[(1, 1, 0, 0)] = 2 # One reflected from (0, 0, 1, 1)
            t_dict[(0, 1, 1, 0)] = 1
            t_dict[(1, 2, 2, 1)] = 1
        elif state == (1, 2, 2, 1):
            t_dict[(1, 2, 2, 1)] = 1
            t_dict[(1, 1, 0, 0)] = 2 # One reflected from (0, 0, 1, 1)
        elif state == (0, 1, 1, 0):
            t_dict[(1, 0, 0, 1)] = 1
        return t_dict
        """
        


    def createTransferAdj(start_states: List[Tuple[int]]) -> List[Dict[int, int]]:
        seen = set()
        qu = deque()
        adj = []
        for state in start_states:
            idx = getStateIndex(state)
            if idx in seen: continue
            seen.add(idx)
            qu.append(idx)
        while qu:
            idx = qu.popleft()
            adj += [{} for _ in range(idx + len(adj) + 1)]
            #print(f"creating out edges for index {idx}, state {states[idx]}")
            adj[idx] = getTransferOutEdges(idx)
            for idx2 in adj[idx].keys() - seen:
                qu.append(idx2)
                seen.add(idx2)
            #print([(states[idx2], f) for idx2, f in adj[idx].items()])
        return adj
    """
    def transferFunction(state: Tuple[int]) -> Generator[Tuple[Tuple[int], int], None, None]:
        if state == (1, 1, 0, 0):
            yield ((1, 1, 2, 2), 1)
            yield ((1, 0, 0, 1), 1)
            return
        elif state == (1, 1, 2, 2):
            yield ((1, 1, 2, 2), 1)
            yield ((1, 0, 0, 1), 1)
            return
        elif state == (1, 0, 1, 0):
            yield ((1, 0, 1, 0), 1) # Reflected from (0, 1, 0, 1)
            return
        elif state == (1, 0, 0, 1):
            yield ((1, 1, 0, 0), 2) # One reflected from (0, 0, 1, 1)
            yield ((0, 1, 1, 0), 1)
            yield ((1, 2, 2, 1), 1)
            return
        elif state == (1, 2, 2, 1):
            yield ((1, 2, 2, 1), 1)
            yield ((1, 1, 0, 0), 2) # One reflected from (0, 0, 1, 1)
            return
        elif state == (0, 1, 1, 0):
            yield ((1, 0, 0, 1), 1)
            return
        return
    """
    #state_dict = {init_state: 1}
    """
    states = [(1, 1, 0, 0), (1, 1, 2, 2), (1, 0, 1, 0), (1, 0, 0, 1), (1, 2, 2, 1), (0, 1, 1, 0)]
    states_dict = {x: i for i, x in enumerate(states)}
    n_states = len(states)
    state_adj = [{} for _ in range(n_states)]
    for idx1 in range(n_states):
        state1 = states[idx1]
        for state2, f in transferFunction(state1):
            idx2 = states_dict[state2]
            state_adj[idx1][idx2] = f
    
    """
    state_adj = createTransferAdj([init_state])
    n_states = len(states)
    #print(states)
    #print(f"number of distinct reachable states = {n_states}")
    def multiplyStateAdj(state_adj1: List[Dict[int, int]], state_adj2: List[Dict[int, int]]) -> List[Dict[int, int]]:
        res = [{} for _ in range(n_states)]
        for idx1 in range(n_states):
            for idx2, f2 in state_adj1[idx1].items():
                for idx3, f3 in state_adj2[idx2].items():
                    res[idx1][idx3] = (res[idx1].get(idx3, 0) + f2 * f3)
                    if md is not None: res[idx1][idx3] %= md
        return res
    
    def applyStateAdj(state_adj: List[Dict[int, int]], state_f_dict: Dict[int, int]) -> Dict[int, int]:
        res = {}
        for idx, f in state_f_dict.items():
            for idx2, f2 in state_adj[idx].items():
                res[idx2] = (res.get(idx2, 0) + f * f2)
                if md is not None: res[idx2] %= md
        return res

    # binary lift
    state_adj_bin = state_adj
    curr = {getStateIndex(init_state): 1}
    m = n_cols - 1
    while True:
        if m & 1:
            curr = applyStateAdj(state_adj_bin, curr)
        m >>= 1
        if not m: break
        state_adj_bin = multiplyStateAdj(state_adj_bin, state_adj_bin)

    """
    curr = {init_state: 1}
    for _ in range(n_cols - 1):
        print(curr)
        prev = curr
        curr = {}
        for state, f in prev.items():
            for state2, f2 in transferFunction(state):
                curr[state2] = curr.get(state2, 0) + f * f2
    """
    # Find which states reached can be connected up so that
    # a single path visiting every square exactly once can
    # be formed, and summing the frequencies of those states
    # for the last column, as identified above
    res = 0
    #print(curr)
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
            #if num in pair_dict.keys():
            #    if pair_dict[num] != prev: break
            #else:
            #    pair_dict[num] = prev
            #    pair_dict[prev] = num
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
            if md is not None: res %= md
    return res

# Problem 238
def infiniteStringTourDigitSumStartSum(n_max: int=2 * 10 ** 15, s_0: int=14025256, s_mod: int=20300713, base: int=10) -> int:
    """
    Solution to Project Euler #238
    """
    # Review- Try to make faster
    # Consider alternative approach where iterate over the sums
    # rather than the starting values.
    it = itertools.chain([s_0], blumBlumShubPseudoRandomGenerator(s_0=s_0, s_mod=s_mod, t_min=0, t_max=s_mod - 1))
    term_dig_counts_cumu = [0]
    term_dig_sums_cumu = [0]
    z_count_cumu = [0]
    terms = []
    terms_cumu = []
    #terms_z_count_cumu = []
    seen = {}
    cycle_start = -1
    for i, num in enumerate(it):
        if num in seen.keys():
            cycle_start = seen[num]
            break
        terms.append(num)
        seen[num] = i
        d_cnt = 0
        d_sm = 0
        z_cnt = 0
        #lst = []
        while num:
            num, d = divmod(num, base)
            #lst.append(d)
            d_sm += d
            d_cnt += 1
            z_cnt += not d
        term_dig_counts_cumu.append(term_dig_counts_cumu[-1] + d_cnt)
        term_dig_sums_cumu.append(term_dig_sums_cumu[-1] + d_sm)
        z_count_cumu.append(z_count_cumu[-1] + z_cnt)
        """        
        terms.append(lst[::-1])
        terms_cumu.append([0])
        terms_z_count_cumu.append([0])
        for d in terms[-1]:
            terms_cumu[-1].append(terms_cumu[-1][-1] + d)
            terms_z_count_cumu[-1].append(terms_z_count_cumu[-1][-1] + (not d))
        """
    print(len(seen))
    print(len(terms))
    print(term_dig_counts_cumu[-1])
    print(term_dig_sums_cumu[-1])
    print(z_count_cumu[-1])
    print(cycle_start)

    def getDigit(idx: int) -> int:
        if idx < term_dig_counts_cumu[-1]:
            idx2 = idx
        else:
            idx0 = term_dig_counts_cumu[cycle_start]
            idx2 = ((idx - idx0) % (term_dig_counts_cumu[-1] - idx0)) + idx0
        i = bisect.bisect_right(term_dig_counts_cumu, idx2) - 1
        j = idx2 - term_dig_counts_cumu[i]
        #print(f"idx = {idx}, i = {i}, j = {j}")
        n_dig = term_dig_counts_cumu[i + 1] - term_dig_counts_cumu[i]
        num = terms[i]
        num //= base ** (n_dig - j - 1)
        return num % base

    def cycleDigitGenerator() -> Generator[int, None, None]:
        for term in terms[cycle_start:]:
            num2 = term
            lst = []
            while num2:
                num2, d = divmod(num2, base)
                lst.append(d)
            for d in reversed(lst):
                yield d
        return

    def reversedCycleDigitGenerator() -> Generator[int, None, None]:
        for term in terms[cycle_start:][::-1]:
            num2 = term
            while num2:
                num2, d = divmod(num2, base)
                yield d
        return
    
    #for idx in range(11):
    #    print(idx, getDigit(idx))

    def getTermCounts(idx: int) -> Tuple[List[int], List[int], List[int]]:
        #print(idx, len(terms))
        num = terms[idx]
        lst = []
        while num:
            num, d = divmod(num, base)
            lst.append(d)
        lst = lst[::-1]
        z_cumu = [0]
        digs_cumu = [0]
        for d in lst:
            z_cumu.append(z_cumu[-1] + (not d))
            digs_cumu.append(digs_cumu[-1] + d)
        return lst, digs_cumu, z_cumu
    
    def calculateIndexStartCounts(start_idx: int, prefixes: List[int]) -> int:
        #print(start_idx, prefixes)
        n_max2 = n_max + (prefixes[0] if prefixes else 0)

        q1, r1 = divmod(start_idx - term_dig_counts_cumu[cycle_start], term_dig_counts_cumu[-1])
        i = bisect.bisect_right(term_dig_counts_cumu, r1 + term_dig_counts_cumu[cycle_start]) - 1
        term_digs, term_digs_cumu, zcc = getTermCounts(i)

        start_z_cnt = q1 * (z_count_cumu[-1] - z_count_cumu[cycle_start]) + z_count_cumu[i] + zcc[r1 - term_dig_counts_cumu[i]]
        res = -(start_idx - start_z_cnt)
        print(f"initial subtraction for start_idx {start_idx} = {res}")

        q2, r2 = divmod(n_max2 - term_dig_sums_cumu[cycle_start], term_dig_sums_cumu[-1] - term_dig_sums_cumu[cycle_start])
        i = bisect.bisect_right(term_dig_sums_cumu, r2 + term_dig_sums_cumu[cycle_start]) - 1
        term_digs, term_digs_cumu, z_counts_cumu = getTermCounts(i)
        #print(i, term_digs_cumu, r2 + term_dig_sums_cumu[cycle_start] - term_dig_sums_cumu[i])
        j = bisect.bisect_right(term_digs_cumu, r2 + term_dig_sums_cumu[cycle_start] - term_dig_sums_cumu[i]) - 1
        res += term_dig_counts_cumu[cycle_start] - z_count_cumu[cycle_start] +\
                q2 * (term_dig_counts_cumu[-1] - term_dig_counts_cumu[cycle_start] - z_count_cumu[-1] + z_count_cumu[cycle_start]) +\
                (term_dig_counts_cumu[i] - term_dig_counts_cumu[cycle_start] - z_count_cumu[i] + z_count_cumu[cycle_start]) +\
                (j - z_counts_cumu[j])
        if not prefixes:
            return res

        prefs2 = sorted([x % (term_dig_sums_cumu[-1] - term_dig_sums_cumu[cycle_start]) for x in prefixes])

        if cycle_start: return -1 # TODO
        #cumu_qu = deque()
        #cumu_set = set()
        cumu_curr = 0
        cumu_lst = [0]
        for d in reversedCycleDigitGenerator():
            if not d: continue
            cumu_curr += d
            if cumu_curr > prefs2[-1]:
                cumu_curr -= d
                break
            cumu_lst.append(cumu_curr)
        cumu_qu = deque(cumu_curr - x for x in reversed(cumu_lst))
        cumu_set = set(cumu_qu)
        #print(cumu_qu, cumu_set)
        #for d in cycleDigitGenerator():
        #    if not d: continue
        #    cumu_curr += d
        #    cumu_mn = cumu_curr - prefs2[-1]
        #    while cumu_qu and cumu_qu[0] < cumu_mn:
        #        cumu_set.remove(cumu_qu.popleft())
        #    cumu_qu.append(cumu_curr)
        #    cumu_set.add(cumu_curr)
        mult = q2 - q1
        print(f"mult = {mult}")
        idx1 = (start_idx - term_dig_counts_cumu[cycle_start]) % (term_dig_counts_cumu[-1] - term_dig_counts_cumu[cycle_start])
        idx2 = term_dig_counts_cumu[i] - term_dig_counts_cumu[cycle_start] + j
        #print(i, j)
        #print(f"idx1 = {idx1}, idx2 = {idx2}")
        rpt_cnt = 0
        for idx, d in enumerate(cycleDigitGenerator()):
            if idx == idx1: mult += 1
            if idx == idx2:
                mult -= 1
                if not mult and idx >= idx1:
                    break
            if not d:
                continue
            cumu_curr += d
            cumu_mn = cumu_curr - prefs2[-1]
            while cumu_qu and cumu_qu[0] < cumu_mn:
                cumu_set.remove(cumu_qu.popleft())
            #print(cumu_qu, {cumu_curr - x for x in prefs2})
            for pref in prefs2:
                if cumu_curr - pref in cumu_set:
                    #print(f"mult = {mult}")
                    #res -= mult
                    rpt_cnt += mult
                    break
            #if not cumu_set.isdisjoint({cumu_curr - x for x in prefs2}):
            #    res -= mult
            #    rpt_cnt += mult
            cumu_qu.append(cumu_curr)
            cumu_set.add(cumu_curr)
        res -= rpt_cnt
        print(f"new value count = {res}, number of repeats = {rpt_cnt}")
        return res
        """
        n_max2 = n_max + prefs[0]
        q1, r1 = divmod(start_idx, term_dig_counts_cumu[-1])
        q2, r2 = divmod(n_max2 - term_dig_sums_cumu[cycle_start], term_dig_sums_cumu[-1] - term_dig_sums_cumu[cycle_start])

        prefs2 = sorted([x % (term_dig_sums_cumu[-1] - term_dig_sums_cumu[cycle_start]) for x in prefixes])
        if 0 in prefs2: return 0

        if q1 == q2:


        mult = max(0, q2 - q1 - 2)
        if mult:
            

        #idx0 = bisect.bisect_right(term_dig_counts_cumu, start_idx) - 1


        #ini = term_dig_sums_cumu[cycle_start]
        """
    
    prefs = []
    n_terms = calculateIndexStartCounts(0, prefs)
    res = n_terms
    k_vals_found = n_terms
    for start_idx in itertools.count(1):
        d = getDigit(start_idx - 1)
        if not d: continue
        prefs = [x + d for x in prefs]
        prefs.append(d)
        print(f"start_idx = {start_idx}, d = {d}, number of k values found = {k_vals_found} of {n_max}, res = {res}")
        print(f"prefs = {prefs}")
        #if start_idx > 50: break
        
        n_terms = calculateIndexStartCounts(start_idx, prefs)

        k_vals_found += n_terms
        res += n_terms * (start_idx + 1)
        if k_vals_found == n_max: break
        
    return res
    """
    q, r = divmod(n_max - term_dig_sums_cumu[cycle_start], term_dig_sums_cumu[-1] - term_dig_sums_cumu[cycle_start])
    i = bisect.bisect_right(term_dig_sums_cumu, r + term_dig_sums_cumu[cycle_start]) - 1
    term_digs, term_digs_cumu, z_counts_cumu = getTermCounts(i)
    j = bisect.bisect_right(term_digs_cumu, term_dig_sums_cumu[i] - term_dig_sums_cumu[cycle_start]) - 1
    n_terms = term_dig_counts_cumu[cycle_start] - z_count_cumu[cycle_start] +\
            q * (term_dig_counts_cumu[-1] - term_dig_counts_cumu[cycle_start] - z_count_cumu[-1] + z_count_cumu[cycle_start]) +\
            (term_dig_counts_cumu[i] - term_dig_counts_cumu[cycle_start] - z_count_cumu[i] + z_count_cumu[cycle_start]) +\
            (j - z_counts_cumu[j])
    k_vals_found = n_terms
    #k_vals_found2 = calculateIndexStartCounts(0, [])
    print(k_vals_found)
    for i in range(16):
        print(i, calculateIndexStartCounts(i, []))
    return -1
    res = n_terms
    prefs = []
    
    for start_idx in itertools.count(1):
        d = getDigit(start_idx - 1)
        if not d: continue
        prefs = [x + d for x in prefs]
        prefs.append(d)
        n_max2 = n_max + prefs[0]

        q, r = divmod(n_max2 - term_dig_sums_cumu[cycle_start], term_dig_sums_cumu[-1] - term_dig_sums_cumu[cycle_start])
        i = bisect.bisect_right(term_dig_sums_cumu, r + term_dig_sums_cumu[cycle_start]) - 1
        term_digs, term_digs_cumu, z_counts_cumu = getTermCounts(i)
        j = bisect.bisect_right(term_digs_cumu, term_dig_sums_cumu[i] - term_dig_sums_cumu[cycle_start]) - 1
        n_terms = term_dig_counts_cumu[cycle_start] - z_count_cumu[cycle_start] +\
                q * (term_dig_counts_cumu[-1] - term_dig_counts_cumu[cycle_start] - z_count_cumu[-1] + z_count_cumu[cycle_start]) +\
                (term_dig_counts_cumu[i] - term_dig_counts_cumu[cycle_start] - z_count_cumu[i] + z_count_cumu[cycle_start]) +\
                (j - z_counts_cumu[j])

        k_vals_found += n_terms
        res += n_terms * (start_idx + 1)
        if k_vals_found == n_max: break
    
    #print(n1)
    print(k_vals_found)
    return -1
    """

# Problem 239
def partialDerangementCount(n_tot: int, n_subset: int, n_subset_deranged: int) -> int:

    if n_subset > n_tot or n_subset_deranged > n_subset: return 0
    neg = False
    n_subset_fixed = n_subset - n_subset_deranged
    res = 0
    for i in range(n_subset_fixed, n_subset + 1):
        #print(i)
        term = math.comb(n_subset_deranged, n_subset - i) * math.factorial(n_tot - i)
        res += -term if neg else term
        neg = not neg
    #print(res)
    return res * math.comb(n_subset, n_subset_deranged)

def partialDerangementProbability(n_tot: int, n_subset: int, n_subset_deranged: int) -> CustomFraction:
    return CustomFraction(partialDerangementCount(n_tot, n_subset, n_subset_deranged), math.factorial(n_tot))

def partialPrimeDerangementProbabilityFraction(n_max: int, n_primes_deranged: int) -> CustomFraction:
    ps = SimplePrimeSieve(n_max)
    n_p = bisect.bisect_right(ps.p_lst, n_max)
    print(n_p)
    res = partialDerangementProbability(n_max, n_p, n_primes_deranged)
    return res

def partialPrimeDerangementProbabilityFloat(n_max: int=100, n_primes_deranged: int=22) -> float:
    res = partialPrimeDerangementProbabilityFraction(n_max, n_primes_deranged)
    print(res)
    return res.numerator / res.denominator

# Problem 240
def topDiceSumCombinations(n_sides: int=12, n_dice: int=20, n_top_dice: int=10, top_sum: int=70) -> int:

    def diceSumEqualsTargetCount(target_score: int, n_sides: int, n_dice: int) -> int:
        if target_score < n_dice: return 0
        #target_score -= n_dice
        memo = {}
        def recur(val: int, remain_score: int, remain_dice: int, cnt: int) -> int:
            #print(val, remain_score, remain_dice, cnt)
            if not remain_score:
                return cnt // math.factorial(remain_dice)
            elif remain_score > remain_dice * val: return 0
            args = (val, remain_score, remain_dice, cnt)
            if args in memo.keys(): return memo[args]

            res = 0
            #print(args)
            #if val == 4 and remain_score == 12:
            #    print(remain_dice, min(remain_dice, remain_score // val))
            for n_val in range(min(remain_dice, remain_score // val) + 1):
                #print(remain_score, n_val, val)
                ans = recur(val - 1, remain_score - n_val * val, remain_dice - n_val, cnt // math.factorial(n_val))
                res += ans

            memo[args] = res
            return res

        res = recur(n_sides - 1, target_score - n_dice, n_dice, math.factorial(n_dice))
        #print(memo)
        return res

    def diceSumCombinationsMinimumDieScoreGenerator(target_score: int, n_dice: int) -> Generator[Tuple[int, int, int], None, None]:
        # yields: min die score, number of that min die score, frequency
        #target_score -= n_dice
        mx_mn_die = target_score // n_dice
        for min_die_val in range(1, mx_mn_die + 1):
            target_score2 = target_score - min_die_val * n_dice
            n_sides2 = n_sides - min_die_val
            for n_min_dice in range(1, n_dice + 1):
                #print(n_dice, n_min_dice, n_sides2, target_score2)
                if (n_dice - n_min_dice) * n_sides2 < target_score2: break
                #print(min_die_val, n_min_dice)
                cnt0 = diceSumEqualsTargetCount(target_score2, n_sides2, n_dice - n_min_dice)
                if not cnt0: continue
                cnt = math.comb(n_dice, n_min_dice) * cnt0
                yield (min_die_val, n_min_dice, cnt)
        return
    
    res = 0
    for (min_top_die_score, n_min_dice, f) in diceSumCombinationsMinimumDieScoreGenerator(top_sum, n_top_dice):
        #print(min_top_die_score, n_min_dice, f)
        for n_extra_min_dice in range(n_dice - n_top_dice + 1):
            n_lt_min_top = n_dice - (n_top_dice + n_extra_min_dice)
            cnt = math.comb(n_dice, n_top_dice + n_extra_min_dice) * (min_top_die_score - 1) ** n_lt_min_top *\
                    f * math.factorial(n_top_dice + n_extra_min_dice) * math.factorial(n_min_dice) // (math.factorial(n_top_dice) * math.factorial(n_min_dice + n_extra_min_dice))
            res += cnt
    return res

# Problem 241
def divisorFunction(num: int) -> int:
    pf = calculatePrimeFactorisation(num)
    res = 1
    for p, f in pf.items():
        res *= (p ** (f + 1) - 1) // (p - 1)
    return res

def halfIntegerPerfectionQuotients(n_max: int=10 ** 18) -> List[int]:

    # Review- try to make faster
    # Review- try to implement without arbitrarily setting a maximum
    # prime factor
    if n_max < 2: return []

    p_max = 400
    ps = SimplePrimeSieve()
    p_gen = iter(ps.endlessPrimeGenerator())
    """
    p_lst = []
    p_dict = {}
    p_pows_sigma = []

    
    def addNextPrime() -> None:
        p = next(p_gen)
        p_dict[p] = len(p_lst)
        p_lst.append(p)
        num = 1
        exp = 0
        p_pows_sigma.append([])
        while True:
            exp += 1
            num *= p
            if num > n_max: break
            sigma = (p ** (exp + 1) - 1) // (p - 1)
            sigma_pf = calculatePrimeFactorisation(sigma)
            p_pows_sigma[-1].append((num, exp, sigma, sigma_pf))
        return
    """
    memo = {}
    def primePowerSigmaPrimeFactorisation(p: int, exp: int) -> Dict[int, int]:
        args = (p, exp)
        if args in memo.keys(): return memo[args]
        sigma = (p ** (exp + 1) - 1) // (p - 1)
        res = calculatePrimeFactorisation(sigma)
        memo[args] = res
        return res
    
    p_lst = []
    p_dict = {}
    def getPrimeAtIndex(idx: int) -> int:
        while len(p_lst) <= idx:
            p = next(p_gen)
            p_dict[p] = len(p_lst)
            p_lst.append(p)
        return p_lst[idx]
    
    def getPrimeIndex(p: int) -> int:
        while not p_lst or p_lst[-1] < p:
            p2 = next(p_gen)
            p_dict[p2] = len(p_lst)
            p_lst.append(p2)
        if p not in p_dict.keys():
            raise ValueError(f"{p} is not prime")
        return p_dict[p]
        
    def search(n_max: int, numerator: int) -> List[int]:
        res = []
        numer_pf = calculatePrimeFactorisation(numerator)

        def recur(p_idx: int, curr: int, target_ratio: CustomFraction) -> None:
            #print(p_idx, curr, nonzero_cnt, bal)
            #if not neg_cnt and not bal[0]:
            #    res.append(curr)
            p = getPrimeAtIndex(p_idx)
            if p > p_max: return
            #curr2 = curr
            recur(p_idx + 1, curr, target_ratio)
            #nonzero_cnt2 = nonzero_cnt
            #target_ratio2 = CustomFraction(target_ratio.numerator, target_ratio.denominator)
            
            while len(bal) <= p_idx:
                bal.append(0)
            bal0 = bal[p_idx]
            start = max(1, bal[p_idx])
            bal[p_idx] -= start - 1
            curr2 = curr * p ** (start - 1)
            target_ratio2 = target_ratio / p ** (start - 1)
            for exp_p in itertools.count(start):
                curr2 *= p
                target_ratio2 /= p
                if curr2 > n_max: break
                #print(p_idx)
                #if not bal[p_idx]: nonzero_cnt2 += 1
                #elif bal[p_idx] == 1: nonzero_cnt2 -= 1
                bal[p_idx] -= 1
                #print(f"starting sigma_pf")
                sigma_pf = primePowerSigmaPrimeFactorisation(p, exp_p)
                #print(f"finishing sigma_pf")
                p2_lst = sorted(sigma_pf.keys())
                if p2_lst[-1] > p_max: continue
                #print("pre:")
                #print(p2_lst)
                delta = {}
                #nonzero_cnt_delta = 0
                target_ratio_mult = 1
                #if p2_lst[0] == 2:
                #    if bal[0] + sigma_pf[2] > 0: continue
                #    #bal[0] += sigma_pf[2]
                #    delta[0] = sigma_pf[2]
                cancel = False
                target_ratio3 = target_ratio2
                for p2_idx in range(len(p2_lst)):#range(p2_lst[0] == 2, len(p2_lst)):
                    p2 = p2_lst[p2_idx]
                    j = getPrimeIndex(p2)
                    f = sigma_pf[p2]
                    if j < p_idx and bal[j] > -f:
                        cancel = True
                        break
                    while len(bal) <= j:
                        bal.append(0)
                    #if not bal[j]: nonzero_cnt_delta += 1
                    #if bal[j] == -f: nonzero_cnt_delta -= 1
                    target_ratio3 *= p2 ** f
                    if target_ratio3 > 1:
                        cancel = True
                        break
                    delta[j] = f
                if cancel: continue
                elif target_ratio3 == 1:
                    print(curr2)
                    res.append(curr2)
                    continue
                for j, f in delta.items():
                    bal[j] += f
                #nonzero_cnt2 += nonzero_cnt_delta
                #if not nonzero_cnt2:
                #    print(curr2)
                #    res.append(curr2)
                recur(p_idx + 1, curr2, target_ratio3)#nonzero_cnt2)
                #print("post:")
                #print(p2_lst)
                for j, f in delta.items():
                    bal[j] -= f
                #nonzero_cnt2 -= nonzero_cnt_delta
                """
                if p2_lst[0] == 2:
                    bal[0] -= sigma_pf[2]
                for p2_idx in range(p2_lst[0] == 2, len(p2_lst)):
                    p2 = p2_lst[p2_idx]
                    
                    j = getPrimeIndex(p2)
                    #print(p2, j, bal)
                    f = sigma_pf[p2]
                    if bal[j] >= 0 and bal[j] < f:
                        neg_cnt2 += 1
                    bal[j] -= f
                """
            bal[p_idx] = bal0
            return

        for exp2 in itertools.count(1):
            print(f"exp2 = {exp2}")
            if (1 << exp2) > n_max: break
            #print(f"starting prime factorisation for 2 ** {exp2}")
            sigma_pf = primePowerSigmaPrimeFactorisation(2, exp2)
            #print(f"finishing prime factorisation for 2 ** {exp2}")
            if max(sigma_pf.keys()) > p_max: continue
            bal = [0] * (getPrimeIndex(max(max(sigma_pf.keys()), max(numer_pf.keys()))) + 1)
            #print("hi0")
            bal[0] = 1 - exp2
            #print("hi1")
            for p, f in numer_pf.items():
                bal[getPrimeIndex(p)] = -f
            
            #print("hi2")
            #nonzero_cnt = len(numer_pf) + bool(bal[0])
            p2_lst = sorted(sigma_pf.keys())
            if p2_lst[-1] > p_max:
                #print("prime in factorisation exceeds the max")
                continue
            #print("hi3")
            #print(sigma_pf)
            #print(nonzero_cnt)
            target_ratio = CustomFraction(1, numerator * 2 **(exp2 - 1))
            for p2 in p2_lst:
                p2_idx = getPrimeIndex(p2)
                #print(p2, p2_idx)
                #if not bal[p2_idx]:
                #    nonzero_cnt += 1
                #if bal[p2_idx] == -sigma_pf[p2]:
                #    nonzero_cnt -= 1
                bal[p2_idx] += sigma_pf[p2]
                #print(bal, nonzero_cnt)
                target_ratio *= p2 ** sigma_pf[p2]
            #print(target_ratio)
            curr = 1 << exp2
            #print("hi4")
            #print("hi")
            #print(bal)
            if target_ratio == 1:#nonzero_cnt:
                print(curr)
                res.append(curr)
                break
            elif target_ratio > 1:
                break
            #print(curr, bal)
            recur(1, curr, target_ratio)#nonzero_cnt)
        #print("hi")
        return res
    
    # Getting an upper bound on the possible numerators using Robin's inequality
    euler_mascheroni = 0.57721566490153286060651209008240243104215933593992 # From Wikipedia
    numer_ub = math.floor(2 * (math.exp(euler_mascheroni) * n_max * math.log(math.log(n_max)) + 0.6483 * n_max / math.log(math.log(n_max))) / n_max)
    print(f"numerator upper bound = {numer_ub}")
    res = [] #search(n_max)
    for numer in range(3, numer_ub + 1, 2):
        print(CustomFraction(numer, 2))
        lst = search(n_max, numer)
        print(lst)
        res += lst
    res.sort()
    print(res)
    return res
    """
    res = 0
    for num in range(2, n_max + 1, 2):
        sigma = divisorFunction(num)
        if sigma % num and not 2 * sigma % num:
            print(num, sigma, CustomFraction(sigma, num), calculatePrimeFactorisation(num))
            res += num
    return res
    """

def halfIntegerPerfectionQuotientsSum(n_max: int=10 ** 18) -> int:
    """
    Solution to Project Euler #241
    """
    res = halfIntegerPerfectionQuotients(n_max=n_max)
    return sum(res)

# Problem 242
def oddSumSubsetCount(n: int, k: int) -> int:
    n_even = n >> 1
    n_odd = n - n_even
    res = 0
    for odd_cnt in range(1, k + 1, 2):
        res += math.comb(n_odd, odd_cnt) * math.comb(n_even, k - odd_cnt)
    return res

def oddTripletsCount(n_max: int=10 ** 12) -> int:
    """
    Solution to Project Euler #142
    """

    memo = {}
    def pascalOddEntriesCount(num: int) -> int:
        if num < 0: return 0
        elif num <= 1: return num
        args = num
        if args in memo.keys(): return memo[args]
        res = 2 * pascalOddEntriesCount(num >> 1) + pascalOddEntriesCount((num >> 1) + 1) if num & 1 else 3 * pascalOddEntriesCount(num >> 1)
        memo[args] = res
        return res
    """
    res = 0
    for n in range(1, n_max + 1, 2):
        cnt = 0
        for k in range(1, n + 1, 2):
            f = oddSumSubsetCount(n, k)
            if f & 1:
                print((n, k, f))
                cnt += 1
        print(f"count for n = {n} is {cnt}")
        res += cnt
    return res
    """
    res = pascalOddEntriesCount((n_max + 3) >> 2)
    return res

# Problem 243
def smallestDenominatorWithSmallerResilience(resilience_upper_bound: CustomFraction=CustomFraction(15499, 94744)) -> int:
    """
    Solution to Project Euler #243
    """
    def testResilience(num: int, sigma: int) -> bool:
        return sigma * resilience_upper_bound.denominator < (num - 1) * resilience_upper_bound.numerator

    ps = SimplePrimeSieve()
    p_gen = iter(ps.endlessPrimeGenerator())
    ub = 1
    p_lst = []
    sigma = 1
    for p in p_gen:
        p_lst.append(p)
        ub *= p
        sigma *= p - 1
        if testResilience(ub, sigma):
            break
    print(ub)
    #print(p_lst[-1])

    res = [ub]

    def recur(p_idx: int, curr_num: int, curr_sigma: int, prev_pow: int) -> None:
        p = p_lst[p_idx]
        curr_num *= p
        if curr_num > res[0]: return
        curr_sigma *= p - 1
        if testResilience(curr_num, curr_sigma):
            res[0] = min(res[0], curr_num)
            return
        recur(p_idx + 1, curr_num, curr_sigma, 1)
        for exp in range(2, prev_pow + 1):
            curr_num *= p
            if curr_num > res[0]: break
            curr_sigma *= p
            if testResilience(curr_num, curr_sigma):
                res[0] = min(res[0], curr_num)
            recur(p_idx + 1, curr_num, curr_sigma, exp)
        return

    p_idx = 0
    p = p_lst[p_idx]
    curr_num = p
    curr_sigma = p - 1
    for exp in itertools.count(2):
        curr_num *= p
        if curr_num >= res[0]: break
        curr_sigma *= p
        if testResilience(curr_num, curr_sigma):
            res[0] = min(res[0], curr_num)
            break
        recur(p_idx + 1, curr_num, curr_sigma, exp)
    
    return res[0]

def sliderPuzzleShortestPathsChecksumValue(
    init_state: List[List[int]]=[[0, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]],
    final_state: List[List[int]]=[[0, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]],
    checksum_mult: int=243,
    checksum_md: int=10 ** 8 + 7,
) -> int:
    
    if len(init_state) != len(final_state):
        raise ValueError("init_state and final_state must have the same length")
    shape = (len(init_state), len(init_state[0]))
    cnts1 = [0, 0]
    cnts2 = [0, 0]
    z1_seen = False
    z2_seen = False
    for (row1, row2) in zip(init_state, final_state):
        if len(row1) != shape[1] or len(row2) != shape[1]:
            raise ValueError("Each row of init_state and final_state must all have "
                            "the same length.")
                            
        for l in row1:
            if l == 0:
                if z1_seen:
                    raise ValueError("There must be exactly be one 0 element in init_state.")
                z1_seen = True
                continue
            elif l not in {1, 2}:
                raise ValueError("init_state can only hold the integers 0 to 2 inclusive.")
            cnts1[l == 2] += 1
        for l in row2:
            if l == 0:
                if z2_seen:
                    raise ValueError("There must be exactly be one 0 element in final_state.")
                z2_seen = True
                continue
            elif l not in {1, 2}:
                raise ValueError("final_state can only hold the integers 0 to 2 inclusive.")
            cnts2[l == 2] += 1
    if cnts1 != cnts2:
        raise ValueError("There must be the same number of 1s and 2s in init_state and final_state.")
    if not z1_seen:
        raise ValueError("There must be exactly be one 0 element in init_state.")
    if not z2_seen:
        raise ValueError("There must be exactly be one 0 element in final_state.")

    def gridIndex2FlatIndex(i1: int, i2: int) -> int:
        return i1 * shape[1] + i2
    
    def flatIndex2GridIndex(idx: int) -> Tuple[int, int]:
        return divmod(idx, shape[1])

    def encodeState(state: List[List[int]]) -> Tuple[int, int]:
        z_pos = None
        bm = 0
        for i1 in reversed(range(shape[0])):
            row = state[i1]
            for i2 in reversed(range(shape[1])):
                l = row[i2]
                bm <<= 1
                if not l:
                    z_pos = (i1, i2)
                    continue
                bm |= int(l == 2)
        return (gridIndex2FlatIndex(*z_pos), bm)
    
    directs = ["D", "U", "R", "L"]
    directs_dict = {l: i for i, l in enumerate(directs)}
    def encodeDirection(direct: str) -> int:
        return directs_dict.get(l, -1)

    directs_ascii = [ord(x) for x in directs]
    #print(directs_ascii)
    def getEncodedDirectionAscii(direct_idx: int) -> int:
        return directs_ascii[direct_idx]
    
    comps = [
        lambda i1, i2: i1 > 0,
        lambda i1, i2: i1 < shape[0] - 1,
        lambda i1, i2: i2 > 0,
        lambda i1, i2: i2 < shape[1] - 1
    ]
    
    incrs = [-shape[1], shape[1], -1, 1]

    def move(state: Tuple[int, int]) -> Generator[Tuple[str, Tuple[int, int]], None, None]:
        i1, i2 = flatIndex2GridIndex(state[0])
        bm = state[1]
        
        for direct_idx, (comp, incr) in enumerate(zip(comps, incrs)):
            if not comp(i1, i2): continue
            state2 = list(state)
            state2[0] += incr
            if bm & (1 << state2[0]):
                state2[1] ^= (1 << state2[0])
                state2[1] |= (1 << state[0])
            yield (direct_idx, tuple(state2))
        return

    start = encodeState(final_state)
    end = encodeState(init_state)
    #print("hi1")
    #print(start, end)
    if start == end: return 0
    #print("hi2")
    Trie = lambda: defaultdict(Trie)

    curr_tries = {start: Trie()}
    curr_tries[start]["end"] = True
    #curr_paths = {start: [1, []]}
    seen = {start}
    steps = 0
    while curr_tries:
        steps += 1
        prev_tries = curr_tries
        curr_tries = {}
        for state, trie in prev_tries.items():
            for direct_idx, state2 in move(state):
                direct_idx2 = direct_idx ^ 1
                if state2 not in curr_tries.keys():
                    if state2 in seen: continue
                    curr_tries[state2] = Trie()
                    #curr_paths[state2] = [n_paths, [list(x) for x in paths]]
                    #curr_paths[state2][1].append([0] * 4)
                curr_tries[state2][direct_idx2] = trie
                #else:
                #    #print(f"repeating for state {state2}")
                #    curr_paths[state2][0] += n_paths
                #    for f_lst1, f_lst2 in zip(curr_paths[state2][1], paths):
                #        for j in range(len(f_lst2)):
                #            f_lst1[j] += f_lst2[j]
                #print(curr_paths, direct_idx)
                #curr_paths[state2][1][-1][direct_idx] += n_paths
                seen.add(state2)
        #print(curr_paths)
        #if steps >= 5:
        #    break
        if end in curr_tries.keys(): break
    else:
        #print(seen)
        print(steps)
        return -1
    
    #trie_depth = [-1]
    def recur(trie: Any, curr: int=0) -> int:
        if "end" in trie.keys(): return curr
        res = 0
        curr = (curr * checksum_mult) % checksum_md
        for direct_idx, t in trie.items():
            res += recur(t, curr=(curr + directs_ascii[direct_idx]) % checksum_md)
        return res

    res = recur(curr_tries[end], curr=0)
    #print(trie_depth[0])
    #print(curr_tries[end])
    return res
    #print(curr_paths)
    #print(len(curr_paths))
    #paths = curr_paths[end]
    #print(paths)
    #res = 0
    #for f_lst in paths[1]:
    #    tot = sum(x * y for x, y in zip(directs_ascii, f_lst))
    #    res = (res * checksum_mult + tot) % checksum_md
    #return res

# Problem 245
def compositeCoresilienceAReciprocalSum(n_max: int=2 * 10 ** 11) -> int:

    ps = SimplePrimeSieve()
    def primeCheck(num: int) -> bool:
        res = ps.millerRabinPrimalityTestWithKnownBounds(num, max_n_additional_trials_if_above_max=10)
        return res[0]

    def eulerTotientFunction(pf: Dict[int, int]) -> int:
        res = 1
        for p, f in pf.items():
            res *= p - 1
            res *= p ** (f - 1)
        return res
    
    #p_mx = isqrt(n_max)
    #p_mx0 = isqrt(n_max)
    res = 0
    #found = []
    for p in ps.endlessPrimeGenerator():
        num = p ** 2 - p + 1
        #if num > n_max: break
        fact_mn = 2 * p - 1
        fact_mx = n_max // p + p + 1
        if fact_mx < fact_mn: break
        for fact in calculateFactorsInRange(num, fact_min=fact_mn, fact_max=fact_mx):
            q = fact - p + 1
            if p == q or not primeCheck(q): continue
            #print(p, q, p * q)
            res += p * q
            #found.append(p * q)
    #return res
    print(res)
    ref = 257 * 359
    def recur(num: int, phi: int, n_remain_p: int, prev_p_idx: int=None) -> int:
        res = 0
        prev_p = ps.p_lst[prev_p_idx] if prev_p_idx is not None else 2
        if n_remain_p == 1:
            #d = num * phi - phi + num
            #mx = n_max + phi - (phi * n_max - 1) // num + 1#(n_max - phi) // (num - phi)
            
            #mult_mn = (prev_p + phi) // (num - phi) + 1
            p_mx3 = n_max // num
            if p_mx3 > ps.p_lst[-1]:
                fact_mn = (num - phi) * ((2 if prev_p is None else prev_p) + 1) + phi
                fact_mx = ((num - phi) * n_max) // num + phi
                for fact in calculateFactorsInRange(num * phi - phi + num, fact_min=fact_mn, fact_max=fact_mx):
                    p, r = divmod(fact - phi, num - phi)
                    #if num == ref:
                    #    print(num, fact, p, r)
                    if r or not primeCheck(p): continue
                    ans = num * p
                    phi2 = phi * (p - 1)
                    if (ans - 1) % (ans - phi2): continue
                    print(ans, phi2, ans - phi2, ans - 1)
                    #print(ans, num, phi, p, phi2, ans - phi2, ans - 1, (num - phi) * p + phi)
                    res += ans
                return res
            p_idx_ub = bisect.bisect_right(ps.p_lst, p_mx3)
            prod = num * phi - phi + num
            #if num == ref:
            #    print(num, phi, p_mx3, prod)
            for p_idx in range(prev_p_idx, p_idx_ub):
                p = ps.p_lst[p_idx]
                fact = (num - phi) * p + phi
                if num == ref and p == 1057477:
                    ans = num * p
                    phi2 = phi * (p - 1)
                    print("hi", num, p, prod % fact, ans, phi2, (ans - 1) % (ans - phi2))
                if (prod % fact): continue
                ans = num * p
                phi2 = phi * (p - 1)
                if (ans - 1) % (ans - phi2): continue
                print(ans, phi2, ans - phi2, ans - 1)
                res += ans
                #found.append(ans)
            return res
            """
            fact_mn = (num - phi) * ((2 if prev_p is None else prev_p) + 1) + phi
            fact_mx = ((num - phi) * n_max) // num + phi
            for fact in calculateFactorsInRange(num * phi - phi + num, fact_min=fact_mn, fact_max=fact_mx):
                p, r = divmod(fact - phi, num - phi)
                #if num == ref:
                #    print(num, fact, p, r)
                if r or not primeCheck(p): continue
                ans = num * p
                phi2 = phi * (p - 1)
                if (ans - 1) % (ans - phi2): continue
                print(ans, phi2, ans - phi2, ans - 1)
                #print(ans, num, phi, p, phi2, ans - phi2, ans - 1, (num - phi) * p + phi)
                res += ans
            return res
            """
            """
            for mult in range(d, mx + 1, d):
                
                p, r = divmod(mult - phi, num - phi)
                if num == ref:
                    print(num, mult, p, r)
                if r or not primeCheck(p): continue
                ans = num * p
                phi2 = phi * (p - 1)
                if (ans - 1) % (ans - phi2): continue
                print(ans, phi2, ans - phi2, ans - 1)
                #print(ans, num, phi, p, phi2, ans - phi2, ans - 1, (num - phi) * p + phi)
                res += ans
            return res
            """
        #j0 = 1 if prev_p is None else bisect.bisect_right(ps.p_lst, prev_p)
        j0 = prev_p_idx + 1 if prev_p_idx is not None else 1
        p_mx2 = integerNthRoot(n_max // num, n_remain_p)
        res2 = 0
        #if n_remain_p == 4: print(f"num = {num}, n_max = {n_max}, p_mx = {p_mx2}")
        for j in range(j0, len(ps.p_lst)):
            p = ps.p_lst[j]
            if p > p_mx2: break
            if prev_p_idx is None:
                since1 = time.time()
            ans = recur(num * p, phi * (p - 1), n_remain_p - 1, prev_p_idx=j)
            res += ans
            res2 += ans
            if prev_p_idx is None:
                print(f"n_p = {n_remain_p}, p1 = {p}, p_max = {p_mx2}, total for this p1 = {ans}, cumulative total = {res}, time taken for this p1 = {time.time() - since1} seconds")
        return res

    #print("hello")
    res2 = 0
    for n_p in range(3, math.floor(math.log(n_max, 2)) + 1):
        print(f"n_p = {n_p}")
        ans = recur(1, 1, n_p, prev_p_idx=None)
        res2 += ans
        res += ans
        print(ans, res2, res)
    print(res2)
    #print(sorted(found))
    return res
    """

    res = 0
    coresilience_dict = {}
    for num in range(4, 5 * 10 ** 6 + 1):
        if primeCheck(num): continue
        pf = calculatePrimeFactorisation(num)
        phi = eulerTotientFunction(pf)
        coresilience = CustomFraction(num - phi, num - 1)
        #print(num, coresilience)
        if coresilience.numerator != 1: continue
        #print(num, pf, phi, coresilience)
        denom = coresilience.denominator
        coresilience_dict.setdefault(denom, [])
        coresilience_dict[denom].append(pf)
        if len(pf) > 2:
            print(num, denom, pf)
            p_lst = sorted(pf.keys())
            pairs = []
            for i2 in range(1, len(p_lst)):
                p2 = p_lst[i2]
                for i1 in range(i2):
                    p1 = p_lst[i1]
                    num2 = p1 * p2
                    pf2 = {p1: 1, p2: 1}
                    phi2 = eulerTotientFunction(pf2)
                    coresilience2 = CustomFraction(num2 - phi2, num2 - 1)
                    if coresilience2.numerator != 1: continue
                    pairs.append((p1, p2))
            print(f"sub-pairs found: {pairs}")
        #res += num
    """
    #for denom in sorted(coresilience_dict.keys()):
    #    print(denom, coresilience_dict[denom])
    return res
    
# Problem 246
def latticePointsAtTangentsToEllipseLatticePoints(upper_bound_angle_deg: float=45, m_x: int=-2000, g_x: int=8000, circle_radius: int=15000, incl_upper_bound_angle: bool=False, incl_ellipse_border: bool=False) -> int:
    """
    Solution to Project Euler #246
    """
    c = CustomFraction(abs(g_x - m_x), 2)
    print(c)
    c_sq = c * c
    a = CustomFraction(circle_radius, 2)
    a_sq = a * a
    b_sq = a_sq - c_sq
    print(a_sq, b_sq)
    sq_ratio = a_sq / b_sq
    print(sq_ratio.numerator / sq_ratio.denominator)
    print(f"ellipse area = {math.pi * math.sqrt(a_sq.numerator / a_sq.denominator) * math.sqrt(b_sq.numerator / b_sq.denominator)}")
    x_hlf = (c.denominator == 2)

    t = math.tan(math.radians(upper_bound_angle_deg))
    t_sq = t ** 2
    print(f"t_sq = {t_sq}")

    y_max = math.floor(math.sqrt(((a_sq + b_sq) * t_sq + 2 * a_sq * (1 + math.sqrt(t_sq + 1))) / t_sq))

    y_max_ellipse = math.floor(math.sqrt(b_sq.numerator / b_sq.denominator))
    
    def getLatticePointCountInsideEllipseAtY(y: int) -> int:
        y_sq = y ** 2
        frac = a_sq * (1 - y_sq / b_sq)
        if frac < 0: return 0
        if x_hlf:
            if frac.denominator == 4:
                rt = CustomFraction(isqrt(frac.numerator), 2)
                x_max = rt - (rt * rt == frac and incl_ellipse_border)
            else:
                x_max_numer = math.floor(4 * frac.numerator / frac.denominator) >> 1
                x_max_numer -= 1 - (x_max_numer & 1)
                x_max = CustomFraction(x_max_numer, 2)
            return x_max.numerator + 1
        if frac.denominator == 1:
            rt = isqrt(frac.numerator)
            x_max = rt - (rt * rt == frac and incl_ellipse_border)
        else:
            x_max = isqrt(frac.numerator // frac.denominator)
        return x_max * 2 + 1
        

    def getLatticePointCountInsideLocusAtY(y: int) -> int:
        y_sq = y ** 2
        sm = a_sq  + b_sq - y_sq
        c2 = t_sq
        c1 = -2 * (sm * t_sq + 2 * b_sq)
        c0 = t_sq * sm * sm + 4 * a_sq * (b_sq - y_sq)
        x_max = math.sqrt((-c1 + math.sqrt(c1 * c1 - 4 * c2 * c0)) / (2 * c2))
        if x_hlf:
            x_max *= 2
            x_max_int = math.floor(x_max)
            x_max_int -= 1 - (x_max_int & 1)
            return x_max + 1 - 2 * ((x_max_int == x_max) and not incl_upper_bound_angle)
        x_max_int = math.floor(x_max)
        return x_max_int * 2 + 1 - 2 * ((x_max_int == x_max) and not incl_upper_bound_angle)

    res = 0#
    locus_cnt = 0
    ellipse_cnt = 0
    for y in range(1, y_max + 1):
        locus_cnt += getLatticePointCountInsideLocusAtY(y)
    for y in range(1, y_max_ellipse + 1):
        ellipse_cnt += getLatticePointCountInsideEllipseAtY(y)
    locus_cnt <<= 1
    ellipse_cnt <<= 1
    locus_cnt += getLatticePointCountInsideLocusAtY(0)
    ellipse_cnt += getLatticePointCountInsideEllipseAtY(0)
    print(f"lattice point count in locus= {locus_cnt}")
    print(f"lattice point count in ellipse = {ellipse_cnt}")
    return locus_cnt - ellipse_cnt

# Problem 247
def squaresUnderHyperbola(target_index: Tuple[int, int]=(3, 3)) -> int:
    """
    Solution to Project Euler #247
    """
    def calculateLength(pos: Tuple[float]) -> float:
        d = pos[0] - pos[1]
        return .5 * (math.sqrt(d ** 2 + 4) - (pos[0] + pos[1]))

    pos0 = (1, 0)
    h = [(-calculateLength(pos0), pos0, (0, 0))]
    target_cnt = math.comb(sum(target_index), target_index[0])
    cnt = 0
    for num in itertools.count(1):
        neg_length, pos, index = heapq.heappop(h)
        length = -neg_length
        #print(num, length, pos, index)
        if index == target_index:
            cnt += 1
            print(f"count = {cnt} of {target_cnt}")
            if cnt == target_cnt: break
        for pos2, index_incr in (((pos[0] + length, pos[1]), (1, 0)), ((pos[0], pos[1] + length), (0, 1))):
            d = pos2[0] - pos2[1]
            length2 = calculateLength(pos2)
            index2 = tuple(x + y for x, y in zip(index, index_incr))
            heapq.heappush(h, (-length2, pos2, index2))
    return num


# Problem 248
def factorialPrimeFactorisation(n: int) -> Dict[int, int]:
    pf = {}
    for num in range(2, n + 1):
        pf2 = calculatePrimeFactorisation(num)
        for p, f in pf2.items():
            pf[p] = pf.get(p, 0) + f
    return pf

def numbersWithEulerTotientFunctionNFactorial(n: int) -> List[int]:
    

    ps = SimplePrimeSieve()
    def primeCheck(num: int) -> bool:
        res = ps.millerRabinPrimalityTestWithKnownBounds(num, max_n_additional_trials_if_above_max=10)
        return res[0]
    
    #for num in range(1, 30):
    #    print(num, primeCheck(num))

    factorial_pf = factorialPrimeFactorisation(n)
    #print(factorial_pf)

    p_lst = sorted(factorial_pf.keys())
    p_dict = {p: i for i, p in enumerate(p_lst)}
    n_p = len(p_lst)
    f_lst = [factorial_pf[p] for p in p_lst]
    z_cnt = [0]

    #print(p_lst, f_lst)

    def recur(idx: int=0, tot: int=1, num: int=1, prev: int=1) -> Generator[int, None, None]:
        #print(idx, num, prev, curr)
        if idx == n_p:
            #print("hi2")
            
            if num <= prev: return
            #if num == 2: print("hello")
            #print(idx, num, prev, f_lst, curr, z_cnt[0])
            p1 = num + 1
            
            #if num == 66530: print("here")
            if not primeCheck(p1):
                return
            #if num > 5 * 10 ** 4:
            #    print(num, p1)
            #if num == 66528: print("here")
            tot2 = tot * p1
            #print(num, p1)
            if z_cnt[0] == n_p:
                yield tot2
                yield tot2 << 1
            else:
                if z_cnt[0] == n_p - 1 and f_lst[0]:
                    yield tot2 << (f_lst[0] + 1)
                yield from recur(idx=0, tot=tot2, num=1, prev=num)
            p1_f = f_lst[p_dict[p1]] if p1 in p_dict.keys() else 0
            if not p1_f: return
            
            p1_idx = p_dict[p1]
            for exp in range(1, p1_f):
                f_lst[p1_idx] -= 1
                tot2 *= p1
                yield from recur(idx=0, tot=tot2, num=1, prev=num)
            f_lst[p1_idx] = 0
            tot2 *= p1
            z_cnt[0] += 1
            if z_cnt[0] == n_p:
                yield tot2
                yield tot2 << 1
            else:
                if z_cnt[0] == n_p - 1 and f_lst[0]:
                    yield tot2 << (f_lst[0] + 1)
                yield from recur(idx=0, tot=tot2, num=1, prev=num)
            f_lst[p1_idx] = p1_f
            z_cnt[0] -= 1
            return
        f0 = f_lst[idx]
        num0 = num
        #print(f"idx = {idx}, f0 = {f0}, f_lst = {f_lst}, z_cnt = {z_cnt}, curr = {curr}")
        if not f0:
            yield from recur(idx=idx + 1, tot=tot, num=num, prev=prev)
            return
        for i in range(f0):
            #print(f"i = {i}")
            yield from recur(idx=idx + 1, tot=tot, num=num, prev=prev)
            #print(f"i = {i}, idx = {idx}, f_lst[idx] = {f_lst[idx]}")
            num *= p_lst[idx]
            f_lst[idx] -= 1
        #print("finished loop")
        z_cnt[0] += 1
        #print(idx, num, prev)
        yield from recur(idx=idx + 1, tot=tot, num=num, prev=prev)
        num = num0
        z_cnt[0] -= 1
        f_lst[idx] = f0
        return
    
    res = list(recur())
    res.sort()
    #for num in recur():
        #tot += 1
        #if num == 6227180929: print("found")
        #if not  tot % 100: print(tot, num)

    return res

def mthSmallestNumbersWithEulerTotientFunctionNFactorial(n: int=13, m: int=15 * 10 ** 4) -> int:
    """
    Solution to Project Euler #248
    """
    nums = numbersWithEulerTotientFunctionNFactorial(n)
    return nums[m - 1] if len(nums) >= m else -1

# Problem 249
def primeSumsetSums(p_max: int=4999, md: Optional[int]=10 ** 16) -> int:
    """
    Solution to Project Euler #249
    """
    ps = SimplePrimeSieve(n_max=p_max)
    def primeCheck(num: int) -> bool:
        res = ps.millerRabinPrimalityTestWithKnownBounds(num, max_n_additional_trials_if_above_max=10)
        return res[0]

    curr = {0: 1}
    for p in ps.endlessPrimeGenerator():
        if p > p_max: break
        print(p, len(curr))
        for num in reversed(sorted(curr.keys())):
            curr[num + p] = curr.get(num + p, 0) + curr[num]
            if md is not None: curr[num + p] %= md
    res = 0
    for num, f in curr.items():
        if primeCheck(num):
            res += f
            if md is not None: res %= md
    return res

# Problem 250
def numberToItsOwnPowerSubsetDivisibleByNumberCount(n_max: int=250250, div: int=250, md: Optional[int]=10 ** 16) -> int:
    """
    Solution to Project Euler #250
    """
    curr = [0] * div
    curr[0] = 1
    for num in range(1, n_max + 1):
        #if not num % 1000: print(num)
        num2 = pow(num, num, div)
        for i, f in enumerate(list(curr)):
            if not f: continue
            curr[(i + num2) % div] += f
        if md is None: continue
        for i, f in enumerate(curr):
            curr[i] = f % md
    res = curr[0] - 1
    if md is not None: res %= md
    return res

##############
project_euler_num_range = (201, 250)

def evaluateProjectEulerSolutions201to250(eval_nums: Optional[Set[int]]=None) -> None:
    if not eval_nums:
        eval_nums = set(range(project_euler_num_range[0], project_euler_num_range[1] + 1))

    if 201 in eval_nums:
        since = time.time()
        res = subsetsOfSquaresWithUniqueSumTotal(n_max=100, k=50)
        #res = subsetsWithUniqueSumTotal({1, 3, 6, 8, 10, 11}, 3)
        print(f"Solution to Project Euler #201 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 202 in eval_nums:
        since = time.time()
        res = equilateralTriangleReflectionCountNumberOfWays(n_reflect=12017639147)
        print(f"Solution to Project Euler #202 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 203 in eval_nums:
        since = time.time()
        res = distinctSquareFreeBinomialCoefficientSum(n_max=51)
        print(f"Solution to Project Euler #203 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 204 in eval_nums:
        since = time.time()
        res = generalisedHammingNumberCount(typ=100, n_max=10 ** 9)
        print(f"Solution to Project Euler #204 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 205 in eval_nums:
        since = time.time()
        res = probabilityDieOneSumWinsFloat(die1_face_values=(1, 2, 3, 4), n_die1=9, die2_face_values=(1, 2, 3, 4, 5, 6), n_die2=6)
        print(f"Solution to Project Euler #205 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 206 in eval_nums:
        since = time.time()
        res = concealedSquare(pattern=[1, None, 2, None, 3, None, 4, None, 5, None, 6, None, 7, None, 8, None, 9, None, 0], base=10)
        print(f"Solution to Project Euler #206 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 207 in eval_nums:
        since = time.time()
        res = findSmallestPartitionBelowGivenProportion(proportion=CustomFraction(1, 12345))
        print(f"Solution to Project Euler #207 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 208 in eval_nums:
        since = time.time()
        res = robotWalks(reciprocal=5, n_steps=70)
        print(f"Solution to Project Euler #208 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 209 in eval_nums:
        since = time.time()
        res = countZeroMappings(n_inputs=6)
        print(f"Solution to Project Euler #209 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 210 in eval_nums:
        since = time.time()
        res = countObtuseTriangles(r=10 ** 9, div=4)
        print(f"Solution to Project Euler #210 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 211 in eval_nums:
        since = time.time()
        res = divisorSquareSumIsSquareTotal(n_max=64 * 10 ** 6 - 1)
        print(f"Solution to Project Euler #211 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 212 in eval_nums:
        since = time.time()
        res = laggedFibonacciCuboidUnionVolume(
            n_cuboids=50000,
            cuboid_smallest_coord_ranges=((0, 9999), (0, 9999), (0, 9999)),
            cuboid_dim_ranges=((1, 399), (1, 399), (1, 399)),
            l_fib_modulus=10 ** 6,
            l_fib_poly_coeffs=(100003, -200003, 0, 300007),
            l_fib_lags=(24, 55),
        )
        print(f"Solution to Project Euler #212 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 213 in eval_nums:
        since = time.time()
        res = fleaCircusExpectedNumberOfUnoccupiedSquaresFloatDirect(dims=(30, 30), n_steps=50)
        print(f"Solution to Project Euler #213 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 214 in eval_nums:
        since = time.time()
        res = primesOfTotientChainLengthSum(p_max=4 * 10 ** 7 - 1, chain_len=25)
        print(f"Solution to Project Euler #214 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 215 in eval_nums:
        since = time.time()
        res = crackFreeWalls(n_rows=10, n_cols=32)
        print(f"Solution to Project Euler #215 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 216 in eval_nums:
        since = time.time()
        res = countPrimesOneLessThanTwiceASquare(n_max=5 * 10 ** 7)
        print(f"Solution to Project Euler #216 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 217 in eval_nums:
        since = time.time()
        res = balancedNumberCount(max_n_dig=47, base=10, md=3 ** 15)
        print(f"Solution to Project Euler #217 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 218 in eval_nums:
        since = time.time() 
        res = nonSuperPerfectPerfectRightAngledTriangleCount(max_hypotenuse=10 ** 16)
        print(f"Solution to Project Euler #218 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 219 in eval_nums:
        since = time.time() 
        res = prefixFreeCodeMinimumTotalSkewCost(n_words=10 ** 9, cost1=1, cost2=4)
        print(f"Solution to Project Euler #219 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 220 in eval_nums:
        since = time.time() 
        res = heighwayDragon(
            order=50,
            n_steps=10 ** 12,
            init_pos=(0, 0),
            init_direct=(0, 1),
            initial_str="Fa",
            recursive_strs={"a": "aRbFR", "b": "LFaLb"},
        )
        print(f"Solution to Project Euler #220 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 221 in eval_nums:
        since = time.time() 
        res = nthAlexandrianInteger(n=15 * 10 ** 4)
        print(f"Solution to Project Euler #221 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 222 in eval_nums:
        since = time.time() 
        res = shortestSpherePackingInTube(tube_radius=50, radii=list(range(30, 51)))
        print(f"Solution to Project Euler #222 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 223 in eval_nums:
        since = time.time() 
        res = countBarelyAcuteIntegerSidedTrianglesUpToMaxPerimeter(max_perimeter=25 * 10 ** 6)
        print(f"Solution to Project Euler #223 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 224 in eval_nums:
        since = time.time() 
        res = countBarelyObtuseIntegerSidedTrianglesUpToMaxPerimeter(max_perimeter=75 * 10 ** 6)
        print(f"Solution to Project Euler #224 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 225 in eval_nums:
        since = time.time() 
        res = nthSmallestTribonacciOddNonDivisors(odd_non_divisor_number=124, init_terms=(1, 1, 1)) 
        print(f"Solution to Project Euler #225 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 226 in eval_nums:
        since = time.time() 
        res = blacmangeCircleIntersectionArea(eps=10 ** -9)
        print(f"Solution to Project Euler #226 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 227 in eval_nums:
        since = time.time() 
        res = chaseGameExpectedNumberOfTurns(die_n_faces=6, n_opts_left=1, n_opts_right=1, n_players=100, separation_init=50)
        print(f"Solution to Project Euler #227 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 228 in eval_nums:
        since = time.time() 
        res = regularPolygonMinkowskiSumSideCount(vertex_counts=list(range(1864, 1910)))
        print(f"Solution to Project Euler #228 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 229 in eval_nums:
        since = time.time() 
        #res = fourRepresentationsUsingSquaresCount(mults=(1, 2, 3, 7), num_max=2 * 10 ** 6)
        res = fourSquaresRepresentationCountSpecialised(num_max=2 * 10 ** 9) 
        print(f"Solution to Project Euler #229 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 230 in eval_nums:
        since = time.time() 
        res = fibonacciWordsSum(
            A=1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679,
            B=8214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196,
            poly_coeffs=(127, 19),
            exp_base=7,
            n_max=17,
            base=10
        )
        print(f"Solution to Project Euler #230 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 231 in eval_nums:
        since = time.time() 
        res = binomialCoefficientPrimeFactorisationSum(n=20 * 10 ** 6, k=15 * 10 ** 6)
        print(f"Solution to Project Euler #231 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 232 in eval_nums:
        since = time.time() 
        res = probabilityPlayer2WinsFloat(points_required=100)
        print(f"Solution to Project Euler #232 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 233 in eval_nums:
        since = time.time() 
        res = circleInscribedSquareSideLengthWithLatticePointCount(n_lattice_points=420, max_inscribed_square_side_length=10 ** 11)
        print(f"Solution to Project Euler #233 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 234 in eval_nums:
        since = time.time()
        res = semiDivisibleNumberCount(n_max=999966663333)
        print(f"Solution to Project Euler #234 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 235 in eval_nums:
        since = time.time() 
        res = arithmeticGeometricSeries(a=900, b=-3, n=5000, val=-6 * 10 ** 11, eps=10 ** -13)
        print(f"Solution to Project Euler #235 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 236 in eval_nums:
        since = time.time() 
        res = largestLuxuryHamperPossibleMValue(
            pairs=[(5248, 640), (1312, 1888), (2624, 3776), (5760, 3776), (3936, 5664)],
        )
        print(f"Solution to Project Euler #236 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 237 in eval_nums:
        since = time.time() 
        res = playingBoardTourCount(n_rows=4, n_cols=10 ** 12, start_row=0, end_row=3, md=10 ** 8)
        print(f"Solution to Project Euler #237 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 238 in eval_nums:
        since = time.time() 
        res = infiniteStringTourDigitSumStartSum(n_max=2 * 10 ** 15, s_0=14025256, s_mod=20300713, base=10)
        print(f"Solution to Project Euler #238 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 239 in eval_nums:
        since = time.time() 
        res = partialPrimeDerangementProbabilityFloat(n_max=100, n_primes_deranged=22)
        print(f"Solution to Project Euler #239 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 240 in eval_nums:
        since = time.time() 
        res = topDiceSumCombinations(n_sides=12, n_dice=20, n_top_dice=10, top_sum=70)
        print(f"Solution to Project Euler #240 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 241 in eval_nums:
        since = time.time() 
        res = halfIntegerPerfectionQuotientsSum(n_max=10 ** 18)
        print(f"Solution to Project Euler #241 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 242 in eval_nums:
        since = time.time() 
        res = oddTripletsCount(n_max=10 ** 12)
        print(f"Solution to Project Euler #242 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 243 in eval_nums:
        since = time.time() 
        res = smallestDenominatorWithSmallerResilience(resilience_upper_bound=CustomFraction(15499, 94744))
        print(f"Solution to Project Euler #243 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 244 in eval_nums:
        since = time.time() 
        res = sliderPuzzleShortestPathsChecksumValue(
            init_state=[[0, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]],
            final_state=[[0, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]],
            checksum_mult=243,
            checksum_md=10 ** 8 + 7,
        )
        print(f"Solution to Project Euler #244 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 245 in eval_nums:
        since = time.time() 
        res = compositeCoresilienceAReciprocalSum(n_max=2 * 10 ** 11)
        print(f"Solution to Project Euler #245 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 246 in eval_nums:
        since = time.time() 
        res = latticePointsAtTangentsToEllipseLatticePoints(
            upper_bound_angle_deg=45,
            m_x=-2000,
            g_x=8000,
            circle_radius=15000,
            incl_upper_bound_angle=False,
            incl_ellipse_border=False
        )
        print(f"Solution to Project Euler #246 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 247 in eval_nums:
        since = time.time()
        res = squaresUnderHyperbola(target_index=(3, 3))
        print(f"Solution to Project Euler #247 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 248 in eval_nums:
        since = time.time() 
        res = mthSmallestNumbersWithEulerTotientFunctionNFactorial(n=13, m=15 * 10 ** 4)
        print(f"Solution to Project Euler #248 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 249 in eval_nums:
        since = time.time() 
        res = primeSumsetSums(p_max=5000, md=10 ** 16)
        print(f"Solution to Project Euler #249 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 250 in eval_nums:
        since = time.time() 
        res = numberToItsOwnPowerSubsetDivisibleByNumberCount(n_max=250250, div=250, md=10 ** 16)
        print(f"Solution to Project Euler #250 = {res}, calculated in {time.time() - since:.4f} seconds")

    #print(f"Total time taken = {time.time() - since0:.4f} seconds")

if __name__ == "__main__":
    eval_nums = {211}
    evaluateProjectEulerSolutions201to250(eval_nums)
