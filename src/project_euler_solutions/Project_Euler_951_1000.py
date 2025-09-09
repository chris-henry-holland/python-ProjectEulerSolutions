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

import numpy as np

from collections import deque, defaultdict
from sortedcontainers import SortedDict, SortedList, SortedSet

from data_structures.fractions import CustomFraction
from data_structures.prime_sieves import PrimeSPFsieve, SimplePrimeSieve

from algorithms.number_theory_algorithms import gcd, lcm, isqrt, integerNthRoot, extendedEuclideanAlgorithm
from algorithms.pseudorandom_number_generators import blumBlumShubPseudoRandomGenerator
from algorithms.string_searching_algorithms import KnuthMorrisPratt
#from algorithms.geometry_algorithms import determinant


##############
project_euler_num_range = (951, 1000)


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


def evaluateProjectEulerSolutions951to1000(eval_nums: Optional[Set[int]]=None) -> None:
    if not eval_nums:
        eval_nums = set(range(project_euler_num_range[0], project_euler_num_range[1] + 1))

    if 959 in eval_nums:
        since = time.time()
        res = randomWalkNumberOfUniquePointsOverPathLengthInfLimit(l_step_len=89, r_step_len=97, eps=10 ** -12)
        print(f"Solution to Project Euler #959 = {res}, calculated in {time.time() - since:.4f} seconds")

if __name__ == "__main__":
    eval_nums = {959}
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