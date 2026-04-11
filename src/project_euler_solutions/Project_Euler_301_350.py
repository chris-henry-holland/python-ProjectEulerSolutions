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


# Problem 301
def nimVariantPlayer2WinsWithPerfectPlayConfigurationsCount(pow2: int=30) -> int:
    """
    Solution to Project Euler #301
    """
    curr = [1, 1]
    for _ in range(pow2):
        curr = [curr[1], sum(curr)]
    return curr[1]



# Problem 306
def paperStripGamePlayer1WinsWithPerfectPlayCount(n_square_pick: int=2, n_max: int=10 ** 6) -> int:
    """
    Solution to Project Euler #306
    """
    # Using Sprague-Grundy theorem
    g_arr = [0] * n_square_pick
    res = 0
    for i in range(n_square_pick, n_max + 1):
        if not i % 1000: print(f"i = {i} of {n_max}")
        seen = set()
        for i2 in range(0, ((i - n_square_pick) >> 1) + 1):
            g = g_arr[i2] ^ g_arr[i - i2 - n_square_pick]
            seen.add(g)
        for g in range(len(seen) + 1):
            if g in seen: continue
            g_arr.append(g)
            break
        res += bool(g_arr[-1])
    return res



##############
project_euler_num_range = (301, 350)

def evaluateProjectEulerSolutions251to300(eval_nums: Optional[Set[int]]=None) -> None:
    if not eval_nums:
        eval_nums = set(range(project_euler_num_range[0], project_euler_num_range[1] + 1))

    since0 = time.time()

    if 301 in eval_nums:
        since = time.time()
        res = nimVariantPlayer2WinsWithPerfectPlayConfigurationsCount(pow2=30)
        print(f"Solution to Project Euler #301 = {res}, calculated in {time.time() - since:.4f} seconds")


    if 306 in eval_nums:
        since = time.time()
        res = paperStripGamePlayer1WinsWithPerfectPlayCount(n_square_pick=2, n_max=10 ** 5)
        print(f"Solution to Project Euler #301 = {res}, calculated in {time.time() - since:.4f} seconds")

    print(f"Total time taken = {time.time() - since0:.4f} seconds")

if __name__ == "__main__":
    eval_nums = {306}
    evaluateProjectEulerSolutions251to300(eval_nums)
