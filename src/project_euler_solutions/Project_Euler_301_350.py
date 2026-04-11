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


    print(f"Total time taken = {time.time() - since0:.4f} seconds")

if __name__ == "__main__":
    eval_nums = {301}
    evaluateProjectEulerSolutions251to300(eval_nums)
