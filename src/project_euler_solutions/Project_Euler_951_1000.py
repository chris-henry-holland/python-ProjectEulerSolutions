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


##############
project_euler_num_range = (951, 1000)

def evaluateProjectEulerSolutions951to1000(eval_nums: Optional[Set[int]]=None) -> None:
    if not eval_nums:
        eval_nums = set(range(project_euler_num_range[0], project_euler_num_range[1] + 1))

    if 959 in eval_nums:
        since = time.time()
        res = None#optimumPolynomial(((1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1)))
        print(f"Solution to Project Euler #101 = {res}, calculated in {time.time() - since:.4f} seconds")

if __name__ == "__main__":
    eval_nums = {959}
    evaluateProjectEulerSolutions951to1000(eval_nums)