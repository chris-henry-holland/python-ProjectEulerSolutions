#!/usr/bin/env python

from typing import (
    List,
    Tuple,
    Optional,
)

import bisect
import gmpy2
import math
import os
import time

import numpy as np
#from PIL import Image
import scipy.signal as sig

from data_structures.prime_sieves import PrimeSPFsieve, SimplePrimeSieve
from data_structures.addition_chains import AdditionChainCalculator

from algorithms.number_theory_algorithms import gcd, lcm, isqrt

def addFractions(frac1: Tuple[int, int], frac2: Tuple[int, int]) -> Tuple[int, int]:
    """
    Finds the sum of two fractions in lowest terms (i.e. such that
    the numerator and denominator are coprime)

    Args:
        frac1 (2-tuple of ints): The first of the fractions to sum,
                in the form (numerator, denominator)
        frac2 (2-tuple of ints): The second of the fractions to sum,
                in the form (numerator, denominator)
    
    Returns:
    2-tuple of ints giving the sum of frac1 and frac2 in the form (numerator,
    denominator). If the result is negative then the numerator is negative
    and the denominator positive.
    """
    denom = lcm(abs(frac1[1]), abs(frac2[1]))
    numer = (frac1[0] * denom // frac1[1]) + (frac2[0] * denom // frac2[1])
    g = gcd(numer, denom)
    return (numer // g, denom // g)

def multiplyFractions(frac1: Tuple[int, int], frac2: Tuple[int, int]) -> Tuple[int, int]:
    """
    Finds the product of two fractions in lowest terms (i.e. such that
    the numerator and denominator are coprime)

    Args:
        frac1 (2-tuple of ints): The first of the fractions to multiply,
                in the form (numerator, denominator)
        frac2 (2-tuple of ints): The second of the fractions to multiply,
                in the form (numerator, denominator)
    
    Returns:
    2-tuple of ints giving the product of frac1 and frac2 in the form (numerator,
    denominator). If the result is negative then the numerator is negative
    and the denominator positive.
    """
    neg = (frac1[1] < 0) ^ (frac1[1] < 0) ^ (frac2[0] < 0) ^ (frac2[1] < 0)
    frac_prov = (abs(frac1[0] * frac2[0]), abs(frac1[1] * frac2[1]))
    g = gcd(frac_prov[0], frac_prov[1])
    return (-(frac_prov[0] // g) if neg else (frac_prov[0] // g), frac_prov[1] // g)

# Problem -1
def ramanujanSummationOfAllMultiples(nums: List[int]=[3, 5]) -> Tuple[int, int]:
    since = time.time()
    n_nums = len(nums)
    res = (0, 1)
    for bm in range(1, 1 << n_nums):
        neg = True
        bm2 = bm
        l = 1
        i = 0
        while bm2:
            if bm2 & 1:
                neg = not neg
                l = lcm(l, nums[i])
            bm2 >>= 1
            i += 1
        contrib = (l if neg else -l, 12)
        #print(l, contrib)
        res = addFractions(res, contrib)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem root 13
def rootExpansionDigits(num: int, n_digs: int, base: int=10) -> List[int]:

    num *= base ** (2 * n_digs)
    num_sqrt = isqrt(num)
    res = []
    for _ in range(n_digs):
        num_sqrt, r = divmod(num_sqrt, base)
        res.append(r)
    res = res[::-1]
    #print(res)
    return res

def rootExpansionDigitSum(num: int=13, n_digs: int=1_000, base: int=10) -> int:
    """
    Solution to Project Euler #root 13
    """
    since = time.time()
    res = sum(rootExpansionDigits(num, n_digs, base=base))
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res


# Problem Heegner
def closestCosPiSqrtInteger(abs_n_max: int=1000) -> int:
    """
    Solution to Project Euler Bonus Problem Heegner

    For integers n with absolute value no greater than
    abs_n_max, finds the value of n such that the value
    of:
        x = cos(pi * sqrt(n))
    is closest to an integer, i.e. the value of n for
    which (abs(x - round(x))) is smallest.

    Args:
        Optional named:
        abs_n_max (int): The largest absolute value of the
                integers considered.
            Default: 1000
    
    Returns:
    Integer (int) giving the value of the integer n with
    absolute value no greater than abs_n_max such that
    cos(pi * sqrt(n)) is closest to an integer.

    Outline of rationale:
    Given that:
        cos(pi * sqrt(n)) = cosh(i * pi * sqrt(n))
    for negative n, we get:
        cos(pi * sqrt(n)) = cosh(i * pi * sqrt(abs(n)))
    This becomes very large very quickly as n becomes
    more negative, and with normal float operations the
    fractional part will rapidly not be calculated accurately
    (if at all).
    We therefore use the gmpy2 package to perform the
    calculations with arbitrary precision, and with each
    calculation we ensure that the current precision is
    sufficient to get sufficient precision in the fractional
    part to compare between answers, increasing the
    precision if not.

    Note that, given that for x >> 1:
        cosh(x) approximately equals exp(x)
    it is to be expected that for relatively small values of
    abs_n_max the solution is the negative of a Heegner number,
    specifically the largest Heegner number no greater than
    n_max, as a property of these numbers m is that
    exp(pi * sqrt(m)) is extremely close to an integer, with
    the larger the Heeneger number the closer to an integer.
    The Heegner numbers are:
        1, 2, 3, 7, 11, 19, 43, 67, and 163
    """
    since = time.time()
    squares = set(i ** 2 for i in range(math.isqrt(abs_n_max) + 1))
    res = (float("inf"), -1)
    for func, filter, mult in ((gmpy2.cos, lambda x: x not in squares, 1), (gmpy2.cosh, lambda x: True, -1)):
        for i in range(1, abs_n_max + 1):
            if not filter(i): continue
            while True:
                val = func(gmpy2.const_pi() * gmpy2.sqrt(gmpy2.mpc(i))).real
                val_str = str(val)
                increase_precision = False
                if "." not in set(val_str[-10:]):
                    zero_run = 0
                    nine_run = 0
                    for j in range(len(val_str)):
                        d = val_str[~j]
                        if d == ".":
                            increase_precision = True
                            break
                        if d == "0": zero_run += 1
                        else: zero_run = 0
                        if d == "9": nine_run += 1
                        else: nine_run = 0
                        #print(j, zero_run, nine_run, j - max(zero_run, nine_run))
                        if j - max(zero_run, nine_run) >= 10: break
                    if not increase_precision: break
                #print(val)
                gmpy2.get_context().precision += 10
            #print(i, val)
            v2 = abs(val - round(val))
            if v2 < res[0]:
                res = (v2, -i)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res[1]


# Problem 18i
def polynomialPrimeProductRemainder(p_min: int=10 ** 9, p_max: int=11 * 10 ** 8) -> int:
    since = time.time()
    ps = PrimeSPFsieve(p_max)
    print("found prime sieve")
    poly_func = lambda x: x ** 3 - 3 * x + 4
    res = 0
    p_i_mn = bisect.bisect_left(ps.p_lst, p_min)
    seen_primes = set()
    largest_poly_arg = -1
    poly_past_range = False
    for p_i in range(p_i_mn, len(ps.p_lst)):
        p = ps.p_lst[p_i]
        if not poly_past_range:
            for largest_poly_arg in range(largest_poly_arg + 1, p):
                val = poly_func(largest_poly_arg)
                #print(f"arg = {largest_poly_arg}, val = {val}")
                if val > p_max:
                    poly_past_range = True
                    break
                p_facts = ps.primeFactors(val)
                seen_primes |= set(p_facts)
                #if ps.isPrime(val):
                #    seen_primes.add(val)
        if p in seen_primes: continue
        ans = 1
        for i in range(p):
            ans = (ans * poly_func(i)) % p
            if not ans: break
        #print(f"p = {p}, product = {ans}")
        res += ans
    #print(seen_primes)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

def integerPlusSquareRootMultiplePower(a: int, b: int, c: int, exp: int, md: Optional[int]=None) -> Tuple[int]:
    """
    Calculates (a + b * sqrt(c)) ** exp in the form (n, m) where:
        (a + b * sqrt(c)) ** exp = n + m * sqrt(c)
    If md is not None, n and m are given modulo md.

    Calculates using binary exponentiation
    """
    if md is None:
        curr = (a, b)
        res = (1, 0)
        e = exp
        while True:
            if e & 1:
                res = (res[0] * curr[0] + c * res[1] * curr[1], res[0] * curr[1] + res[1] * curr[0])
            e >>= 1
            if not e: break
            curr = (curr[0] ** 2 + c * curr[1] ** 2, 2 * curr[0] * curr[1])
        return res
    curr = (a % md, b % md)
    res = (1, 0)
    e = exp
    while True:
        if e & 1:
            res = ((res[0] * curr[0] + c * res[1] * curr[1]) % md, (res[0] * curr[1] + res[1] * curr[0]) % md)
        e >>= 1
        if not e: break
        curr = ((pow(curr[0], 2, md) + c * pow(curr[1], 2, md)) % md, (2 * curr[0] * curr[1]) % md)
    return res

def sumOfPolynomialProductRemainders(n_min: int=10 ** 9, n_max: int=11 * 10 ** 8) -> int:
    """
    Solution to Project Euler Bonus Problem #18i
    """
    since = time.time()
    ps = SimplePrimeSieve()

    def primeChecker(num: int) -> bool:
        res = ps.millerRabinPrimalityTest(num, n_trials=10)
        # res = ps.isPrime(num, extend_sieve=False, extend_sieve_sqrt=False, use_miller_rabin_screening=True, n_miller_rabin_trials=3)
        return res
    
    res = 0
    i0 = n_min + (-n_min) % 12
    for i in range(i0, n_max + 1, 12):
        p1 = i + 1
        if p1 > n_max: break
        if primeChecker(p1):
            exp = (p1 - 1) // 3
            r_tup1 = integerPlusSquareRootMultiplePower(a=-2, b=1, c=3, exp=exp, md=p1)
            ans = (-18 * r_tup1[1] * (1 - 2 * r_tup1[0])) % p1
            res += ans
        p2 = i + 5
        if p2 > n_max: break
        if primeChecker(p2):
            exp = (p2 + 1) // 3
            r_tup2 = integerPlusSquareRootMultiplePower(a=-2, b=1, c=3, exp=exp, md=p2)
            ans = (18 * r_tup2[1] * (1 - 2 * r_tup2[0])) % p2
            res += ans

    """
    ps = SimplePrimeSieve(n_max)
    print(f"calculated prime sieve after {time.time() - since:.4f} seconds")
    res = 0
    i0 = bisect.bisect_left(ps.p_lst, n_min)
    for i in range(i0, len(ps.p_lst)):
        p = ps.p_lst[i]
        if p > n_max: break
        pmd12 = p % 12
        ans = 0
        if pmd12 == 1:
            exp = (p - 1) // 3
            r_tup = integerPlusSquareRootMultiplePower(a=-2, b=1, c=3, exp=exp, md=p)
            ans = -18 * r_tup[1] * (1 - 2 * r_tup[0])
        elif pmd12 == 5:
            exp = (p + 1) // 3
            r_tup = integerPlusSquareRootMultiplePower(a=-2, b=1, c=3, exp=exp, md=p)
            
            ans = 18 * r_tup[1] * (1 - 2 * r_tup[0])
        else: continue
        #print(p, exp, r_tup)
        ans %= p
        #print(f"{p}: {ans}")
        res += ans
    """
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem Secret
"""
def loadBlackAndWhitePNGImage(filename: str, relative_to_program_file_directory: bool=False) -> np.ndarray:
    ""
        Optional named:
        relative_to_program_file_directory (bool): If True then
                if doc is specified as a relative path, that
                path is relative to the directory containing
                the program file, otherwise relative to the
                current working directory.
            Default: False
    ""
    if relative_to_program_file_directory and not filename.startswith("/"):
        filename = os.path.join(os.path.dirname(__file__), filename)
    image_np = np.array(Image.open(filename))
    #shape = image_np.shape[:2]
    res = image_np[:, :, 0]

    #print(res)
    print(res.shape)
    return res

def repeatedAdjacentConvolutions(arr: np.ndarray, n_convolutions: int, md: Optional[int]=None) -> np.ndarray:
    pattern = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int)
    shp = arr.shape
    arr %= md
    arr2 = np.array(arr) #np.zeros((shp[0] + 1, shp[1] + 1), dtype=int)
    cycle_len = math.lcm(*shp, md)
    if md is not None:
        n_convolutions = n_convolutions % cycle_len
    print(f"n_convolutions = {n_convolutions}")
    print(f"cycle length = {cycle_len}")
    #arr2[1:-1, 1:-1] = arr
    print(arr2[:4, :4])
    print(arr2[:4,-4:])
    print(arr2[-4:,:4])
    print(arr2[-4:,-4:])
    for i in range(2 * cycle_len):
        arr2 = sig.convolve2d(arr2, pattern, boundary="wrap")[1:-1, 1:-1]
        if md is not None: arr2 %= md
        
        if not i % 100:
            print(i)
        if np.all(arr2 == arr): print("back to original after {i} convolutions")
        if not (i + 1) % cycle_len:
            print(i)
            print(arr2[:4, :4])
            print(arr2[:4,-4:])
            print(arr2[-4:,:4])
            print(arr2[-4:,-4:])
    return arr2

def repeatedAdjacentConvolutionsLoadedImage(n_convolutions: int=10 ** 12, in_filename: str="bonus_secret_statement.png", out_filename: str="bonus_secret_statement_decoded.png", relative_to_program_file_directory: bool=True, md: Optional[int]=7) -> np.ndarray:
    
    since = time.time()
    arr = loadBlackAndWhitePNGImage(in_filename, relative_to_program_file_directory=relative_to_program_file_directory)
    res = repeatedAdjacentConvolutions(arr, n_convolutions, md=md)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
"""
if __name__ == "__main__":
    to_evaluate = {"secret"}

    if not to_evaluate or "-1" in to_evaluate:
        res = ramanujanSummationOfAllMultiples(nums=[3, 5])
        print(f"Solution to Project Euler #-1 = {res}")

    if not to_evaluate or "root_13" in to_evaluate:
        res = rootExpansionDigitSum(num=13, n_digs=1_000, base=10)
        print(f"Solution to Project Euler #root 13 = {res}")

    if not to_evaluate or "heegner" in to_evaluate:
        res = closestCosPiSqrtInteger(abs_n_max=1000)
        print(f"Solution to Project Euler #heegner = {res}")

    if not to_evaluate or "18i" in to_evaluate:
        res = sumOfPolynomialProductRemainders(n_min=10 ** 9, n_max=11 * 10 ** 8)
        print(f"Solution to Project Euler #18i = {res}")

    """
    if not to_evaluate or "secret" in to_evaluate:
        res = repeatedAdjacentConvolutionsLoadedImage(n_convolutions=10 ** 12, in_filename="bonus_secret_statement.png", out_filename="bonus_secret_statement_decoded.png", relative_to_program_file_directory=True, md=7)
        print(f"Solution to Project Euler #secret = {res}")
    """