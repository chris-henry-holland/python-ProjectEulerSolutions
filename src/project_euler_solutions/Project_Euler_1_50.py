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
)

import bisect
import copy
import functools
import heapq
import itertools
import math
import os
import random
import time

from pathlib import Path
from sortedcontainers import SortedList

from data_structures.prime_sieves import PrimeSPFsieve

# Problem 1
def multipleSum(n_max: int=999, fact_list: Tuple[int]=(3, 5)) -> int:
    """
    Solution to Project Euler #1

    Finds the sum of all natural numbers no greater than n_max which
    are multiples of any of the integers in fact_list.
    
    Args:
        Optional named:
        n_max (int): The largest natural number considered
            Default: 999
        fact_list (tuple of ints): The natural numbers of which each
                of the numbers considered should be a multiple of
                at least one.
            Default: (3, 5)
    
    Returns:
    Integer (int) giving the sum of all natural numbers no greater
    than n_max which are multiples of any of the integers in fact_list
    """
    #since = time.time()
    # Constructing list whose ith element specifies whether i is
    # divisible by any of fact_list (i.e. a sieve)
    i_incl_list = [False] * (n_max + 1)
    for fact in fact_list:
        for i in range(0, n_max + 1, fact):
            i_incl_list[i] = True
    res = 0
    # Summing the indices in i_incl_list whose corresponding value
    # is True
    for i, i_incl in enumerate(i_incl_list):
        if i_incl: res += i
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 2
def multFibonacciSum(n_max: int=4 * 10 ** 6, fact: int=2) -> int:
    """
    Solution to Project Euler #2

    Finds the sum of Fibonacci numbers (starting with 1, 2, ...)
    whose values do not exceed n_max and are divisible by n_fact.
    
    Args:
        Optional named:
        n_max (int): The largest value of Fibonacci numbers
                considered.
            Default: 4 * 10 ** 6
        fact (int): The strictly positive integer which must
                divide every Fibonacci number counted.
            Default: 2
    
    Returns:
    Integer (int) giving the sum of Fibonacci numbers (starting
    with 1, 2, ...) whose values are divisible by fact and do not
    exceed n_max
    """
    #since = time.time()
    n_sum = 0
    n0, n1 = 1, 2
    n_sum = n0 if n0 % fact == 0 else 0
    start = True
    while n1 <= n_max:
        if not start:
            # Calculating the new term, allocating it to n1 and
            # allocating the old n1 term to n0
            n0, n1 = n1, n0 + n1
        else: start = False
        # Checking if new term divisible by fact
        if n1 % fact == 0:
            n_sum += n1
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return n_sum

# Problem 3
def primeFactorisation(num: int) -> Dict[int, int]:
    """
    Finds the prime factorisation of a strictly positive integer num.
    
    Args:
        Required positional:
        num (int): The strictly positive integer whose prime
                factorisation is being calculated
    
    Returns:
    Dictionary (dict) whose keys are ints representing each of the
    distinct prime factors of n and whose corresponding values are the
    number of occurrences of that prime in the prime factorisation
    of num.
    """
    p = 2
    out_dict = {}
    while p * p <= num:
        while num % p == 0:
            num //= p
            out_dict.setdefault(p, 0)
            out_dict[p] += 1
        if p == 2: p = 3
        else: p += 2
    if num != 1:
        out_dict[num] = 1
    return out_dict

def primeFactorisationChk(num: int) -> bool:
    """
    Checks whether prime factorisation of strictly positive integer num
    calculated by the function primeFactorisation() is a factorisation
    of num.
    
    Args:
        Required positional:
        num (int): The strictly positive integer whose prime
                factorisation by primeFactorisation() is being checked.
    
    Returns:
    Boolean (bool), True if product of the terms in the factorisation
    for num found by primeFactorisation() is indeed equal to num,
    otherwise False
    """
    fact_dict = primeFactorisation(num)
    prod = 1
    for k, v in fact_dict.items():
        prod *= k ** v
    return prod == num
    
def largestPrimeFactor(num: int=600851475143) -> int:
    """
    Solution to Project Euler #3

    Finds the largest prime factor of integer num.
    
    Args:
        Optional named:
        num (int): Integer strictly greater than 1 for which the
                largest prime factor is sought.
            Default: 600851475143
    
    Returns:
    Integer (int) giving the largest prime factor of num.
    """
    #since = time.time()
    fact_dict = primeFactorisation(num)
    res = max(fact_dict.keys())
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 4
def isPalindromeString(s: str) -> bool:
    """
    Checks if the string s is palindromic (i.e. the string when
    reversed is equal to the original string)
    
    Args:
        Required positional:
        s (str): The string being checked for whether it is a
                palindrome.
    
    Returns:
    Boolean (bool), True if s a palindrome, otherwise False.
    """
    n = len(s)
    for i in range(n >> 1):
        if s[i] != s[~i]: return False
    return True

def isPalindromeInteger(num: int, base: int=10) -> bool:
    """
    Checks if the representation of non-negative integer num in the
    chosen base is palindromic (i.e. the digits of the
    representation of num are in the same order when read both
    forwards and backwards). Note that the representation of
    num in the chosen base does not include leading zeros.
    
    Args:
        Required positional:
        num (int): The non-negative integer being checked for whether
                it is palindromic.
        
        Optional named:
        base (int): The base in which num is to be represented
                when assessing whether it is palindromic.
            Default: 10
    
    Returns:
    Boolean (bool), True if num is palindromic, otherwise False.
    """
    num_lst = []
    while num:
        num, r = divmod(num, base)
        num_lst.append(r)
    n = len(num_lst)
    for i in range(n >> 1):
        if num_lst[i] != num_lst[~i]: return False
    return True

def largestPalindromicProductProperties(
    prod_nums_n_digit: int=3,
    n_prod_nums: int=2,
    base: int=10,
) -> Tuple[int, List[int]]:
    """
    Finds the largest palindromic integer in the chosen base (without
    leading zeros) that is the product of n_prod_nums integers which
    each contain prod_nums_n_digit digits in their representation in
    the chosen base (again without leading zeros), providing the value
    of the palindromic number and the n_prod_nums integers whose product
    forms that palindromic integer.
    
    Args:
        Opitional named:
        prod_nums_n_digit (int): The number of digits in their 
                representation in the chosen base of the integers
                whose product forms the resulting palindromic integer.
            Default: 2
        n_prod_nums (int): The number of integers whose product forms
                the result.
            Default: 3
        base (int): The base used to represent the integers in
                question.
            Default: 10
    
    Returns:
    2-tuple for which index 0 contains an integer (int) giving the
    largest such palindromic integer in the chosen base and whose
    index 1 contains a list containing n_prod_nums integers with
    prod_nums_n_digit digits in their representation in the chosen
    base whose product forms the identified palindromic integer in
    order of increasing value. If no palindromic integers in the
    chosen base that satisfy the conditions exist, then (-1, [])
    is returned.
    """
    #since = time.time()
    curr_best = 0
    curr_inds = None
    iter_obj = [range(int(base ** prod_nums_n_digit) - 1,
            int(base ** (prod_nums_n_digit - 1)) - 1, -1)] * n_prod_nums
    for inds in itertools.product(*iter_obj):
        if inds[0] ** len(inds) <= curr_best: break
        # To ensure no repeating, skip to next iteration if any of
        # the indices exceeds the index to its left
        skip = True
        for j in range(1, len(inds)):
            if inds[j] > inds[j - 1]:
                break
        else:
            skip = False
        if skip: continue
        # Checking that at least one of the indices is larger
        # than its corresponding index in the current best
        # solution- otherwise no chance it will be larger
        # than the current best solution, so go to next
        # iteration
        if curr_inds is not None:
            for i, i0 in zip(inds, curr_inds):
                if i > i0: break
            else:
                continue
        # Calculating the product
        prod = 1
        for i in inds: prod *= i
        # Checking whether the product exceeds the current
        # best (if not goes to next iteration)
        if prod <= curr_best:
            continue
        # Checking whether the product is a palindrome
        if isPalindromeInteger(prod, base=base):
            curr_best = prod
            curr_inds = inds
    
    res = (curr_best, curr_inds[::-1]) if curr_best else (-1, ())
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return (res[0], list[res[1]])

def largestPalindromicProduct(
    prod_nums_n_digit: int=3,
    n_prod_nums: int=2,
    base: int=10,
) -> int:
    """
    Solution to Project Euler #4

    Finds the largest palindromic integer in the chosen base (without
    leading zeros) that is the product of n_prod_nums integers which
    each contain prod_nums_n_digit digits in their representation in
    the chosen base (again without any leading zeros), providing the
    value of the palindromic integer.
    
    Args:
        Opitional named:
        prod_nums_n_digit (int): The number of digits in their
                representation in the chosen base of the integers whose
                product forms the resulting palindromic integer.
            Default: 2
        n_prod_nums (int): The number of integers whose product forms
                the result.
            Default: 3
        base (int): The base used to represent the integers in
                question.
            Default: 10
    
    Returns:
    Integer (int) giving the largest palindromic integer that can
    be produced as described above. If no such palindromic integers
    exist, then returns -1.
    """
    return largestPalindromicProductProperties(
        prod_nums_n_digit=prod_nums_n_digit,
        n_prod_nums=n_prod_nums, 
        base=base,
    )[0]
    
# Problem 5
def gcd(a: int, b: int) -> int:
    """
    Finds the greatest common divisor of two non-negative integers
    a and b using the Euclidean algorithm
    
    Args:
        Required positional:
        a (int): One of the integers whose greatest common
                divisor with the other is sought.
        b (int): One of the integers whose greatest common
                divisor with the other is sought.
    
    Returns:
    Integer (int) giving the greatest common divisor of a and b
    """
    if a == 0: return b
    return gcd(b % a, a)

def lcm(a: int, b: int) -> int:
    """
    Finds the lowest common multiple of two non-negative integers
    a and b.
    
    Args:
        Required positional:
        a (int): One of the integers whose lowest common
                multiple with the other is sought.
        b (int): One of the integers whose lowest common
                multiple with the other is sought.
    
    Returns:
    Integer (int) giving the lowest common multiple of a and b
    """
    return a * (b // gcd(a, b))

def smallestMultiple(div_max: int=20) -> int:
    """
    Solution to Project Euler #5
    
    Finds the smallest strictly positive integer that is an exact
    multiple of every strictly positive integer up to div_max.
    
    Args:
        Optional named:
        div_max (int): The strictly postive integer for which
                the result must be a multiple of all strictly
                positive integers no greater than this.
            Default: 20
    
    Returns:
    Integer (int) giving the smallest integer that is a multiple of
    every strictly positive integer up to div_max.
    """
    #since = time.time()
    res = 1
    for num in range(2, div_max + 1):
        res = lcm(res, num)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 6
def sumSquareDiff(n_max: int=100) -> int:
    """
    Solution to Project Euler #6

    Finds the difference between the sum of the squares and the
    square of the sum of the first n_max strictly positive integers.
    
    Args:
        Optional named:
        n_max (int): The largest integer included in either sum
            Default: 100
    
    Returns:
    Integer (int) giving the difference between the square of the sum
    and the sum of the squares of the first n natural numbers.
    
    Outline of rationale:
    Uses the formula for the sum of the first n integers:
        (n * (n + 1)) / 2
    and the formula for the sum of the first n squares:
        (n * (n + 1) * (2 * n + 1)) / 6
    """
    #since = time.time()
    sq_sum = (n_max * (n_max + 1) * (2 * n_max + 1)) // 6
    sum_sq = (n_max ** 2 * (n_max + 1) ** 2) >> 2
    res = sum_sq - sq_sum
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
     
# Problem 7
def firstNPrimes(n_primes: int) -> Tuple[int]:
    """
    Finds the first n_primes primes
    
    Args:
        Required positional:
        n_primes (int): The number of the initial primes to be
                found.
    
    Returns:
    A tuple of ints containing the first n_primes prime numbers
    in increasing order.
    """

    # Upper bound for the n_primes:th prime number
    if n_primes > 17:
        n_max = math.floor(1.26 * n_primes * math.log(n_primes))
    elif n_primes > 3:
        n_max = math.ceil(n_primes * (math.log(n_primes) +
                math.log(math.log(n_primes))))
    else:
        n_max = 5
    prime_list = []
    sieve = [True for x in range(n_max + 1)]
    for p in range(2, n_max + 1):
        if sieve[p]:
            prime_list.append(p)
            if len(prime_list) >= n_primes:
                break
            for i in range(p ** 2, n_max + 1, p):
                sieve[i] = False
    return tuple(prime_list)

def findPrime(n: int=10001) -> int:
    """
    Solution to Project Euler #7

    Finds the nth prime number
    
    Args:
        Optional named:
        n (int): The position in the sequence of primes of the prime
                to be found.
            Default: 10001
    
    Returns:
    Integer giving the nth prime number.
    """
    #since = time.time()
    res = firstNPrimes(n)[-1]
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
     
# Problem 8
num_str = '73167176531330624919225119674426574742355349194934'\
                '96983520312774506326239578318016984801869478851843'\
                '85861560789112949495459501737958331952853208805511'\
                '12540698747158523863050715693290963295227443043557'\
                '66896648950445244523161731856403098711121722383113'\
                '62229893423380308135336276614282806444486645238749'\
                '30358907296290491560440772390713810515859307960866'\
                '70172427121883998797908792274921901699720888093776'\
                '65727333001053367881220235421809751254540594752243'\
                '52584907711670556013604839586446706324415722155397'\
                '53697817977846174064955149290862569321978468622482'\
                '83972241375657056057490261407972968652414535100474'\
                '82166370484403199890008895243450658541227588666881'\
                '16427171479924442928230863465674813919123162824586'\
                '17866458359124566529476545682848912883142607690042'\
                '24219022671055626321111109370544217506941658960408'\
                '07198403850962455444362981230987879927244284909188'\
                '84580156166097919133875499200524063689912560717606'\
                '05886116467109405077541002256983155200055935729725'\
                '71636269561882670428252483600823257530420752963450'

def largestSubstringProductProperties(num_str: str=num_str, n_char: int=13):
    """
    For a given string of digits, num_str, identifies the value product,
    starting index and the characters of a non-empty sub-string with
    length n_char for which the product of the digits it contains is at
    least as large as any other such sub-string of that length.
    
    Args:
        Optional named:
        num_str (str): The string of digits from which the sub-strings
                are extracted.
            Default: num_str (as defined in Project_Euler_1_50.py)
        n_char (int): The length of the sub-strings in question.
            Default: 13
    
    Returns:
    Dictionary (dict) with keys:
        "best_value": the product of the numbers in the sub-string
            that has the largest such product
        "location": the index of the first character in a sub-string
            whose product is equal to that in "best_value"
        "largest_string": the sub-string of length n_char starting
            at the index given by "location", and whose product
            is equal to the value given by "best_value"
            
    """
    #since = time.time()
    # Any product including 0 as one of the elements of the product
    # has value 0, so there does not exist a sub-string of length
    # n_char that does not contain any zeros, the sub-string
    # sought cannot contain the character "0". Therefore, we split the
    # string into sections around the "0" characters and look at
    # each one separately.
    num_split = num_str.split("0")
    for s in num_split:
        if len(s) >= n_char: break
    else:
        return {"best_value": 0, "location": 0,\
                "largest_string": num_str[:n_char]}
    
    start_index = 0
    
    best_val = 0
    best_index = 0
    best_str = num_str[:n_char]
    
    for section in num_split:
        pos = 0
        while pos <= len(section) - n_char:
            while pos < len(section) - n_char and\
                    int(section[pos]) <= int(section[pos + n_char]):
                pos += 1
            prod = 1
            for j in range(0, n_char):
                prod *= int(section[pos + j])
            if prod > best_val:
                best_val = prod
                best_index = start_index + pos
                best_str = section[pos:pos + n_char]
            pos += 2
                
        start_index += len(section) + 1
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return {"best_value": best_val, "location": best_index, 
            "largest_string": best_str}

def largestSubstringProduct(num_str: str=num_str, n_char: int=13):
    """
    Solution to Project Euler #8

    For a given string of digits, num_str, identifies a non-empty
    sub-string with length n_char for which the product of the digits
    it contains is at least as large as any other such sub-string of
    that length.
    
    Args:
        Optional named:
        num_str (str): The string of digits from which the sub-strings
                are extracted.
            Default: num_str (as defined in Project_Euler_1_50.py)
        n_char (int): The length of the sub-strings in question.
            Default: 13
    
    Returns:
    Integer (int) giving the product of the numbers in the sub-string
    that has the largest such product.
    """
    res = largestSubstringProductProperties(num_str=num_str, n_char=n_char)
    return res["best_value"]

# Problem 9
# Consider implementing using Euclid's formula for Pythagorean triples
def pythagoreanTripleWithGivenSum(side_sum: int) -> List[Tuple[int]]:
    """
    For a given strictly positive integer side_sum, finds all
    right-angled triangle whose sides lengths are integers and
    collectively sum to side_sum.
    
    Args:
        Required positional:
        side_sum (int): The sum of the sides of the right-angled
                triangles with integer side lengths being sought.
    
    Returns:
    List of 3-tuples of ints, where each 3-tuple gives the side lengths
    of one of the right-angled triangles whose side lengths sum to
    side_sum. Collectively the entries of the list represent all such
    right-angled triangles.
    """
    res = []
    for c in range(1, side_sum // 2 + 1):
        # The minimum length c can be given the total side length
        # constraint is if this is an isosceles right
        # triangle, where c = (sqrt 2) * a = (sqrt 2) * b so
        # side_sum = a + b + c 
        #          >= c * (1 + 2/(sqrt 2))
        #          = c * (1 + (sqrt 2))
        # so (since both sides positive prior to squaring):
        # (side_sum - c) ** 2 >= 2 * c ** 2
        # leading to:
        # c ** 2 + 2 * side_sum * c <= side_sum ** 2
        if c ** 2 + 2 * side_sum * c <= side_sum ** 2: 
            continue
        for b in range((side_sum - c) - (side_sum - c) // 2,\
                side_sum - c): 
            a = side_sum - c - b
            if a ** 2 + b ** 2 == c ** 2:
                #print(f"a = {a}, b = {b}, c = {c}")
                res.append(tuple(sorted((a, b, c))))
    #if not res:
    #    print('Failed to find triple')
    return res

def specialPythagoreanTripletProducts(side_sum: int=1000) -> Set[int]:
    """
    For a given strictly positive integer side_sum, finds all
    right-angled triangle whose sides lengths are integers and
    collectively sum to side_sum and finds the product of the
    side lengths.
    
    Args:
        Optional named:
        side_sum (int): The sum of the sides of the right-angled
                triangles with integer side lengths being sought.
            Default: 1000
    
    Returns:
    Set of ints, representing set of products of side lengths of
    the right-angled triangles whose side lengths sum to side_sum.
    """
    #since = time.time()
    lst = pythagoreanTripleWithGivenSum(side_sum)
    res = {a * b * c for (a, b, c) in lst}
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

def smallestSpecialPythagoreanTripletProduct(side_sum: int=1000) -> int:
    """
    Solution to Project Euler #9

    For a given strictly positive integer side_sum, finds the
    smallest integer for which there exists a right-angled
    triangle whose sides lengths are integers and collectively
    sum to side_sum whose product is equal to that integer.
    
    Args:
        Optional named:
        side_sum (int): The sum of the sides of the right-angled
                triangles with integer side lengths being sought.
            Default: 1000
    
    Returns:
    Integer (int) representing the smallest integer for which there
    exists a right-angled triangle whose sides lengths are integers
    and collectively sum to side_sum whose product is equal to that
    integer. If no such integer exists, returns -1.
    """
    res = specialPythagoreanTripletProducts(side_sum=side_sum)
    return min(res) if res else -1

# Problem 10
def sumPrimes(n_max: int=2 * 10 ** 6 - 1) -> int:
    """
    Solution to Project Euler #10

    Finds the sum of all prime numbers no greater than n_max.
    
    Args:
        Optional named:
        n_max (int): The largest value that may be included in the
                sum.
            Default: 2 * 10 ** 6 - 1
    
    Returns:
    The sum of all prime numbers no greater than n_max.
    """
    #since = time.time()
    p_sum = 0
    sieve = [True for x in range(n_max)]
    for p in range(2, n_max):
        if sieve[p]:
            p_sum += p
            for i in range(p ** 2, n_max, p):
                sieve[i] = False
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return p_sum
    

# Problem 11
num_grid = [[8, 2, 22, 97, 38, 15, 00, 40, 00, 75, 4, 5, 7, 78, 52, 12, 50, 77, 91, 8],
            [49, 49, 99, 40, 17, 81, 18, 57, 60, 87, 17, 40, 98, 43, 69, 48, 4, 56, 62, 00],
            [81, 49, 31, 73, 55, 79, 14, 29, 93, 71, 40, 67, 53, 88, 30, 3, 49, 13, 36, 65],
            [52, 70, 95, 23, 4, 60, 11, 42, 69, 24, 68, 56, 1, 32, 56, 71, 37, 2, 36, 91],
            [22, 31, 16, 71, 51, 67, 63, 89, 41, 92, 36, 54, 22, 40, 40, 28, 66, 33, 13, 80],
            [24, 47, 32, 60, 99, 3, 45, 2, 44, 75, 33, 53, 78, 36, 84, 20, 35, 17, 12, 50],
            [32, 98, 81, 28, 64, 23, 67, 10, 26, 38, 40, 67, 59, 54, 70, 66, 18, 38, 64, 70],
            [67, 26, 20, 68, 2, 62, 12, 20, 95, 63, 94, 39, 63, 8, 40, 91, 66, 49, 94, 21],
            [24, 55, 58, 5, 66, 73, 99, 26, 97, 17, 78, 78, 96, 83, 14, 88, 34, 89, 63, 72],
            [21, 36, 23, 9, 75, 00, 76, 44, 20, 45, 35, 14, 00, 61, 33, 97, 34, 31, 33, 95],
            [78, 17, 53, 28, 22, 75, 31, 67, 15, 94, 3, 80, 4, 62, 16, 14, 9, 53, 56, 92],
            [16, 39, 5, 42, 96, 35, 31, 47, 55, 58, 88, 24, 00, 17, 54, 24, 36, 29, 85, 57],
            [86, 56, 00, 48, 35, 71, 89, 7, 5, 44, 44, 37, 44, 60, 21, 58, 51, 54, 17, 58],
            [19, 80, 81, 68, 5, 94, 47, 69, 28, 73, 92, 13, 86, 52, 17, 77, 4, 89, 55, 40],
            [4, 52, 8, 83, 97, 35, 99, 16, 7, 97, 57, 32, 16, 26, 26, 79, 33, 27, 98, 66],
            [88, 36, 68, 87, 57, 62, 20, 72, 3, 46, 33, 67, 46, 55, 12, 32, 63, 93, 53, 69],
            [4, 42, 16, 73, 38, 25, 39, 11, 24, 94, 72, 18, 8, 46, 29, 32, 40, 62, 76, 36],
            [20, 69, 36, 41, 72, 30, 23, 88, 34, 62, 99, 69, 82, 67, 59, 85, 74, 4, 36, 16],
            [20, 73, 35, 29, 78, 31, 90, 1, 74, 31, 49, 71, 48, 86, 81, 16, 23, 57, 5, 54],
            [1, 70, 54, 71, 83, 51, 54, 69, 16, 92, 33, 48, 61, 43, 52, 1, 89, 19, 67, 48]]

def largestLineProductProperties(num_grid: List[List[int]]=num_grid, line_len: int=4) -> Dict[int, Any]:
    """
    For a 2D grid of integers num_grid, finds the product, start
    position, orientation and elements of the line_len consecutive
    entries on a  horizontal, vertical or diagonal straight line in
    num_grid whose product is at least as large as any other such line.
    
    Args:
        Optional named:
        num_grid (list of lists of ints): The 2D grid of integers being
                considered. All of the lists in the main list must have
                the same length.
            Default: num_grid (as defined in Project_Euler_1_50.py)
        line_len (int): The number of integers in the lines in
                num_grid considered.
            Default: 4
    
    Returns:
    Dictionary (dict) with entries:
        "max_product" (int): the value of maximum product of the
            line_len consecutive integers found
        "start_pos" (2-tuple of ints): the indices in the lists where
            a the consecutive integers start.
        "orient" (int): the orientation of the consecutive integers,
            starting at "start_pos". 0: horizontally right,
            1: vertically down, 2: diagonally down-right,
            3: diagonally down-left
        "elements" (line_len-tuple of ints): the values of the
            consecutive integers in the line specified by "start_pos"
            and "orient"
    """
    #since = time.time()
    
    shape = (len(num_grid), len(num_grid[0]))
    
    max_prod = 0
    pos = None
    orient = None
    max_el_list = None
    # Horizontal
    for i in range(shape[0]):
        for j in range(shape[1] - line_len + 1):
            prod = 1
            el_list = []
            for k in range(line_len):
                #print(f"k = {k}")
                #print(num_grid[i][j + k])
                prod *= num_grid[i][j + k]
                el_list.append(num_grid[i][j + k])
            if prod > max_prod:
                max_prod = prod
                pos = (i, j)
                orient = 0
                max_el_list = tuple(el_list)
    # Vertical and diagonal
    for i in range(shape[0] - line_len + 1):
        # Vertical
        for j in range(shape[1]):
            prod = 1
            el_list = []
            for k in range(line_len):
                prod *= num_grid[i + k][j]
                el_list.append(num_grid[i + k][j])
            if prod > max_prod:
                max_prod = prod
                pos = (i, j)
                orient = 1
                max_el_list = tuple(el_list)
        # Diagonal
        for j in range(shape[1] - line_len + 1):
            # Diagoal left
            prod = 1
            el_list = []
            for k in range(line_len):
                prod *= num_grid[i + k][j + k]
                el_list.append(num_grid[i + k][j + k])
            if prod > max_prod:
                max_prod = prod
                pos = (i, j)
                orient = 2
                max_el_list = tuple(el_list)
            # Diagonal right
            prod = 1
            el_list = []
            for k in range(line_len):
                prod *= num_grid[i + k][j - k + line_len - 1]
                el_list.append(num_grid[i + k][j - k + line_len - 1])
            if prod > max_prod:
                max_prod = prod
                pos = (i, j + line_len)
                orient = 3
                max_el_list = tuple(el_list)
    res = {"max_product": max_prod, "start_position": pos,
            "orientation": orient, "elements": tuple(max_el_list)}
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

def largestLineProduct(num_grid: List[List[int]]=num_grid, line_len: int=4) -> int:
    """
    Solution to Project Euler #11

    For a 2D grid of integers num_grid, finds the line_len consecutive
    entries on a  horizontal, vertical or diagonal straight line in
    num_grid whose product is at least as large as any other such line.
    
    Args:
        Optional named:
        num_grid (list of lists of ints): The 2D grid of integers being
                considered. All of the lists in the main list must have
                the same length.
            Default: num_grid (as defined in Project_Euler_1_50.py)
        line_len (int): The number of integers in the lines in
                num_grid considered.
            Default: 4
    
    Returns:
    Integer (int) giving the value of the maximum product of line_len
    consecutive integers found
    """
    res = largestLineProductProperties(num_grid=num_grid, line_len=4)
    return res["max_product"]

# Problem 12
def triangleNDiv(target_ndiv: int=501):
    """
    Solution to Project Euler #12

    Finds the smallest triangle number to have at least target_ndiv
    positive integer divisors.
    
    Args:
        Optional named:
        target_ndiv (int): Non-negative integer giving the target
                number of divisors.
            Default: 501
    
    Returns:
    Integer (int) giving the smallest triangle number to have at least
    n_div positive divisors
    
    Outline of rationale:
    Utilises that the number of divisors function, tau, is
    multiplicative, i.e. for any coprime positive integers a, b:
        tau(a * b) = tau(a) * tau(b)
    and we note that for any positive integer n, n and n + 1 are
    coprime.
    Also utilises that tau(1) = 1, for any prime p and any natural
    number a tau(p^a) = a + 1, and that the formula
    for triangle numbers is n * (n + 1) / 2, so by separating this
    into either n / 2 and (n + 1) if n is even or n and (n + 1) / 2
    if n is odd, the tau(n / 2) or tau(n) as applicable will have
    already been calculated for the previous triangle number.
    """
    if target_ndiv <= 1: return 1
    #since = time.time()
    n1 = 1
    tau_even = 0
    tau_odd = 1
    i = 1
    while tau_even * tau_odd < target_ndiv:
        i += 1
        if i & 1 == 0:
            # From problem 3
            fact_dict = primeFactorisation(i + 1)
            tau_even = 1
            for v in fact_dict.values():
                tau_even *= v + 1
        else:
            # From problem 3
            fact_dict = primeFactorisation((i + 1) >> 1)
            tau_odd = 1
            for v in fact_dict.values():
                tau_odd *= v + 1
    res = (i * (i + 1)) >> 1
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
            
# Problem 13
num_list = (37107287533902102798797998220837590246510135740250,
        46376937677490009712648124896970078050417018260538,
        74324986199524741059474233309513058123726617309629,
        91942213363574161572522430563301811072406154908250,
        23067588207539346171171980310421047513778063246676,
        89261670696623633820136378418383684178734361726757,
        28112879812849979408065481931592621691275889832738,
        44274228917432520321923589422876796487670272189318,
        47451445736001306439091167216856844588711603153276,
        70386486105843025439939619828917593665686757934951,
        62176457141856560629502157223196586755079324193331,
        64906352462741904929101432445813822663347944758178,
        92575867718337217661963751590579239728245598838407,
        58203565325359399008402633568948830189458628227828,
        80181199384826282014278194139940567587151170094390,
        35398664372827112653829987240784473053190104293586,
        86515506006295864861532075273371959191420517255829,
        71693888707715466499115593487603532921714970056938,
        54370070576826684624621495650076471787294438377604,
        53282654108756828443191190634694037855217779295145,
        36123272525000296071075082563815656710885258350721,
        45876576172410976447339110607218265236877223636045,
        17423706905851860660448207621209813287860733969412,
        81142660418086830619328460811191061556940512689692,
        51934325451728388641918047049293215058642563049483,
        62467221648435076201727918039944693004732956340691,
        15732444386908125794514089057706229429197107928209,
        55037687525678773091862540744969844508330393682126,
        18336384825330154686196124348767681297534375946515,
        80386287592878490201521685554828717201219257766954,
        78182833757993103614740356856449095527097864797581,
        16726320100436897842553539920931837441497806860984,
        48403098129077791799088218795327364475675590848030,
        87086987551392711854517078544161852424320693150332,
        59959406895756536782107074926966537676326235447210,
        69793950679652694742597709739166693763042633987085,
        41052684708299085211399427365734116182760315001271,
        65378607361501080857009149939512557028198746004375,
        35829035317434717326932123578154982629742552737307,
        94953759765105305946966067683156574377167401875275,
        88902802571733229619176668713819931811048770190271,
        25267680276078003013678680992525463401061632866526,
        36270218540497705585629946580636237993140746255962,
        24074486908231174977792365466257246923322810917141,
        91430288197103288597806669760892938638285025333403,
        34413065578016127815921815005561868836468420090470,
        23053081172816430487623791969842487255036638784583,
        11487696932154902810424020138335124462181441773470,
        63783299490636259666498587618221225225512486764533,
        67720186971698544312419572409913959008952310058822,
        95548255300263520781532296796249481641953868218774,
        76085327132285723110424803456124867697064507995236,
        37774242535411291684276865538926205024910326572967,
        23701913275725675285653248258265463092207058596522,
        29798860272258331913126375147341994889534765745501,
        18495701454879288984856827726077713721403798879715,
        38298203783031473527721580348144513491373226651381,
        34829543829199918180278916522431027392251122869539,
        40957953066405232632538044100059654939159879593635,
        29746152185502371307642255121183693803580388584903,
        41698116222072977186158236678424689157993532961922,
        62467957194401269043877107275048102390895523597457,
        23189706772547915061505504953922979530901129967519,
        86188088225875314529584099251203829009407770775672,
        11306739708304724483816533873502340845647058077308,
        82959174767140363198008187129011875491310547126581,
        97623331044818386269515456334926366572897563400500,
        42846280183517070527831839425882145521227251250327,
        55121603546981200581762165212827652751691296897789,
        32238195734329339946437501907836945765883352399886,
        75506164965184775180738168837861091527357929701337,
        62177842752192623401942399639168044983993173312731,
        32924185707147349566916674687634660915035914677504,
        99518671430235219628894890102423325116913619626622,
        73267460800591547471830798392868535206946944540724,
        76841822524674417161514036427982273348055556214818,
        97142617910342598647204516893989422179826088076852,
        87783646182799346313767754307809363333018982642090,
        10848802521674670883215120185883543223812876952786,
        71329612474782464538636993009049310363619763878039,
        62184073572399794223406235393808339651327408011116,
        66627891981488087797941876876144230030984490851411,
        60661826293682836764744779239180335110989069790714,
        85786944089552990653640447425576083659976645795096,
        66024396409905389607120198219976047599490197230297,
        64913982680032973156037120041377903785566085089252,
        16730939319872750275468906903707539413042652315011,
        94809377245048795150954100921645863754710598436791,
        78639167021187492431995700641917969777599028300699,
        15368713711936614952811305876380278410754449733078,
        40789923115535562561142322423255033685442488917353,
        44889911501440648020369068063960672322193204149535,
        41503128880339536053299340368006977710650566631954,
        81234880673210146739058568557934581403627822703280,
        82616570773948327592232845941706525094512325230608,
        22918802058777319719839450180888072429661980811197,
        77158542502016545090413245809786882778948721859617,
        72107838435069186155435662884062257473692284509516,
        20849603980134001723930671666823555245252804609722,
        53503534226472524250874054075591789781264330331690)

def sumNumbers(
    num_list: Union[Tuple[int], List[int]]=num_list,
    n_digits: int=10,
    base: int=10
) -> int:
    """
    Solution to Project Euler #13

    Gives the first n_digits digits of the sum a list of
    integers, num_list when represented in the chosen base.
    
    Args:
        Optional named:
        num_list (list/tuple of ints): The list of integers to be
                summed.
            Default: num_lst (as defined in Project_Euler_1_50.py)
        n_digits (int): The number of the initial digits of the
                sum (when represented in the chosen base) to be
                returned.
            Default: 10
        base (int): The base used to express the sum.
            Default: 10
    
    Returns:
    String giving the first n_digits digits of the sum of the integers
    in num_list when expressed in the chosen base.
    If base exceeds 10 then the characters representing the digits for
    10, 11, ... are represented by a, b, ... respectively.
    """
    #since = time.time()
    num_sum = sum(num_list)
    digs = []
    while num_sum:
        num_sum, r = divmod(num_sum, base)
        digs.append(r)
    while len(digs) < n_digits: digs.append(0)
    ord_a = ord("a")
    def dig2Char(d: int) -> str:
        if d < 10: return str(d)
        return chr(d - 10 + ord_a)
    res = "".join([dig2Char(digs[~i]) for i in range(n_digits)])
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 14
def longestCollatzChainProperties(n_max: int=10 ** 6 - 1) -> Tuple[int, int]:
    """
    Solution to Project Euler #14

    Finds the strictly positive integer no greater than n_max whose
    Collatz chain is longer than that of any strictly positive integer
    smaller than it and at least as long as that of any strictly
    positive integer larger than it but no greater than n_max, and
    the length of the Collatz chain or that number.
    
    For any integer, its Collatz sequence takes its first term as
    that integer and every subsequent term is calculated from the
    previous term by dividing by 2 if that term is even and multiplying
    by 3 and adding 1 if that term is odd. The length of the Collatz
    chain for a given integer is the smallest positive integer m
    such that the mth term of the Collatz sequence of that integer
    is equal to 1.
    
    This assumes that the Collatz conjecture is true for all
    strictly positive integers no greater than n_max (i.e. that
    for every Collatz sequence starting at any number n_max, there
    exists an integer m such that the mth term in the Collatz
    sequence is 1, or alternatively that the length of the
    Collatz chain is defined for every strictly positive integer).
    
    Args:
        Optional named:
        n_max (int): The largest number considered.
            Default: 10 ** 6 - 1
    
    Returns:
    2-tuple whose index 0 contains the integer satisfying the
    given conditions and whose index 1 contains the length
    of the Collatz chain for that number.
    """
    #since = time.time()
    chain_lens = {}
    chain_lens[1] = 1
    l_chain = 0
    l_chain_no = 0
    for i in range(1, n_max + 1):
        chain = [i]
        # If the current number has already been encountered
        # it is already part of a larger chain and so cannot
        # be the largest.
        if chain_lens.get(i) is not None: continue
        while chain_lens.get(chain[-1]) is None:
            j = chain[-1]
            if j % 2 == 0:
                chain.append(j // 2)
            else:
                prov = 3 * j + 1
                chain.append(prov)
                chain.append(prov // 2)
        for k, el in enumerate(chain[::-1][1:]):
            #if el <= n_max:
            chain_lens[el] = chain_lens[chain[-1]] + k + 1
        if chain_lens[i] > l_chain:
            l_chain = chain_lens[i]
            l_chain_no = i
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return l_chain_no, l_chain

def longestCollatzChain(n_max: int=10 ** 6 - 1) -> int:
    """
    Solution to Project Euler #14

    Finds the strictly positive integer no greater than n_max whose
    Collatz chain is longer than that of any strictly positive integer
    smaller than it and at least as long as that of any strictly
    positive integer larger than it but no greater than n_max.
    
    For any integer, its Collatz sequence takes its first term as
    that integer and every subsequent term is calculated from the
    previous term by dividing by 2 if that term is even and multiplying
    by 3 and adding 1 if that term is odd. The length of the Collatz
    chain for a given integer is the smallest positive integer m
    such that the mth term of the Collatz sequence of that integer
    is equal to 1.
    
    This assumes that the Collatz conjecture is true for all
    strictly positive integers no greater than n_max (i.e. that
    for every Collatz sequence starting at any number n_max, there
    exists an integer m such that the mth term in the Collatz
    sequence is 1, or alternatively that the length of the
    Collatz chain is defined for every strictly positive integer).
    
    Args:
        Optional named:
        n_max (int): The largest number considered.
            Default: 10 ** 6 - 1
    
    Returns:
    Integer (int) giving the integer satisfying the conditions
    specified above.
    """
    res = longestCollatzChainProperties(n_max=n_max)
    return res[0]

# Problem 15
def countLatticePaths(r: int=20, d: int=20) -> int:
    """
    Solution to Project Euler #15

    Gives the number of distinct paths on a 2x2 grid with exactly
    with r steps to the right and d steps down.
    
    Args:
        Optional named:
        r (int): The number of steps to the right.
            Default: 20
        d (int): The number of steps down.
            Default: 20
    
    Returns:
    Integer (int) giving the number of distinct paths with exactly
    r steps to the right and d steps down.
    """
    #since = time.time()
    # This is simply (r+d) choose r
    n1 = max(r, d)
    n2 = min(r, d)
    sol = 1
    for i in range(n1 + 1, n1 + n2 + 1):
        sol *= i
    for i in range(1, n2+1):
        sol //= i
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return sol
    
# Problems 16 and 20
def digitSum(num: int=2 ** 1000, base: int=10) -> int:
    """
    Solution to Project Euler #16

    Calculates the sum of the digits of the non-negative integer num
    when expressed in the chosen base.
    
    Args:
        Optional named:
        num (int): The non-negative integer whose digits in its
                representation in the chosen base are being summed.
            Default: 2 ** 1000
        base (int): The base in which num is to be represented.
            Default: 10
    
    Returns:
    Integer (int) giving the sum of the digits of the representation of
    num in the chosen base.
    """
    #since = time.time()
    res = 0
    while num:
        num, r = divmod(num, base)
        res += r
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 17
def freqDictCombine(f_dict1, f_dict2):
    """
    Combines two dictionaries that represent frequency counts
    (i.e. of the form {k1: freq1, k2: freq2, ..., kn: freqn}
    where freq1, freq2, ..., freqn are natural numbers),
    summing the frequencies of any shared key.
    
    Args:
        Required positional
        f_dict1 (dict): The first of the frequency dictionaries
                being combined
        f_dict2 (dict): The second of the frequency dictionaries
                being combined
    
    Returns:
    Dictionary (dict) representing the combined frequency
    counts of the keys of f_dict1 and f_dict2.
    """
    out_dict = copy.deepcopy(f_dict1)
    for k, v in f_dict2.items():
        if k in out_dict.keys():
            out_dict[k] += v
        else: out_dict[k] = v
    return out_dict

def numberWordFrequency(n_max: int) -> Dict[str, int]:
    """
    Calculates the total number of occurrences of each word making up
    all integers from 1 to n_max inclusive when expressed in English.
    
    Args:
        Required positional:
        n_max (int): Strictly positive integer less than 10 ** 15
                (one quadrillion) up to which the words making up
                the initial integers (starting at 1) are to be counted.
    
    Returns:
    Dictionary (dict) whose keys are the words occurring in the
    English expression of all integers from 1 to n_max inclusive
    and whose corresponding values are the frequency with which
    the corresponding word occurs in the English expression of
    all those integers.
    """
    digits = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                6: "six", 7: "seven", 8: "eight", 9: "nine"}
    teens = {10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
                14: "fourteen", 15: "fifteen", 16: "sixteen",
                17: "seventeen", 18: "eighteen", 19: "nineteen"}
    tens = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
                60: "sixty", 70: "seventy", 80: "eighty",
                90: "ninety"}
    powers_of_ten = {2: "hundred", 3: "thousand", 6: "million",\
                    9: "billion", 12: "trillion"}
        
    def underHundred(n=99, top_level=True):
        # n < 100
        if n == 0: return {}
        # Working out units
        d, rem = divmod(n, 10)
        out_dict = {v: d + (k <= rem) - (d + (k <= rem) >= 2)\
                        for k, v in digits.items()}
        # Working out teens
        out_dict = freqDictCombine(out_dict,\
                    {v: int(k <= n) for k, v in teens.items()})
        # Working out tens
        if d > 1:
            tens_dict = {v: 10 for k, v in tens.items() if\
                            k <= 10 * (d - 1)}
            tens_dict[tens[10 * d]] = rem + 1
            out_dict = freqDictCombine(out_dict, tens_dict)
        for k in list(out_dict.keys()):
            if out_dict[k] == 0: out_dict.pop(k)
        return out_dict

    def hundredPlus(n, top_level=True):
        # Any strictly positive n
        level = 2 if len(str(n)) == 3 else ((len(str(n)) - 1) // 3) * 3
        #print(level)
        if level == 0:
            return underHundred(n)
        elif level == 2:
            next_func = underHundred
        else:
            next_func = hundredPlus
        if level not in powers_of_ten.keys():
            raise ValueError("n too large")
        
        d, rem = divmod(n, 10 ** level)
        out_dict = next_func(10 ** level - 1, top_level=False)
        #print(out_dict)
        for k in out_dict.keys():
            out_dict[k] *= d
        out_dict = freqDictCombine(out_dict,\
                        next_func(rem, top_level=False))
        n_level = max(0, n - 10 ** level + 1)
        level_units0 = hundredPlus(d - 1, top_level=True)
        #print(level_units0)
        level_units = copy.deepcopy(level_units0)
        n_and = n - 99 - (n // 100) if top_level else 0
        #if top_level: print(f"n_and = {n_and}")
        for k in level_units.keys():
            level_units[k] *= 10 ** level
        extra_dict = {k: (v - level_units0.get(k, 0)) * (rem + 1)\
                for k, v in hundredPlus(d, top_level=True).items()}
        #print(extra_dict)
        
        level_units = freqDictCombine(level_units, extra_dict)
        out_dict = freqDictCombine(out_dict, {"and": n_and})
        out_dict = freqDictCombine(out_dict,\
                    {powers_of_ten[level]: n_level, **level_units})
        for k in list(out_dict.keys()):
            if out_dict[k] == 0: out_dict.pop(k)
        return out_dict
    
    return hundredPlus(n_max, top_level=True)

def numberLetterCount(n_max: int=1000) -> int:
    """
    Solution to Project Euler #17

    Calculates the total number of letters used when writing out all
    integers from 1 to n_max inclusive in English, excluding spaces
    and hyphens but including and (e.g. 101 as one hundered and one
    is counted as 17 letters).
    
    Args:
        Required positional:
        n_max (int): Strictly positive integer less than 10 ** 15
                (one quadrillion) up to which the words making up
                the initial integers (starting at 1) are to be counted.
    
    Returns:
    Integer(int) giving the total number of letters used when writing
    out all integers from 1 to n_max inclusive in English, subject
    to the rules given above.
    """
    #since = time.time()
    str_freq_dict = numberWordFrequency(n_max)
    res = 0
    for k, v in str_freq_dict.items():
        res += len(k) * v
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
    
# Problems 18
triangle1 = ["75",
            "95 64",
            "17 47 82",
            "18 35 87 10",
            "20 04 82 47 65",
            "19 01 23 75 03 34",
            "88 02 77 73 07 63 67",
            "99 65 04 28 06 16 70 92",
            "41 41 26 56 83 40 80 70 33",
            "41 48 72 33 47 32 37 16 94 29",
            "53 71 44 65 25 43 91 52 97 51 14",
            "70 11 33 28 77 73 17 78 39 68 17 57",
            "91 71 52 38 17 14 91 43 58 50 27 29 48",
            "63 66 04 68 89 53 67 30 73 16 69 87 40 31",
            "04 62 98 27 23 09 70 98 73 93 38 53 60 04 23"]

def loadTriangle(doc: str, rel_package: bool=False) -> List[str]:
    """
    Loads triangle of integers from .txt file located at doc.
    The file should contain the rows of the triangle in order,
    separated by line breaks ('\\n') and the integers in each
    row separated by single spaces. For rows labelled in order
    starting from 0, for any non-negative integer i less than the
    number of rows of the triangle, the ith row must contain exactly
    (i + 1) integers.
    
    Args:
        Required positional:
        doc (str): The relative or absolution location of the .txt
                file containing the triangle of integers.

        Optional named:
        rel_package (bool): Whether a relative path given by doc
                is relative to the current directory (False) or
                the package directory (True).
            Default: False
    
    Returns:
    List of strings (str), with each entry in the list representing
    a row of the triangle, and for each row the integers are
    separated by single spaces.
    """
    doc = doc.strip()
    if rel_package and not doc.startswith("/"):
        doc = os.path.join(__file__, doc)
    with open(doc) as f:
        txt = f.read()
    res = txt.split("\n")
    while not res[-1]: res.pop()
    return res

def triangleMaxSum(
    triangle: List[List[int]]=triangle1,
    preserve_triangle: bool=True
) -> int:
    """
    Solution to Project Euler #18

    For a given triangle of integers, (arranged as a grid where the
    first row has length 1 and every other row has length exactly one
    more than the previous row) finds the maximum total that can be
    achieved by travelling along a path from the top to the bottom,
    starting on the single element of the first row and each step
    moving to the next row, to either the element of that row with
    the same index as the index on the current row or the element
    with index one greater, and taking the total of the numbers of
    all elements of the triangle this path encounters.
    
    Args:
        Optional named:
        triangle (str, list of strs or list of list of ints): Either
                a string giving the path to a .txt file containing
                the triangle of integers (where the rows are separated
                by line breaks- '\\n'- and the integers in each row
                are separated by single spaces), or a representation
                of the triangle directly. This representation may
                either be as a list of strings (where each entry
                represents a row and the integers of each row are
                separated by single spaces), or as a list of lists
                of integers (where the lists in the main list are the
                rows of the triangle).
                For rows labelled in order starting from 0, for any
                non-negative integer i less than the number of rows of
                the triangle, the ith row must contain exactly (i + 1)
                integers.
            Default: triangle1
        preserve_triangle (bool): If True, requires that the object given
                for the fellow input argument triangle should not be
                altered by this process. Otherwise, allow this object
                to be altered.
            Default: True
        
    Returns:
    The maximum sum (int) that can be achieved by any traversal
    of the triangle following the restrictions given.
    """
    #since = time.time()
    if isinstance(triangle, str):
        triangle = loadTriangle(triangle)
        preserve_triangle = False
    # Converting the triangle into lists of integers
    if isinstance(triangle[0], str):
        if preserve_triangle:
            triangle = list(triangle)
        for i, row in enumerate(triangle):
            triangle[i] = [int(x) for x in row.strip().split(" ")]
    elif preserve_triangle or not isinstance(triangle, list) or\
            any(not isinstance(x, list) for x in triangle):
        triangle = [list(x) for x in triangle]
    n = len(triangle)
    for i in reversed(range(n - 1)):
        for j in range(i + 1):
            triangle[i][j] += max(triangle[i + 1][j],
                    triangle[i + 1][j + 1])
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return triangle[0][0]
    
# Problem 19
def monthLength(month_no: int, year: int) -> int:
    """
    For a given month number (i.e. Jan = 1, Feb = 2, etc.) and year,
    finds the number of days in that month
    
    Args:
        Required positional:
        month_no (int): The number of the month (Jan = 1, Feb = 2 etc.)
                from 1 to 12 inclusive
        year (int): The year A.D.
    
    Returns:
    Integer (int) giving the number of days in the specified month.
    """
    if month_no == 2:
        return 28 + (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    elif month_no in {1, 3, 5, 7, 8, 10, 12}:
        return 31
    elif month_no in {4, 6, 9, 11}:
        return 30
    else:
        ValueError("The month number must be an integer from "
                        "1 to 12")

def countDaysBetweenYears(start_year=1900, end_year=1901) -> int:
    """
    For given start year start_year and end year end_year, finds the number
    of days (24 hour periods) between midnight on the morning of 1st Jan
    start_year and midnight in the morning of 1st Jan end_year.
    
    Args:
        Optional named:
        start_year (int): The year (A.D.) the day count starts.
            Default: 1900
        end_year (int): The year (A.D.) the day count ends.
            Default: 1901
    
    Returns:
    Integer giving the number of days as described above.
    """
    diff = end_year - start_year
    return 365 * diff + (end_year - 1) // 4 - (start_year - 1) // 4\
            - (end_year - 1) // 100 + (start_year - 1) // 100\
            + (end_year - 1) // 400 - (start_year - 1) // 400
    

def countMonthsStartingDoW(
    day: int=0,
    start_year: int=1901,
    end_year: int=2000,
) -> int:
    """
    Solution to Project Euler #19

    Gives the number of months between start_year and end_year
    (inclusive) starting on a given day of the week (where 0 = Sunday,
    1 = Monday, ..., 6 = Saturday).
    
    Args:
        Optional named:
        day (int): The day of the week counted (0 = Sunday, 1 = Monday,
                ..., 6 = Saturday).
            Default: 0 (Sunday)
        start_year (int): The first year considered.
            Default: 1901
        end_year (int): The last year considered. Should be no less
                than start_year.
            Default: 2000
    
    Returns:
    The number of months between start_year and end_year inclusive
    starting on the given day of the week.
    """
    #since = time.time()
    # Jan 1st 1900 was a Monday
    zero_day = 1
    zero_month = 1
    zero_year = 1900
    
    curr_day = (zero_day + countDaysBetweenYears(zero_year, start_year)) % 7
    #print(curr_day)
    curr_year = start_year
    curr_month = zero_month
    count = 0
    while curr_year <= end_year:
        curr_day = (curr_day + monthLength(curr_month, curr_year)) % 7
        if curr_day == day: count += 1
        curr_month += 1
        if curr_month == 13:
            curr_month = 1
            curr_year += 1
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return count
        
# Problem 20- See Problem 16 (use digitSum(math.factorial(100)))

# Problem 21
def factorSum(num: int, proper: bool=False) -> int:
    """
    Finds the sum of the positive factors of an integer num (if
    specified, proper factors).
    
    Args:
        Required positional:
        num (int): The number whose sum of factors is to be calculated.
        
        Optional named:
        proper (bool): If True, then the sum only includes proper
                factors (i.e. excludes num itself from the factors
                summed), otherwise includes all factors.
            Default: False
    
    Returns:
    Integer (int) giving the sum of the factors of num, or proper
    factors of num if argument proper given as True.
    
    Outline of rationale:
    Uses the multiplicity property of the sum of factors function
    sigma, i.e. for any coprime integers a, b:
        sigma(a * b) = sigma(a) * sigma(b)
    and that if p1 and p2 are distinct primes then for any natural
    numbers a, b, p1^a and p2^b are coprime. Furthermore,
        sigma(p1^a) = (p^(a + 1) - 1) / (p - 1)
    """
    # From Problem 3
    factor_dict = primeFactorisation(num)
    sigma = 1
    for k, v in factor_dict.items():
        sigma *= (k ** (v + 1) - 1) // (k - 1)
    return sigma - num if proper else sigma

def findAmicablePairs(n_max: int) -> Set[Tuple[int]]:
    """
    A pair of distinct strictly positive integers is an amicable
    pair if and only if the sum of proper factors of each of the
    pair is equal to the value of the other.
    This function finds all amicable pairs such that at least one
    of the pair is no greater than n_max.
    
    Args:
        Required positional:
        n_max (int): The largest value for which the smaller value
                of an amicable pair can take for any amicable pair
                considered.
    
    Returns:
    Set of 2-tuples of ints containing all amicable pairs such that
    at least one of the pair is no greater than n_max, where each
    pair has the smaller value first.
    """
    factsum_dict = {1: 0}
    amicable_set = set()
    for i in range(2, n_max + 1):
        if i in factsum_dict.keys(): continue
        factsum_dict[i] = factorSum(i, proper=True)
        if factsum_dict[i] == i: continue
        if factsum_dict[i] not in factsum_dict.keys():
            factsum_dict[factsum_dict[i]] =\
                    factorSum(factsum_dict[i], proper=True)
        if factsum_dict[factsum_dict[i]] == i:
            amicable_set.add((i, factsum_dict[i]))
    return amicable_set

def amicableNumbersSum(n_max: int=10000) -> int:
    """
    Solution to Project Euler #21

    A strictly positive integer is amicable if and only if it belongs
    to an amicable pair. A pair of distinct strictly positive integers
    is an amicable pair if and only if the sum of proper factors of
    each of the pair is equal to the value of the other.
    This function finds the sum of all amicable numbers not exceeding
    n_max.
    
    Args:
        Optional named:
        n_max (int): The largest value of amicable numbers considered.
            Default: 10000
    
    Returns:
    Integer (int) giving the sum of all amicable numbers not exceeding
    n_max.
    """
    #since = time.time()
    res = sum(x[0] + (x[1] if x[1] <= n_max else 0)\
            for x in findAmicablePairs(n_max))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 22
def letterAlphabetValue(char: str, a_val: int=1) -> int:
    """
    For a single letter (i.e. a string of length 1 with an
    alphabet character, either upper or lower case) char, determines
    its numeric value, where a and A have value a_val, b and B have
    value a_val + 1, ..., and z or Z have value a_val + 26.
    
    Args:
        Required positional:
        char (str): The letter in question. Can only be a single
                alphabet character (i.e. a, b, c, ..., y or z, or the
                upper case version and len(char) is exactly 1)
    
        Optional named:
        a_val (int): The numerical value of a or A, determining the
                offset for the value of every other letter.
            Default: 1
    
    Returns:
    Integer (int) giving the numerical value of the letter char for the
    given a_val
    """
    if len(char) != 1 or not char.isalpha():
        raise ValueError("The input char is not a single alphabet "
                "character as is required.")
    return ord(char.lower()) - ord("a") + a_val
    
    #alphabet = "abcdefghijklmnopqrstuvwxyz"
    #char = char.strip().lower()
    #for i, char2 in enumerate(alphabet, start=1):
    #    if char == char2: return i
    #else:
    #    raise ValueError("The input char is not a single alphabet "
    #            "character as is required.")

#def alphabetPairSort(word1: str, word2: str):
#    """
#    For two words (strings of alphabet characters, where case is
#    ignored) word1 and word2, sorts them alphabetically.
#    
#    Args:
#        Required positional:
#        word1 (str): The first word being sorted. Should only contain
#                alphabet characters (case is ignored).
#        word2 (str): The second word being sorted. Should only contain
#                alphabet characters (case is ignored).
#    
#    Returns:
#    2-tuple of strs, (word3, word4) where word3 and word4 are the
#    smaller and larger alphabetically respectively of word1 and word2.
#    """
#    word1 = word1.strip()
#    word2 = word2.strip()
#    for i in range(min(len(word1), len(word2))):
#        l1 = word1[i].lower()
#        l2 = word2[i].lower()
#        v1, v2 = list(map(letterAlphabetValue, (l1, l2))
#        if l1 < l2: return (word1, word2)
#        elif l1 > l2: return (word2, word1)
#    return (word1, word2) if len(word1) <= len(word2)\
#            else (word1, word2)

def alphabetComparitor(word1: str, word2: str) -> int:
    """
    For two words (strings of alphabet characters, where case is
    ignored) word1 and word2, determines the alphabetical ordering
    of the two (giving -1 if word1 comes before word2 alphabetically,
    1 if word2 comes after word1 and otherwise 0 (i.e. when ignoring
    case, word1 is equal to word2).
    Note that this defines a total preorder on the set of strings
    containing only alphabet characters, since for any strings a, b
    and c (where for strings a and b, a <= b denotes that a does not
    come after b in the ordering and b >= a has the same meaning),
    a <= b and b <= c implies a <= c (transitivity) and at least one
    of a <= b or a >= b is true (strong connectedness).
    
    Args:
        Required positional:
        word1 (str): The first word being compared. Should only contain
                alphabet characters (case is ignored).
        word2 (str): The second word being compared. Should only
                contain alphabet characters (case is ignored).
    
    Returns:
    Integer (int) giving -1 if word1 comes before word2 alphabetically,
    1 if word1 comes after word2 alphabetically, and otherwise 0
    """
    word1 = word1.strip()
    word2 = word2.strip()
    for i in range(min(len(word1), len(word2))):
        l1 = word1[i].lower()
        l2 = word2[i].lower()
        v1, v2 = list(map(letterAlphabetValue, (l1, l2)))
        if l1 == l2: continue
        return -1 if (l1 < l2) else 1
    d = len(word1) - len(word2)
    if not d: return 0
    return -1 if d < 0 else 1

def pairSort(
    pair: List[int],
    comp_func: Callable[[Any, Any], int]
) -> Tuple[int]:
    
    return tuple(pair) if comp_func(*pair) <= 0 else tuple(pair[::-1])

def bubbleSort(
    obj_list: List[Any],
    comp_func: Callable[[Any, Any], int]=alphabetComparitor,
) -> Tuple[Any]:
    """
    For a list of objects, obj_list, from which any two elements
    can be ordered using the comparitor function comp_func that defines
    a total preorder on the set of objects in this list (i.e. for any
    elements of the set a, b and c, a <= b and b <= c implies a <= c -
    transitivity and at least one of a <= b or b <= a is true - strong
    connectedness), performs a bubble sort.
    
    Args:
        Required positional:
        obj_list (list/tuple): The list of objects to be sorted.
        
        Optional named:
        comp_func (function): Comparitor function that defines a
                total preorder on the set of objects in obj_list.
                For any two elements a and b of the set of objects in
                obj_list, comp_func(a, b) = -1 if the comparitor
                assesses that a <= b but not b >= a,
                comp_func(a, b) = 1 if the comparitor assesses that
                a >= b but not b >= a and comp_func(a, b) = 0
                otherwise (i.e. both a >= b and a <= b).
            Default: alphabetPairSort
    
    Returns:
    Tuple containing the same objects as obj_list in sorted order
    based on the comparitor function comp_func.
    """
    #print("Using bubbleSort()")
    obj_list_prev = tuple(obj_list)
    last_swap_prev = len(obj_list) - 1
    prev_swaps = set(range(len(obj_list) - 1))
    first = True
    count = 0
    while True:
        if count % 500 == 0:
            print(count)
        count += 1
        obj_list = list(obj_list_prev)
        last_swap = None
        swaps = set()
        for i in range(last_swap_prev):
            if (i + 2 < len(obj_list) and i + 1 not in prev_swaps)\
                    and (i >= 1 or i - 1 not in swaps):
                continue
            obj_list[i: i + 2] = pairSort(obj_list[i: i + 2],\
                    comp_func=alphabetComparitor)
            if obj_list[i + 1] != obj_list_prev[i + 1]:
                last_swap = i
                swaps.add(i)
        obj_list = tuple(obj_list)
        if len(swaps) == 0: break
        obj_list_prev = obj_list
        last_swap_prev = last_swap
        prev_swaps = swaps
        first = False
    return obj_list

def mergeSort(
    obj_list: List[Any],
    comp_func: Callable[[Any, Any], int]=alphabetComparitor
) -> Tuple[Any]:
    """
    For a list of objects, obj_list, from which any two elements
    can be ordered using the comparitor function comp_func that defines
    a total preorder on the set of objects in this list (i.e. for any
    elements of the set a, b and c, a <= b and b <= c implies a <= c -
    transitivity and at least one of a <= b or b <= a is true - strong
    connectedness), performs a merge sort.
    
    Args:
        Required positional:
        obj_list (list/tuple): The list of objects to be sorted.
        
        Optional named:
        comp_func (function): Comparitor function that defines a
                total preorder on the set of objects in obj_list.
                For any two elements a and b of the set of objects in
                obj_list, comp_func(a, b) = -1 if the comparitor
                assesses that a <= b but not b >= a,
                comp_func(a, b) = 1 if the comparitor assesses that
                a >= b but not b >= a and comp_func(a, b) = 0
                otherwise (i.e. both a >= b and a <= b).
            Default: alphabetComparitor
    
    Returns:
    Tuple containing the same objects as obj_list in sorted order
    based on the comparitor function comp_func.
    """
    #print("Using mergeSort()")
    def mergeSortRecur(obj_list):
        if len(obj_list) == 1:
            return obj_list
        split_i = len(obj_list) // 2
        list1 = mergeSortRecur(obj_list[:split_i])
        list2 = mergeSortRecur(obj_list[split_i:])
        out_list = []
        i1 = 0
        el1 = list1[i1]
        finished_list1 = False
        for i2, el2 in enumerate(list2):
            #print(comp_func(el1, el2))
            while comp_func(el1, el2) <= 0:
                out_list.append(el1)
                i1 += 1
                if i1 >= len(list1):
                    finished_list1 = True
                    break
                el1 = list1[i1]
            out_list.append(el2)
            if finished_list1:
                break
        out_list.extend(list1[i1:])
        out_list.extend(list2[i2 + 1:])
        return tuple(out_list)
    return mergeSortRecur(obj_list)

def quickSort(
    obj_list: List[Any],
    comp_func: Callable[[Any, Any], int]=alphabetComparitor
) -> Tuple[Any]:
    """
    For a list of objects, obj_list, from which any two elements
    can be ordered using the comparitor function comp_func that defines
    a total preorder on the set of objects in this list (i.e. for any
    elements of the set a, b and c, a <= b and b <= c implies a <= c -
    transitivity and at least one of a <= b or b <= a is true - strong
    connectedness), performs a Quicksort.
    
    Args:
        Required positional:
        obj_list (list/tuple): The list of objects to be sorted.
        
        Optional named:
        comp_func (function): Comparitor function that defines a
                total preorder on the set of objects in obj_list.
                For any two elements a and b of the set of objects in
                obj_list, comp_func(a, b) = -1 if the comparitor
                assesses that a <= b but not b >= a,
                comp_func(a, b) = 1 if the comparitor assesses that
                a >= b but not b >= a and comp_func(a, b) = 0
                otherwise (i.e. both a >= b and a <= b).
            Default: alphabetComparitor
    
    Returns:
    Tuple containing the same objects as obj_list in sorted order
    based on the comparitor function comp_func.
    """
    #print("Using quickSort()")
    def quickSortRecur(obj_list):
        if len(obj_list) <= 1:
            return obj_list
        pivot_i = random.randrange(len(obj_list))
        pivot = obj_list[pivot_i]
        list_gt = []
        list_lt = []
        list_eq = []
        for el in obj_list:
            if el == pivot: list_eq.append(el)
            elif comp_func(pivot, el) <= 0:
                list_gt.append(el)
            else:
                list_lt.append(el)
        list_gt = quickSortRecur(list_gt)
        list_lt = quickSortRecur(list_lt)
        return [*list_lt, *list_eq, *list_gt]
    return quickSortRecur(obj_list)


class ComparitorWrapperFactory:
    """
    Factory class that wraps objects with ComparitorWrapper for a
    given custom comparitor.
    
    Initialisation args:
        Required positional:
        comp_func (function): Defines the attribute comp_func for this
                instance (see comp_func attribute) which is to be used
                by two ComparitorWrapper instances created by this
                ComparitorWrapperFactory object to compare the objects
                wrapped.
        
    Attributes:
        comp_func (function): Function which is to be used by two
                ComparitorWrapper instances created by this
                ComparitorWrapperFactory object to compare the objects
                wrapped.
                This comparitor should represent a total preorder on
                the set of objects wrapped by this
                ComparitorWrapperFactory object (i.e. for any elements
                of the set a, b and c, a <= b and b <= c implies
                a <= c - transitivity and at least one of a <= b or
                b <= a is true - strong connectedness). For any two
                elements a and b of the set of objects to be wrapped,
                comp_func(a, b) = -1 if the comparitor assesses that
                a <= b but not b >= a, comp_func(a, b) = 1 if the
                comparitor assesses that a >= b but not b >= a and
                comp_func(a, b) = 0 otherwise (i.e. both a >= b and
                a <= b).
    
    Methods:
            (see documentation for methods themselves for more detail)
        wrapObject(): Takes an object that may be used as either
                argument for the comparitor function attribute
                comp_func() and returns that object wrapped inside an
                instance of ComparitorWrapper.
    """
    def __init__(self, comp_func: Callable[[Any, Any], int]):
        self.comp_func = comp_func
    
    def wrapObject(self, obj: Any):
        """
        Wraps the object obj that may be used as either argument for
        the comparitor function attribute comp_func() and returns
        that object wrapped inside an instance of ComparitorWrapper.
        
        Args:
            Required positional:
            obj (object): An object that may be used as either argument
                    for the comparitor function attribute comp_func()
                    that is to be wrapped inside an instance of
                    ComparitorWrapper.
        
        Returns:
        ComparitorWrapper object containing obj, where the attribute
        obj is the argument obj and the attribute comp_func is the
        comparison function attribute of this ComparisonWrapperFactory
        instance comp_func.
        """
        return ComparitorWrapper(obj, self.comp_func)
    
class ComparitorWrapper:
    """    
    Wrapper of objects that enables custom comparisons between the
    wrapped objects.
    Uses a comparitor function comp_func() (see below) which defines
    a custom comparitor representing a total preorder on the set of
    objects wrapped in ComparitorWrapper objects using this comp_func
    comparitor (i.e. for any elements of the set a, b and c, a <= b and
    b <= c implies a <= c - transitivity and at least one of a <= b or
    b <= a is true - strong connectedness) to define the comparison
    dunder method between two objects of this set __lt__() (i.e. <) and
    __le__() (i.e. <=) directly and through these __gt__() (i.e. >)
    and __ge__() (i.e. >=) indeirectly.
    Note that two objects of this class are considered equal if and
    only if the object wrapped and the comparison function it uses
    are both equal between the two objects.
    
    Initialisation args:
        Required positional:
        obj (object): The object to be wrapped in ComparitorWrapper.
                Should be able to be used as either argument in the
                comparitor function argument comp_func().
        comp_func (function): Defines the attribute comp_func for this
                instance (see comp_func attribute) which is to be used
                to compare the object wrapped in this ComparitorWrapper
                object with another ComparitorWrapper object with the
                same comp_func attribute.
        
    Attributes:
        obj (object): The object wrapped in this ComparitorWrapper
                instance, which uses fellow attribute comp_func to
                compare this object with other objects wrapped in
                ComparitorWrapper instances with the same comp_func
                attribute.
        comp_func (function): Function which is to be used to compare
                the object wrapped in this ComparitorWrapper object
                with another ComparitorWrapper object with the same
                comp_func attribute.
                This comparitor should represent a total preorder on
                the set of objects wrapped in ComparitorWrapper objects
                with this comp_func attribute (i.e. for any elements of
                the set a, b and c, a <= b and b <= c implies
                a <= c - transitivity and at least one of a <= b or
                b <= a is true - strong connectedness). For any two
                objects a and b wrapped in ComparitorWrapper instances
                with the same comp_func attribute, comp_func(a, b) = -1
                if the comparitor assesses that a <= b but not b >= a,
                comp_func(a, b) = 1 if the comparitor assesses that
                a >= b but not b >= a and comp_func(a, b) = 0 otherwise
                (i.e. both a >= b and a <= b).
    """
    def __init__(self, obj: Any, comp_func: Callable[[Any, Any], int]):
        self.obj = obj
        self.comp_func = comp_func
    
    def __lt__(self, other: Any):
        if not isinstance(other, ComparitorWrapper) and\
                self.comp_func == other.comp_func:
            raise TypeError("The object being compared to must be of "
                    "type ComparitorWrapper() with the same comparitor.")
        return self.comp_func(self.obj, other.obj) == -1
    
    def __le__(self, other: Any):
        if not isinstance(other, ComparitorWrapper) and\
                self.comp_func == other.comp_func:
            raise TypeError("The object being compared to must be of "
                    "type ComparitorWrapper() with the same comparitor.")
        return self.comp_func(self.obj, other.obj) <= 0
    
    def __eq__(self, other: Any):
        if not isinstance(other, ComparitorWrapper) and\
                self.comp_func == other.comp_func:
            return False
        return self.obj == other.obj

def heapSort(
    obj_list: List[Any],
    comp_func: Callable[[Any, Any], int]=alphabetComparitor,
) -> Tuple[Any]:
    """
    For a list of objects, obj_list, from which any two elements
    can be ordered using the comparitor function comp_func that defines
    a total preorder on the set of objects in this list (i.e. for any
    elements of the set a, b and c, a <= b and b <= c implies a <= c -
    transitivity and at least one of a <= b or b <= a is true - strong
    connectedness), performs a heap sort.
    
    Args:
        Required positional:
        obj_list (list/tuple): The list of objects to be sorted.
        
        Optional named:
        comp_func (function): Comparitor function that defines a
                total preorder on the set of objects in obj_list.
                For any two elements a and b of the set of objects in
                obj_list, comp_func(a, b) = -1 if the comparitor
                assesses that a <= b but not b >= a,
                comp_func(a, b) = 1 if the comparitor assesses that
                a >= b but not b >= a and comp_func(a, b) = 0
                otherwise (i.e. both a >= b and a <= b).
            Default: alphabetComparitor
    
    Returns:
    Tuple containing the same objects as obj_list in sorted order
    based on the comparitor function comp_func.
    """
    #print("Using heapSort()")
    comparitor_factory = ComparitorWrapperFactory(comp_func)
    obj_list = [comparitor_factory.wrapObject(x) for x in obj_list]
    heapq.heapify(obj_list)
    out_list = []
    while len(obj_list) > 0:
        out_list.append(heapq.heappop(obj_list))
        #print(out_list)
    return [x.obj for x in out_list]
    #return out_list
    
def wordsSort(
    word_list: Union[List[str], Tuple[str]],
    sort_func: Callable[[List[Any], Callable[[Any, Any], Tuple[Any]]], Tuple[Any]]=None,
    comp_func: Callable[[Any, Any], int]=alphabetComparitor,
) -> Tuple[str]:
    """
    Sorts a list of words (strings of alphabet characters where case
    is ignored) word_list based on the sorting function sort_func (or,
    if sort_func is None then on the in-build Python sorted()
    function), using the comparitor function comp_func that defines a
    total preorder on the set of words in this list (i.e. for any
    elements of the set a, b and c, a <= b and b <= c implies a <= c -
    transitivity and at least one of a <= b or b <= a is true - strong
    connectedness).
    
    Args:
        Required positional:
        word_list (list/tuple of strs): The list of words to be sorted.
        
        Optional named:
        sort_func (Function or None): The algorithm used to sort the
                words in word_list. If not given or given as None,
                uses the inbuilt sorted() function.
            Default: None
        comp_func (function): Comparitor function that defines a
                total preorder on the set of words in word_list.
                For any two elements a and b of the set of words in
                word_list, comp_func(a, b) = -1 if the comparitor
                assesses that a <= b but not b >= a,
                comp_func(a, b) = 1 if the comparitor assesses that
                a >= b but not b >= a and comp_func(a, b) = 0
                otherwise (i.e. both a >= b and a <= b).
            Default: alphabetComparitor
    
    Returns:
    Tuple of strs containing the elements of word_list in sorted
    order based on the comparitor function comp_func.
    """
    if sort_func is None:
        return tuple(sorted(word_list, key=functools.cmp_to_key(comp_func)))
    return tuple(sort_func(word_list, comp_func))

def nameScore(word: str) -> int:
    """
    For word, a string of alphabet characters (either upper or lower
    case), determines its name score (the sum of the position in
    the alphabet of each of its letters ignoring case, starting
    at 1 for a).
    
    Args:
        word (str): The string whose name score is to be calculated.
                May only contains alphabet characters (upper or
                lower case are allowed for any of the characters).
    
    Returns:
    Integer (int) giving the name score of word.
    """
    return sum(letterAlphabetValue(x) for x in word.strip())

def nameListScore(
    word_list: List[str],
    sort_func: Optional[Callable[[List[Any], Callable[[Any, Any], Tuple[Any]]], Tuple[Any]]]=None,
    comp_func: Callable[[Any, Any], int]=alphabetComparitor
) -> int:
    """
    For a list of words (strings of alphabet characters, either
    upper or lower case) word_list and a comparitor function, comp_func
    that defines a total preorder on the set of words in this list
    (i.e. for any elements of the set a, b and c, a <= b and b <= c
    implies a <= c - transitivity and at least one of a <= b or b <= a
    is true - strong connectedness), calculates the combined name score
    of this list of words for this comparitor function.
    The combined name score for a given list of words for a given
    comparitor function that defines a preorder on the set of words
    in that list is defined to be the sum over the values given for
    each word by the product of the word's name score (see nameScore
    for the definition of the name score) and its position in the
    ordering of the list of words based on the given comparitor
    function (where the first element in the ordering has position 1
    and each subsequent element in the ordering has position one
    greater than the previous element in the ordering).
    The sorting is performed by sort_func() if this is given and
    not given as None, otherwise uses the inbuild sorted() function.
    
    Args:
        Required positional:
        word_list (list of strs): The list of words whose combined
                name score is to be calculated. Each element in
                word_list may only contain alphabet characters (both
                upper and lower case are allowed for any of the
                characters in any of the words).
        
        Optional named:
        sort_func (Function or None): The algorithm used to sort the
                words in word_list. If not given or given as None,
                uses the inbuilt sorted() function.
            Default: None
        comp_func (function): Comparitor function that defines a
                total preorder on the set of words in word_list.
                For any two elements a and b of the set of words in
                word_list, comp_func(a, b) = -1 if the comparitor
                assesses that a <= b but not b >= a,
                comp_func(a, b) = 1 if the comparitor assesses that
                a >= b but not b >= a and comp_func(a, b) = 0
                otherwise (i.e. both a >= b and a <= b).
            Default: alphabetComparitor
    
    Returns:
    Integer (int) giving the combined name score for word_list (as
    defined above) for the comparitor function comp_func.
    """
    ordering = wordsSort(
        word_list,
        sort_func=sort_func,
        comp_func=comp_func
    )
    #print(ordering)
    #return sum(i * nameScore(x) for i, x in\
    #            enumerate(wordsSort(word_list,
    #            sort_func=sort_func, comp_func=comp_func), start=1))
    return sum(i * nameScore(x) for i, x in\
                enumerate(ordering, start=1))

def loadStrings(doc: str, rel_package_src: bool=False) -> List[str]:
    """
    Loads a list of strings from .txt file located at relative or
    absolute location doc. The file should contain the words separated
    by commas (',') with each word surrounded by double quotation marks
    ('"').
    
    Args:
        Required positional:
        doc (str): The relative or absolution location of the .txt
                file containing the list of strings.
        
        Optional named:
        rel_package_src (bool): Whether a relative path given by doc
                is relative to the current directory (False) or
                the package src directory (True).
            Default: False
    
    Returns:
    List of strings (str), with each entry in the list representing one
    of the words in the .txt file at doc. The list contains all the
    words in that .txt file in the same order as they appear there.
    """
    
    #print(src_directory.name, type(src_directory))
    doc = doc.strip()
    if rel_package_src and not doc.startswith("/"):
        src_directory = Path(__file__).resolve()
        while src_directory.name != "src":
            #print(src_directory.name)
            src_directory = src_directory.parent
        doc = (src_directory / doc).resolve()
    #print(doc)
    if not os.path.isfile(doc):
        raise FileNotFoundError(f"There is no file at location {doc}.")
    with open(doc) as f:
        txt = f.read()
    return txt.strip("\"").split("\",\"")

def nameListScoreFromFile(
    word_file: str="project_euler_problem_data_files/p022_names.txt",
    sort_func: Optional[Callable[[List[Any], Callable[[Any, Any], Tuple[Any]]], Tuple[Any]]]=None,
    comp_func: Callable[[Any, Any], int]=alphabetComparitor,
    rel_package_src: bool=True,
) -> int:
    """
    Solution to Project Euler #22

    For a list of words (strings of alphabet characters, either
    upper or lower case) contained in the .txt file at relative or
    absolute location word_file (where the words in this file are
    separated by commas and each word surrounded by double quotation
    marks), and a comparitor function, comp_func that defines a
    total preorder on the set of words in this list (i.e. for any
    elements of the set a, b and c, a <= b and b <= c implies a <= c -
    transitivity and at least one of a <= b or b <= a is true - strong
    connectedness), calculates the combined name score of this list of
    words for this comparitor function.
    The combined name score for a given list of words for a given
    comparitor function that defines a preorder on the set of words
    in that list is defined to be the sum over the values given for
    each word by the product of the word's name score (see nameScore
    for the definition of the name score) and its position in the
    ordering of the list of words based on the given comparitor
    function (where the first element in the ordering has position 1
    and each subsequent element in the ordering has position one
    greater than the previous element in the ordering).
    The sorting is performed by sort_func() if this is given and
    not given as None, otherwise uses the inbuild sorted() function.
    
    Args:
        Optional named:
        word_file (str): The relative or absolution location of the
                .txt file containing the list of words whose combined
                name score is to be calculated. In the file at location
                word_file, the words should be separated by a comma
                and each word should be surrounded by double quotation
                marks. Furthermore, each of the words in the list
                contained in the file at location word_file may only
                contain alphabet characters (both upper and lower case
                are allowed for any of the characters in any of the
                words).
            Default: "project_euler_problem_data_files/p022_names.txt"
        sort_func (Function or None): The algorithm used to sort the
                words in word_list. If not given or given as None,
                uses the inbuilt sorted() function.
            Default: None
        comp_func (function): Comparitor function that defines a
                total preorder ordering on the set of words in
                word_file.
                For any two elements a and b of the set of words in
                word_file, comp_func(a, b) = -1 if the comparitor
                assesses that a <= b but not b >= a,
                comp_func(a, b) = 1 if the comparitor assesses that
                a >= b but not b >= a and comp_func(a, b) = 0
                otherwise (i.e. both a >= b and a <= b).
            Default: alphabetComparitor
        rel_package_src (bool): Whether a relative path given by word_file
                is relative to the current directory (False) or the
                package src directory (True).
            Default: True
    
    Returns:
    Integer (int) giving the combined name score (as defined above) of
    the list of words in the file at location word_file for the
    comparitor function comp_func.
    """
    #since = time.time()
    word_list = loadStrings(word_file, rel_package_src=rel_package_src)
    res = nameListScore(
        word_list,
        sort_func=sort_func,
        comp_func=comp_func,
    )
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
    
# Problem 23
def findAbundantNumbers(n_max: int) -> List[int]:
    """
    Finds all abundant numbers not exceeding n_max.
    A number is an abundant number if and only if it is a strictly
    positive integer whose sum of positive proper divisors is
    strictly greater than the number itself.
    
    Args:
        Required positional:
        n_max (int): The largest integer considered for membership
                in the returned list.
    
    Returns:
    List of ints containing all abundant numbers not exceeding n_max
    in increasing order.
    """
    # Using solution to Problem #21
    return [i for i in range(1, n_max + 1) if\
                    factorSum(i, proper=True) > i]

def notExpressibleAsSumOfTwoAbundantNumbers(
    n_max: Optional[int]=None
) -> Set[int]:
    """
    Finds all strictly positive integers (if n_max specified as an
    integer value, not exceeding n_max) which cannot be written as the
    sum of two abundant numbers (see documentation of
    findAbundantNumbers() for definition of an abundant number).
    Uses the fact that all integers greater than 28123 can be written
    as the sum of two abundant numbers.
    
    Args:
        Optional named:
        n_max (int or None): If specified as an integer value, the
                largest integer considered for membership in the
                returned list. Otherwise, there is no such upper
                bound.
            Default: None
    
    Returns:
    List of ints containing all strictly positive integers (if n_max
    specified as an integer value, not exceeding n_max) which can be
    written as the sum of two abundant numbers. The elements of the
    list are guaranteed to be in increasing order.
    """
    n_max_ub = 28123
    if n_max is None:
        n_max = n_max_ub
    else: n_max = min(n_max, n_max_ub)
    abundant_sum_set = set()
    abundant_lst = findAbundantNumbers(n_max)
    #print(abundant_lst)
    for i1, num1 in enumerate(abundant_lst):
        for i2 in range(i1 + 1):
            num_sum = num1 + abundant_lst[i2]
            if num_sum > n_max: break
            abundant_sum_set.add(num_sum)
    return [num for num in range(1, n_max + 1)\
            if num not in abundant_sum_set]

#def notExpressibleAsSumOfTwoAbundantNumbers(n_max=28123):
#    """
#    Finds all strictly positive integers not exceeding n_max which
#    cannot be written as the sum of two abundant numbers (see
#    documentation of findAbundantNumbers() for definition of an
#    abundant number).
#    
#    Args:
#        Required positional:
#        n_max (int): The largest integer considered for membership
#                in the returned set.
#    
#    Returns:
#    Set of ints containing all strictly positive integers not exceeding
#    n_max which cannot be written as the sum of two abundant numbers.
#    """
#    return set(range(1, n_max + 1))\
#        .difference(expressibleAsSumOfTwoAbundantNumbers(n_max=28123))

def notExpressibleAsSumOfTwoAbundantNumbersSum(
    n_max: Optional[int]=None,
) -> int:
    """
    Solution to Project Euler #23

    Finds the sum of all strictly positive integers not exceeding n_max
    which cannot be written as the sum of two abundant numbers (see
    documentation of findAbundantNumbers() for definition of an
    abundant number).
    
    Args:
        Optional named:
        n_max (int or None): If specified as an integer value, the
                largest integer considered for membership in the
                returned sum. Otherwise there is no such upper bound.
            Default: None
    
    Returns:
    Integer (int) giving the sum all strictly positive integers up to
    (if n_max specified as an integer value, not exceeding n_max) which
    cannot be written as the sum of two abundant numbers.
    """
    #since = time.time()
    res = sum(notExpressibleAsSumOfTwoAbundantNumbers(n_max))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 24
def basicComparitor(obj1: Any, obj2: Any) -> int:
    """
    Comparitor function for two objects obj1 and obj2 relying on
    the dunder methods __le__() and __ge__() of obj1 and obj2
    to evaluate the comparison.
    
    Args:
        Required positional:
        obj1 (object): The first object to be compared.
        obj2 (object): The second object to be compared.
    
    Returns:
    If the __le__() and/or __ge__() methods of obj1 and/or
    obj2 collectively determine that obj1 is less than obj2, returns
    -1, while if those methods collectively determine that obj1 is
    greater than obj2, returns 1. Otherwise returns 0.
    """
    if obj1 <= obj2 and obj1 >= obj2: return 0
    elif obj1 <= obj2: return -1
    elif obj1 >= obj2: return 1
    return 0

def nthPermutation(
    n: int=10 ** 6,
    objs: Union[Tuple[Any], Set[Hashable], List[Any]]=tuple(range(10)),
    comp_func: Callable[[Any, Any], int]=basicComparitor,
) -> Tuple[Any]:
    """
    For a collection of unique objects, objs, with an associated
    comparitor function comp_func that defines a total order on the
    set of objects in objs (i.e. for any elements of the set a, b and
    c, a <= a - reflexivity, a <= b and b <= a implies b = a -
    antisymmetry, a <= b and b <= c implies a <= c - transitivity and
    for every element a and b in the set at least one of a <= b or
    a >= b is true - strong connectedness), consider the following.
    Define the list of all of the permutations of the set of objects
    in objs, such that permutations are ordered lexicographically using
    the comparitor function comp_func. This function returns the
    permutation that would appear at index (n - 1) in this list (i.e.
    is the nth permutation in this ordering).
    
    Args:
        Optional named:
        n (int): The position of the desired permutation in the
                sorted list of permutations (where the permutation at
                the beginning of the list is at position 1 and all
                other permutations are at the position exactly one
                greater than that of the previous permutation in the
                list.
            Default: 10 ** 6
        objs (tuple/list/set of objects): The collection of objects
                on which the list of permutations is based.
            Default: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        comp_func (function): A comparitor function which gives a total
                ordering of the set of objects in objs (see above for
                definition of a total ordering). This function
                accepts exactly two elements of objs as inputs and
                outputs -1 if the the first of these objects is
                assessed by the comparitor to be larger than the
                second, 1 if the second is larger assessed to be larger
                than the first and 0 if the comparitor assesses neither
                of the objects to be greater than or less than the
                other.
                Given that this is a total ordering, comp_func should
                return 0 if and only if the two objects being compared
                are the same object.
            Default: basicComparitor
    
    Returns:
    The nth permutation of objs based on lexicographic ordering where
    each element is compared by the comparitor function comp_func.
    """
    # Review- currently, comp_func not actually being used and is
    # always using the default comparison (lexicographic)

    #since = time.time()
    # Check that n is less that the factorial of len(obj)
    if n > math.factorial(len(objs)):
        raise ValueError("The number of permutation specified "
            "is greater than the number of permutations.")
    if len(objs) == 1:
        return tuple(objs)
    
    remain = SortedList(objs)
    n -= 1
    
    m = len(objs)
    res = [None] * m
    for i in range(m):
        fact_term = math.factorial(len(remain) - 1)
        j, n = divmod(n, fact_term)
        res[i] = remain.pop(j)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
    """
    def recur(i: int):
        return
    
    # Identify the first digit
    obj_sorted = sorted(list(obj))
    fact_term = math.factorial(len(obj) - 1)
    i = (n - 1) // fact_term
    en = obj_sorted[i]
    tail = NthPermutation(obj=obj.difference({en}), n=(n - fact_term * i))
    return [en, *tail]
    """

def nthDigitPermutation(
    n: int=10 ** 6,
    digs: Union[Tuple[int], Set[int], List[int]]=tuple(range(10)),
    base: int=10,
) -> int:
    """
    Solution to Project Euler #24

    For a collection of unique digits, digs, consider the following.
    Define the list of all of the permutations of the set of digits
    in digs, such that permutations are ordered lexicographically
    (where digits are compared with each other based on their value).
    This function returns the integer which, when expressed in the
    chosen base (with as many leading zeros as required to make that
    expression have the same number of digits as are in digs)
    gives, when its digits are read left to right the digit permutation
    that would appear at index (n - 1) in this list (i.e. is the nth
    permutation in this ordering).
    
    Args:
        Optional named:
        n (int): The position of the desired permutation in the
                sorted list of permutations (where the permutation at
                the beginning of the list is at position 1 and all
                other permutations are at the position exactly one
                greater than that of the previous permutation in the
                list.
            Default: 10 ** 6
        digs (tuple/list/set of ints): The collection of unique integers
                giving the values of the digits on which the list of
                permutations is based.
            Default: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        base (int): Integer no less than 2 and strictly greater than the
                largest value in digs giving the base which, when
                expressed in that base (possibly with leading zeros) the
                returned integer when read from left to right is to give
                the specified digit permutation.
            Default: 10

    Returns:
    Integer (int) representing the nth permutation of the digits digs
    based on lexicographic ordering (where the digits are compared by
    their value) when expressed in the chosen base (possibly with
    leading zeros) as described above.
    """

    if base <= max(digs):
        raise ValueError("The value of base must exceed the largest value "
                "in digs")
    dig_lst = nthPermutation(
        n=n,
        objs=digs,
        comp_func=basicComparitor,
    )
    res = 0
    for d in dig_lst:
        res = res * base + d
    return res

# Problem 25
def firstFibonacciGEn(n: int=10**999) -> Tuple[int, int]:
    """
    Finds the first Fibonacci number at least as large as n
    (where terms 1 and 2 of the Fibonacci sequence are taken to
    be 1, 1).
    
    Args:
        Optional named:
        n (int): The minimum size for the value of the Fibonacci
                term returned.
            Default: 10 ** 999
    
    Returns:
    2-tuple whose index 0 contains the term number of the first
    Fibonacci number (with the Fibonacci sequence terms 1 and 2
    both being 1) at least as large as n and whose index 0 contains
    the value of this Fibonacci number.
    """
    #since = time.time()
    n_sum = 0
    n1, n2 = 1, 1
    if n2 >= n: return (1, n1)
    i = 2
    while n2 < n:
        # Calculating the new term, allocating it to n2 and
        # allocating the old n2 term to n1
        n1, n2 = n2, n1 + n2
        i += 1
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return (i, n2)

def firstFibonacciGEnTermNumber(n: int=10**999) -> int:
    """
    Solution to Project Euler #25

    Finds the term number of the first Fibonacci number at least
    as large as n (where terms 1 and 2 of the Fibonacci sequence
    are taken to be 1, 1).
    
    Args:
        Optional named:
        n (int): The minimum size for the value of the Fibonacci
                term returned.
            Default: 10 ** 999
    
    Returns:
    Integer (int) giving the term number of the first Fibonacci
    number (with the Fibonacci sequence terms 1 and 2 both being
    1) at least as large as n.
    """
    return firstFibonacciGEn(n=n)[0]

# Problem 26
def basimalCycleLength(num: int, base: int=10):
    """
    Finds the cycle length of a basimal expansion of a reciprocal
    of a strictly positive integer num in the chosen base.
    Basimal is a term used for generalising decimal to bases other
    than 10
    
    Args:
        Required positional:
        num (int): The strictly positive integer for which the
                cycle length of the basimal expansion of the
                reciprocal is to be found.
        
        Optional named:
        base (int): The base in which the basimal expansion is
                to be expressed.
            Default: 10
        
    Returns:
    Integer (int) giving the cycle length of the basimal expansion
    of the reciprocal of num in the chosen base. If the basimal
    expansion terminates, returns 0.
    """
    expansion_list = []
    remain_dict = {}
    # Essentially simple division until find a repeat
    remain = 1
    while remain not in remain_dict.keys():
        remain_dict[remain] = len(expansion_list)
        if remain == 0:
            # Terminating decimal expansions
            return 0
        remain *= base
        while remain < num:
            remain *= base
            expansion_list.append(0)
        exp_term, remain = divmod(remain, num)
        expansion_list.append(exp_term)
    return len(expansion_list) - remain_dict[remain]

def maxReciprocalBasimalCycleProperties(n_max: int=999, base: int=10) -> Tuple[int, int]:
    """
    Finds the strictly positive integer num no greater than n_max such
    that the basimal expansion of the reciprocal of num has a
    recurring cycle and the length of this cycle is longer than that
    for any strictly positive integer less than num and is at least
    as long as that of any strictly positive integer no greater than
    n_max. Also returns the length of that cycle (which by the
    definition of the number is the longest recurring cycle length of
    any reciprocal of any strictly positive integer no greater than
    n_max).
    
    Args:
        Optional named:
        n_max (int): Strictly positive integer giving the largest
                integer considered.
            Default: 999
        base (int): Strictly positive integer giving the base in
                which the basimal expansions of the reciprocals
                are expressed.
            Default: 10
    
    Returns:
    2-tuple whose index 0 contains the strictly positive integer
    satisfying the given conditions and whose index 1 contains the
    length of the recurring cycle in the basimal expansion of the
    reciprocal of this number.
    """
    #since = time.time()
    res = (1, 0)
    for i in range(1, n_max + 1):
        cycle_len = basimalCycleLength(i, base=base)
        if cycle_len > res[1]:
            res = (i, cycle_len)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

def reciprocalWithLargestBasimalCycleLength(n_max: int=999, base: int=10) -> int:
    """
    Solution to Project Euler #26

    Finds the strictly positive integer num no greater than n_max such
    that the basimal expansion of the reciprocal of num has a
    recurring cycle and the length of this cycle is longer than that
    for any strictly positive integer less than num and is at least
    as long as that of any strictly positive integer no greater than
    n_max.
    
    Args:
        Optional named:
        n_max (int): Strictly positive integer giving the largest
                integer considered.
            Default: 999
        base (int): Strictly positive integer giving the base in
                which the basimal expansions of the reciprocals
                are expressed.
            Default: 10
    
    Returns:
    Integer (int) giving the strictly positive integer satisfying
    the conditions described above.
    """
    return maxReciprocalBasimalCycleProperties(n_max=n_max, base=base)[0]

# Problem 27
# Based on solution to Problem 7
def primesUpToN(n_max: int) -> List[int]:
    """
    Finds all primes up to not exceeding n_max
    
    Args:
        Required positional:
        n_max (int): The largest integer considered for inclusion
                in the result.
    
    Returns:
    A tuple with all primes not exceeding n_max in increasing order.
    """
    prime_list = []
    sieve = [True for x in range(n_max + 1)]
    for p in range(2, n_max + 1):
        if sieve[p]:
            prime_list.append(p)
            for i in range(p ** 2, n_max + 1, p):
                sieve[i] = False
    return prime_list

def maxConsecutiveQuadraticPrimes(abs_a_max: int, abs_b_max: int,
        n_start: int) -> Tuple[int]:
    """
    For the ordered pair of integers (a, b) where:
        abs(a) <= abs_a_max and abs(b) <= abs_b_max
    let f(a, b) be the largest non-negative integer such that the
    quadratic expression:
        n ** 2 + a * n + b
    is a prime for this number of consecutive integer values of n,
    starting at n_start and increasing.
    This function finds the integer values of a and b such that
    there is no other such ordered pair such that f gives a larger
    value, and for any ordered pair (a', b') such that
    abs(a') <= abs_a_max and abs(b') <= abs_b_max and
    f(a', b') = f(a, b), either a' > a or a' = a and b' >= b.
    
    Args:
        Required positional:
        abs_a_max (int): Non-negative integer giving the largest
                absolute value of the parameter a considered.
        abs_b_max (int): Non-negative integer giving the largest
                absolute value of the parameter b considered.
        n_start (int): The value of n in the quadratic expression
                from which the count of consecutive primes as n
                increases is started.
    
    Returns:
    2-tuple whose index 0 contains a 2-tuple of ints containing
    the values of a and b in index 0 and 1 respectively and whose
    index 1 is the number of consecutive integer values of n,
    for which the quadradic expression for this a and b (starting
    from n_start and increasing) is prime.
    """
    # We observe that the polynomial cannot be prime if n is a
    # multiple of b
    prime_upper_bound = (abs(n_start) // abs_b_max + 1) * abs_b_max
    prime_set = set(primesUpToN(prime_upper_bound))
    
    out_prov = (None, 0)
    for a in range(-abs_a_max, abs_a_max + 1):
        for b in range(-abs_b_max, abs_b_max + 1):
            #if abs(b) not in prime_set or (a - b) % 2 != 0:
            #    continue
            n = n_start
            while True:
                val = n ** 2 + a * n + b
                if val not in prime_set:
                    break
                n += 1
            n_diff = n - n_start
            if n_diff > out_prov[1]:
                out_prov = ((a, b), n_diff)
    return out_prov

def maxConsecutiveQuadraticPrimesProduct(
    abs_a_max: int=999,
    abs_b_max: int=1000,
    n_start: int=0,
) -> int:
    """
    Solution to Project Euler #27

    For the ordered pair of integers (a, b) where:
        abs(a) <= abs_a_max and abs(b) <= abs_b_max
    let f(a, b) be the largest non-negative integer such that the
    quadratic expression:
        n ** 2 + a * n + b
    is a prime for this number of consecutive integer values of n,
    starting at n_start and increasing.
    This function finds the integer values of a and b such that
    there is no other such ordered pair such that f gives a larger
    value, and for any ordered pair (a', b') such that
    abs(a') <= abs_a_max and abs(b') <= abs_b_max and
    f(a', b') = f(a, b), either a' > a or a' = a and b' >= b, and
    returns the product of the values of a and b identified.
    
    Args:
        Optional named:
        abs_a_max (int): Non-negative integer giving the largest
                absolute value of the parameter a considered.
            Default: 999
        abs_b_max (int): Non-negative integer giving the largest
                absolute value of the parameter b considered.
            Default: 999
        n_start (int): The value of n in the quadratic expression
                from which the count of consecutive primes as n
                increases is started.
            Default: 0
    
    Returns:
    Integer (int) giving the product of a and b for the ordered pair
    (a, b) fulfilling the requirements given.
    """
    #since = time.time()
    mcp_out = maxConsecutiveQuadraticPrimes(abs_a_max=abs_a_max,
                        abs_b_max=abs_b_max, n_start=n_start)
    res = mcp_out[0][0] * mcp_out[0][1]
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 28
def spiralDiagonalsGenerator(
    max_side_len: Optional[int]=None,
) -> Generator[int, None, None]:
    """
    Generator that yields the values on the diagonals of the
    spiral grid in increasing order.
    The spiral grid is the infinite 2 dimensional grid formed by
    placing 1 on one of the grid spaces, then taking a single step to
    the right and placing a 2 on that grid space. For every subsequent
    step, based on the direction previously stepped, considers whether
    to go to the next direction option in the cyclic sequence
    (right, down, left, right). If the grid space exactly one step in
    the next direction option has not been filled, the step is taken
    to that grid space, otherwise takes a single step in the direction
    of the previous step. In either case, enters the number exactly
    one more than the number previously used in the square landed on.
    This creates a pattern that starts as follows:
        21 22 23 24 25
        20  7  8  9 10
        19  6  1  2 11
        18  5  4  3 12
        17 16 15 14 13
    The pattern ends up tracing out concentric squares in the grid,
    each of which has an odd number of numbers in each side of the
    square, exactly two more than the square directly inside it.
    We refer to these squares as layers of the square spiral and
    the number of numbers in a given square as the side length
    of the corresponding layer. The diagonals of the square spiral
    are those elements that are at the vertices of these squares,
    including 1.
    
    Args:
        Optional named:
        max_side_len (int or None): If given as an integer, this is
                the maximum side length of the layers of the spiral
                matrix considered (once the generator reaches a layer
                with side length exceeding this, the generator
                terminates). Otherwise, the generator continues
                indefinitely.
            Default: None
                
    Yields:
    Integers (ints) giving the numbers appearing on the diagonals of
    the spiral grid in increasing order. If square_len is given as
    an integer, this generator terminates once it reaches a layer
    with side length square_len, otherwise continues indefinitely.
    """
    increment = 0
    curr = 1
    yield 1
    max_n = max_side_len ** 2 if isinstance(max_side_len, int)\
            else float("inf")
    while curr < max_n:
        increment += 2
        for i in range(4):
            curr += increment
            yield curr
    return
        
def numSpiralDiagonalsSum(max_side_len: int=1001) -> int:
    """
    Solution to Project Euler #28

    Finds the sum of the integers on the diagonals of the square
    spiral in the layers which have side length not exceeding
    max_side_len.
    See documentation of spiralDiagonalsGenerator() for details
    regarding the definition of the square spiral and the
    terminology used relating to it.
    
    Args:
        Optional named:
        max_side_len (int): The upper bound (inclusive) for the
                side length of layers of the square spiral grid
                whose diagonals are included in the sum.
            Default: 1001
    
    Returns:
    Integer (int) giving the sum of the integers on the diagonals of
    the square spiral grid belonging to layers of the square spiral
    grid whose side length does not exceed max_side_len
    """
    #since = time.time()
    res = 0
    for num in spiralDiagonalsGenerator(max_side_len=max_side_len):
        res += num
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 29
def distinctPowersNum(
    a_range: Tuple[int]=(2, 100),
    b_range: Tuple[int]=(2,100),
) -> int:
    """
    Solution to Project Euler #29

    Finds the number of integers n for which n = a ** b for some
    integers a and b such that:
    a_range[0] <= a <= a_range[1] and b_range[0] <= a <= b_range[1].
    
    Args:
        a_range (2-tuple of ints): The range of values the base of
                the exponentiation can take, where index 0 containes
                the minimum value and index 1 contains the maximum
                value.
            Default: (2, 100)
        b_range (2-tuple of ints): The range of values the exponent of
                the exponentiation can take, where index 0 containes
                the minimum value and index 1 contains the maximum
                value.
            Default: (2, 100)
    
    Returns:
    Integer (int) giving the number of integers that can be expressed
    as some integer to the power of some other integer where the base
    and the exponent are restricted to the range of values given by
    a_range and b_range respectively.
    """
    #since = time.time()
    seen = set()
    res = int(not b_range[0])
    for num in range(a_range[0], a_range[1] + 1):
        if num in seen: continue
        #print(f"num = {num}")
        pows = set()
        num2 = num
        exp = 1
        while num2 <= a_range[1]:
            seen.add(num2)
            pows |= {exp * x for x in range(b_range[0], b_range[1] + 1)}
            num2 *= num
            exp += 1
        res += len(pows)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
    #return len(DistinctPowersSet(a_range=a_range, b_range=b_range))

# Previous solution to Problem 29
#def DistinctPowersSet(a_range=(2, 100), b_range=(2,100)):
#    """
#    Finds the set of integers n for which n = a^b for some
#    a_range[0] <= a <= a_range[1] and b_range[0] <= a <= b_range[1].
#    
#    Args:
#    a_range (2-tuple of ints)
#    b_range (2-tuple of ints)
#    
#    Returns:
#    Set (set) of the integers satisfying the above conditions.
#    """
#    out_set = set()
#    for a in range(a_range[0], a_range[1] + 1):
#        for b in range(b_range[0], b_range[1] + 1):
#            out_set.add(a ** b)
#    return out_set

# Problem 30- try to make faster
# Similar to solution to Problem 16
def digitPowSum(num: int, exp: int, dig_pow_dict: Dict[int, int],\
        base: int=10) -> int:
    """
    Calculates the sum of the exp:th power of digits of a non-negative
    integer num when expressed in the chosen base (referred to as the
    digit power sum of num in the chosen base for exponent exp)
    
    Args:
        Required positional:
        num (int): The non-negative integer whose digits in its
                representation in the chosen base are used to compute
                the result.
        exp (int): The power to which each digit of the representation
                of num in the chosen base should be taken to in the
                sum.
        
        Optional named:
        dig_pow_dict (dict or None): If given as a value which is not
                None, is a dictionary representing pre-computed values
                for powers of digits, whose keys are non-negative
                integers with the corresponding value of that integer
                to the power of exp. If given as None these are
                powers of digits are computed directly as needed.
                Note that if given as a dictionary, this may be updated
                for instance if a missing digit is encountered.
            Default: None
        base (int): Strictly positive integer giving the base in which
                the integer num is to be represented when considering
                the digit power sum.
            Default: 10
    
    Returns:
    Integer giving the power digit sum of num in the chosen base for
    exponent exp.
    """
    res = 0
    if dig_pow_dict is None:
        while num:
            num, r = divmod(num, base)
            res += r ** exp
    else:
        while num:
            num, r = divmod(num, base)
            if r not in dig_pow_dict.keys():
                dig_pow_dict[r] = r ** exp
            res += dig_pow_dict[r]
    return res

def digitPowSumEqualsSelf(exp: int, base: int=10) -> List[int]:
    """
    Finds the strictly positive integers for which the sum of the
    exp:th power of digits of that number when expressed in the
    chosen base (referred to as the digit power sum of the given
    integer in the chosen base for exponent exp) is equal to the
    number itself.
    
    Args:
        Required positional:
        exp (int): The exponent used in the digit power sums.
        
        Optional named:
        base (int): Strictly positive integer giving the base in which
                the integers considered are to be represented when
                calculating the digit power sum for that integer.
            Default: 10
    
    Returns:
    List containing all the strictly positive integers whose digit
    power sum in the chosen base for exponent exp is equal to itself
    in increasing order.
    """
    # Preparing the powers of digits in a dictionary
    pow_dict = {i: i ** exp for i in range(base)}
    
    # Finding the upper bound for the number of digits
    n_digit = 2
    while len(str(n_digit * (base - 1) ** exp)) >= n_digit:
        n_digit += 1
    n_max = (n_digit - 1) * (base - 1) ** exp
    #print(n_max)
    
    res = []
    for i in range(base, n_max + 1):
        if digitPowSum(i, exp, dig_pow_dict=pow_dict, base=base) == i:
            res.append(i)
    return res

def digitPowSumEqualsSelfSum(exp: int=5, base: int=10) -> int:
    """
    Solution to Project Euler #30

    Finds the sum of strictly positive integers for which the sum of
    the exp:th power of digits of that number when expressed in the
    chosen base (referred to as the digit power sum of the given
    integer in the chosen base for exponent exp) is equal to the
    number itself.
    
    Args:
        Required positional:
        exp (int): The exponent used in the digit power sums.
        
        Optional named:
        base (int): Strictly positive integer giving the base in which
                the integers considered are to be represented when
                calculating the digit power sum for that integer.
            Default: 10
    
    Returns:
    The sum of all the strictly positive integers whose digit power
    sum in the chosen base for exponent exp is equal to itself.
    """
    #since = time.time()
    res = sum(digitPowSumEqualsSelf(exp=exp, base=base))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 31
def coinCombinations(
    amount_p: int=200,
    coins_allowed: Tuple[int]=(1, 2, 5, 10, 20, 50, 100, 200),
) -> int:
    """
    Solution to Project Euler #31

    Finds the number of distinct combinations of coins with values
    in pence as given by coins_allowed that together are worth
    amount_p pence, where two combinations of coins are considered
    distinct if and only if at least one coin value appears with
    a different frequency in the two combinations.
    
    Args:
        Optional named:
        amount_p (int): The amount of money in pence that is to be
                constituted by the combinations of coins in
                coins_allowed
            Default: 200 (£2.00)
        coins_allowed (tuple of ints): A list of the values of the
                coins that may be used.
            Default: (1, 2, 5, 10, 20, 50, 100, 200)
    
    Returns:
    The number of distinct combinations of coins with values in pence
    as given by coins_allowed that together are worth amount_p pence.
    """
    #since = time.time()
    n = len(coins_allowed)
    coins_allowed = sorted(coins_allowed)
    if not n: return int(not amount_p)
    if n == 1:
        return int(not (amount_p % coins_allowed[0]))
    
    # 1D dynamic programming
    curr = [0] * (amount_p + 1)
    curr[0] = 1
    for num in coins_allowed:
        for i in range(amount_p - num + 1):
            curr[i + num] += curr[i]
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return curr[-1]

# Problem 32
def pandigitalProducts(
    min_dig: int=1,
    max_dig: Optional[int]=None,
    base: int=10,
) -> int:
    """
    Finds all possible combinations of strictly positive integers a, b
    and c such that a <= b, a * b = c and when each of a, b and c are
    expressed in the chosen base, each digit between min_dig and
    max_dig inclusive appears exactly once in exactly one of a, b or c,
    where none of the expressions have leading zeros.
    
    Args:
        Optional named:
        min_dig (int): Non-negative integer giving the smallest digit
                in the range of digits which should all appear exactly
                once in any product included.
                Should be between 0 and (base - 1) inclusive.
            Default: 1
        max_dig (int): Strictly positive integer giving the largest
                digit in the range of digits which should all appear
                exactly once in any product included.
                Should be between (min_dig + 1) and (base - 1)
                inclusive.
            Default: base - 1
        base (int): Strictly positive integer giving the base in which
                the integers are to be expressed.
            Default: 10
    
    Returns:
    Set of 3-tuple of ints containing all possible tuples (a, b, c)
    such that collectively the integers a, b and c satisfy the
    specified requirements.
    """
    if max_dig is None: max_dig = base - 1
    # Check which combinations of number of digits is allowed
    # for the multiplying numbers
    allowed_combs = set()
    nd = max_dig - min_dig + 1
    iter_objs = set()
    for i1 in range(1, ((nd + 1) >> 1)):
        min1 = 0
        if not min_dig:
            min1 = 1
            if i1 > 1: min1 *= base
        for j in range(2 * (not min_dig), i1):
            min1 = min1 * base + min_dig + j
        max1 = 0
        for j in range(i1):
            max1 = max1 * base + max_dig - j
        #min1 = int("".join([str(x) for x in range(1, i + 1)]))
        #max1 = int("".join([str(n - x) for x in range(i)]))
        for i2 in range(1, i1 + 1):
            min2 = 0
            if not min_dig:
                min2 = 1
                if i2 > 1: min2 *= base
            for j in range(2 * (not min_dig), i2):
                min2 = min2 * base + min_dig + j
            max2 = 0
            for j in range(i2):
                max2 = max2 * base + max_dig - j
            #min2 = int("".join([str(x) for x in range(1, i2 + 1)]))
            #max2 = int("".join([str(n - x) for x in range(i2)]))
            min_prod = min1 * min2
            min_prod_n_dig = 0
            while min_prod:
                min_prod //= base
                min_prod_n_dig += 1
            if min_prod_n_dig + i1 + i2 > nd: break
            max_prod = max1 * max2
            max_prod_n_dig = 0
            #print(i1, i2, max_prod)
            while max_prod:
                max_prod //= base
                max_prod_n_dig += 1
            if max_prod_n_dig + i1 + i2 < nd: continue
            iter_objs.add((range(min1, max1 + 1),\
                    range(min2, max2 + 1)))
    
    def digitsUniqueSet(num: int, allowed_dig: Set[int],\
            base: int=base) -> Tuple[Union[bool, Set[int]]]:
        res = set()
        while num:
            num, r = divmod(num, base)
            if r in res or r not in allowed_dig: return (False, set())
            res.add(r)
        return (True, res)
    
    # Review- try iterating over the possible permutations of digits
    # rather than almost pair of numbers with a respective number of
    # digits that may give a correct product.
    res = set()
    dig_set = set(range(min_dig, max_dig + 1))
    for iter_pair in iter_objs:
        for i1 in iter_pair[0]:
            b, digs1 = digitsUniqueSet(i1, dig_set, base=base)
            if not b: continue
            remain_digs = dig_set.difference(digs1)
            for i2 in iter_pair[1]:
                if i2 > i1: break
                b, digs2 = digitsUniqueSet(i2, remain_digs, base=base)
                if not b: continue
                remain_digs2 = remain_digs.difference(digs2)
                prod = i1 * i2
                b, prod_digs = digitsUniqueSet(prod, remain_digs2,\
                        base=base)
                if not b or len(prod_digs) != len(remain_digs2):
                    continue
                res.add((i1, i2, prod))
    return res
            
def pandigitalProductsSum(
    min_dig: int=1,
    max_dig: Optional[int]=None,
    base: int=10,
) -> int:
    """
    Solution to Project Euler #32

    Finds the sum of all possible strictly positive integers c such
    that there exist strictly positive integers a and b for which
    a * b = c and when each of a, b and c are expressed in the chosen
    base, each digit between min_dig and max_dig inclusive appears
    exactly once in exactly one of a, b or c, where none of the
    expressions have leading zeros.
    
    Args:
        Optional named:
        min_dig (int): Non-negative integer giving the smallest digit
                in the range of digits which should all appear exactly
                once in any product included.
                Should be between 0 and (base - 1) inclusive.
            Default: 1
        max_dig (int): Strictly positive integer giving the largest
                digit in the range of digits which should all appear
                exactly once in any product included.
                Should be between (min_dig + 1) and (base - 1)
                inclusive.
            Default: base - 1
        base (int): Strictly positive integer giving the base in which
                the integers are to be expressed.
            Default: 10
    
    Returns:
    Integer (int) giving the sum over all strictly positive integers c
    such that there exist strictly positive integers a and b for which
    a * b = c and when each of a, b and c are expressed in the chosen
    base, each digit between min_dig and max_dig inclusive appears
    exactly once in exactly one of a, b or c, where none of the
    expressions have leading zeros.
    """
    #since = time.time()
    if max_dig is None: max_dig = base - 1
    prods = pandigitalProducts(min_dig=min_dig, max_dig=max_dig,\
            base=base)
    res = sum({x[2] for x in prods})
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
        
# Problem 33
# Look at documentation- may need to simplify and/or clarify
def digitCancellationSet(
    numer: int,
    denom: int,
    base: int=10,
    leading_zeros_allowed: bool=False,
    simplify: bool=False,
) -> Set[Tuple[int]]:
    """
    For a given fraction (numer / denom), finds all possible digit
    cancellations in the given base.
    A digit cancellation of a fraction in a given base is result of
    the application of the following process a finite and non-zero
    number of times:
    For the representation of the numerator and denominator of the
    fraction in the chosen base, for any digits that appear in
    both representations, one of those digits is removed from each
    representation and the value of both numerator and denominator
    are recalculated to be consistent with the new representation.
    Note that this process may only be applied if the number of digits
    in both representations is greater than 1.
    Note the choice of which pair of digits to remove in each step
    of the process is free, as long as the process is applied at
    least once.
    Also note that steps in the process that digit cancellations
    such that either of the expressions of the numerator and
    denominator have leading zeros before simplification, then
    these are only included if leading_zeros_allowed is True. A
    single zero on its own in an expression in the chosen base is
    not considered to be a leading zero. Furthermore, digit
    cancellations for which the denominator becomes zero are not
    included. 
    
    Args:
        Required positional:
        numer (int): Strictly positive integer giving the numerator of
                the fraction whose digit cancellations in the given
                base are being calculated.
        denom (int): Strictly positive integer giving the denominator
                of the fraction whose digit cancellations in the given
                base are being calculated.
        
        Optional named:
        base (int): Strictly positive integer giving the base in which
                the numerator and denominator are to be represented
                during the applications of the process of digit
                cancellation.
            Default: 10
        leading_zeros_allowed (bool): Whether digit cancellations that
                result in either the expression for the numerator
                or denominator in the chosen base before simplification
                having any leading zeros is allowed.
            Default: False
        simplify (bool): Whether the fractions returned should be
                simplified into lowest terms (i.e. the numerator
                and denominator both divided by their greatest
                common divisor)
            Default: False
    
    Returns:
    Set containing 2-tuples representing each possible distinct digit
    cancellation of the fraction in the given base (subject to the
    specified restrictions regarding leading zeros and not allowing
    zero denominators) whose index 0 and 1 contain the value of the
    numerator and denominator respectively of the corresponding digit
    cancellation of the fraction for the given base.
    """
    numer_digs = []
    numer_ind_dict = {}
    while numer:
        numer, r = divmod(numer, base)
        numer_ind_dict.setdefault(r, [])
        numer_ind_dict[r].append(len(numer_digs))
        numer_digs.append(r)
    if len(numer_digs) <= 1: return set()
    denom_digs = []
    denom_ind_dict = {}
    while denom:
        denom, r = divmod(denom, base)
        denom_ind_dict.setdefault(r, [])
        denom_ind_dict[r].append(len(denom_digs))
        denom_digs.append(r)
    if len(denom_digs) <= 1: return set()
    
    shared_digs = sorted(set(numer_ind_dict.keys())\
            .intersection(set(denom_ind_dict.keys())))
    if not shared_digs: return set()
    mx_excl = min(len(numer_digs), len(denom_digs))
    inds = []
    #mx_lens = []
    for k in shared_digs:
        inds.append([numer_ind_dict[k], denom_ind_dict[k]])
    curr_pair_lsts = [[[set()], [set()]]]
    res = set()
    for i, (numer_inds, denom_inds) in enumerate(inds):
        mx_len = min(len(x) for x in (numer_inds, denom_inds))
        prev_len = len(curr_pair_lsts)
        for j in range(prev_len):
            prev_sets = curr_pair_lsts[j]
            length0 = len(prev_sets[0][0])
            for length in range(1, mx_len + (length0 + mx_len < mx_excl)):
                single_dig = [len(numer_digs) - (length0 + length) == 1,\
                        len(denom_digs) - (length0 + length) == 1]
                curr_pair_lsts.append([[], []])
                numer_vals, denom_vals = [], []
                for idx, (inds_lst, vals, digs) in\
                        enumerate(((numer_inds, numer_vals,\
                        numer_digs), (denom_inds, denom_vals,\
                        denom_digs))):
                    for ind_set in prev_sets[idx]:
                        for inds in itertools.combinations(inds_lst,\
                                length):
                            ind_set2 = ind_set.union(set(inds))
                            for k in reversed(range(len(digs))):
                                if k not in ind_set2: break
                            else: continue # should not happen
                            if digs[k] == 0 and\
                                    not leading_zeros_allowed and\
                                    (idx or not single_dig[idx]):
                                continue
                            val = digs[k]
                            for k in reversed(range(k)):
                                if k not in ind_set2:
                                    val = val * base + digs[k]
                            if not val and idx == 1: continue
                            vals.append(val)
                            curr_pair_lsts[-1][idx].append(ind_set2)
                if simplify:
                    for numer in numer_vals:
                        for denom in denom_vals:
                            g = gcd(numer, denom)
                            res.add((numer // g, denom // g))
                else:
                    for numer in numer_vals:
                        for denom in denom_vals:
                            res.add((numer, denom))
    return res
                

def digitCancellationsEqualToSelf(
    denom_max: int,
    base: int=10,
    leading_zeros_allowed: bool=False,
    exclude_trivial: bool=True,
) -> Tuple[Union[Tuple[int], Set[Tuple[int]]]]:
    """
    Finds all the fractions no less than 0 and less than 1, not
    necessarily expressed in lowest terms, whose denominator as given
    does not exceed denom_max, for which at least one of the fractions
    resulting from a digit cancellation in the given base (satisfying
    the given requirements) has the same value as the original
    fraction. These fractions are returned in unsimplified form (i.e.
    they are not reduced to lowest terms, but gives the numerator and
    denominator values as they were as given and as they were when the
    digit cancellations were applied). As such, the function may return
    multiple fractions with the same value (i.e. different numerators
    and denominators from each but for which the ratio of the numerator
    and denominator is the same). Additionally, for each of these
    fractions identified, the set of the fractions resulting from the
    digit cancellations of that fraction in the given base which have
    the same value as the original fraction are also returned, again
    in unsimplified form (i.e. not reduced to lowest form, but giving
    the numerator and denominator as they were directly after the digit
    cancellations were performed).
    A digit cancellation of a fraction in a given base is result of
    the application of the following process a finite and non-zero
    number of times:
    For the representation of the numerator and denominator of the
    fraction in the chosen base, for any digits that appear in
    both representations, one of those digits is removed from each
    representation and the value of both numerator and denominator
    are recalculated to be consistent with the new representation.
    Note that this process may only be applied if the number of digits
    in both representations is greater than 1.
    Note the choice of which pair of digits to remove in each step
    of the process is free, as long as the process is applied at
    least once.
    A trivial digit cancellation is one which only involves
    removing zeros from the right hand side of the numerator and
    denominator of the given fraction when expressed in the given
    base. If exclude_trivial is given as True, such digit cancellations
    are not considered.
    Also note that steps in the process that digit cancellations
    such that either of the expressions of the numerator and
    denominator have leading zeros before simplification, then
    these are only included if leading_zeros_allowed is True. A
    single zero on its own in an expression in the chosen base is
    not considered to be a leading zero. Furthermore, digit
    cancellations for which the denominator becomes zero are not
    included. 
    
    Args:
        Required positional:
        numer (int): Strictly positive integer giving the numerator of
                the fraction whose digit cancellations in the given
                base are being calculated.
        denom (int): Strictly positive integer giving the denominator
                of the fraction whose digit cancellations in the given
                base are being calculated.
        
        Optional named:
        base (int): Strictly positive integer giving the base in which
                the numerator and denominator are to be represented
                during the applications of the process of digit
                cancellation.
            Default: 10
        leading_zeros_allowed (bool): Whether digit cancellations that
                result in either the expression for the numerator
                or denominator in the chosen base before simplification
                having any leading zeros is allowed.
            Default: False
        exclude_trivial (bool): If given as True, trivial digit
                cancellations in the given base are not considered
                (see above for the definition of trivial digit
                cancellations), otherwise these are considered.
            Default: True
    
    Returns:
    Tuple (tuple) containing 2-tuples whose index 0 contains
    a 2-tuple whose index 0 and 1 contain the numerator and denominator
    respectively of each of the unsimplified fractions no less than
    zero and less than one whose denominator does not exceed denom_max
    for which at least one of the permitted digit cancellations in the
    chosen base yields a fraction with the same value as that fraction
    and whose index 1 contains the set of all the unsimplified
    fractions resulting from one of these digit cancellations that have
    the same value as the fraction in index 0, similarly represented
    by 2-tuples whose index 0 contains the numerator and index 1
    contains the denominator of the unsimplified fraction.
    """
    res = []
    base_pows = set()
    if exclude_trivial:
        curr = base
        while curr <= denom_max:
            base_pows.add(curr)
            curr *= base
    for denom in range(base, denom_max + 1):
        for numer in range(base, denom):
            dc_set = digitCancellationSet(numer, denom, base=base,\
                    leading_zeros_allowed=leading_zeros_allowed,\
                    simplify=False)
            out_tup = ((numer, denom), set())
            for (i, j) in dc_set:
                if j == 0 or numer * j != denom * i:
                    continue
                # Ignoring trivial examples
                if exclude_trivial:
                    q, r = divmod(denom, j)
                    if not r and q in base_pows:
                        continue
                out_tup[1].add((i, j))
            if out_tup[1]: res.append(out_tup)
    #print(res)
    return tuple(res)

def digitCancellationsEqualToSelfProdDenom(
    denom_max: int=99,
    base: int=10,
    leading_zeros_allowed: bool=False,
    exclude_trivial: bool=True,
) -> int:
    """
    Solution to Project Euler #33
    
    Finds the product all the fractions no less than 0 and less than 1,
    not necessarily expressed in lowest terms, whose denominator as
    given does not exceed denom_max, for which at least one of the
    fractions resulting from a digit cancellation in the given base
    (satisfying the given requirements) has the same value as the
    original fraction. Returns the denominator of the fraction
    resulting from this product in lowest terms.
    A digit cancellation of a fraction in a given base is result of
    the application of the following process a finite and non-zero
    number of times:
    For the representation of the numerator and denominator of the
    fraction in the chosen base, for any digits that appear in
    both representations, one of those digits is removed from each
    representation and the value of both numerator and denominator
    are recalculated to be consistent with the new representation.
    Note that this process may only be applied if the number of digits
    in both representations is greater than 1.
    Note the choice of which pair of digits to remove in each step
    of the process is free, as long as the process is applied at
    least once.
    A trivial digit cancellation is one which only involves
    removing zeros from the right hand side of the numerator and
    denominator of the given fraction when expressed in the given
    base. If exclude_trivial is given as True, such digit cancellations
    are not considered.
    Also note that steps in the process that digit cancellations
    such that either of the expressions of the numerator and
    denominator have leading zeros before simplification, then
    these are only included if leading_zeros_allowed is True. A
    single zero on its own in an expression in the chosen base is
    not considered to be a leading zero. Furthermore, digit
    cancellations for which the denominator becomes zero are not
    included. 
    
    Args:
        Required positional:
        numer (int): Strictly positive integer giving the numerator of
                the fraction whose digit cancellations in the given
                base are being calculated.
        denom (int): Strictly positive integer giving the denominator
                of the fraction whose digit cancellations in the given
                base are being calculated.
        
        Optional named:
        base (int): Strictly positive integer giving the base in which
                the numerator and denominator are to be represented
                during the applications of the process of digit
                cancellation.
            Default: 10
        leading_zeros_allowed (bool): Whether digit cancellations that
                result in either the expression for the numerator
                or denominator in the chosen base before simplification
                having any leading zeros is allowed.
            Default: False
        exclude_trivial (bool): If given as True, trivial digit
                cancellations in the given base are not considered
                (see above for the definition of trivial digit
                cancellations), otherwise these are considered.
            Default: True
    
    Returns:
    Integer (int) giving the denominator of the fraction in lowest
    terms resulting from the described product.
    """
    #since = time.time()
    dc_tup = digitCancellationsEqualToSelf(denom_max, base=base,\
            leading_zeros_allowed=leading_zeros_allowed,\
            exclude_trivial=exclude_trivial)
    num_prod = 1
    denom_prod = 1
    for frac_tup in dc_tup:
        g = gcd(*frac_tup[0])
        num_prod *= frac_tup[0][0] // g
        denom_prod *= frac_tup[0][1] // g
    nd_gcd = gcd(num_prod, denom_prod)
    #print(num_prod, denom_prod)
    #print(num_prod // nd_gcd, denom_prod // nd_gcd)
    res = denom_prod // nd_gcd
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 34
def digitFactorialSumEqualsSelf(
    n_dig_min: int,
    base: int=10,
) -> Set[int]:
    """
    Finds all strictly positive integers such that when expressed
    in the chosen base, contains at least n_dig_min digits and the
    sum of the factorials of the digits is exactly equal to the
    integer itself.
    It is straightforward to show that for any finite base, the
    number of such integers is finite.
    
    Args:
        Required positional:
        n_dig_min (int): Non-negative integer giving the minimum
                number of digits in the chosen base of the integers
                considered.
        
        Optional named:
        base (int): Strictly positive integer giving the base in
                which the integers in question are expressed for the
                assessment of the number of digits and the calculation
                of the digit factorial sum of the integer.
            Default: 10
    
    Returns:
    A set containing every strictly positive integer such that when
    expressed in the chosen base contains at least n_dig_min digits
    and the sum of factorials of the digits in this expression is
    exactly equal to the integer itself.
    """
    since = time.time()
    factorials = [math.factorial(i) for i in range(base)]
    
    # Finding the upper bound for the number of digits
    curr = factorials[base - 1]
    n_max = 1
    n_dig_max = 0
    while curr >= n_max:
        curr += factorials[base - 1]
        n_max *= base
        n_dig_max += 1
    #print(n_max)
    #print(n_dig_max)
    
    res = set()
    dig_dict = {}
    def backtrack(d: int, n_dig: int, curr: int,\
            target_rng: Tuple[int]) -> None:
        if not n_dig:
            curr_dict = {}
            num = curr
            while num:
                num, r = divmod(num, base)
                curr_dict[r] = curr_dict.get(r, 0) + 1
                if curr_dict[r] > dig_dict.get(r, 0):
                    return
            if curr_dict == dig_dict:
                #print(curr)
                res.add(curr)
            return
        if target_rng[0] > factorials[d] * n_dig or\
                target_rng[1] < factorials[0] * n_dig:
            return
        if not d:
            add = n_dig * factorials[d]
            dig_dict[d] = n_dig
            backtrack(0, 0, curr + add, tuple(x - add for x in target_rng))
            dig_dict.pop(d)
            return
        add = factorials[d]
        for f in range(n_dig + 1):
            backtrack(d - 1, n_dig - f, curr, target_rng)
            dig_dict[d] = dig_dict.get(d, 0) + 1
            curr += add
            target_rng = tuple(x - add for x in target_rng)
            if target_rng[1] <= 0: break
        dig_dict.pop(d)
        return
    
    for n_dig in range(n_dig_min, n_dig_max + 1):
        backtrack(base - 1, n_dig, 0,\
                target_rng=(base ** (n_dig - 1), base ** n_dig))
    return res

def digitFactorialSumEqualsSelfSum(n_dig_min: int=2, base: int=10) -> int:
    """
    Solution to Project Euler #34

    Finds the sum of all strictly positive integers such that when
    expressed in the chosen base, contains at least n_dig_min digits
    and the sum of the factorials of the digits is exactly equal to the
    integer itself.
    It is straightforward to show that for any finite base, the number
    of terms in this sum is finite and so the sum itself must be
    finite.
    
    Args:
        Required positional:
        n_dig_min (int): Non-negative integer giving the minimum
                number of digits in the chosen base of the integers
                considered.
        
        Optional named:
        base (int): Strictly positive integer giving the base in
                which the integers in question are expressed for the
                assessment of the number of digits and the calculation
                of the digit factorial sum of the integer.
            Default: 10
    
    Returns:
    The sum over every strictly positive integer such that when
    expressed in the chosen base contains at least n_dig_min digits
    and the sum of factorials of the digits in this expression is
    exactly equal to the integer itself.
    """
    #since = time.time()
    res = sum(digitFactorialSumEqualsSelf(n_dig_min, base=base))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
    
# Problem 35- try to make faster
def circularPrimes(n_max: int, base: int=10) -> Tuple[int]:
    """
    Finds all circular primes for the chosen base not exceeding n_max.
    A circular prime for the chosen base is a strictly positive integer
    for which, when expressed in the chosen base, any integer produced
    by rotating the digits by any amount (including not at all) results
    in a prime number in the chosen base.
    For example, in base 10, 197 is a circular prime because 197,
    971 and 719 (when interpreted in base 10) are all primes.
    
    Args:
        Required positional:
        n_max (int): The largest integer considered.
        
        Optional named:
        base (int): The base in which the integers are to be
                represented and the rotating of digits is to be
                performed.
            Default: 10
    
    Returns:
    Tuple of ints containing every circular prime for the chosen base
    not exceeding n_max in increasing order.
    """
    # From solution to Problem 27
    prime_list = primesUpToN(n_max)
    prime_set = set(prime_list)
    out_set = set()
    for p in prime_list:
        # Checking not already encountered
        if p not in prime_set:
            continue
        
        #digit_list = [int(x) for x in str(p)]
        digit_list = []
        p2 = p
        while p2:
            p2, d = divmod(p2, base)
            digit_list.append(d)
        digit_list = digit_list[::-1]
        #if digit_list[0] != min(digit_list):
        #    continue
        if len(digit_list) > 1 and len({0, 2, 4, 5, 6, 8}\
                .intersection(set(digit_list))) != 0:
            continue
        cycle_list = [p]
        nonprime_found = False
        for i in range(1, len(digit_list)):
            #digit_list = [digit_list[-1], *digit_list[:-1]]
            #cycle_list.append(int("".join([str(x) for\
            #        x in digit_list])))
            num = 0
            for d in digit_list[i:]:
                num = num * base + d
            for d in digit_list[:i]:
                num = num * base + d
            cycle_list.append(num)
            if not nonprime_found and cycle_list[-1] not in prime_set:
                nonprime_found = True
        cycle_set = set(cycle_list)
        prime_set = prime_set.difference(cycle_set)
        if not nonprime_found:
            #print(cycle_list)
            out_set = out_set.union(cycle_set)
        
    return tuple(sorted(list(out_set)))

def circularPrimesCount(n_max: int=10 ** 6 - 1, base: int=10) -> int:
    """
    Solution to Project Euler #35

    Finds the number of circular primes for the chosen base not
    exceeding n_max.
    A circular prime for the chosen base is a strictly positive integer
    for which, when expressed in the chosen base, any integer produced
    by rotating the digits by any amount (including not at all) results
    in a prime number in the chosen base.
    For example, in base 10, 197 is a circular prime because 197,
    971 and 719 (when interpreted in base 10) are all primes.
    
    Args:
        Optional named:
        n_max (int): The largest integer considered.
            Default: 10 ** 6 - 1
        base (int): The base in which the integers are to be
                represented and the rotating of digits is to be
                performed.
            Default: 10
    
    Returns:
    Number of circular primes for the chosen base not exceeding n_max
    (int)
    """
    #since = time.time()
    res = len(circularPrimes(n_max, base=base))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 36
def palindromeGenerator(
    n_dig_mn: int=1,
    n_dig_mx: Optional[int]=None,
    base: int=10,
) -> Generator[int, None, None]:
    """
    Generator iterating over all strictly positive integers in
    increasing order whose representation in the chosen base is
    no less than n_dig_mn digits long, (if n_dig_mx given as not None)
    no more than n_dig_mx digits long and is a palindrome
    An integer is a palindrome when expressed in a given base if the
    digits of that expression of the integer reads the same forwards
    and backwards. Leading zeros are not allowed in such expressions.
    
    Note that if n_dig_mx is not given or is given as None, this
    generator will not itself terminate, so for a loop using this
    generator for which n_dig_mx is not given or given as None, the
    loop should contain a loop breaking statement (e.g. break or
    return), otherwise the program containing that loop will not halt.
    
    Args:
        Optional named:
        n_dig_mn (int): The least number of digits in the chosen base
                allowed for any integer yielded.
            Default: 1
        n_dig_mx (int or None): If given as not None, the most number
                of digits in the chosen base allowed for any integer
                yielded. If not given or given as None, there is no
                upper bound on the size of the integers yielded, so
                the generator will not itself terminate. Therefore,
                if n_dig_mx is given as None for a loop using this
                generator, the loop should include a loop breaking
                statement (e.g. break or return), otherwise the program
                containing that loop will not halt.
            Default: None
        base (int): Strictly positive integers giving the base in which
                the expression of all integers yielded by the generator
                must be palindromes.
            Default: 10
    
    Yields:
    Strictly positive integers representing all the integers whose
    expression in the chosen base is no less than n_dig_mn digits long,
    (if n_dig_mx given as not None) no more than n_dig_mx digits
    long and is a palindrome. These integers are yielded in strictly
    increasing order.
    """
    n_dig_mn = max(n_dig_mn, 1)
    if isinstance(n_dig_mx, int) and n_dig_mx < n_dig_mn:
        return
    
    def palindromeSingle(num: int, odd: bool=True) -> int:
        num2 = num
        if odd: num2 //= base
        res = num
        while num2:
            num2, r = divmod(num2, base)
            res = res * base + r
        return res
    
    def palindromeDouble(num: int) -> Tuple[int]:
        res = (num // base, num)
        while num:
            num, r = divmod(num, base)
            res = tuple(x * base + r for x in res)
        return res
        
    mx = base ** (((n_dig_mn + 1) >> 1) - 1)
    
    if not n_dig_mn & 1:
        mn = mx
        mx = mn * base
        for num in range(mn, mx):
            yield palindromeSingle(num, odd=False)
    iter_obj = range(n_dig_mn >> 1, n_dig_mx >> 1)\
            if isinstance(n_dig_mx, int) else\
            itertools.count(n_dig_mn >> 1)
    for hlf_n_dig in iter_obj:
        mn = mx
        mx = mn * base
        lst = []
        for num in range(mn, mx):
            pair = palindromeDouble(num)
            yield pair[0]
            lst.append(pair[1])
        for pal in lst: yield pal
    if isinstance(n_dig_mx, int) and n_dig_mx & 1:
        mn = mx
        mx = mn * base
        for num in range(mn, mx):
            yield palindromeSingle(num, odd=True)
    return

def multiBasePalindromes(n_max: int, bases: Tuple[int]) -> List[int]:
    """
    Finds all the strictly positive integers no greater than n_max that
    are palindromes when expressed in each of the bases specified.
    These integers are returned as a list given in strictly increasing
    order.
    An integer is a palindrome when expressed in a given base if the
    digits of that expression of the integer reads the same forwards
    and backwards. Leading zeros are not allowed in such expressions.
    
    Args:
        Required positional:
        n_max (int): Strictly positive integer giving the largest
                integer considered.
        bases (tuple of ints): Tuple of strictly positive integers
                giving the bases in which the expression of all
                integers included in the output must be palindromes.
    
    Returns:
    List of ints giving the integers no greater than n_max in
    increasing order whose representation in each of the bases
    specified is a palindrome.
    """
    bases = sorted(bases)
    base0 = bases[-1]
    n_base = len(bases)
    n_dig = 0
    num = n_max
    while num:
        num //= base0
        n_dig += 1
    res = []
    for pal in palindromeGenerator(n_dig_mn=1, n_dig_mx=n_dig - 1,\
            base=base0):
        #print(pal)
        for i in range(n_base - 1):
            # From solution to Problem 4
            if not isPalindromeInteger(pal, base=bases[i]):
                break
        else: res.append(pal)
    for pal in palindromeGenerator(n_dig_mn=n_dig, n_dig_mx=n_dig,\
            base=base0):
        if pal > n_max: break
        #print(pal)
        for i in range(n_base - 1):
            # From solution to Problem 4
            if not isPalindromeInteger(pal, base=bases[i]):
                break
        else: res.append(pal)
    return res

def multiBasePalindromesSum(
    n_max: int=10 ** 6 - 1,
    bases: Tuple[int]=(2, 10),
) -> int:
    """
    Solution to Project Euler #36

    Finds the sum of all the strictly positive integers no greater
    than n_max that are palindromes when expressed in each of the bases
    specified. These integers are returned as a list given in strictly
    increasing order.
    An integer is a palindrome when expressed in a given base if the
    digits of that expression of the integer reads the same forwards
    and backwards. Leading zeros are not allowed in such expressions.
    
    Args:
        Optional named:
        n_max (int): Strictly positive integer giving the largest
                integer considered.
            Default: 10 ** 6 - 1
        bases (tuple of ints): Tuple of strictly positive integers
                giving the bases in which the expression of all
                integers included in the output must be palindromes.
            Default: (2, 10)
    
    Returns:
    Integer giving sum over the integers no greater than n_max in
    increasing order whose representation in each of the bases
    specified is a palindrome.
    """
    #since = time.time()
    res = sum(multiBasePalindromes(n_max, bases))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 37
def isqrt(n: int) -> int:
    """
    For a non-negative integer n, finds the largest integer m
    such that m ** 2 <= n (or equivalently, the floor of the
    positive square root of n).
    Uses Newton's method.
    
    Args:
        Required positional:
        n (int): The number for which the above process is
                performed.
    
    Returns:
    Integer (int) giving the largest integer m such that
    m ** 2 <= n.
    
    Examples:
    >>> isqrt(4)
    2
    >>> isqrt(15)
    3
    """
    x2 = n
    x1 = (n + 1) >> 1
    while x1 < x2:
        x2 = x1
        x1 = (x2 + n // x2) >> 1
    return x2

# Similar to primeFactors in solution to Problem 3, just allowing
# to exit early if proper factor found
def isPrime(num: int):
    """
    Determines whether strictly positive integer num is prime
    
    Args:
        num (int): The strictly positive integer whose primality is
                being checked.
    
    Returns:
    Boolean (bool) giving True if num is prime, False otherwise.
    """
    if num <= 1: return False
    elif num == 2: return True
    if not num & 1: return False
    for p in range(3, isqrt(num) + 1, 2):
        if not num % p: return False
    return True

def leftTruncatablePrimes(
    mx_n_dig: Optional[int]=None,
    base: int=10,
) -> List[int]:
    """
    Finds all left-truncatable primes for the given base, if mx_n_dig
    given as a number, which (if mx_n_dig given as a number), contain
    more than mx_n_dig in its representation in the given base.
    An integer is a left-truncatable prime in a given base if and only
    if it is strictly positive and when expressed in the given base,
    removing any number of digits from the left of this expression
    (including removing no digits) results in a prime number when
    interpreted in the given base.
    For example, 3797 is a left-truncatable prime in base 10 given that
    3797, 797, 97 and 7 are all primes. 
    
    Args:
        Optional named:
        mx_n_dig (int or None): If given and not None, gives the
                largest number of digits allowed when expressed in
                the given base for integers considered. Otherwise,
                all strictly positive integers are considered.
            Default: None
        base (int): Strictly positive integer giving the base in
                which the left-truncation of the integers is performed.
            Default: 10
    
    Returns:
    List of ints giving all the left-truncatable primes for the given
    base in increasing order, with (if mx_n_dig given as not None)
    no more than mx_n_dig digits in when expressed in the given base.
    """
    if isinstance(mx_n_dig, int) and mx_n_dig < 1: return []
    res = []
    curr = []
    for num in range(2, base):
        if not isPrime(num): continue
        if base % num: curr.append(num)
    #print(res)
    #print(curr)
    iter_obj = range(2, mx_n_dig + 1) if isinstance(mx_n_dig, int)\
            else itertools.count(2)
    for n_dig in iter_obj:
        mult = base ** (n_dig - 1)
        prev = curr
        curr = []
        for term in range(1, base):
            add = mult * term
            for num in prev:
                num2 = num + add
                if not isPrime(num2): continue
                curr.append(num2)
                res.append(num2)
        #print(curr)
        if not curr: break
    return res

def rightTruncatablePrimes(
    mx_n_dig: Optional[int]=None,
    base: int=10,
) -> List[int]:
    """
    Finds all right-truncatable primes for the given base, if mx_n_dig
    given as a number, which (if mx_n_dig given as a number), contain
    more than mx_n_dig in its representation in the given base.
    An integer is a right-truncatable prime in a given base if and only
    if it is strictly positive and when expressed in the given base,
    removing any number of digits from the right of this expression
    (including removing no digits) results in a prime number when
    interpreted in the given base.
    For example, 3797 is a right-truncatable prime in base 10 given
    that 3797, 379, 37 and 3 are all primes. 
    
    Args:
        Optional named:
        mx_n_dig (int or None): If given and not None, gives the
                largest number of digits allowed when expressed in
                the given base for integers considered. Otherwise,
                all strictly positive integers are considered.
            Default: None
        base (int): Strictly positive integer giving the base in
                which the right-truncation of the integers is
                performed.
            Default: 10
    
    Returns:
    List of ints giving all the right-truncatable primes for the given
    base in increasing order, with (if mx_n_dig given as not None)
    no more than mx_n_dig digits in when expressed in the given base.
    """
    
    if isinstance(mx_n_dig, int) and mx_n_dig < 1: return []
    append_candidates = [num for num in range(1, base)\
                        if gcd(num, base) == 1]
    res = []
    curr = [num for num in range(1, base) if isPrime(num)]
    #print(res)
    #print(curr)
    iter_obj = range(2, mx_n_dig + 1) if isinstance(mx_n_dig, int)\
            else itertools.count(2)
    for n_dig in iter_obj:
        prev = curr
        curr = []
        for num in prev:
            num2 = num * base
            for term in append_candidates:
                num3 = num2 + term
                if isPrime(num3):
                    curr.append(num3)
                    res.append(num3)
        if not curr: break
    return res

def bothTruncatablePrimes(
    mx_n_dig: Optional[int]=None,
    base: int=10,
) -> List[int]:
    """
    Finds all integers which are left-truncatable primes and
    right-truncatable primes for the given base, which (if mx_n_dig
    given as a number), contain more than mx_n_dig in its
    representation in the given base.
    An integer is a left-truncatable prime in a given base if and only
    if it is strictly positive and when expressed in the given base,
    removing any number of digits from the left of this expression
    (including removing no digits) results in a prime number when
    interpreted in the given base.
    Similarly, an integer is a right-truncatable prime in a given base
    if and only if it is strictly positive and when expressed in the
    given base, removing any number of digits from the right of this
    expression (including removing no digits) results in a prime number
    when interpreted in the given base.
    For example, 3797 is both a left-truncatable and right-truncatable
    prime in base 10 given that 3797, 797, 97, 7, 379, 37, and 3 are
    all primes. 
    
    Args:
        Optional named:
        mx_n_dig (int or None): If given and not None, gives the
                largest number of digits allowed when expressed in
                the given base for integers considered. Otherwise,
                all strictly positive integers are considered.
            Default: None
        base (int): Strictly positive integer giving the base in
                which both the left-truncation and right-truncation
                of the integers is performed.
            Default: 10
    
    Returns:
    List of ints giving all integers which are both left-truncatable
    and right-truncatable primes for the given base in increasing
    order, with (if mx_n_dig given as not None) no more than mx_n_dig
    digits in when expressed in the given base.
    """
    
    if isinstance(mx_n_dig, int) and mx_n_dig < 1: return []
    # Look at right truncatable primes first as the longest length of
    # right truncatable primes tends to be much shorter.
    rt_primes = rightTruncatablePrimes(mx_n_dig=mx_n_dig, base=base)
    mx_n_dig = 0
    num = rt_primes[-1]
    while num:
        mx_n_dig += 1
        num //= base
    lt_primes = leftTruncatablePrimes(mx_n_dig=mx_n_dig, base=base)
    res = []
    i2 = 0
    for i1, p1 in enumerate(lt_primes):
        for i2 in range(i2, len(rt_primes)):
            if rt_primes[i2] >= p1: break
        else: break
        if rt_primes[i2] == p1:
            res.append(p1)
            i2 += 1
    return res
            
def truncatablePrimesSum(
    mx_n_dig: Optional[int]=None,
    left_truncatable: bool=True,
    right_truncatable: bool=True,
    base: int=10,
) -> int:
    """
    Solution to Project Euler #37

    Finds the sum of all integers which are (if left_truncatable given
    as True) left-truncatable primes and (if right_truncatable given
    as True) right-truncatable primes for the given base, which
    (if mx_n_dig given as a number), contain more than mx_n_dig in its
    representation in the given base.
    An integer is a left-truncatable prime in a given base if and only
    if it is strictly positive and when expressed in the given base,
    removing any number of digits from the left of this expression
    (including removing no digits) results in a prime number when
    interpreted in the given base.
    Similarly, an integer is a right-truncatable prime in a given base
    if and only if it is strictly positive and when expressed in the
    given base, removing any number of digits from the right of this
    expression (including removing no digits) results in a prime number
    when interpreted in the given base.
    For example, 3797 is both a left-truncatable and right-truncatable
    prime in base 10 given that 3797, 797, 97, 7, 379, 37, and 3 are
    all primes. 
    
    Args:
        Optional named:
        mx_n_dig (int or None): If given and not None, gives the
                largest number of digits allowed when expressed in
                the given base for integers considered. Otherwise,
                all strictly positive integers are considered.
            Default: None
        left_truncatable (bool): If True, the returned integers must
                be left-truncatable primes in the given base,
                otherwise they are not required to be left-truncatable.
                Note that at least one of left_truncatable and
                right_truncatable must be given as True.
            Default: True
        right_truncatable (bool): If True, the returned integers must
                be right-truncatable primes in the given base,
                otherwise they are not required to be
                right-truncatable.
                Note that at least one of left_truncatable and
                right_truncatable must be given as True.
            Default: True
        base (int): Strictly positive integer giving the base in
                which both the left-truncation and right-truncation
                of the integers is performed.
            Default: 10
    
    Returns:
    List of ints giving all integers which are (if left_truncatable
    is given as True) left-truncatable and (if right_truncatable is
    given as True) right-truncatable primes for the given base in
    increasing order, with (if mx_n_dig given as not None) no more than
    mx_n_dig digits in when expressed in the given base.
    """
    #since = time.time()
    if not left_truncatable and not right_truncatable:
        raise ValueError("At least one of the input arguments "
                "left_truncatable and right_truncatable must be given "
                "as True.")
    if not left_truncatable:
        res = sum(rightTruncatablePrimes(mx_n_dig=mx_n_dig, base=base))
    elif not right_truncatable:
        res = sum(leftTruncatablePrimes(mx_n_dig=mx_n_dig, base=base))
    else:
        res = sum(bothTruncatablePrimes(mx_n_dig=mx_n_dig, base=base))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
            
# Problem 38
# Review- try to simplify and make more efficient
def pandigitalGenerator(
    n_dig: Optional[int]=None,
    min_dig: int=1,
    max_dig: Optional[int]=None,
    base: int=10,
    reverse: bool=False,
) -> Generator[int, None, None]:
    """
    Generator that iterates over all the integers whose representation
    in the given base contains exactly n_dig digits (with no leading
    zeros) and contains at most one of each of the digits between
    min_dig and max_dig inclusive and no other digits. If reverse is
    given as False, these are yielded in order of increasing value,
    while if reverse is given as True, they are yielded in order of
    decreasing value.
    
    Args:
        Optional named:
        n_dig (int): Strictly positive integer giving the number of
                digits in the representation of each integer yielded in
                the chosen base (with no leading zeros)
            Default: min_dig - max_dig + 1
        min_dig (int): Non-negative integer giving the smallest digit
                in the range of digits which should each appear at most
                once in each integer yielded.
                Should be between 0 and (base - 1) inclusive.
            Default: 1
        max_dig (int): Strictly positive integer giving the largest
                digit in the range of digits which should each appear
                at most once in each integer yielded.
                Should be between (min_dig + 1) and (base - 1)
                inclusive.
            Default: base - 1
        base (int): Strictly positive integer giving the base in which
                the integers are to be expressed.
            Default: 10
        reverse (bool): Whether the integers produced by the generator
                should be yielded in order of increasing value (reverse
                given as True) or in order of decreasing value (reverse
                given as False)
            Default: False
    
    Yields:
    Integers (int) collectively giving all the integers whose
    representation in the given base that contain exactly n_dig digits
    (with no leading zeros) and contain at most one of each of the
    digits between min_dig and max_dig inclusive and no other digits.
    If reverse is given as False, these are yielded in order of
    increasing value, while if reverse is given as True, they are
    yielded in order of decreasing value.
    """
    if min_dig >= base:
        raise ValueError("Argument min_dig must be strictly less "
                "than base.")
    if max_dig is None: max_dig = base - 1
    elif max_dig >= base or max_dig < min_dig:
        raise ValueError("Argument max_dig must be no less than "
                "min_dig and strictly less than base.")
    if n_dig is None: n_dig = max_dig - min_dig + 1
    elif n_dig > max_dig - min_dig + 1 or not n_dig:
        raise ValueError("Argument n_dig must be strictly positive "
                "and no greater than (max_dig - min_dig + 1).")
    
    iter_func = (lambda x: reversed(x)) if reverse else (lambda x: x)
    
    if n_dig == 1:
        for num in iter_func(range(min_dig, max_dig + 1)):
            yield num
        return
    
    remain = SortedList(range(min_dig, max_dig + 1))
    has_zero = not min_dig
    def recur(curr: int=0, dig_count: int=0)\
            -> Generator[int, None, None]:
        if dig_count == n_dig:
            yield curr
            return
        for d in iter_func(list(remain)):
            remain.remove(d)
            yield from recur(curr=curr * base + d,\
                    dig_count=dig_count + 1)
            remain.add(d)
        return
            
    for d in iter_func(range(max(1, min_dig), max_dig + 1)):
        remain.remove(d)
        yield from recur(curr=d, dig_count=1)
        remain.add(d)
    return

def multiplesCollectivelyPandigital(
    n_mult: int,
    min_dig: int=1,
    max_dig: Optional[int]=None,
    base: int=10,
) -> Set[int]:
    """
    Finds all strictly positive integers a such that collectively the
    representations of the first n_mult multiples of a (i.e. a * 1,
    a * 2, ..., a * n_mult) in the chosen base (without leading zeros)
    between them contain the digits from min_dig to max_dig inclusive
    exactly once and no other digits.
    For example, for n_mult = 3, min_dig = 1, max_dig = 9 and
    base = 10, the integer 192 is one such integer given that, when
    represented in base 10 the first 3 multiples of 192 are:
        192 * 1 = 192, 192 * 2 = 384,  192 * 3 = 576
    and the representations of numbers 192, 384, 576 in base 10 between
    them contain the digits between 1 and 9 inclusive exactly once and
    no other digits.
    
    Args:
        Required positional:
        n_mult (int): Strictly positive integer giving the number of
                multiples of the initial multiples of the integer
                considered.
        
        Optional named:
        min_dig (int): Non-negative integer giving the smallest digit
                in the range of digits which should each appear at most
                once between the representations of the multiples of
                the integer in question in the chosen base.
                Should be between 0 and (base - 1) inclusive.
            Default: 1
        max_dig (int): Strictly positive integer giving the largest
                digit in the range of digits which should each appear
                at most once between the representations of the
                multiples of the integer in question in the chosen
                base.
                Should be between (min_dig + 1) and (base - 1)
                inclusive.
            Default: base - 1
        base (int): Strictly positive integer giving the base in which
                the integers are to be expressed.
            Default: 10
    
    Returns:
    Set (set) of 2-tuples, each of whose index 0 contains the integer
    whose multiples up to n_mult when expressed in the chosen base
    (with no leading zeros) between them contain the digits from
    min_dig to max_dig inclusive exactly once and no other digits, and
    whose index 1 contains a n_mult-tuple containing those multiples
    in increasing order.
    Collectively, these represent all possible strictly positive
    integers satisfying the given requirements.
    """
    if max_dig is None: max_dig = base - 1
    tot_n_dig = max_dig - min_dig + 1
    #print(tot_n_dig)
    # Working out range of the possible number of digits
    def digitCountMeetsTarget(a: int, n_mult: int, target_n_dig: int)\
            -> bool:
        n_dig_sum = 0
        num = a
        
        mn_n_dig = 0
        mn = a
        num = mn
        while num:
            num //= base
            mn_n_dig += 1
        target_n_dig -= mn_n_dig * n_mult
        if target_n_dig <= 0:
            return True
        elif n_mult == 1: return False
        mn_div = base ** mn_n_dig
        
        mx_extra_dig = 0
        mx = a * n_mult
        num = mx // mn_div
        while num:
            num //= base
            mx_extra_dig += 1
        target_n_dig -= mx_extra_dig
        if target_n_dig <= 0:
            return True
        if mx_extra_dig * (n_mult - 2) < target_n_dig:
            return False
        
        a2 = a
        for mult in range(2, n_mult):
            a2 += mult
            num = a2 // mn_div
            extra = 0
            while num:
                num //= base
                extra += 1
            target_n_dig -= extra * (n_mult - mult)
            if target_n_dig <= 0: return True
            mx_extra_dig -= extra
            mn_div *= base ** extra
            if mx_extra_dig * (n_mult - mult - 1) < target_n_dig:
                return False
        return False
    
    lower = None
    upper = tot_n_dig + 1
    a1, a2 = 0, 0
    for n_dig in range(1, tot_n_dig + 1):
        a2 = a2 * base + (min_dig + n_dig - 1)
        if lower is None:
            a1 = a1 * base + (max_dig - n_dig + 1)
            if digitCountMeetsTarget(a1, n_mult, tot_n_dig):
                lower = n_dig
            else: continue
        if digitCountMeetsTarget(a2, n_mult, tot_n_dig + 1):
            upper = n_dig
            break
    #print(f"{n_mult}, {lower}, {upper}")
    if lower >= upper:
        # The correct number of digits (9) cannot be achieved so there
        # are not possible such integers
        return set()
    res = set()
    for n_dig in range(lower, upper):
        for a in pandigitalGenerator(n_dig=n_dig, min_dig=min_dig,\
                max_dig=max_dig, base=base, reverse=False):
            remain = set(range(min_dig, max_dig + 1))
            num2 = a
            while num2:
                num2, d = divmod(num2, base)
                remain.remove(d)
            num = a << 1
            nums = [a]
            for mult in range(2, n_mult + 1):
                num2 = num
                while num2:
                    num2, d = divmod(num2, base)
                    if d not in remain: break
                    remain.remove(d)
                else:
                    nums.append(num)
                    num += a
                    continue
                break
            else:
                if remain: continue
                res.add((a, tuple(nums)))
    return res
    
def multiplesConcatenatedPandigitalMaxProperties(
    min_n_mult: int=2,
    min_dig: int=1,
    max_dig: Optional[int]=None,
    base: int=10,
) -> Tuple[int, int, int]:
    """
    Finds the largest min_dig to max_dig pandigital number (i.e. a
    non-negative integer whose representation in the chosen base
    without leading zeros contains each of the digits between min_dig
    and max_dig inclusive exactly once and no other digits) such that
    there exists strictly positive integers a and n where
    n >= min_n_mult and the concatenation of the representations of
    the first n multiples of a (i.e. a * 1, a * 2, ..., a * n) in the
    chosen base in that order is the same as the representation of the
    min_dig to max_dig pandigital number in the chosen base. The value
    of this pandigital number is returned alongside the corresponding
    values of a and n.

    For example, for min_n_mult = 2, min_dig = 1, max_dig = 9 and
    base = 10, the 1 to 9 pandigital number 192384576 is a candidate
    for the solution given that, when represented in base 10, the first
    3 multiples of 192 are:
        192 * 1 = 192, 192 * 2 = 384,  192 * 3 = 576
    and the concatenation of the representations of numbers 192, 384,
    576 in base 10 is 192384576, so there exists positive integers
    a and n (namely a = 192 and n = 3 >= 2) as described for this 1 to
    9 pandigital number.
    
    Args:
        Optional named:
        min_n_mult (int): Strictly positive integer giving the smallest
                number of concatenated multiples considered.
            Default: 2
        min_dig (int): Non-negative integer giving the smallest digit
                in the range of digits which should each appear at most
                once in the representation of the solution in the
                chosen base.
                Should be between 0 and (base - 1) inclusive.
            Default: 1
        max_dig (int): Strictly positive integer giving the largest
                digit in the range of digits which should each appear
                at most once in the representation of the solution in
                the chosen base.
                Should be between (min_dig + 1) and (base - 1)
                inclusive.
            Default: base - 1
        base (int): Strictly positive integer giving the base in which
                the solution is to be min_dig to max_dig pandigital and
                in which the multiples are to be expressed when
                concatenated.
            Default: 10
    
    Returns:
    3-tuple of ints, whose index 0 contains the largest min_dig to
    max_dig pandigital number satisfying the requirements and whose
    indices 1 and 2 contain a pair of values of a and n respectively
    such that n > 1 and the concatenation of the representations of
    a * 1, a * 2, ..., a * n in the chosen base in that order is the
    same as the representation of the min_dig to max_dig pandigital
    number in the chosen base contained in index 0. If no such
    pandigital number exists, returns an empty tuple.
    """
    # Review- wording of documentation for clarity
    #since = time.time()
    if max_dig is None: max_dig = base - 1
    res = (-float("inf"), None, None)
    n_dig = max_dig - min_dig + 1
    if min_n_mult <= 1:
        for num in pandigitalGenerator(n_dig=n_dig,\
                min_dig=min_dig, max_dig=max_dig, base=base,\
                reverse=True):
            res = (num, num, 1)
            break
    
    for n_mult in range(max(2, min_n_mult), n_dig + 1):
        #print(f"n_mult = {n_mult}")
        pandig_products = multiplesCollectivelyPandigital(n_mult,\
                min_dig=min_dig, max_dig=max_dig, base=base)
        if not pandig_products: continue
        #print(pandig_products)
        for pair in pandig_products:
            curr = 0
            for num2 in pair[1]:
                mult = 1
                while mult <= num2:
                    mult *= base
                    curr *= base
                curr += num2
            if curr > res[0]:
                res = (curr, pair[0], n_mult)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    if not isinstance(res[0], int): return ()
    return res

def multiplesConcatenatedPandigitalMax(
    min_n_mult: int=2,
    min_dig: int=1,
    max_dig: Optional[int]=None,
    base: int=10,
) -> int:
    """
    Finds the largest min_dig to max_dig pandigital number (i.e. a
    non-negative integer whose representation in the chosen base
    without leading zeros contains each of the digits between min_dig
    and max_dig inclusive exactly once and no other digits) such that
    there exists strictly positive integers a and n where
    n >= min_n_mult and the concatenation of the representations of
    the first n multiples of a (i.e. a * 1, a * 2, ..., a * n) in the
    chosen base in that order is the same as the representation of the
    min_dig to max_dig pandigital number in the chosen base.

    For example, for min_n_mult = 2, min_dig = 1, max_dig = 9 and
    base = 10, the 1 to 9 pandigital number 192384576 is a candidate
    for the solution given that, when represented in base 10, the first
    3 multiples of 192 are:
        192 * 1 = 192, 192 * 2 = 384,  192 * 3 = 576
    and the concatenation of the representations of numbers 192, 384,
    576 in base 10 is 192384576, so there exists positive integers
    a and n (namely a = 192 and n = 3 >= 2) as described for this 1 to
    9 pandigital number.
    
    Args:
        Optional named:
        min_n_mult (int): Strictly positive integer giving the smallest
                number of concatenated multiples considered.
            Default: 2
        min_dig (int): Non-negative integer giving the smallest digit
                in the range of digits which should each appear at most
                once in the representation of the solution in the
                chosen base.
                Should be between 0 and (base - 1) inclusive.
            Default: 1
        max_dig (int): Strictly positive integer giving the largest
                digit in the range of digits which should each appear
                at most once in the representation of the solution in
                the chosen base.
                Should be between (min_dig + 1) and (base - 1)
                inclusive.
            Default: base - 1
        base (int): Strictly positive integer giving the base in which
                the solution is to be min_dig to max_dig pandigital and
                in which the multiples are to be expressed when
                concatenated.
            Default: 10
    
    Returns:
    Integer (int), giving the largest min_dig to max_dig pandigital number
    in the chosen base satisfying the requirements given above. If no
    such  number exists, returns -1.
    """
    res = multiplesConcatenatedPandigitalMaxProperties(
        min_n_mult=min_n_mult,
        min_dig=min_dig,
        max_dig=max_dig,
        base=base,
    )
    return res[0] if res else -1

# Problem 39
def pythagTriple(k: int, m: int, n: int) -> Tuple[int]:
    """
    Calculates the Pythagorean triple based on Euclid's formula, for
    strictly positive integers k, m, n such that m > n, exactly one
    of m and n is even and gcd(m, n) = 1:
        a' = k * (m ** 2 - n ** 2), b' = k * (2 * m * n),
        c = k * (m ** 2 + n ** 2),
        a = min(a', b'), b = max(a', b')
    It can be shown that every Pythagorean triple (a, b, c) where
    a, b, c are strictly positive integers and a < b < c corresponds
    to a unique triple of strictly positive integers (k, m, n) such
    that m > n, exactly one of m and n is even and gcd(m, n) = 1 by
    these formulas and conversely, each distinct triple of strictly
    positive integers (k, m, n) such that m > n, exactly one of
    m and n is even and gcd(m, n) corresponds to a distinct Pythagorean
    triple (a, b, c) where a, b, c are strictly positive integers and
    a < b < c.
    
    Args:
        k (int): The strictly positive integer k in the above
                equations.
        m (int): The strictly positive integer m in the above
                equations.
        n (int): The strictly positive integer n in the above
                equations. Must be strictly less than m, a different
                parity to m (i.e. if m is even, n is odd and if m is
                odd n is even) and gcd(m, n) = 1
    
    Returns:
    3-tuple of integers giving the Pythagorean triple (a, b, c) where
    a < b < c corresponding to the given values of k, m, n.
    Note that index 2 of the returned tuple represents the length of
    the hypotenuse of the corresponding right-angled triangle.
    """
    a_, b_, c = (k * (m ** 2 - n ** 2), 2 * k * m * n,\
            k * (m ** 2 + n ** 2))
    return (min(a_, b_), max(a_, b_), c)

def pythagTripleMaxSolsPerim(perim_max: int=1000):
    """
    Solution to Project Euler #39

    Calculates the strictly positive integer not exceeding perim_max
    that is equal to the perimeter of at least as many distinct
    right-angled triangles with integer side lengths (i.e. sums of
    Pythagorean triples) as any other strictly positive integer not
    exceeding perim_max and more distinct right-angled triangles as
    any smaller strictly positive integer.
    Two right-angled triangles are considered the same if and only
    if the lists of their side lengths in increasing order is the
    same for both triangles.
    
    Args:
        perim_max (int): Strictly positive integer giving the largest
                perimeter of right-angled triangles considered.
            Default: 1000
    
    Returns:
    Integer (int) no greater than perim_max such that there are no
    other such integers for which there exist more distinct
    right-angled triangles with integer side lengths and perimeter
    equal to the integer in question, and there are no integers
    smaller than that integer for which there exist as many
    such right-angled triangles with a perimeter equal to the
    integer in question.
    
    Outline of rationale:
    Based on the formula for the general Pythagorean triple, for
    any Pythagorean triple there exists natural numbers m, n, k
    such that m > n, exactly one of m and n is even and
    gcd(m, n) = 1 for which the perimeter is
     perim = 2 * k * m * (m + n)
    As any of m, n or k increase keeping the other values
    constant, it is clear that the perimeter increases.
    Thus, for given m, n, perim >= 2 * m * (m + n) and for
    given m, perim >= 2 * m * (m + 1)
    """
    #since = time.time()
    perim_fdict = {}
    for m in range(1, perim_max + 1):
        if 2 * m * (m + 1) > perim_max:
            break
        for n in range(1 + (m & 1), m, 2):
            if gcd(m, n) > 1: continue
            base_val = 2 * m * (m + n)
            for k in range(1, perim_max // base_val + 1):
                perim = k * base_val
                perim_fdict[perim] = perim_fdict.get(perim, 0) + 1
    f_mx = 0
    for perim in sorted(perim_fdict.keys()):
        f = perim_fdict[perim]
        if f <= f_mx: continue
        res = perim
        f_mx = f
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 40
def champernowneConstantNthDigit(n: int, base: int=10) -> int:
    """
    Finds the the nth basimal digit after the basimal point of
    Champernowne's constant for the given base (where n = 1 refers
    to the digit directly to the right of the basimal point).
    Champernowne's constant for a given base is the irrational number
    whose expression in the chosen base is constructed by
    concatentating the natural numbers as expressed in the chosen base
    after the basimal point, starting with 1.
    For example, the first several decimal places of Champenowne's
    constant for base 10 is:
        0.123456789101112131415161718192021...
    
    Args:
        Required positional:
        n (int): n strictly positive integer specifying the position
                of the digit after the basimal point that should be
                returned where position 1 refers to the digit directly
                following the basimal point.
        
        Optional named:
        base (int): The base of Champernowne's constant referred to
                and the base in which the constant is expressed in
                when the nth basimal digit after the basimal point
                is identified.
            Default: 10
    
    Returns:
    Integer (int) giving the nth digit after the basimal point of
    Champernowne's constant for the given base, when represented
    in the given base.
    """
    n_digits = 0
    start = start_prov = 1
    while start_prov <= n:
        n_digits += 1
        start = start_prov
        start_prov = start + (base - 1) * base ** (n_digits - 1) * n_digits
        #print(start_prov)
    (m, pos) = divmod(n - start, n_digits)
    #print(n_digits, m, pos)
    pos = n_digits - pos
    m += base ** (n_digits - 1)
    for i in range(pos - 1):
        m //= base
    return m % base
    #print(str(m + 10 ** (n_digits - 1)))
    #return int(str(m + 10 ** (n_digits - 1))[pos])

def champernowneConstantProduct(
    n_list: List[int]=[10 ** x for x in range(7)],
    base: int=10,
) -> int:
    """
    Solution to Project Euler #40

    Calculates the product of each nth digit of Champernowne's
    constant for the given base after the basimal point when
    represented in that base over the integers n in n_list (where
    n = 1 refers to the digit directly following the basimal point).
    For the definition of Champernowne's constant for a given base,
    see the documentation of the function
    champernowneConstantNthDigit().
    
    Args:
        Optional named:
        n_list (list or tuple of ints): List of strictly positive
                integers representing the positions of the digits
                of Champernowne's constant for the given base after
                the basimal point when expressed in that base that
                are to be included in the product, where position 1
                refers to the digit directly following the basimal
                point.
            Default: (1, 10, 100, 1000, 10000, 100000, 1000000)
        base (int): The base of Champernowne's constant referred to
                and the base in which the constant is expressed in
                when the digits after the basimal point at the
                positions given by n_list are identified.
            Default: 10
    
    Returns:
    Integer (int) giving the product of the specified digits of
    Champernowne's constant for the given base when represented in that
    base.
    """
    #since = time.time()
    prod = 1
    for n in n_list:
        prod *= champernowneConstantNthDigit(n, base=base)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return prod

# Problem 41
def largestNDigitPandigitalPrime(
    min_dig: int=1,
    max_dig: Optional[int]=None,
    base: int=10
) -> int:
    """
    Finds the largest prime whose representation in the given base
    contains the digits between min_dig and max_dig exactly once and
    no other digits (with no leading zeros), if such a prime exists
    
    Args:
        Optional named:
        min_dig (int): Non-negative integer giving the smallest digit
                in the range of digits which should each appear exactly
                once in the prime returned.
                Should be between 0 and (base - 1) inclusive.
            Default: 1
        max_dig (int): Strictly positive integer giving the largest
                digit in the range of digits which should each appear
                exactly once in the prime returned.
                Should be between (min_dig + 1) and (base - 1)
                inclusive.
            Default: base - 1
        base (int): Strictly positive integer giving the base in which
                the integers considered are to be expressed.
            Default: 10
    
    Returns:
    Integer (int) giving the largest prime number satisfying the
    given requirements, otherwise 0.
    """
    
    if max_dig is None: max_dig = base - 1
    n_dig = max_dig - min_dig + 1
    
    # Ruling out certain combinations of digits whose digit sums
    # guarantee that any integer containing these digits when
    # represented in the chosen base cannot be prime. In base 10,
    # this is ruling out any combination of digits whose sum
    # is a multiple of 3, as this guarantees that any integer
    # composed of these digits is divisible by 3 and so (unless
    # it is itself 3) is not a prime.
    dig_sum = (max_dig * (max_dig + 1) - min_dig * (min_dig - 1)) >> 1
    if n_dig > 1 and gcd(dig_sum, base - 1) != 1: return 0
    # From solution to Problem 38
    for num in pandigitalGenerator(n_dig=n_dig, min_dig=min_dig,\
            max_dig=max_dig, base=base, reverse=True):
        if isPrime(num): break
    else: return 0
    return num
    
def largestPandigitalPrime(base: int=10, incl_zero: bool=False) -> int:
    """
    Solution to Project Euler #41

    Finds the largest 1 to n (or if incl_zero given as True, 0 to n)
    pandigital number in the chosen base that is prime, for any
    strictly positive integer (or non-negative integer if incl_zero
    given as True) n strictly less than base.
    A m to n pandigital number in a given base and m and n non-negative
    integers such that m <= n < base is a non-negative integer whose
    representation in the given base contains each of the digits from m
    to n inclusive exactly once and no other digits.
    
    Args:
        Optional named:
        base (int): Strictly positive integer giving the base in which
                the primes considered are to be expressed when
                assessing their pandigital status.
            Default: 10
        incl_zero (bool): Whether the prime returned should be 1 to n
                pandigital (if incl_zero given as False) or 0 to n
                pandigital for some integer n.
            Default: False
    
    Returns:
    Integer (int) giving the largest 1 to n (or, if incl_zero given as
    True, 0 to n) pandigital number in the chosen base that is prime
    for any strictly positive integer (or non-negative integer if
    incl_zero given as True) n strictly less than base, if such a
    number exists, otherwise 0.
    """
    #since = time.time()
    min_dig = int(not incl_zero)
    res = 0
    for i in reversed(range(min_dig, base)):
        res = largestNDigitPandigitalPrime(min_dig=min_dig,\
                max_dig=i, base=base)
        if res: break
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 42
def isSquare(num: int) -> bool:
    """
    For a given integer determines whether it is the square of an
    integer.
    
    Args:
        Required positional:
        num (int): The integer being assessed as to whether it is a
                the square of an integer
    
    Returns:
    Boolean (bool) with value True if num is the square of an integer
    and False if it is not the square of an integer.
    """
    if num < 0: return False
    sqrt = isqrt(num)
    return sqrt * sqrt == num

def isTriangleNumber(num: int):
    """
    Determines whether an integer num is a triangle number.
    A triangle number is one which is equal to the sum of all strictly
    positive integers not exceeding n for some strictly positive
    integer n.
    
    Args:
        Required positional:
        num (int): The integer whose status as a triangle number is
                being determined.
    
    Returns:
    Boolean with value True if num is a triangle number, otherwise
    False.
    
    Outline of rationale:
    Uses that a number n is a triangle number if and only if num is
    a strictly positive integer and (8 * num + 1) is a square number.
    """
    return num > 0 and isSquare(8 * num + 1)

def isTriangleWord(word: str) -> bool:
    """
    For the string containing only alphabet characters word determines
    whether it is a triangle word.
    A triangle word is any word such that when its letters are
    converted to numbers based on their position in the alphabet
    ignoring case (i.e. A or a is 1, B or b is 2, ..., Z or z is 26)
    and the resulting numbers are summed, this sum is a triangle
    number.
    A triangle number is one which is equal to the sum of all strictly
    positive integers not exceeding n for some integer n.
    
    Args:
        Required positional:
        word (str): The string containing only alphabet characters
                whose status as a triangle word is being determined.
    
    Returns:
    Boolean with value True if word is a triangle word, otherwise
    False.
    """
    num = 0
    for l in word.strip():
        if not l.isalpha():
            raise ValueError("The input argument word may only "
                    "contain alphabet characters.")
        # Using letterAlphabetValue from solution to Problem 22
        num += letterAlphabetValue(l, a_val=1)
    #print(out)
    #print(num, isTriangleNumber(num))
    return isTriangleNumber(num)

def countTriangleWordsInTxtDoc(
    doc: str="project_euler_problem_data_files/p042_words.txt",
    rel_package_src: bool=True,
) -> int:
    """
    Solution to Project Euler #42

    For a .txt file at location given by doc containing a list
    of words containing only alphabet characters which are separated
    by a single comma (',') and each word surrounded by double
    quotation marks ('"'), reads the words from the file and determines
    how many of the total number of words in the file are triangle
    words (as defined in the description of the function
    isTriangleWord()).
    
    Args:
        Optional named:
        doc (str): The relative or absolution location of the .txt
                file containing the list of words.
            Default: "project_euler_problem_data_files/p042_words.txt"
        rel_package_src (bool): Whether a relative path given by doc
                is relative to the current directory (False) or
                the package src directory (True).
            Default: True
    
    Returns:
    Integer (int) giving the number of words in the .txt file at
    location doc that are triangle words.
    """
    # Using loadStrings() from solution to Project Euler #22
    #since = time.time()
    words = loadStrings(doc, rel_package_src=rel_package_src)
    n_triangle = 0
    for word in words:
        n_triangle += isTriangleWord(word)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return n_triangle

# Problem 43
def pandigitalDivProps(
    subnum_n_dig: int,
    n_prime: int,
    min_dig: int=0,
    max_dig: Optional[int]=None,
    base: int=10,
) -> Set[int]:
    """
    Finds all min_dig to max_dig pandigital numbers in the chosen base
    (i.e. non-negative integers such that their representation in the
    chosen base with no leading zeros contain the digits between
    min_dig and max_dig inclusive and no other digits) that have
    the following property.
    Let m be equal to (max_dig - min_dig - n_prime - subnum_n_dig + 2).
    Then for each integer n between 1 and n_prime inclusive, the
    integer whose representation in the chosen base is formed by
    concatenating the (m + n)th to the (m + n + subnum_n_dig - 1)th
    digit (inclusive) of the pandigital number is divisible by the mth
    prime (where the 1st prime is 2, the 2nd is 3, the 3rd is, etc.).
    For example, for subnum_n_dig = 3, n_prime = 6, min_dig = 0,
    max_dig = 9 and base = 10, the pandigital number 1406357289
    satisfies this property since (noting that
    m = 9 - 0 - 6 - 3 + 2 = 2, and all numbers are expressed in base
    10):
        406 is divisible by 2
        063 is divisible by 3
        635 is divisible by 5
        357 is divisible by 7
        572 is divisible by 11
        728 is divisible by 13
        289 is divisible by 17
    
    Args:
        Required positional:
        subnum_n_dig (int): Strictly positive integer giving the number
                of digits in the concatenations of digits that must be
                divisible by some prime.
        n_prime (int): Non-negative integer giving the number of the
                initial prime numbers that must divide a specific
                concatentation of digits of any number returned.
        
        Optional named:
        min_dig (int): Non-negative integer giving the smallest digit
                in the range of digits which should each appear exactly
                once in any number returned.
                Should be between 0 and (base - 1) inclusive.
            Default: 1
        max_dig (int): Strictly positive integer giving the largest
                digit in the range of digits which should each appear
                exactly once in any number returned.
                Should be between (min_dig + 1) and (base - 1)
                inclusive.
            Default: base - 1
        base (int): Strictly positive integer giving the base in which
                the integers considered are to be expressed and the
                integers formed by concatenations of digits of this
                number are to be interpreted.
            Default: 10
    
    Returns:
    Set of ints giving all the min_dig to max_dig pandigital numbers
    that satisfy the stated property.
    """
    if max_dig is None: max_dig = base - 1
    dig_set = set(range(min_dig, max_dig + 1))
    n_dig = len(dig_set)
    if n_prime + subnum_n_dig - 1 > n_dig:
        return set()
    
    # From solution to Problem 7
    p_lst = firstNPrimes(n_prime)
    
    res = set()
    mn = base ** (subnum_n_dig - 1)
    mx = mn * base
    
    def recur(p_i: int, curr: int) -> None:
        if p_i >= n_prime:
            if not dig_set:
                res.add(curr)
            elif dig_set != {0}:
                for d in list(dig_set):
                    dig_set.remove(d)
                    recur(p_i, curr +\
                            d * base ** (n_dig - len(dig_set) - 1))
                    dig_set.add(d)
            return
        num = base * (curr % (base ** (subnum_n_dig - 1)))
        curr *= base
        for d in range((-num) % p_lst[p_i], base, p_lst[p_i]):
            if d not in dig_set: continue
            dig_set.remove(d)
            recur(p_i + 1, curr + d)
            dig_set.add(d)
        return
    
    for num in range(-((-mn) // p_lst[0]) * p_lst[0], mx, p_lst[0]):
        num2 = num
        num_digs = set()
        while num2:
            num2, r = divmod(num2, base)
            if r in num_digs or r not in dig_set: break
            num_digs.add(r)
        else:
            dig_set -= num_digs
            recur(1, num) 
            dig_set |= num_digs
    return res

def pandigitalDivPropsSum(
    subnum_n_dig: int=3,
    n_prime: int=7,
    min_dig: int=0,
    max_dig: Optional[int]=None,
    base: int=10,
) -> int:
    """
    Solution to Project Euler #43

    Finds the sum of all min_dig to max_dig pandigital numbers in the
    chosen base (i.e. non-negative integers such that their
    representation in the chosen base with no leading zeros contain the
    digits between min_dig and max_dig inclusive and no other digits)
    that have the following property.
    Let m be equal to (max_dig - min_dig - n_prime - subnum_n_dig + 2).
    Then for each integer n between 1 and n_prime inclusive, the
    integer whose representation in the chosen base is formed by
    concatenating the (m + n)th to the (m + n + subnum_n_dig - 1)th
    digit (inclusive) of the pandigital number is divisible by the mth
    prime (where the 1st prime is 2, the 2nd is 3, the 3rd is, etc.).
    For example, for subnum_n_dig = 3, n_prime = 6, min_dig = 0,
    max_dig = 9 and base = 10, the pandigital number 1406357289
    satisfies this property since (noting that
    m = 9 - 0 - 6 - 3 + 2 = 2, and all numbers are expressed in base
    10):
        406 is divisible by 2
        063 is divisible by 3
        635 is divisible by 5
        357 is divisible by 7
        572 is divisible by 11
        728 is divisible by 13
        289 is divisible by 17
    
    Args:
        Required positional:
        subnum_n_dig (int): Strictly positive integer giving the number
                of digits in the concatenations of digits that must be
                divisible by some prime.
        n_prime (int): Non-negative integer giving the number of the
                initial prime numbers that must divide a specific
                concatentation of digits of any number returned.
        
        Optional named:
        min_dig (int): Non-negative integer giving the smallest digit
                in the range of digits which should each appear exactly
                once in any number returned.
                Should be between 0 and (base - 1) inclusive.
            Default: 1
        max_dig (int): Strictly positive integer giving the largest
                digit in the range of digits which should each appear
                exactly once in any number returned.
                Should be between (min_dig + 1) and (base - 1)
                inclusive.
            Default: base - 1
        base (int): Strictly positive integer giving the base in which
                the integers considered are to be expressed and the
                integers formed by concatenations of digits of this
                number are to be interpreted.
            Default: 10
    
    Returns:
    Integer (int) giving the sum of all the min_dig to max_dig
    pandigital numbers that satisfy the stated property.
    """
    #since = time.time()
    if max_dig is None: max_dig = base - 1
    res = sum(pandigitalDivProps(subnum_n_dig, n_prime,\
            min_dig=min_dig, max_dig=max_dig, base=base))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 44
# Review- This has an alternative, much faster method than that
# currently being used (based on solution to Pell's equation),
# but it has not been rigerously proven to always give the smallest
# possible solution, though it gives does give the correct answer
# for Project Euler #44. However, such a proof is likely
# to be extremely difficult to find (assuming the method does
# indeed give the smallest solution).
def nthConvergent(n: int, cf_func: Callable[[int], int]) -> Tuple[int]:
    """
    Finds the nth convergent of a given continued fraction
    representation of a non-negative number, with terms as
    given in cf_func()
    
    Args:
        Required positional:
        n (int): Strictly positive integer giving the
                convergent of the continued fraction is to
                be calculated
        cf_func (callable): A function accepting a single int
                as an argument. For given i, this should
                return the ith value in the 0-indexed continued
                fraction sequence. If the sequence has
                terminated before the ith index, it should
                return -1. Note that cf_func(0) must be a
                non-negative integer.
    
    Returns:
    2-tuple of ints where index 0 is the numerator and
    index 1 is the denominator
    """
    if n < 1: raise ValueError("n should be strictly positive")
    for i in reversed(range(n)):
        a = cf_func(i)
        if a != -1: break
    else: raise ValueError("The function cf_func returned -1 "
            "for 0, which is not allowed.")
    res = (a, 1)
    for i in reversed(range(i)):
        res = (res[1] + cf_func(i) * res[0], res[0])
    return res

def sqrtCF(num: int) -> Tuple[Union[Tuple[int], int]]:
    """
    Finds the continued fraction representation of the
    square root of num
    
    Args:
        Required positional:
        num (int): The number whose square root to be
                represented as a continued fraction
    
    Returns:
    2-tuple whose 0th index contains a tuple of ints which is
    the sequence of terms in the continued fraction
    representation up to the point where the sequence repeats
    and whose 1st index contains the index of the sequence
    the repetition goes back to (where 0-indexing is used).
    For any positive integer that is not an exact square,
    the sequence is guaranteed to repeat.
    If num is an exact square (e.g. 1, 4, 9, ...) then the
    1st index contains -1.
    """
    seen = {}
    res = []
    curr = (0, 1)
    sqrt = isqrt(num)
    if sqrt ** 2 == num: return ((sqrt,), -1)
    while True:
        if curr in seen.keys():
            return (tuple(res), seen[curr])
        seen[curr] = len(res)
        a = (sqrt + curr[0]) // curr[1]
        res.append(a)
        b = curr[0] - a * curr[1]
        curr = (-b, (num - b ** 2) // curr[1])
        prev = curr
    return ()

def pellFundamentalSolution(D: int) -> Tuple[int]:
    """
    Finds the solution to Pell's equation and Pell's
    negative equation (if they exist):
        x ** 2 - D * y ** 2 = 1
        and
        x ** 2 - D * y ** 2 = -1
    respectively for given strictly positive integer D and
    strictly positive integers x and y such that there does
    not exist another such solution with smaller x (the
    so-called fundamental solution).
    Uses the standard method of solving Pell's equation using
    continued fractions.
    
    Args:
        Required positional:
        D (int): Strictly positive integer that is the value
                of D in the above equation
    
    Returns:
    2-tuple, giving the fundamental solution to Pell's equation
    (index 0) and Pell's negative equation (index 1). Each
    solution is either None (if no solution for strictly
    positive x and y exists for this D) or in the form of a
    2-tuple where indices 0 and 1 are the values of x and y
    for the fundamental solution.
    If D is a square, there is no solution with both strictly
    positive integers x and y, for either Pell's equation or
    Pell's negative equation so (None, None) is returned.
    Otherwise, there is always a solution to Pell's equation,
    so index 0 gives the value None if and only if D is a
    square.
    """
    #(since (1, 0) is the only possible solution to Pell's
    #equation with square D and integer x and y, but not
    #one satisfying the requirement that x and y are strictly
    #positive, and there is no solution not even a trivial one
    #to Pell's negative equation with square D and integer x and y).
    D_cf = sqrtCF(D)
    if D_cf[1] == -1:
        return (None, None)#(1, 0)
    def cf_func(i: int) -> int:
        if i < len(D_cf[0]): return D_cf[0][i]
        j = D_cf[1]
        return D_cf[0][j + (i - j) % (len(D_cf[0]) - j)]
        
    res = nthConvergent(len(D_cf[0]) - 1, cf_func)
    if (len(D_cf[0]) - D_cf[1]) & 1:
        # For continued fractions with odd cycle lengths
        # the solution found is the fundamental solution for:
        #  x ** 2 + D * y ** 2 = -1
        # The following converts this into the fundamental
        # solution for:
        #  x ** 2 + D * y ** 2 = 1
        x, y = res
        res2 = ((x ** 2 + D * y ** 2), (2 * x * y))
        return (res2, res)
    return (res, None)

def pellSolutionGenerator(D: int, negative: bool=False)\
        -> Generator[Tuple[int], None, None]:
    """
    Generator that yields the positive integer solutions to Pell's
    equation or Pell's negative equation:
        x ** 2 - D * y ** 2 = 1 (Pell's equation)
        or x ** 2 - D * y ** 2 = - 1 (Pell's negative equation)
    for given strictly positive integer D, in order of increasing
    size of x.
    Note that these equations either has no integer solutions or
    an infinite number of integer solutions. Pell's equation has
    and infinite number of integer solutions for any non-square
    positive integer value of D, while for square values of D
    it has no integer solutions, while Pell's negative equation
    some non-square positive integer values of D give an infinite
    number of integer solutions, while the other non-square positive
    integer values of D and all square values of D give no
    integer solutions.
    Given that for many values of D the generator does not by
    itself terminate, any loop over this generator should contain a
    break or return statement.
    
    Args:
        Required positional:
        D (int): The strictly positive integer number D used in
                Pell's equation or Pell's negative equation
        
        Optional named:
        negative (bool): If True, iterates over the solutions to
                Pell's equation for the given value of D, otherwise
                iterates over the solutions to Pell's negative
                equation for the given value of D.
            Default: False
    
    Yields:
    2-tuple of ints with the 0th index containing the value of x and
    1st index containing the value of y for the current solution to
    Pell's equation or Pell's negative equation (based on the input
    argument negative given).
    These solutions are yielded in increasing size of x (which by the
    form of Pell's equation and Pell's negative equation and the
    requirement that x and y are strictly positive implies the solutions
    are also yielded in increasing size of y), and it if the generator
    terminates, there are no solutions other than those yielded, and
    for any two consecutive solutions yielded (x1, y1) and (x2, y2), for
    any integer x where x1 < x < x2 there does not exist a positive
    integer y such that (x, y) is also a solution.
    """
    f_sol_pair = pellFundamentalSolution(D=D)
    f_sol = f_sol_pair[negative]
    if f_sol is None:
        # No solution
        return
    curr = f_sol
    if not negative:
        while True:
            yield curr
            curr = (curr[0] * f_sol[0] + D * curr[1] * f_sol[1],\
                    curr[1] * f_sol[0] + curr[0] * f_sol[1])
        return
    while True:
        yield curr
        curr = (curr[0] * f_sol[0] ** 2 + D * curr[0] * f_sol[1] ** 2\
                + 2 * D * curr[1] * f_sol[1] * f_sol[0],\
                curr[1] * f_sol[0] ** 2 + D * curr[1] * f_sol[1] ** 2\
                + 2 * curr[0] * f_sol[1] * f_sol[0])
    return

# Review- see if there exists a more efficient method, as for larger n
# this becomes very slow
# Additionally, consider using congruences to identify cases with no
# solution.
def generalisedPellFundamentalSolutions(
    D: int,
    n: int,
    pell_basic_sol: Optional[Tuple[int]]=None,
) -> List[Tuple[int]]:
    """
    Finds a fundamental solution set to the generalised Pell's equation
    (if they exist):
        x ** 2 - D * y ** 2 = n
    for given strictly positive integer D, given non-zero integer n
    corresponding to the solution to Pell's equation with the same D
    value pell_basic_sol if this is given as not None or the
    fundamental solution to that Pell's equation otherwise (see
    documentation for pellFundamentalSolution() for definition of the
    fundamental solution for Pell's equation for a given value of D).
    A set S of ordered pairs of integers is fundamental solution set
    of the generalised Pell's equation for given D and n corresponding
    to a solution (x0, y0) to Pell's equation with x0 and y0 strictly
    positive integers for the same D value if and only if:
     - Any integer solution of the generalised Pell's equation
       in question (x, y) can be expressed as:
        x + sqrt(D) * y = (x1 + sqrt(D) * y1) * (x0 + sqrt(D) * y0) ** k
       where k is an integer and (x1, y1) an element of S
     - For any two distinct elements of S (x1, y1) and (x2, y2) there
       does not exist an integer k such that:
        x1 + sqrt(D) * y1 = (x2 + sqrt(D) * y2) * (x0 + sqrt(D) * y0) ** k
    Note that k may be negative and from the definition of Pell's
    equation, for any integer k:
        (x0 + sqrt(D) * y0) ** -k = (x0 - sqrt(D) * y0) ** k
    
    Args:
        Required positional:
        D (int): Strictly positive integer that is the value of D in
                the above equation
        n (int): Non-zero integer that is the value of n in the above
                equation
        
        Optional named:
        pell_basic_sol (2-tuple of ints or None): If given as not None,
                gives the integer solution to Pell's equation with the
                same D value the fundamental solution set returned
                corresponds to. Otherwise, the solution returned will
                correspond to the fundamental solution to Pell's
                equation for that D value (see documentation for
                pellFundamentalSolution() for definition of the
                fundamental solution for Pell's equation for a given
                value of D).
                Note that the solution to Pell's equation with the same
                D value used must have strictly positive x and y values
                but given that if (x, y) is a solution, then so are
                (-x, y), (x, -y) and (-x, -y), rather than throwing an
                error if one or both components is negative, the
                absolute value for both components is used.
            Default: None
    
    Returns:
    List of 2-tuples of ints, where each 2-tuple (x1, y1) represents
    exactly two distinct solutions to the generalised Pell's equation
    for the given D and n, namely:
        x = x1 and y = y1
        x = -x1 and y = -y1
    Collectively, the solutions represented by the elements of the list
    form a fundamental solution set of the generalised Pell's equation
    for the given D and n corresponding to the solution to Pell's
    equation with the same D value pell_basic_sol if this is given as
    not None or the fundamental solution to that Pell's equation
    otherwise.
    An empty list signifies that the generalised Pell's equation for
    the given D and n has no integer solutions.
    
    Outline of rationale/proof:
    See https://kconrad.math.uconn.edu/blurbs/ugradnumthy/pelleqn2.pdf
    In this paper, Theorem 3.3 implies that for any solution (x0, y0)
    of Pell's equation for given D there exists a corresponding
    fundamental solution set for the generalised Pell's equation with
    given D and n that is a subset of the solutions (x, y) of this
    generalised Pell's equation such that:
        abs(y) <= sqrt(abs(n)) * (sqrt(u) + 1 / sqrt(u)) / (2 * sqrt(D))
    where:
        u = x0 + sqrt(D) * y0
    We therefore check all y values within these bounds using the
    fundamental solution to Pell's equation for this D to find these
    solutions, the set of which contains as a subset a fundamental
    solution set for this generalised Pell's equation.
    """
    #print(D, n)
    if n == 1:
        #print("hi1")
        res = pellFundamentalSolution(D)[0]
        return [] if res is None else [res]
    elif n == -1:
        #print("hi2")
        res = pellFundamentalSolution(D)[1]
        return [] if res is None else [res]
    if pell_basic_sol is None:
        pell_basic_sol = pellFundamentalSolution(D)[0]
        if pell_basic_sol is None: return []
    
    # Checking congruences:
    for p, sq_congs in ((3, {0, 1}), (4, {0, 1}), (5, {0, 1, 4})):
        ysqD_congs = {(x * D) % p for x in sq_congs}
        n_cong = n % p
        for xsq_cong in sq_congs:
            if (xsq_cong - n_cong) % p in ysqD_congs:
                break
        else:
            #print(f"failed {p} congruence")
            return []
    
    sqrt_D = math.sqrt(D)
    abs_val_func = lambda x: abs(x[0] + x[1] * sqrt_D)
    x0, y0 = [abs(x) for x in pell_basic_sol]
    u = abs_val_func((x0, y0))
    u_sqrt = math.sqrt(u)
    y_ub = math.floor(math.sqrt(abs(n)) * (u_sqrt + 1 / u_sqrt) /\
            (2 * math.sqrt(D)))
    #print(y_ub)
    res = []
    for y in range(y_ub + 1):
        x_sq = n + D * y ** 2
        x = isqrt(x_sq)
        if x ** 2 == x_sq:
            res.append((x, y))
            if y: res.append((x, -y))
    if not res: return []
    # Removing solutions that can be reached by another solution
    # via a power of a solution to Pell's equation with the same
    # value of D.
    res.sort(key=abs_val_func)
    res_set = {v for v in res}
    excl_inds = set()
    abs_val_mx = abs_val_func(res[-1])
    for i1, (x1, y1) in enumerate(res):
        #print(abs_val_mx, abs_val_func((x1, y1)))
        mx_u_pow = math.floor(math.log(abs_val_mx /\
                abs_val_func((x1, y1))))
        if mx_u_pow < 1: break
        x, y = (x1 * x0 + y1 * y0 * D, x1 * y0 + y1 * x0)
        for _ in range(mx_u_pow):
            if (x, y) in res_set:
                #print("removed degenerate")
                excl_inds.add(i1)
                break
            x, y = (x * x0 + y * y0 * D, x * y0 + y * x0)
    return [x for i, x in enumerate(res) if i not in excl_inds] if\
            excl_inds else res

def generalisedPellSolutionGenerator(
    D: int,
    n: int,
    excl_trivial: bool=True,
) -> Generator[Tuple[int], None, None]:
    """
    Generator that yields the solutions to the generalised Pell's
    equation:
        x ** 2 - D * y ** 2 = n
    for given strictly positive integer D and integer n, with strictly
    positive (or if excl_trivial is False, non-negative) integers x and
    y in order of increasing size of x.
    Note that these equations either has no strictly positive integer
    solutions or an infinite number of strictly positive integer
    solutions.
    Given that for many values of D the generator does not by
    itself terminate, any loop over this generator should contain a
    break or return statement.
    
    Args:
        Required positional:
        D (int): The strictly positive integer D used in the
                generalised Pell's equation.
        n (int): The integer n used in the generalised Pell's equation.
        
        Optional named:
        excl_trivial (bool): If True, excludes solutions for which
                x or y is zero. Otherwise includes such solutions.
            Default: True
    
    Yields:
    2-tuple of ints with the 0th index containing the value of x and
    1st index containing the value of y for the current solution to
    the generalised Pell's equation for the given values of D and n.
    These solutions are yielded in increasing size of x, and if the
    generator terminates, there are no solutions for strictly positive
    (or if excl_trivial is given as True non-negative) x and y other
    than those yielded, and for any two consecutive solutions yielded
    (x1, y1) and (x2, y2), for any integer x where x1 < x < x2 there
    does not exist a positive integer y such that (x, y) is also a
    solution.
    """
    if not excl_trivial:
        if n >= 0:
            n_sqrt = isqrt(n)
            if n_sqrt ** 2 == n:
                yield (n_sqrt, 0)
        elif not n % D:
            n_div_D = -n // D
            n_div_D_sqrt = isqrt(n_div_D)
            if n_div_D_sqrt ** 2 == n_div_D:
                yield (0, n_div_D_sqrt)
    if n == 1:
        yield from pellSolutionGenerator(D, negative=False)
        return
    elif n == -1:
        yield from pellSolutionGenerator(D, negative=True)
        return
    elif not n:
        # If n = 0, has solutions if and only if D is a square
        D_sqrt = isqrt(D)
        if D_sqrt ** 2 != D: return
        for y in itertools.count(1):
            yield (y * D_sqrt, y)
        return
    pell_basic_sol = pellFundamentalSolution(D)[0]
    if pell_basic_sol is None: return
    x0, y0 = pell_basic_sol
    pell_heap = []
    for x, y in generalisedPellFundamentalSolutions(D, n,\
            pell_basic_sol=pell_basic_sol):
        #print(x, y)
        pell_heap.append((abs(x), abs(y), x, y, True))
        if y: pell_heap.append((abs(x), abs(y), x, y, False))
    heapq.heapify(pell_heap)
    prev_x = None
    #print(f"pell_heap = {pell_heap}")
    while pell_heap:
        x2, y2, x, y, b = heapq.heappop(pell_heap)
        #print(f"x = {x}, y = {y}")
        #x2, y2 = abs(x), abs(y)
        if x2 != prev_x and x2 and y2:
            yield (x2, y2)
        prev_x = x2
        if b:
            x_, y_ = (x * x0 + y * y0 * D, x * y0 + y * x0)
            heapq.heappush(pell_heap,\
                    (abs(x_), abs(y_), x_, y_, b))
        else:
            x_, y_ = (x * x0 - y * y0 * D, -x * y0 + y * x0)
            heapq.heappush(pell_heap,\
                    (abs(x_), abs(y_), x_, y_, b))
    return

def kPolygonalMinKPolyDiffExperimental(
    k: int=5,
    max_k_poly_diff: Optional[int]=10 ** 6,
) -> None:
    """
    Note that while this gives the correct answer to Project Euler
    #44 and is much faster than the kPolygonalMinKPolyDiff2(), the
    approach is not proven to be guaranteed to give the smallest
    possible answer.
    For the given value of k, finds a pair of k-polygonal numbers
    such that sum of and the difference between the pair are also
    k-polygonal numbers and the difference is no larger than the
    difference between any other such pair of k-polygonal numbers.
    A k-polygonal number (for k an integer no less than 3) is a
    strictly positive integer that for some strictly positive integer
    n can be expressed as:
        n * ((k - 2) * n - k + 4) / 2
    where for a given strictly positive integer n, this number is
    referred to as the nth k-polygonal number, or the number at
    position n of the k-polygonal sequence.
    If max_k_poly_diff is given as not None, only considers pairs
    for which the difference is not greater than the max_k_poly_diff:th
    k-polygonal number.
    
    Args:
        Optional named:
        k (int): Integer no less than 3 giving the k-value of the
                k-polygonal numbers.
            Default: 5
        max_k_poly_diff (int or None): If given as not None, a strictly
                positive integer giving the position of the number
                in the k-polygonal sequence for which only pairs with
                the difference no greater than this k-polygonal number
                are considered.
                Note that if this is given as None, for values of k
                for which there is no solution, the function will
                not terminate.
            Default: 10 ** 6
    
    Returns:
    If a solution is found, 4-tuple of 2-tuples of ints, with each
    2-tuple containing the position in the sequence of k-polygonal
    numbers in index 0 and the value of that k-polygonal number in
    index 1. Index 0 and 1 correspond to the smaller and larger
    k-polygonal number in the identified pair respectively, while
    index 2 corresponds to the sum of the pair and index 3 to the
    difference between the pair.
    If max_k_poly_diff given as None and no pairs of k-polygonal
    numbers exist such that their difference is no greater than
    the number at position k_poly_diff of the k-polygonal sequence,
    returns an empty tuple.
    """
    # Review- try to prove that this works
    #since = time.time()
    if k == 4: return None
    eqn = ((k - 2) << 1, 4 - k)
    facts = [i for i in range(2, eqn[0] + 1) if isPrime(i) and\
            not eqn[0] % i]
    r = (k - 4) ** 2
    start = 1#-(eqn[1] // eqn[0])
    res = tuple(float("inf") for _ in range(4))
    iter_obj = itertools.count(start) if max_k_poly_diff is None else\
            range(start, ((max_k_poly_diff - eqn[1]) // eqn[0]) + 1)
    #print(((max_k_poly_diff - eqn[1]) // eqn[0]))
    for n0 in iter_obj:
        d0 = eqn[0] * n0 + eqn[1]
        if d0 >= res[-1]: break
        print(f"d0 = {d0}")
        for d in (-d0, d0):
            #print(f"d = {d}")
            #for m2_, m1 in generalisedPellFundamentalSolutions(2, d):
            cnt0 = 0
            for m2_, m1 in generalisedPellSolutionGenerator(2, d,\
                    excl_trivial=True):
                if cnt0 > 1: break
                cnt0 += 1
            
                m2 = m2_ - m1
                if m1 < 0 or m2 < 0 or m1 < m2 or gcd(m1, m2) != 1 or\
                        m1 ** 2 + 2 * m1 * m2 - m2 ** 2 ==\
                        abs(m2 ** 2 + 2 * m1 * m2 - m1 ** 2):
                    continue
                
                D = 4 * m1 * m2 * (m1 ** 2 - m2 ** 2)
                print(f"d = {d}, m1 = {m1}, m2_ = {m2_}, m2 = {m2}, D = {D}")
                #print(f"D = {D}")
                f_sols = generalisedPellFundamentalSolutions(D, r)
                print(f"f_sols = {f_sols}")
                sol_heap = []
                for a, u in f_sols:
                    sol_heap.append((a, u, eqn[0] - 1, True))
                    if u: sol_heap.append((a, u, eqn[0] - 1, False))
                if not sol_heap: continue
                heapq.heapify(sol_heap)
                b_sol = pellFundamentalSolution(D)[0]
                #print(f"b_sol = {b_sol}")
                if b_sol is not None:
                    x0, y0 = b_sol
                    for fact in facts:
                        if not x0 % fact or not y0 % fact:
                            b_sol = None
                            break
                prev_a = None
                #print(f"sol_heap = {sol_heap}")
                while sol_heap:
                    x, y, remain, b = heapq.heappop(sol_heap)
                    #print(x, y)
                    a, u = abs(x), abs(y)
                    if a != prev_a and a and u:
                        if u * d0 >= res[-1]: break
                        ans = (a, u * (m1 ** 2 + m2 ** 2),\
                                u * (m1 ** 2 + 2 * m1 * m2 - m2 ** 2), u * d0)
                        if all(not (x - eqn[1]) % eqn[0] for x in ans):
                            res = ans
                            print(ans)
                            break
                    prev_a = a
                    if remain > 0 and b_sol is not None:
                        if b:
                            heapq.heappush(sol_heap,\
                                    (x * x0 + y * y0 * D, x * y0 + y * x0,\
                                    remain - 1, b))
                        else:
                            heapq.heappush(sol_heap,\
                                    (x * x0 - y * y0 * D, -x * y0 + y * x0,\
                                    remain - 1, b))
                
                """
                mx_cnt = len(sol_heap) * eqn[0]
                cnt = 0
                for a, u in generalisedPellSolutionGenerator(D, r):
                    if cnt > eqn[0]: break
                    #print(cnt)
                    if u * d0 >= res[-1]: break
                    ans = (a, u * (m1 ** 2 + m2 ** 2),\
                            u * (m1 ** 2 + 2 * m1 * m2 - m2 ** 2), u * d0)
                    if all(not (x - eqn[1]) % eqn[0] for x in ans):
                        res = ans
                        #print(D, a, u)
                        #print(m1, m2, d0)
                        #print(res)
                        break
                    cnt += 1
                """
    res2 = []
    if isinstance(res[0], int):
        for num in res:
            k_poly_num = (num - eqn[1]) // eqn[0]
            res2.append((k_poly_num,\
                    nthKPolygonalNumber(k_poly_num, k)))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return tuple(res2)

def nthKPolygonalNumber(n: int, k: int) -> int:
    return (n * ((k - 2) * n - k + 4)) >> 1

def kPolygonalGenerator(
    k: int,
    n_max: Optional[int]=None,
) -> Generator[int, None, None]:
    
    iter_obj = range(1, n_max + 1) if isinstance(n_max, int) else\
            itertools.count(1)
    for n in iter_obj:
        yield nthKPolygonalNumber(n, k)
    return

def isKPolygonal(num: int, k: int=3) -> bool:
    """
    Identifies whether the integer num is a k-polygonal number.
    A k-polygonal number (for k an integer no less than 3) is a
    strictly positive integer that for some strictly positive integer
    n can be expressed as:
        n * ((k - 2) * n - k + 4) / 2
    
    Args:
        Required positional:
        num (int): The number whose status as a k-polygonal number
                is being determined.
        
        Optional named:
        k (int): Integer no less than 3 giving the k-value of the
                k-polygonal numbers of which the membership of num
                is being determined.
            Default: 3
    
    Returns:
    Boolean (bool) with the value True if num is a k-polygonal number
    and False otherwise.
    
    Outline of rationale:
    This uses the formula for k-polygonal numbers given above, it can
    be deduced that a strictly positive integer m is a k-polygonal
    number if and only if:
        k ** 2 + 8 * (m - 1) * (k - 2)
    is the squre of some non-negative integer num2 and:
        k - 4 + num2
    is divisible by (2 * k - 4)
    """
    if num <= 0: return False
    num2 = k ** 2 + 8 * (num - 1) * (k - 2)
    m = isqrt(num2)
    if m * m != num2: return False
    return not (k - 4 + m) % (2 * k - 4)

def kPolygonalMinKPolyDiffProperties(
    k: int=5,
    max_k_poly_diff: Optional[int]=10 ** 6,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    For the given value of k, finds a pair of k-polygonal numbers
    such that sum of and the absolute difference between the pair
    are also k-polygonal numbers and the difference is no larger
    than the difference between any other such pair of k-polygonal
    numbers, returning the details of the numbers involved including
    the two k-polygonal numbers, their sum and absolute difference
    and where in the sequence of k-polygonal numbers each of these
    is found.
    
    A k-polygonal number (for k an integer no less than 3) is a
    strictly positive integer that for some strictly positive integer
    n can be expressed as:
        n * ((k - 2) * n - k + 4) / 2
    where for a given strictly positive integer n, this number is
    referred to as the nth k-polygonal number, or the number at
    position n of the k-polygonal sequence.
    If max_k_poly_diff is given as not None, only considers pairs
    for which the difference is not greater than the max_k_poly_diff:th
    k-polygonal number.
    
    Args:
        Optional named:
        k (int): Integer no less than 3 giving the k-value of the
                k-polygonal numbers.
            Default: 5
        max_k_poly_diff (int or None): If given as not None, a strictly
                positive integer giving the position of the number
                in the k-polygonal sequence for which only pairs with
                the difference no greater than this k-polygonal number
                are considered.
                Note that if this is given as None, for values of k
                for which there is no solution, the function will
                not terminate.
            Default: 10 ** 6
    
    Returns:
    If a solution is exists, 4-tuple of 2-tuples of ints, with each
    2-tuple containing the position in the sequence of k-polygonal
    numbers in index 0 and the value of that k-polygonal number in
    index 1. Index 0 and 1 correspond to the smaller and larger
    k-polygonal number in the identified pair respectively, while
    index 2 corresponds to the sum of the pair and index 3 to the
    difference between the pair.
    If no such solution exists (most likely due to max_k_poly_diff
    being specified as a relatively small positive integer), returns
    an empty tuple.
    """
    #since = time.time()
    # Search in terms of the sum and the larger of the two numbers
    polygonal_lst = []
    polygonal_dict = {}
    i_start = 0
    curr_diff = float("inf") if max_k_poly_diff is None else\
            nthKPolygonalNumber(max_k_poly_diff, k) + 1
    res = ()
    for num_sum in kPolygonalGenerator(k):
        sum_idx = len(polygonal_lst) + 1
        #print(sum_idx, num_sum)
        target = (num_sum + 1) >> 1
        for i_start in range(i_start, len(polygonal_lst)):
            if polygonal_lst[i_start] > target: break
        else:
            i_start = len(polygonal_lst)
            polygonal_lst.append(num_sum)
            polygonal_dict[num_sum] = sum_idx
            continue
        if i_start and polygonal_lst[i_start] -\
                polygonal_lst[i_start - 1] >= curr_diff:
            break
        for i in range(i_start, len(polygonal_lst)):
            num2 = polygonal_lst[i]
            num1 = num_sum - num2
            if num2 - num1 >= curr_diff: break
            if num1 not in polygonal_dict.keys():
                continue
            num_diff = num2 - num1
            if num_diff not in polygonal_dict.keys():
                continue
            curr_diff = num_diff
            res = ((polygonal_dict[num1], num1),\
                    (polygonal_dict[num2], num2),\
                    (sum_idx, num_sum),
                    (polygonal_dict[num_diff], num_diff))
            #print(curr_diff, res)
        polygonal_lst.append(num_sum)
        polygonal_dict[num_sum] = sum_idx
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

def kPolygonalMinKPolyDiff(
    k: int=5,
    max_k_poly_diff: Optional[int]=10 ** 6,
) -> int:
    """
    For the given value of k, the smallest strictly positive integer
    such that there exists a distinct pair of k-polygonal numbers
    whose absolute difference and sum are both k-polygonal and
    whose difference is equal to that stirctly positive integer.

    A k-polygonal number (for k an integer no less than 3) is a
    strictly positive integer that for some strictly positive integer
    n can be expressed as:
        n * ((k - 2) * n - k + 4) / 2
    where for a given strictly positive integer n, this number is
    referred to as the nth k-polygonal number, or the number at
    position n of the k-polygonal sequence.
    If max_k_poly_diff is given as not None, only considers pairs
    for which the difference is not greater than the max_k_poly_diff:th
    k-polygonal number.
    
    Args:
        Optional named:
        k (int): Integer no less than 3 giving the k-value of the
                k-polygonal numbers.
            Default: 5
        max_k_poly_diff (int or None): If given as not None, a strictly
                positive integer giving the position of the number
                in the k-polygonal sequence for which only pairs with
                the difference no greater than this k-polygonal number
                are considered.
                Note that if this is given as None, for values of k
                for which there is no solution, the function will
                not terminate.
            Default: 10 ** 6
    
    Returns:
    Integer (int) which, if such a number exists is the smallest
    difference between k-polygonal numbers satisfying the properties
    outlined above. If no such number exists, then -1 is returned.
    """

    res = kPolygonalMinKPolyDiffProperties(k=k, max_k_poly_diff=max_k_poly_diff)
    return res[3][1] if res else -1

# Problem 45
def kPolygonalSequenceNumber(num: int, k: int=3) -> bool:
    """
    Identifies whether the integer num is a k-polygonal number and
    if so which term number in the sequence of k-polygonal numbers it
    is.
    A k-polygonal number (for k an integer no less than 3) is a
    strictly positive integer that for some strictly positive integer
    n can be expressed as:
        n * ((k - 2) * n - k + 4) / 2
    where n, if it exists, is the term number in the k-polygonal
    sequence of that num. 
    
    Args:
        num (int): The number whose term number in the k-polygonal
                sequence (if it is in the sequence) is being
                determined.
        
        Optional named:
        k (int): Integer no less than 3 giving the k-value of the
                k-polygonal numbers of which the membership of num
                is being determined.
            Default: 3
    
    Returns:
    Integer (int) with the term number in the k-polygonal sequence
    of num if num is in the k-polygonal sequence, otherwise -1.
    
    Outline of rationale:
    This uses the formula for k-polygonal numbers given above,
    inverting the formula using the quadratic formula.
    """
    if num <= 0: return False
    num2 = k ** 2 + 8 * (num - 1) * (k - 2)
    m = isqrt(num2)
    if m * m != num2 or (k - 4 + m) % (2 * k - 4): return -1
    return (k - 4 + m) // (2 * k - 4)

def nSmallestTriangularPentagonalAndHexagonalNumbersProperties(
    n_smallest: int=3
) -> List[Tuple[Tuple[int, int, int], int]]:
    """
    Finds the smallest n_smallest strictly positive integers that are
    simultaneously triangle, pentagonal and hexagonal numbers, giving
    both the numbers themselves and which triangle, pentagonal and
    hexagonal numbers they are (i.e. where in each of the sequences
    of triangle, pentagonal and hexagonal numbers the calculated
    number is found).

    A triangle number n is a strictly positive integer such that there
    exists a strictly positive integer n such that:
        n = m * (m + 1) / 2
    The integer m corresponds to which triangle number n is.
    A pentagonal number n is a strictly positive integer such that there
    exists a strictly positive integer n such that:
        n = m * (3 * m - 1) / 2
    The integer m corresponds to which pentagonal number n is.
    A hexagonal number n is a strictly positive integer such that there
    exists a strictly positive integer n such that:
        n = m * (2 * m - 1) / 2
    The integer m corresponds to which hexagonal number n is.
    
    Args:
        Optional named:
        n_smallest (int): Strictly positive integer giving the
                number of the initial integers that are simultaneously
                triangle, pentagonal and hexagonal numbers.
            Default: 3
    
    Returns:
    n_smallest-tuple with the entries representing (in order of
    increasing size) the smallest n_smallest natural numbers that are
    simultaneously triangle, pentagonal and hexagonal numbers. Each
    entry is a 2-tuple, whose index 0 contains to a 3-tuple with
    integer entries representing the sequence number in the triangle,
    pentagonal and hexagonal number sequences respectively and whose
    index 1 contains the value of that number.
    """
    # Observe that every hexagonal number is also triangular, since
    # for strictly positive integer m:
    #  H_m = m(2m - 1) = (2m - 1)(2m)/2 = (2m - 1)((2m - 1) + 1)/2
    #                  = T_(2m - 1)
    #since = time.time()
    res = []
    
    # From solution to Problem 44
    for i, num in enumerate(kPolygonalGenerator(6, n_max=None)):
        # From solution to Problem 44
        j = kPolygonalSequenceNumber(num, k=5)
        if j != -1:
            res.append(((2 * i + 1, j, i + 1), num))
            if len(res) == n_smallest: break
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

def nthSmallestTriangularPentagonalAndHexagonalNumbers(n: int=3) -> int:
    """
    Solution to Project Euler #45

    Finds the n:th smallest strictly positive integer that is
    simultaneously a triangle, pentagonal and hexagonal number.

    A triangle number n is a strictly positive integer such that there
    exists a strictly positive integer n such that:
        n = m * (m + 1) / 2
    A pentagonal number n is a strictly positive integer such that there
    exists a strictly positive integer n such that:
        n = m * (3 * m - 1) / 2
    A hexagonal number n is a strictly positive integer such that there
    exists a strictly positive integer n such that:
        n = m * (2 * m - 1) / 2
    
    Args:
        Optional named:
        n (int): Strictly positive integer specifying which integer that
                is simultaneously a triangle, pentagonal and hexagonal
                number to return, with the n:th smallest such integer
                being the one to return.
            Default: 3
    
    Returns:
    Integer (int) giving the n:th smallest strictly positive integer
    that is simultaneously a triangle, pentagonal and hexagonal number.
    """
    res = nSmallestTriangularPentagonalAndHexagonalNumbersProperties(n)
    #print(res)
    return res[-1][1]

# Problem 46
def goldbachOtherChk(n_max: int=10 ** 6) -> int:
    """
    Solution to Project Euler #46

    Looks for counterexamples up to n_max to Christian Goldbach's
    conjecture that every odd composite number can be written as the
    sum of a prime and twice a square.
    
    Args:
        Optional named:
        n_max (int): The strictly positive integer up to which a
                counterexample to the conjecture is searched.
            Default: 10 ** 6
    
    Returns:
    Integer (int) giving the smallest counterexample found or, if no
    counterexample exists for integers no greater than n_max, returns
    -1.
    """
    #since = time.time()
    # Finding odd composite sieve and prime list (based on solution to
    # Problem 27)
    chk_freq = 1 # How often check if solution found
    sieve = [False for x in range(n_max + 1)]
    sieve_p = [x % 2 != 0 for x in range(n_max + 1)]
    sieve_p[2] = True
    saved_p = 2
    for p in range(3, n_max + 1, 2):
        if not sieve_p[p]: continue
        for i in range(p ** 2, n_max + 1, 2 * p):
            sieve_p[i] = False
            sieve[i] = True
        i = 1
        while True:
            num = p + 2 * i ** 2
            if num > n_max: break
            sieve[num] = False
            i += 1
        # See whether can exit early
        if i % chk_freq == 0:
            for i in range(saved_p + 2, p + 2):
                if not sieve[i]: continue
                #print(f"Time taken = {time.time() - since:.4f} seconds")
                return i
            saved_p = p
    
    for i, sieve_val in enumerate(sieve):
        if sieve_val:
            res = i
            break
    else: res = -1
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 47
def smallestNConsecutiveMDistinctPrimeFactors(
    n: int,
    m: int,
    n_max: int,
    n_min: int=1,
    p_sieve: Optional[PrimeSPFsieve]=None,
) -> int:
    """
    Finds the smallest n consecutive strictly positive integers such
    that each has exactly m distinct prime factors in the range n_min
    to n_max (inclusive).
    
    Args:
        Required positional:
        n (int): Strictly positive integer giving the number of
                consecutive integers being sought.
        m (int): Strictly positive integer giving the number of
                distinct prime factors each of the consecutive
                integers must have.
        n_max (int): Stictly positive integer giving the integer
                up to which the search is performed.
        
        Optional named:
        n_min (int): Strictly positive integer giving the integer
                from which the search is performed.
            Default: 1
        p_sieve (PrimeSPFsieve object or None): If given as not
                None, gives a PrimeSPFsieve object which contains
                a prime sieve used for finding the prime factors
                of a given integer.
                If not given or given as None, a new such object is
                defined for use by this function.
                Note that this object may be updated if the sieve
                encounters larger numbers requiring extension of
                the sieve
            Default: None
    
    Returns:
    Integer (int) giving the smallest number in the smallest
    consecutive sequence of length n of strictly positive integers with
    all elements no less than n_min and no greater than n_max and
    exactly m distinct prime factors if such a sequence exists. If
    no such sequence exists, returns -1.
    """
    n_min = max(n_min, 1)
    if p_sieve is None:
        p_sieve = PrimeSPFsieve()
    p_sieve.extendSieve(n_max)
    num0 = n_min
    for num in range(n_min, n_max + 1):
        if len(p_sieve.primeFactors(num)) == m:
            if num - num0 + 1 == n:
                return num0
        else: num0 = num + 1
    return -1

def smallestnConsecutiveMDistinctPrimeFactorsUnlimited(
    n: int=4,
    m: int=4,
    p_sieve: Optional[PrimeSPFsieve]=None,
) -> int:
    """
    Solution to Project Euler #47

    Finds the smallest n consecutive strictly positive integers such
    that each has exactly m distinct prime factors in the range 1 to
    n_max (inclusive).
    Note- if m and n are chosen such that there is no solution
    then this will not terminate.
    
    Args:
        Required positional:
        n (int): Strictly positive integer giving the number of
                consecutive integers being sought.
        m (int): Strictly positive integer giving the number of
                distinct prime factors each of the consecutive
                integers must have.
        
        Optional named:
        p_sieve (PrimeSPFsieve object or None): If given as not
                None, gives a PrimeSPFsieve object which contains
                a prime sieve used for finding the prime factors
                of a given integer.
                If not given or given as None, a new such object is
                defined for use by this function.
                Note that this object may be updated if the sieve
                encounters larger numbers requiring extension of
                the sieve
            Default: None
    
    Returns:
    Integer (int) giving the smallest number in the smallest
    consecutive sequence of length n of strictly positive integers each
    with exactly m distinct prime factors.
    """
    #since = time.time()
    if p_sieve is None:
        p_sieve = PrimeSPFsieve()
    n_min = 1
    n_max = 10
    while True:
        res = smallestNConsecutiveMDistinctPrimeFactors(n, m, n_max,\
                n_min=n_min, p_sieve=p_sieve)
        if res != -1: break
        n_min = n_max - n + 2
        n_max *= 10
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 48
def eulerTotientFunction(num: int) -> int:
    """
    For given strictly positive integer number n, calculates the
    Euler totient function (the number of strictly positive integers
    less than num which are relatively prime to num).
    
    Args:
        Required positional:
        num (int): Strictly positive integer for which the Euler
                totient function is to be calculated.
    
    Returns:
    Integer (int) giving the Euler totient function value for num.
    """
    p_fact = primeFactorisation(num)
    out = 1
    for p, v in p_fact.items():
        out *= p ** (v - 1) * (p - 1)
    return out
    
def powMod(a: int, b: int, n: int, et_n: Optional[int]=None):
    """
    Calculates a ** b (mod n) for non-negatve integers a and b (not
    both zero) and strictly positive integer n.
    Utilises Euler's theorem (a generalisation of Fermat's little
    theorem), that if a and n are relatively prime then:
        a ** phi(n) is congruent to 1 modulo n
    where phi(n) is the value Euler totient function for n.
    
    Args:
        Required positional:
        a (int): Non-negative integer giving the base of the
                exponentiation
        b (int): Non-negative integer giving the exponent of the
                exponentiation. If a is 0, this cannot also be zero.
        n (int): Strictly positive integer giving the modulus to
                which the exponentiation is to be taken.
        
        Optional named:
        et_n (None or int): If provided as not None, the previously
                calculated value of the Euler totient function for n.
            Default: None
    
    Returns:
    a ** b (mod n)
    """
    # From solution to Problem 33
    if gcd(a, n) == 1:
        if et_n is None:
            et_n = eulerTotientFunction(n)
        b = b % et_n
    a = a % n
    # Review- try to find a way to simplify if a and n are not coprime
    out = 1
    for i in range(b):
        out = (out * a) % n
        if out == 0: return 0
    return out

def selfExpIntSumLastDigits(
    n_max: int=1000,
    n_digits: int=10,
    base: int=10,
) -> int:
    """
    Solution to Project Euler #48

    Calculates the final n_digits digits of the sum:
        1 ** 1 + 2 ** 2 + 3 ** 3 + ... + n_max ** n_max
    in the chosen base where n_max and n_digits are strictly positive
    integers.
    
    Args:
        Optional named:
        n_max (int): Strictly positive integer giving the base and
                exponent of the final base and exponent in the above
                sum.
            Default: 1000
        n_digits (int): Strictly positive integer giving the number
                of the last digits in the given base of the sum are to
                be included.
            Default: 10
        base (int): The base in which the sum is to be represented when
                giving the last n_digits digits.
            Default: 10
    
    Returns:
    Integer (int) giving the value of the final n_digit above sum in
    the chosen base.
    """
    #since = time.time()
    modulus = base ** n_digits
    et_mod = eulerTotientFunction(modulus)
    res = 0
    for i in range(1, n_max + 1):
        res = (res + powMod(i, i, modulus, et_n=et_mod)) % modulus
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 49
def digitsSorted(num: int, base: int=10) -> Tuple[int]:
    """
    Gives the digits of the representation of non-negative integer num
    in the given base in sorted order.
    
    Args:
        Required positional:
        num (int): Non-negative integer whose digits when expressed in
                the chosen base are to be sorted.
        
        Optional named:
        base (int): Strictly positive integer giving the base in which
                num is to be expressed when finding its digits.
            Default: 10
    
    Returns:
    Tuple of ints representing the value of each digit in the
    representation of num in the chosen base in order of increasing
    size.
    """
    if not num: return (0,)
    digs = []
    while num:
        num, r = divmod(num, base)
        digs.append(r)
    return tuple(sorted(digs))


def primePermutArithmeticProgression(
    n_dig: int,
    seq_len: int,
    base: int=10
) -> List[Tuple[int]]:
    """
    Finds all possible integer sequences of length seq_len whose
    elements are prime numbers, form a strictly increasing arithmetic
    progresion and their representations in the chosen base are all
    n_dig digits long containing the same digits as each other (where
    leading zeros are not allowed).
    
    Args:
        Required positional:
        n_dig (int): The number of digits each element in the sequence
                must contain when expressed in the chosen base.
        seq_len (int): The length of the arithmetic sequences being
                sought.
        
        Optional named:
        base (int): Strictly positive integer giving the base in which
                the representation of all primes in a returned sequence
                must contain the same n_dig digits.
            Default: 10
    
    Returns:
    A list of n_dig-tuples of ints containing the sequences satisfying
    the given requirements in increasing lexicographical order.
    """
    mn = base ** (n_dig - 1)
    mx = mn * base
    ps = PrimeSPFsieve(mx)
    p_mn_i = bisect.bisect_left(ps.p_lst, mn)
    p_mx_i = bisect.bisect_right(ps.p_lst, mx)
    if seq_len == 1:
        return [(ps.p_lst[i],) for i in range(p_mn_i, p_mx_i)]
    seen = {}
    res = []
    for p_i in reversed(range(p_mn_i, p_mx_i)):
        p1 = ps.p_lst[p_i]
        #add = []
        digs = digitsSorted(p1, base=base)
        seen.setdefault(digs, [])
        seq_dict1 = {}
        for p2, seq_dict2 in seen[digs]:
            diff = p2 - p1
            length = seq_dict2.get(diff, 1) + 1
            seq_dict1[diff] = length
            if length < seq_len: continue
            ans = [p1]
            for _ in range(seq_len - 1):
                ans.append(ans[-1] + diff)
            res.append(tuple(ans))
        seen[digs].append([p1, seq_dict1])
    return res[::-1]

def primePermutArithmeticProgressionConcat(
    n_dig: int=4,
    seq_len: int=3,
    base: int=10,
) -> List[str]:
    """
    Solution to Project Euler #49
    Finds all possible integer sequences of length seq_len whose
    elements are prime numbers, form a strictly increasing arithmetic
    progresion and their representations in the chosen base are all
    n_dig digits long containing the same digits as each other (where
    leading zeros are not allowed). These are returned as a list of the
    sequences where the expressions of each number of the sequence in
    the given base are concatenated in order. For bases exceeding 10,
    lower case letters are used for digits exceeding 9 (i.e. 10 is
    given by a, 11 by b, ..., 35 by z. Further digit characters are
    not defined so this implementation requires that the base does not
    exceed 36.
    
    Args:
        Optional named:
        n_dig (int): The number of digits each element in the sequence
                must contain when expressed in the chosen base.
            Default: 4
        seq_len (int): The length of the arithmetic sequences being
                sought.
            Default: 3
        base (int): Strictly positive integer not exceeding 36 giving
                the base in which the representation of all primes in a
                returned sequence must contain the same n_dig digits.
            Default: 10
    
    Returns:
    A list of strings (str) containing the concatenated sequences
    satisfying the given requirements with the numbers in the sequence
    represented in the chosen base.
    """
    #since = time.time()
    if base > 36:
        raise ValueError("Input argument base cannot exceed 36.")
    ord_a = ord("a")
    def dig2Char(d: int) -> str:
        if d < 10: return str(d)
        return chr(d - 10 + ord_a)
    seqs = primePermutArithmeticProgression(n_dig, seq_len, base=base)
    res = []
    for seq in seqs:
        ans = []
        for num in seq:
            dig_lst = []
            while num:
                num, r = divmod(num, base)
                dig_lst.append(r)
            for d in reversed(dig_lst):
                ans.append(dig2Char(d))
        res.append("".join(ans))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 50
def primeSumOfMostConsecutivePrimesProperties(
    n_max: int=10 ** 6 - 1,
) -> Tuple[Union[int, int, List[int]]]:
    """
    Solution to Project Euler #50

    Finds the prime number not exceeding n_max that can be expressed as
    the sum of a number of consecutive primes that is strictly greater
    than that of any prime smaller than it and is no smaller than
    that of any other prime not exceeding n_max, returning this prime
    number, the largest number of consecutive primes of which this
    number can be written as a sum, and a list of those primes in
    increasing order.
    
    Args:
        Optional named:
        n_max (int): The integer no less than 2 for which no primes
                greater are considered.
            Default: 10 ** 6 - 1
    
    Returns:
    3-tuple whose index 0 contains an integer (int) giving the prime
    satisfying the given conditions, whose index 1 contains the largest
    number of consecutive primes that sum to give this prime and whose
    index 2 contains a tuple of ints containing a sequence of
    consecutive primes of length equal to the integer in index 1
    whose sum is equal to the identified prime.
    """
    #since = time.time()
    if n_max < 2:
        raise ValueError("n_max must be no less than 2")
    # Special case
    if n_max == 2: return (2, 1, (2,))
    # From solution to Problem 27
    p_list = primesUpToN(n_max)
    p_cumu = []
    curr = 0
    p_set = set(p_list)
    res = (0, 0, ())
    for i2, p2 in enumerate(p_list):
        p_cumu.append(curr)
        curr += p2
        if curr <= n_max and curr in p_set:
            res = (i2 + 1, curr, (0, i2 + 1))
            continue
        start = bisect.bisect_left(p_cumu, curr - n_max)
        # All primes except the first (2) are odd, so only of the sums
        # not including the first must contain an odd number of terms
        # in order to be prime (observing that any non-empty sum of
        # primes greater than 2 is itself greater than 2).
        if i2 - start + 1 <= res[0]: break
        if not start: start += 1 + (not (i2 & 1))
        elif ((i2 - start) & 1): start += 1
        for i1 in range(start, i2 - res[0], 2):
            num = curr - p_cumu[i1]
            if num not in p_set: continue
            res = (i2 - i1 + 1, num, (i1, i2 + 1))
            break
    res = (res[1], res[0], [p_list[i] for i in range(*res[2])])
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

def primeSumOfMostConsecutivePrimes(
    n_max: int=10 ** 6 - 1,
) -> int:
    """
    Solution to Project Euler #50

    Finds the prime number not exceeding n_max that can be expressed as
    the sum of a number of consecutive primes that is strictly greater
    than that of any prime smaller than it and is no smaller than
    that of any other prime not exceeding n_max.
    
    Args:
        Optional named:
        n_max (int): The integer no less than 2 for which no primes
                greater are considered.
            Default: 10 ** 6 - 1
    
    Returns:
    Integer (int) giving the prime number satisfying the above stated
    conditions.
    """
    res = primeSumOfMostConsecutivePrimesProperties(n_max=n_max)
    return res[0]

##############
project_euler_num_range = (1, 50)

def evaluateProjectEulerSolutions1to50(eval_nums: Optional[Set[int]]=None) -> None:
    if not eval_nums:
        eval_nums = set(range(project_euler_num_range[0], project_euler_num_range[1] + 1))
    
    if 1 in eval_nums:
        since = time.time()
        res = multipleSum(n_max=999, fact_list=(3, 5))
        print(f"Solution to Project Euler #1 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 2 in eval_nums:
        since = time.time()
        res = multFibonacciSum(n_max=4 * 10 ** 6, fact=2)
        print(f"Solution to Project Euler #2 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 3 in eval_nums:
        since = time.time()
        res = largestPrimeFactor(num=600851475143)
        print(f"Solution to Project Euler #3 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 4 in eval_nums:
        since = time.time()
        res = largestPalindromicProduct(prod_nums_n_digit=3, n_prod_nums=2, base=10)
        print(f"Solution to Project Euler #4 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 5 in eval_nums:
        since = time.time()
        res = smallestMultiple(div_max=20)
        print(f"Solution to Project Euler #5 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 6 in eval_nums:
        since = time.time()
        res = sumSquareDiff(n_max=100)
        print(f"Solution to Project Euler #6 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 7 in eval_nums:
        since = time.time()
        res =findPrime(n=10001)
        print(f"Solution to Project Euler #7 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 8 in eval_nums:
        since = time.time()
        res = largestSubstringProduct(num_str=num_str, n_char=13)
        print(f"Solution to Project Euler #8 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 9 in eval_nums:
        since = time.time()
        res = smallestSpecialPythagoreanTripletProduct(side_sum=1000)
        print(f"Solution to Project Euler #9 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 10 in eval_nums:
        since = time.time()
        res = sumPrimes(n_max=2 * 10 ** 6 - 1)
        print(f"Solution to Project Euler #10 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 11 in eval_nums:
        since = time.time()
        res = largestLineProduct(num_grid=num_grid, line_len=4)
        print(f"Solution to Project Euler #11 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 12 in eval_nums:
        since = time.time()
        res = triangleNDiv(target_ndiv=501)
        print(f"Solution to Project Euler #12 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 13 in eval_nums:
        since = time.time()
        res = sumNumbers(num_list=num_list, n_digits=10, base=10)
        print(f"Solution to Project Euler #13 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 14 in eval_nums:
        since = time.time()
        res = longestCollatzChain(n_max=10 ** 6 - 1)
        print(f"Solution to Project Euler #14 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 15 in eval_nums:
        since = time.time()
        res = countLatticePaths(r=20, d=20)
        print(f"Solution to Project Euler #15 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 16 in eval_nums:
        since = time.time()
        res = digitSum(num=2 ** 1000, base=10)
        print(f"Solution to Project Euler #16 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 17 in eval_nums:
        since = time.time()
        res = numberLetterCount(n_max=1000)
        print(f"Solution to Project Euler #17 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 18 in eval_nums:
        since = time.time()
        res = triangleMaxSum(triangle=triangle1, preserve_triangle=True)
        print(f"Solution to Project Euler #18 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 19 in eval_nums:
        since = time.time()
        res = countMonthsStartingDoW(day=0, start_year=1901, end_year=2000)
        print(f"Solution to Project Euler #19 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 20 in eval_nums:
        since = time.time()
        res = digitSum(num=math.factorial(100), base=10)
        print(f"Solution to Project Euler #20 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 21 in eval_nums:
        since = time.time()
        res = amicableNumbersSum(n_max=10000)
        print(f"Solution to Project Euler #21 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 22 in eval_nums:
        since = time.time()
        res = nameListScoreFromFile(
            word_file="project_euler_problem_data_files/p022_names.txt",
            sort_func=None,
            comp_func=alphabetComparitor,
            rel_package_src=True,
        )
        print(f"Solution to Project Euler #22 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 23 in eval_nums:
        since = time.time()
        res = notExpressibleAsSumOfTwoAbundantNumbersSum(n_max=None)
        print(f"Solution to Project Euler #23 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 24 in eval_nums:
        since = time.time()
        res = nthDigitPermutation(n=10 ** 6, digs=tuple(range(10)), base=10)
        print(f"Solution to Project Euler #24 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 25 in eval_nums:
        since = time.time()
        res = firstFibonacciGEnTermNumber(n=10**999)
        print(f"Solution to Project Euler #25 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 26 in eval_nums:
        since = time.time()
        res = reciprocalWithLargestBasimalCycleLength(n_max=999, base=10)
        print(f"Solution to Project Euler #26 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 27 in eval_nums:
        since = time.time()
        res = maxConsecutiveQuadraticPrimesProduct(abs_a_max=999, abs_b_max=1000, n_start=0)
        print(f"Solution to Project Euler #27 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 28 in eval_nums:
        since = time.time()
        res = numSpiralDiagonalsSum(max_side_len=1001)
        print(f"Solution to Project Euler #28 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 29 in eval_nums:
        since = time.time()
        res = distinctPowersNum(a_range=(2, 100), b_range=(2,100))
        print(f"Solution to Project Euler #29 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 30 in eval_nums:
        since = time.time()
        res = digitPowSumEqualsSelfSum(exp=5, base=10)
        print(f"Solution to Project Euler #30 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 31 in eval_nums:
        since = time.time()
        res = coinCombinations(
            amount_p=200,
            coins_allowed=(1, 2, 5, 10, 20, 50, 100, 200),
        )
        print(f"Solution to Project Euler #31 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 32 in eval_nums:
        since = time.time()
        res = pandigitalProductsSum(min_dig=1, max_dig=None, base=10)
        print(f"Solution to Project Euler #32 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 33 in eval_nums:
        since = time.time()
        res = digitCancellationsEqualToSelfProdDenom(
            denom_max=99,
            base=10,
            leading_zeros_allowed=False,
            exclude_trivial=True,
        )
        print(f"Solution to Project Euler #33 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 34 in eval_nums:
        since = time.time()
        res = digitFactorialSumEqualsSelfSum(n_dig_min=2, base=10)
        print(f"Solution to Project Euler #34 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 35 in eval_nums:
        since = time.time()
        res = circularPrimesCount(n_max=10 ** 6 - 1, base=10)
        print(f"Solution to Project Euler #35 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 36 in eval_nums:
        since = time.time()
        res = multiBasePalindromesSum(n_max=10 ** 6 - 1, bases=(2, 10))
        print(f"Solution to Project Euler #36 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 37 in eval_nums:
        since = time.time()
        res = truncatablePrimesSum(
            mx_n_dig=None,
            left_truncatable=True,
            right_truncatable=True,
            base=10,
        )
        print(f"Solution to Project Euler #37 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 38 in eval_nums:
        since = time.time()
        res = multiplesConcatenatedPandigitalMax(
            min_n_mult=2,
            min_dig=1,
            max_dig=None,
            base=10,
        )
        print(f"Solution to Project Euler #38 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 39 in eval_nums:
        since = time.time()
        res = pythagTripleMaxSolsPerim(perim_max=1000)
        print(f"Solution to Project Euler #39 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 40 in eval_nums:
        since = time.time()
        res = champernowneConstantProduct(
            n_list=[10 ** x for x in range(7)],
            base=10
        )
        print(f"Solution to Project Euler #40 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 41 in eval_nums:
        since = time.time()
        res = largestPandigitalPrime(base=10, incl_zero=False)
        print(f"Solution to Project Euler #41 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 42 in eval_nums:
        since = time.time()
        res = countTriangleWordsInTxtDoc(
            doc="project_euler_problem_data_files/p042_words.txt",
            rel_package_src=True,
        )
        print(f"Solution to Project Euler #42 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 43 in eval_nums:
        since = time.time()
        res = pandigitalDivPropsSum(
            subnum_n_dig=3,
            n_prime=7,
            min_dig=0,
            max_dig=None,
            base=10,
        )
        print(f"Solution to Project Euler #43 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 44 in eval_nums:
        since = time.time()
        res = kPolygonalMinKPolyDiff(k=5, max_k_poly_diff=10 ** 6)
        print(f"Solution to Project Euler #44 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 45 in eval_nums:
        since = time.time()
        res = nthSmallestTriangularPentagonalAndHexagonalNumbers(n=3)
        print(f"Solution to Project Euler #45 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 46 in eval_nums:
        since = time.time()
        res = goldbachOtherChk(n_max=10 ** 6)
        print(f"Solution to Project Euler #46 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 47 in eval_nums:
        since = time.time()
        res = smallestnConsecutiveMDistinctPrimeFactorsUnlimited(
            n=4,
            m=4,
            p_sieve=None,
        )
        print(f"Solution to Project Euler #47 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 48 in eval_nums:
        since = time.time()
        res = selfExpIntSumLastDigits(n_max=1000, n_digits=10, base=10)
        print(f"Solution to Project Euler #48 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 49 in eval_nums:
        since = time.time()
        res = primePermutArithmeticProgressionConcat(
            n_dig=4,
            seq_len=3,
            base=10,
        )
        print(f"Solution to Project Euler #49 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 50 in eval_nums:
        since = time.time()
        res = primeSumOfMostConsecutivePrimes(
            n_max=10 ** 6 - 1,
        )
        print(f"Solution to Project Euler #50 = {res}, calculated in {time.time() - since:.4f} seconds")

    return

if __name__ == "__main__":
    eval_nums = {50}
    evaluateProjectEulerSolutions1to50(eval_nums)