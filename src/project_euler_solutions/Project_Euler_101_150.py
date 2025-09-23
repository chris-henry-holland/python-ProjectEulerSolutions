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
    Iterable,
)

import bisect
import heapq
import itertools
import math
import time

import sympy as sym

from collections import deque
from sortedcontainers import SortedDict, SortedList, SortedSet

from data_structures.addition_chains import AdditionChainCalculator
from data_structures.fractions import CustomFraction
from data_structures.prime_sieves import (
    PrimeSPFsieve,
    SimplePrimeSieve,
)

from algorithms.continued_fractions_and_Pell_equations import (
    generalisedPellSolutionGenerator,
    pellSolutionGenerator,
)
from algorithms.number_theory_algorithms import (
    gcd,
    lcm,
    isqrt,
    integerNthRoot,
)

from project_euler_solutions.utils import (
    loadTextFromFile,
    UnionFind,
)

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
    #print(frac1, frac2, (numer // g, denom // g))
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
    neg = (frac1[0] < 0) ^ (frac1[1] < 0) ^ (frac2[0] < 0) ^ (frac2[1] < 0)
    frac_prov = (abs(frac1[0] * frac2[0]), abs(frac1[1] * frac2[1]))
    #print(frac_prov)
    g = gcd(frac_prov[0], frac_prov[1])
    return (-(frac_prov[0] // g) if neg else (frac_prov[0] // g), frac_prov[1] // g)

# Problem 101
# Review- Look into Lagrange polynomial interpolation
def polynomialFit(seq: List[int], n0=0) -> Tuple[Tuple[int], int]:
    """
    For an integer sequence seq such that the first value in seq
    corresponds to n = n0, finds the coefficients of the lowest order
    polynomial P(n) such that P(n) = seq[n - n0] for each integer n
    between n0 and n0 + len(seq) - 1 inclusive.
    
    Args:
        Required positional:
        seq (List of ints): The integer sequence in question
        
        Optional named:
        n0 (int): The value of n to which the first element of seq
                corresponds
            Default: 0
    
    Returns:
    Tuple whose 0th index contains a tuple of integers representing
    the numerators of the coefficients of the polynomial, where the
    entry at index i represents the coefficient of the n ** i term,
    and whose 1st index contains an integer (int) representing the
    denominator of all those coefficients.
    """
    m = len(seq)
    #mat = np.zeros((m, m), dtype=np.uint64)
    mat = [[0] * m for _ in range(m)]
    for i1 in range(n0, n0 + m):
        mat[i1 - n0][0] = 1
        if not i1: continue
        for i2 in range(1, m):
            mat[i1 - n0][i2] = i1 ** i2
    # Review- try to avoid using sympy
    mat = sym.matrices.Matrix(mat)
    vec = sym.matrices.Matrix(m, 1, seq)
    #res = np.linalg.solve(mat, np.array(seq, dtype=np.int64))
    res = mat.LUsolve(vec)
    res = list(res)
    while len(res) > 1 and not res[-1]:
        res.pop()
    denom = 1
    for frac in res:
        if hasattr(frac, "denominator"):
            denom = lcm(denom, frac.denominator)
    return (tuple(int(x * denom) for x in res), denom)

def optimumPolynomial(
    coeffs: Tuple[int]=(1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1),
) -> Union[int, float]:
    """
    Solution to Project Euler #101

    Consider a polynomial P(n) (where n is the variable of the
    polynomial) whose coefficients are given by the tuple of ints
    coeffs, where the value at the ith index of coeffs is the
    coefficient of the n ** i term in the polynomial.
    
    For a given sequence of finite length, the optimum polynomial Q(n)
    is the polynomial with minimum degree such that for Q(i) is equal
    to the ith element of the sequence for all integers i between 1
    and the length of the sequence. It can be shown that this uniquely
    defines such a polynomial.
    
    For the polynomial P(n) define OP(k, m) to be the mth term of the
    optimum polynomial for the sequence of length k such that the
    ith term in the sequence is equal to P(i).
    
    This function calculates the sum of OP(k, m(k)) over all
    strictly positive integers k for which m(k) is defined, where for
    each k, m(k) is the smallest positive integer m for which
    OP(k, m) is not equal to P(m), if any, and is undefined if there
    is no such m.
    
    Note that for k > (degree of P(n)), OP(k, m) = P(m) for all m,
    given that for any sequence defined to be a polynomial with more
    terms than the degree of the polynomial, a consequence of the
    fundamental theorem of algebra is that the optimal polynomial
    must be equal to the polynomial that defined the sequence.
    Therefore, only the terms where k is between 1 and the degree of
    P(n) contribute to the sum.
    
    Args:
        Optional named:
        coeffs (tuple of ints): The coefficients of the polynomial P(n)
                where the value at the ith index of the tuple is the
                coefficient of the n ** i term in the polynomial.
            Default: (1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1)
    
    Returns:
    Number (int/float) giving the described sum.
    """
    #since = time.time()
    n = len(coeffs)
    res = [0, 1]
    ans_real = sum(coeffs)
    seq = [ans_real]
    skip_count = 0
    i = 2
    while i <= n or skip_count:
        if not skip_count:
            poly, denom = polynomialFit(seq, n0=1)
        #print(seq)
        #print(poly)
        ans_poly = poly[0]
        for j in range(1, len(poly)):
            ans_poly += poly[j] * i ** j
        ans_real = coeffs[0]
        for j in range(1, n):
            ans_real += coeffs[j] * i ** j
        seq.append(ans_real)
        #print(i, ans_poly, ans_real, skip_count)
        if ans_poly * res[1] != ans_real * denom:
            #print("hi")
            denom2 = lcm(denom, res[1])
            res = [res[0] * (denom2 // res[1]) +\
                    (skip_count + 1) * ans_poly * (denom2 // denom), denom2]
            #print(res)
            g = gcd(*res)
            res = [x // g for x in res]
            skip_count = 0
        else: skip_count += 1
        i += 1
    #print(seq)
    res2 = res[0] if res[1] == 1 else res[0] / res[1]
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res2

# Problem 102
def loadTrianglesFromFile(
    doc: str,
    rel_package_src: bool=False,
) -> List[Tuple[Tuple[int]]]:
    """
    Loads the coordinates in the plane of the vertices of a
    sequence of triangles from the .txt file at relative or
    absolute location doc.
    In this .txt file, the set of coordinates representing each
    triangle are separated by a line break ('\\n') and each set
    of coordinates is a list of 6 numbers, with each consecutive
    pair representing the 2-dimensional Cartesian coordinates of
    each vertex of the triangle.
    
    Args:
        Required positional:
        doc (str): The relative or absolution location of the .txt
                file containing the coordinates of the vertices
                of the triangles.

        Optional named:
        rel_package_src (bool): Whether a relative path given by doc
                is relative to the current directory (False) or
                the package src directory (True).
            Default: False
    
    Returns:
    A list of 3-tuples of 2-tuples of ints. Each element of the list
    represents one of the triangles in the .txt file at location
    doc in the same order as they appear in that file. Each entry
    of the list is a 3-tuple, with the entries of this 3-tuple being
    the 2-dimensional Cartesian coordinates (as a 2-tuple of ints)
    of the vertices of the triangle.
    """
    txt = loadTextFromFile(doc, rel_package_src=rel_package_src)
    res = []
    for s in txt.split("\n"):
        if not s: continue
        nums = [int(x.strip()) for x in s.split(",")]
        triangle = []
        for i in range(0, len(nums), 2):
            triangle.append((nums[i], nums[i + 1]))
        res.append(tuple(triangle))
    return res

def lineEquation(v1: Tuple[int], v2: Tuple[int]) -> Tuple[int]:
    """
    Given Cartesian coordinates of two distinct points on the plane v1
    and v2 with integer coordinates, finds the equation for the unique
    straight line that passes through both of these points in the form:
        a * x + b * y = c
    where x and y are the Cartesian coordinates and a, b and c are
    constants such that gcd(a, b) = 1 and a > 0 or a = 0 and b > 0.
    
    Args:
        Required positional:
        v1 (2-tuple of ints): The Cartesian coordinates in the form
                (x, y) of one of the points the line must pass through
        v2 (2-tuple of ints): The Cartesian coordinates in the form
                (x, y) the other points the line must pass through.
                Note that v2 must be different to v1 in at least one
                of its entries.
    
    Returns:
    3-tuple of ints representing the equation found for the straight
    line found, in the form:
        (a, b, c)
    where the equation for the line is given by:
        a * x + b * y = c
    """
    dx, dy = [x1 - x2 for x1, x2 in zip(v1, v2)]
    if not dx and not dy:
        raise ValueError("v1 and v2 must be different")
    if (dy, dx) < (0, 0): dx, dy = -dx, -dy
    return (dy, -dx, dy * v1[0] - dx * v1[1])

def lineAxisIntersectionSign(line_eqn: Tuple[int], axis: int=0)\
        -> Optional[int]:
    """
    For a straight line represented by line_eqn, such that if
    line_eqn = (a, b, c) then the line has the equation:
        a * x + b * y = c
    where (x, y) are Cartesian coordinates of the plane, finds
    whether the point of intersection with the given axis (where
    axis=0 represents the x-axis and axis=1 the y-axis) exists,
    and if so whether the value of that axis at the point of
    intersection is positive, negative or zero.
    
    Args:
         Required positional:
         line_eqn (3-tuple of ints): The representation of the
                equation of the line in the plane to be considered,
                where for line_eqn = (a, b, c) the line it
                represents has the equation:
                    a * x + b * y = c
                where (x, y) are Cartesian coordinates of the plane
         
         Optional named:
         axis (int): Either 0 or 1, with 0 representing that the
                intersection with the x-axis is to be considered
                and 1 representing that the intersection with the
                y-axis is to be considered.
    
    Returns:
    Integer (int) between -1 and 1 inclusive or None. If the line
    is parallel to the chosen axis (meaning either the line never
    meets the axis or the line coincides with the axis along its
    whole length) then returns None. Otherwise, returns -1 if the
    point of intersection is on the negative portion of the chosen
    axis, 1 if the point of intersection is on the positive portion
    of the chosen axis and 0 if it passes through exactly 0 on the
    chosen axis (i.e. the line passes through the origin of the
    Cartesian coordinate system used).
    """
    if not line_eqn[axis]: return None
    if not line_eqn[-1]: return 0
    return -1 if (line_eqn[axis] < 0) ^ (line_eqn[-1] < 0) else 1
    
def crossProduct2D(vec1: Tuple[int], vec2: Tuple[int]) -> int:
    """
    Finds the value of the cross product of two vectors in two
    dimensions, where the vectors are represented in terms of a
    right-handed orthonormal basis (e.g. Cartesian coordinates).
    The cross product of two vectors v1 and v2 in two dimensions is
    the scalar value:
        vec1 x vec2 = (length of v1) * (length of v2) *\
                            sin(angle between v1 and v2)
    where the angle from v1 to v2 (i.e. the angle vector v1 needs to be
    turned by in order to be made parallel to vector v2) is positive
    if it is an anti-clockwise turn and negative if it is a clockwise
    turn. Note that this is antisymmetric so:
       vec2 x vec1 = -vec1 x vec2.
    
    Args:
        vec1 (2-tuple of ints): The representation of the vector
                appearing first in the cross product in terms of the
                basis vectors (i.e. for basis vectors i and j, the
                vector represented is vec1[0] * i + vec1[1] * j.
                In this case, an right-handed orthonormal basis,
                i and j are unit vectors orthogonal to each other,
                and if i is turned pi/2 radians in an anticlockwise
                direction it will be parallel to j. In terms of
                Cartesian coordinates, here i is a unit vector parallel
                with the x-axis and pointing in the direction of
                increasing x, while j is a unit vector parallel with
                the y-axis and pointing in the direction of increasing
                y).
        vec2 (2-tuple of ints): The representation of the vector
                appearing second in the cross product in terms of the
                basis vectors, similarly to vec1.
    
    Returns:
    Integer (int) giving the value of the cross product of the vector
    represented by vec1 with the vector represented by vec2.
    """
    return vec1[0] * vec2[1] - vec1[1] * vec2[0]

def triangleContainsPoint(
    p: Tuple[int],
    triangle_vertices: Tuple[Tuple[int]],
    include_surface: bool=False,
) -> bool:
    """
    Using the 2-dimensional cross product, finds whether the point
    with 2-dimensional Cartesian coordinates p falls inside a triangle
    whose vertices are at Cartesian coordinates given by
    triangle_vertices.
    Points falling exactly on the edges and vertices of the triangle
    are considered as being inside the triangle if and only if
    include_surface is given as True.
    
    Args:
        Required positional:
        p (2-tuple of ints): The 2-dimensional Cartesian coordinate
                of the point in the plane under consideration.
        triangle_vertices (3-tuple of 2-tuples of ints): The
                2-dimensional Cartesian coordinates of the vertices
                of the triangle being considered.
        
        Optional named:
        include_surface (bool): Whether of not points falling
                exactly on the edges or vertices of the triangle
                are considered to be inside the triangle.
    
    Returns:
    Boolean (bool) giving whether the point with 2-dimensional
    Cartesian coordinates p is inside the triangle whose vertices
    have the 2-dimensional Cartesian coordinates triangle_vertices
    (where points falling exactly on the edges and vertices of the
    triangle are considered as being inside the triangle if and only
    if include_surface was given as True).
    
    Outline of rationale:
    If we arbitrarily label the vertices v1, v2 and v3. Consider the
    set of values given by the 2d cross product of the vector from
    the chosen point to v1 with the vector from v1 to v2, the 2d cross
    product of the vector from the chosen point to v2 with the vector
    from v2 to v3 and the 2d cross product of the vector from the
    chosen point to v3 with the vector from v3 to v1.
    
    It can be shown that if all values in this set are either all
    positive or all negative, then the point is strictly inside
    the triangle, while if there are both positive and negative
    values in this set then the point is strictly outside the
    triangle. For the remaining possible cases, the point is
    exactly on a vertex or edges- if two of the values are 0
    then the point is on a vertex, while if one value is 0 while
    the other two values are either both positive or both negative
    then the point is exactly on one of the edges.
    """
    res = set()
    v2 = triangle_vertices[2]
    for i in range(3):
        v1 = v2
        v2 = triangle_vertices[i]
        ans = crossProduct2D(tuple(x - y for x, y in zip(v1, p)),\
                tuple(x - y for x, y in zip(v2, v1)))
        if not ans:
            if not include_surface: return False
            continue
        res.add(ans > 0)
        if len(res) > 1: return False
    return True

"""
    Detail:
    Consider the vector from the point at p to some point on one of the
    surface of the triangle (i.e. on one of the edges or vertices of
    the triangle), and consider how the direction of this vector
    changes as we move p around one complete circuit of the surface of
    the triangle. If the point represented by p is strictly inside the
    triangle (i.e. within the triangle and not on the surface), then
    during this process the vector is always turning in the same
    direction, either clockwise or anticlockwise and turns through
    exactly 2 * pi radians. On the other hand if the point represented
    by p is strictly outside the triangle then the vector first turns
    in one direction, then the other, and the net angle it turns
    through is exactly 0 radians. We consider the case where the point
    is on the surface separately.
    
    We now consider how the vector turns when traversing an edge, from
    one vertex of the triangle to another. Either the vector turns
    only in one direction, or (as is the case when the edge and the
    vector are parallel, and recalling that we are not yet considering
    the case when the point is on one of the edges) not at all. In
    either case, the direction of turn does not change. Consequently,
    the direction the vector turns can only change at one of the
    vertices, when transitioning from one edge to the next.
    
    With this in mind, as long as the point is not on the surface of
    the triangle, if we consider turning the vector to point from
    one vertex to the next moving round the triangle in one direction
    and if we always choose to turn the vector in the direction that
    results in the vector turning the least (i.e. choosing the
    angle that is strictly less than pi / 2 noting that given that
    the point is not on an edge, a turn of pi / 2 cannot occur),
    if the point is inside the triangle then the angle will always
    turn in the same direction, while if the point is outside the
    triangle the direction will sometimes be in one direction and
    sometimes in the opposite direction (and in some cases may
    not turn at all).
    
    This can be quantified by using the 2-dimensional cross product.
    If we take the cross product of the vector from the point to
    the current vertex with the vector from that vertex to the
    next vertex in the traversal, then a positive result signifies
    that between these two vertices, the vector turns anti-clockwise,
    a negative result that the vector turns clockwise and a zero
    result that the vector does not turn.
    
    Thus, if two of the cross products between the vector from
    the point to a vertex and the vector from that vertex to
    the next vertex in that traversal have a different sign,
    this signifies that the point is outside the triangle.
    
    Conversely, if these cross products are all positive or
    all negative then the point is inside the triangle.
    
    We now consider what happens when the point is on the
    surface of the triangle. If it is at a vertex, then since
    the vector from the point to that vertex is length 0, the
    cross product of that vector with any other vector is also
    zero. Additionally, the cross product of the vector to
    the preceding vertex in the traversal is exactly the
    negative of the vector from the preceding vertex to this
    vertex, so the cross product for the preceding vertex
    is also 0. This leaves only one potentially non-zero
    cross product (between the other two vertices). In fact,
    since the corresponding edge cannot be parallel to
    either of the other two in order for the vertices to
    be considered a triangle, this cross product must be non-zero.
    
    TODO- revise
    
    
    If we label the vertices v1, v2 and v3 and consider the vectors
    vec1 from v2 to v1, vec2 from v2 to v3 and vec3 from v3 to v1,
    
"""

def triangleContainsOrigin(v1: Tuple[int], v2: Tuple[int], v3: Tuple[int]) -> bool:
    """
    Finds whether the point at the origin (0, 0) of a given Cartesian
    coordinate system falls inside a triangle whose vertices are at
    Cartesian coordinates given by v1, v2 and v3.
    Points falling exactly on the edges and vertices of the triangle
    are considered as being inside the triangle.
    
    Args:
        Required positional:
        v1 (2-tuple of ints): The 2-dimensional Cartesian coordinates
                of one of the vertices of the triangle being
                considered.
        v2 (2-tuple of ints): The 2-dimensional Cartesian coordinates
                of another of the vertices of the triangle being
                considered.
        v3 (2-tuple of ints): The 2-dimensional Cartesian coordinates
                of the final vertex of the triangle being considered.
    
    Returns:
    Boolean (bool) giving whether the origin of the 2-dimensional
    Cartesian coordinate system used is inside the triangle whose
    vertices have the 2-dimensional Cartesian coordinates v1, v2
    and v3 (where points falling exactly on the edges and vertices of
    the triangle are considered as being inside the triangle).
    
    Not currently used- superceded by triangleContainsPoint()
    """
    if v1 == (0, 0) or v2 == (0, 0) or v3 == (0, 0):
        return True
    if (v1[0] > 0 and v2[0] > 0 and v3[0] > 0) or\
            (v1[0] < 0 and v2[0] < 0 and v3[0] < 0) or\
            (v1[1] > 0 and v2[1] > 0 and v3[1] > 0) or\
            (v1[1] < 0 and v2[1] < 0 and v3[1] < 0):
        return False
    
    intercept_sgns = [set(), set()]
    
    for pair in ((v1, v2), (v2, v3), (v3, v1)):
        u1, u2 = pair
        #print(pair)
        #print([x * y > 0 for x, y in zip(*pair)])
        if all(x * y > 0 for x, y in zip(*pair)):
            continue
        eqn = lineEquation(*pair)
        #print(pair)
        #print(eqn)
        for i in range(2):
            if pair[0][~i] * pair[1][~i] > 0: continue
            ans = lineAxisIntersectionSign(eqn, axis=i)
            #print(pair, i, ans)
            if ans is not None:
                intercept_sgns[i].add(ans)
    #print(intercept_sgns)
    return all(len(x) > 1 for x in intercept_sgns)

def countTrianglesContainingPointFromFile(
    p: Tuple[int]=(0, 0),
    doc: str="project_euler_problem_data_files/0102_triangles.txt",
    rel_package_src: bool=True,
    include_surface: bool=True,
) -> int:
    """
    Solution to Project Euler #102

    Given the list of triangles represented by the 2-dimensional
    Cartesian coordinates of their vertices in the .txt file at
    location doc (see loadTriangles()), counts how many of these
    triangles contain the point with 2-dimensional Cartesian
    coordinates p.

    Args:
        Optional named:
        p (2-tuple of ints): The 2-dimensional Cartesian coordinates
                of the point in the plane of interest.
            Default: (0, 0)
        doc (str): The relative or absolute location of the .txt file
                containing the coordinates of the vertices of the
                triangles.
            Default: "project_euler_problem_data_files/0102_triangles.txt"
        rel_package_src (bool): Whether a relative path given by doc
                is relative to the current directory (False) or
                the package src directory (True).
            Default: True
        include_surface (bool): If True, considers points that are
                exactly on the edge or on a vertex of a given triangle
                as being inside that triangle, otherwise considers
                these points to be outside the triangle.
            Default: True
    
    Returns:
    Integer (int) giving the number of triangles from the .txt file
    at location doc contain the point with Cartesian coordinates
    p, subject to the specified classification of points falling
    exactly on an edge or vertex of a triangle.
    """
    #since = time.time()
    triangles = loadTrianglesFromFile(
        doc,
        rel_package_src=rel_package_src,
    )
    res = sum(
        triangleContainsPoint(
            p,
            x,
            include_surface=include_surface,
        ) for x in triangles
    )
    #res = sum(triangleContainsOrigin(*x) for x in triangles)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 103
def isSpecialSumSet(nums: Tuple[int], nums_sorted: bool=False) -> bool:
    """
    For a given list of integers nums, identifies whether it is
    a special sum set.

    A special sum set is a set of distinct positive integers for
    which:
     1) For any two disjoint non-empty subsets (i.e. subsets that
        contain at least one element and have no common element)
        the sums over all elements is different for the two subsets
     2) For any subset, the sum over all elements in that subset
        is strictly greater than that of any other subset that is
        disjoint with the chosen set that contains fewer elements.
    
    
    Args:
        Required positional:
        nums (tuple of ints): The set of integers to be assessed
                for whether it is a special sum set.
        
        Optional named:
        nums_sorted (bool): Whether the contents of nums has
                already been sorted.
            Default: False
    
    Returns:
    Boolean (bool) giving True if nums represents a special sum
    set and False if not.
    
    Note- the two conditions for a special sum set are equivalent
    to the same conditions with the disjoint requirement being
    replaced by a distinct requirement. This is because the
    distinct requirement encompasses the disjoint requirement, and
    in both of the conditions, if there exist two distinct non-empty
    subsets that violate that condition, then by removing the common
    elements of the two sets, we can construct disjoint sets that
    violate the condition, at least one of which must be non-empty.
    If the constructed sets are both non-empty then there exist
    disjoint non-empty sets that violate one of the conditions. On
    the other hand it is actually impossible for either of the
    constructed sets to be empty if the original sets violate one
    of the conditions, as for sets containing strictly positive
    integers an empty set always has a strictly smaller sum (0)
    than a non-empty set. Thus, this replacement of distinct for
    disjoint gives rise equivalent conditions for the special sum
    set. As this condition is easier to work with, it is used
    instead in the calculation.
    """
    n = len(nums)
    # Sorting and ensuring no repeated elements
    if not nums_sorted:
        if len(set(nums)) != n: return False
        nums = sorted(nums)
    else:
        for i in range(n - 1):
            if nums[i] == nums[i + 1]: return False
    # Checking that all elements are strictly positive
    if nums[0] < 1: return False
    # Checking that all subsets have sums strictly greater
    # than any subsets with fewer elements
    curr = [0, 0]
    for i in range(-((-n) >> 1)):
        curr[0] += nums[i]
        if curr[0] <= curr[1]: return False
        curr[1] += nums[~i]
    # Checking that there are no repeated sums
    seen = set()
    for bm in range(1, 1 << n):
        cnt = 0
        sm = 0
        for i in range(n):
            if bm & 1:
                cnt += 1
                sm += nums[i]
            bm >>= 1
        if sm in seen:
            return False
        seen.add(sm)
    return True

def findOptimalSpecialSumSets(n: int) -> List[Tuple[int]]:
    """
    Identifies every optimal special sum set with n elements.

    A special sum set is a set of distinct positive integers for
    which:
     1) For any two disjoint non-empty subsets (i.e. subsets that
        contain at least one element and have no common element)
        the sums over all elements is different for the two subsets
     2) For any subset, the sum over all elements in that subset
        is strictly greater than that of any other subset that is
        disjoint with the chosen set that contains fewer elements.
    
    An optimal special sum set for a given number of elements is
    a special sum set such that the sum of its elements is no
    greater than than of any other special sum set with the same
    number of elements.

    Args:
        Required positional:
        n (int): The number of elements for which an optimal special
                sum set is sought.
    
    Returns:
    List of n-tuples containing strictly positive integers (int),
    representing every optimal special sum set with n elements,
    each sorted in strictly increasing order. The optimal special
    sum sets are sorted in lexicographically increasing order
    over the elements from left to right.
    
    Outline of rationale:
    We can simplify the requirements by making the following
    observations:
     1) Two non-empty disjoint subsets of A exist with equal sum
        if and only if two unequal subsets of A exist with equal sum
     2) Two non-empty disjoint subsets of A exist such that one of
        the subsets has more elements than the other but a sum not
        exceeding that of the other, if and only if two unequal
        subsets of A exist such that one of the subsets has more
        elements than the other but a sum not exceeding that of the
        other.
     3) Two unequal subsets of A exist such that one of the
        subsets has more elements than the other but a sum not
        exceeding that of the other if and only if there exists an
        integer m such that 1 <= m <= n / 2 (where n is the
        number of elements in A) and:
            (sum of smallest m + 1 elements of A) <=
                                (sum of largest m elements of A)
    TODO
    
    
    We use a backtracking algorithm. We prune the search space
    by noting that the sum of the two smallest elements of the
    set must be strictly greater than the largest element. For
    a given sum of the two smallest elements, this makes the number
    of cases to check finite.
    
    We search for increasing sum of the two smallest elements until
    we have found a candidate. This allows us to further reduce the
    search space until we reach a sum of two smallest elements such
    that the smallest possible sum of a set with this sum of the
    smallest two elements is at least as large as the current best sum,
    at which point we can conclude that the current best candidate
    is an optimal special sum set, and so return this.
    """
    if n == 1: return (1,)
    elif n == 2: return (1, 2)
    
    curr_best = float("inf")
    curr = [0] * n
    curr_sums = [0, 0]
    
    def recur(sum_set: Set[int], i1: int=2, i2: int=n - 2)\
            -> Generator[Tuple[int], None, None]:
        #if i1 == 2:
        #    print(curr, curr_sums)
        #print(i1, i2, curr, sum_set, i1 + n - 1 - i2, len(sum_set))
        tot_sum = sum(curr_sums)
        if tot_sum >= curr_best: return
        if i1 > i2:
            yield tuple(curr)
            return
        elif i1 == i2:
            for num in range(curr[i2 - 1] + 1, min(curr[i1 + 1], curr_best - tot_sum)):
                if num in sum_set: continue
                for x in sum_set:
                    if x + num in sum_set:
                        break
                else:
                    curr[i1] = num
                    curr_sums[0] += num
                    yield tuple(curr)
                    curr_sums[0] -= num
                #if isSpecialSumSet(curr):
                #    curr_sums[0] += num
                #    yield tuple(curr)
                #    curr_sums[0] -= num
            return
        n_remain = i2 - i1 + 1
        lb = ((curr[i1 - 1] + n_remain) * (curr[i1 - 1] + n_remain + 1) -\
                curr[i1 - 1] * (curr[i1 - 1] + 1)) >> 1
        
        lb2 = lb - curr[i1 - 1] - n_remain
        #if i1 == 2 and curr[0] == 11 and curr[1] == 18:
        #    print(i1, i2, curr, curr_sums, curr_best)
        #    print(lb, lb2, max(curr[i1 - 1], curr_sums[1] - curr_sums[0]) + 1,\
        #        min(curr[i1 - 1] + (curr_best - tot_sum - lb) // n_remain + 2,\
        #        curr[i2 + 1] - n_remain + 1))
        #print(curr[i1 - 1] + (curr_best - tot_sum - lb) // n_remain + 2,\
        #        curr[i2 + 1] - n_remain + 1)
        rng_mx = curr[i2 + 1] - n_remain + 1
        if isinstance(curr_best, int):
            rng_mx = min(rng_mx,\
                    curr[i1 - 1] + (curr_best - tot_sum - lb) // n_remain + 2)
        for num1 in range(max(curr[i1 - 1], curr_sums[1] - curr_sums[0]) + 1,\
                rng_mx):
            #if (num - 2 - curr[i1 - 1]) * n_remain > curr_best - tot_sum - lb:
            if num1 in sum_set: continue
            sum_set2 = set(sum_set)
            sum_set2.add(num1)
            for x in sum_set:
                x2 = x + num1
                if x2 in sum_set2: break
                sum_set2.add(x2)
            else:
                curr[i1] = num1
                curr_sums[0] += num1
                lb2 += n_remain - 1
                for num2 in range(num1 + n_remain - 1,\
                        min(curr_sums[0] - curr_sums[1],\
                        curr[i2 + 1] + 1,\
                        curr_best - tot_sum - lb2 + 1)):
                    if num2 in sum_set2: continue
                    sum_set3 = set(sum_set2)
                    sum_set3.add(num2)
                    for x in sum_set2:
                        x2 = x + num2
                        if x2 in sum_set3: break
                        sum_set3.add(x2)
                    else:
                        curr[i2] = num2
                        curr_sums[1] += num2
                        yield from recur(sum_set=sum_set3, i1=i1 + 1, i2=i2 - 1)
                        curr_sums[1] -= num2
                curr_sums[0] -= num1
        return
        """
        for num in range(num2 + i - 1, curr[i + 1]):
            curr[i] = num
            curr_sm[0] += num
            if curr_sm[0] + (((curr[1] + i - 3) * (curr[1] + i - 2)\
                    - curr[1] * (curr[1] + 1)) >> 1) >= curr_best:
                curr_sm[0] -= num
                break
            yield from recur(i=i - 1)
            curr_sm[0] -= num
        
        return
        """
    
    res = []
    curr_best = float("inf")
    pair1_sum = ((n - 1) << 1) + 1
    while True:
        #print(f"pair 1 sum = {pair1_sum}")
        looped = False
        curr_sums = [pair1_sum, 0]
        for num1 in reversed(range(n - 1, -((-pair1_sum) >> 1))):
            num2 = pair1_sum - num1
            #print(pair1_sum, num1, num2)
            lb = num1 + (((num2 + n - 1) * (num2 + n - 2) - num2 * (num2 - 1)) >> 1)
            if lb >= curr_best: break
            looped = True
            curr[0], curr[1] = num1, num2
            for num_mx in range(num2 + n - 2, pair1_sum):
                curr[-1] = num_mx
                curr_sums[1] = num_mx
                sum_set = {num1, num2, num_mx, num1 + num2,\
                        num1 + num_mx, num2 + num_mx, num1 + num2 + num_mx}
                for seq in recur(sum_set=sum_set,i1=2, i2=n - 2):
                    sm = sum(seq)
                    if sm > curr_best: continue
                    if sm < curr_best:
                        curr_best = sm
                        res = []
                    res.append(seq)
        if not looped: break
        pair1_sum += 1
    #print(res)
    return sorted(res)

def specialSubsetSumsOptimum(n: int=7) -> str:
    """
    Solution to Project Euler #103

    Identifies the lexicographically smallest optimum special
    subset sum for n elements (where the lexicographic sorting is
    over the elements in the set from smallest to largest). The
    result is given as the string concatenation of the numbers
    in the identified optimum special subset sum from smallest to
    largest, each expressed in base 10.

    A special sum set is a set of distinct positive integers for
    which:
     1) For any two disjoint non-empty subsets (i.e. subsets that
        contain at least one element and have no common element)
        the sums over all elements is different for the two subsets
     2) For any subset, the sum over all elements in that subset
        is strictly greater than that of any other subset that is
        disjoint with the chosen set that contains fewer elements.
    
    An optimal special sum set for a given number of elements is
    a special sum set such that the sum of its elements is no
    greater than than of any other special sum set with the same
    number of elements.

    Args:
        Optional named:
        n (int): The number of elements for which an optimal special
                sum set is sought.
            Default: 7
    
    Returns:
    String (str) containing the concatenation of the numbers in the
    lexicographically smallest optimum special subset sum for n
    elements (where the lexicographic sorting is over the elements
    in the set from smallest to largest), where the numbers are
    concatenated in order from smallest to largest.
    """
    #since = time.time()
    res = findOptimalSpecialSumSets(n)[0]
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return "".join([str(x) for x in res])

# Problem 104
def isPandigital(num: int, base: int=10, chk_rng: bool=True) -> bool:
    """
    Function assessing whether an integer num is pandigital in a given
    base (i.e. which num is expressed in the chosen base each digit from
    0 to (base - 1)) appears as one of the digits in this expression and
    0 is not the first digit).

    Args:
        Required positional:
        num (int): The number whose status as pandigital in the chosen
                base is being assessed.
        
        Optional named:
        base (int): Integer strictly greater than 1 giving the base in
                which num is to be expressed for its pandigital status.
            Default: 10
        chk_rng (bool): Whether to check that the number is in the
                value range that is a necessary condition for it to
                be pandigital in the chosen base. If this is given
                as False, it is assumed that this has already been
                tested and the test was passed.
            Default: True
    
    Returns:
    Boolean (bool) which is True if num is pandigital in the chosen base,
    False otherwise.
    """
    if chk_rng and not base ** (base - 2) <= num < base ** (base - 1):
        return False
    dig_set = set()
    while num:
        num, r = divmod(num, base)
        if not r or r in dig_set: return False
        dig_set.add(r)
    return True

"""
def startAndEndPandigital(num: int, base: int=10, target: Optional[int]=None,\
        md: Optional[int]=None) -> bool:
    if target is None: target = base ** (base - 2)
    if num < target: return False
    if md is None: md = target * base
    if not isPandigital(num % md): return False
    while num > md: num //= base
    return isPandigital(num)
"""
def FibonacciFirstKDigits(i: int, k: int, base: int=10) -> int:
    """
    Finds the first k digits in the i:th Fibonacci number (where
    the 0th term is 0 and the 1st term is 1) when expressed in
    the chosen base.

    Calculated using Binet's formula.

    Args:
    Required positional:
        i (int): Non-negative integer giving the term in the Fibonacci
                sequence for which the first k digits when expressed
                in the chosen base are to be calculated.
        k (int): Strictly positive integer giving the number of digits
                to be calculated.
        
        Optional named:
        base (int): Integer strictly greater than 1 giving the base
                in which the i:th Fibonacci number is to be expressed
                when finding the first k digits.
            Default: 10
    
    Returns:
    Integer (int) giving the value of the first k digits of the i:th
    Fibonacci number when expressed in the chosen base, when interpreted
    as a number in the chosen base.
    """
    phi = (1 + math.sqrt(5)) / 2 # The golden ratio
    lg_rt5 = math.log(math.sqrt(5), base)
    lg_num = i * math.log(phi, base) - lg_rt5
    if lg_num > k + 1:
        # Ensuring no rounding error
        lg_num2 = (lg_num % 1) + (k - 1)
        diff = round(lg_num - lg_num2)
        cnt = 0
        div = 1
        while cnt < diff:
            res = math.floor(base ** lg_num2)
            if res % base != base - 1:
                return res // div
            #print("hi")
            #print(res)
            div *= base
            cnt += 1
            lg_num2 += 1
    # Calculating exactly
    psi = (1 - math.sqrt(5)) / 2
    res = (phi ** i - psi ** i) / math.sqrt(5)
    #print(res)
    res = round(res)
    mx = base ** k
    while res > mx:
        res //= base
    return res
    

def pandigitalFibonacciStart(i: int, base: int=10) -> bool:
    """
    Finds whether the first base digits in the i:th Fibonacci number 
    where the 0th term is 0 and the 1st term is 1) when expressed in
    the chosen base are pandigital in that base. Leading zeroes are
    not allowed.

    Calculated using Binet's formula (via FibonacciFirstKDigits()).

    Args:
    Required positional:
        i (int): Non-negative integer giving the term in the Fibonacci
                sequence for which the first base digits when expressed
                in the chosen base are to be assessed for their
                pandigital status in that base.
        
        Optional named:
        base (int): Integer strictly greater than 1 giving the base
                in which the i:th Fibonacci number is to be expressed
                when assessing whether its first base digits are
                pandigital.
            Default: 10
    
    Returns:
    Boolean (bool) giving True if the first base digits in the i:th
    Fibonacci number when expressed in that base without leading zeroes
    are pandigital in the chosen base.
    """
    return isPandigital(FibonacciFirstKDigits(i, k=base - 1, base=10))
    
    """
    if target is None: target = base ** (base - 2)
    lg_num = i * math.log((1 + math.sqrt(5)) / 2, base)
    #if lg_num < base - 1: continue
    #num = 10 ** 
    num = round(((1 + math.sqrt(5)) / 2) ** i / math.sqrt(5))
    if num < target: return False
    if md is None: md = target * base
    while num > md: num //= base
    print(i, num)
    res = isPandigital(num, chk_rng=False)
    print(res)
    return res
    """

def pandigitalFibonacciEnds(base: int=10) -> int:
    """
    Solution to Project Euler #104

    Finds the smallest Fibonacci number such that when expressed in
    the chosen base, the first base digits and the last base digits
    are both pandigital in that base. Leading zeroes are not allowed
    for the first base digits.

    Args:
    Required positional:
        Optional named:
        base (int): Integer strictly greater than 1 giving the base
                in which the i:th Fibonacci number is to be expressed
                when assessing whether its first base digits and
                last base digits are pandigital.
            Default: 10
    
    Returns:
    Integer (int) giving the value of the smallest Fibonacci number that
    fulfills the stated requirement.
    """
    #since = time.time()
    if base == 2: return 1
    curr = [1, 1]
    i = 2
    target = base ** (base - 2)
    md = base ** (base - 1)
    while curr[1] < target:
        curr = [curr[1], sum(curr)]
        i += 1
        #if not i % 1000:
        #    print(i)
    curr[1] %= md
    #not startAndEndPandigital(curr[1]):
    while not isPandigital(curr[1], chk_rng=False) or\
            not pandigitalFibonacciStart(i, base=base):
        #if isPandigital(curr[1]) and pandigitalFibonacciStart(i, base=base):
        #    return i
        curr = [curr[1], sum(curr) % md]
        i += 1
        #if not i % 1000:
        #    print(i)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return i

# Problem 105
def loadSetsFromFile(
    doc: str,
    rel_package_src: bool=False,
) -> List[Tuple[int]]:
    """
    Loads sets of integers stored in a .txt file doc.

    The sets are stored in the .txt file as strings separated by
    line breaks ("\n"), with the integers in each set expressed
    in base 10 and each separated by a comma (",").

    Args:
        Required positional
        doc (str): The relative or absolute location of the .txt file
                containing the sets of integers

        Optional named:
        rel_package_src (bool): Whether a relative path given by doc
                is relative to the current directory (False) or
                the package src directory (True).
            Default: False
    
    Returns:
    List of tuples of ints with each entry in the list representing
    one of the sets of integers stored in doc, and each tuple
    containing each of the integers in the corresponding set.
    """
    #if relative_to_program_file_directory and not doc.startswith("/"):
    #    doc = os.path.join(os.path.dirname(__file__), doc)
    #with open(doc) as f:
    #    txt = f.read()
    txt = loadTextFromFile(doc, rel_package_src=rel_package_src)
    return [tuple(int(y.strip()) for y in x.split(",")) for x in txt.split("\n")]

def specialSubsetSumsTestingFromFile(
    doc: str="project_euler_problem_data_files/0105_sets.txt",
    rel_package_src: bool=True,
) -> int:
    """
    Solution to Project Euler #105

    For the sets of integers contained in the .txt file doc,
    identifies which of those sets are special sum sets, and for
    those calculates the total sum of elements over all of those
    sets.

    A special sum set is a set of distinct positive integers for
    which:
     1) For any two disjoint non-empty subsets (i.e. subsets that
        contain at least one element and have no common element)
        the sums over all elements is different for the two subsets
     2) For any subset, the sum over all elements in that subset
        is strictly greater than that of any other subset that is
        disjoint with the chosen set that contains fewer elements.
    
    The sets are stored in the .txt file as strings separated by
    line breaks ("\n"), with the integers in each set expressed
    in base 10 and each separated by a comma (",").
    
    Args:
        Optional named:
        doc (str): The relative or absolute location of the .txt file
                containing the sets of integers
            Default: "project_euler_problem_data_files/0105_sets.txt"
        rel_package_src (bool): Whether a relative path given by doc
                is relative to the current directory (False) or
                the package src directory (True).
            Default: True

    Returns:
    Integer (int) giving the total of the sum of elements in all of the
    sets of integers contained in doc that are special sum sets.
    """
    #since = time.time()
    sp_sets = loadSetsFromFile(doc, rel_package_src=rel_package_src)
    #print(sp_sets)
    res = sum(sum(x) for x in sp_sets if isSpecialSumSet(x, nums_sorted=False))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 106
def specialSubsetSumsComparisons(n: int=12) -> int:
    """
    Solution to Project Euler #106
    
    A special sum set is a set of distinct positive integers for
    which:
     1) For any two disjoint non-empty subsets (i.e. subsets that
        contain at least one element and have no common element)
        the sums over all elements is different for the two subsets
     2) For any subset, the sum over all elements in that subset
        is strictly greater than that of any other subset that is
        disjoint with the chosen set that contains fewer elements.
    
    This function calculates, for sets of distinct positive integers
    with exactly n elements for which the second property holds, the
    minimum number of comparisons of sum equality between subsets
    that need to be made to confirm with certainty whether or not
    any such set also satisfies the first property (and so is or is
    not a special sum set).

    Args:
        Optional named:
        n (int): Non-negative integer giving the number of elements
                in the sets under consideration.
            Default: 12
    
    Returns:
    Integer (int) giving the minimum number of comparisons of sum
    equality between subsets that need to be made to confirm with
    certainty whether or not a set of distinct positive integers
    with exactly n elements satisfying the second property given
    above is or is not a special sum set.

    Outline of rationale:
    TODO
    """
    res = 0
    for i in range(2, (n >> 1) + 1):
        # Using Catalan numbers
        res += (math.comb(n, 2 * i) * math.comb(2 * i, i) *\
                (i - 1)) // (2 * (i + 1))
    return res

# Problem 107
def loadNetworkFromFile(
    doc: str,
    rel_package_src: bool=False,
) -> Tuple[Union[int, List[Tuple[int]]]]:
    """
    Loads a the weighted edges from a weighted adjacency matrix
    for an undirected network with integer weights stored in the
    .txt file located at doc.
    
    The file should contain the rows of the matrix in order,
    separated by line breaks ('\\n'), with each entry of the row
    being either a non-negative integer represented in base 10 by
    arabic numerals representing that a single undirected edge
    exists between the corresponding vertices in the network with
    weight equal to that number or a hyphen ('-') denoting that
    no edge exists between those two vertices, with each entry
    separated by a single comma (',') only (i.e. no space before
    or after the comma).

    The matrix should be square (i.e. each row contains the same
    number of entries as there are rows) and symmetric about the
    leading diagonal (so the entry in the ith row and jth column
    is equal to the entry in the jth row and ith column for all
    valid choiced of i and j), and no self-edges (so no edges
    along the leading diagonal).
    
    Args:
        Required positional:
        doc (str): The relative or absolution location of the .txt
                file containing the triangle of integers.

        Optional named:
        rel_package_src (bool): Whether a relative path given by doc
                is relative to the current directory (False) or
                the package src directory (True).
            Default: False
    
    Returns:
    2-tuple whose index 0 contains an integer (int) giving the number
    of rows (and columns) of the matrix and whose index 1 contains
    a list of 3-tuples of ints, each representing the edges of the
    network, with indices 0 and 1 containing the indices of the
    connected vertices (with the index corresponding to the row and
    column number, starting at zero) in increasing order and index
    2 containing the weight of that edge.
    """
    #if relative_to_program_file_directory and not doc.startswith("/"):
    #    doc = os.path.join(os.path.dirname(__file__), doc)
    #with open(doc) as f:
    #    txt = f.read()
    txt = loadTextFromFile(doc, rel_package_src=rel_package_src)
    res = []
    arr = txt.split("\n")
    n = len(arr)
    for i1, row in enumerate(arr):
        if not row: continue
        row2 = row.split(",")
        for i2 in range(i1):
            v = row2[i2].strip()
            #print(v)
            if v == "-": continue
            res.append((i2, i1, int(v)))
    return (n, res)


# Review- documentation for clarity
def kruskallAlgorithm(n: int, edges: List[Tuple[int, int, int]]):
    """
    Implementation of Kruskall's algorithm for finding a
    minimum spanning forest of a weighted undirected graph
    with integer weights.

    A spanning forest of an undirected graph is a subgraph
    of that graph containing all of its vertices, for which
    all vertices that were in the same connected component
    in the original graph are still in the same connected
    component and each connected component is a tree.

    A connected component of an undirected graph is a part
    of the graph consisting of a subset of its vertices and
    edges for which:
     1) For every pair of vertices in the connected component,
        a path through the graph exists between them.
     2) There exists no path through the graph between any
        vertex in the connected component and any vertex not
        in the connected component.
     3) An edge is in the connected component if and only if
        the two vertices it connects are in the connected
        component (note that by definition, either both
        adjacent vertices to the edge are in the connected
        component or both are not in the connected component).
    An undirected graph whose every vertex are in the same
    connected component (and so there exists a path through
    the graph between every pair of vertices in the graph) is
    referred to as a connected graph.

    A tree is a graph that contains no cycles (a cycle being
    a path through the graph that leads back to the starting
    point without traversing the same edge more than once).
    
    A minimum spanning forest of a weighted undirected graph
    is a spanning forest of the graph whose sum of edge weights
    is no larger than any other such spanning forest. A
    minimum spanning forest of a connected weighted undirected
    graph is referred to as a minimum spanning tree.

    Args:
        Required positional:
        n (int): The number of vertices in the graph.
        edges (list of 3-tuples of ints): The weighted
                undirected edges of the graph, with indices
                0 and 1 giving the indices of the vertices
                connected by that edge (0-indexed) and index
                2 giving the weight of the edge (a non-negative
                integer).

    Returns:
    List of 3-tuples of integers (int) giving the weighted
    undirected edges of a minimum spanning forest of the given
    weighted undirected graph, with indices 0 and 1 giving the
    indices of the vertices onnected by that edge (as per the
    vertex indexing of the original graph) and index 2 giving
    the weight of the edge.
    If there are several such minimum spanning forests, this
    returns the first one constructed by the algorithm.
    Note that if the given graph is connected, the edges will
    be the edges of a minimum spanning tree.
    """
    edges = sorted(edges, key=lambda x: x[2])
    res = []
    uf = UnionFind(n)
    for e in edges:
        if uf.connected(e[0], e[1]):
            continue
        uf.union(e[0], e[1])
        res.append(e)
    return res

def minimalNetworkFromFile(
    doc: str="project_euler_problem_data_files/0107_network.txt",
    rel_package_src: bool=True,
) -> int:
    """
    Solution to Project Euler #107

    For a weighted network in the .txt file doc for which every
    vertex is connected, identifies the largest sum of weights
    of any subset of the edges for which the removal of those
    edges would still leave the network connected.

    The file should contain the rows of the matrix in order,
    separated by line breaks ('\\n'), with each entry of the row
    being either a non-negative integer represented in base 10 by
    arabic numerals representing that a single undirected edge
    exists between the corresponding vertices in the network with
    weight equal to that number or a hyphen ('-') denoting that
    no edge exists between those two vertices, with each entry
    separated by a single comma (',') only (i.e. no space before
    or after the comma).

    Args:
        Optional named:
        doc (str): The relative or absolution location of the .txt
                file containing the network in matrix form.
            Default: "project_euler_problem_data_files/0107_network.txt"
        rel_package_src (bool): Whether a relative path given by doc
                is relative to the current directory (False) or
                the package src directory (True).
            Default: True
    
    Returns:
    Integer (int) giving the largest sum of weights of any subset of
    the edges in the network such that the removal of those edges still
    leaves the network connected.

    Outline of rationale:
    This problem is equivalent to finding a minimum spanning tree of
    the network (which is the smallest total weight of a subset of
    edges of a network such that the network remains connected), with
    the result being the difference between the total weight of all edges
    in the network and the total weight of all edges in the minimum
    spanning tree.
    """
    n, edges = loadNetworkFromFile(doc, rel_package_src=rel_package_src)
    mst_edges = kruskallAlgorithm(n, edges)
    return sum(x[2] for x in edges) - sum(x[2] for x in mst_edges)

# Problem 108 & 110
def diophantineReciprocals(min_n_solutions: int=1001) -> int:
    """
    Solution to Project Euler #108 and Project Euler #110 (the latter
    with min_n_solutions=4 * 10 ** 6 + 1)

    Calculates the smallest strictly positive integer n such that
    the equation:
        1 / x + 1 / y = 1 / n
    has at least min_n_solutions distinct solutions for the ordered
    pair of integer (x, y) where:
        0 < x <= y
    
    Args:
        Optional named:
        min_n_solutions (int): The smallest number of solutions to
                the above equation the returned value of n must
                have.
            Default: 1001
    
    Returns:
    Integer (int) giving the smallest strictly positive integer
    n such that the above equation has at least min_n_solutions.
    """
    #since = time.time()
    n_p = 0
    num = 1
    while num < min_n_solutions:
        num *= 3
        n_p += 1
    p_lst = []
    ps = PrimeSPFsieve()
    p_gen = ps.endlessPrimeGenerator()
    for i in range(n_p):
        p_lst.append(next(p_gen))
    curr_best = [float("inf")]
    target = (min_n_solutions << 1) - 1
    #print(p_lst)
    def recur(i: int=0, curr_num: int=1, curr_n_solutions: int=1,\
            mx_count: Union[int, float]=float("inf")) -> None:
        #print(i, curr_num, curr_n_solutions)
        if curr_num >= curr_best[0] or i >= len(p_lst): return
        if curr_n_solutions >= target:
            curr_best[0] = curr_num
            #print(i, curr_num, curr_n_solutions)
            #print(f"curr_best = {curr_best[0]}")
            return
        for j in range(1, min(mx_count, (-((-target) // curr_n_solutions) - 1) >> 1) + 1):
            curr_num *= p_lst[i]
            if curr_num >= curr_best[0]: break
            recur(i + 1, curr_num=curr_num,\
                    curr_n_solutions=curr_n_solutions * ((j << 1) + 1),\
                    mx_count=j)
        return
    recur()
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return curr_best[0]
            

# Problem 109
def dartCheckouts(mx_score: int=99) -> int:
    """
    Solution to Project Euler #109
    For a standard dart board, calculates the sum of the number
    of ways to check out without missing over all scores no greater
    than mx_score where two ways of checking out are distinct
    if and only if the sets of regions hit in the three darts are
    different or the final double is different.
    
    Args:
        Optional named:
        mx_score (int): The largest checkout score included in
                the sum.
            Default: 99
    
    Returns:
    Integer (int) giving the sum of the number of ways to check
    out without missing over all scores no greater than mx_score
    subject to the definition of distinct checkouts given above.
    """
    #since = time.time()
    double_dict = SortedDict()
    for num in range(1, min(20, mx_score >> 1) + 1):
        double_dict[num << 1] = 1
    if 50 <= mx_score: double_dict[50] = 1
    
    score_dict = SortedDict({0: 1})
    for num in range(1, min(20, mx_score) + 1):
        score_dict[num] = score_dict.get(num, 0) + 1
        dbl = num << 1
        if dbl > mx_score: continue
        score_dict[dbl] = score_dict.get(dbl, 0) + 1
        trpl = num * 3
        if trpl > mx_score: continue
        score_dict[trpl] = score_dict.get(trpl, 0) + 1
    if 25 <= mx_score:
        score_dict[25] = score_dict.get(25, 0) + 1
        if 50 <= mx_score:
            score_dict[50] = score_dict.get(50, 0) + 1
    
    prev = SortedDict(score_dict)
    curr = SortedDict()
    for num1, f1 in prev.items():
        for num2, f2 in score_dict.items():
            sm = num1 + num2
            if sm > mx_score: break
            elif num2 == num1:
                curr[sm] = curr.get(sm, 0) + ((f1 * (f1 + 1)) >> 1)
                break
            curr[sm] = curr.get(sm, 0) + f1 * f2
    prev = curr
    curr = SortedDict()
    for num1, f1 in prev.items():
        for num2, f2 in double_dict.items():
            sm = num1 + num2
            if sm > mx_score: break
            curr[sm] = curr.get(sm, 0) + f1 * f2
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return sum(curr.values())
    
# Problem 110- see Problem 108


# Problem 111
def permutationsWithRepeats(
    objs: List[Any],
    freqs: List[int],
) -> Generator[Tuple[Any], None, None]:
    """
    Generator yielding every permutation of the distinct
    objects in the list objs, each with a number of
    occurrences equal to the corresponding entry in the
    list freqs.

    Args:
        Required postional:
        objs (list of objects): List of distinct objects for
                which the permutations are to be found.
        freqs (list of ints): List of strictly positive integers
                with the same length as that of objs, with each
                entry giving the number of occurrences the
                corresponding entry in objs should have in each
                permutation yielded.

    Yields:
    Each permutation of the objects in objs, each with a
    frequency in each permutation according to the corresponding
    entry in freqs.
    The permutations are yielded in lexicographically increasing
    order, where objects earlier in the list objs are considered
    to be smaller than every object later in the list.
    """
    n = len(objs)
    m = sum(freqs)
    remain = SortedSet(range(n))
    curr = [None] * m
    def recur(i: int=0) -> Generator[Tuple[Any], None, None]:
        if i == m:
            yield tuple(curr)
            return
        for j in list(remain):
            freqs[j] -= 1
            if not freqs[j]: remain.remove(j)
            curr[i] = objs[j]
            yield from recur(i + 1)
            if not freqs[j]: remain.add(j)
            freqs[j] += 1
        return
    
    yield from recur(i=0)
    return

# Review- consider trying to yield the integers in strictly
# increasing order
def digitCountIntegerGenerator(
    n_dig: int,
    rpt_dig: int,
    n_rpt: int,
    base: int=10,
) -> Generator[int, None, None]:
    """
    Generator yielding all strictly positive integers which when
    represented in the chosen base (without leading zeros) contain
    exactly n_dig digits, exactly n_rpt of which are the digit
    rpt_dig.

    Args:
        Required positional:
        n_dig (int): Strictly positive integer giving the number
                of digits each yielded integer should have when
                represented in the chosen base (without leading
                zeros).
        rpt_dig (int): Integer between 0 and (base - 1) inclusive
                specifying the value of the digit in the chosen
                base that should be present exactly n_rpt times
                in the representation of each yielded integer
                in that base (without leading zeros).
        n_rpt (int): Non-negative integer giving the exact number
                of times rpt_dig should appear in the representataion
                of each number yielded in the chosen base (without
                leading zeros).

        Optional named:
        base (int): The integer strictly greater than 1 giving
                the base in which each number is to be expressed
                when assessing whether it has the correct number
                of digits and the correct number of the digit
                rpt_dig when assessing whether it should be
                yielded.
            Default: 10

    Yields:
    Integers (int), collectively representing all strictly positive
    integers which when represented in the chosen base without
    leading zeros contain exactly n_dig digits, exactly n_rpt of
    which have the value rpt_dig.
    The integers are not yielded in a specific order.
    """
    if n_dig < n_rpt: return
    digs = [x for x in range(base) if x != rpt_dig]
    if n_rpt:
        objs = [rpt_dig]
        freqs = [n_rpt]
    else:
        objs = []
        freqs = []
    
    def recur(i: int, n_remain: int) -> Generator[int, None, None]:
        if not n_remain or i == base - 2:
            if n_remain:
                objs.append(digs[-1])
                freqs.append(n_remain)
            #print(objs, freqs)
            for dig_tup in permutationsWithRepeats(objs, freqs):
                #print(dig_tup)
                if not dig_tup[0]: continue
                ans = 0
                for d in dig_tup:
                    ans = ans * base + d
                yield ans
            if n_remain:
                objs.pop()
                freqs.pop()
            return
        yield from recur(i + 1, n_remain)
        objs.append(digs[i])
        freqs.append(0)
        for j in range(1, n_remain + 1):
            freqs[-1] += 1
            yield from recur(i + 1, n_remain - j)
        freqs.pop()
        objs.pop()
        return
    
    yield from recur(0, n_dig - n_rpt)
    return

def mostRepeatDigitPrimes(
    n_dig: int,
    rpt_dig: int,
    base: int=10,
    ps: Optional[SimplePrimeSieve]=None,
) -> Tuple[Union[List[int], int]]:
    """
    Calculates all prime numbers which represented in the
    chosen base without leading zeros contain exactly n_dig
    repeat digits, and contain at least as many digits with
    value rpt_dig as any other such prime.

    Args:
        Required positional:
        n_dig (int): The number of digits (without leading zeros)
                that each prime number considered should have in
                its representation in the chosen base.
        rpt_dig (int): Integer between 0 and (base - 1) inclusive
                giving the value of the digit for which the
                returned primes should have at least as many
                occurrences in their representations in the chosen
                base (without leading zeros) as any other such
                prime.

        Optional named:
        base (int): The integer strictly greater than 1 giving
                the base in which each number is to be expressed
                when assessing whether it has the correct number
                of digits and sufficient number of the digit with
                value rpt_dig to be included in the output.
            Default: 10
        ps (SimplePrimeSieve or None): If specified, the prime
                sieve object used to assess whether a given number
                is prime. If not specified, a new SimplePrimeSieve
                object will be constructed for this purpose.
                This option is included to prevent the need for
                repeated construction of prime sieve objects
                across multiple function calls, potentially
                improving efficiency.
            Default: None

    Returns:
    2-tuple whose index 0 contains a list of integers (int)
    representing all the prime numbers which, when represented in
    the chosen base (without leading zeros) contain exactly n_dig
    digits and at least as many occurrences of the digit with
    value rpt_dig as any other such prime (and so necessarily
    the same number of such occurrences as all other elements
    of the list), and whose index 1 contains an integer (int)
    representing the number of occurrences of the digit with value
    rpt_dig of those primes given in index 0 when represented
    in the chosen base (without leading zeros).
    """
    
    if ps is None: ps = SimplePrimeSieve()
    def primeCheck(num: int) -> int:
        return ps.millerRabinPrimalityTestWithKnownBounds(num)[0]
    
    for n_rpt in reversed(range(n_dig + 1)):
        p_lst = []
        for num in digitCountIntegerGenerator(n_dig, rpt_dig,\
                n_rpt, base=base):
            if primeCheck(num):
                p_lst.append(num)
        if p_lst: break
    return (sorted(p_lst), n_rpt)

def primesWithRuns(
    n_dig: int=10,
    base: int=10,
    ps: Optional[SimplePrimeSieve]=None,
) -> int:
    """
    Solution for Project Euler #111

    Calculates the sum of the function S(n_dig, d, base) over integers
    d between 0 and (base - 1) inclusive.

    For strictly positive integers m and b and integer d between
    0 and (b - 1) inclusive, S(m, d, b) is the sum of all
    prime numbers which, when represented in base b without leading
    zeros contain exactly m digits and at least as many occurrences
    of the digit with value d as any other such prime.

    Args:
        Optional named:
        n_dig (int): Strictly positive integer giving the value
                of m in all of the S(m, d, b) in the described
                sum, representing the number of digits of all
                primes considered when expressed in the chosen
                base (without leading zeros).
            Default: 10
        base (int): The integer strictly greater than 1 giving
                the value of b in oll of the S(m, d, b) in the
                described sum, representing the base in which each
                number is to be expressed when assessing whether
                it has the correct number of digits and sufficient
                number of the digit with value rpt_dig to be
                included in the total when calculating each value
                of S(m, d, b).
            Default: 10
        ps (SimplePrimeSieve or None): If specified, the prime
                sieve object used to assess whether a given number
                is prime. If not specified, a new SimplePrimeSieve
                object will be constructed for this purpose.
                This option is included to prevent the need for
                repeated construction of prime sieve objects
                across multiple function calls, potentially
                improving efficiency.
            Default: None
    
    Returns:
    Integer (int) giving the value of the function S(n_dig, d, base)
    (with the function S as defined above) over integers d between 0
    and (base - 1) inclusive.
    """
    #since = time.time()
    if ps is None: ps = SimplePrimeSieve()
    res = sum(
        sum(mostRepeatDigitPrimes(n_dig, d, base=10, ps=ps)[0])
        for d in range(base)
    )
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res   

# Problem 112
def isBouncy(num: int, base: int=10) -> bool:
    """
    Assesses whether positive integer num is bouncy for the
    chosen base.

    A strictly positive integer is bouncy for a given base if
    and only if the sequence of digits of the expression of
    that integer in the given base (without leading zeros) is
    not weakly increasing or weakly decreasing (i.e. there
    exists at least one pair of consecutive digits for which
    the first is strictly greater than the second and another
    pair of consecutive digits for which the second is strictly
    greater than the first).

    Args:
        Required postional:
        num (int): The strictly positive integer for which its
                status as bouncy for the chosen base is to be
                assessed.

        Optional named:
        base (int): The integer strictly greater than 1 giving
                the base for which num is to be assessed as bouncy
                or not bouncy.
            Default: 10
    
    Returns:
    Boolean (bool) specifying whether num is bouncy (True) or
    not bouncy (False) for the chosen base.
    """
    incr = True
    decr = True
    num, curr = divmod(num, base)
    while num:
        prev = curr
        num, curr = divmod(num, base)
        if curr < prev:
            if not decr: return True
            incr = False
        elif curr > prev:
            if not incr: return True
            decr = False
    return False

def bouncyProportions(
    prop_numer: int=99,
    prop_denom: int=100,
    base: int=10,
) -> int:
    """
    Solution to Project Euler #112

    Calculates the smallest strictly positive integer for which the
    proportion of strictly positive integers no greater than that
    integer that are bouncy for the chosen base is exactly equal
    to the fraction (prop_numer / prop_denom).

    A positive integer is bouncy for a given base if and only
    if the sequence of digits of the expression of that integer
    in the given base (without leading zeros) is not weakly
    increasing or weakly decreasing (i.e. there exists at least
    one pair of consecutive digits for which the first is
    strictly greater than the second and another pair of
    consecutive digits for which the second is strictly greater
    than the first).

    Args:
        Optional named:
        prop_numer (int): Strictly positive integer giving the
                numerator of the fraction for the target proportion
                of bouncy numbers for the chosen base.
            Default: 99
        prop_denom (int): Strictly positive integer giving the
                denominator of the fraction for the target proportion
                of bouncy numbers for the chosen base.
            Default: 99
        base (int): The integer strictly greater than 1 giving
                the base for which the strictly positive integers are
                to be assessed as bouncy or not bouncy.
            Default: 10

    Returns:
    Integer (int) giving the smallest strictly positive integer for
    which the proportion of strictly positive integers no greater than
    this integer that are bouncy for the chosen base is exactly equal
    to the fraction (prop_numer / prop_denom).

    Outline of rationale:
    We simply iterate over the strictly positive integers, assessing
    whether each is bouncy or not for the chosen base. The number of
    times the proportion is checked is reduced by noting that the
    proportion can only be exact when the number is a multiple of the
    denominator of the proportion when expressed as a fraction in lowest
    terms. 
    """
    #since = time.time()
    bouncy_cnt = 0
    g = gcd(prop_numer, prop_denom)
    prop_numer //= g
    prop_denom //= g
    rng = (1, prop_denom + 1)
    while True:
        bouncy_cnt += sum(isBouncy(x, base=base) for x in range(*rng))
        if bouncy_cnt * prop_denom == (rng[1] - 1) * prop_numer:
            break
        rng = (rng[1], rng[1] + prop_denom)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return rng[1] - 1

# Problem 113
class NonBouncyCounter(object):
    """
    Class used to calculate the number of strictly positive integers
    that are not bouncy for a given base that contain up to a given
    number of digits when represented in that base.

    A positive integer is bouncy for a given base if and only
    if the sequence of digits of the expression of that integer
    in the given base (without leading zeros) is not weakly
    increasing or weakly decreasing (i.e. there exists at least
    one pair of consecutive digits for which the first is
    strictly greater than the second and another pair of
    consecutive digits for which the second is strictly greater
    than the first).

    Initialisation args:
        Optional named:
        n_dig (int): Non-negative integer giving the number of digits
                for which the counts should be precomputed on
                initialisation.
            Default: 1
        base (int): The integer strictly greater than 1 giving
                the base for which the strictly positive integers are
                to be assessed as bouncy or not bouncy. This becomes
                the attribute base.
            Default: 10
    
    Attributes:
        Attributes that should not be changed during the lifetime
        of the instance:
        base (int): The integer strictly greater than 1 giving
                the base for which the strictly positive integers are
                to be assessed as bouncy or not bouncy. This is
                specified on initialisation by the argument base.
    
    Function call:
        Provides the number of strictly positive integers that are
        not bouncy for the base given by the attribute base that when
        represented in that base have no more that n_dig digits.

        Args:
            Required postional:
            n_dig (int): The maximum number of digits when represented
                    in the chosen base of all strictly positive integers
                    to be considered for inclusion in the returned
                    count.

        Returns:
        Integer (int) giving the number of strictly positive integers
        that contain no more than n_dig digits when represented in the
        base given by the attribute base (without leading zeros) and
        are not bouncy for that base.
    """
    def __init__(self, n_dig: int=1, base: int=10):
        self._base = base
        self._memo = [[[0, 0] for _ in range(base)],\
                [[1, 1] for _ in range(base)]]
        self._extendMemo(n_dig)
    
    @property
    def base(self) -> int:
        return self._base
    
    def _extendMemo(self, n_dig: int) -> None:
        for i in range(len(self._memo), n_dig + 1):
            self._memo.append([[0, 0] for _ in range(self.base)])
            curr = 0
            for j, pair in enumerate(self._memo[i - 1]):
                curr += pair[0]
                self._memo[i][j][0] = curr
            curr = 0
            for j in reversed(range(len(self._memo[i - 1]))):
                curr += self._memo[i - 1][j][1]
                self._memo[i][j][1] = curr
        return
    
    def __call__(self, n_dig: int) -> int:
        if not n_dig: return 0
        self._extendMemo(n_dig)
        # Subtract self.base - 1 since numbers with all one digit
        # are double counted
        return sum(sum(x) for x in self._memo[n_dig][1:]) - (self.base - 1)

def nonBouncyNumbers(mx_n_dig: int=100, base: int=10) -> int:
    """
    Solution to Project Euler #113

    Calculates the number of strictly positive integers that are
    not bouncy for the given base that contain up to a given
    number of digits when represented in that base.

    A positive integer is bouncy for a given base if and only
    if the sequence of digits of the expression of that integer
    in the given base (without leading zeros) is not weakly
    increasing or weakly decreasing (i.e. there exists at least
    one pair of consecutive digits for which the first is
    strictly greater than the second and another pair of
    consecutive digits for which the second is strictly greater
    than the first).

    Args:
        Optional named:
        mx_n_dig (int): Non-negative integer giving the largest
                number of digits any strictly positive integer
                included in the returned count should have
                when represented in the chosen base (without leading
                zeros).
            Default: 10
        base (int): The integer strictly greater than 1 giving
                the base for which the strictly positive integers are
                to be assessed as bouncy or not bouncy.
            Default: 10

    Returns:
    Integer (int) giving the number of strictly positive integers
    that contain no more than n_dig digits when represented in the
    chosen base (without leading zeros) and are not bouncy for that
    base.
    """
    #since = time.time()
    nbc = NonBouncyCounter(n_dig=mx_n_dig, base=base)
    res = sum(nbc(i) for i in range(1, mx_n_dig + 1))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 114
def countingBlockCombinations(tot_len: int=50, min_large_len: int=3) -> int:
    """
    Solution to Project Euler #114

    Calculates the number of distinct ordered partitionings of
    a line of length tot_len such that all the parts so produced
    are of integer length, with that integer being either 1 or
    strictly greater than min_large_len and for every pair of
    adjacent parts, at least one is length 1 (i.e. no two parts
    with length greater than 1 are adjacent).

    Args:
        Optional named:
        tot_len (int): Strictly positive integer giving the length
                of the line for which the number of ordered
                partitionings described above are to be found.
            Default: 50
        min_large_len (int): Smallest possible value for the length
                of parts produced by the partitioning that do not
                have length 1.
            Default: 3
    
    Returns:
    Integer (int) giving the number of distinct ordered partitionings
    exist for a line of length tot_len subject to the above described
    constraints.
    """
    #since = time.time()
    qu = deque([1] * (min_large_len + 1))
    tot = 0
    for _ in range(min_large_len + 1, tot_len + 2):
        tot += qu.popleft()
        qu.append(qu[-1] + tot)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return qu[-1]

# Problem 115
def countingBlockCombinationsII(min_large_len: int=50, target_count: int=10 ** 6 + 1) -> int:
    """
    Solution to Project Euler #115

    Calculates the smallest non-negative integer such that the
    number of distinct ordered partitionings of line of that
    length such that:
     1) all the parts so produced are of integer length, with
        that integer being either 1 or strictly greater than
        min_large_len
     2) for every pair of adjacent parts, at least one is length
        1 (i.e. no two parts with length greater than 1 are
        adjacent).
    is no less than target_count

    Args:
        Optional named:
        min_large_len (int): Smallest possible value for the length
                of parts produced by the partitioning that do not
                have length 1.
            Default: 50
        target_count (int): The minimum number of distinct ordered
                partitionings of a line of the returned length that
                should be possible, with all shorter lines having
                fewer such possible ordered partitions.
            
    Returns:
    Integer (int) giving smallest non-negative integer satisfying the
    properties given above.
    """
    # Review- documentation for clarity
    #since = time.time()
    if target_count <= 1: return 0
    qu = deque([1] * (min_large_len + 1))
    n = min_large_len - 1
    tot = 0
    while qu[-1] < target_count:
        tot += qu.popleft()
        qu.append(qu[-1] + tot)
        n += 1
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return n

# Problem 116
def redGreenOrBlueTiles(
    tot_len: int=50,
    min_large_len: int=2,
    max_large_len: int=4,
) -> int:
    """
    Solution to Project Euler #116

    Calculates the number of distinct ordered partitionings of
    a line of length tot_len such that all the parts so produced
    are of integer length, each of which are either length 1 or
    a single chosen integer length between min_large_len and
    max_large_len (so all of the parts that are not length 1 in
    a given partitioning are the same length), and not all of
    which are length 1.

    Args:
        Optional named:
        tot_len (int): Strictly positive integer giving the length
                of the line for which the number of ordered
                partitionings described above are to be found.
            Default: 50
        min_large_len (int): Smallest possible value for the length
                of parts produced by the partitioning that do not
                have length 1.
            Default: 2
        max_large_len (int): Largest possible value for the length
                of parts produced by the partitioning that do not
                have length 1.
            Default: 4
    
    Returns:
    Integer (int) giving the number of distinct ordered partitionings
    exist for a line of length tot_len subject to the above described
    constraints.
    """
    #since = time.time()
    res = 0
    for large_len in range(min_large_len, max_large_len + 1):
        qu = deque([1] * (large_len))
        for _ in range(large_len, tot_len + 1):
            qu.append(qu[-1] + qu.popleft())
        res += qu[-1]
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res - (max_large_len - min_large_len + 1)

# Problem 117
def redGreenAndBlueTiles(
    tot_len: int=50,
    min_large_len: int=2,
    max_large_len: int=4
) -> int:
    """
    Solution to Project Euler #117

    Calculates the number of distinct ordered partitionings of
    a line of length tot_len such that all the parts so produced
    are of integer length, each of which are either length 1 or
    any integer length between min_large_len and max_large_len.

    Args:
        Optional named:
        tot_len (int): Strictly positive integer giving the length
                of the line for which the number of ordered
                partitionings described above are to be found.
            Default: 50
        min_large_len (int): Smallest possible value for the length
                of parts produced by the partitioning that do not
                have length 1.
            Default: 2
        max_large_len (int): Largest possible value for the length
                of parts produced by the partitioning that do not
                have length 1.
            Default: 4
    
    Returns:
    Integer (int) giving the number of distinct ordered partitionings
    exist for a line of length tot_len subject to the above described
    constraints.
    """
    #since = time.time()
    qu1 = deque([1] * (min_large_len))
    tot = 0
    qu2 = deque()
    for _ in range(min_large_len, max_large_len + 1):
        qu2.append(qu1.popleft())
        tot += qu2[-1]
        qu1.append(qu1[-1] + tot)
    for _ in range(max_large_len + 1, tot_len + 1):
        qu2.append(qu1.popleft())
        tot += qu2[-1]
        tot -= qu2.popleft()
        qu1.append(qu1[-1] + tot)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return qu1[-1]

# Problem 118- try to make faster
def pandigitalPrimeSets(base: int=10) -> int:
    """
    Solution to Project Euler #118

    Calculates the number of sets of prime numbers for which the
    digits from 1 to (base - 1) appear as a digit exactly once in
    exactly one of the elements' representation in the chosen base
    (without leading zeros).

    Args:
        Optional named:
        base (int): The integer strictly greater than 1 giving
                the base for which the prime numbers in a given
                set are to be represented when assessing whether
                every digit from 1 to (base - 1) appears exactly
                once in exactly one of the primes' representation
                in that base.
            Default: 10
    
    Returns:
    Integer (int) giving the number of distinct set of primes that
    satisfy the above described property.
    """
    #since = time.time()
    ps = SimplePrimeSieve()#isqrt(base ** base))
    def primeCheck(num: int) -> int:
        return ps.millerRabinPrimalityTestWithKnownBounds(num)[0]
    res = [0]
    def recur(nums: Tuple[int], i: int=0, prev: int=0, prev_n_dig: int=0) -> None:
        n = len(nums)
        #print(i, prev, prev_n_dig)
        num = 0
        for i2 in range(i, i + prev_n_dig):
            num = num * base + nums[i2]
        i2 = i + prev_n_dig
        if num <= prev:
            if i2 == n: return
            num = num * base + nums[i2]
            i2 += 1
        if n - i2 >= i2 - i:
            #if i2 > i + 1 and nums[i2 - 1] not in disallowed_last and\
            #        ps.isPrime(num):
            if i2 > i and primeCheck(num):
                #print(num)
                recur(nums, i=i2, prev=num, prev_n_dig=i2 - i)
            i3 = i + ((n - i) >> 1)
            for i2 in range(i2, i3):
                num = num * base + nums[i2]
                #if nums[i2] not in disallowed_last and ps.isPrime(num):
                if ps.isPrime(num):
                    #print(num)
                    recur(nums, i=i2 + 1, prev=num,\
                            prev_n_dig=i2 - i + 1)
        else: i3 = i2
        for i3 in range(i3, n):
            num = num * base + nums[i3]
        #if num > prev and ps.isPrime(num): print(num)
        res[0] += (num > prev) and primeCheck(num)
        return
    
    disallowed_last = {x for x in range(1, base) if gcd(base, x) != 1}
    for perm in itertools.permutations(range(1, base)):
        if perm[-1] in disallowed_last: continue
        recur(perm)
        #print(perm, res)
    res[0] += all(ps.isPrime(x) for x in range(1, base))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res[0]

# Problem 119
def isPower(num: int, base: int) -> int:
    """
    Identifies whether a strictly positive integer num is
    a power of the strictly positive integer base (i.e.
    num = base^n for some non-negative integer n), and if
    so what the exponent is (i.e. the integer n in the
    previous formula).

    Args:
        Required positional:
        num (int): Strictly positive integer whose status
                as an integer power of the integer base.
        base (int): Strictly positive integer for which the
                status of num as an integer power of this
                number is being assessed.
    
    Returns:
    Integer (int) which is the integer power to which base is
    to be taken to get num if such a number exists, and if
    not -1.
    """
    num2 = base
    res = 1
    while num2 < num:
        num2 *= base
        res += 1
    return res if num2 == num else -1

def digitCountAndDigitSum(
    num: int,
    base: int=10,
) -> Tuple[int, int]:
    """
    Calculates the number and sum of digits of a strictly
    positive integer when expressed terms of a given base.

    Args:
        Required positional:
        num (int): The strictly positive integer whose number
                and sum of digits when expressed in the
                chosen base is to be calculated.

        Optional named:
        base (int): The integer strictly exceeding 1 giving
                the base in which num is to be expressed when
                assessing the digit number and sum.
            Default: 10

    Returns:
    2-tuple whose index 0 contains the number of digits (without
    leading 0s) of num, and whose index 1 contains the sum of
    digits of num, both when num is expressed in the chosen base.

    Examples:
        >>> digitCountAndDigitSum(5496, base=10)
        (4, 24)

        This signifies that 5496 when expressed in base 10 (i.e.
        the standard base) has 4 digits which sum to 24
        (5 + 4 + 9 + 6).

        >>> digitCountAndDigitSum(6, base=2)
        (3, 2)

        This signifies that 6 when expressed in base 2 (binary,
        in which 6 is expressed as 101) has 3 digits which sum
        to 2 (1 + 0 + 1).
    """
    res = [0, 0]
    while num:
        num, r = divmod(num, base)
        res[0] += 1
        res[1] += r
    return tuple(res)

def powerDigitSumEqualNumDigitCountUpperBound(
    exp: int,
    base: int=10,
) -> int:
    """
    For a given exponent and base, finds an upper bound for the
    number of digits a strictly positive integer can have in
    that base and it be possible for the sum over the chosen exponent
    of each of its digits in the chosen base to be equal to the
    integer itself.

    Args:
        Required positional:
        exp (int): Non-negative positive integer giving the exponent
                to which each of the digits in the chosen base is
                taken in the described sum.

        Optional named:
        base (int): Integer strictly greater than 1 giving the base
                in which the integers are expressed when taking
                the exponentiated digit sums as described.
    
    Returns:
    Strictly positive integer (int) giving an upper bound on the
    number of digits in the chosen base an integer may have, and
    for the described exponentiated digit sum of that integer
    to be equal to the integer itself. That is to say, there
    may exist integers with this property with this number of
    digits or fewer in the chosen base, but there cannot be
    any with more.
    """
    def check(n_dig: int) -> bool:
        num = ((base - 1) * n_dig) ** exp
        return num >= base ** (n_dig - 1)

    mult = 10
    prev = 0
    curr = 1
    while check(curr):
        prev = curr
        curr *= mult
    #print(prev, curr)
    lft, rgt = prev, curr - 1
    while lft < rgt:
        mid = lft - ((lft - rgt) >> 1)
        if check(mid): lft = mid
        else: rgt = mid - 1
    return lft
    
    """
    n_dig = 0
    curr = 0
    comp = 1
    while True:
        curr += base - 1
        if curr ** exp < comp: break
        n_dig += 1
        comp *= base
    #print(n_dig)
    return n_dig
    """

def digitPowerSumSequence(
    n_terms: int,
    base: int=10,
) -> List[Tuple[int, int, int]]:
    """
    Calculates the first n_terms terms in the integer sequence whose
    terms are the integers in strictly increasing order which
    when represented in the chosen base contain at least two digits and
    there exists a non-negative integer power such that the sum of its
    digits in that representation taken to that power equals the integer
    itself.

    Args:
        Required positional:
        n_terms (int): Non-negative integer giving the number of terms
                of the described sequence to return.

        Optional named:
        base (int): Integer strictly greater than 1 giving the base
                in which the integers are expressed when assessing whether
                the sum of their digits taken to some non-negative power
                equals the integer itself.
            Default: 10
    
    Returns:
    List of 2-tuples of integers, representing the first n_terms terms of
    the described sequence in order, with index 0 containing the integer,
    index 1 containing the sum of its digits when represented in the
    chosen base and index 2 containing the smallest non-negative integer
    power for which the sum of digits, when taken to that power, equals
    the integer.
    The list has length n_terms.
    """
    res = []
    if n_terms <= 0: return []
    heap = [(2 ** 2, 2, 2)]
    n_dig_limit_heap = [(powerDigitSumEqualNumDigitCountUpperBound(2, base=base), 2)]
    mx_b = -float("inf")
    mx_a = 2
    curr_n_dig = 0
    while True:
        num, a, b = heapq.heappop(heap)
        n_dig, dig_sum = digitCountAndDigitSum(num, base=base)
        if dig_sum == a:
            res.append((num, a, b))
            #print(len(res), num, a, b)
            #print(n_dig_limit_heap[0])
            if len(res) == n_terms: break
        if n_dig > curr_n_dig:
            curr_n_dig = n_dig
            #print(n_dig_limit_heap)
            while n_dig_limit_heap and n_dig_limit_heap[0][0] < n_dig:
                heapq.heappop(n_dig_limit_heap)
                #print(f"new min = {n_dig_limit_heap[0]}")
        heapq.heappush(heap, (num * a, a, b + 1))
        if b + 1 > mx_b:
            mx_b = b + 1
            heapq.heappush(
                n_dig_limit_heap,
                (powerDigitSumEqualNumDigitCountUpperBound(mx_b, base=base), mx_b),
            )
            #print(n_dig_limit_heap)
        if a == mx_a:
            mx_a += 1
            b2 = n_dig_limit_heap[0][1]
            heapq.heappush(heap, (mx_a ** b2, mx_a, b2))
    return res

def digitPowerSum(n: int=30, base: int=10) -> int:
    """
    Solution to Project Euler #119

    Finds the n:th smallest integer no less than base, such
    that when expressed in that base, there exists a non-negative
    integer exponent for which the sum over the digits taken
    to the power of that exponent is equal to the integer
    itself.

    Args:
        Optional named:
        n (int): Strictly positive integer specifying the term
                in the sequence of the integers with the described
                property in ascending order, starting at 1 is to
                be found.
            Default: 30
        base (int): Strictly positive integer specifying the
                base in which the integers should be expressed
                when assessing the described property.
            Default: 10
    
    Returns:
    Integer (int), giving the n:th smallest integer no less than
    base such that when expressed in that base, there exists a
    non-negative integer exponent for which the sum over the
    digits taken to the power of that exponent is equal to the
    integer itself.
    """
    #since = time.time()
    seq = digitPowerSumSequence(n, base=base)
    #print(seq)
    res = seq[-1][0]
    #print(seq)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 120
def squareRemainders(a_min: int=3, a_max: int=1000) -> int:
    """
    Solution to Project Euler #120

    For given non-negative integers a and n, consider the maximum
    value of the remainder of (a - 1)^n + (a + 1)^n when divided
    by a^2.
    Keeping a the fixed, consider the value of this remainder for
    all possible values of n and choose the largest. This function
    calculates the sum of these largest values for all values of
    a between a_min and a_max inclusive.
    
    Args:
        Optional named:
        a_min (int): Non-negative integer giving the smallest value
                of a considered in the sum.
            Default: 3
        a_max (int): Non-negative integer giving the largest value
                of a considered in the sum.
    
    Returns:
    Non-negative integer giving the calculated value of the sum
    described above.

    Outline of rationale:
    
    In the binomial expansion of (a + 1)^n and (a - 1)^n, all
    terms except the constant term and linear term are divisible
    by a^2. As such,
        (a + 1)^n = a * n + 1 (mod a^2)
        (a - 1)^n = (-1)^n * (1 - a * n) (mod a^2)
    Therefore, for even n:
        (a - 1)^n + (a + 1)^n = 2 (mod a^2)
    and for odd n:
        (a - 1)^n + (a + 1)^n = 2 * a * n (mod a^2)
    Thus, for even n the maximum value modulo a^2 is 2 mod a^2,
    and for odd n the maximum value modulo a^2 is when n
    is the largest odd number strictly smaller than an integer
    multiple of half of a modulo a^2. By considering the
    different remainders of a modulo 4, it can be found that
    the maximising value of n modulo a^2 is:
        floor((a - 1) / 2)
    Thus, for given a, the maximum value of (a - 1)^n + (a + 1)^n
    modulo a^2 for non-negative integers n is:
        max(2 (mod a^2), 2 * a * floor((a - 1) / 2))
    """
    res = 0
    for a in range(a_min, a_max + 1):
        md = a ** 2
        res += max(2 % md, (2 * ((a - 1) >> 1) * a) % md)
    return res

# Problem 121
def diskGameBlueDiskProbability(n_turns: int, min_n_blue_disks: int) -> Tuple[int, int]:
    """
    Consider a game consisting of a bag and red and blue disks.
    Initially, the bag contains one red and one blue disk. A
    turn of the game consists of randomly choosing a disk from
    the bag such that the probability fo drawing each individual
    disk is the same. After each turn the drawn disk is replaced
    and an additional red disk is placed in the bag.

    This function calculates the probability that after n_turns
    turns of this game the total number of times the blue disk
    is drawn is at least min_n_blue_disks as a fraction.

    Args:
        Required positional:
        n_turns (int): The number of turns in the game considered.
        min_n_blue_disks (int): The number of blue disks for which
                the probability of drawing at least this many in
                the number of turns is to be calculated.
    
    Returns:
    2-tuple giving the probability that in a run of the described
    game with n_turns turns a total of at least min_n_blue_disks are
    drawn over the course of the game, expressed as a fraction
    (numerator, denominator).
    """
    if min_n_blue_disks > n_turns: return (0, 1)
    row = [(1, 1)]
    n_red = 1
    n_tot = 2
    for i in range(n_turns):
        prev = row
        row = [(1, 1)]
        for i in range(1, min(len(prev), min_n_blue_disks) + 1):
            row.append(
                addFractions(
                    multiplyFractions(prev[i - 1], (n_tot - n_red, n_tot)),
                    multiplyFractions(prev[i], (n_red, n_tot)) if i < len(prev) else (0, 1)
                )
            )
        n_red += 1
        n_tot += 1
    #print(row[-1])
    return row[-1]

def diskGameMaximumNonLossPayout(n_turns: int=15) -> int:
    """
    Solution to Project Euler #121

    Consider a game consisting of a bag and red and blue disks.
    Initially, the bag contains one red and one blue disk. A
    turn of the game consists of randomly choosing a disk from
    the bag such that the probability fo drawing each individual
    disk is the same. After each turn the drawn disk is replaced
    and an additional red disk is placed in the bag.

    The player wins if the number of blue disks drawn over the
    course of the game strictly exceeds the number of red disks
    drawn.

    Given a wager of £1 that the player will win a game consisting
    of n_turns turns, this function calculates the whole pound
    maximum payout such that as the number of attempts approaches
    infinity, the organisation running the game should not expect
    to make a net loss (with the payout including the player's
    initial wager).

    Args:
        Required positional:
        n_turns (int): The number of turns in the game considered.

    Returns:
    Integer (int) giving the whole pound maximum payout such that
    the organisation should not expect to make a net loss in the
    long term repeated running of the described game with n_turns
    turns.
    """
    player_win_n_blue_disks = (n_turns >> 1) + 1
    p_player_win = diskGameBlueDiskProbability(n_turns, player_win_n_blue_disks)
    return math.floor(p_player_win[1] / p_player_win[0])

# Problem 122
def efficientExponentiation(sum_min: int=1, sum_max: int=200, method: Optional[str]="exact") -> float:
    """
    Solution to Project Euler #122

    Calculates the sum over the least number of multiplications
    required to achieve each of the powers individually from
    sum_min to sum_max using a specified method.

    Args:
        Optional named:
        sum_min (int): Strictly positive integer giving the smallest
                exponent considered
            Default: 1
        sum_max (int): Strictly positive integer giving the largest
                exponent considered
            Default: 200
        method (string or None): Specifies the method:
                "exact": calculates exactly, in a way that is
                        guaranteed to give the correct answer
                        for any sum_min and sum_max. This is an
                        exponential time algorithm, so can be very
                        slow for larger values of sum_max. Specifying
                        the method as None defaults to this method
                "Brauer": Uses the Brauer method, which restricts
                        the search space, giving faster evaluation but
                        not guaranteeing that the result found is
                        optimum. Gives the optimum number for all
                        exponents less than 12509
                "approx": A method that further restricts the search
                        space, giving still faster evaluation but
                        again not guaranteeing that the result found
                        is optimum. Gives the optimum number for all
                        exponents less than 77.
                "binary": Uses the binary method, where the path is
                        constructed on exponents that are powers of
                        2. This is the fastest but least accurate
                        method (as it can be calculated directly from
                        the binary expression for the exponent). Gives
                        the optimum number for all exponents less than
                        14.
            Default: "exact"

    Returns:
    Integer giving the sum over the least number of multiplications
    required to achieve each of the powers individually from
    sum_min to sum_max for the chosen method, with this being guranteed
    to be the optimum if the method "exact" is chosen.
    """
    #since = time.time()
    if method is None:
        method = "exact"
    
    addition_chain_calculator = AdditionChainCalculator()
    if method == "approx":
        func = addition_chain_calculator.shortestAddPathApprox
    elif method == "Brauer":
        func = addition_chain_calculator.shortestAddPathBrauer
    elif method == "exact":
        func = addition_chain_calculator.shortestAddPathExact
    elif method == "binary":
        func = addition_chain_calculator.shortestAddPathBinary
    
    res = sum(len(func(i)) - 1 for i in range(sum_min, sum_max + 1))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 123
def calculateSquareRemainder(num: int, exp: int) -> int:
    """
    For the integer num and the non-negative integer exp,
    calculates the remainder when:
        (num - 1)^exp + (num + 1)^exp
    is divided by num^2.

    Args:
        Required positional:
        num (int): The integer num used in the above expression.
        exp (int): The non-negative integer exp in the above
                expression.
    
    Returns:
    Integer between 0 and (num^2 - 1) inclusive giving the
    remainder when:
        (num - 1)^exp + (num + 1)^exp
    is divided by num^2.
    """
    md = num ** 2
    return (pow(num - 1, exp, md) + pow(num + 1, exp, md)) % md

def primeSquareRemainders(target_remainder: int=10 ** 10 + 1):
    """
    Solution to Project Euler #123

    Finds the smallest number n such that if p_n is the n:th prime
    (where p_1 = 2, p_2 = 3, ...) then the remainder when:
        (p_n - 1)^n + (p_n - 1)^n
    is divided by p_n^2, the result is at least target_remainder.

    Args:
        Optional named:
        target_remainder (int): The minimum target result of the
                calculation given above.
            Default: 10 ** 10 + 1
    
    Returns:
    Strictly positive integer (int) giving the smallest number
    n such that the remainder when:
        (p_n - 1)^n + (p_n - 1)^n
    is divided by p_n^2, the result is at least target_remainder.
    """
    # Review- prove that for even n the remainder is always 2 and
    # try to find further rules that restricts the search space,
    # or enables direct calculation of the answer- see solution to
    # Project Euler #120
    #since = time.time()
    ps = PrimeSPFsieve()
    # p_n^2 must be strictly greater than the square root of target_remainder,
    # as the remainder on dividing by p_n^2 is strictly smaller than p_n^2.
    start = isqrt(target_remainder) + 1
    mx = start * 10
    ps.extendSieve(mx)
    
    i = bisect.bisect_left(ps.p_lst, start)
    if i & 1: i += 1
    # For even i, the result is always 2
    while True:
        while i >= len(ps.p_lst):
            mx *= 10
            ps.extendSieve(mx)
        #print(i + 1, ps.p_lst[i], calculateSquareRemainder(ps.p_lst[i], i + 1))
        if calculateSquareRemainder(ps.p_lst[i], i + 1) >= target_remainder:
            #print(f"Time taken = {time.time() - since:.4f} seconds")
            return i + 1
        i += 2
    return -1

# Problem 124
def radicalCount(p_facts: Set[int], mx: int) -> int:
    """
    For a given set of distinct primes p_facts, finds the number of positive
    integers up to and including mx whose radical is the product of those
    primes.

    The radical of a positive integer is the product of its distinct prime
    factors (note that as 1 has no prime factoris, it has a radical of the
    multiplicative identity, 1).

    Args:
        Required positional:
        p_facts (set of ints): List of distinct prime numbers for which
                the number of integers not exceeding mx whose radicals
                are equal to the product of these primes is to be
                calculated.
                It is assumed that these are indeed primes, and this
                property is not checked.
        mx (int): The largest number considered.
    
    Returns:
    Integer (int) equal to the number of positive integers not exceeding
    mx whose radicals are equal to the product of p_facts.
    """
    n_p = len(p_facts)
    p_facts_lst = list(p_facts)
    mn = 1
    for p in p_facts_lst: mn *= p
    mx2 = mx // mn
    #if mn < 1: return 0
    #elif mn == 1: return 1
    #res = [0]

    memo = {}
    def recur(curr: int, p_i: int=0) -> int:
        if not curr: return 0
        args = (curr, p_i)
        if args in memo.keys():
            return memo[args]
        res = 1
        for i in range(p_i, n_p):
            p = p_facts_lst[i]
            res += recur(curr // p, p_i=i)
        memo[args] = res
        return res
    
    return recur(mx2, p_i=0)

def orderedRadicals(n: int=100000, k: int=10000) -> int:
    """
    Solution to Project Euler #124

    Consider all the integers between 1 and n inclusive. Sort
    these into a list based on:

    1) The radical of the integer from smallest to largest
    2) For numbers with the same radical, the size of the integer
       from smallest to largest.

    The radical of a positive integer is the product of its distinct prime
    factors (note that as 1 has no prime factoris, it has a radical of the
    multiplicative identity, 1).

    This function calculates the k:th item on that list (1-indexed)

    Args:
        Named positional:
        n (int): The largest number considered
        k (int): Which item on the list to be returned (with k = 1
                corresponding to the first item on the list).
        
    Returns:
    Integer (int) giving the k:th number on the list constructed as
    described above.

    Examples:
        >>> orderedRadicals(n=10, k=4)
        8

        >>> orderedRadicals(n=10, k=6)
        9

        Of the numbers between 1 and 10, the number 1 has radical
        1, the numbers 2, 4 and 8 have radical 2, the numbers
        3 and 9 have radical 3, and the numbers 5, 6, 7 and 10
        do not have any repeated prime factors and so they are
        each their own radical. As per the instructions for sorting
        as given above, the list for n = 10 becomes:
            [1, 2, 4, 8, 3, 9, 5, 6, 7, 10]
        Thus, for n = 10, k = 4 gives the 4th item on this list
        (8) while k = 6 gives the 6th item in this list (9).
    """
    #since = time.time()
    if k == 1:
        #print(f"Time taken = {time.time() - since:.4f} seconds")
        return 1
    ps = PrimeSPFsieve(n_max=k)
    """
    chk = 1
    for i in range(2, n + 1):
        pf = ps.primeFactorisation(i)
        if max(pf.values()) > 1: continue
        p_lst = sorted(pf.keys())
        rad_cnt = radicalCount(p_lst, n)
        chk += rad_cnt
    print(f"Total = {chk}")
    """

    k2 = k - 1 # Minus one to account for 1
    for i in range(2, k + 1):
        pf = ps.primeFactorisation(i)
        if max(pf.values()) > 1: continue
        #p_lst = sorted(pf.keys())
        rad_cnt = radicalCount(pf.keys(), n)
        #print(i, rad_cnt)
        if rad_cnt >= k2: break
        k2 -= rad_cnt
    #print(p_lst)
    #print(k2)
    rad = 1
    mx = n
    p_lst = sorted(pf.keys())
    n_p = len(p_lst)
    for p in p_lst:
        mx //= p
        rad *= p
    heap = [-1]
    def recur(curr: int, p_i: int=0) -> None:
        for i in range(p_i, n_p):
            p = p_lst[i]
            nxt = curr * p
            if nxt > mx: break
            if len(heap) < k2:
                heapq.heappush(heap, -nxt)
            else:
                heapq.heappushpop(heap, -nxt)
            recur(nxt, p_i=i)
        return

    recur(1, p_i=0)
    res = -heap[0] * rad
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 125
def isPalindromic(num: int, base: int=10) -> bool:
    """
    For a given non-negative integer, assesses whether it is
    palindromic when expressed in the chosen base (i.e. the
    digits in the expression read the same forwards and
    backwards).

    Args:
        Required positional:
        num (int): The non-negative integer to be assessed
                for its status as palindromic when expressed
                in the chosen base.
        
        Optional named:
        base (int): Integer strictly greater than 1 giving
                the base in which num is to be expressed
                when assessing whether or not it is
                palindromic.
            Default: 10
    
    Returns:
    Boolean (bool) giving True if num is palindromic when
    expressed in the chosen base and False otherwise.
    """
    digs = []
    num2 = num
    while num2:
        num2, r = divmod(num2, base)
        digs.append(r)
    for i in range(len(digs) >> 1):
        if digs[i] != digs[~i]: return False
    return True

def palindromicConsecutiveSquareSumStart(start: int, mx: int, base: int=10) -> List[int]:
    """
    For a given integer start, finds all of the integers with
    value no greater than mx that can be expressed as the
    sum of at least two consecutive integer squares, starting
    with start^2, and are palindromic in the chosen base (i.e.
    the digits in the expression read the same forwards and
    backwards).

    Args:
        Required positional:
        start (int): The integer whose square is the first
                in the consecutive integer square sums
                considered.
        mx (int): The maximum allowed returned value.

        Optional named:
        base (int): Integer strictly greater than 1 giving the
                base in which integers are to be expressed
                when assessing whether or not they are
                palindromic.
            Default: 10
    
    Returns:
    List of integers (ints) giving all the sums of at least 2
    consecutive squares starting at start^2 that are 
    palindromic and no greater than mx, in strictly increasing
    order.
    """
    curr = start
    tot = curr * curr
    res = []
    while True:
        curr += 1
        tot += curr * curr
        if tot > mx: break
        if isPalindromic(tot): res.append(tot)
    return res

def palindromicConsecutiveSquareSums(
    mx: int=100000000 - 1,
    base: int=10,
) -> int:
    """
    Solution to Project Euler #125

    Finds the sum of all of the integers with value no greater than mx
    that are palindromic in the chosen base (i.e. the digits in the
    expression of the integer in the chosen base read the same forwards
    and backwards) and can be expressed as the sum of at least two
    consecutive integer squares.
    Note that if a palindromic integer can be expressed as the
    sum of consecutive squares in more than one way, it is still
    only included once in the sum.

    Args:
        Optional named:
        mx (int): The maximum value allowed to be included in the
                sum.
            Default: 99999999
        base (int): Integer strictly greater than 1 giving the
                base in which integers are to be expressed
                when assessing whether or not they are
                palindromic.
            Default: 10
    
    Returns:
    Integer (int) giving the sum of all the integers with value no
    greater than mx that are palindromic and can be expressed as the
    sum of at least two consecutive integer squares.
    """
    #since = time.time()
    end = isqrt(mx >> 1)
    res = 0
    palindromic_set = set()
    #count = 0
    for start in range(1, end + 1):
        lst = palindromicConsecutiveSquareSumStart(start, mx, base=10)
        palindromic_set |= set(lst)
        #print(start, lst)
        #count += len(lst)
        res += sum(lst)
    #print(count)
    res = sum(palindromic_set)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 126
def cuboidLayerSizes(
    dims: Tuple[int, int, int],
    min_layer_size: int,
    max_layer_size: int,
) -> List[int]:
    """
    For a cuboid with integer side lengths, consider an iterative process
    where on the first iteration, the faces of the cuboid are completely
    covered by unit cubes (1 x 1 x 1), and on every subsequent iteration
    the visible faces of the unit cubes placed on the previous iteration
    are completely covered by unit cubes. The number of such cubes placed
    in a given iteration is referred to as the layer size of that iteration.

    This function calculates all layer sizes of the given cuboid directions
    for iterations whose layer size is between min_layer_size and
    max_layer_size inclusive.

    Args:
        Required positional:
        dims (3-tuple of ints): 3-tuple of strictly positive integers
                specifying the side lengths of the cuboid around which the
                described cuboid is based.
        min_layer_size (int): Integer giving the smallest layer size that
                it is permitted to include in the result.
        max_layer_size (int): Integer giving the largest layer size that
                it is permitted to include in the result.
    
    Returns:
    List of integers (int) giving the layer sizes for a cuboid with side
    lengths dims that are between min_layer_size and max_layer_size in
    the iterative process described. These are given in strictly increasing
    order.
    """
    n_faces = (dims[0] * dims[1] + dims[0] * dims[2] + dims[1] * dims[2]) << 1
    n_edges = sum(dims) << 2
    if n_faces > max_layer_size: return []
    #n_internal_edges = 0
    #n_internal_corners = 0
    i = 0
    a = 4
    b = n_edges - 4
    c = n_faces - (min_layer_size - 1)
    rad = b ** 2 - 4 * a * c
    if rad >= 0:
        rad_sqrt = isqrt(rad)
        i = max(0, ((rad_sqrt - b) // (2 * a)) + 1)
    
    func = lambda x : a * x ** 2 + b * x + n_faces
    if func(i) < min_layer_size: print(f"i too small")
    #if i > 0 and func(i - 1) >= min_layer_size:
    #    print(f"i too large")
    #    print(f"i = {i}, func(i - 1) = {func(i - 1)}, min_layer_size = {min_layer_size}")
    #print(i)
    res = []
    #n_internal_edges = n_edges
    #n_internal_corners = 6
    while True:
        nxt = func(i)
        #print(nxt, max_layer_size)
        if nxt > max_layer_size: break
        if nxt >= min_layer_size:
            res.append(nxt)
        i += 1
    return res

def cuboidHasLayerSize(
    dims: Tuple[int, int, int],
    target_layer_size: int,
) -> bool:
    """
    For a cuboid with integer side lengths, consider an iterative process
    where on the first iteration, the faces of the cuboid are completely
    covered by unit cubes (1 x 1 x 1), and on every subsequent iteration
    the visible faces of the unit cubes placed on the previous iteration
    are completely covered by unit cubes. The number of such cubes placed
    in a given iteration is referred to as the layer size for that
    iteration.

    This function calculates whether a cuboid with integer side lengths
    given by dims contains an iteration whose layer size is exactly
    target_layer_size.

    Args:
        Required positional:
        dims (3-tuple of ints): 3-tuple of strictly positive integers
                specifying the side lengths of the cuboid around which the
                described cuboid is based.
        target_layer_size (int): Integer giving the layer size which
                one of the iterations must have if True is to be returned.
    
    Returns:
    Boolean (bool) specifying whether for the cuboid with side lengths
    given by dims, any of the iterations in the above described process
    has the layer size of exactly target_layer_size.
    """
    n_faces = (dims[0] * dims[1] + dims[0] * dims[2] + dims[1] * dims[2]) << 1
    n_edges = sum(dims) << 2
    a = 4
    b = n_edges - 4
    c = n_faces - target_layer_size
    rad = b ** 2 - 4 * a * c
    if rad < 0: return False
    rad_sqrt = isqrt(rad)
    if rad_sqrt ** 2 != rad: return False
    return rad_sqrt >= b

def cuboidLayers(
    target_layer_size_count: int=1000,
    batch_size: int=10000,
) -> int:
    """
    Solution to Project Euler #126

    For a cuboids with integer side lengths, consider an iterative process
    where on the first iteration, the faces of the cuboid are completely
    covered by unit cubes (1 x 1 x 1), and on every subsequent iteration
    the visible faces of the unit cubes placed on the previous iteration
    are completely covered by unit cubes. The number of such cubes placed
    in a given iteration is referred to as the layer size for that
    iteration.

    This function calculates the smallest positive integer for which
    exactly target_layer_size_count distinct cuboids with integer side length
    have an iteration of the described process with a layer size equal
    to that integer.

    Two cuboids are considered distinct if and only if they cannot be
    rotated to lie on top of each other, or equivalently, their lists
    of side lengths are not permutations of each other.

    Args:
        Optional named:
        target_layer_size_count (int): Strictly positive integer giving
                the exact number of distinct cuboids with integer side
                length that should have an iteration of the above described
                process equal to the returned value.
            Default: 1000
        batch_size (int): The number of layer sizes simultaneously checked
                in the implementation. Larger sizes mean more efficient
                computation, but would be expected to cause a greater
                number of layer sizes in excess of the eventual solution
                (that are therefore not required for the solution) to be
                processed.
    
    Returns:
    Integer (int) giving the smallest positive integer for which exactly
    target_layer_size_count distinct cuboids with integer side length
    have an iteration of the described process with a layer size equal
    to that integer.
                
    Outline of rationale:
    TODO
    """
    #since = time.time()

    #step_size = 20000
    sz_rng = [1, batch_size]
    print(sz_rng)
    #tot = 0
    #tot2 = 0
    while True:
        counts = {}
        candidates = SortedList()
        a_mx = (sz_rng[1] - 2) // 4
        for a in range(1, a_mx + 1):
            #print(f"a = {a}")
            b_mx = (sz_rng[1] - 2 * a) // (2 * (a + 1))
            for b in range(1, min(a, b_mx) + 1):
                #print(f"b = {b}")
                c_mx = (sz_rng[1] - 2 * a * b) // (2 * (a + b))
                for c in range(1, min(b, c_mx) + 1):
                    #print(f"c = {c}")
                    #print(a, b, c)
                    lst = cuboidLayerSizes((a, b, c), min_layer_size=sz_rng[0], max_layer_size=sz_rng[1])
                    #print(a, b, c)
                    #print(lst)
                    for sz in set(lst):
                        #tot += 1
                        counts[sz] = counts.get(sz, 0) + 1
                        if counts[sz] == target_layer_size_count:
                            candidates.add(sz)
                        elif counts[sz] == target_layer_size_count + 1:
                            candidates.remove(sz)
        #print(sz_rng)
        #print(counts)
        #if 154 in counts.keys():
        #    print(f"C(154) = {counts[154]}")
        #tot2 += sum(counts.values())
        if candidates: break
        sz_rng = [sz_rng[1] + 1, sz_rng[1] + batch_size]
        print(sz_rng)
    #print(f"tot = {tot}")
    #print(f"tot2 = {tot2}")
    res = candidates[0]
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 127
def abcHits(c_max: int=199999) -> int:
    """
    Solution to Project Euler #127

    Consider strictly positive integers a, b and c such that
     1) a + b = c
     2) gcd(a, b) = 1, gcd(a, c) = 1 and gcd(b, c) = 1
     3) a < b
    
    Find the sum of the values of c for all such combinations
    for which c is no greater than c_max and the radical of
    the product of a, b and c is strictly less than c.

    The radical of a non-negative integer is the product of
    its unique prime factors (for the multiplicative identity
    1, it is defined to be 1).

    Args:
        Optional named
        c_max (int): The largest value of c for the a, b, c
                combinations considered.
            Default: 199999
    
    Returns:
    Integer (int) giving the sum over the c values for all
    unique a, b, c combinations that satisfy all of the
    described properties.
    """
    #since = time.time()

    # Note that if a + b = c and gcd(b, c) = 1 then gcd(a, c) = 1
    # and gcd(a, b) = 1 and rad(abc) = rad(a) * rad(b) * rad(c).

    def radical(p_facts: List[int]) -> int:
        res = 1
        for p in p_facts: res *= p
        return res

    ps = PrimeSPFsieve(n_max=c_max, use_p_lst=True)
    radicals = [1] * (c_max + 1)
    for p in ps.p_lst:
        for i in range(p, c_max + 1, p):
            radicals[i] *= p
    
    b_radicals = SortedList()

    res = 0
    for c in range(5, c_max + 1):
        if not c % 1000: print(f"c = {c}")
        b_radicals.add((radicals[c - 2], c - 2))
        if not c & 1:
            b_radicals.remove((radicals[c >> 1], c >> 1))
        #c_facts = ps.primeFactors(c)
        rad_c = radicals[c]
        if rad_c == c: continue
        rad_ab_mx = (c - 1) // rad_c
        rad_b_mx = rad_ab_mx >> 1
        i_mx = b_radicals.bisect_right((rad_b_mx, float("inf")))
        for i in range(i_mx):
            rad_b, b = b_radicals[i]
            if gcd(rad_b, rad_c) != 1: continue
            a = c - b
            if radicals[a] * rad_b <= rad_ab_mx:
                res += c
        b = c - 1
        #b_facts = ps.primeFactors(b)
        if radicals[b] <= rad_ab_mx:
            res += c
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
    
# Problem 128
def hexagonalLayerPrimeDifferenceCountIs3(layer: int, ps: PrimeSPFsieve) -> List[int]:
    """
    Consider a tessellating tiling of numbered regular hexagons
    of equal size constructed in the following manner. First,
    place hexagon 1. This is layer 0. For each subsequent layer
    (labelled with the number one greater than that of the
    previous layer), place the next unused positive integer
    hexagon so that it shares an edge (i.e. neighbours) only the
    first placed hexagon of the previous layer. Then, place the
    next unused positive integer hexagon so that it neighbours
    the first hexagon of that layer and a hexagon in the previous
    layer on the anticlockwise side of the first hexagon of that
    layer (where rotation is around hexagon 1, the very first
    hexagon placed). Then place repeatedly place the hexagon with
    the next unused positive integer in the unique position
    neighbouring the immediately previously placed hexagon and
    a hexagon in the previous layer until there are no such
    positions available. The last hexagon placed in the layer
    will be neighbouring the first hexagon placed in the layer.
    Then repeat the process with the next layer.

    For a given layer, numbered as described and with number
    no less than 2, this identifies the numbers of the hexagons
    in that layer for which three of the neighbouring hexagons
    in the tiling (i.e. hexagons that share an edge with the
    chosen hexagon) have numbers that differ from the number of
    the chosen hexagon (either above or below) by a prime number. 

    Args:
        Required positional:
        layer (int): Non-negative integer giving the layer number
                to be considered, where layer 0 consists of hexagon
                1 only, layer 1 consists of the hexagons 2 to 7,
                layer 2 consists of the hexagons 8 to 19 etc.
        ps (PrimeSPFsieve object): Object representing a prime
                sieve, enabling the rapid assessment of whether
                a number is prime in the case of repeated testing
                of relatively small number (<= 10 ** 6).
    
    Returns:
    List of integers (int) giving the numbers of hexagons in the
    chosen layer for which three of the neighbouring hexagons
    in the tiling (i.e. hexagons that share an edge with the
    chosen hexagon) have numbers that differ from the number of
    the chosen hexagon (either above or below) by a prime number.
    These are sorted in strictly increasing order.
    
    Note that this uses the fact that the only hexagons in a
    layer with number no less than 2 that can possibly have the
    described property are the first and last hexagons placed
    in that layer. For an outline of a proof of this, see
    documentation for hexagonalTileDifferences().
    """
    # layer >= 2
    #if idx not in {0, 1, 3, 5}: return False
    #num = findHexagonalCorner(layer, idx)
    if not ps.isPrime(layer * 6 - 1, extend_sieve=False, extend_sieve_sqrt=True):
        return []
    diffs = [
        (6 * layer + 1, 12 * layer + 5),
        (12 * layer - 7, 6 * layer + 5)
    ]
    res = []
    if ps.isPrime(6 * layer + 1, extend_sieve=False, extend_sieve_sqrt=True) and ps.isPrime(12 * layer + 5, extend_sieve=False, extend_sieve_sqrt=True):
        #print(layer, 0)
        res.append(3 * layer * (layer - 1) + 2)
    
    if ps.isPrime(6 * layer + 5, extend_sieve=False, extend_sieve_sqrt=True) and ps.isPrime(12 * layer - 7, extend_sieve=False, extend_sieve_sqrt=True):
        #print(layer, 1)
        res.append(3 * layer * (layer + 1) + 1)
    
    return res

def hexagonalTileDifferences(sequence_number: int=2000) -> int:
    """
    Consider a tessellating tiling of numbered regular hexagons
    of equal size constructed in the following manner. First,
    place hexagon 1. This is layer 0. For each subsequent layer
    (labelled with the number one greater than that of the
    previous layer), place the next unused positive integer
    hexagon so that it shares an edge (i.e. neighbours) only the
    first placed hexagon of the previous layer. Then, place the
    next unused positive integer hexagon so that it neighbours
    the first hexagon of that layer and a hexagon in the previous
    layer on the anticlockwise side of the first hexagon of that
    layer (where rotation is around hexagon 1, the very first
    hexagon placed). Then place repeatedly place the hexagon with
    the next unused positive integer in the unique position
    neighbouring the immediately previously placed hexagon and
    a hexagon in the previous layer until there are no such
    positions available. The last hexagon placed in the layer
    will be neighbouring the first hexagon placed in the layer.
    Then repeat the process with the next layer.

    Now consider all the hexagons in this tiling for which three
    of the neighbouring hexagons in the tiling (i.e. hexagons that
    share an edge with the chosen hexagon) have numbers that differ
    from the number of the chosen hexagon (either above or below)
    by a prime number. Let the numbers of all such hexagons,
    organised in strictly increasing order form a sequence. This
    function identifies term sequence_number in that sequence
    (where the first term is term 1).

    Args:
        Optional named:
        sequence_number (int): Strictly positive integer
                specifying the term in the sequence described to be
                returned, with the sequence starting with term
                1.
    
    Returns:
    Integer (int) giving term sequence_number in the sequence
    described above.

    Solution to Project Euler #128

    Outline of rationale:

    In the first two layers (up to hexagon 7) the hexagons
    with 3 neighbours with prime differences are hexagons
    1 and 2

    Counting the layers from 0, from layer 2 onwards (the layer
    starting with number 8) the only possible hexagons for which
    three adjacent hexagons have prime difference are the
    first and last hexagon in that layer. This can be proved as
    follows.
    
    First note that after the first two layers, the difference
    between any two neighbours is either 1 or strictly greater
    than 2 (so any neighbour difference divisible by 2 for these
    layers means that the difference is not prime).
    
    Now, consider the hexagons for which the preceding and
    succeeding values are opposite. We refer to these as
    edge hexagons. The differences between this tile and
    the preceding and suceeding tiles are both one, which
    is not prime. Considering the two neighbouring hexagons
    on the next layer in. These are two consecutive numbers
    and so the difference with the chosen hexagon must be
    even for one of them and so (since as established the
    difference cannot be 2) not prime. Thus, the difference
    can only be prime for at most one of these hexagons.
    Using identical reasoning, we can also conclude that
    for the two neighbouring hexagons on the next layer
    out, at most one of the differences with the chosen
    hexagon can be prime. Thus, for edge hexagons, the
    largest number of prime differences with neighbouring
    hexagons is 2, so none of these hexagons will be
    counted.

    Consider the hexagons for which the preceding and
    succeeding values are neighbouring but not opposite.
    We refer to these as corner hexagons. As for the edge
    hexagons, the differences between this tile and the
    preceding and succeeding tiles are both one, which
    is not prime. Considering the three neighbouring
    hexagons on the next layer out, we first note that
    the middle of these is a corner hexagon of the next
    layer out, which we refer to as the corresponding
    corner hexagon of the next layer out. These three
    hexagons contain three consecutive numbers. At most
    two of these can have prime difference with the
    chosen hexagon, and when that is the case they must
    be the two hexagons other than the corresponding
    corner hexagon of the next layer out. The remaining
    neighbouring hexagon is on the next layer in and
    is also a corner hexagon, which we refer to as the
    corresponding corner hexagon of the next layer in.
    As such, in order for three of the differences to
    be prime, the difference with the corresponding
    corner hexagon of the next layer in must be prime
    and the difference with the preceding and
    succeeding hexagons of the corresponding corner
    hexagon of the next layer out must both be
    prime. It can be shown that the corresponding corner
    hexagons of the next layer out and in must either
    both be odd or both be even, and so the corresponding
    corner hexagon on the next layer in must have
    different parity from the preceding and succeeding
    hexagons of the corresponding corner hexagon of the
    next layer out. This implies that the differences
    of these three hexagons and the chosen hexagon
    cannot all be odd and so (since as established the
    differences are all strictly greater than 2) cannot
    all be prime. Thus, like for edge hexagons, for
    corner hexagons, the largest number of prime
    differences with neighbouring hexagons is 2, so none
    of these hexagons will be counted.

    The only cases that remain for layers 2 and out
    (i.e. the only hexagons on these layers that are
    not classified as either an edge hexagon or as
    a corner hexagon) are when the hexagon does not
    neighbour to both its preceding and succeeding
    hexagon, which is the case if and only if the
    hexagon is the first in its layer or the last in
    its layer. Therefore, we restrict our search to
    those two cases. In both cases there are only 3
    neighbouring hexagons that may have prime difference,
    and a formula can be derived based on the layer
    number to calculate those candidate differences.
    TODO
    """
    #since = time.time()
    ps = PrimeSPFsieve(12 * sequence_number)

    if sequence_number <= 2: return sequence_number
    count = 2
    layer = 2
    while True:
        layer_candidates = hexagonalLayerPrimeDifferenceCountIs3(layer, ps)
        count += len(layer_candidates)
        #if layer_candidates: print(layer_candidates)
        if count >= sequence_number:
            #print(f"Time taken = {time.time() - since:.4f} seconds")
            return layer_candidates[~(count - sequence_number)]

        layer += 1
    return -1

# Problem 129
def findSmallestRepunitDivisibleByK(k: int, base: int=10) -> int:
    """
    For a given base, finds the smallest repunit in that base
    that is divisible by k. If no such repunit exists, then
    returns -1.

    In a given base, a repunit of length n (where n is strictly
    positive) is the strictly positive integer that when
    expressed in the chosen base is the concatenation of
    n 1s. For instance, the repunit of length 3 for base 10
    is 111 and the repunit of length 4 for base 2 is
    15 (which, when expressed in base 2 i.e. binary is 1111).

    Args:
        Required positional:
        k (int): Strictly positive integer giving the quantity
                for which the returned repunit must be divisible.
        
        Optional named:
        base (int): The base in which the repunits are expressed.
            Default: 10

    Returns:
    Integer (int) giving the value of the smallest repunit in
    the chosen base that is divisible by k if any such
    repunit exists, otherwise -1.

    Note that we have used the property that if k and
    base are not coprime then no repunit in this base can
    be divisible by k. To see this, suppose k and base share
    a prime divisor p and there exists a repunit r in the
    chosen base that is divisible by k. Then r = 0 (mod p)
    and base = 0 (mod p). Now, r - 1 ends in a 0 when
    expressed in the chosen base and so is divisible by base,
    and therefore p. Consequently:
        r - 1 = 0 (mod p)
        r = p - 1 (mod p)
        0 = p - 1 (mod p)
    Given that no prime is less than 2, this cannot occur,
    so we have a contradiction. Therefore, if k and base
    share a prime divisor then there cannot exist a repunit
    in that base that is divisible by k.

    We have also used the property that if there exists a
    repunit divisible by k then the shortest such repunit
    will be at most length k (see documentation of
    repunitDivisibility() for an outline of the proof of
    this).
    """
    if gcd(k, base) != 1: return -1
    if k == 1: return 1
    base_mod_k = base % k
    res = 1
    curr = 1
    while curr:
        curr = (curr * base_mod_k + 1) % k
        res += 1
        if res > k: return -1
    return res
    
def repunitDivisibility(target_repunit_length: int=1000000, base: int=10) -> int:
    """
    Solution to Project Euler #129

    For a given base, finds the smallest integer k such that
    the smallest repunit in that base divisible by k exists and
    is no smaller than target_repunit_length.

    In a given base, a repunit of length n (where n is strictly
    positive) is the strictly positive integer that when
    expressed in the chosen base is the concatenation of
    n 1s. For instance, the repunit of length 3 for base 10
    is 111 and the repunit of length 4 for base 2 is
    15 (which, when expressed in base 2 i.e. binary is 1111).

    Args:
        Optional named:
        target_repunit_length (int): Strictly positive integer
                giving the target size of the smallest repunit
                divisible by the returned value k.
            Default: 1000000
        base (int): The base in which the repunits are expressed.
            Default: 10

    Returns:
    Integer (int) giving the value k such that the smallest
    repunit in that base divisible by k exists and is no smaller
    than target_repunit_length.
    
    Outline of rationale:

    We observe that if there exists a repunit that is divisible
    by an integer k, then the smallest such repunit must have
    a length that does not exceed k. This can be seen by finding
    the remainder on division by k as we build up the repunit
    adding one 1 at a time. At each stage, we can calculate
    the value by multiplying the value for the previous repunit
    by 10 and adding 1, then taking the modulus. Thus, the
    value of a repunit can be calculated solely from the repunit
    with one fewer digit. Suppose the value calculated has been
    seen before. Then this will give rise to an infinite cycle.
    Thus, if the value 0 occurs in any of the repunits, then
    there can be no repeated values for the remainder among all
    of the repunits smaller than it. As there are only (k - 1)
    other possible remainders, this implies that if any of the
    repunits have remainder 0 on division by k (and thus the
    repunit is divisible by k) then the first such occurrence
    must be for a repunit of length no greater than k.
    """
    #since = time.time()
    num = target_repunit_length
    while True:
        if findSmallestRepunitDivisibleByK(num, base=base) >= target_repunit_length:
            break
        num += 1
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return num

def compositesWithPrimeRepunitProperty(n_smallest: int, base: int=10) -> List[int]:
    """
    For a given base, finds the n_smallest smallest composite
    integers such that for each such integer, the smallest repunit
    in that base divisible by the integer exists and the number of
    digits it contains in the chosen base exactly divides one less
    than the integer.

    A composite integer is a strictly positive integer such that
    there exists a prime number that exactly divides it that is not
    equal to that integer. 

    In a given base, a repunit of length n (where n is strictly
    positive) is the strictly positive integer that when
    expressed in the chosen base is the concatenation of
    n 1s. For instance, the repunit of length 3 for base 10
    is 111 and the repunit of length 4 for base 2 is
    15 (which, when expressed in base 2 i.e. binary is 1111).

    Args:
        Required positional:
        n_smallest (int): The number of integers with the described
                property to be found.
        
        Optional named:
        base (int): The base in which the repunits are expressed.
            Default: 10
    
    Returns:
    List of integers (int) giving the smallest n_smallest composite
    integers with the described property in strictly increasing
    order.
    """

    ps = PrimeSPFsieve()
    p_gen = ps.endlessPrimeGenerator()
    p0 = 2
    res = []
    for p in p_gen:
        #print(p0, p)
        for i in range(p0 + 1, p):
            val = findSmallestRepunitDivisibleByK(i, base=base)
            if val > 0 and not (i - 1) % val:
                res.append(i)
                if len(res) == n_smallest: break
        else:
            p0 = p
            continue
        break
    print(res)
    return res


def sumCompositesWithPrimeRepunitProperty(n_to_sum=25, base: int=10) -> List[int]:
    """
    Solution to Project Euler #130
    
    For a given base, finds sum of the n_to_sum smallest composite
    integers such that for each such integer, the smallest repunit
    in that base divisible by the integer exists and the number of
    digits it contains in the chosen base exactly divides one less
    than the integer.

    A composite integer is a strictly positive integer such that
    there exists a prime number that exactly divides it that is not
    equal to that integer. 

    In a given base, a repunit of length n (where n is strictly
    positive) is the strictly positive integer that when
    expressed in the chosen base is the concatenation of
    n 1s. For instance, the repunit of length 3 for base 10
    is 111 and the repunit of length 4 for base 2 is
    15 (which, when expressed in base 2 i.e. binary is 1111).

    Args:
        Required positional:
        n_to_sum (int): The number of integers with the described
                property to be included in the sum.
        
        Optional named:
        base (int): The base in which the repunits are expressed.
            Default: 10
    
    Returns:
    Integer (int) giving the sum of the smallest n_to_sum composite
    integers with the described property in strictly increasing
    order.
    """
    #since = time.time()
    res = sum(compositesWithPrimeRepunitProperty(n_to_sum, base=base))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 131
def findPrimeCubePartnerships(p_max: int) -> List[Tuple[int, int]]:
    """
    

    Outline of rationale:

    We can factor n^3 + n^2 * p to get:
        n^2 * (n + p)
    Given that p is prime, n is either coprime with p or a
    multiple of p. If n is a multiple of p then n = a * p
    for some strictly positive integer a. Then the expression
    becomes:
        a^2 * (a + 1) * p^3
    For this to be a perfect cube, a^2 * (a + 1) must be
    a perfect cube. For a^2 to be a perfect cube, a must be a
    perfect cube. Since they differ by 1, a and (a + 1) are
    coprime, for a^2 * (a + 1) to be a perfect cube, both
    a and (a + 1) must be perfect cubes. Since there are no
    two strictly positive consecutive cubes that differ by
    1 this is not possible and we have a contradiction.
    Therefore, n^3 + n^2 * p cannot be a perfect cube if
    n is a multiple of p.

    Consequently, n and p must be coprime. In that case, n
    and (n + p) are coprime, so for n^2 * (n + p) to be a
    perfect cube both n and (n + p) must be perfect cubes.
    Then since n and p are strictly positive integers, there
    exist strictly positive integer n = a^3 and (n + p) = b^3.
    Note that since p is strictly positive, b is strictly
    greater than a, This means that in terms of a and b:
        p = b^3 - a^3 = (b - a) * (a^2 + a * b + b^2)
    Now, since a and b are strictly positive and b is strictly
    greater than a, (a^2 + a * b + b^2) is an integer strictly
    greater than 1 and (b - a) is a strictly positive integer.
    Therefore, in order for p to be prime, (b - a) must equal
    1. Thus, n and (n + p) must be consecutive perfect cubes.
    
    Conversely, consider two strictly positive consecutive cubes
    whose difference is prime, a and (a + 1). Let n = a^3 and
    p = (a + 1)^3 - a^3. Then:
        n^2 * (n + p) = a^6 * (a + 1)^3 = (a^2 * (a + 1))^3
    which is a perfect cube. Therefore, there is a one-to-one
    correspondence between the values of strictly positive
    integer n and prime p pairs such that (n^3 + n^2 * p) and
    consecutive perfect cubes a^3 and (a + 1)^3 pairs that
    differ by a prime number (where a is a strictly positive
    integer), linked by the relations:
        n = a^3, p = (a + 1)^3 - a^3
    
    We are therefore searching for positive integer a and
    prime p no greater than p_max pairs such that:
        (a + 1)^3 - a^3 - p = 0
        3 * a^2 + 3 * a - (p - 1) = 0
    Solving this in terms of a, we get the discriminant:
        12 * p - 3
    To get an integer solution for a, this must be a square.
    Since p is an integer, (12 * p - 3) is divisible by 3,
    so if this is a square then the square root must also
    be divisible by 3, so (12 * p - 3) is divisible by 3^2 = 9.
    Therefore, for a prime p a solution exists as long as:
        (4 * p - 1) / 3
    is a perfect square. Thus, p can only be part of a solution
    if there exists an integer d such that:
        d^2 = (4 * p - 1) / 3
    which can be rearranged to get:
        p = (3 * d^2 + 1) / 4
    and which in such a case, by finding the solution to the
    quadratic we find:
        a = (-3 +/- 3 * d) / 6
    Since a must be positive and the negative branch always gives
    a negative answer, we must choose the positive branch, which
    gives the only possible n to give a solution with this p (as
    long as this is an integer):
        n = ((d - 1) / 2)^3

    Conversely, consider a non-negative integer d. If:
        p = (3 * d^2 + 1) / 4
    is a prime integer, and we let n = ((d - 1) / 2)^3, then
        n^2 * (n + p) = n^2 * (d^3 - 3 * d^2 + 3 * d - 1 + 6 * d^2 + 2) / 8
                      = n^2 * (d^3 + 3 * d^2 + 3 * d + 1) / 8
                      = (((d - 1) / 2) * ((d + 1) / 2)) ^ 3
    Note that for p to be an integer, (3 * d^2 + 1) must be
    divisible by 4 and so d must be odd, which also guarantees
    that (d - 1) and (d + 1) are even, so n is an integer and
    n^2 * (n + p) is a perfect cube.

    Consequently, all (p, n) pairs for prime p and strictly positive
    integer n for which n^3 + n^2 * p is a perfect cube have a
    one-to-one correspondence with the odd positive integers d,
    where (3 * d^2 + 1) / 4 is a prime number, where the
    corresponding (p, n) pair to such a value of d is:
        (3 * d^2 + 1) / 4, ((d - 1) / 2)^3)

    Therefore, given that increasing d gives increasing p, we can
    find all possible solutions with p <= p_max by iterating over
    increasing odd values of d up to and including the floor of the
    square root of (4 * p_max - 1) divided by 3, choosing those for
    which p = (3 * d^2 + 1) / 4 is a prime number and adding:
        ((3 * d^2 + 1) / 4, ((d - 1) / 2)^3)
    as a (p, n) solution pair for each such value of d.
    """
    ps = PrimeSPFsieve(n_max=isqrt(p_max))#n_max=12 * n_max)
    res = []
    for d in range(1, isqrt((4 * p_max - 1) // 3) + 1, 2):
        p, r = divmod(3 * d ** 2 + 1, 4)
        if not r and ps.isPrime(p, extend_sieve=False, extend_sieve_sqrt=True):
            res.append((p, ((d - 1) // 2) ** 3))
    #print(res)
    return res
    """
    ps = PrimeSPFsieve()
    res = []
    for p in ps.endlessPrimeGenerator():
        if p > n_max: break
        q, r = divmod(4 * p - 1, 3)
        if r: continue
        num = q
        #num = 12 * p - 3
        #pf = ps.primeFactorisation(num)
        #if all(not x & 1 for x in pf.values()):
        #    res.append(p)
        if isqrt(num) ** 2 == num:
            res.append(p)
    print(res)
    return res
    """

def primeCubePartnership(p_max: int=999999) -> int:
    """
    Solution to Project Euler #131
    """
    #since = time.time()
    res = len(findPrimeCubePartnerships(p_max))
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 132
def repunitPrimeFactors(n_ones: int, n_p: int, base: int=10) -> List[int]:
    """
    Finds the n_p smallest distinct prime factors of the
    chosen base's repunit of length n_ones.

    In a given base, a repunit of length n (where n is strictly
    positive) is the strictly positive integer that when
    expressed in the chosen base is the concatenation of
    n 1s. For instance, the repunit of length 3 for base 10
    is 111 and the repunit of length 4 for base 2 is
    15 (which, when expressed in base 2 i.e. binary is 1111).

    Args:
        Required positional:
        n_ones (int): The length of the repunit in the chosen
                base whose prime factors are being sought.
        n_p (int): The number of distinct prime factors to
                be found.
        
        Optional named:
        base (int): The base in which the repunit is expressed,
                and in which it consists of the concatenation
                of n_ones ones.
            Default: 10

    Returns:
    List of n_p integers (ints), giving the n_p smallest distinct
    prime factors of the chosen base's repunit of length n_ones
    in strictly increasing order.
    """
    ps = PrimeSPFsieve(base)
    base_facts = ps.primeFactors(base)
    mx_base_fact = max(base_facts)
    p_gen = ps.endlessPrimeGenerator()
    #print(format(n_ones, "b"))

    def pDividesRepunit(n_ones: int, p: int, base: int=10) -> bool:
        #print(f"p = {p}")
        if n_ones == 1: return False
        res = 0
        #n_ones >>= 1
        curr = 1
        #print(curr, res)
        base_pow_md = base % p
        while True:
            if n_ones & 1:
                res = (res * base_pow_md + curr) % p
            n_ones >>= 1
            if not n_ones: break
            curr = (curr * (base_pow_md + 1)) % p
            if not curr: break
            #print(curr, res)
            base_pow_md = pow(base_pow_md, 2, p)
        #print(res)
        return not res
    res = []
    for p in p_gen:
        if p in base_facts:
            if p == mx_base_fact: break
            continue
        #if n_ones % len(repunitDivisorCycle(p, base=base)): continue
        if not pDividesRepunit(n_ones, p, base=base): continue
        res.append(p)
        #print(len(res), p)
        if len(res) == n_p:
            return res
    for p in p_gen:
        #if n_ones % len(repunitDivisorCycle(p, base=base)): continue
        if not pDividesRepunit(n_ones, p, base=base): continue
        res.append(p)
        #print(len(res), p)
        if len(res) == n_p:
            return res
    return res

def repunitPrimeFactorsSum(n_ones: int=1000000000, n_p: int=40, base: int=10) -> int:
    """
    Solution to Project Euler #132

    Finds the sum of the n_p smallest distinct prime factors of
    the chosen base's repunit of length n_ones.

    In a given base, a repunit of length n (where n is strictly
    positive) is the strictly positive integer that when
    expressed in the chosen base is the concatenation of
    n 1s. For instance, the repunit of length 3 for base 10
    is 111 and the repunit of length 4 for base 2 is
    15 (which, when expressed in base 2 i.e. binary is 1111).

    Args:
        Required positional:
        n_ones (int): The length of the repunit in the chosen
                base whose prime factors are being sought.
        n_p (int): The number of distinct prime factors to
                be found.
        
        Optional named:
        base (int): The base in which the repunit is expressed,
                and in which it consists of the concatenation
                of n_ones ones.
            Default: 10

    Returns:
    Integer (int), giving the sum of the n_p smallest distinct
    prime factors of the chosen base's repunit of length n_ones.
    """
    #since = time.time()
    p_lst = repunitPrimeFactors(n_ones, n_p, base=base)
    #print(p_lst)
    res = sum(p_lst)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 133
"""
def repunitDivisorCycle(p: int, base: int=10) -> List[int]:
    res = [0, 1]
    while res[-1] != 0:
        res.append((res[-1] * base + 1) % p)
    res.pop()
    return res
"""

def repunitPowBaseNonFactors(p_max: int, base: int=10) -> List[int]:
    """
    Finds the primes not exceeding p_max that do not divide any
    of the chosen base's repunits with length base^n for any
    non-negative integer n.

    In a given base, a repunit of length n (where n is strictly
    positive) is the strictly positive integer that when
    expressed in the chosen base is the concatenation of
    n 1s. For instance, the repunit of length 3 for base 10
    is 111 and the repunit of length 4 for base 2 is
    15 (which, when expressed in base 2 i.e. binary is 1111).

    Args:
        Required positional:
        p_max (int): The largest possible value of the prime
                numbers considered.
        
        Optional named:
        base (int): The base in which the repunit is expressed,
                and in which it consists of the concatenation
                of n_ones ones.
            Default: 10

    Returns:
    List of integers (ints), giving the prime numbers no greater
    than p_max that do not divide any of the chosen base's repunits
    with length base^n for any non-negative integer n, in strictly
    increasing order.
    """
    ps = PrimeSPFsieve()
    base_facts = set(ps.primeFactors(base))
    res = []
    for p in ps.endlessPrimeGenerator():
        if p > p_max: break
        elif p in base_facts:
            res.append(p)
            continue
        #cycle_len = len(repunitDivisorCycle(p, base=base))
        cycle_len = findSmallestRepunitDivisibleByK(p, base=base)
        p_facts = set(ps.primeFactors(cycle_len))
        if not p_facts.issubset(base_facts):
            res.append(p)
        else:
            print(p)
    return res

def repunitPowBaseNonFactorsSum(p_max: int=99_999, base: int=10) -> List[int]:
    """
    Finds the sum of all prime numbers not exceeding p_max that
    do not divide any of the chosen base's repunits with length
    base^n for any non-negative integer n.

    In a given base, a repunit of length n (where n is strictly
    positive) is the strictly positive integer that when
    expressed in the chosen base is the concatenation of
    n 1s. For instance, the repunit of length 3 for base 10
    is 111 and the repunit of length 4 for base 2 is
    15 (which, when expressed in base 2 i.e. binary is 1111).

    Args:
        Optional named:
        p_max (int): The largest possible value of the prime
                numbers included in the sum.
            Default: 10^5 - 1
        base (int): The base in which the repunit is expressed,
                and in which it consists of the concatenation
                of n_ones ones.
            Default: 10

    Returns:
    Integer (int), giving sum over the prime numbers no greater
    than p_max that do not divide any of the chosen base's repunits
    with length base^n for any non-negative integer n.
    """
    #since = time.time()
    repunit_nonfactors = repunitPowBaseNonFactors(p_max, base=10)
    print(repunit_nonfactors)
    res = sum(repunit_nonfactors)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 134
def extendedEuclideanAlgorithm(a: int, b: int) -> Tuple[int, Tuple[int, int]]:
    """
    Implementation of the extended Euclidean Algorithm to find the
    greatest common divisor (gcd) of integers a and b and finds an
    ordered pair of integers (m, n) such that:
        m * a + n * b = gcd(a, b)
    
    Args:
        Required positional:
        a (int): The first of the integers on which the extended
                Euclidean Algorithm is to be applied
        b (int): The second of the integers on which the extended
                Euclidan Algorithm is to be applied
    
    Returns:
    2-tuple whose index 0 contains a non-negative integer giving
    the greatest common divisor (gcd) of a and b, and whose
    index 1 contains a 2-tuple of integers, giving an ordered
    pair of integers (m, n) such that:
        m * a + n * b = gcd(a, b)
    """
    if b > a:
        swapped = True
        a, b = b, a
    else: swapped = False
    #print(a, b)
    q_stk = []
    curr = [a, b]
    while True:
        q, r = divmod(*curr)
        if not r: break
        q_stk.append(q)
        curr = [curr[1], r]
    #if not q_stk:

    g = curr[1]
    mn_pair = [0, 1]
    #print(mn_pair)
    #qr_pair = qr_pair_stk.pop()
    while q_stk:
        q = q_stk.pop()
        mn_pair = [mn_pair[1], mn_pair[0] + mn_pair[1] * (-q)]
        #print(mn_pair)
    #print(mn_pair[0] * a + mn_pair[1] * b, g)
    if swapped: mn_pair = mn_pair[::-1]
    return (g, tuple(mn_pair))

def solveLinearCongruence(a: int, b: int, md: int) -> int:
    """
    Finds the smallest non-negative integer k such that solves
    the linear congruence:
        k * a = b (mod md)
    if such a value exists.

    A congruence relation for two integers m and n over a given
    modulus md:
        m = n (mod md)
    is a relation such that there exists an integer q such that:
        m + q * md = n
    
    Args:
        Required positional:
        a (int): Integer specifying the value of a in the above
                congruence to be solved for k.
        b (int): Integer specifying the value of b in the above
                linear congruence to be solved for k.
        md (int): Strictly positive integer specifying the
                modulus of the congruence (i.e. the value md in
                the linear congruence to be solved for k)
        
    Returns:
    Integer (int) giving the smallest non-negative integer value
    of k for which the linear congruence:
        k * a = b (mod md)
    is true if any such value exists, otherwise -1.

    Outline of method:
    Solves by first using the extended Euclidean algorithm to
    find the greatest common divisor (gcd) of a and md and
    an integer pair (m, n) for which:
        m * a + n * md = gcd(a, md)
    This implies the congruence:
        m * a = gcd(a, md) (mod md)
    If gcd(a, md) does not divide b then the linear congruence
    has no solution, as any linear combination of a and md with
    integer coefficients is a multiple of gcd(a, md). Otherwise,
    a solution to the linear congruence is:
        k = m * (b / gcd(a, md))
    A known property of linear congruences is that if there
    exists a solution, then any other integer is a solution
    if and only if it is congruent to the known solution under
    the chosen modulus.
    Therefore, to find the smallest non-negative such value,
    we take the smallest non-negative integer to which this
    value is congruent modulo md (which in Python can be found
    using k % md).
    """
    a %= md
    g, (m, n) = extendedEuclideanAlgorithm(a, md)
    #print(m, n, g)
    b %= md
    q, r = divmod(b, g)
    return -1 if r else (q * m) % md

def primeConnection(p1: int, p2: int, base: int=10) -> int:
    """
    For two prime number p1 and p2, finds the smallest positive
    multiple of p2 which, when expressed in the chosen
    base, contains the expression of p1 in than base (without
    leading zeroes) as a suffix, if such a value exists.

    Args:
        Required positional:
        p1 (int): Prime number whose representation in the
                chosen base (without leading zeroes) should
                be a suffix of the returned value when expressed
                in the chosen base.
        p2 (int): Prime number which the solution must divide.

        Optional named:
        base (int): The base in which the solution and p2 are
                to be expressed when assessing whether the
                representation of p2 is a suffix of the
                representation of the solution.
            Default: 10
    
    Returns:
    Integer (int) giving the smallest positive multiple of
    p2 such that when expressed in the chosen base, the
    representation of p1 in that base without losing zeros
    is one of its suffixes, if such a value exists, otherwise
    -1.
    
    Outine of rationale:
    We can identify the solution by finding the smallest
    non-negative integer k that is a multiple of p2 and:
        k * p2 = p1 (mod base ** n_dig1)
    where n_dig1 is the number of digits without leading zeros
    in the representation of p1 in the chosen base. This is
    a linear congruence and can be solved via the extended
    Euclidean algorithm (see solveLinearCongruence() and
    extendedEuclideanAlgorithm()) as long as gcd(p2, base * n_dig1)
    divides p1 (which is guaranteed if p2 is larger than
    the largest prime factor of base), otherwise there
    is no solution.
    """
    p1_n_dig = 0
    p1_2 = p1
    md = 1
    while p1_2:
        p1_n_dig += 1
        p1_2 //= base
        md *= base
    #print(p1, p2, md)
    cong_sol = solveLinearCongruence(p2, p1, md)
    return -1 if cong_sol == -1 else cong_sol * p2

def primePairConnectionsSum(p1_min: int=5, p1_max: int=1_000_000, base: int=10) -> int:
    """
    Solution to Project Euler #134

    Finds the sum over all pairs of consecutive primes p1 and
    p2 (where p2 is the larger) with p1_min <= p1 <= p1_max of
    the smallest positive multiple of p2 which, when expressed
    in the chosen base, contains the expression of p1 in than
    base (without leading zeroes) as a suffix, if such a value
    exists (otherwise the term is 0).

    Args:
        Optional named:
        p1_min (int): The smallest possible value of p1 among
                the pairs of primes included in the sum.
            Default: 5
        p1_max (int): The largest possible value of p1 among
                the pairs of primes included in the sum.
            Default: 10^6
        base (int): The base in which for each (p1, p2) pair
                the sum term and p2 are to be expressed when
                assessing whether the representation of p2 is
                a suffix of the representation of the sum term.
            Default: 10
    
    Returns:
    Integer (int) giving the sum over all the pairs of
    consecutive primes p1, p2 (where p2 is the larger) with
    p1_min <= p1 <= p1_max of the smallest positive multiple of
    p2 which, when expressed in the chosen base, contains the
    expression of p1 in than base (without leading zeroes) as
    a suffix, if such a value exists (otherwise the term is 0).
    """
    #since = time.time()
    if p1_min > p1_max:
        #print(f"Time taken = {time.time() - since:.4f} seconds")
        return 0
    ps = PrimeSPFsieve()
    p_gen = ps.endlessPrimeGenerator()
    
    for p in p_gen:
        if p >= p1_min: break
    res = 0
    p1 = p
    for p2 in p_gen:
        if p1 > p1_max: break
        connect = primeConnection(p1, p2, base=base)
        res += 0 if connect == -1 else connect
        p1 = p2
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 135
def sameDifferences(n_max: int=999_999, target_count: int=10) -> int:
    """
    Solution to Project Euler #135

    Finds the number of strictly positive integers no greater than
    n_max for which there are exactly target_count distinct
    integer triples (a, b, c) such that:
     1) a, b and c are strictly positive
     2) a, b and c form an arithmetic progression (i.e. a - b = b - c)
     3) a^2 - b^2 - c^2 = n (where n is the integer in question)
    
    Args:
        Optional named:
        n_max (int): Strictly positive integer giving the largest
                value of n considered for the count.
            Default: 10^6 - 1
        target_count (int): Non-negative integer giving the exact
                number of distinct (a, b, c) triples there exists
                for a given integer n for it to be included in the
                count.
            Default: 10
    
    Returns:
    The number of strictly positive integers no greater than n_max
    such that there exists exactly target_count distinct positive
    integer triples that satisfy all of the stated conditions.
    """
    #since = time.time()
    counts = {}
    rng = [float("inf"), -float("inf")]
    for a in range(1, n_max + 1):
        for b in range((a >> 2) + 1, min(a, (((n_max // a) + a) >> 2) + 1)):
            num = a * (4 * b - a)
            rng[0] = min(rng[0], num)
            rng[1] = max(rng[1], num)
            counts[num] = counts.get(num, 0) + 1
    #print(counts)
    #print(rng)
    res = []
    for k, v in counts.items():
        if v != target_count: continue
        res.append(k)
    #print(sorted(res))
    #res.sort()
    #print([x >> 2 for x in res if not x & 3])
    #print([x >> 4 for x in res if not x & 15])
    #print([x >> 5 for x in res if not x & 31])
    #print([x for x in res if x & 1])
    #res = sum(x for x in counts.values() if x == target_count)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return len(res)

def singletonDifferences(n_max: int=49_999_999) -> int:
    """
    Solution to Project Euler #136

    Finds the number of strictly positive integers no greater than
    n_max for which there is one and only one  integer triple
    (a, b, c) such that:
     1) a, b and c are strictly positive
     2) a, b and c form an arithmetic progression (i.e. a - b = b - c)
     3) a^2 - b^2 - c^2 = n (where n is the integer in question)
    
    Args:
        Optional named:
        n_max (int): Strictly positive integer giving the largest
                value of n considered for the count.
            Default: 10^6 - 1
    
    Returns:
    The number of strictly positive integers no greater than n_max
    such that there exists one and only one positive integer triples
    that satisfies all of the stated conditions.

    Outline of rationale:
    Empirically, it appears that n has a unique solution
    if and only if:
     1) n is a prime congruent to 3 modulo 4
     2) n is 1 or an odd prime multiplied by 4 or 16
    TODO- prove this
    """
    #since = time.time()
    ps = PrimeSPFsieve(n_max=n_max)
    print("finished creating prime sieve")
    res = 0
    if n_max >= 4:
        res += 1
        if n_max >= 16: res += 1
    #next(p_gen)
    for p in ps.p_lst[1:]:
        if p > n_max: break
        if p & 3 == 3: res += 1
        if 4 * p <= n_max:
            res += 1
            if 16 * p <= n_max:
                res += 1
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 137
def modifiedFibonacciGoldenNuggetsList(n_nuggets: int, G0: int=0, G1: int=1) -> List[int]:
    """
    Identifies the smallest n_nuggets modified Fibonacci nuggets for initial
    Fibonacci-like sequence terms of G0 and G1.

    A Fibonacci-like sequence is a sequence for which:
        t_0 = G0,
        t_1 = G1,
        t_n = t_(n - 2) + t_(n - 1) for integer n >= 2
    
    A modified Fibonacci nugget for a Fibonacci-like sequence with intial terms
    G0 and G1 are the strictly positive integers such that there exists a real
    number x for which the chosen integer equals the series:
        A(x) = (sum n from 1 to inf) x ** n * t_n
    
    Args:
        n_nuggets (int)
    """
    # x ** 2 - D * y ** 2 = n
    res = []
    r = G1 + 2 * G0
    r_md = r % 5
    d = 4 * (G0 ** 2 + G0 * G1 - G1 ** 2)
    for (x, y) in generalisedPellSolutionGenerator(5, d, excl_trivial=False):
        #print((x, y))
        if x <= r or x % 5 != r_md: continue
        res.append((x - r) // 5)
        #print((x - 1) // 5)
        if len(res) >= n_nuggets: break
    return res

def modifiedFibonacciGoldenNugget(nugget_number: int=15, G1: int=1, G2: int=1) -> int:
    """
    Solution to Project Euler #137
    """
    #since = time.time()
    G0 = G2 - G1
    res = modifiedFibonacciGoldenNuggetsList(nugget_number, G0=G0, G1=G1)
    print(res)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res[-1]

# Problem 138
def specialIsocelesTriangles(n_smallest: int) -> List[Tuple[int, int, int]]:
    """
    Consider isoceles triangles with a base b and the two matching
    sides L. Let the height from the base be h. This finds the
    n_smallest triangles of this form with the smallest L for which
    L and b are integers and h is one greater or less than b.

    Args:
        Required positional:
        n_smallest (int): The number of triangles to be found.
    
    Returns:
    List of 3-tuples of integers (int), containing the dimensions
    of each of the triangles satisfying the given constraints with
    the n_smallest values of L in the form (b, h, L).
    These are given in order of increasing L.
    
    Outline of rationale:
    Consider a solution (b, h, L). Given that L and b are integers,
    and h is exactly one greater or less than b, h is an integer.
    Furthermore, since h, b / 2 and L form a right angled triangle,
    b / 2 must be an integer, so b is even. Consequently, h must
    be odd.
    Consider the right angled triangle (hb, h, L), where hb is
    half of b. As established, these are all integers and so form
    a Pythagorean triple, and additionally h is odd and equal to
    either (2 * hb + 1) or (2 * hb - 1).
    A Pythagorean triple can always be uniquely expressed as:
        (k * (m ** 2 - n ** 2), k * (2 * m * n), k * (m ** 2 + n ** 2))
    where the final length is the hypotenuse (corresponding to L),
    k, m and n are strictly positive integers, m > n, m and n are
    coprime and not both odd. As h must be odd, the value of k must
    be 1 and h must correspond to the first of these lengths (and
    therefore bh corresponds to the second). We therefore have:
        h = m ** 2 - n ** 2
        hb = 2 * m * n -> b = 4 * m * n
        L = m ** 2 + n ** 2
    for strictly positive coprime integers m and n that are not
    both odd with m > n.
    If we now set h to be one greater or less than hb, we get
    the equation:
        m ** 2 - n ** 2 = 4 * m * n +/- 1
    which can be rearranged to get:
        (m - 2 * n) ** 2 - 5 * n ** 2 = +/- 1
    This is a variant of Pell's equation (with the + option and
    Pell's negative equation with the - option) with D = 5.
    Note that for both versions of the equation, for positive
    m and n the larger n gets the larger also m gets, and
    consequently (given L = m ** 2 + n ** 2) the larger L gets.
    We can therefore find the smallest solutions (in terms of
    their value for L) by finding the smallest collective
    non-negative solutions of Pell's equation and Pell's negative
    equation:
        x ** 2 - 5 * y ** 2 = +/- 1
    such that when m = x + 2 * y and n = y, m and n are strictly
    positive coprime integers that are not both odd and m > n.
    This can be done using a standard technique with continued
    fractions for the square root of 5 (see
    pellSolutionGenerator()).
    Then, the corresponding triangle for the solution (m, n)
    is:
        h = m ** 2 - n ** 2
        b = 4 * m * n
        L = m ** 2 + n ** 2
    """
    gens = []
    mn_heap = []
    for j, neg in enumerate((False, True)):
        gen = pellSolutionGenerator(5, negative=neg)
        try:
            first = next(gen)
        except TypeError:
            continue
        gens.append(gen)
        heapq.heappush(mn_heap, (first, j))
    if not gens: return []
    res = []
    for _ in range(n_smallest):
        while True:
            pair, j = mn_heap[0]
            heapq.heappushpop(mn_heap, (next(gens[j]), j))
            n = pair[1]
            m = pair[0] + 2 * n
            if gcd(m, n) != 1 or m <= n or (m & 1 == 1 and n & 1 == 1):
                continue
            res.append((4 * m * n, m ** 2 - n ** 2, m ** 2 + n ** 2))
            break
    return res

def specialIsocelesTriangleSum(n_smallest_to_sum: int=12) -> int:
    """
    Solution to Project Euler #138

    Consider isoceles triangles with a base b and the two matching
    sides L. Let the height from the base be h. This finds the sum
    of the values of L over the n_smallest_to_sum triangles of this
    form with the smallest L for which L and b are integers and h is
    one greater or less than b.

    Args:
        Optional named:
        n_smallest (int): The number of triangles whose value of L
                is to be included in the sum.
            Default: 12
    
    Returns:
    Sum over the values of L for the triangles satisfying the given
    constraints with the n_smallest_to_sum values of L.
    
    Outline of rationale:
    See documentation for specialIsocelesTriangles()
    """
    #since = time.time()
    lst = specialIsocelesTriangles(n_smallest_to_sum)
    res = sum(x[2] for x in lst)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 139
def pythagoreanTripleGeneratorByHypotenuse(primitive_only: bool=False, max_hypotenuse: Optional[int]=None) -> Generator[Tuple[Tuple[int, int, int], bool], None, None]:
    """
    Generator iterating over Pythagorean triples, yielding them
    in order of increasing size of the hypotenuse (i.e. the largest
    value in the Pythagorean triple).

    Args:
        Optional named:
        primitive_only (bool): Boolean specifying whether to
                iterate only over primitive Pythagorean triples
                (i.e. those whose side lengths are coprime) or
                all Pythagorean triples, with True specifying
                only primitive Pythagorean triples are to be
                iterated over.
            Default: False
        max_hypotenuse (None or int): If given, specifies the
                largest possible value of the hypotenuse of
                any Pythagorean triple yielded.
                If this is not given or given as None, the
                iterator will not self-terminate, so any loop
                based around this iterator must contain a
                mechanism to break the loop (e.g. break or
                return) to avoid an infinite loop.
            Default: None
    
    Yields:
    2-tuple whose index 0 contains a 3-tuple of integers
    specifying the corresponding Pythagorean triple, with
    the 3 items ordered in increasing size (so the hypotenuse
    is last) and whose index 1 contains a boolean denoting
    whether this Pythagorean triple is primitive (with True
    indicating that it is primitive).
    The triples are yielded in order of increasing size of
    hypotenuse, with triples with the same hypotenuse yielded
    in increasing order of their next longest side.
    """
    m = 1
    heap = []
    if max_hypotenuse is None: max_hypotenuse = float("inf")
    while True:
        m += 1
        m_odd = m & 1
        n_mn = 1 + m_odd
        m_sq = m ** 2
        min_hyp = m_sq + n_mn ** 2
        while heap and heap[0][0][0] < min_hyp:
            ans = heapq.heappop(heap) if primitive_only or heap[0][0][0] + heap[0][1][0] > max_hypotenuse else heapq.heappushpop(heap, (tuple(x + y for x, y in zip(*heap[0][:2])), heap[0][1], False))
            yield (tuple(ans[0][::-1]), ans[2])
        if min_hyp > max_hypotenuse: break
        max_n = min(m - 1, isqrt(max_hypotenuse - m_sq)) if max_hypotenuse != float("inf") else m - 1
        # Note that since m and n are coprime and not both can be odd,
        # m and n must have different parity (as if they were both
        # even then they would not be coprime)
        for n in range(1 + m_odd, max_n + 1, 2):
            if gcd(m, n) != 1: continue
            a, b, c = m_sq - n ** 2, 2 * m * n, m_sq + n ** 2
            if b < a: a, b = b, a
            heapq.heappush(heap, ((c, b, a), (c, b, a), True))
    return

def pythagoreanTripleGeneratorByPerimeter(primitive_only: bool=False, max_perimeter: Optional[int]=None) -> Generator[Tuple[Tuple[int, int, int], int, bool], None, None]:
    """
    Generator iterating over Pythagorean triples, yielding them
    in order of increasing size of the perimeter (i.e. the sum
    over the three values in the Pythagorean triple).

    Args:
        Optional named:
        primitive_only (bool): Boolean specifying whether to
                iterate only over primitive Pythagorean triples
                (i.e. those whose side lengths are coprime) or
                all Pythagorean triples, with True specifying
                only primitive Pythagorean triples are to be
                iterated over.
            Default: False
        max_perimeter (None or int): If given, specifies the
                largest possible value of the perimeter of
                any Pythagorean triple yielded.
                If this is not given or given as None, the
                iterator will not self-terminate, so any loop
                based around this iterator must contain a
                mechanism to break the loop (e.g. break or
                return) to avoid an infinite loop.
            Default: None
    
    Yields:
    3-tuple whose index 0 contains a 3-tuple of integers
    specifying the corresponding Pythagorean triple, with
    the 3 items ordered in increasing size (so the hypotenuse
    is last), whose index 1 contains an integer giving the
    perimeter of this Pythagorean triple and whose index 2
    contains a boolean denoting whether this Pythagorean
    triple is primitive (with True indicating that it is
    primitive).
    The triples are yielded in order of increasing size of
    perimeter, with triples with the same perimeter yielded
    in increasing order of hypotenuse, with triples with the
    same perimeter and hypotenuse yielded in increasing order
    of their next longest side.
    """
    m = 1
    heap = []
    if max_perimeter is None: max_perimeter = float("inf")
    while True:
        m += 1
        m_odd = m & 1
        n_mn = 1 + m_odd
        m_sq = m ** 2
        min_perim = m * (m + n_mn)
        while heap and heap[0][0] < min_perim:
            new_perim = heap[0][0] + sum(heap[0][2])
            ans = heapq.heappop(heap) if primitive_only or new_perim > max_perimeter else heapq.heappushpop(heap, (new_perim, tuple(x + y for x, y in zip(*heap[0][1:3])), heap[0][2], False))
            yield (tuple(ans[1][::-1]), ans[0], ans[3])
        if min_perim > max_perimeter: break
        max_n = min(m - 1, max_perimeter // (2 * m) - m) if max_perimeter != float("inf") else m - 1
        for n in range(1 + m_odd, max_n + 1, 2):
            if gcd(m, n) != 1: continue
            a, b, c = m_sq - n ** 2, 2 * m * n, m_sq + n ** 2
            if b < a: a, b = b, a
            heapq.heappush(heap, ((a + b + c), (c, b, a), (c, b, a), True))
    return

def pythagoreanTiles(max_triangle_perimeter: int=99_999_999) -> int:
    """
    Solution to Project Euler #139

    For all right angled triangles with integer side lengths (the
    side lengths of which are known as a Pythagorean triple)
    with perimeter (i.e. the sum of the side lengths) no greater
    than max_triangle_perimeter, counts the number of these such
    that a square with side length equal to the hypotenuse of the
    triangle can be tiled exactly by square tiles with side length
    equal to the difference between the other two triangle side
    lengths.

    Args:
        Optional named:
        max_triangle_perimeter (int): Strictly positive integer
                specifying the largest perimeter (i.e. the sum of
                the side lengths) for the right angled triangles
                with integer side lengths considered.
            Default: 10 ** 8 - 1
    
    Returns:
    Integer giving the number of right angled triangles with
    integer side lengths and perimeter no greater than
    max_triangle_perimeter for which the described exact tiling
    is possible.

    Outline of rationale:
    We first observe that for a Pythagorean triple, a tiling
    is possible if and only if it is possible for its primitive
    (i.e. the Pythagorean triple created when all sides of that
    triple are divided by their collective greatest common
    divisor). We can therefore restrict our attention to the
    primitive Pythagorean triples, and for each primitive
    Pythagorean triple for which a tiling is possible, add the
    floor of max_triangle_perimeter divided by the triple's
    perimeter. This also means that we need only consider
    the primitive Pythagorean triples with a perimeter no
    greater than max_triangle_perimeter, as any with a larger
    perimeter will not contribute to the answer.

    Now, suppose (a, b, c) is a primitive Pythagorean triple
    with hypotenuse c (so that gcd(a, b, c) = 1). Note that a
    and b must have opposite parity, as if a and b are both even,
    then c is even so gcd(a, b, c) >= 2 and if a and b are both
    odd then a ** 2 + b ** 2 = 2 (mod 4) which is impossible.
    Additionally, note that gcd(a, b) = 1 if a prime divides
    both a and b then k also divides a ** 2 + b ** 2 = c ** 2
    and so must divide c, meaning that (a, b, c) is not a
    primitive Pythagorean triple.
    Suppose there exists a prime p that divides both (b - a)
    and c. Then p divides (b - a) ** 2 and so divides
    (b ** 2 + a ** 2 - 2 * a * b). But b ** 2 + a ** 2 = c ** 2
    and p divides c and so c ** 2. Therefore, p must divide
    2 * a * b. As p is prime, it must divide at least one of 2,
    a and b. If p divides 2 then p = 2, meaning (b - a) is even.
    But a and b have different parity so (b - a) must be odd, and
    thus p cannot divide 2. If p divides a then since p divides
    (b - a), p must also divide b, which contradicts the
    observation that gcd(a, b) = 1. Thus, p cannot divide a.
    Similarly, p cannot divide b. Consequently, p cannot divide
    2 * a * b, which is a contradiction. Therefore, there cannot
    exist a prime p that divides both (b - a) and c, implying
    that a tiling is only possible for a primitive Pythagorean
    triple if the difference between the two non-hypotenuse sides
    is not divisible by any prime, which can only be the case
    when the difference is +/- 1.
    Conversely, for any primitive Pythagorean triple whose
    non-hypotenuse sides differ by 1, the tile has dimensions
    1 x 1 and so trivially, this can exactly tile the c x c
    rectangle. Thus, a tiling is possible for a primitive
    Pythagorean triple if and only if its non-hypotenuse sides
    differ by exactly 1.

    A primitive Pythagorean triple can always be uniquely
    expressed as:
        ((m ** 2 - n ** 2), (2 * m * n), (m ** 2 + n ** 2))
    where the final length is the hypotenuse m and n are
    strictly positive integers, m > n, m and n are
    coprime and not both odd. Note that this has a perimeter
    of 2 * m * (m + n)
    As we require the non-hypotenuse sides to differ by 1
    we get:
        m ** 2 - n ** 2 = 2 * m * n +/- 1
    which can be rearranged to get:
        (m - n) ** 2 - 2 * n ** 2 = +/- 1
    This is a variant of Pell's equation (with the + option and
    Pell's negative equation with the - option) with D = 2.
    Note that for both versions of the equation, for positive
    m and n the larger n gets the larger also m gets, and
    consequently (given the perimeter is 2 * m * (m + n)) the
    larger the perimeter gets.
    We can therefore find all of the primitive Pythagorean
    triples for which a tiling is possible and whose perimeter
    is no greater than max_triangle_perimeter by finding the
    non-negative solutions of Pell's equation and Pell's negative
    equation:
        x ** 2 - 2 * y ** 2 = +/- 1
    such that when m = x + y and n = y, m and n are strictly
    positive coprime integers that are not both odd, m > n and
    2 * m * (m + n) <= max_triangle_perimeter.
    This can be done using a standard technique with continued
    fractions for the square root of 2 (see
    pellSolutionGenerator()).
    Then, as initially observed, for each of these primitive
    Pythagorean triples, the amount contributed to the sum by
    its multiples is then the floor of max_triangle_perimeter
    divided by the perimeter of the primitive Pythagorean
    triple (given by 2 * m * (m + n)).
    TODO- review wording of the explanation for the logic and
    clarity of the arguments
    """
    """
    since = time.time()
    res = 0
    for triple in pythagoreanTripleGeneratorByPerimeter(primitive_only=True, max_perimeter=max_triangle_perimeter):
        a, b, c = triple[0]
        perim = triple[1]
        if c % (b - a): continue
        cnt = max_triangle_perimeter // perim
        print((a, b, c), cnt)
        res += cnt
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
    """
    #since = time.time()
    res = 0
    for neg in (False, True):
        for x, y in pellSolutionGenerator(2, negative=neg):
            #print(x, y)
            m, n = x + y, y
            #print((m - n) ** 2 - 2 * n ** 2)
            perim = 2 * m * (m + n)
            if perim > max_triangle_perimeter: break
            if gcd(m, n) != 1 or m & 1 == n & 1: continue
            #print(sorted([m ** 2 - n ** 2, 2 * m * n, m ** 2 + n ** 2]))
            res += max_triangle_perimeter // perim
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
    
    
# Problem 140
def modifiedFibonacciGoldenNuggetSum(n_nugget_numbers: int=30, G1: int=1, G2: int=4) -> int:
    """
    Solution to Project Euler #140
    """
    #since = time.time()
    G0 = G2 - G1
    lst = modifiedFibonacciGoldenNuggetsList(n_nugget_numbers, G0=G0, G1=G1)
    res = sum(lst)
    #print(lst)
    #print(res)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 141
def squareProgressiveNumbers(n_max: int) -> List[Tuple[int, int, Tuple[int, int, int]]]:
    """
    
    Review- prove and optimise
    """
    m_max = isqrt(n_max)
    a_max = integerNthRoot(n_max, 3)
    res = []
    for a in range(1, a_max + 1):
        for b in range(1, a):
            if gcd(a, b) != 1: continue
            r_max = (isqrt(b ** 4 + 4 * a ** 3 * b * n_max) - b ** 2) // (2 * a ** 3 * b)
            for r_ in range(1, r_max + 1):
                m_sq = r_ ** 2 * a ** 3 * b + r_ * b ** 2
                #if m_sq > n_max: continue
                m = isqrt(m_sq)
                if m ** 2 != m_sq: continue
                res.append((m, m_sq, (r_ * a * b, r_ * a ** 2, r_ * b ** 2)))
                print(a, b, r_, res[-1])
                #sqrt_r_ = isqrt(r_)
                #print(r_, sqrt_r_, sqrt_r_ ** 2 == r_)
    return sorted(res)

def squareProgressiveNumbersSum(n_max: int=10 ** 12 - 1) -> int:
    """
    Solution to Project Euler #141
    """
    #since = time.time()
    nums = {x[1] for x in squareProgressiveNumbers(n_max)}
    res = sum(nums)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

#Problem 142
def perfectSquareCollectionGenerator() -> Generator[Tuple[int, Tuple[int, int, int]], None, None]:
    #ps = PrimeSPFsieve()
    # TODO- find a less brute force solution. Look up Rolle's puzzle
    seen = {}
    candidates_heap = []
    for triple in pythagoreanTripleGeneratorByHypotenuse(primitive_only=False, max_hypotenuse=None):
        a, b, c = triple[0]
        #print(c)
        while candidates_heap and c + 3 >= candidates_heap[0][0]:
            yield heapq.heappop(candidates_heap)
        for (x, y, m) in ((c, a, b), (c, b, a)):
            num1 = x + y
            rt1 = isqrt(num1)
            if rt1 ** 2 != num1: continue
            seen.setdefault(x, set())
            seen[x].add(y)
            #print((x, y))
            if y not in seen.keys(): continue
            for z in seen[y].intersection(seen[x]):
                print((x + y + z), (x, y, z))
                heapq.heappush(candidates_heap, ((x + y + z), (x, y, z)))
    return

def perfectSquareCollection() -> int:
    """
    Solution to Project Euler #142
    """
    #since = time.time()
    #for coll in perfectSquareCollectionGenerator():
    #    break
    res = next(perfectSquareCollectionGenerator())[0]#coll[0]
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 143
def torricelliTriangleUniqueLengthSum(sm_max: int=12 * 10 ** 4) -> int:
    """
    Solution to Project Euler Problem 143

    For triangles whose angles are all strictly less than 120 degrees,
    the Fermat-Toricelli point is the point on the interior of the
    triangle for which the sum of the distances to each of the vertices
    of the triangle is minimised.

    We define a Toricelli triangle to be a triangle whose angles are
    all strictly less than 120 degrees, whose sides are all integer
    lengths for which all of the distances from the Fermat-Toricelli
    point are integers.

    This function considers all distinct Toricelli triangles for which
    the sum of the distances from the Fermat-Toricelli point to the
    three vertices of the triangle is no greater than sm_max and
    returns the sum over these sums of distances for all such
    Toricelli triangles.

    Two Toricelli triangles are considered distinct if and only if
    the sorted lists of their respective side lengths are distinct
    (so for instance a Toricelli triangle constructed by reflecting
    another Toricelli triangle is not considered to be distinct
    from the Toricelli triangle from which it was reflected).

    Args:
        Optional named:
        sm_max (int): Non-negative integer giving the largest sum
                of distances from the Fermat-Toricelli point to the
                three triangle vertices for the Toricelli triangles
                considered.
            Default: 12 * 10 ** 4
    
    Returns:
    Integer (int) giving the total of all sums of distances from the
    Fermat-Toricelli point to the three vertices of the triangle for
    all distinct Toricelli triangles considered (that being all such
    triangles for which this distance does not exceed sm_max).

    Outline of rationale:
    It can be shown that for any triangle with all angles strictly
    less than 120 degrees, the triangle whose vertices are two
    vertices of the triangle and the third is the Fermat-Toricelli
    point of the triangle, the angle at the Fermat-Toricelli point
    is equal to 120 degrees.
    Consider the following construction and labelling based on such
    a triangle. Label the sides a, b and c, and add straight lines
    from the Fermat-Toricelli point to each of the three vertices,
    giving the lines to the vertex opposite a, b and c with the
    labels p, q and r respectively.
    We are interested in Toricelli triangles, which are precisely
    those triangles with all angles strictly less than 120 degrees
    for which this construction yields integer lengths of the
    sides labelled a, b, c, p, q and r are all integers.
    Using our initial observation, by the cosine rule (given that
    cos(120 degrees) is minus one half), we find:
        a ** 2 = q ** 2 + r ** 2 + q * r
        b ** 2 = p ** 2 + r ** 2 + p * r
        c ** 2 = p ** 2 + q ** 2 + p * q
    TODO- complete outline of rationale
    """
    #since = time.time()
    m_mx = isqrt(sm_max) - 1
    seen = {}
    lengths = set()
    for m in range(2, m_mx + 1):
        n_mx = min(m - 1, (sm_max - m ** 2 - 1) // (2 * m))
        for n in range(1, n_mx + 1):
            if gcd(m, n) != 1: continue
            x0, y0, z0 = n * (2 * m + n), m ** 2 - n ** 2, m ** 2 + n ** 2 + m * n
            #print(x0, y0, z0)
            if gcd(gcd(x0, y0), z0) != 1: continue
            for mult in range(1, (sm_max // (x0 + y0)) + 1):
                x, y, z = x0 * mult, y0 * mult, z0 * mult
                if y > x: x, y = y, x
                for y2, z2 in seen.get(x, {}).items():
                    y1_, y2_ = sorted([y, y2])
                    if y1_ not in seen.get(y2_, {}).keys(): continue
                    #triangle = sorted([z, z2, seen[y2_][y1_]])
                    #if gcd(gcd(triangle[0], triangle[1]), triangle[2]) != 1: continue
                    #yield tuple(triangle)
                    #print(tuple(triangle))
                    length = x + y + y2
                    if length <= sm_max:
                        lengths.add(x + y + y2)

                seen.setdefault(x, {})
                seen[x][y] = z
                seen.setdefault(y, {})
                seen[y][x] = z
    #print(seen)
    res = sum(lengths)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res


# Problem 144
def ellipseInternalNorm(
    ellipse: Tuple[int, int, int],
    pos: Tuple[Tuple[int, int], Tuple[int, int]]
) -> Tuple[int, int]:
    """
    Given a rational ellipse in the x-y plane with its semi-major
    axes parallel to the x and y axes giving by the equation:
        ellipse[0] * x ** 2 + ellipse[1] * y ** 2 = ellipse[2]
    and a point at position in Cartesian coordinates:
        (pos[0][0] / pos[0][1], pos[1][0] / pos[1][1])
    on the ellipse, finds a vector in Cartesian coordinates
    with integer coefficients normal to the ellipse at that
    point, pointing towards the interior of the ellipse.
    Note that the returned vector is not in general normalized.

    Args:
        Required positional:
        ellipse (3-tuple of ints): 3 integers specifying the
                equation of the ellipse as shown above.
        pos (2-tuple of 2-tuples of ints): Two fractions, given
                as 2-tuples of ints (numerator then denominator)
                specifying the point on the ellipse in Cartesian
                coordinates.
    
    Returns:
    2-tuple of integers (ints) giving a normal vector for the
    ellipse at the given point expressed Cartesian coordinates
    with integer coefficients, pointing to the interior of the
    ellipse.
    """
    
    # Pointing into the ellipse
    return (-ellipse[0] * pos[0][0] * pos[1][1], -ellipse[1] * pos[0][1] * pos[1][0])

def otherRationalEllipseIntersection(
    ellipse: Tuple[int, int, int],
    pos: Tuple[Tuple[int, int], Tuple[int, int]],
    vec: Tuple[Tuple[int, int], Tuple[int, int]]
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Given a rational point on a rational ellipse and a vector with a
    rational Cartesian representation, for a line parallel to that
    vector intersecting that ellipse at that rational point, identifies
    the other point at which the line and ellipse intersect.
    """
    #print(pos, vec)
    A, B, C = ellipse
    x0, y0 = pos
    beta, alpha = vec
    #pos_f = (x0[0] / x0[1], y0[0] / y0[1])
    #vec_f = (alpha[0] / alpha[1], beta[0] / beta[1])
    #print(f"pos_f = {pos_f}, vec_f = {vec_f}")
    #x_f = ((B * vec_f[0] ** 2 - A * vec_f[1] ** 2) * pos_f[0] - 2 * B * vec_f[0] * vec_f[1] * pos_f[1]) / (A * vec_f[1] ** 2 + B * vec_f[0] ** 2)
    #print(f"x float = {x_f}")
    #y_f = ((A * vec_f[1] ** 2 - B * vec_f[0] ** 2) * pos_f[1] - 2 * A * vec_f[1] * vec_f[0] * pos_f[0]) / (A * vec_f[1] ** 2 + B * vec_f[0] ** 2)
    #print(f"y float = {y_f}")
    denom = addFractions((B * alpha[0] ** 2, alpha[1] ** 2), (A * beta[0] ** 2, beta[1] ** 2))
    #print(f"denom: {denom[0] / denom[1]} vs {(A * vec_f[1] ** 2 + B * vec_f [0] ** 2)}")
    x0_term_x = multiplyFractions(addFractions((B * alpha[0] ** 2, alpha[1] ** 2), (-A * beta[0] ** 2, beta[1] ** 2)), x0)
    #print(addFractions((B * alpha[0] ** 2, alpha[1] ** 2), (-A * beta[0] ** 2, beta[1] ** 2)), x0)
    #print((B * vec_f[0] ** 2 - A * vec_f[1] ** 2), pos_f[0])
    #print(f"x0_term_x: {x0_term_x[0] / x0_term_x[1]} vs {(B * vec_f[0] ** 2 - A * vec_f[1] ** 2) * pos_f[0]}")
    y0_term_x = multiplyFractions((-2 * B * alpha[0] * beta[0], alpha[1] * beta[1]), y0)
    #print(f"y0_term_x: {y0_term_x[0] / y0_term_x[1]} vs {- 2 * B * vec_f[0] * vec_f[1] * pos_f[1]}")
    numer_x = addFractions(x0_term_x, y0_term_x)
    x = multiplyFractions(numer_x, (denom[1], denom[0]))
    x0_term_y = multiplyFractions((-2 * A * alpha[0] * beta[0], alpha[1] * beta[1]), x0)
    y0_term_y = multiplyFractions(addFractions((A * beta[0] ** 2, beta[1] ** 2), (-B * alpha[0] ** 2, alpha[1] ** 2)), y0)
    numer_y = addFractions(x0_term_y, y0_term_y)
    y = multiplyFractions(numer_y, (denom[1], denom[0]))
    return (x, y)

def nextEllipseReflectedRay(
    ellipse: Tuple[int, int, int],
    pos: Tuple[Tuple[int, int], Tuple[int, int]],
    vec: Tuple[Tuple[int, int], Tuple[int, int]]
) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int]]]: 

    norm = ellipseInternalNorm(ellipse, pos)
    #print(f"norm = {norm}")
    norm_mag_sq = sum(x * x for x in norm)
    dot_prod = addFractions(*[(y * x[0], x[1]) for x, y in zip(vec, norm)])
    mult = multiplyFractions(dot_prod, (2, norm_mag_sq))
    add_vec = tuple((-x * mult[0], mult[1]) for x in norm)
    #print(f"add_vec = {add_vec}")
    vec2 = tuple(addFractions(x, y) for x, y in zip(vec, add_vec))
    pos2 = otherRationalEllipseIntersection(ellipse, pos, vec2)
    return (pos2, vec2)

def laserBeamEllipseReflectionPointGenerator(
    ellipse: Tuple[int, int, int],
    pos0: Tuple[Tuple[int, int], Tuple[int, int]],
    reflect1: Tuple[Tuple[int, int], Tuple[int, int]]
) -> Generator[Tuple[Tuple[int, int], Tuple[int, int]], None, None]:
    pos0_neg = ((-pos0[0][0], pos0[0][1]), (-pos0[1][0], pos0[1][1]))
    #print(reflect1, pos0_neg)
    vec = tuple(addFractions(x, y) for x, y in zip(reflect1, pos0_neg))
    #print(f"vec0 = {vec} = {(vec[0][0] / vec[0][1], vec[1][0] / vec[1][1])}")
    #print(f"reflect1 = {reflect1} = {(reflect1[0][0] / reflect1[0][1], reflect1[1][0] / reflect1[1][1])}")
    
    pos = reflect1
    while True:
        pos, vec = nextEllipseReflectedRay(ellipse, pos, vec)
        #print(f"vec = {vec} = {(vec[0][0] / vec[0][1], vec[1][0] / vec[1][1])}")
        #print(f"pos = {pos} = {(pos[0][0] / pos[0][1], pos[1][0] / pos[1][1])}")
        
        yield pos
    return

def ellipseInternalNormFloat(
    ellipse: Tuple[int, int, int],
    pos: Tuple[float, float]
) -> Tuple[float]:
    """
    Given a rational ellipse in the x-y plane with its semi-major
    axes parallel to the x and y axes giving by the equation:
        ellipse[0] * x ** 2 + ellipse[1] * y ** 2 = ellipse[2]
    and a point at position in Cartesian coordinates:
        (pos[0], pos[1])
    on the ellipse, finds a vector in Cartesian coordinates
    normal to the ellipse at that point, pointing towards the
    interior of the ellipse.
    Note that the returned vector is not in general normalized.

    Args:
        Required positional:
        ellipse (3-tuple of ints): 3 integers specifying the
                equation of the ellipse as shown above.
        pos (2-tuple of 2-tuples of ifloats): The point on the
                ellipse in Cartesian coordinates.
    
    Returns:
    2-tuple of floats giving a normal vector for the ellipse at
    the given point expressed Cartesian coordinates, pointing to
    the interior of the ellipse.
    """
    
    # Pointing into the ellipse
    return (-ellipse[0] * pos[0], -ellipse[1] * pos[1])

def otherEllipseIntersectionFloat(
    ellipse: Tuple[int, int, int],
    pos: Tuple[float, float],
    vec: Tuple[float, float]
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Given a rational point on a rational ellipse and a vector with a
    rational Cartesian representation, for a line parallel to that
    vector intersecting that ellipse at that rational point, identifies
    the other point at which the line and ellipse intersect.
    """
    #print(pos, vec)
    A, B, C = ellipse
    x0, y0 = pos
    beta, alpha = vec
    #pos_f = (x0[0] / x0[1], y0[0] / y0[1])
    #vec_f = (alpha[0] / alpha[1], beta[0] / beta[1])
    #print(f"pos_f = {pos_f}, vec_f = {vec_f}")
    #x_f = ((B * vec_f[0] ** 2 - A * vec_f[1] ** 2) * pos_f[0] - 2 * B * vec_f[0] * vec_f[1] * pos_f[1]) / (A * vec_f[1] ** 2 + B * vec_f[0] ** 2)
    #print(f"x float = {x_f}")
    #y_f = ((A * vec_f[1] ** 2 - B * vec_f[0] ** 2) * pos_f[1] - 2 * A * vec_f[1] * vec_f[0] * pos_f[0]) / (A * vec_f[1] ** 2 + B * vec_f[0] ** 2)
    #print(f"y float = {y_f}")
    denom = (B * alpha ** 2) + (A * beta ** 2)
    #print(f"denom: {denom[0] / denom[1]} vs {(A * vec_f[1] ** 2 + B * vec_f [0] ** 2)}")
    x0_term_x = (B * alpha ** 2 - A * beta ** 2) * x0
    y0_term_x = -2 * B * alpha * beta * y0
    numer_x = x0_term_x + y0_term_x
    x = numer_x / denom
    x0_term_y = (-2 * A * alpha * beta) * x0
    y0_term_y = (A * beta ** 2 - B * alpha ** 2) * y0
    numer_y = x0_term_y + y0_term_y
    y = numer_y / denom
    return (x, y)

def nextEllipseReflectedRayFloat(
    ellipse: Tuple[int, int, int],
    pos: Tuple[float, float],
    vec: Tuple[float, float]
) -> Tuple[Tuple[float, float], Tuple[float, float]]: 

    norm = ellipseInternalNormFloat(ellipse, pos)
    #print(f"norm = {norm}")
    norm_mag_sq = sum(x * x for x in norm)
    dot_prod = sum((y * x) for x, y in zip(vec, norm))
    mult = dot_prod * 2 / norm_mag_sq
    add_vec = tuple((-x * mult) for x in norm)
    #print(f"add_vec = {add_vec}")
    vec2 = tuple(x + y for x, y in zip(vec, add_vec))
    pos2 = otherEllipseIntersectionFloat(ellipse, pos, vec2)
    return (pos2, vec2)

def laserBeamEllipseReflectionPointFloatGenerator(
    ellipse: Tuple[int, int, int],
    pos0: Tuple[float, float],
    reflect1: Tuple[float, float]
) -> Generator[Tuple[float, float], None, None]:
    pos0_neg = (-pos0[0], -pos0[1])
    #print(reflect1, pos0_neg)
    vec = tuple(x + y for x, y in zip(reflect1, pos0_neg))
    #print(f"vec0 = {vec} = {(vec[0][0] / vec[0][1], vec[1][0] / vec[1][1])}")
    #print(f"reflect1 = {reflect1} = {(reflect1[0][0] / reflect1[0][1], reflect1[1][0] / reflect1[1][1])}")
    
    pos = reflect1
    while True:
        pos, vec = nextEllipseReflectedRayFloat(ellipse, pos, vec)
        #print(f"vec = {vec} = {(vec[0][0] / vec[0][1], vec[1][0] / vec[1][1])}")
        #print(f"pos = {pos} = {(pos[0][0] / pos[0][1], pos[1][0] / pos[1][1])}")
        
        yield pos
    return

def laserBeamEllipseReflectionCount(
    ellipse: Tuple[int, int, int]=(4, 1, 100),
    pos0: Tuple[Tuple[int, int], Tuple[int, int]]=((0, 1), (101, 10)),
    reflect1: Tuple[Tuple[int, int], Tuple[int, int]]=((7, 5), (-48, 5)),
    x_window: Tuple[Tuple[int, int], Tuple[int, int]]=((-1, 100), (1, 100)),
    use_float: bool=True
) -> int:
    """
    Solution to Project Euler #144

    Note that this solves the problem very quickly if use_float is set to
    True, even though this means that we cannot be completely confident
    in the answer due to rounding errors. If use_float is set to False then
    the solution is exact, though this results in fractions whose numberator
    and denominator that get larger with each reflection (it appears to be
    exponentially so), rapidly resulting in unmanageably large integers
    which not only makes it extremely slow, but also likely to exhaust all
    memory resources. The exact version has never been run to the extent
    that it solves for the parameters used in the Project Euler problem
    due to the extreme slow down (it reached reflection 56 before being
    abandoned). However, the option to solve exactly has been left in
    to illustrate that at least in theory, an exact solution in rationals
    is possible.
    """
    #since = time.time()
    #print(reflect1)
    res = 1
    closest = float("inf")
    if use_float:
        x_window_float = tuple(x[0] / x[1] for x in x_window)
        #print(x_window_float)
        pos0_float = tuple(x[0] / x[1] for x in pos0)
        reflect1_float = tuple(x[0] / x[1] for x in reflect1)
        for pos in laserBeamEllipseReflectionPointFloatGenerator(ellipse, pos0_float, reflect1_float):
            #if res == 354:
            #    print(f"354 result: {pos}")
            #if res < 10:
            #    print(res, pos)
            if pos[1] > 0 and pos[0] >= x_window_float[0] and pos[0] <= x_window_float[1]:
                break
            
            res += 1
            
            #dist = min(abs(pos[0] - x_window_float[0]), abs(pos[0] - x_window_float[1]))
            #if pos[1] > 0 and dist < closest:
            #    closest = dist
            #    print(f"closest yet on reflection {res} at distance of {dist} at position {pos}")
            
            #print(f"n_reflections = {res}")
    else:
        for pos in laserBeamEllipseReflectionPointGenerator(ellipse, pos0, reflect1):
            if pos[1][0] > 0 and pos[0][0] * x_window[0][1] >= pos[0][1] * x_window[0][0] and pos[0][0] * x_window[1][1] <= pos[0][1] * x_window[1][0]:
                break
            res += 1
            #print(f"n_reflections = {res}")
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 145
def nDigitReversibleNumbersCount(n_dig: int, base: int=10) -> int:
    """
    Calculates the number of integers which, when expressed in the
    chosen base are n_dig digits long without leading zeros, do not
    end with a 0 and when added to the number consisting of the same
    digits reversed (without leading zeros) when expressed in the
    chosen base results in an integer which when expressed in the
    chosen base contains only odd digits.

    Args:
        Required positional:
        n_dig (int): Strictly positive integer giving the number
                of digits in the integers considered when expressed
                in the chosen base
        
        Optional named:
        base (int): Integer strictly greater than 1 giving the base
                in which the integers are expressed, in particular
                when reversing the digits of the integer and
                assessing whether the sum consists only of odd
                digits.
    
    Returns:
    Integer (int) giving the number of integers which when expressed
    in the chosen base consists of n_dig digits without leading
    zeros that satisfy the stated requirements.
    """
    # TODO- look into the closed form solution, and adapt it to  work
    # with odd bases

    n_dig_hlf = n_dig >> 1
    odd_n_dig = n_dig & 1
    base_odd = base & 1
    tail = []
    memo = {}
    def recur(idx: int=0, head_needs_carry: bool=False, carry: bool=False) -> int:
        if idx == n_dig_hlf:
            if not odd_n_dig: return head_needs_carry == carry
            if not base_odd:
                #print(idx, head_needs_carry, carry)
                if not carry: return 0
                return base >> 1
            else:
                if carry != head_needs_carry: return 0
                if not carry: return (base >> 1) + 1
                return (base >> 1) - (not idx)

        args = (idx, head_needs_carry, carry)
        if args in memo.keys(): return memo[args]

        tail_d = tail[idx]

        res = 0
        if carry:
            cnt10 = 0
            cnt11 = 0
            # TODO- try to find a closed form expression for the
            # counts without looping over the digit options
            for d in range(not idx, base):
                d_wo_carry = tail_d + d
                if base & 1:
                    if d_wo_carry < base - 1:
                        needs_carry = not d_wo_carry & 1
                    elif d_wo_carry >= base:
                        needs_carry = d_wo_carry & 1
                    else: continue
                else:
                    needs_carry = not d_wo_carry & 1
                if head_needs_carry and tail_d + d + needs_carry < base:
                    continue
                elif not head_needs_carry and tail_d + d >= base:
                    continue
                carries = tail_d + d + carry >= base
                if carries: cnt11 += needs_carry
                else: cnt10 += needs_carry
            if cnt10: res += cnt10 * recur(idx=idx + 1, head_needs_carry=True, carry=False)
            if cnt11: res += cnt11 * recur(idx=idx + 1, head_needs_carry=True, carry=True)
            memo[args] = res
            return res
        cnt00 = 0
        cnt01 = 0
        for d in range(not idx, base):
            d_wo_carry = tail_d + d
            if base & 1:
                if d_wo_carry < base - 1:
                    needs_carry = not d_wo_carry & 1
                elif d_wo_carry >= base:
                    needs_carry = d_wo_carry & 1
                else: continue
            else:
                needs_carry = not d_wo_carry & 1
            if head_needs_carry and tail_d + d + needs_carry < base:
                continue
            elif not head_needs_carry and tail_d + d >= base:
                continue
            carries = tail_d + d + carry >= base
            if carries: cnt01 += not needs_carry
            else: cnt00 += not needs_carry
        #print(tail, args, cnt00, cnt01, cnt10, cnt11)
        
        if cnt00 and not carry: res += cnt00 * recur(idx=idx + 1, head_needs_carry=False, carry=False)
        if cnt01 and not carry: res += cnt01 * recur(idx=idx + 1, head_needs_carry=False, carry=True)
        #if cnt10 and carry: res += cnt10 * recur(idx=idx + 1, head_needs_carry=True, carry=False)
        #if cnt11 and carry: res += cnt11 * recur(idx=idx + 1, head_needs_carry=True, carry=True)
        memo[args] = res
        return res
        """
        for opt in range(not idx, base):
            d_wo_carry = tail_d + opt
            if base & 1:
                if d_wo_carry < base - 1:
                    needs_carry = not d_wo_carry & 1
                elif d_wo_carry >= base:
                    needs_carry = d_wo_carry & 1
                else: continue
            else:
                needs_carry = not d_wo_carry & 1
            if head_needs_carry and tail_d + opt + needs_carry < base:
                continue
            elif not head_needs_carry and tail_d + opt >= base:
                continue
            carries = tail_d + opt + carry >= base
            if carries:
                if needs_carry: cnt11 += 1
                else: cnt01 += 1
            else:
                if needs_carry: cnt10 += 1
                else: cnt00 += 1
        #print(tail, args, cnt00, cnt01, cnt10, cnt11)
        
        if cnt00 and not carry: res += cnt00 * recur(idx=idx + 1, head_needs_carry=False, carry=False)
        if cnt01 and not carry: res += cnt01 * recur(idx=idx + 1, head_needs_carry=False, carry=True)
        #if cnt10 and carry: res += cnt10 * recur(idx=idx + 1, head_needs_carry=True, carry=False)
        #if cnt11 and carry: res += cnt11 * recur(idx=idx + 1, head_needs_carry=True, carry=True)
        memo[args] = res
        return res
        """
            

    res = 0
    for tail_num in range(1, base ** n_dig_hlf):
        if not tail_num % base: continue
        tail_num2 = tail_num
        
        tail = []
        for _ in range(n_dig_hlf):
            tail_num2, d = divmod(tail_num2, base)
            tail.append(d)
        memo = {}
        res += recur(idx=0, head_needs_carry=False, carry=False) + recur(idx=0, head_needs_carry=True, carry=False)
    return res

    """
    n_dig_hlf = -((-n_dig) >> 1)
    odd = n_dig & 1

    def recur(tail: List[int], head: List[int], carry: bool=False) -> int:
        if len(tail) == len(head) + odd:
            if odd:
                if not carry: return 0
                carry = (tail[-1] << 1) + 1 >= base
            for i in reversed(range(len(head))):
                carry, d = divmod(tail[i] + head[i] + carry, base)
                if not d & 1: return 0
            return 1
        d_odd = not (tail[len(head)] + carry) & 1
        start = d_odd
        if not start and not head:
            start = 2
        head.append(0)
        res = 0
        for d in range(start, base, 2):
            head[-1] = d
            carry = d + tail[len(head) - 1] + carry >= base
            res += recur(tail, head, carry=carry)
        head.pop()
        return res
            

    res = 0
    for tail_num in range(1, base ** n_dig_hlf):
        if not tail_num % base: continue
        tail_num2 = tail_num
        tail = []
        for _ in range(n_dig_hlf):
            tail_num2, d = divmod(tail_num2, base)
            tail.append(d)
        res += recur(tail, [], carry=False)
    return res
    """

def reversibleNumbersCount(n_dig_max: int=9, base: int=10) -> int:
    """
    Solution to Project Euler #145

    Calculates the number of integers which, when expressed in the
    chosen base are no more than n_dig digits long without leading
    zeros, do not end with a 0 and when added to the number
    consisting of the same digits reversed (without leading zeros)
    when expressed in the chosen base results in an integer which
    when expressed in the chosen base contains only odd digits.

    Args:
        Required positional:
        n_dig (int): Strictly positive integer giving the largest
                number of digits in the integers considered when
                expressed in the chosen base
        
        Optional named:
        base (int): Integer strictly greater than 1 giving the base
                in which the integers are expressed, in particular
                when reversing the digits of the integer and
                assessing whether the sum consists only of odd
                digits.
    
    Returns:
    Integer (int) giving the number of integers which when expressed
    in the chosen base consists of at most n_dig digits without
    leading zeros that satisfy the stated requirements.
    """
    #since = time.time()
    res = 0
    for n_dig in range(1, n_dig_max + 1):
        ans = nDigitReversibleNumbersCount(n_dig, base=base)
        #print(f"n_dig = {n_dig}, count = {ans}")
        res += ans
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 146
def investigatingAPrimePatternList(n_max: int=150 * 10 ** 6 - 1, add_nums: List[int]=[1, 3, 7, 9, 13, 27]) -> List[int]:
    """
    For a given list of strictly positive integers add_nums in
    strictly increasing order finds the non-negative integers n
    no greater than n_max such that the sequence with terms:
        t_i = n ** 2 + a[i - 1]
    for integers 1 <= i <= len(a) is a sequence of consecutive
    prime numbers.

    Args:
        Optional named:
        n_max (int): Strictly positive integer specifying the
                largest value of n considered.
            Default: 150 * 10 ** 6 - 1
        add_nums (list of ints): List of strictly positive
                integers in strictly increasing order, for which
                an integer is included in the returned list if
                and only if squaring that integer and adding the
                values in this list individually in order results
                in a sequence of consecutive prime numbers.
            Default: [1, 3, 7, 9, 13, 27]

    Returns:
    List of integers (ints) containing all the strictly positive
    integers no greater than n_max such that when squared and
    added to the integers in add_nums individually in order, the
    result is a sequence of consecutive prime numbers.

    Example:
        >>> investigatingAPrimePatternList(n_max=500000, add_nums=[1, 3, 7, 9, 13, 27])
        [10, 315410]

        This signifies that the only strictly positive integers
        no greater than 500000 for which the integer squared plus
        each of the numbers in the list [1, 3, 7, 9, 13, 27] in
        order results in a sequence of consecutive primes are
        10 and 315410. For 10, this sequence is:
            101, 103, 107, 109, 113, 127
        which can readily be verified to be a sequence of
        consecutive primes, while for 315410 this sequence is:
            99483468101, 99483468103, 99483468107, 99483468109,
            99483468113, 99483468127
        which can also (somewhat more arduously) be confirmed
        as a sequence of consecutive primes.

    Brief outline of rationale:
    For smaller primes, we create a filter that restricts the
    search, finding those values modulo the product of the
    smaller primes considered (utilising the Chinese remainder
    theorem) such that the number squared plus any of the values
    in add_values is not 0 modulo any of the smalelr primes
    considered. For each such number, integers between the
    maximum and minimum values of add_nums not in add_nums that
    the number squared plus that integer is not 0 modulo any
    of the smaller primes considered, and thus may result in
    a prime number. This reduces the search space considerably.
    
    A full check over values which when squared and added
    to a value in add_nums may be equal to one of the smaller
    primes considered in the previous step (and so potentially
    have been erroneously filtered out by that step) is then
    performed, appending each value for which all numbers
    required to be prime are prime and no numbers in between
    are prime to the final result.

    Finally, all of the larger values (i.e. those not checked
    by the previous step) that passed the filter are checked
    to ensure that all the numbers required to be prime are
    prime and none of the numbers in between (with candidates
    identified in the filter stage to reduce the number of
    checks needed) are prime, appending the values for which
    this occurs to the final result.

    The prime checks are performed using an implementation
    of the Miller-Rabin test to identify definite non-primes
    (see documentation of millerRabinPrimalityTest() method
    of the PrimeSPFsieve class), with cases where none of
    the values required to be prime have been ruled as
    non-prime having a full check performed, using a prime
    sieve to check divisibility by primes up to 10 ** 6 and
    then (for numbers greater than 10 ** 12) checking
    divisibility by odd numbers from 10 ** 6 + 1 up to
    the square root of the candidate for those numbers
    required to be prime, and further Miller-Rabin tests
    followed by (if necessary) full prime checks for numbers
    between those required to be prime not already determined
    to be non-prime (e.g. by the filter). At each test,
    if the test is failed (i.e. a number required to be
    prime is definitely non-prime or vice versa), the
    remaining tests scheduled are immediately abandoned
    to save unnecessary calculation.
    """

    # Try to make more efficient
    ps = PrimeSPFsieve(min(10 ** 6, n_max))
    add_nums_set = set(add_nums)
    mx_add = max(add_nums)
    mn_add = min(add_nums)
    curr_md = 1
    r_lst = [(0, set(range(mx_add)).difference(set(add_nums)))]
    filter_p_max = 4 * mx_add
    for p in ps.p_lst:
        if p > filter_p_max: break
        #print(f"p = {p}")
        prev_r_lst = r_lst
        r_lst = []
        for r in range(p):
            #if p == 11:
            #    print(f"r = {r}")
            r_sq_md = r ** 2 % p
            num0 = (-r_sq_md) % p
            neg_chk_rm_set = set()
            for num in range(num0, mx_add + 1, p):
                if num in add_nums_set: break
                neg_chk_rm_set.add(num)
            else:
                for r0, neg_chk_set in prev_r_lst:
                    for r2 in range(r0, min(curr_md * p, n_max + 1), curr_md):
                        if r2 % p == r:
                            break
                    else: continue
                    r_lst.append((r2, neg_chk_set.difference(neg_chk_rm_set)))
        #r_lst.sort()
        #print(f"p = {p}, curr_md = {curr_md}, number of remainders = {len(r_lst)}")
        #print(r_lst[:min(5, len(r_lst))])
        if curr_md <= n_max: curr_md *= p
    r_lst.sort()
    #print(r_lst[:5])
    #print(len(r_lst))
    #print(f"modulus = {curr_md}, number of remainders to check = {len(r_lst)}")
    #print(r_lst)

    # To enusure not ruling out a number for having a number divisible
    # by one of the primes iterated over above that is actually that
    # prime
    res = []
    small_end = isqrt(filter_p_max - mn_add)
    #print(f"small_end = {small_end}")
    for num in range(small_end + 1):
        num_sq = num ** 2
        #print(f"num = {num}, num_sq = {num_sq}")
        for num2 in range(min(add_nums), max(add_nums) + 1):
            #print(num2, ps.isPrime(num_sq + num2), (num2 in add_nums_set))
            if ps.isPrime(num_sq + num2) != (num2 in add_nums_set): break
        else:
            res.append(num)
            #print(num)
    
    for num0 in range(0, n_max + 1, curr_md):
        #print(f"num0 = {num0}")
        for r, neg_chk_set in r_lst:
            num = num0 + r
            if num <= small_end: continue
            elif num > n_max: break
            num_sq = num ** 2
            for num2 in add_nums:
                if not ps.millerRabinPrimalityTest(num_sq + num2, n_trials=1):
                    break
            else:
                for num2 in add_nums:
                    if not ps.isPrime(num_sq + num2, use_miller_rabin_screening=False):
                        break
                else:
                    for num2 in neg_chk_set:
                        if ps.isPrime(num_sq + num2, use_miller_rabin_screening=True, n_miller_rabin_trials=3):
                            break
                    else:
                        res.append(num)
                        #print(num)
        else: continue
        break
    return res

    """
    add_lst = [1, 3, 7, 9, 13, 27]
    neg_chk_lst = [21]
    res = 0
    sq_cnt = 0
    for i in range(10, 150 * 10 ** 6, 10):
        i_sq = i ** 2
        if i_sq % 420 != 100: continue
        #i_rt = isqrt(i)
        #if not i_rt ** 2 == i: continue
        sq_cnt += 1
        for j in add_lst:
            if not ps.isPrime(i_sq + j, extend_sieve=False, extend_sieve_sqrt=False, use_miller_rabin_screening=True, n_miller_rabin_trials=3):
                break
        else:
            for j in neg_chk_lst:
                if ps.isPrime(i_sq + j, extend_sieve=False, extend_sieve_sqrt=False, use_miller_rabin_screening=True, n_miller_rabin_trials=3):
                    break
            else:
                print(i)
            res += i
    print(f"Time taken = {time.time() - since:.4f} seconds")
    print(f"square count = {sq_cnt}")
    print(f"sum = {res}")
    """

def investigatingAPrimePatternSum(n_max: int=150 * 10 ** 6 - 1, add_nums: List[int]=[1, 3, 7, 9, 13, 27]) -> int:
    """
    Solution to Project Euler #146

    For a given list of strictly positive integers add_nums in
    strictly increasing order finds sum of the non-negative
    integers n no greater than n_max such that the sequence with
    terms:
        t_i = n ** 2 + a[i - 1]
    for integers 1 <= i <= len(a) is a sequence of consecutive
    prime numbers.

    Args:
        Optional named:
        n_max (int): Strictly positive integer specifying the
                largest value of n that may be included in the
                sum.
            Default: 150 * 10 ** 6 - 1
        add_nums (list of ints): List of strictly positive
                integers in strictly increasing order, for which
                an integer is included in the sum if and only if
                squaring that integer and adding the values in
                this list individually in order results
                in a sequence of consecutive prime numbers.
            Default: [1, 3, 7, 9, 13, 27]

    Returns:
    Integer (int) giving the sum over all the strictly positive
    integers no greater than n_max such that when squared and
    added to the integers in add_nums individually in order, the
    result is a sequence of consecutive prime numbers.

    Outline of rationale:
    See Brief outline of rationale in the documentation for the
    function investigatingAPrimePatternList().
    """
    #since = time.time()
    lst = investigatingAPrimePatternList(n_max=n_max, add_nums=add_nums)
    res = sum(lst)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 147
def rectanglesInCrossHatchedGrids(m: int=47, n: int=43) -> int:
    """
    Solution to Project Euler #147
    """
    #since = time.time()
    if n < m: m, n = n, m
    res = math.comb(m + 2, 3) * math.comb(n + 2, 3)\
            + 24 * math.comb(m + 5, 6)\
            - 8 * (2 * n + 7) * math.comb(m + 4, 5)\
            + 2 * (4 * math.comb(n + 1, 2) + 12 * n + 21) * math.comb(m + 3, 4)\
            - (8 * math.comb(n + 1, 2) + 9 * n + 10) * math.comb(m + 2, 3)\
            + math.comb(n + 1, 2) * math.comb(m + 1, 2)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 148
def pascalTrianglePrimeNondivisorCount(p: int=7, n_rows: int=10 ** 9) -> int:
    """
    Solution to Project Euler #148

    For a prime p, calculates how many entries in the first n_rows
    rows of Pascal's triangle (starting at the row with a single
    1 entry as row 1) are not divisible by p.

    Pascal's triangle is a triangle of integers constructed row
    by row by starting with the row consisting of the number 1
    only, and constructing each subsequent row by taking the sum
    of each pair of adjacent numbers, placing the result
    below and between the adjacent pair, and additionally placing
    a 1 at the beginning and end of the new row.

    The first few rows of Pascal's triangle are:

            1
          1   1
        1   2   1
      1   3   3   1
    1   4   6   4   1

    Args:
        Optional named:
        p (int): Integer specifying the prime number for which
                divisibility of entries in Pascal's triangle are
                to be assessed. The function is only guaranteed to
                give the correct answer if this is a prime number.
            Default: 7
        n_rows (int): The number of initial rows of Pascal's
                triangle whose entries are considered in the
                count.
            Default: 10 ** 9
    
    Returns:
    Integer (int) giving the number of entries in the first n_rows
    rows of Pascal's triangle that are divisible by p.

    Outline of rationale:
    For non-negative integers m and n and n < m, the (n + 1)th
    entry from the left in the (m + 1)th row of Pascal's triangle
    is given by (m choose n), where for non-negative integers m
    and n:
        (m choose n) = m! / (n! * (m - n)!) if m >= n
                       0                    otherwise
    We observe that for prime p, m! is only divisible by p if
    m >= p, implying that for non-negative integers m and n with
    m < p, (m choose n) is divisible by p if and only if n > m.
    By Lucas' theorem, for prime p, non-negative integer m and
    integer n where 0 <= n <= m:
        (m choose n) = product (i = 0 to k) (m_i choose n_i) (mod p)
    where k is the number of digits in m when expressed in base
    p and for non-negative i, m_i and n_i are the ith digits from
    the right (with the rightmost digit corresponding to i = 0) of
    m and n respectively when expressed in base p, or equivalently
    are the unique integers between 0 and p - 1 inclusive for which:
        m = m_k * p ** k + m_(k - 1) * p ** (k - 1) + ... + m_1 * p + m_0
        n = n_k * p ** k + n_(k - 1) * p ** (k - 1) + ... + n_1 * p + n_0
    
    An integer not being divisible by p is equivalent to not being
    equal to 0 modulo p. Since for prime p, a product modulo p is
    zero if and only if at least of the numbers being multiplied is
    zero modulo p, (m choose n) is non-zero modulo p and so is not
    divisible by p if and only if (m_i choose n_i) is non-zero for
    all integers 0 <= i <= k. Given that each m_i and n_i are
    non-negative integers and strictly less than p, as previously
    observed (m_i choose n_i) is divisible by p and so zero modulo
    p if and only if m_i < n_i. Thus, for each m_i there are
    (m_i + 1) values of n_i (i.e. 0, 1, ... m_i - 1, m_i) for
    which (m_i choose n_i) is non-zero modulo p. Given that if
    0 <= n_i <= m_i for each integer 0 <= i <= k results in a
    value of n between 0 and m (and so an entry in Pascal's
    triangle) and each distinct combination of values of n_i
    gives rise to a different value of n_i, all combinations of
    the values of n_i for which 0 <= n_i <= m_i map injectively
    to an entry in the mth row of Pascal's triangle. Consequently,
    the number of entries in the mth row of Pascal's triangle
    that are not divisible by p is given by:
        (prod i = 0 to k) (m_i + 1)
    
    We now consider summing this over multiple consecutive rows.
    For an integer a, consider the number of entries that are
    not divisible by p in the rows with m between a * p and
    (a + 1) * p - 1 inclusive. Using the above equation (defining
    the a_i as for m_i and n_i to be the digits in the expression
    of a in base p from right to left) we can find this number to
    be:
        ((prod i = 0 to k) (a_i + 1)) * sum(j = 0 to p - 1) (j + 1)
        = ((prod i = 0 to k) (a_i + 1)) * p * (p + 1) / 2
    It can straightforwardly be shown by induction that the
    sum for the rows with m between a * p ** b and
    (a + 1) * p ** b - 1 inclusive (for positive b) is:
        ((prod i = 0 to k) (a_i + 1)) * (p * (p + 1) / 2) ** b
    Similarly, it can be shown that for integer 0 <= c < p,
    the sum for the rows with m between a * p ** b and
    a * p ** b + c * p ** (b - 1) is:
        ((prod i = 0 to k) (a_i + 1)) * (p * (p + 1) / 2) ** (b - 1) * (c * (c + 1) / 2)
    From these it can be calulated that for all rows with m
    less than a (which given that m corrseponds to the (m + 1)th
    row of Pascal's triangle is the first a rows of Pascal's
    triangle), the number of entries not divisible by p is:
        (sum i = 0 to k) (p * (p + 1) / 2) ** i * (a_i * (a_i + 1))
                * ((prod j = i + 1 to k) (a_j + 1))
    """
    #since = time.time()
    n2 = n_rows
    #base_p_digs = []
    res = 0
    i = 0
    digs = []
    while n2:
        n2, r = divmod(n2, p)
        #base_p_digs.append(r)
        digs.append(r)
        #term = ((p * (p + 1) // 2) ** i) * (r * (r + 1) // 2)
        #res += term
        #print(i, term)
        #i += 1
    k = len(digs) - 1
    curr = 1
    for i in reversed(range(k + 1)):
        d = digs[i]
        term = curr * ((p * (p + 1) // 2) ** i) * (d * (d + 1) // 2)
        res += term
        curr *= d + 1
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
    #for i, d in enumerate(base)


# Problem 149
def kadane(seq: Iterable[int]) -> int:
    """
    Implementation of Kadane's algorithm for finding the
    largest sum over all of the non-empty contiguous
    subsequences of an integer sequence.

    Args:
        Required positional:
        seq (finite ordered iterable object containing ints):
                the sequence of integers whose largest sum
                over contiguous subsequences is to be found.
    
    Returns:
    Integer (int) giving the largest sum over all of the
    non-empty contiguous subsequences of seq.
    """
    #it = iter(seq)
    curr = 0
    res = -float("inf")
    for num in seq:
        #print(curr, num)
        curr = max(curr, 0) + num
        res = max(res, curr)
    return res

def maximumGridSumSubsequence(grid: List[List[int]]) -> int:
    """
    For a 2D array of integers, finds the largest sum elements
    that appear in a straight line in the array, where the allowed
    lines are vertical, horizontal, diagonal or antidiagonal and
    if two elements on the line are included then all elements
    between them on that line must also be included (we refer
    to this as the line being continuous).

    A horizontal line in the grid is one in which for two
    consecutive elements the first index is the same and the
    second index differs by exactly 1.

    A vertical line in the grid is one in which for two
    consecutive elements the second index is the same and the
    index index differs by exactly 1.

    A diagonal line in the grid is one in which for two
    consecutive elements, if their indices are (i1, i2) and
    (j1, j2) then i1 - j1 = i2 - j2 and equals either
    1 or -1.

    Aa antidiagonal line in the grid is one in which for two
    consecutive elements, if their indices are (i1, i2) and
    (j1, j2) then i1 - j1 = -(i2 - j2) and equals either
    1 or -1.

    Args:
        Required positional:
        grid (list of lists of ints): The 2D integer array in
                question, represented as a list of lists, where
                all the contained lists are the same length.
    
    Returns:
    Integer (int) giving the the largest sum elements that
    appear in a continuous straight line in the array.

    Outline of rationale:
    For each possible line that goes across the whole grid along
    that line in each of the four line directions, identifies a start
    point (one of the elements where a step in one direction along
    the line goes out of the grid), and produces a generator that
    iterates over the elements along that line in order starting with
    that start element until all elements on the line are exhausted.
    Kadane's algorithm is applied to the sequence this generator
    produces (see kadane()) to find the largest sum of the contiguous
    subsequences of this sequence. The largest such result amoung all
    of the line direction and start point pairs is then the solution.
    """
    shape = (len(grid), len(grid[0]))
    def horizontalGenerator(i1: int) -> Generator[int, None, None]:
        #print("horiz")
        for num in grid[i1]:
            yield num
        return

    def verticalGenerator(i2: int) -> Generator[int, None, None]:
        #print("vert")
        for i1 in range(shape[0]):
            yield grid[i1][i2]
        return

    def diagGenerator(i1_0: int, i2_0: int) -> Generator[int, None, None]:
        #print("diag")
        n_terms = min(shape[0] - i1_0, shape[1] - i2_0)
        for j in range(n_terms):
            yield grid[i1_0 + j][i2_0 + j]
        return

    def antidiagGenerator(i1_0: int, i2_0: int) -> Generator[int, None, None]:
        #print("antidiag")
        n_terms = min(i1_0 + 1, shape[1] - i2_0)
        for j in range(n_terms):
            yield grid[i1_0 - j][i2_0 + j]
        return

    res = max(kadane(verticalGenerator(0)), kadane(diagGenerator(0, 0)))
    for i1 in range(1, shape[0]):
        #print(f"i1 = {i1}")
        res = max(res,
                  kadane(horizontalGenerator(i1)),
                  kadane(diagGenerator(i1, 0)),
                  kadane(antidiagGenerator(i1, 0)),
            )
    res = max(res, kadane(verticalGenerator(i1)))
    for i2 in range(1, shape[1]):
        #print(f"i2 = {i2}")
        res = max(res,
                  kadane(verticalGenerator(i2)),
                  kadane(diagGenerator(0, i2)),
                  kadane(antidiagGenerator(shape[0] - 1, i2))
            )
    return res

def generalisedLaggedFibonacciGenerator(poly_coeffs: Tuple[int]=(100003, -200003, 0, 300007), lags: Tuple[int]=(24, 55), min_val: int=-5 * 10 ** 5, max_val: int=5 * 10 ** 5 - 1) -> Generator[int, None, None]:
    """
    Generator iterating over the terms in a generalisation
    of a lagged Fibonacci generator sequence for given for a
    given initial polynomial and given lag lengths within n a
    given range.

    The generalisation of the lagged Fibonacci generator
    sequence for the given tuple of integers poly_coeffs, tuple
    of strictly positive integers lags, and the integers min_val
    and max_val is the sequence such that for integer i >= 1,
    the i:th term in the sequence is:
        t_i = (sum j from 0 to len(poly_coeffs) - 1) (poly_coeffs[j] * i ** j) % md + min_val
                for i <= max(lags)
              ((sum j fro 0 to len(lags) - 1) (t_(i - lags[i]))) % md + min_val
                otherwise
    where md is one greater than the difference between
    min_value and max_value and % signifies modular division
    (i.e. the remainder of the integer preceding that symbol
    by the integer succeeding it). This sequence contains integer
    values between min_value and max_value inclusive.

    The terms where i <= max(lags) are referred as the polynomial
    terms and the terms where i > max(lags) are referred to as the
    recursive terms.

    In the case that lags is length 2 with those two elements
    distinct, this is a traditional lagged Fibonacci generator
    sequence.

    For well chosen values of poly_coeffs and lags for given
    min_value and max_value, this can potentially be used as a
    generator of pseudo-random integers between min_value and
    max_value inclusive.

    Note that the generator never terminates and thus any
    iterator over this generator must include provision to
    terminate (e.g. a break or return statement), otherwise
    it would result in an infinite loop.

    Args:
        Optional named:
        poly_coeffs (tuple of ints): Tuple of integers giving
                the coefficients of the polynomial used to
                generate the polynomial terms.
            Default: (100003, -200003, 0, 300007)
        lags (tuple of ints): Strictly positive integers,
                which when generating the recursive terms,
                indicates how many steps back in the sequence
                the previous terms summed should each be
                from the position of the term being generated.
                Additionally, the maximum value determines
                at which term the transition from the polynomial
                terms to the recursive terms will occur.
            Default: (24, 55)
        min_value (int): Integer giving the smallest value
                possible for terms in the sequence.
            Default: -5 * 10 ** 5
        max_value (int): Integer giving the largest value
                possible for terms in the sequence. Must
                be no smaller than min_value.
            Default: 5 * 10 ** 5 - 1
    
    Yields:
    Integer (int) between min_value and max_value inclusive,
    with the i:th term yielded (for strictly positive integer
    i) representing the i:th term in the generalisation of
    the lagged Fibonacci generator defined above for the
    given parameters.
    """
    qu = deque()
    md = max_val - min_val + 1
    #print(md)
    lags = sorted(lags)
    max_lag = lags[-1]
    for k in range(1, max_lag + 1):
        num = (sum(c * k ** i for i, c in enumerate(poly_coeffs)) % md) + min_val
        #num = ((100003 - 200003 * k + 300007 * k ** 3) % md) - 5 * 10 ** 5
        #print(num)
        qu.append(num)
        yield num
    #cnt = 0
    while True:
        num = qu.popleft()
        for i in range(len(lags) - 1):
            num += qu[-lags[i]]
        num = (num % md) + min_val
        #if cnt < 10:
        #    print(num)
        #cnt += 1
        #num = ((qu[-24] + qu.popleft() + 10 ** 6) % md) - 5 * 10 ** 5
        qu.append(num)
        yield num
    return
    """
    qu = deque()
    md = 10 ** 6
    lags = sorted(lags)
    for k in range(1, 56):
        num = ((100003 - 200003 * k + 300007 * k ** 3) % md) - 5 * 10 ** 5
        print(num)
        qu.append(num)
        yield num
    print("switch")
    cnt = 0
    while True:
        num = ((qu[-24] + qu.popleft() + 10 ** 6) % md) - 5 * 10 ** 5
        if cnt < 10:
            print(num)
        cnt += 1
        qu.append(num)
        yield num
    return
    """

def constructLaggedFibonacciGrid(
    shape: Tuple[int, int],
    l_fib_poly_coeffs: Tuple[int]=(100003, -200003, 0, 300007),
    l_fib_lags: Tuple[int]=(24, 55),
    min_grid_val: int=-5 * 10 ** 5,
    max_grid_val: int=5 * 10 ** 5 - 1,
) -> List[List[int]]:
    

    it = generalisedLaggedFibonacciGenerator(poly_coeffs=l_fib_poly_coeffs, lags=l_fib_lags, min_val=min_grid_val, max_val=max_grid_val)
    grid = []
    for i1 in range(shape[0]):
        grid.append([])
        for i2 in range(shape[1]):
            grid[-1].append(next(it))
    return grid

def maximumLaggedFibonacciGridSumSubsequence(
    shape: Tuple[int, int]=(2000, 2000),
    l_fib_poly_coeffs: Tuple[int]=(100003, -200003, 0, 300007),
    l_fib_lags: Tuple[int]=(24, 55),
    min_grid_val: int=-5 * 10 ** 5,
    max_grid_val: int=5 * 10 ** 5 - 1,
) -> int:
    """
    Solution to Project Euler #149
    """
    #since = time.time()
    grid = constructLaggedFibonacciGrid(shape, l_fib_poly_coeffs=l_fib_poly_coeffs, l_fib_lags=l_fib_lags, min_grid_val=min_grid_val, max_grid_val=max_grid_val)
    """
    it = laggedFibonacciGenerator()
    grid = []
    for i1 in range(shape[0]):
        grid.append([])
        for i2 in range(shape[1]):
            grid[-1].append(next(it))
    """
    res = maximumGridSumSubsequence(grid)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 150
def subTriangleMinSum(triangle: List[List[int]]) -> int:
    """
    Calculates the smallest total of all possible non-empty
    sub-triangle array in an integer triangle array.

    A triangle array is a 1D array whose first element is
    a 1D array of integers with length one, and each other
    element is a 1D array of integers with length one
    greater than that of the previous element. We refer
    to the 1D arrays inside the main array as rows, and
    refer to each element of these 1D arrays as an element
    of the triangle array, and the index in that array
    is referred to as its location in the row, and the
    integer it contains is its value.

    A sub-triangle array of a triangle array is a triangle
    array which can be constructed in the following way.
    First, an element of the triangle array is selected.
    The value of this element is used as the value of the
    single element in the first row of the new triangle array.
    At each subsequent step, either the construction is
    complete (this is forced if the last row of the original
    triangle array has been reached) or a new row is
    constructed by moving one row down in the triangle array
    and taking for the new row the contiguous subarray
    starting at the same position in the row as the originally
    selected element in its row, ith length one greater than
    that of the previous row of the new triangle array.
    Note that the original triange array is considered to be
    a sub-triangle array of itself, as it can be constructed
    in the described manner.

    Args:
        Required positional:
        triangle (list of lists of ints): The triangle array
                in question, represented as a list of lists.
                The first element of the outer list is
                required to be length 1 and each subsequent
                element is required to be length one greater
                than that of the previous element.

    Returns:
    Integer (int) giving the smallest sum of elements
    possible for any sub-triangle array of the given triangle
    array.

    Outline of rationale:
    We calculate the possible sub-triangle sums using bottom
    up dynamic programming, starting from the bottom of the
    triangle array going up, at each level considering the
    sub-triangle arrays whose initial elements are in that
    array using the corresponding results from the two rows
    after it.
    For each row, and each element in that row we calculate
    the sub-triangle array sum of each size (i.e. the number
    of arrays it contains) with that element as the initial
    element for the sub-triangle array, storing these in a
    list. For size 1 (i.e. the sub-triangle consisting of
    just that element), this is just the value of that
    element, and for size 2 this is that value plus the
    value of the two elements in the subsequent array at
    the corresponding position relative to the start of
    their array and the next position. For the larger sizes,
    it can be calculated by adding together the value for
    the sub-triangle size one less for those two elements in
    the subsequent array (already calculated as we are working
    from the bottom up) and to avoid double-counting of the
    elements shared by these two triangles, subtracting
    the value for the sub-triangle size one less still for
    the element in the array subsequent to those two
    elements, at the position from the beginning of the
    array one after the corresponding position of the
    original element. This process is continued for
    increasing sizes until the maximum size for that element
    is reached (which is one greater than that of the
    elements of the subsequent array, and corresponds to
    the sub-triangles reaching the last array in the
    original triangle array).
    For each value obtained, we compare with the current
    result and if it is smaller, we update the result.
    This process iterates over all of the possible
    sub-triangle arrays of the chosen triangle array.
    """
    # TODO- look for alternative approaches to make it faster
    n = len(triangle)
    if n == 1: return triangle[0][0]
    res = min(triangle[-1])
    
    
    prev = [[x] for x in triangle[-1]]
    curr = []
    row = triangle[-2]
    #print(f"i = {n - 2}")
    for j, val in enumerate(row):
        res = min(res, val)
        curr.append([val])
        lst1 = prev[j]
        lst2 = prev[j + 1]
        for v1, v2 in zip(lst1, lst2):
            curr[-1].append(val + v1 + v2)
            res = min(res, curr[-1][-1])

    for i in reversed(range(n - 2)):
        #print(f"i = {i}")
        prevprev = prev
        prev = curr
        curr = []
        row = triangle[i]
        for j, val in enumerate(row):
            res = min(res, val)
            curr.append([val])
            lst1 = prev[j]
            lst2 = prev[j + 1]
            lst3 = prevprev[j + 1]
            curr[-1].append(val + lst1[0] + lst2[0])
            res = min(res, curr[-1][-1])
            for k in range(len(lst3)):
                curr[-1].append(val + lst1[k + 1] + lst2[k + 1] - lst3[k])
                res = min(res, curr[-1][-1])
    return res

def linearCongruentialGenerator(
    k: int=615949,
    m: int=797807,
    min_value: int=-(1 << 19),
    max_value: int=(1 << 19) - 1
) -> Generator[int, None, None]:
    """
    Generator iterating over the terms in a linear congruential
    sequence for given linear and constant coefficients (k and
    m respectively) within a given range.

    A linear congruential sequence is one where the value for
    the i:th term is (t_i - mn_value) for integer i >= 1,
    where t_0 = 0 and:
        t_i = (k * t + m) % md
    where md is one greater than the difference between
    min_value and max_value and % signifies modular division
    (i.e. the remainder of the integer preceding that symbol
    by the integer succeeding it). This sequence contains integer
    values between min_value and max_value inclusive.

    For well chosen values of k and m for given min_value and
    max_value, this can potentially be used as a generator of
    pseudo-random integers between min_value and max_value
    inclusive.

    Note that the generator never terminates and thus any
    iterator over this generator must include provision to
    terminate (e.g. a break or return statement), otherwise
    it would result in an infinite loop.

    Args:
        Optional named:
        k (int): Integer giving the value of k (the linear
                coefficient) in the above function used
                for calculating the terms in the sequence.
            Default: 615949
        m (int): Integer giving the value of m (the constant
                coefficient) in the above function used
                for calculating the terms in the sequence.
        min_value (int): Integer giving the smallest value
                possible for terms in the sequence.
        max_value (int): Integer giving the largest value
                possible for terms in the sequence. Must
                be no smaller than min_value.
    
    Yields:
    Integer (int) between min_value and max_value inclusive,
    with the i:th term yielded (for strictly positive integer
    i) representing the i:th term in the linear congruential
    sequence for given linear and constant coefficients (k and
    m respectively).
    """
    t = 0
    md = max_value - min_value + 1
    while True:
        t = (k * t + m) % md
        yield t + min_value
    return

def constructLinearCongruentialTriangle(
    n_rows: int,
    l_cong_k: int=615949,
    l_cong_m: int=797807,
    min_triangle_value: int=-(1 << 19),
    max_triangle_value: int=(1 << 19) - 1
) -> List[List[int]]:
    """
    Constructs a triangle array of a given size whose elements
    are generated by a linear congruential generator, where
    elements from the linear congruential generateor fill the
    values in the triangle generator row by row from the top
    and from start to end of each row.

    A triangle array is a 1D array whose first element is
    a 1D array of integers with length one, and each other
    element is a 1D array of integers with length one
    greater than that of the previous element. We refer
    to the 1D arrays inside the main array as rows, and
    refer to each element of these 1D arrays as an element
    of the triangle array, and the index in that array
    is referred to as its location in the row, and the
    integer it contains is its value.

    For details regarding the linear congruential generator,
    see linearCongruentialGenerator(). The parameters used
    in this case are:
        k = l_cong_k,
        m = l_cong_m,
        min_value = min_triangle_value
        max_value = max_triangle value
    
    Args:
        Required positional:
        n_rows (int): The number of rows in the constructed
                triangle array.
        
        Optional named:
        l_cong_k (int): The value of the parameter k used in the
                linear congruential generator.
            Default: 615949
        l_cong_m (int): The value of the parameter m used in the
                linear congruential generator.
            Default: 797807
        min_triangle_value (int): The value of the parameter
                min_value used in the linear congruential
                generator, representing the smallest possible
                value in the constructed triangle array.
            Default: -2 ** 19
        max_triangle_value (int): The value of the parameter
                max_value used in the linear congruential
                generator, representing the largest possible
                value in the constructed triangle array.
            Default: 2 ** 19 - 1
    
    Returns:
    A list of lists of integers (int) representing the
    constructed triangle array. The first list is length 1
    and each other list is length one greater than the
    length of the previous list. Every element of these
    lists has an integer value between min_triangle_value
    and max_triangle_value inclusive.
    """
    it = linearCongruentialGenerator(k=l_cong_k, m=l_cong_m, min_value=min_triangle_value, max_value=max_triangle_value)
    triangle = []
    for i in range(n_rows):
        triangle.append([])
        for _ in range(i + 1):
            triangle[-1].append(next(it))
    return triangle

def subLinearCongruentialTriangleSubTriangleSum(n_rows: int=1000, l_cong_k: int=615949, l_cong_m: int=797807, min_triangle_value: int=-(1 << 19), max_triangle_value: int=(1 << 19) - 1) -> int:
    """
    Solution to Project Euler #150

    Calculates the smallest total of all possible non-empty
    sub-triangle arrays of the triangle array constructed
    by the function constructLinearCongruentialTriangle()
    using the parameters n_rows, l_cong_k, l_cong_m,
    min_triangle_value and max_triangle_value, whose values
    are produced using a linear congruential generator (see
    constructLinearCongruentialTriangle() and
    linearCongruentialGenerator() for details)

    A triangle array is a 1D array whose first element is
    a 1D array of integers with length one, and each other
    element is a 1D array of integers with length one
    greater than that of the previous element. We refer
    to the 1D arrays inside the main array as rows, and
    refer to each element of these 1D arrays as an element
    of the triangle array, and the index in that array
    is referred to as its location in the row, and the
    integer it contains is its value.

    A sub-triangle array of a triangle array is a triangle
    array which can be constructed in the following way.
    First, an element of the triangle array is selected.
    The value of this element is used as the value of the
    single element in the first row of the new triangle array.
    At each subsequent step, either the construction is
    complete (this is forced if the last row of the original
    triangle array has been reached) or a new row is
    constructed by moving one row down in the triangle array
    and taking for the new row the contiguous subarray
    starting at the same position in the row as the originally
    selected element in its row, ith length one greater than
    that of the previous row of the new triangle array.
    Note that the original triange array is considered to be
    a sub-triangle array of itself, as it can be constructed
    in the described manner.

    Args:
        Required positional:
        n_rows (int): The number of rows in the constructed
                triangle array.
        
        Optional named:
        l_cong_k (int): The value of the parameter k used in the
                linear congruential generator for construction
                of the triangle by constructLinearCongruentialTriangle().
            Default: 615949
        l_cong_m (int): The value of the parameter m used in the
                linear congruential generator for construction
                of the triangle by constructLinearCongruentialTriangle().
            Default: 797807
        min_triangle_value (int): The value of the parameter
                min_value used in the linear congruential
                generator for construction of the triangle by
                constructLinearCongruentialTriangle(), representing
                the smallest possible value in the triangle array.
            Default: -2 ** 19
        max_triangle_value (int): The value of the parameter
                max_value used in the linear congruential
                generator for construction of the triangle by
                constructLinearCongruentialTriangle(), representing
                the largest possible value in the triangle array.
            Default: 2 ** 19 - 1
    
    Returns:
    Integer (int) giving the smallest sum of elements
    possible for any sub-triangle array of the triangle
    array constructed by constructLinearCongruentialTriangle()
    with the specified parameters.

    Outline of rationale:
    See Outline of rationale in documentation for the function
    subTriangleMinSum().
    """
    
    #since = time.time()
    triangle = constructLinearCongruentialTriangle(n_rows, l_cong_k=l_cong_k, l_cong_m=l_cong_m, min_triangle_value=min_triangle_value, max_triangle_value=max_triangle_value)
    #print(triangle)
    #print("Generated triangle")
    res = subTriangleMinSum(triangle)
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

##############
project_euler_num_range = (51, 100)

def evaluateProjectEulerSolutions101to150(eval_nums: Optional[Set[int]]=None) -> None:
    if not eval_nums:
        eval_nums = set(range(project_euler_num_range[0], project_euler_num_range[1] + 1))

    if 101 in eval_nums:
        since = time.time()
        res = optimumPolynomial(((1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1)))
        print(f"Solution to Project Euler #101 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 102 in eval_nums:
        since = time.time()
        res = countTrianglesContainingPointFromFile(
            p=(0, 0),
            doc="project_euler_problem_data_files/0102_triangles.txt",
            rel_package_src=True,
            include_surface=True,
        )
        print(f"Solution to Project Euler #102 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 103 in eval_nums:
        since = time.time()
        res = specialSubsetSumsOptimum(n=7)
        print(f"Solution to Project Euler #103 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 104 in eval_nums:
        since = time.time()
        res = pandigitalFibonacciEnds(base=10)
        print(f"Solution to Project Euler #104 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 105 in eval_nums:
        since = time.time()
        res = specialSubsetSumsTestingFromFile(
            doc="project_euler_problem_data_files/0105_sets.txt",
            rel_package_src=True,
        )
        print(f"Solution to Project Euler #105 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 106 in eval_nums:
        since = time.time()
        res = specialSubsetSumsComparisons(n=12)
        print(f"Solution to Project Euler #106 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 107 in eval_nums:
        since = time.time()
        res = minimalNetworkFromFile(
            doc="project_euler_problem_data_files/0107_network.txt",
            rel_package_src=True,
        )
        print(f"Solution to Project Euler #107 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 108 in eval_nums:
        since = time.time()
        res = diophantineReciprocals(min_n_solutions=1001)
        print(f"Solution to Project Euler #108 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 109 in eval_nums:
        since = time.time()
        res = dartCheckouts(mx_score=99)
        print(f"Solution to Project Euler #109 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 110 in eval_nums:
        since = time.time()
        res = diophantineReciprocals(min_n_solutions=4 * 10 ** 6 + 1)
        print(f"Solution to Project Euler #110 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 111 in eval_nums:
        since = time.time()
        res = primesWithRuns(n_dig=10, base=10, ps=None)
        print(f"Solution to Project Euler #111 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 112 in eval_nums:
        since = time.time()
        res = bouncyProportions(prop_numer=99, prop_denom=100)
        print(f"Solution to Project Euler #112 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 113 in eval_nums:
        since = time.time()
        res = nonBouncyNumbers(mx_n_dig=100, base=10)
        print(f"Solution to Project Euler #113 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 114 in eval_nums:
        since = time.time()
        res = countingBlockCombinations(tot_len=50, min_large_len=3)
        print(f"Solution to Project Euler #114 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 115 in eval_nums:
        since = time.time()
        res = countingBlockCombinationsII(min_large_len=50, target_count=10 ** 6 + 1)
        print(f"Solution to Project Euler #115 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 116 in eval_nums:
        since = time.time()
        res = redGreenOrBlueTiles(tot_len=50, min_large_len=2, max_large_len=4)
        print(f"Solution to Project Euler #116 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 117 in eval_nums:
        since = time.time()
        res = redGreenAndBlueTiles(tot_len=50, min_large_len=2, max_large_len=4)
        print(f"Solution to Project Euler #117 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 118 in eval_nums:
        since = time.time()
        res = pandigitalPrimeSets(base=10)
        print(f"Solution to Project Euler #118 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 119 in eval_nums:
        since = time.time()
        res = digitPowerSum(n=30, base=10)
        print(f"Solution to Project Euler #119 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 120 in eval_nums:
        since = time.time()
        res = squareRemainders(a_min=3, a_max=1000)
        print(f"Solution to Project Euler #120 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 121 in eval_nums:
        since = time.time()
        res = diskGameMaximumNonLossPayout(15)
        print(f"Solution to Project Euler #121 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 122 in eval_nums:
        since = time.time()
        res = efficientExponentiation(sum_min=1, sum_max=200, method="exact")
        print(f"Solution to Project Euler #122 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 123 in eval_nums:
        since = time.time()
        res = primeSquareRemainders(target_remainder=10 ** 10 + 1)
        print(f"Solution to Project Euler #123 = {res}, calculated in {time.time() - since:.4f} seconds")
        
    if 124 in eval_nums:
        since = time.time()
        res = orderedRadicals(n=100000, k=10000)
        print(f"Solution to Project Euler #124 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 125 in eval_nums:
        since = time.time()
        res = palindromicConsecutiveSquareSums(mx=100000000 - 1, base=10)
        print(f"Solution to Project Euler #125 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 126 in eval_nums:
        since = time.time()
        res = cuboidLayers(target_layer_size_count=1000, step_size=10000)
        print(f"Solution to Project Euler #126 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 127 in eval_nums:
        since = time.time()
        res = abcHits(c_max=119999)
        print(f"Solution to Project Euler #127 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 128 in eval_nums:
        since = time.time()
        res = hexagonalTileDifferences(sequence_number=2000)
        print(f"Solution to Project Euler #128 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 129 in eval_nums:
        since = time.time()
        res = repunitDivisibility(target_repunit_length=1000000, base=10)
        print(f"Solution to Project Euler #129 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 130 in eval_nums:
        since = time.time()
        res = sumCompositesWithPrimeRepunitProperty(n_to_sum=25, base=10)
        print(f"Solution to Project Euler #130 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 131 in eval_nums:
        since = time.time()
        res = primeCubePartnership(p_max=999_999)
        print(f"Solution to Project Euler #131 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 132 in eval_nums:
        since = time.time()
        res = repunitPrimeFactorsSum(n_ones=1_000_000_000, n_p=40, base=10) 
        print(f"Solution to Project Euler #132 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 133 in eval_nums:
        since = time.time()
        res = repunitPowBaseNonFactorsSum(p_max=99_999, base=10)
        print(f"Solution to Project Euler #133 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 134 in eval_nums:
        since = time.time()
        res = primePairConnectionsSum(p1_min=5, p1_max=1_000_000, base=10)
        print(f"Solution to Project Euler #134 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 135 in eval_nums:
        since = time.time()
        res = sameDifferences(n_max=999_999, target_count=10)
        print(f"Solution to Project Euler #135 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 136 in eval_nums:
        since = time.time()
        res = singletonDifferences(n_max=49_999_999)
        print(f"Solution to Project Euler #136 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 137 in eval_nums:
        since = time.time()
        res = modifiedFibonacciGoldenNugget(nugget_number=15, G1=1, G2=1)
        print(f"Solution to Project Euler #137 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 138 in eval_nums:
        since = time.time()
        res = specialIsocelesTriangleSum(n_smallest_to_sum=12)
        print(f"Solution to Project Euler #138 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 139 in eval_nums:
        since = time.time()
        res = pythagoreanTiles(max_triangle_perimeter=99_999_999)
        print(f"Solution to Project Euler #139 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 140 in eval_nums:
        since = time.time()
        res = modifiedFibonacciGoldenNuggetSum(n_nugget_numbers=30, G1=1, G2=4)
        print(f"Solution to Project Euler #140 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 141 in eval_nums:
        since = time.time()
        res = squareProgressiveNumbersSum(n_max=10 ** 12)
        print(f"Solution to Project Euler #141 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 142 in eval_nums:
        since = time.time()
        res = perfectSquareCollection()
        print(f"Solution to Project Euler #142 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 143 in eval_nums:
        since = time.time()
        res = torricelliTriangleUniqueLengthSum(sm_max=12 * 10 ** 4)
        print(f"Solution to Project Euler #143 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 144 in eval_nums:
        since = time.time()
        res = laserBeamEllipseReflectionCount(
            ellipse=(4, 1, 100),
            pos0=((0, 1), (101, 10)),
            reflect1=((7, 5), (-48, 5)),
            x_window=((-1, 100), (1, 100)),
            use_float=True
        )
        print(f"Solution to Project Euler #144 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 145 in eval_nums:
        since = time.time()
        res = reversibleNumbersCount(n_dig_max=9, base=10)
        print(f"Solution to Project Euler #145 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 146 in eval_nums:
        since = time.time()
        res = investigatingAPrimePatternSum(
            n_max=150 * 10 ** 6 - 1,
            add_nums=[1, 3, 7, 9, 13, 27],
        )
        print(f"Solution to Project Euler #146 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 147 in eval_nums:
        since = time.time()
        res = rectanglesInCrossHatchedGrids(m=47, n=43)
        print(f"Solution to Project Euler #147 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 148 in eval_nums:
        since = time.time()
        res = pascalTrianglePrimeNondivisorCount(p=7, n_rows=10 ** 9)
        print(f"Solution to Project Euler #148 = {res}, calculated in {time.time() - since:.4f} seconds")

    if 149 in eval_nums:
        since = time.time()
        res = maximumLaggedFibonacciGridSumSubsequence(shape=(2000, 2000))
        print(f"Solution to Project Euler #149 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if 150 in eval_nums:
        since = time.time()
        #res = subTriangleMinSum(triangle=[[15], [-14, -7], [20, -13, -5], [-3, 8, 23, -26], [1, -4, -5, -18, 5], [-16, 31, 2, 9, 28, 3]])
        res = subLinearCongruentialTriangleSubTriangleSum(
            n_rows=1000,
            l_cong_k=615949,
            l_cong_m=797807,
            min_triangle_value=-(1 << 19),
            max_triangle_value=(1 << 19) - 1,
        )
        print(f"Solution to Project Euler #150 = {res}, calculated in {time.time() - since:.4f} seconds")


if __name__ == "__main__":
    eval_nums = {126}
    evaluateProjectEulerSolutions101to150(eval_nums)