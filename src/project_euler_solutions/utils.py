#!/usr/bin/env python

from typing import (
    List,
)

import os

from pathlib import Path

def loadTextFromFile(doc: str, rel_package_src: bool=False) -> str:
    """
    Loads the text contained in a .txt file located at relative or
    absolute location doc as a single string.

    Args:
        Required positional:
        doc (str): The relative or absolution location of the .txt
                file containing the text to be loaded.
        
        Optional named:
        rel_package_src (bool): Whether a relative path given by doc
                is relative to the current directory (False) or
                the package src directory (True).
            Default: False
    
    Returns:
    String representing the contents of the .txt file at location
    doc.
    """

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
    return txt

def loadStringsFromFile(doc: str, rel_package_src: bool=False) -> List[str]:
    """
    Loads a comma-separated list of strings from .txt file located at
    relative or absolute location doc. The file should contain the words
    separated by commas (',') with each word surrounded by double
    quotation marks ('"').
    
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
    
    txt = loadTextFromFile(doc=doc, rel_package_src=rel_package_src)
    return txt.strip("\"").split("\",\"")

class UnionFind:
    def __init__(self, n: int):
        self.n = n
        self.root = list(range(n))
        self.rank = [1] * n
    
    def find(self, v: int) -> int:
        r = self.root[v]
        if r == v: return v
        res = self.find(r)
        self.root[v] = res
        return res
    
    def union(self, v1: int, v2: int) -> None:
        r1, r2 = list(map(self.find, (v1, v2)))
        if r1 == r2: return
        d = self.rank[r1] - self.rank[r2]
        if d < 0: r1, r2 = r2, r1
        elif not d: self.rank[r1] += 1
        self.root[r2] = r1
        return
    
    def connected(self, v1: int, v2: int) -> bool:
        return self.find(v1) == self.find(v2)