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