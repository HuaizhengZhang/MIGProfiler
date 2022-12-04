#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 11/3/2020
"""
import re


def camelcase_to_snakecase(camel_str):
    """
    Convert string in camel case to snake case.
    References:
        https://www.geeksforgeeks.org/python-program-to-convert-camel-case-string-to-snake-case/
    Args:
        camel_str: String in camel case.
    Returns:
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
