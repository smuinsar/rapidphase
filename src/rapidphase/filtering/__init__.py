"""
Filtering module for interferogram preprocessing.

Provides adaptive filtering algorithms to reduce noise in interferograms
before phase unwrapping.
"""

from rapidphase.filtering.goldstein import GoldsteinFilter, filter_multi_gpu

__all__ = ["GoldsteinFilter", "filter_multi_gpu"]
