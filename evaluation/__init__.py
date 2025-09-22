"""
Evaluation Package

Contains performance analysis, statistical validation, and result evaluation tools.

Components:
- PerformanceComparator: Statistical comparison framework
- ResultsAnalyzer: Comprehensive result analysis
- Hyperparameter/Reward validation tools
"""

from .performance_comparison import PerformanceComparator
from .results_analysis import ResultsAnalyzer

__all__ = ['PerformanceComparator', 'ResultsAnalyzer']
