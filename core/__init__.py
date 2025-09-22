"""
Core Package

Contains core environment and simulation components for traffic signal control.

Components:
- TrafficEnvironment: SUMO environment wrapper with advanced metrics
"""

from .traffic_env import TrafficEnvironment

__all__ = ['TrafficEnvironment']
