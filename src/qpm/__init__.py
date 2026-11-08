"""A faithful Python rebuild of a US equity momentum and inverse-volatility study.

The package reconstructs, from monthly prices and Fama-French factors, a decile momentum
strategy, an inverse-volatility portfolio against an equal-weight benchmark, and a rolling
factor-model decomposition of the active return.
"""

from qpm import data, factors, momentum, portfolio, stats, turnover

__all__ = ["data", "factors", "momentum", "portfolio", "stats", "turnover"]
