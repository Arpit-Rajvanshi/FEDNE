"""
FEDANCHOR v7 - Shared Global Anchors
=====================================
Research version testing the hypothesis that a shared/global anchor set
provides a common low-dimensional geometric reference frame across federated
clients and improves dimensionality-reduction quality.

Key architectural change from V5:
    LOCAL CLIENT-SPECIFIC ANCHORS → SHARED GLOBAL ANCHORS

One anchor matrix A ∈ R^(K × 2) is used by every federated client.
"""

__version__ = "6.0.0"
__research_question__ = (
    "Can a shared/global anchor set provide a common low-dimensional geometric "
    "reference frame across federated clients and improve dimensionality-reduction "
    "quality compared with independently learned client-specific anchors?"
)
