
"""
metrics_tier2.py
----------------
Legacy Tier-1 metrics initialiser kept for backward compatibility.
"""

K_LIST = [1, 3, 5, 10]

def init_metrics():
    """Create an empty metrics dict with keys for each K in K_LIST."""
    m = {}
    for k in K_LIST:
        m[f"P@{k}"] = []
        m[f"R@{k}"] = []
        m[f"MAP@{k}"] = []
        m[f"NDCG@{k}"] = []
    return m
