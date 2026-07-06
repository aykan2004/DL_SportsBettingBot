"""Back-compat wrapper; the implementation lives in quantbet.retrain.

Equivalent to: python -m quantbet retrain
"""

from quantbet.retrain import retrain

if __name__ == "__main__":
    retrain()
