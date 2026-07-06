"""Back-compat wrapper; the implementation lives in quantbet.settlement.

Equivalent to: python -m quantbet settle
"""

from quantbet.settlement import run_settlement

if __name__ == "__main__":
    run_settlement()
