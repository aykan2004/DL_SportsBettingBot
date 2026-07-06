"""Back-compat wrapper; the implementation lives in quantbet.slip.

Equivalent to: python -m quantbet slip --profile safe
"""

from quantbet.slip import run
from quantbet.strategy import SAFE

if __name__ == "__main__":
    run(SAFE)
