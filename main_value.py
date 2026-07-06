"""Back-compat wrapper; the implementation lives in quantbet.slip.

Equivalent to: python -m quantbet slip --profile value
"""

from quantbet.slip import run
from quantbet.strategy import VALUE

if __name__ == "__main__":
    run(VALUE)
