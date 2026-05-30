"""Enable ``python -m loopgain`` to invoke the CLI."""

import sys

from loopgain.cli import main

if __name__ == "__main__":
    sys.exit(main())
