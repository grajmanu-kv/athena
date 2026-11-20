"""`python -m athena.web` is the same as `athena serve`."""

import sys

from athena.cli import main

raise SystemExit(main(["serve", *sys.argv[1:]]))
