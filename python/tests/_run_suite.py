"""Streaming unittest runner with a faulthandler watchdog.

Plain ``unittest discover`` buffered through a shell pipe hides *which* test is
running when the suite hangs (as it does on some platforms). This runner streams
output live and arms faulthandler to dump every thread's stack and abort if the
suite exceeds the watchdog timeout - so a hang shows up as a traceback pointing
at the stuck test rather than an opaque CI step timeout.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import unittest

# Dump all thread stacks and exit non-zero if the suite runs longer than this.
# Kept under the CI step's own timeout so we capture the stack before the job is
# killed. Override with GV_TEST_WATCHDOG_SECONDS.
_WATCHDOG = int(os.environ.get("GV_TEST_WATCHDOG_SECONDS", "480"))


def main() -> int:
    faulthandler.enable()
    faulthandler.dump_traceback_later(_WATCHDOG, exit=True)
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        loader = unittest.TestLoader()
        suite = loader.discover(start_dir=here)
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=False)
        result = runner.run(suite)
        return 0 if result.wasSuccessful() else 1
    finally:
        # Disarm the watchdog so its pending timer does not perturb interpreter
        # shutdown (on Windows an armed exit=True timer leaves a non-zero code).
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    sys.exit(main())
