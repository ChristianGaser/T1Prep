"""Importing t1prep must not change the state of the host process.

Up to 0.8.0 ``t1prep.utils`` and ``t1prep.segment`` called
``warnings.filterwarnings("ignore")`` at import.  The package ``__init__``
imports ``utils``, so every program that did ``import t1prep`` -- fMRIPrep,
Nipype, a notebook -- lost all of its warnings, NaN and divide-by-zero
``RuntimeWarning`` included.  The pipeline steps now quieten themselves only
when they run as their own process (their ``__main__`` blocks).
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"

_PROBE = """
import warnings
before = list(warnings.filters)
import t1prep, t1prep.segment, t1prep.surface_estimation
added = [f for f in warnings.filters if f not in before]
# ("ignore", message=None, category=Warning): a blanket filter
print(sum(f[0] == "ignore" and f[1] is None and f[2] is Warning for f in added))
warnings.warn("host warning")
"""


class TestImportingKeepsTheHostsWarnings(unittest.TestCase):
    def test_no_blanket_filter_and_host_warnings_still_show(self):
        env = dict(os.environ, PYTHONPATH=str(_SRC))
        done = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True, text=True, timeout=300, env=env,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "0")
        self.assertIn("host warning", done.stderr)


if __name__ == "__main__":
    unittest.main()
