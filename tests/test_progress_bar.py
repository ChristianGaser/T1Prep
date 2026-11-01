"""Tests for the progress bars: ``scripts/progress_bar_multi.sh`` and its
Python mirror :func:`t1prep.utils.progress_bar`.

On a terminal each step redraws the line in place (``\\r``), so a step's line
must cover the previous one whatever the label lengths; otherwise the tail of
the longer line stays visible behind the shorter one's ETA.
"""

import io
import os
import re
import select
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from t1prep.utils import progress_bar  # noqa: E402

try:
    import pty
except ImportError:  # Windows
    pty = None

#: Labels of consecutive volume steps, the second one unpadded and shorter.
LABELS = ["DeepMriPrep segmentation     ", "Remove non-cortical GM", "Atlas creation"]

_ESCAPE = re.compile(r"\x1b(\[[0-9;?]*[A-Za-z]|\([AB0])")


def _frames(output):
    """The redrawn lines of *output*, without escape codes."""
    return [_ESCAPE.sub("", f) for f in re.split(r"[\r\n]+", output) if "%" in f]


class _Terminal(io.StringIO):
    def isatty(self):
        return True


class TestPythonBar(unittest.TestCase):
    def test_a_step_covers_the_previous_line_at_80_columns(self):
        out = _Terminal()
        with (
            mock.patch("sys.stdout", out),
            mock.patch("time.time", side_effect=[100.0, 110.0, 120.0]),
            mock.patch(
                "shutil.get_terminal_size", return_value=os.terminal_size((80, 24))
            ),
        ):
            for step, label in enumerate(LABELS, 1):
                progress_bar(step, len(LABELS) + 1, label)
        frames = _frames(out.getvalue())
        self.assertEqual(len(frames), 3)
        self.assertTrue(all(len(frame) <= 80 for frame in frames), frames)
        self.assertEqual(len({len(f) for f in frames}), 1, frames)
        self.assertEqual(frames[1].index("ETA"), frames[2].index("ETA"))
        self.assertEqual(out.getvalue().count("\x1b[K\r"), 3)

    def test_a_log_gets_no_padding(self):
        out = io.StringIO()
        with mock.patch("sys.stdout", out):
            progress_bar(3, 3, "Remove non-cortical GM")
        self.assertIn("100% Remove non-cortical GM \n", out.getvalue())


@unittest.skipIf(
    pty is None or shutil.which("bash") is None, "needs bash and a pseudo-terminal"
)
class TestShellBar(unittest.TestCase):
    def test_a_step_covers_the_previous_line_at_80_columns(self):
        master, slave = pty.openpty()
        self.addCleanup(os.close, master)
        self.addCleanup(os.close, slave)
        import fcntl
        import struct
        import termios

        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
        env = dict(os.environ, TERM="xterm-256color")
        output = b""
        for step, label in enumerate(LABELS, 1):
            subprocess.run(
                [
                    str(_ROOT / "scripts" / "progress_bar_multi.sh"),
                    *("1", "", str(step), str(len(LABELS)), label, "40", "0"),
                ],
                stdout=slave,
                stderr=subprocess.DEVNULL,
                env=env,
                check=True,
            )
            # Read while the slave is open: macOS drops what is left unread
            # once it is closed.
            while select.select([master], [], [], 0.2)[0]:
                output += os.read(master, 4096)
        frames = _frames(output.decode("utf-8"))
        self.assertEqual(len(frames), 3)
        self.assertTrue(all(len(frame) <= 80 for frame in frames), frames)
        self.assertEqual(len({len(f) for f in frames}), 1, frames)


if __name__ == "__main__":
    unittest.main()
