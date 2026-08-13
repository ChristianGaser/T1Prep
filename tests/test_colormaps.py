import sys
import unittest
from pathlib import Path

# Allow running tests without installing the package (repo checkout / editable dev)
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from t1prep.gui.colormaps import (
        FIRE,
        JET,
        apply_discrete,
        build_overlay_lut,
        clipped_lut_indices,
        get_lookup_table,
        invert_lut,
    )
except Exception as exc:  # pragma: no cover - depends on VTK
    raise unittest.SkipTest(f"colormaps unavailable: {exc}")


class TestSharedLookupTables(unittest.TestCase):
    """One implementation of "overlay colours" for both viewers."""

    def test_inverse_reverses_the_table(self):
        lut = get_lookup_table(JET, 1.0)
        first, last = lut.GetTableValue(0), lut.GetTableValue(255)
        invert_lut(lut)
        self.assertEqual(lut.GetTableValue(0), last)
        self.assertEqual(lut.GetTableValue(255), first)

    def test_discrete_creates_bands_of_equal_width(self):
        lut = get_lookup_table(JET, 1.0)
        apply_discrete(lut, 4)
        colors = {tuple(lut.GetTableValue(i)[:3]) for i in range(256)}
        self.assertEqual(len(colors), 4)
        # every band holds the colour of its first entry
        for band in range(4):
            start = band * 64
            for i in range(start, start + 64):
                self.assertEqual(lut.GetTableValue(i)[:3], lut.GetTableValue(start)[:3])

    def test_clipped_indices_include_the_clamped_edges(self):
        """-range 6 16 -clip -100 6 must clip the lower end, not nothing."""
        indices = clipped_lut_indices(256, 6.0, 16.0, -100.0, 6.0)
        self.assertIn(0, indices)
        self.assertNotIn(255, indices)
        self.assertEqual(clipped_lut_indices(256, 0.0, 1.0, 0.0, -1.0), [])

    def test_build_overlay_lut_combines_everything(self):
        lut = build_overlay_lut(FIRE, 0.5, value_range=[0.0, 10.0],
                                clip=(-1.0, 1.0), discrete=8)
        self.assertEqual(tuple(round(v, 3) for v in lut.GetTableRange()), (0.0, 10.0))
        self.assertEqual(lut.GetTableValue(0)[3], 0.0)          # clipped away
        # the table is 8-bit, so 0.5 comes back as 128/255
        self.assertAlmostEqual(lut.GetTableValue(200)[3], 0.5, places=2)
        colors = {tuple(lut.GetTableValue(i)[:3]) for i in range(256)}
        self.assertEqual(len(colors), 8)                        # bands kept

    def test_without_a_range_the_table_is_left_open(self):
        """The surface viewer sets the range on the mapper, not the table."""
        lut = build_overlay_lut(JET, 1.0)
        self.assertEqual(lut.GetTableValue(0)[3], 1.0)


if __name__ == "__main__":
    unittest.main()
