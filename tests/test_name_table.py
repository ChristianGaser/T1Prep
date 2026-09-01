"""Every filename ``Names.tsv`` produces has to be one the writers accept.

CAT-Surface infers a file's format from its extension and refuses the ones it
does not know.  A pattern with a bad extension therefore fails only when that
particular output is produced — which for the debug-only intensity maps meant a
``.txt`` suffix survived unnoticed until someone ran with ``--debug``.
"""

import numpy as np
import pytest

from t1prep.utils import DATA_PATH_T1PREP, NameTable

cat_surf = pytest.importorskip("cat_surf")

NAMES_TSV = DATA_PATH_T1PREP / "Names.tsv"

#: Codes written through ``cat_surf.write_values`` in surface_estimation.
VALUE_OUTPUTS = [
    "PBT_shape",
    "Area_shape",
    "Sulc_shape",
    "GMT_shape",
    "Intensity_Mid",
    "Intensity_Pial",
    "Intensity_WM",
    "Mask_label",
]

#: Column 1 is the CAT12 pattern, column 2 the BIDS one; both are reachable.
COLUMNS = {1: "lh", 2: "L"}


@pytest.fixture(scope="module")
def table():
    return NameTable(NAMES_TSV)


@pytest.mark.parametrize("code", VALUE_OUTPUTS)
@pytest.mark.parametrize("column", sorted(COLUMNS))
def test_value_output_names_are_writable(tmp_path, table, code, column):
    name = table.substitute(
        code, column, bname="sub-01_T1w", hemi=COLUMNS[column], nii_ext="nii.gz"
    )
    values = np.linspace(0.0, 1.0, 64, dtype=np.float64)
    cat_surf.write_values(str(tmp_path / name), values)

    written = tmp_path / name
    assert written.exists() and written.stat().st_size > 0


@pytest.mark.parametrize("code", VALUE_OUTPUTS)
@pytest.mark.parametrize("column", sorted(COLUMNS))
def test_value_output_names_carry_no_unsupported_suffix(table, code, column):
    """``.txt`` is the trap: plausible-looking, and the one thing rejected."""
    name = table.substitute(
        code, column, bname="sub-01_T1w", hemi=COLUMNS[column], nii_ext="nii.gz"
    )
    assert not name.endswith(".txt")
