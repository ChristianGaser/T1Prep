"""Where T1Prep's data folder is and what the output files are called.

The data folder and ``Names.tsv`` are needed by every part of T1Prep: the
pipeline, and the launcher and the viewers as well.  They live here, apart
from ``t1prep.utils``, because ``utils`` imports torch: a window that only
shows results should not wait for the networks' framework to load.
``t1prep.utils`` re-exports every name of this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    from importlib.resources import files as _res_files
except Exception:
    _res_files = None


# We are inside src/t1prep; repo root is two levels up (dev fallback)
ROOT_PATH = Path(__file__).resolve().parents[2]

def _package_data_root() -> Optional[Path]:
    if _res_files is not None:
        try:
            # files('t1prep.data') returns a Traversable to the data package dir
            return Path(str(_res_files('t1prep.data')))
        except Exception:
            return None
    return None

# Resolve T1Prep's data directory.  Prefer importlib.resources (correct
# for installed wheels, where __file__'s parent points into site-packages),
# then fall back to the directory adjacent to this file — which works for
# both editable installs and direct source-tree imports.  The legacy
# ``ROOT_PATH / "src" / "t1prep" / "data"`` form was broken for non-editable
# installs because it spliced ``src/`` under site-packages.
DATA_PATH_T1PREP: Path = (
    _package_data_root()
    or (Path(__file__).resolve().parent / "data")
)


# Prefer packaged Names.tsv; fall back to repo root for editable/dev mode
def _resolve_names_tsv() -> Path:
    if _res_files is not None:
        try:
            p = _res_files('t1prep.data').joinpath('Names.tsv')
            if p.is_file():
                return Path(str(p))
        except Exception:
            pass
    # Fallback
    return DATA_PATH_T1PREP / "Names.tsv"

name_file = _resolve_names_tsv()


# ===========================================================================
# Names.tsv: pattern lookup and substitution (shared with surface_estimation)
# ===========================================================================

class NameTable:
    """In-memory representation of ``Names.tsv``.

    The file is whitespace-separated: column 0 is a symbolic code,
    column 1 is the CAT12-style filename pattern, column 2 (if present)
    is the BIDS-style filename pattern.  Patterns contain placeholders
    ``{bname}``, ``{side}``, ``{desc}``, ``{space}``, ``{atlas}``,
    ``{nii_ext}``.
    """

    def __init__(self, path: str | os.PathLike):
        self._rows: dict[str, list[str]] = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 2:
                    continue
                self._rows[parts[0]] = parts

    def codes(self) -> list:
        """Every symbolic code defined in the table."""
        return list(self._rows)

    def pattern(self, code: str, column: int) -> str:
        row = self._rows.get(code)
        if row is None or column >= len(row):
            raise KeyError(
                f"Names.tsv: no pattern for code={code!r} column={column}")
        return row[column]

    def substitute(self, code: str, column: int, *,
                   bname: str, hemi: str, desc: str = "",
                   space: str = "", atlas: str = "",
                   nii_ext: str = "") -> str:
        pat = self.pattern(code, column)
        bname_clean = bname[:-4] if bname.endswith("_T1w") else bname
        out = (pat
               .replace("{bname}", bname_clean)
               .replace("{side}", hemi)
               .replace("{desc}", desc)
               .replace("{space}", space)
               .replace("{atlas}", atlas)
               .replace("{nii_ext}", nii_ext)
               .replace("..", "."))
        return out
