"""Turn trained torch models into T1Prep network files.

Two sources: deepmriprep's TorchScript archives (``.pt``), converted once
when T1Prep prepares its models, and the ``state_dict`` files
deepmriprep-train writes (``.pth``).  A retrained model should be exported
straight from its ``state_dict`` -- :func:`save_state_dict` in the training
script, or afterwards::

    python -m t1prep.nn.convert data/models/*.pth --out-dir models/

instead of being traced to TorchScript, which torch is retiring.  The keys
stay those of deepmriprep-train's modules; the architecture is inferred from
them and stored in the file.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from .architectures import architecture_names, get_architecture
from .weights import save_safetensors

__all__ = ["convert_model", "save_state_dict"]


def save_state_dict(state_dict, path, arch: str | None = None, **metadata) -> None:
    """Store a torch ``state_dict`` as one of T1Prep's network files.

    Args:
        state_dict: ``model.state_dict()`` of a registered architecture.
        path: Destination ``.safetensors`` file.
        arch: Architecture name; inferred from the keys when not given.
        **metadata: Extra strings for the file header (e.g. ``source``).
    """
    arrays = {k: v.detach().cpu().contiguous().numpy() for k, v in state_dict.items()}
    architecture = get_architecture(arch, arrays)
    save_safetensors(
        path, architecture.canonical(arrays), {"arch": architecture.name, **metadata}
    )


def _is_torchscript(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return any(
                n.endswith("/constants.pkl") or "/code/" in n
                for n in archive.namelist()
            )
    except zipfile.BadZipFile:
        return False


def convert_model(src, dst, arch: str | None = None) -> None:
    """Convert a TorchScript archive or a saved ``state_dict`` to a network file.

    Args:
        src: ``.pt`` TorchScript archive (as deepmriprep ships them) or a
            ``torch.save``'d ``state_dict`` (as deepmriprep-train writes them).
        dst: Destination ``.safetensors`` file.
        arch: Architecture name; inferred from the keys when not given.
    """
    import torch

    src = Path(src)
    if _is_torchscript(src):
        state = torch.jit.load(str(src), map_location="cpu").state_dict()
    else:
        state = torch.load(str(src), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"{src}: neither a TorchScript archive nor a state_dict")
    save_state_dict(state, dst, arch=arch, source=src.name)


def main(argv: list | None = None) -> int:
    """``python -m t1prep.nn.convert``: convert trained models to network files."""
    from ..cli_help import ArgumentParser

    argv = sys.argv[1:] if argv is None else argv
    parser = ArgumentParser(
        prog="python -m t1prep.nn.convert",
        description="Convert TorchScript archives or saved state_dicts (.pt, .pth) to "
        "T1Prep's safetensors network files, named like each input.",
    )
    parser.add_argument(
        "models", nargs="+", help="TorchScript archives or state_dict files"
    )
    parser.add_argument(
        "--out-dir", default=None, help="output directory (default: beside each input)"
    )
    parser.add_argument(
        "--arch",
        choices=architecture_names(),
        help="architecture (default: from the keys)",
    )
    parser.exit_without_arguments(argv)
    args = parser.parse_args(argv)
    for src in map(Path, args.models):
        dst = (
            Path(args.out_dir) if args.out_dir else src.parent
        ) / f"{src.stem}.safetensors"
        dst.parent.mkdir(parents=True, exist_ok=True)
        convert_model(src, dst, arch=args.arch)
        print(f"{src} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
