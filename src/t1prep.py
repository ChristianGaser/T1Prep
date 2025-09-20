import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union


def _as_atlas_list(arg: Optional[Union[str, Sequence[str]]]) -> Optional[str]:
    """
    Normalize atlas arguments to the script's expected string format.

    The T1Prep bash script expects a string like:  "'atlas1','atlas2'".
    Accepts a str (passed through) or a sequence of strings.
    """

    if arg is None:
        return None
    if isinstance(arg, str):
        return arg
    items = [str(a) for a in arg if a is not None and str(a) != ""]
    if not items:
        return None
    return ", ".join(f"'{a}'" for a in items)


def run_t1prep(
    inputs: Union[str, Sequence[str]],
    *,
    defaults: Optional[Union[str, os.PathLike]] = None,
    install: bool = False,
    re_install: bool = False,
    python: Optional[Union[str, os.PathLike]] = None,
    out_dir: Optional[Union[str, os.PathLike]] = None,
    pre_fwhm: Optional[Union[int, float]] = None,
    downsample: Optional[Union[int, float]] = None,
    median_filter: Optional[int] = None,
    vessel: Optional[Union[int, float]] = None,
    bin_dir: Optional[Union[str, os.PathLike]] = None,
    no_overwrite: Optional[str] = None,
    thickness_method: Optional[int] = None,
    multi: Optional[Union[int, float]] = None,
    seed: Optional[int] = None,
    atlas: Optional[Union[str, Sequence[str]]] = None,
    atlas_surf: Optional[Union[str, Sequence[str]]] = None,
    min_memory: Optional[Union[int, float]] = None,
    gz: bool = False,
    hemisphere: bool = False,
    no_surf: bool = False,
    no_seg: bool = False,
    no_sphere_reg: bool = False,
    pial_white: bool = False,
    lesions: bool = False,
    no_mwp: bool = False,
    wp: bool = False,
    rp: bool = False,
    p: bool = False,
    csf: bool = False,
    amap: bool = False,
    bids: bool = False,
    no_correct_folding: bool = False,
    debug: bool = False,
    fast: bool = False,
    cwd: Optional[Union[str, os.PathLike]] = None,
    env: Optional[dict] = None,
    check: bool = True,
    log_file: Optional[Union[str, os.PathLike]] = None,
    stream_output: bool = True,
) -> int:
    """
    Run the T1Prep bash pipeline (scripts/T1Prep) from Python with full CLI parity.

    Parameters
    - inputs: one path or a list/sequence of paths to .nii/.nii.gz files
    - defaults: path to an alternative defaults file (equivalent to --defaults)
    - install / re_install: trigger dependency installation (equivalent to --install / --re-install)
    - python: interpreter for the script to use internally (equivalent to --python)
    - out_dir: output directory (equivalent to --out-dir)
    - pre_fwhm, downsample, median_filter, vessel, thickness_method: numeric options
    - bin_dir: path to CAT/Cortex tool binaries (equivalent to --bin-dir)
    - no_overwrite: filename pattern to skip existing results (equivalent to --no-overwrite)
    - multi, seed, min_memory: parallelization and reproducibility options
    - atlas, atlas_surf: atlas lists; can be a string (passed through) or a sequence
    - gz, hemisphere, no_surf, no_seg, no_sphere_reg, pial_white, lesions, no_mwp,
      wp, rp, p, csf, amap, bids, no_correct_folding, debug, fast: boolean flags
    - cwd, env: working directory and env overrides for the subprocess
    - check: raise if the process returns non-zero
    - log_file: tee stdout/stderr to this file if provided
    - stream_output: when True, stream stdout/stderr live; when False, suppress output

    Returns
    - int exit code from the T1Prep script
    """

    root = Path(__file__).resolve().parent.parent
    script = root / "scripts" / "T1Prep"
    if not script.exists():
        raise FileNotFoundError(f"T1Prep script not found: {script}")

    # Normalize inputs to list
    if isinstance(inputs, (str, os.PathLike)):
        input_list: List[str] = [str(inputs)]
    else:
        input_list = [str(p) for p in inputs]
    if not input_list:
        raise ValueError("No input files provided")

    args: List[str] = [str(script)]

    def add_flag(flag: str, enabled: bool):
        if enabled:
            args.append(flag)

    def add_opt(flag: str, value):
        if value is not None:
            args.extend([flag, str(value)])

    # Options mapping (prefer documented long flags)
    if defaults is not None:
        add_opt("--defaults", defaults)
    if install:
        args.append("--install")
    if re_install:
        args.append("--re-install")
    add_opt("--python", python)
    add_opt("--out-dir", out_dir)
    add_opt("--pre-fwhm", pre_fwhm)
    add_opt("--downsample", downsample)
    add_opt("--median-filter", median_filter)
    add_opt("--vessel", vessel)
    # The bash script accepts --bin-dir or --bindir
    add_opt("--bin-dir", bin_dir)
    add_opt("--no-overwrite", no_overwrite)
    add_opt("--thickness-method", thickness_method)
    add_opt("--multi", multi)
    add_opt("--seed", seed)
    atl = _as_atlas_list(atlas)
    if atl is not None:
        add_opt("--atlas", atl)
    atl_s = _as_atlas_list(atlas_surf)
    if atl_s is not None:
        add_opt("--atlas-surf", atl_s)
    add_opt("--min-memory", min_memory)

    add_flag("--gz", gz)
    add_flag("--hemisphere", hemisphere)
    add_flag("--no-surf", no_surf)
    add_flag("--no-seg", no_seg)
    add_flag("--no-sphere-reg", no_sphere_reg)
    add_flag("--pial-white", pial_white)
    add_flag("--lesions", lesions)
    add_flag("--no-mwp", no_mwp)
    add_flag("--wp", wp)
    add_flag("--rp", rp)
    add_flag("--p", p)
    add_flag("--csf", csf)
    add_flag("--amap", amap)
    add_flag("--bids", bids)
    add_flag("--no-correct-folding", no_correct_folding)
    add_flag("--debug", debug)
    add_flag("--fast", fast)

    # Positionals: input files
    args.extend(input_list)

    # Run the process
    # - When stream_output is True and log_file provided, tee output to file and stdout
    # - Otherwise inherit stdout/stderr or suppress
    if stream_output and log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as lf:
            proc = subprocess.Popen(
                args,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                lf.write(line)
            proc.wait()
            rc = proc.returncode
    else:
        stdout = None if stream_output else subprocess.DEVNULL
        stderr = None if stream_output else subprocess.DEVNULL
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
        rc = completed.returncode

        # If requested, also write a basic log when log_file is given (without tee)
        if log_file and not stream_output:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "w") as f:
                f.write("")  # placeholder; detailed logs require stream_output=True

    if check and rc != 0:
        # Surface the exact command to help debugging
        cmd_str = " ".join(shlex.quote(a) for a in args)
        raise subprocess.CalledProcessError(rc, cmd_str)
    return rc


__all__ = ["run_t1prep"]
