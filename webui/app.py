from __future__ import annotations

import glob
import os
import shutil
import socket
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify, redirect, render_template, request, url_for

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT_DIR / "scripts" / "T1Prep"
TEMPLATE_ATLAS_DIR = ROOT_DIR / "src" / "t1prep" / "data" / "templates_MNI152NLin2009cAsym"
SURFACE_ATLAS_DIR = ROOT_DIR / "src" / "t1prep" / "data" / "atlases_surfaces_32k"
DEFAULT_UPLOAD_ROOT = ROOT_DIR / "webui_uploads"
DEFAULT_JOB_ROOT = Path("/tmp") / "webui_jobs"

app = Flask(__name__)

scheduler = BackgroundScheduler()
scheduler.start()

JOBS: Dict[str, dict] = {}
LOCK = threading.Lock()


def ensure_dirs() -> None:
    DEFAULT_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    DEFAULT_JOB_ROOT.mkdir(parents=True, exist_ok=True)


def parse_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if (value.startswith("\"") and value.endswith("\"")) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def load_defaults(path: Path) -> dict:
    defaults: dict[str, str] = {}
    if not path.exists():
        return defaults

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        defaults[key.strip()] = _strip_quotes(value.strip())
    return defaults


def _to_int(value: str, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def format_command_for_log(cmd: list[str]) -> str:
    formatted = []
    for arg in cmd:
        if any(ch in arg for ch in [" ", ",", "\t"]):
            escaped = arg.replace('"', '\\"')
            formatted.append(f'"{escaped}"')
        else:
            formatted.append(arg)
    return " ".join(formatted)


def list_atlas_names() -> list[str]:
    if not TEMPLATE_ATLAS_DIR.exists():
        return []
    names = []
    for path in sorted(TEMPLATE_ATLAS_DIR.glob("*.nii*")):
        name = path.name
        if name.endswith(".nii.gz"):
            name = name[:-7]
        elif name.endswith(".nii"):
            name = name[:-4]
        csv_path = TEMPLATE_ATLAS_DIR / f"{name}.csv"
        if not csv_path.exists():
            continue
        if name and name not in names:
            names.append(name)
    return names


def list_surface_atlas_names() -> list[str]:
    if not SURFACE_ATLAS_DIR.exists():
        return []
    names = []
    for path in sorted(SURFACE_ATLAS_DIR.glob("lh.*.annot")):
        name = path.name
        if name.startswith("lh."):
            name = name[3:]
        if name.endswith(".annot"):
            name = name[:-6]
        if name and name not in names:
            names.append(name)
    return names


def parse_atlas_string(value: str) -> list[str]:
    if not value:
        return []
    parts = [item.strip().strip("'\"") for item in value.split(",")]
    return [item for item in parts if item]


def build_atlas_arg(atlas_items: list[str]) -> Optional[str]:
    if not atlas_items:
        return None
    quoted = [f"'{item}'" for item in atlas_items]
    return ",".join(quoted)


def resolve_defaults_path(value: Optional[str]) -> Path:
    if not value:
        return ROOT_DIR / "T1Prep_defaults.txt"
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    return candidate


def resolve_folder_pattern_inputs(form: dict) -> list[Path]:
    folder_root = form.get("folder_root", "").strip()
    pattern = form.get("file_pattern", "").strip()
    if not folder_root or not pattern:
        return []

    root = Path(folder_root).expanduser()
    if not root.is_absolute():
        root = ROOT_DIR / root
    if not root.exists() or not root.is_dir():
        return []

    matches = [p for p in root.glob(pattern) if p.is_file()]
    filtered = [p for p in matches if p.name.endswith((".nii", ".nii.gz"))]
    return sorted(filtered)


def build_command(
    form: dict,
    input_files: list[Path],
    out_dir: Optional[Path] = None,
    atlas_override: Optional[str] = None,
    surface_atlas_override: Optional[str] = None,
    remove_input: bool = False,
) -> list[str]:
    cmd = [str(SCRIPT_PATH)]

    def add_flag(name: str) -> None:
        if form.get(name):
            cmd.append(name)

    def add_value(name: str) -> None:
        value = form.get(name)
        if value:
            cmd.extend([name, value])

    add_value("--multi")
    add_value("--min-memory")
    add_flag("--debug")

    if out_dir is not None:
        cmd.extend(["--out-dir", str(out_dir)])
    add_flag("--bids")
    add_value("--no-overwrite")
    add_flag("--gz")
    add_flag("--skullstrip-only")
    add_flag("--no-skullstrip")
    add_flag("--no-surf")
    add_flag("--no-seg")
    add_flag("--amap")
    add_flag("--no-sphere-reg")

    # WebUI uses a positive checkbox (save_mwp=1) but the CLI option is inverted
    # (--no-mwp). If the form explicitly sends --no-mwp, honor it.
    if form.get("--no-mwp"):
        cmd.append("--no-mwp")
    else:
        # If save_mwp checkbox is unchecked (missing), disable MWP via --no-mwp.
        if not form.get("save_mwp"):
            cmd.append("--no-mwp")

    add_flag("--no-atlas")
    add_flag("--pial-white")
    add_flag("--hemi")
    add_flag("--wp")
    add_flag("--rp")
    add_flag("--p")
    add_flag("--csf")
    add_flag("--lesions")
    if atlas_override:
        cmd.extend(["--atlas", atlas_override])
    else:
        add_value("--atlas")
    if surface_atlas_override:
        cmd.extend(["--atlas-surf", surface_atlas_override])
    else:
        add_value("--atlas-surf")

    # Remove input files after successful processing (WebUI copies files to upload dir)
    if remove_input:
        cmd.append("--remove-input")

    cmd.extend([str(path) for path in input_files])
    return cmd


def find_parallelize_log() -> Optional[Path]:
    """Find the most recent parallelize log file from /tmp."""
    hostname = socket.gethostname()
    pattern = f"/tmp/parallelize_{hostname}_*.log"
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    # Return the most recently modified one
    return Path(max(candidates, key=os.path.getmtime))


def run_job(job_id: str) -> None:
    with LOCK:
        job = JOBS[job_id]
        job["status"] = "running"
        job["started_at"] = datetime.now().isoformat(timespec="seconds")

    log_path = Path(job["log_path"])
    cmd = job["command"]

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["T1PREP_WEBUI"] = "1"  # Enable progress output in non-TTY mode
    progress_dir = job.get("progress_dir")
    if progress_dir:
        env["PROGRESS_DIR"] = progress_dir

    # Initialize live progress tracking
    with LOCK:
        JOBS[job_id]["live_progress"] = {"vol_percent": 0, "surf_percent": 0, "current_phase": "volume"}

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("Command:\n")
        log_file.write(format_command_for_log(cmd) + "\n\n")
        log_file.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )

        with LOCK:
            JOBS[job_id]["pid"] = process.pid

        assert process.stdout is not None
        last_progress_line = ""
        current_phase = "volume"
        for line in process.stdout:
            stripped = line.strip()

            # Detect phase transitions
            if "Surface Estimation" in stripped:
                current_phase = "surface"
                with LOCK:
                    JOBS[job_id]["live_progress"]["current_phase"] = "surface"
                    JOBS[job_id]["live_progress"]["vol_percent"] = 100
                log_file.write(line)
                log_file.flush()
                continue

            # Filter out repetitive progress bar lines (keep only changes)
            if stripped.startswith("[") and "]" in stripped and "%" in stripped:
                # Extract percentage from progress bar line
                import re
                pct_match = re.search(r"\]\s*(\d{1,3})%", stripped)
                if pct_match:
                    pct = int(pct_match.group(1))
                    with LOCK:
                        if current_phase == "volume":
                            JOBS[job_id]["live_progress"]["vol_percent"] = pct
                        else:
                            JOBS[job_id]["live_progress"]["surf_percent"] = pct

                # Only log if it changed significantly
                if stripped != last_progress_line:
                    last_progress_line = stripped
                    # Only write 100% completion to log
                    if "100%" in stripped:
                        log_file.write(line)
                        log_file.flush()
                continue

            log_file.write(line)
            log_file.flush()
        return_code = process.wait()

        # Mark both phases as 100% when finished successfully
        if return_code == 0:
            with LOCK:
                JOBS[job_id]["live_progress"]["vol_percent"] = 100
                JOBS[job_id]["live_progress"]["surf_percent"] = 100

        # Append parallelize log if available
        parallelize_log = find_parallelize_log()
        if parallelize_log and parallelize_log.exists():
            log_file.write("\n\n--- Parallelize Log ---\n")
            try:
                log_file.write(parallelize_log.read_text(encoding="utf-8"))
            except Exception:
                pass

    with LOCK:
        job = JOBS[job_id]
        job["return_code"] = return_code
        job["finished_at"] = datetime.now().isoformat(timespec="seconds")
        job["status"] = "finished" if return_code == 0 else "failed"
        upload_dir = job.get("upload_dir")

    # Clean up uploaded files after successful completion
    if return_code == 0 and upload_dir:
        upload_path = Path(upload_dir)
        if upload_path.exists():
            try:
                shutil.rmtree(upload_path)
            except Exception:
                pass  # Ignore cleanup errors


def schedule_or_run(job_id: str, run_at: Optional[datetime]) -> None:
    if run_at and run_at > datetime.now():
        scheduler.add_job(run_job, "date", run_date=run_at, args=[job_id])
        with LOCK:
            JOBS[job_id]["status"] = "scheduled"
            JOBS[job_id]["scheduled_for"] = run_at.isoformat(timespec="seconds")
        return

    threading.Thread(target=run_job, args=(job_id,), daemon=True).start()


@app.route("/")
def index():
    default_file = ROOT_DIR / "T1Prep_defaults.txt"
    defaults = load_defaults(default_file)

    atlas_names = list_atlas_names()
    atlas_selected = set(parse_atlas_string(defaults.get("atlas_vol", "")))
    surface_atlas_names = list_surface_atlas_names()
    surface_atlas_selected = set(parse_atlas_string(defaults.get("atlas_surf", "")))

    context = {
        "python": defaults.get("python", ""),
        "multi": defaults.get("multi", ""),
        "min_memory": defaults.get("min_memory", ""),
        "debug": _to_int(defaults.get("debug", "0"), 0) == 1,
        "outdir": defaults.get("outdir", ""),
        "use_bids": _to_int(defaults.get("use_bids_naming", "0"), 0) == 1,
        "no_overwrite": defaults.get("no_overwrite", ""),
        "gz": defaults.get("nii_ext", "nii").lower() == "nii.gz",
        "skullstrip_only": _to_int(defaults.get("skullstrip_only", "0"), 0) == 1,
        "skip_skullstrip": _to_int(defaults.get("skip_skullstrip", "0"), 0) == 1,
        "no_surf": _to_int(defaults.get("estimate_surf", "1"), 1) == 0,
        "no_seg": _to_int(defaults.get("estimate_seg", "1"), 1) == 0,
        "amap": _to_int(defaults.get("use_amap", "0"), 0) == 1,
        "no_sphere_reg": _to_int(defaults.get("estimate_spherereg", "1"), 1) == 0,
        "save_mwp": (
            (_to_int(defaults.get("save_mwp", "1"), 1) == 1)
            and (_to_int(defaults.get("no_mwp", "0"), 0) == 0)
        ),
        "no_atlas": defaults.get("atlas_vol", "") == "",
        "pial_white": _to_int(defaults.get("save_pial_white", "0"), 0) == 1,
        "hemi": _to_int(defaults.get("save_hemi", "0"), 0) == 1,
        "wp": _to_int(defaults.get("save_wp", "0"), 0) == 1,
        "rp": _to_int(defaults.get("save_rp", "0"), 0) == 1,
        "p": _to_int(defaults.get("save_p", "0"), 0) == 1,
        "csf": _to_int(defaults.get("save_csf", "0"), 0) == 1,
        "lesions": _to_int(defaults.get("save_lesions", "0"), 0) == 1,
        "atlas": defaults.get("atlas_vol", ""),
        "atlas_surf": defaults.get("atlas_surf", ""),
        "atlas_names": atlas_names,
        "atlas_selected": atlas_selected,
        "surface_atlas_names": surface_atlas_names,
        "surface_atlas_selected": surface_atlas_selected,
    }

    return render_template("index.html", **context)


@app.route("/atlas-help")
def atlas_help():
    kind = request.args.get("kind", "volume").strip().lower()
    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({"error": "Missing atlas name."}), 400

    if kind == "surface":
        core_name = name
        if core_name.startswith("lh.") or core_name.startswith("rh."):
            core_name = core_name[3:]
        for suffix in (".annot", ".txt"):
            if core_name.endswith(suffix):
                core_name = core_name[: -len(suffix)]
        txt_candidates = [
            SURFACE_ATLAS_DIR / f"{core_name}.txt",
            SURFACE_ATLAS_DIR / f"lh.{core_name}.txt",
            SURFACE_ATLAS_DIR / f"rh.{core_name}.txt",
        ]
    else:
        valid_names = set(list_atlas_names())
        txt_candidates = [TEMPLATE_ATLAS_DIR / f"{name}.txt"]

    if kind != "surface" and name not in valid_names:
        return jsonify({"error": "Unknown atlas."}), 404

    txt_path = next((path for path in txt_candidates if path.exists()), None)
    if txt_path is None:
        return jsonify({"name": name, "text": "No description available.", "truncated": False})

    lines = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    preview = "\n".join(lines[:20]).strip() or "No description available."
    truncated = len(lines) > 20

    return jsonify({"name": name, "text": preview, "truncated": truncated})
@app.route("/submit", methods=["POST"])
def submit():
    ensure_dirs()

    input_files = request.files.getlist("inputs")
    all_inputs = [file for file in input_files if file and file.filename]
    pattern_files = resolve_folder_pattern_inputs(request.form)
    removed_pattern = set(
        p for p in request.form.get("removed_pattern_files", "").split("||") if p
    )
    if removed_pattern:
        pattern_files = [p for p in pattern_files if str(p) not in removed_pattern]

    if not all_inputs and not pattern_files:
        return "No input files selected.", 400

    # Get mandatory output directory
    out_dir_str = request.form.get("out_dir", "").strip()
    if not out_dir_str:
        return "Output directory is required.", 400
    out_dir = Path(out_dir_str)
    if not out_dir.is_absolute():
        out_dir = ROOT_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    job_id = uuid.uuid4().hex

    # Use temp directories for job management, but output goes to user-specified dir
    upload_dir = DEFAULT_UPLOAD_ROOT / job_id
    job_dir = DEFAULT_JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    upload_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "t1prep.log"
    progress_dir = job_dir / "progress"

    progress_dir.mkdir(parents=True, exist_ok=True)

    # Track whether we have uploaded files that should be removed after processing
    has_uploaded_files = len(all_inputs) > 0

    saved_paths = []
    for file_storage in all_inputs:
        safe_name = Path(file_storage.filename).name
        target_path = upload_dir / safe_name
        file_storage.save(target_path)
        saved_paths.append(target_path)

    saved_paths.extend(pattern_files)

    if not saved_paths:
        return "No valid input files selected.", 400

    deduped_paths = []
    seen_paths = set()
    for path in saved_paths:
        path_str = str(path)
        if path_str in seen_paths:
            continue
        seen_paths.add(path_str)
        deduped_paths.append(path)
    saved_paths = deduped_paths

    atlas_items = []
    atlas_items.extend(request.form.getlist("atlas_choice"))

    atlas_files = request.files.getlist("atlas_file")
    for atlas_file in atlas_files:
        if not atlas_file or not atlas_file.filename:
            continue
        if not atlas_file.filename.endswith((".nii", ".nii.gz")):
            continue
        safe_name = Path(atlas_file.filename).name
        target_path = upload_dir / f"atlas_{job_id}_{safe_name}"
        atlas_file.save(target_path)
        atlas_items.append(str(target_path))

    deduped = []
    seen = set()
    for item in atlas_items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)

    atlas_arg = build_atlas_arg(deduped)

    surface_items = []
    surface_items.extend(request.form.getlist("surface_atlas_choice"))

    surface_files = request.files.getlist("surface_atlas_file")
    for surface_file in surface_files:
        if not surface_file or not surface_file.filename:
            continue
        if not surface_file.filename.endswith(".annot"):
            continue
        safe_name = Path(surface_file.filename).name
        target_path = upload_dir / f"surface_atlas_{job_id}_{safe_name}"
        surface_file.save(target_path)
        surface_items.append(str(target_path))

    surface_deduped = []
    surface_seen = set()
    for item in surface_items:
        if item in surface_seen:
            continue
        surface_seen.add(item)
        surface_deduped.append(item)

    surface_atlas_arg = build_atlas_arg(surface_deduped)

    cmd = build_command(
        request.form,
        saved_paths,
        out_dir=out_dir,
        atlas_override=atlas_arg,
        surface_atlas_override=surface_atlas_arg,
        remove_input=False,  # Don't use --remove-input, we'll clean up after job completes
    )
    job_info = {
        "id": job_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "queued",
        "command": cmd,
        "command_display": format_command_for_log(cmd),
        "log_path": str(log_path),
        "progress_dir": str(progress_dir),
        "inputs": [str(path) for path in saved_paths],
        "options": {key: value for key, value in request.form.items()},
        "upload_dir": str(upload_dir) if has_uploaded_files else None,
    }

    with LOCK:
        JOBS[job_id] = job_info

    run_at = parse_datetime(request.form.get("start_at", ""))
    schedule_or_run(job_id, run_at)

    return redirect(url_for("job_detail", job_id=job_id))


@app.route("/jobs")
def jobs():
    with LOCK:
        items = list(JOBS.values())
    items.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return render_template("jobs.html", jobs=items)


@app.route("/jobs/<job_id>")
def job_detail(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
    if not job:
        return "Job not found.", 404

    log_content = ""
    log_path = Path(job["log_path"])
    if log_path.exists():
        log_content = log_path.read_text(encoding="utf-8")

    return render_template("job_detail.html", job=job, log_content=log_content)


@app.route("/jobs/<job_id>/progress")
def job_progress(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    progress_dir = Path(job.get("progress_dir", ""))
    log_path = Path(job["log_path"])

    # Parse log to determine phases and their completion status
    vol_complete = False
    vol_time = ""
    surf_started = False
    surf_complete = False
    surf_time = ""

    if log_path.exists():
        try:
            content = log_path.read_text(encoding="utf-8")
            lines = content.split("\n")
            for line in lines:
                if "Volume Segmentation" in line:
                    pass  # Volume phase started
                elif "Surface Estimation" in line:
                    surf_started = True
                    vol_complete = True  # Volume must be done if surface started
                elif "Finished after" in line:
                    # Extract time from "Finished after: Xmin Ys" or "Finished after Xmin Ys"
                    import re
                    time_match = re.search(r"Finished after:?\s*(.+)", line)
                    if time_match:
                        finish_time = time_match.group(1).strip()
                        if surf_started and not surf_complete:
                            surf_complete = True
                            surf_time = finish_time
                        elif not vol_complete:
                            vol_complete = True
                            vol_time = finish_time
        except Exception:
            pass

    # Read current progress from files
    current_done = 0
    current_total = 0
    failed = False

    if progress_dir.exists():
        progress_files = sorted(progress_dir.glob("job*.progress"))
        for progress_file in progress_files:
            try:
                raw = progress_file.read_text(encoding="utf-8").strip()
                done_str, total_str = raw.split("/", 1)
                done = int(done_str)
                total = int(total_str)
                current_done += done
                current_total += max(total, 1)

                status_file = progress_dir / progress_file.name.replace(".progress", ".status")
                if status_file.exists():
                    try:
                        if int(status_file.read_text(encoding="utf-8").strip() or "0") != 0:
                            failed = True
                    except ValueError:
                        pass
            except Exception:
                continue

    # Determine current phase progress
    # First, check live progress from stdout parsing (works for serial/single-job mode)
    live_progress = job.get("live_progress", {})
    live_vol_percent = live_progress.get("vol_percent", 0)
    live_surf_percent = live_progress.get("surf_percent", 0)
    live_phase = live_progress.get("current_phase", "volume")

    if surf_complete:
        # Both phases complete
        vol_percent = 100
        surf_percent = 100
    elif surf_started:
        # Surface estimation in progress
        vol_percent = 100
        # Use progress files if available, otherwise use live progress
        if current_total > 0:
            surf_percent = int(min(100, max(0, (current_done * 100) // current_total)))
        else:
            surf_percent = live_surf_percent
    elif vol_complete:
        # Volume done, surface not started yet (brief transition)
        vol_percent = 100
        surf_percent = 0
    else:
        # Volume segmentation in progress
        # Use progress files if available, otherwise use live progress from stdout
        if current_total > 0:
            vol_percent = int(min(100, max(0, (current_done * 100) // current_total)))
        else:
            vol_percent = live_vol_percent
        surf_percent = 0

    return jsonify(
        {
            "available": True,
            "volume": {
                "percent": vol_percent,
                "complete": vol_complete,
                "time": vol_time,
                "failed": failed and not surf_started,
            },
            "surface": {
                "percent": surf_percent,
                "complete": surf_complete,
                "time": surf_time,
                "started": surf_started,
                "failed": failed and surf_started,
            },
            "done": current_done,
            "total": current_total,
            "failed": failed,
        }
    )


@app.route("/jobs/<job_id>/status")
def job_status(job_id: str):
    """Return current job status and phase information."""
    with LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    status = job.get("status", "unknown")
    phase = "Initializing"

    # Parse phase from log file
    log_path = Path(job["log_path"])
    if log_path.exists():
        try:
            content = log_path.read_text(encoding="utf-8")
            # Detect phase from log content
            if "Surface Estimation" in content:
                phase = "Surface Estimation"
            elif "Volume Segmentation" in content:
                phase = "Volume Segmentation"
            elif "Check" in content:
                phase = "Checking files"
        except Exception:
            pass

    return jsonify(
        {
            "status": status,
            "phase": phase,
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
            "return_code": job.get("return_code"),
        }
    )


@app.route("/resolve-pattern")
def resolve_pattern():
    folder_root = request.args.get("folder_root", "").strip()
    file_pattern = request.args.get("file_pattern", "").strip()
    if not folder_root or not file_pattern:
        return jsonify({"files": []})

    fake_form = {"folder_root": folder_root, "file_pattern": file_pattern}
    files = resolve_folder_pattern_inputs(fake_form)
    return jsonify({"files": [str(path) for path in files]})


@app.route("/resolve-dirname")
def resolve_dirname():
    """Return the directory name from a file path or folder path.

    Query parameters:
        path: A file or folder path

    Returns:
        JSON with 'dirname' key containing the resolved directory,
        or an empty string if the path is invalid.
    """
    raw_path = request.args.get("path", "").strip()
    if not raw_path:
        return jsonify({"dirname": ""})

    p = Path(raw_path).expanduser()
    if not p.is_absolute():
        p = ROOT_DIR / p

    # If path exists and is a directory, use it directly
    if p.exists() and p.is_dir():
        return jsonify({"dirname": str(p)})

    # If path exists and is a file, return its parent
    if p.exists() and p.is_file():
        return jsonify({"dirname": str(p.parent)})

    # If path doesn't exist, try to determine if it looks like a file or directory
    # A path ending with / or without extension is likely a directory
    if raw_path.endswith("/") or raw_path.endswith(os.sep):
        return jsonify({"dirname": str(p)})

    # Check if it looks like a NIfTI file path
    if p.name.endswith((".nii", ".nii.gz", ".gz")):
        return jsonify({"dirname": str(p.parent)})

    # Otherwise return the path as-is (assume it's a directory)
    return jsonify({"dirname": str(p)})


@app.route("/get-default-output")
def get_default_output():
    """Return a default output directory for uploaded files.

    Returns:
        JSON with 'default_output' key containing a suggested output directory.
    """
    # Use user's home directory with a T1Prep_output subfolder as default
    home = Path.home()
    default_output = home / "T1Prep_output"
    return jsonify({"default_output": str(default_output)})


import platform
import webbrowser


def open_chrome_app_mode(url: str, width: int = 1100, height: int = 900) -> bool:
    """Open Google Chrome in app mode with specified window size.

    Args:
        url: The URL to open.
        width: Window width in pixels.
        height: Window height in pixels.

    Returns:
        True if Chrome was successfully launched, False otherwise.
    """
    system = platform.system()

    # Chrome paths for different operating systems
    if system == "Darwin":  # macOS
        chrome_paths = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
    elif system == "Linux":
        chrome_paths = [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
            "/snap/bin/chromium",
        ]
    else:
        # Unsupported OS, fall back to default browser
        webbrowser.open(url)
        return False

    # Find the first available Chrome executable
    chrome_path = None
    for path in chrome_paths:
        if os.path.exists(path):
            chrome_path = path
            break

    if chrome_path is None:
        # Chrome not found, fall back to default browser
        print("Chrome not found, opening in default browser...")
        webbrowser.open(url)
        return False

    # Launch Chrome in app mode
    try:
        cmd = [
            chrome_path,
            f"--app={url}",
            f"--window-size={width},{height}",
            "--window-position=100,100",
        ]
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except Exception as e:
        print(f"Failed to open Chrome: {e}")
        webbrowser.open(url)
        return False


if __name__ == "__main__":
    import sys
    ensure_dirs()
    host = "127.0.0.1"
    port = 5000
    # Parse --port argument
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            try:
                port = int(sys.argv[i + 1])
            except ValueError:
                print(f"Invalid port: {sys.argv[i + 1]}")
                sys.exit(1)
    url = f"http://{host}:{port}"
    # Check if --no-browser flag is passed
    open_browser = "--no-browser" not in sys.argv
    if open_browser:
        # Open Chrome in app mode after a short delay to allow server to start
        def delayed_open():
            import time
            time.sleep(1.0)
            open_chrome_app_mode(url, width=1100, height=900)
        threading.Thread(target=delayed_open, daemon=True).start()
    app.run(host=host, port=port, debug=True, use_reloader=False)
