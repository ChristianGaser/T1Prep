from __future__ import annotations

import os
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


def resolve_defaults_path(value: Optional[str]) -> Path:
    if not value:
        return ROOT_DIR / "T1Prep_defaults.txt"
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    return candidate


def resolve_storage_root(form: dict) -> Optional[Path]:
    storage_root = form.get("storage_root")
    if storage_root:
        root = Path(storage_root)
        if not root.is_absolute():
            root = ROOT_DIR / root
        return root

    out_dir = form.get("--out-dir")
    if not out_dir:
        return None
    root = Path(out_dir)
    if not root.is_absolute():
        root = ROOT_DIR / root
    return root


def build_command(
    form: dict,
    input_files: list[Path],
    storage_root: Optional[Path] = None,
) -> list[str]:
    cmd = [str(SCRIPT_PATH)]

    def add_flag(name: str) -> None:
        if form.get(name):
            cmd.append(name)

    def add_value(name: str) -> None:
        value = form.get(name)
        if value:
            cmd.extend([name, value])

    add_value("--defaults")
    add_value("--python")
    add_value("--multi")
    add_value("--min-memory")
    add_flag("--debug")

    if form.get("--out-dir"):
        add_value("--out-dir")
    elif storage_root is not None:
        cmd.extend(["--out-dir", str(storage_root)])
    add_flag("--bids")
    add_value("--no-overwrite")
    add_flag("--gz")
    add_flag("--skullstrip-only")
    add_flag("--no-skullstrip")
    add_flag("--no-surf")
    add_flag("--no-seg")
    add_flag("--no-sphere-reg")
    add_flag("--no-mwp")
    add_flag("--no-atlas")
    add_flag("--pial-white")
    add_flag("--hemi")
    add_flag("--wp")
    add_flag("--rp")
    add_flag("--p")
    add_flag("--csf")
    add_flag("--lesions")
    add_value("--atlas")
    add_value("--atlas-surf")

    cmd.extend([str(path) for path in input_files])
    return cmd


def run_job(job_id: str) -> None:
    with LOCK:
        job = JOBS[job_id]
        job["status"] = "running"
        job["started_at"] = datetime.now().isoformat(timespec="seconds")

    log_path = Path(job["log_path"])
    cmd = job["command"]

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    progress_dir = job.get("progress_dir")
    if progress_dir:
        env["PROGRESS_DIR"] = progress_dir

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
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()

    with LOCK:
        job = JOBS[job_id]
        job["return_code"] = return_code
        job["finished_at"] = datetime.now().isoformat(timespec="seconds")
        job["status"] = "finished" if return_code == 0 else "failed"


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
    requested_defaults = request.args.get("defaults")
    default_file = resolve_defaults_path(requested_defaults)
    defaults = load_defaults(default_file)
    defaults_missing = bool(requested_defaults and not defaults)

    defaults_options = [
        str(ROOT_DIR / "T1Prep_defaults.txt"),
    ]
    if requested_defaults and str(default_file) not in defaults_options:
        defaults_options.append(str(default_file))

    context = {
        "defaults_file": str(default_file),
        "defaults_missing": defaults_missing,
        "defaults_options": defaults_options,
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
        "no_sphere_reg": _to_int(defaults.get("estimate_spherereg", "1"), 1) == 0,
        "no_mwp": _to_int(defaults.get("save_mwp", "1"), 1) == 0,
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
    }

    return render_template("index.html", **context)


@app.route("/submit", methods=["POST"])
def submit():
    ensure_dirs()

    input_files = request.files.getlist("inputs")
    all_inputs = [file for file in input_files if file and file.filename]

    if not all_inputs:
        return "No input files selected.", 400

    job_id = uuid.uuid4().hex
    storage_root = resolve_storage_root(request.form)
    if storage_root is None:
        upload_dir = DEFAULT_UPLOAD_ROOT / job_id
        job_dir = DEFAULT_JOB_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        upload_dir.mkdir(parents=True, exist_ok=True)
        log_path = job_dir / "t1prep.log"
        progress_dir = job_dir / "progress"
    else:
        storage_root.mkdir(parents=True, exist_ok=True)
        upload_dir = storage_root
        job_dir = storage_root
        log_path = job_dir / f"t1prep_{job_id}.log"
        progress_dir = job_dir / f"progress_{job_id}"

    progress_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for file_storage in all_inputs:
        safe_name = Path(file_storage.filename).name
        if storage_root is None:
            target_path = upload_dir / safe_name
        else:
            target_path = upload_dir / f"{job_id}_{safe_name}"
        file_storage.save(target_path)
        saved_paths.append(target_path)

    if not saved_paths:
        return "No valid input files uploaded.", 400

    cmd = build_command(request.form, saved_paths, storage_root)
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
    if not progress_dir.exists():
        return jsonify({"available": False})

    progress_files = sorted(progress_dir.glob("job*.progress"))
    if not progress_files:
        return jsonify({"available": False})

    total_done = 0
    total_items = 0
    failed = False

    for progress_file in progress_files:
        try:
            raw = progress_file.read_text(encoding="utf-8").strip()
            done_str, total_str = raw.split("/", 1)
            done = int(done_str)
            total = int(total_str)
        except Exception:
            continue

        total_done += done
        total_items += max(total, 1)

        status_file = progress_dir / progress_file.name.replace(".progress", ".status")
        if status_file.exists():
            try:
                if int(status_file.read_text(encoding="utf-8").strip() or "0") != 0:
                    failed = True
            except ValueError:
                pass

    if total_items <= 0:
        return jsonify({"available": False})

    percent = int(min(100, max(0, (total_done * 100) // total_items)))
    return jsonify(
        {
            "available": True,
            "percent": percent,
            "done": total_done,
            "total": total_items,
            "failed": failed,
            "label": "T1Prep",
        }
    )


if __name__ == "__main__":
    ensure_dirs()
    app.run(host="127.0.0.1", port=5000, debug=True)
