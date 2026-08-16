# Installing T1Prep

T1Prep is a pure-Python package. Pick whichever route suits you: `pip` for an
existing Python, the bootstrapper for a self-contained source tree, WSL on
Windows, or Docker.

- [pip / PyPI](#pip--pypi-recommended-for-python-users)
- [Bash bootstrapper](#bash-bootstrapper-full-source-tree)
- [Windows via WSL](#windows-installation-via-wsl-recommended)
- [Manual installation](#manual-installation)
- [Docker](#docker)

Once it is installed, see [usage.md](usage.md) for running it and
[viewers.md](viewers.md) for the viewers. Back to the [README](../README.md).

## Requirements

 [Python 3.9-3.12](https://www.python.org/downloads/) is required (3.10+ recommended), and all necessary libraries are automatically installed the first time T1Prep is run or is called with the flag "--install".

> **Why prefer 3.10+?** Python 3.9 works, but PyTorch publishes no wheels for it after 2.8, so a 3.9 install is pinned to PyTorch 2.8. The GPU (MPS) kernels T1Prep relies on — `max_pool3d_with_indices`, `avg_pool3d` and `grid_sampler_3d` — only arrived in PyTorch 2.9; on 2.8 they silently fall back to the CPU. On Apple Silicon that is worth a measurable amount: one subject takes **3:02 min on Python 3.9 / PyTorch 2.8 versus 2:39 min on Python 3.12 / PyTorch 2.13**. On Linux/CUDA/CPU the two stacks perform the same, so 3.9 is a fine choice there.

---

## pip / PyPI (Recommended for Python users)

If you have Python 3.9–3.12 available (3.10+ recommended), install directly from PyPI:

```bash
# Latest release
python3 -m pip install T1Prep

# Pin a specific version (any PEP 440 spec)
python3 -m pip install "T1Prep==0.4.4"

# Optional: pipx keeps T1Prep isolated in its own venv
pipx install T1Prep
```

> **Multiple Python versions?** Install with the *exact* interpreter you intend
> to run T1Prep with, e.g. `python3.12 -m pip install T1Prep` (T1Prep requires
> Python 3.9–3.12; a `pip` bound to a Python older than 3.9 will refuse to
> install). On macOS, prefer 3.10+ — a 3.9 install is pinned to PyTorch 2.8 and
> runs slower (see [Requirements](#requirements)).
> T1Prep first tries the interpreter it was installed into and the newest
> Python 3.9–3.12 on your `PATH`. If auto-detection picks the wrong one, point
> it explicitly — either per invocation (`T1Prep --python /path/to/python …`) or
> for the whole session (`export T1PREP_PYTHON=/path/to/python`). Using `pipx`
> or a dedicated venv sidesteps the ambiguity entirely.

Model weights are not bundled with the wheel; they are fetched lazily on
the first run into your user cache (or downloaded ahead of time with
`t1prep-download-models`).

Use it from Python:

```python
from t1prep import run_t1prep
run_t1prep("/path/to/sub-01_T1w.nii.gz")
```

…or from the command line. A `pip install` places every entry point into the
active environment's `bin/` directory, so once that directory is on your `PATH`
the following commands are available:

```bash
T1Prep file.nii.gz                              # main CLI (batch + parallel, --multi)
t1prep-run --input file.nii.gz --out-dir out/   # single-subject Python entry
t1prep-ui                                       # web UI
CAT_SurfView lh.central.gii                     # surface viewer
CAT_VolView T1.nii.gz                           # volume viewer
t1prep-download-models                          # fetch the model weights now
t1prep-make-apps                                # macOS: build the viewer .app bundles
```

> The `T1Prep` command is the bash orchestrator (full features including
> `--multi` batch parallelism); `t1prep-run` is the equivalent single-subject
> Python entry. Add the environment's `bin/` to `PATH` and you never need to
> call anything from the source `scripts/` folder.

What to do with them: [usage.md](usage.md) for the pipeline,
[viewers.md](viewers.md) for the viewers, [tools.md](tools.md) for the rest.

## Bash bootstrapper (full source tree)

To install everything (the package, all dependencies, and every entry point)
into a self-contained environment, use the bundled bootstrapper:

```bash
curl -fsSL https://raw.githubusercontent.com/ChristianGaser/T1Prep/refs/heads/main/scripts/install.sh | bash
```

It creates a virtualenv, installs T1Prep into it, and prints the `export PATH`
line to add its `bin/` directory to your shell. After that, run `T1Prep`,
`t1prep-ui`, etc. directly.

The installer will interactively prompt you to:
1. **Select a version**: Latest release, development (main branch), or choose from available releases
2. **Choose installation directory**: Current folder, temporary folder, or custom path
3. **Select a Python interpreter**: if several supported Pythons (3.9–3.12) are
   found, pick which one to install into (the newest is offered as the default).
   If only one is found it is used automatically.

#### Non-Interactive Installation
Use environment variables to skip the interactive prompts:
```bash
# Install latest release to current directory
T1PREP_VERSION=latest T1PREP_INSTALL_DIR="$PWD/T1Prep" \
  curl -fsSL https://raw.githubusercontent.com/ChristianGaser/T1Prep/refs/heads/main/scripts/install.sh | bash

# Install specific version to custom directory
T1PREP_VERSION=v1.0.0 T1PREP_INSTALL_DIR=/opt/T1Prep \
  curl -fsSL https://raw.githubusercontent.com/ChristianGaser/T1Prep/refs/heads/main/scripts/install.sh | bash

# Pin the Python interpreter to install into (skips the Python prompt)
T1PREP_VERSION=latest T1PREP_INSTALL_DIR="$PWD/T1Prep" T1PREP_PYTHON=python3.12 \
  curl -fsSL https://raw.githubusercontent.com/ChristianGaser/T1Prep/refs/heads/main/scripts/install.sh | bash
```

| Environment Variable | Description |
|---------------------|-------------|
| `T1PREP_VERSION` | Release tag (e.g., `v1.0.0`) or `latest` |
| `T1PREP_INSTALL_DIR` | Absolute path for installation |
| `T1PREP_PYTHON` | Python interpreter to install into (e.g., `python3.12` or an absolute path); must be Python 3.9–3.12 |

## Windows Installation via WSL (Recommended)

T1Prep requires a Linux environment to run. On Windows, we recommend using **Windows Subsystem for Linux (WSL)**, which provides a complete Linux environment with full compatibility.

#### WSL Requirements

| Windows Version | WSL Support |
|-----------------|-------------|
| Windows 11 (all versions) | WSL 2 ✓ |
| Windows 10 version 2004+ (Build 19041+) | WSL 2 ✓ |
| Windows 10 version 1903-1909 | WSL 2 (with manual kernel update) |
| Windows 10 version 1607-1903 | WSL 1 only |
| Windows Server 2019+ | WSL ✓ |

#### Installing WSL and T1Prep

1. **Install WSL** (run PowerShell as Administrator):
   ```powershell
   wsl --install
   ```
   This installs WSL 2 with Ubuntu by default. Restart your computer when prompted.

2. **Open Ubuntu** from the Start menu and complete the initial setup (create username/password).

3. **Install T1Prep** inside WSL (Ubuntu terminal):
   ```bash
   curl -fsSL https://raw.githubusercontent.com/ChristianGaser/T1Prep/main/scripts/install.sh | bash
   ```

4. **Access Windows files** from WSL at `/mnt/c/` (C: drive), `/mnt/d/` (D: drive), etc.:
   ```bash
   # Process a file from your Windows Documents folder
   T1Prep --out-dir /mnt/c/Users/YourName/T1Prep_output /mnt/c/Users/YourName/Documents/scan.nii.gz
   ```

#### Alternative: Docker on Windows

If you prefer not to install WSL directly, you can use Docker Desktop for Windows (which uses WSL 2 internally):

```powershell
docker run --rm -it -v C:\path\to\data:/data t1prep:latest --out-dir /data/out /data/file.nii.gz
```

See the [Docker](#docker) section for build instructions.

## Manual Installation

If you do not want pip to manage the environment for you (e.g. for an
offline or air-gapped setup), you can install the dependencies yourself.

**From PyPI into your own virtualenv:**
```bash
python3.12 -m venv env
source env/bin/activate         # or add env/bin to PATH
pip install T1Prep
```

**From a source checkout (developers):**
```bash
git clone https://github.com/ChristianGaser/T1Prep.git
cd T1Prep
python3.12 -m venv env
source env/bin/activate
pip install -e .                    # editable; tracks local edits
# – or –
pip install -r requirements.txt     # dependencies only (no T1Prep itself)
```

Either way the entry points (`T1Prep`, `t1prep-ui`, `t1prep-run`,
`CAT_SurfView`, `CAT_VolView`, `t1prep-make-apps`, `t1prep-download-models`) are
placed in `env/bin`. Activating
the venv — or adding `env/bin` to your `PATH` — is all that is needed; the
source `scripts/` folder is only a dev fallback and should not be put on `PATH`.

**Source ZIP plus bash bootstrapper** (kept for parity with older docs):
```bash
unzip T1Prep_$version.zip -d your_installation_folder
./scripts/T1Prep --python python3.12 --install   # creates env/ and installs into it
export PATH="$PWD/env/bin:$PATH"                  # then use T1Prep, t1prep-ui, …
```

## Docker

A Dockerfile is provided that installs T1Prep from PyPI on top of a slim
Python 3.12 base image. No source checkout is needed — the image is a
pure-Python distribution with model weights fetched lazily on first run.

### Build

**Latest release from PyPI:**
```bash
docker build -t t1prep:latest .
```

**Pinned release:**

```bash
docker build \
  --build-arg T1PREP_VERSION=0.4.4 \
  -t t1prep:0.4.4 .
```

The `T1PREP_VERSION` build-arg accepts any PEP 440 version string (no
leading `v`) and is forwarded to `pip install "T1Prep==..."`. Leave it
unset to track the latest release on PyPI.

### Run

Mount your data directory into the container (replace /path/to/data with your folder):

```bash
docker run --rm -it \
  -v /path/to/data:/data \
  t1prep:latest \
  --out-dir /data/out /data/file.nii.gz
```
Append `--gpus all` to `docker run` to enable GPU acceleration when available.

### Memory & performance

Make sure that the container has at least 10-16 GB of RAM available. If you are using Docker Desktop/WSL2, increase the VM memory in the settings if needed. If you receive an error message stating that there is no space left on the device: /tmp/, you can try the following:
If you obtain an error that no space is left on device: /tmp/ you can try that:
```bash
docker run --rm -it \
  --tmpfs /tmp:rw,exec,nosuid,nodev,size=16g \
  -v /path/to/data:/data \
  t1prep:latest \
  --out-dir /data/out /data/file.nii.gz
```

## Environment helpers

A source checkout carries wrappers that activate the bundled environment for
you — `scripts/activate_env.sh` and `scripts/run_with_env.sh`. See
[ENVIRONMENT_USAGE.md](../ENVIRONMENT_USAGE.md) and
[scripts/README.md](../scripts/README.md).
