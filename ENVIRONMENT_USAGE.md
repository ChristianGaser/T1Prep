# Environment Usage Guide for T1Prep

This guide explains how to ensure you're using the correct virtual environment when running T1Prep scripts, particularly `cat_viewsurf.py`.

## 🎯 The Problem

When you call `python src/cat_viewsurf.py`, you need to ensure that:
1. The virtual environment is activated
2. All required dependencies (PyQt6, VTK, etc.) are available
3. The correct Python version is being used

## ✅ Solutions

### 1. **Shell Script Wrapper (Recommended)**

Use the dedicated wrapper script for `cat_viewsurf.py`:

```bash
# From anywhere in your project
./scripts/cat_viewsurf_wrapper.sh --help
./scripts/cat_viewsurf_wrapper.sh mesh_file.gii
./scripts/cat_viewsurf_wrapper.sh mesh_file.gii -overlay overlay.gii
```

**What it does:**
- ✅ Automatically detects if environment is activated
- ✅ Activates environment if needed
- ✅ Verifies correct Python path and version
- ✅ Runs the script with all arguments
- ✅ Provides clear status messages

### 2. **Generic Script Runner**

For any Python script in the project:

```bash
# Run any script with automatic environment activation
./scripts/run_with_env.sh src/cat_viewsurf.py --help
./scripts/run_with_env.sh src/segment.py [arguments...]
./scripts/run_with_env.sh src/utils.py [arguments...]
```

### 3. **Manual Environment Activation**

Activate the environment manually, then run scripts:

```bash
# Activate environment
source scripts/activate_env.sh

# Now you can run any script normally
python src/cat_viewsurf.py --help
python src/segment.py [arguments...]
```

### 4. **Direct Environment Activation**

Traditional approach:

```bash
# Navigate to project directory
cd /path/to/T1prep

# Activate environment
source env/bin/activate

# Run scripts
python src/cat_viewsurf.py --help
```

## 🔍 How to Verify You're Using the Correct Environment

### Check Environment Status

```bash
# Check if environment is activated
echo $VIRTUAL_ENV
# Should show: /path/to/T1prep/env

# Check Python path
which python
# Should show: /path/to/T1prep/env/bin/python

# Check Python version
python --version
# Should show: Python 3.9.6
```

### Visual Indicators

- **Prompt indicator**: `(env)` prefix in your terminal prompt
- **Wrapper scripts**: Show status messages with ✅/❌ indicators
- **Error messages**: Clear feedback if environment is not activated

## 🚀 Quick Start Examples

### Running CAT_ViewSurf

```bash
# Method 1: Using wrapper (recommended)
./scripts/cat_viewsurf_wrapper.sh data/templates_surfaces_32k/lh.pial.gii

# Method 2: Using generic runner
./scripts/run_with_env.sh src/cat_viewsurf.py data/templates_surfaces_32k/lh.pial.gii

# Method 3: Manual activation
source scripts/activate_env.sh
python src/cat_viewsurf.py data/templates_surfaces_32k/lh.pial.gii
```

### Running Other Scripts

```bash
# Using generic runner for any script
./scripts/run_with_env.sh src/segment.py input.nii.gz
./scripts/run_with_env.sh src/utils.py --help
```

## 🛠️ Troubleshooting

### Environment Not Found
```
❌ Error: Virtual environment not found: /path/to/env
```
**Solution**: Run `python3 -m venv env` to create the environment

### Missing Dependencies
```
ModuleNotFoundError: No module named 'PyQt6'
```
**Solution**: The environment is not activated or dependencies are missing. Use a wrapper script or run `pip install -r requirements.txt`

### Wrong Python Version
```
Python 3.8.10  # Wrong version
```
**Solution**: Ensure you're using the correct environment. Check with `which python` and `echo $VIRTUAL_ENV`

## 📁 File Structure

```
T1prep/
├── env/                          # Virtual environment
│   ├── bin/
│   │   ├── activate             # Environment activation script
│   │   └── python               # Environment Python executable
│   └── pyvenv.cfg               # Environment configuration
├── scripts/
│   ├── cat_viewsurf_wrapper.sh  # CAT_ViewSurf wrapper
│   ├── run_with_env.sh          # Generic script runner
│   └── activate_env.sh          # Environment activation helper
├── src/
│   ├── cat_viewsurf.py          # Main CAT_ViewSurf script
│   ├── segment.py               # Segmentation script
│   └── utils.py                 # Utility functions
└── requirements.txt             # Python dependencies
```

## 💡 Best Practices

1. **Always use wrapper scripts** when possible - they handle environment activation automatically
2. **Check the prompt** - look for `(env)` prefix to confirm environment is active
3. **Use the generic runner** for any Python script in the project
4. **Keep dependencies updated** - run `pip install -r requirements.txt` after environment changes
5. **Document your workflow** - use these scripts in your own automation

## 🔧 Advanced Usage

### Creating Custom Wrappers

You can create custom wrapper scripts for other tools:

```bash
#!/bin/bash
# Custom wrapper example
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_DIR="$PROJECT_DIR/env"

# Activate environment if needed
if [[ "$VIRTUAL_ENV" != "$ENV_DIR" ]]; then
    source "$ENV_DIR/bin/activate"
fi

# Run your custom script
cd "$PROJECT_DIR"
python src/your_script.py "$@"
```

### Environment Variables

The wrapper scripts set these environment variables:
- `VIRTUAL_ENV`: Path to the virtual environment
- `PATH`: Updated to include environment's bin directory
- `PYTHONPATH`: Updated to include project source directory

---

**Remember**: The wrapper scripts are your best friends! They ensure you always use the correct environment without having to remember activation steps.
