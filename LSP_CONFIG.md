# LSP Configuration for Odak Project

## Overview

This project uses **Pylance** (Pyright-based) as the default Python language server in VS Code. The configurations below help the LSP understand the nested package structure (`odak/odak/`) and properly resolve imports.

## Files Created

### 1. `pyrightconfig.json`
Configures Pyright type checker for this project:
- **Execution environment**: Root directory (`.`) added to PYTHONPATH
- **Includes**: `odak/` (source) and `test/` directories
- **Type checking mode**: Basic (balanced between strictness and pragmatism)
- **Python version**: 3.9 (minimum supported)

### 2. `.vscode/settings.json`
VS Code-specific Python settings:
- **Language server**: Pylance
- **Extra paths**: Current directory added for import resolution
- **Diagnostic mode**: Workspace-wide analysis
- **Auto-complete**: Includes current directory in search path

### 3. `setup.cfg`
Central configuration for tools (flake8, mypy, pytest):
- **Flake8**: Aligned with AGENTS.md standards (127 char limit, complexity ≤ 10)
- **Mypy**: Lenient checks for third-party imports
- **Pytest**: Test discovery settings

## How This Fixes LSP Errors

### The Problem
The repository has a nested package structure:
```
odak/              ← Repository root (where setup.py is)
└── odak/          ← Actual Python package
    ├── __init__.py
    ├── tools/
    └── ...
```

Without proper LSP configuration, the language server cannot resolve:
- Absolute imports like `import odak.tools` when package isn't installed in editable mode
- Relative imports like `from ..log import logger` without understanding workspace root

### The Solution
By adding `.` to both:
- `python.analysis.extraPaths` (VS Code settings)
- `executionEnvironments.extraPaths` (Pyright config)

The LSP treats the repository root as a Python path, allowing it to resolve `odak` imports just like your installed virtual environment does.

## Verification

After applying these configurations:

1. **Reload VS Code window**: Press `Ctrl+Shift+P` → "Developer: Reload Window"
2. **Check Python interpreter**: Should show `/home/kaan/venvs/odak/bin/python3`
3. **Open a file** (e.g., `odak/tools/file.py`) - imports should resolve without errors

### Quick Test
Run this in terminal from project root:
```bash
python3 -c "import odak; print(odak.__version__)"
```

Expected output: `0.2.7`

## Additional Notes

- The package is already installed in development mode at `/mnt/yedek/bulut/depolar/odak`
- These configs help LSP match the runtime behavior of your virtual environment
- No changes needed to source code - import patterns remain unchanged

## Alternative Solutions (If Issues Persist)

1. **Ensure editable install is active**:
   ```bash
   pip install -e .
   ```

2. **Add to `.gitignore`** if you want these configs versioned:
   ```
   # LSP configurations (optional - uncomment if you want them in repo)
   # pyrightconfig.json
   # setup.cfg
   # .vscode/
   ```

3. **Create `venv.cfg`** for explicit venv detection:
   ```ini
   [virtualenv]
   include-system-site-packages = false
   ```
