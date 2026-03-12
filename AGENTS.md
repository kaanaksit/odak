# Odak Agent Guidelines

## Build, Lint, and Test Commands

### Running Tests
- Run all tests: `pytest`
- Run a single test file: `pytest test/test_import.py`
- Run a specific test function: `pytest test/test_import.py::test_function_name`
- Run with verbose output: `pytest -v`
- Run tests for perception modules: `pytest test/test_learn_perception_*.py`
- Run tests with coverage: `pytest --cov=odak`

### Linting
- Check critical errors only: `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
- Full check with warnings: `flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics`

### Build Process
- Install dependencies: `pip install -r requirements.txt`
- Install in development mode: `pip install -e .`
- Build package: `python setup.py sdist bdist_wheel`

## Code Style Guidelines

### General Python Style
- Use PEP 8 style conventions
- Lines should not exceed **127 characters**
- Maximum complexity (McCabe) per function is **10**
- Use meaningful variable and function names with descriptive docstrings
- Follow existing codebase patterns for consistency

### Imports (Order Matters)
1. Standard library: `import os`, `import sys`, `import numpy as np`
2. Third-party: `import torch`, `import cv2`, `import plotly`
3. Local package: `from ..log import logger`, `from odak.tools import ...`

### Naming Conventions
- Functions/variables: **snake_case** (e.g., `validate_positive_parameter`)
- Classes: **PascalCase** (e.g., `MultiLayerPerceptron`)
- Constants: **ALL_CAPS** (e.g., `ALLOWED_COMMANDS`, `DANGEROUS_PATTERNS`)
- Test functions: `test_<function_name>` (e.g., `test_validate_positive_parameter`)

### Documentation
- All functions must have **NumPy-style docstrings** with Parameters, Returns, Raises sections
- Class documentation should include purpose and key methods
- Use `...` in Returns/Raises when appropriate
- Include Examples section when useful: `>>> function_call()`

### Error Handling & Security
- Always validate inputs (paths, tensors, parameters)
- Use `raise ValueError("descriptive message")` for invalid inputs
- Use `raise TypeError("expected type, got {type(value).__name__}")` for wrong types
- For torch.load(): **ALWAYS use `weights_only=True`** for security
- Use `odak.log.logger` for logging (info, debug, warning levels)
- Block path traversal (`../`), null bytes (`\x00`), and URL protocols in file paths

### Type Safety & Validation
- Validate all function parameters at the start of functions
- For tensor operations: handle device placement explicitly (CPU/CUDA) with `validate_device()`
- Use `validate_path()` for file/directory paths (blocks traversal, null bytes, UNC paths)
- Use `validate_positive_parameter()` for numeric parameters (supports int, float, numpy arrays, torch Tensors)

### Test Structure
- Location: `/test/` directory
- Naming: `test_<module_or_function>.py`
- Pattern: `def test_<function_name>()` or `class TestSomething(unittest.TestCase)`
- Use temporary directories/files for filesystem operations (`tempfile`, `shutil.rmtree()`)
- All tests must pass before committing code

### Project Structure (Modular)
```
odak/
├── tools/          # General utility functions (path validation, file ops)
├── learn/          # Learning components (models, losses, optimization)
├── wave/           # Wave optics and propagation
├── raytracing/     # Ray tracing operations
├── catalog/        # Component catalogs (lenses, diffusers)
├── measurement/    # Image quality metrics
├── log/            # Logging utilities
└── visualize/      # Visualization tools (plotly)
```

### Frameworks & Libraries
- Primary: **PyTorch** for tensor operations and neural networks
- Image processing: **OpenCV** (cv2), Pillow
- Scientific computing: **NumPy**
- Visualization: **Plotly**
- Progress tracking: **Tqdm**
- File validation: Use `validate_path()` before any file operations

## Implementation Patterns

### Security Checklist for File Operations
Before writing code that handles files/paths:
1. ✅ Call `validate_path(path)` or `validate_path(path, allowed_extensions=[...])`
2. ✅ Check file existence before reading (`os.path.exists`, `os.path.isfile`)
3. ✅ Validate directory is actual directory (`os.path.isdir`)
4. ✅ Use `weights_only=True` in all `torch.load()` calls
5. ✅ Handle exceptions with descriptive error messages

### When Adding New Functions
1. Check existing patterns in same module for consistency
2. Add comprehensive docstring with Parameters/Returns/Raises
3. Include type hints in docstring (not Python hints - follow NumPy style)
4. Add corresponding test in `/test/test_<function>.py`
5. Run `pytest test/test_<function>.py` before committing

### Agent-Specific Guidelines
- Always load AGENTS.md context before implementing new features
- Use ContextScout for discovering project standards and existing patterns
- Check where functions are used (grep) before modifying/deleting
- Ensure backward compatibility when modifying existing functions
- All tensor operations must handle device placement explicitly
- When fixing dimensions, handle irregular sizes (e.g., 2400×4094)
- Maintain parameter validation and defensive coding throughout