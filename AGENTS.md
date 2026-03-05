# Odak Agent Guidelines

## Build, Lint, and Test Commands

### Running Tests
- Run all tests: `pytest`
- Run a single test file: `pytest test/test_import.py`
- Run a specific test function: `pytest test/test_import.py::test`
- Run with verbose output: `pytest -v`

### Linting
- Check code style with flake8: `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
- Check with warnings as errors: `flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics`

### Build Process
- Install dependencies: `pip install -r requirements.txt`
- Install in development mode: `pip install -e .`
- Build package: `python setup.py sdist bdist_wheel`

## Code Style Guidelines

### General Python Style
- Use PEP 8 style conventions
- Lines should not exceed 127 characters
- Maximum complexity per function is 10
- Use meaningful variable and function names with descriptive docstrings
- Follow the existing codebase structure and naming convention

### Imports
- Standard library imports first (e.g., `import os`, `import sys`)
- Third-party library imports next (e.g., `import torch`, `import numpy`)
- Local package imports last (e.g., `import odak`)

### Naming Conventions
- Use snake_case for functions and variables
- Use PascalCase for classes
- Use ALL_CAPS for constants
- Use descriptive names that explain intent clearly

### Error Handling
- Use try/except blocks for error handling where appropriate
- Validate inputs with descriptive error messages
- Log important events using `odak.log.logger` when available

### Documentation
- All functions should have docstrings in NumPy style
- Class documentation should include class purpose and key methods
- Module-level documentation should be included in __init__.py when applicable

### Testing
- Test files are located in the `/test/` directory
- Each test file should be named `test_<module_or_function>.py`
- Tests should follow the pattern: `def test_<function_name>()`
- Tests can use pytest fixtures for common setup operations
- All tests should pass before committing code

### Project Structure
This project uses a modular structure:
- Main odak module in `odak/` directory
- Submodules in `odak/learn/`, `odak/tools/`, `odak/wave/`, `odak/raytracing/`, etc.
- Data files in `test/data/` for test fixtures

### Frameworks and Libraries
- Primary framework: PyTorch
- Image processing: OpenCV, Pillow
- Visualization: Plotly
- Scientific computing: NumPy
- Progress bars: Tqdm