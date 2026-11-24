# Contributing to Numerical Methods in Python

Contributions and bug reports are welcome!

## Development Workflow

1. Fork the repository and create a branch:
   ```bash
   git checkout -b feat/your-method
   ```
2. Set up a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run the unit test suite before and after making changes:
   ```bash
   python3 -m unittest discover -s tests -v
   ```

## Guidelines

- Validate mathematical preconditions (positive step sizes, distinct nodes, domain constraints) and raise descriptive `ValueError` exceptions.
- Support `--no-plot` and `--save-plot` flags on plotting scripts so they remain runnable in headless CI environments.
- Add analytical or reference-solution assertions in `tests/test_numerical_methods.py` for every new algorithm.
