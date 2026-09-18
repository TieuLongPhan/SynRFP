# Repository scripts

Run these from any working directory:

- `lint.sh`: file-size policy, Sphinx docstring style and Flake8
  (default paths: `synrfp tests scripts Experiments`; pass paths to narrow it);
- `pytest.sh`: the full test suite, or pytest with the supplied arguments;
- `build_doc.sh`: Sphinx HTML build of `doc/` into `docs/`;
- `check_package_artifacts.py`: inspect `dist/` (wheel + sdist) and smoke-test
  the installed wheel. CI runs it after `python -m build`.

`check_python_file_size.py` (max 1000 non-docstring lines per file) and
`python_file_size_baseline.json` (grandfathered exceptions; currently none)
back the size policy. `check_docstring_style.py` requires Sphinx field lists
(`:param:`, `:type:`, `:return:`, `:rtype:`, `:raises X:`) and rejects
Google/NumPy sections.

Set `PYTHON=/path/to/python` to use a specific interpreter.
