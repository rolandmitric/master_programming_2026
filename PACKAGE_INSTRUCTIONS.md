# How To Turn A Directory Of Python Files Into A Package

## Goal

Take a directory of Python source files and make it importable with:

```python
from mypkg.module import some_function
```

## Minimal Working Layout

```text
myproject/
  pyproject.toml
  src/
    mypkg/
      __init__.py
      module.py
      other_module.py
```

## Steps

1. Create a project root directory.
2. Put your Python files inside `src/mypkg/`.
3. Add an empty `__init__.py` inside `src/mypkg/`.
4. Add a `pyproject.toml` file in the project root.
5. Run `pip install -e .` from the project root.

## Minimal `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "mypkg"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
```

## Minimal Example

File: `src/mypkg/module.py`

```python
def hello():
    return "hello"
```

File: `src/mypkg/__init__.py`

```python
# can be empty
```

## Install

From the project root:

```bash
pip install -e .
```

The `-e` means editable install. You can keep changing the source files without reinstalling every time.

## Test It

```bash
python -c "from mypkg.module import hello; print(hello())"
```

If everything is set up correctly, Python will import your package and print:

```text
hello
```

## What Each Part Does

- `mypkg/` is the package name you import.
- `__init__.py` tells Python this directory is a package.
- `module.py` is a module inside the package.
- `pyproject.toml` tells Python packaging tools how to install the code.
- `pip install -e .` makes the package importable in the current environment.

## Mental Model

- A file like `module.py` becomes a module.
- A directory like `mypkg/` with `__init__.py` becomes a package.
- After installation, `src/mypkg/module.py` can be imported as `mypkg.module`.

## Simplest Alternative

You can also skip the `src/` layout and use:

```text
myproject/
  pyproject.toml
  mypkg/
    __init__.py
    module.py
```

This also works, but the `src/` layout is a cleaner default for real projects.
