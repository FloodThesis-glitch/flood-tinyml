"""Setup check: confirm every package imports and report its version.

Run from the project root:  python check_setup.py
"""
from __future__ import annotations

import importlib
import sys

# (module name to import, label shown in the report)
PACKAGES: list[tuple[str, str]] = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("tensorflow", "tensorflow"),
    ("keras", "keras"),
    ("emlearn", "emlearn"),
    ("serial", "pyserial"),
    ("yaml", "pyyaml"),
    ("jupyter_core", "jupyter (core)"),
    ("ipykernel", "ipykernel"),
]


def check(module_name: str) -> tuple[bool, str]:
    """Try to import a module. Return (success, version-or-error text)."""
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # report any failure instead of crashing
        return False, f"{type(exc).__name__}: {exc}"
    return True, str(getattr(module, "__version__", "version not exposed"))


def main() -> int:
    """Check the interpreter and all packages. Return 0 if all is well."""
    in_venv = sys.prefix != sys.base_prefix
    print(f"Python {sys.version.split()[0]}")
    print(f"Interpreter: {sys.executable}")
    print(f"Inside a virtual environment: {in_venv}\n")

    failures = 0 if in_venv else 1
    for module_name, label in PACKAGES:
        ok, info = check(module_name)
        status = "OK  " if ok else "FAIL"
        print(f"[{status}] {label:<16} {info}")
        if not ok:
            failures += 1

    print()
    if failures:
        print(f"{failures} problem(s) found. Fix these before continuing.")
        return 1
    print("All checks passed. The environment is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
