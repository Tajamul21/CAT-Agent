"""Test-wide safety: never let a test write into the real data folder (usage ledger, logs, outputs).

pytest_configure runs before test modules are imported, so every Config built during the tests points
at a throwaway directory unless a test sets its own OPHBENCH_DATA_DIR / OPHBENCH_LOGS_DIR.
"""
import os
import tempfile


def pytest_configure(config):
    base = tempfile.mkdtemp(prefix="ophbench_tests_")
    os.environ.setdefault("OPHBENCH_DATA_DIR", os.path.join(base, "data"))
    os.environ.setdefault("OPHBENCH_LOGS_DIR", os.path.join(base, "logs"))
    os.environ.setdefault("OPHBENCH_PACKAGES_DIR", os.path.join(base, "packages"))
