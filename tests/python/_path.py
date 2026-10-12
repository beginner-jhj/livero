import os

PYTHON_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(PYTHON_TESTS_DIR)
ROOT_DIR = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
INCLUDE_DIR = os.path.join(ROOT_DIR, "include")
