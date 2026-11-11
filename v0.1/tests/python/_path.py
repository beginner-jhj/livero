import os

PYTHON_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(PYTHON_TESTS_DIR)
ROOT_DIR = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
INCLUDE_DIR = os.path.join(ROOT_DIR, "include")

if __name__ == "__main__":
    print("ROOT_DIR:", ROOT_DIR)
    print("PYTHON_TESTS_DIR:", PYTHON_TESTS_DIR)
    print("TESTS_DIR:", TESTS_DIR)
    print("SRC_DIR:", SRC_DIR)
    print("INCLUDE_DIR:", INCLUDE_DIR)