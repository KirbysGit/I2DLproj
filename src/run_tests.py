import pytest
import sys
from pathlib import Path

# Define the root test directory
ROOT_DIR = Path(__file__).parent  # The directory where run_tests.py is located

# Define test directories
TEST_DIRS = ["model", "training", "utils"]

def find_test_files():
    """
    Recursively finds all test files in the specified directories.
    """
    test_files = []
    for test_dir in TEST_DIRS:
        test_path = ROOT_DIR / test_dir
        if test_path.exists():
            test_files.extend(str(path) for path in test_path.rglob("test_*.py"))  # Recursively find test files

    return test_files

def run_tests():
    """
    Runs all tests in the project using pytest.
    """
    print("\n================ RUNNING TESTS =================\n")

    # Get all test files
    test_files = find_test_files()

    if not test_files:
        print("No test files found! Make sure your tests start with 'test_' and are inside the correct directories.")
        sys.exit(1)

    # Run tests using pytest
    exit_code = pytest.main(test_files)
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests()
