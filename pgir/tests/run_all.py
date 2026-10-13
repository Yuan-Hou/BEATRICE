"""Run the full pgir test suite:  cd <package_root> && python -m pgir.tests.run_all"""
import sys
import unittest

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(start_dir=__file__.rsplit("/", 1)[0],
                                                pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
