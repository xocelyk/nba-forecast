#!/usr/bin/env python3
"""
Test runner script for NBA simulation system.

Usage:
    python run_tests.py              # Run all tests with coverage
    python run_tests.py --fast       # Run tests without coverage
    python run_tests.py --unit       # Run only unit tests
    python run_tests.py --integration # Run only integration tests
    python run_tests.py --coverage   # Generate coverage report only
"""

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path


def run_command(cmd, description=""):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}")
    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def run_tests(args):
    """Run test suite based on arguments."""
    base_cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]

    if not args.fast:
        base_cmd.extend(
            [
                "--cov=.",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml",
            ]
        )

    if args.unit:
        base_cmd.extend(["-m", "unit"])
    elif args.integration:
        base_cmd.extend(["-m", "integration"])

    success = run_command(base_cmd, "Running test suite")

    if success and not args.fast:
        coverage_file = Path("htmlcov/index.html")
        if coverage_file.exists():
            print(f"\n📊 Coverage report generated: {coverage_file.absolute()}")
            if args.open_coverage:
                webbrowser.open(f"file://{coverage_file.absolute()}")

    return success


def check_coverage_threshold(threshold=50):
    """Check if coverage meets minimum threshold."""
    try:
        import xml.etree.ElementTree as ET

        tree = ET.parse("coverage.xml")
        root = tree.getroot()
        coverage = float(root.attrib["line-rate"]) * 100

        print(f"\n📈 Current coverage: {coverage:.2f}%")

        if coverage >= threshold:
            print(f"✅ Coverage threshold met ({threshold}%)")
            return True
        else:
            print(f"⚠️  Coverage below threshold ({threshold}%)")
            return False

    except FileNotFoundError:
        print("❌ Coverage report not found. Run tests with coverage first.")
        return False
    except Exception as e:
        print(f"❌ Error reading coverage report: {e}")
        return False


def run_linting():
    """Run code quality checks."""
    commands = [
        (
            [sys.executable, "-m", "black", "--check", "."],
            "Code formatting check (black)",
        ),
        (
            [sys.executable, "-m", "isort", "--check-only", "."],
            "Import sorting check (isort)",
        ),
        ([sys.executable, "-m", "flake8", "."], "Linting check (flake8)"),
    ]

    all_passed = True
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            all_passed = False

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="NBA Simulation Test Runner")
    parser.add_argument(
        "--fast", action="store_true", help="Run tests without coverage reporting"
    )
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument(
        "--integration", action="store_true", help="Run only integration tests"
    )
    parser.add_argument(
        "--coverage-only", action="store_true", help="Only check coverage threshold"
    )
    parser.add_argument("--lint", action="store_true", help="Run linting checks")
    parser.add_argument(
        "--open-coverage", action="store_true", help="Open coverage report in browser"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=50,
        help="Coverage threshold percentage (default: 50)",
    )

    args = parser.parse_args()

    print("🧪 NBA Simulation Test Runner")
    print("=" * 40)

    success = True

    if args.coverage_only:
        success = check_coverage_threshold(args.threshold)
    elif args.lint:
        success = run_linting()
    else:
        # Run tests
        success = run_tests(args)

        # Check coverage threshold if coverage was generated
        if success and not args.fast:
            threshold_met = check_coverage_threshold(args.threshold)
            if not threshold_met:
                print("\n💡 To improve coverage, consider adding tests for:")
                print("  - utils.py: Core rating and calculation functions")
                print("  - eval.py: Model training and evaluation")
                print("  - forecast.py: Prediction functions")
                print("  - data_loader.py: Data processing")
                print("  - sim_season.py: Simulation logic")

    if success:
        print("\n🎉 All checks passed!")
        sys.exit(0)
    else:
        print("\n💥 Some checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
