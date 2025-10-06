#!/usr/bin/env python3
"""
Comprehensive training loop test runner.
This script provides a unified interface to run all training loop validation tests.
"""

import sys
import os
import argparse
import traceback
from typing import Dict, Tuple, List

# Import all test modules
from test_training_basic import test_basic_training_loop
from test_training_overfit import test_overfitting_tiny_dataset, test_generalization_gap
from test_training_gradients import test_gradient_flow
from test_training_lr_schedule import (
    test_lr_schedule_behavior,
    test_lr_schedule_in_training,
    compare_with_without_schedule
)

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def run_test_suite(tests_to_run: List[str], verbose: bool = False) -> Dict:
    """
    Run the selected test suite and collect results.

    Args:
        tests_to_run: List of test names to run
        verbose: Whether to show detailed output

    Returns:
        Dictionary with test results
    """
    results = {}

    # Define test mapping
    test_functions = {
        'basic': ('Basic Training Loop', test_basic_training_loop),
        'overfit': ('Overfitting Test', lambda: test_overfitting_tiny_dataset()[:3]),
        'generalization': ('Generalization Gap', test_generalization_gap),
        'gradients': ('Gradient Flow', test_gradient_flow),
        'lr_schedule': ('LR Schedule', test_lr_schedule_behavior),
        'lr_training': ('LR in Training', test_lr_schedule_in_training),
        'lr_comparison': ('LR Comparison', compare_with_without_schedule),
    }

    print(f"{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}COMPREHENSIVE TRAINING LOOP TEST SUITE{Colors.ENDC}")
    print(f"{Colors.HEADER}{'=' * 70}{Colors.ENDC}\n")

    # Run selected tests
    for test_name in tests_to_run:
        if test_name not in test_functions:
            print(f"{Colors.WARNING}Warning: Unknown test '{test_name}', skipping...{Colors.ENDC}")
            continue

        test_title, test_func = test_functions[test_name]

        print(f"\n{Colors.CYAN}Running: {test_title}{Colors.ENDC}")
        print("-" * 50)

        try:
            result = test_func()

            # Determine success based on return value
            if isinstance(result, tuple) and len(result) >= 2:
                success = result[-1] if isinstance(result[-1], bool) else True
            else:
                success = True

            results[test_name] = {
                'status': 'PASSED' if success else 'FAILED',
                'success': success
            }

            if success:
                print(f"{Colors.GREEN}✓ {test_title} PASSED{Colors.ENDC}")
            else:
                print(f"{Colors.FAIL}✗ {test_title} FAILED{Colors.ENDC}")

        except Exception as e:
            print(f"{Colors.FAIL}✗ {test_title} CRASHED{Colors.ENDC}")
            if verbose:
                print(f"{Colors.FAIL}{traceback.format_exc()}{Colors.ENDC}")
            else:
                print(f"{Colors.FAIL}Error: {str(e)}{Colors.ENDC}")
            results[test_name] = {
                'status': 'CRASHED',
                'success': False,
                'error': str(e)
            }

    return results


def print_summary(results: Dict):
    """Print a summary of all test results."""
    print(f"\n{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}TEST SUMMARY{Colors.ENDC}")
    print(f"{Colors.HEADER}{'=' * 70}{Colors.ENDC}\n")

    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r['success'])
    failed_tests = total_tests - passed_tests

    # Print individual test results
    for test_name, result in results.items():
        status = result['status']
        if status == 'PASSED':
            color = Colors.GREEN
            symbol = '✓'
        elif status == 'FAILED':
            color = Colors.WARNING
            symbol = '✗'
        else:  # CRASHED
            color = Colors.FAIL
            symbol = '✗'

        print(f"  {color}{symbol} {test_name:15s}: {status}{Colors.ENDC}")

    # Print overall summary
    print(f"\n{Colors.BOLD}Overall Results:{Colors.ENDC}")
    print(f"  Total Tests: {total_tests}")
    print(f"  {Colors.GREEN}Passed: {passed_tests}{Colors.ENDC}")
    if failed_tests > 0:
        print(f"  {Colors.FAIL}Failed: {failed_tests}{Colors.ENDC}")

    # Final verdict
    print(f"\n{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
    if passed_tests == total_tests:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED!{Colors.ENDC}")
        print(f"{Colors.GREEN}Your training loop implementation appears to be working correctly!{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}✗ SOME TESTS FAILED{Colors.ENDC}")
        print(f"{Colors.WARNING}Please review the failed tests and fix your implementation.{Colors.ENDC}")
    print(f"{Colors.HEADER}{'=' * 70}{Colors.ENDC}")


def print_recommendations():
    """Print recommendations for using the test suite."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}RECOMMENDATIONS FOR TESTING YOUR TRAINING LOOP:{Colors.ENDC}\n")

    recommendations = [
        ("1. Start with the basic test",
         "Run with --test basic first to ensure fundamental operations work"),

        ("2. Check overfitting capability",
         "Run with --test overfit to verify the model can memorize small data"),

        ("3. Verify gradient flow",
         "Run with --test gradients to ensure gradients propagate correctly"),

        ("4. Test learning rate schedule",
         "Run with --test lr_schedule to validate LR scheduling works"),

        ("5. Run all tests",
         "Use --all to run the complete test suite"),

        ("6. Check generated plots",
         "Review the .png files generated for visual validation"),
    ]

    for title, desc in recommendations:
        print(f"  {Colors.BOLD}{title}{Colors.ENDC}")
        print(f"    {desc}\n")

    print(f"{Colors.CYAN}Example commands:{Colors.ENDC}")
    print(f"  python {os.path.basename(__file__)} --test basic")
    print(f"  python {os.path.basename(__file__)} --test overfit gradients")
    print(f"  python {os.path.basename(__file__)} --all")
    print(f"  python {os.path.basename(__file__)} --quick  # Runs only essential tests")


def main():
    """Main entry point for the test suite."""
    parser = argparse.ArgumentParser(
        description='Comprehensive training loop test suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_training_comprehensive.py --all              # Run all tests
  python test_training_comprehensive.py --quick            # Run quick essential tests
  python test_training_comprehensive.py --test basic       # Run specific test
  python test_training_comprehensive.py --test overfit gradients  # Run multiple tests
  python test_training_comprehensive.py --list             # List available tests
        """
    )

    parser.add_argument('--test', nargs='+',
                       choices=['basic', 'overfit', 'generalization', 'gradients',
                                'lr_schedule', 'lr_training', 'lr_comparison'],
                       help='Specific tests to run')
    parser.add_argument('--all', action='store_true',
                       help='Run all tests')
    parser.add_argument('--quick', action='store_true',
                       help='Run only essential tests (basic, overfit, gradients)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed error messages')
    parser.add_argument('--list', action='store_true',
                       help='List all available tests')
    parser.add_argument('--recommendations', action='store_true',
                       help='Show testing recommendations')

    args = parser.parse_args()

    # Handle special flags
    if args.list:
        print(f"{Colors.BOLD}Available Tests:{Colors.ENDC}")
        tests_info = {
            'basic': 'Basic training loop functionality',
            'overfit': 'Model\'s ability to overfit small dataset',
            'generalization': 'Training vs validation gap analysis',
            'gradients': 'Gradient flow and health checks',
            'lr_schedule': 'Learning rate schedule function',
            'lr_training': 'LR schedule integration with training',
            'lr_comparison': 'Compare training with/without LR schedule',
        }
        for test_name, description in tests_info.items():
            print(f"  {Colors.CYAN}{test_name:15s}{Colors.ENDC}: {description}")
        return

    if args.recommendations:
        print_recommendations()
        return

    # Determine which tests to run
    if args.all:
        tests_to_run = ['basic', 'overfit', 'generalization', 'gradients',
                       'lr_schedule', 'lr_training', 'lr_comparison']
    elif args.quick:
        tests_to_run = ['basic', 'overfit', 'gradients']
    elif args.test:
        tests_to_run = args.test
    else:
        # Default: run quick tests
        print(f"{Colors.WARNING}No tests specified, running quick essential tests...{Colors.ENDC}")
        tests_to_run = ['basic', 'overfit', 'gradients']

    # Run tests
    results = run_test_suite(tests_to_run, verbose=args.verbose)

    # Print summary
    print_summary(results)

    # Exit with appropriate code
    all_passed = all(r['success'] for r in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()