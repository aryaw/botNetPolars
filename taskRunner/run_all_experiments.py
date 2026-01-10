"""
run all hyperparameter tuning experiments
"""

import sys
import traceback


def run_optuna():
    print(" RUNNING OPTUNA (Bayesian Optimization / TPE)")
    from taskRunner.optuna.run_all_tuning import run_all
    run_all()


def run_random_search():
    print(" RUNNING RANDOM SEARCH")
    from taskRunner.random_search.run_all_tuning import run_all
    run_all()


def run_evolutionary():
    print(" RUNNING EVOLUTIONARY / GENETIC")
    from taskRunner.evolutionary.run_all_tuning import run_all
    run_all()


def safe_run(fn, name):
    try:
        fn()
    except Exception as e:
        print(f"\n[ERROR] {name} failed")
        traceback.print_exc()
        print("\n continuing to next experiment...\n")


def main():
    print("# botnet tuning pipeline #")

    safe_run(run_optuna, "optuna")
    safe_run(run_random_search, "random search")
    safe_run(run_evolutionary, "evolutionary")

    print("# ----completed#")


if __name__ == "__main__":
    main()
