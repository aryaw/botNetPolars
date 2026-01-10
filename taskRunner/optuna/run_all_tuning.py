import json
import os
from taskRunner.optuna.tune_adaboost import tune_adaboost
from taskRunner.optuna.tune_decision_tree import tune_decision_tree
from taskRunner.optuna.tune_logistic import tune_logistic
from taskRunner.optuna.tune_random_forest import tune_random_forest
from taskRunner.optuna.tune_svm import tune_svm
from libInternal.dFHelper import setFileLocation
from libInternal.globalCompare import export_global_comparison_table

def run_all():
    results = []

    ts, out = setFileLocation()

    print("=== Running AdaBoost ===")
    results.append({
        "Algorithm": "AdaBoost",
        "BestParams": tune_adaboost()
    })

    print("=== Running Decision Tree ===")
    results.append({
        "Algorithm": "Decision Tree",
        "BestParams": tune_decision_tree()
    })

    print("=== Running Logistic Regression ===")
    results.append({
        "Algorithm": "Logistic Regression",
        "BestParams": tune_logistic()
    })

    print("=== Running Random Forest ===")
    results.append({
        "Algorithm": "Random Forest",
        "BestParams": tune_random_forest()
    })

    print("=== Running SVM (RBF) ===")
    results.append({
        "Algorithm": "SVM (RBF)",
        "BestParams": tune_svm()
    })

    with open(
        os.path.join(out["root"], f"best_params_{ts}.json"),
        "w"
    ) as f:
        json.dump(results, f, indent=2)

    export_global_comparison_table(out)

    print("=== ALL TUNING COMPLETED ===")


if __name__ == "__main__":
    run_all()
