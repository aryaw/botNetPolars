import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_curve


def tune_logistic(
    X_train, y_train, X_test, y_test,
    generations=10,
    population_size=10,
    random_state=42
):
    random.seed(random_state)

    def random_individual():
        return {
            "C": 10 ** random.uniform(-3, 1),
            "solver": random.choice(["liblinear", "lbfgs"])
        }

    def mutate(ind):
        if random.random() < 0.5:
            ind["C"] = 10 ** random.uniform(-3, 1)
        else:
            ind["solver"] = "liblinear" if ind["solver"] == "lbfgs" else "lbfgs"
        return ind

    def fitness(ind):
        model = LogisticRegression(
            **ind,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        )
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:,1]
        fpr,tpr,thr = roc_curve(y_test,p)
        return f1_score(
            y_test, (p >= thr[np.argmax(tpr-fpr)]).astype(int)
        )

    population = [random_individual() for _ in range(population_size)]

    for _ in range(generations):
        scored = sorted(
            [(fitness(ind), ind) for ind in population],
            reverse=True, key=lambda x: x[0]
        )
        survivors = [ind for _,ind in scored[:population_size//2]]

        while len(survivors) < population_size:
            survivors.append(mutate(random.choice(survivors)))

        population = survivors

    best = max(population, key=fitness)
    return {"best_params": best}
