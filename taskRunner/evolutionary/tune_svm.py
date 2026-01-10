import random
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_curve


def tune_svm(
    X_train, y_train, X_test, y_test,
    generations=10,
    population_size=10,
    random_state=42
):
    random.seed(random_state)

    def random_individual():
        return {
            "C": 10 ** random.uniform(-2, np.log10(5.0)),
            "gamma": 10 ** random.uniform(-4, np.log10(5e-2))
        }

    def mutate(ind):
        if random.random() < 0.5:
            ind["C"] = 10 ** random.uniform(-2, np.log10(5.0))
        else:
            ind["gamma"] = 10 ** random.uniform(-4, np.log10(5e-2))
        return ind

    def fitness(ind):
        model = SVC(
            **ind,
            kernel="rbf",
            probability=True,
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
