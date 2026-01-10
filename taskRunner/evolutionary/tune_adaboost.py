import random
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score, roc_curve


def tune_adaboost(
    X_train, y_train, X_test, y_test,
    generations=10,
    population_size=8,
    random_state=42
):
    random.seed(random_state)

    def random_individual():
        return {
            "n_estimators": random.randint(100,200),
            "learning_rate": 10 ** random.uniform(-2, -0.3)
        }

    def mutate(ind):
        if random.random() < 0.5:
            ind["n_estimators"] = max(
                50, min(300, ind["n_estimators"] + random.randint(-20,20))
            )
        else:
            ind["learning_rate"] *= random.uniform(0.7,1.3)
        return ind

    def fitness(ind):
        model = AdaBoostClassifier(**ind, random_state=random_state)
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
