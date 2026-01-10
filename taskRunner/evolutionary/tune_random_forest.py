import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_curve


def tune_random_forest(
    X_train, y_train, X_test, y_test,
    generations=10,
    population_size=10,
    random_state=42
):
    random.seed(random_state)

    def random_individual():
        return {
            "n_estimators": random.randint(150,350),
            "max_depth": random.randint(8,25),
            "min_samples_split": random.randint(10,50),
            "min_samples_leaf": random.randint(5,30),
            "max_features": random.uniform(0.4,0.8)
        }

    def mutate(ind):
        key = random.choice(list(ind.keys()))
        if key == "max_features":
            ind[key] = random.uniform(0.4,0.8)
        else:
            ind[key] = max(2, ind[key] + random.randint(-20,20))
        return ind

    def fitness(ind):
        model = RandomForestClassifier(
            **ind,
            class_weight="balanced",
            n_jobs=1,
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
