import random
import numpy as np
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =====================================================
# Load Dataset
# =====================================================

data = load_breast_cancer()

X = data.data
y = data.target
feature_names = data.feature_names

n_features = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# Genetic Algorithm Parameters
# =====================================================

POPULATION_SIZE = 50
GENERATIONS = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1

# =====================================================
# Fitness Function
# =====================================================

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register(
    "attr_bool",
    random.randint,
    0,
    1
)

toolbox.register(
    "individual",
    tools.initRepeat,
    creator.Individual,
    toolbox.attr_bool,
    n=n_features
)

toolbox.register(
    "population",
    tools.initRepeat,
    list,
    toolbox.individual
)

# =====================================================
# Evaluation Function
# =====================================================

def evaluate(individual):

    selected_features = [
        i for i in range(n_features)
        if individual[i] == 1
    ]

    if len(selected_features) == 0:
        return (0,)

    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    clf.fit(X_train_selected, y_train)

    predictions = clf.predict(X_test_selected)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return (accuracy,)

toolbox.register("evaluate", evaluate)

# =====================================================
# Genetic Operators
# =====================================================

toolbox.register(
    "mate",
    tools.cxTwoPoint
)

toolbox.register(
    "mutate",
    tools.mutFlipBit,
    indpb=0.05
)

toolbox.register(
    "select",
    tools.selTournament,
    tournsize=3
)

# =====================================================
# Main Program
# =====================================================

def main():

    population = toolbox.population(
        n=POPULATION_SIZE
    )

    stats = tools.Statistics(
        lambda ind: ind.fitness.values
    )

    stats.register(
        "avg",
        np.mean
    )

    stats.register(
        "max",
        np.max
    )

    best_fitness_history = []

    for generation in range(GENERATIONS):

        offspring = algorithms.varAnd(
            population,
            toolbox,
            cxpb=CROSSOVER_RATE,
            mutpb=MUTATION_RATE
        )

        fitnesses = list(
            map(
                toolbox.evaluate,
                offspring
            )
        )

        for fit, ind in zip(
            fitnesses,
            offspring
        ):
            ind.fitness.values = fit

        population = toolbox.select(
            offspring,
            k=len(population)
        )

        best_ind = tools.selBest(
            population,
            1
        )[0]

        best_accuracy = best_ind.fitness.values[0]

        best_fitness_history.append(
            best_accuracy
        )

        print(
            f"Generation {generation+1}: "
            f"Best Accuracy = {best_accuracy:.4f}"
        )

    # =================================================
    # Final Results
    # =================================================

    best_solution = tools.selBest(
        population,
        1
    )[0]

    selected_features = [
        feature_names[i]
        for i in range(n_features)
        if best_solution[i] == 1
    ]

    final_accuracy = evaluate(
        best_solution
    )[0]

    print("\n========================")
    print("FINAL RESULTS")
    print("========================")

    print(
        f"Accuracy: {final_accuracy:.4f}"
    )

    print(
        f"Selected Features: "
        f"{len(selected_features)}"
    )

    print("\nFeatures Selected:")

    for feature in selected_features:
        print(feature)

    # =================================================
    # Plot Convergence Graph
    # =================================================

    plt.figure(figsize=(10,5))

    plt.plot(
        best_fitness_history,
        linewidth=2
    )

    plt.title(
        "GA Feature Selection Convergence"
    )

    plt.xlabel(
        "Generation"
    )

    plt.ylabel(
        "Best Accuracy"
    )

    plt.grid(True)

    plt.savefig(
        "convergence_graph.png"
    )

    plt.show()

if __name__ == "__main__":
    main()