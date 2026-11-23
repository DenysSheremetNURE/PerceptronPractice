import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

from src.perceptron import MyPerceptron

def boundary(w, b, x1):
    """x2 coordinates of the line w1*x1 + w2*x2 + b = 0 for given x1."""
    return -(w[0] * x1 + b) / w[1]

if __name__ == "__main__":
    RANDOM_STATE = 2525
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    # mu = np.mean(X_train, axis=0)
    # sigma = np.std(X_train, axis=0)
    # X_train_s = (X_train - mu) / sigma
    # X_test_s = (X_test - mu) / sigma
    #
    # my_p = MyPerceptron(eta=0.1, epochs=50)
    # my_p.fit(X_train_s, y_train)
    # y_pred = my_p.predict(X_test_s)
    #
    # # Test accuracy
    # acc = (y_pred == y_test).mean()
    # print(f"Accuracy: {acc:.3f}")
    #
    # # Errors per epoch plot
    # errors = my_p.errors_
    # epochs = range(1, len(errors) + 1)
    #
    # plt.figure(figsize=(8, 4))
    # plt.plot(epochs, errors, marker="o", markersize=3)
    # plt.xlabel("Epoch")
    # plt.ylabel("Number of errors")
    # plt.title("Perceptron errors per epoch")
    # plt.grid(alpha=0.3)
    # plt.savefig("errors.png", dpi=150, bbox_inches="tight")
    # plt.show()



    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Hyperparameter grid
    perc_grid = {
        "clf__eta0":     [0.0001, 0.001, 0.01, 0.1, 1.0],
        "clf__max_iter": [5, 10, 50, 100, 1000],
        "clf__penalty":  [None, "l2", "l1", "elasticnet"],
        "clf__alpha":    [0.0001, 0.001, 0.01],
    }

    # Pipeline. tol=None disables early stopping
    perc_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", Perceptron(random_state=RANDOM_STATE, tol=None)),
    ])

    # Grid search object
    perc_search = GridSearchCV(perc_pipeline, perc_grid, cv=cv, scoring="f1", n_jobs=-1)

    # Run the search
    perc_search.fit(X_train, y_train)
    #
    # # Results
    # print("Best params:", perc_search.best_params_)
    # print(f"Best CV F1: {perc_search.best_score_:.3f}")
    #
    # # Evaluation
    # y_pred_perc = perc_search.best_estimator_.predict(X_test)
    # print(metrics.classification_report(y_test, y_pred_perc, target_names=dataset.target_names, digits=3))




    # Baseline perceptron (default params)
    baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", Perceptron(random_state=RANDOM_STATE)),
    ])
    baseline.fit(X_train, y_train)

    # Logistic regression
    lr_search = GridSearchCV(
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=5000))]),
        {"clf__C": [0.01, 0.1, 1, 10, 100]},
        cv=cv, scoring="f1", n_jobs=-1)
    lr_search.fit(X_train, y_train)
    print("LogReg best params:", lr_search.best_params_)

    # Linear SVM
    svm_search = GridSearchCV(
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LinearSVC(max_iter=10000, random_state=RANDOM_STATE))]),
        {"clf__C": [0.001, 0.01, 0.1, 1, 10]},
        cv=cv, scoring="f1", n_jobs=-1)
    svm_search.fit(X_train, y_train)
    print("SVM best params:", svm_search.best_params_)

    # Comparison
    models = [
        ("Perceptron (baseline)", baseline),
        ("Perceptron (tuned)",    perc_search.best_estimator_),
        ("Logistic regression",   lr_search.best_estimator_),
        ("Linear SVM",            svm_search.best_estimator_),
    ]

    # print(f"\n{'Model':<24}{'Acc':>7}{'P mal':>7}{'R mal':>7}{'F1 mal':>8}{'F1 ben':>8}")
    # for name, model in models:
    #     pred = model.predict(X_test)
    #     acc = metrics.accuracy_score(y_test, pred)
    #     p_mal = metrics.precision_score(y_test, pred, pos_label=0)
    #     r_mal = metrics.recall_score(y_test, pred, pos_label=0)
    #     f1_mal = metrics.f1_score(y_test, pred, pos_label=0)
    #     f1_ben = metrics.f1_score(y_test, pred, pos_label=1)
    #     print(f"{name:<24}{acc:>7.3f}{p_mal:>7.3f}{r_mal:>7.3f}{f1_mal:>8.3f}{f1_ben:>8.3f}")

    # PCA and comparison

    # Classification reports
    for name, model in models:
        print(f"\n=== {name} ===")
        print(metrics.classification_report(y_test, model.predict(X_test),
                                            target_names=dataset.target_names, digits=3))

    # Confusion matrices
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4))
    for ax, (name, model) in zip(axes, models):
        metrics.ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, display_labels=dataset.target_names, ax=ax, colorbar=False)
        ax.set_title(name, fontsize=10)
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")

    # Decision boundary: project 30 features to 2D with PCA
    scaler_2d = StandardScaler().fit(X_train)
    pca = PCA(n_components=2).fit(scaler_2d.transform(X_train))
    Z_train = pca.transform(scaler_2d.transform(X_train))
    Z_test = pca.transform(scaler_2d.transform(X_test))
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    # Same hyperparameters
    perc_params = {k.replace("clf__", ""): v for k, v in perc_search.best_params_.items()}
    models_2d = [
        ("Perceptron (tuned)", Perceptron(random_state=RANDOM_STATE, tol=None, **perc_params)),
        ("Logistic regression", LogisticRegression(C=lr_search.best_params_["clf__C"], max_iter=5000)),
        ("Linear SVM", LinearSVC(C=svm_search.best_params_["clf__C"], max_iter=10000,
                                 random_state=RANDOM_STATE)),
    ]

    xx, yy = np.meshgrid(
        np.linspace(Z_train[:, 0].min() - 1, Z_train[:, 0].max() + 1, 300),
        np.linspace(Z_train[:, 1].min() - 1, Z_train[:, 1].max() + 1, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, len(models_2d), figsize=(6 * len(models_2d), 5))
    for ax, (name, model) in zip(axes, models_2d):
        model.fit(Z_train, y_train)

        ax.contourf(xx, yy, model.predict(grid).reshape(xx.shape), alpha=0.25, cmap="coolwarm_r")

        xs = np.array([xx.min(), xx.max()])
        ax.plot(xs, boundary(model.coef_[0], model.intercept_[0], xs),
                "k--", lw=1.5, label="Decision boundary")

        ax.scatter(*Z_test[y_test == 0].T, c="tab:red", edgecolor="k", s=30, label="malignant")
        ax.scatter(*Z_test[y_test == 1].T, c="tab:blue", edgecolor="k", s=30, label="benign")

        ax.set_xlim(xx.min(), xx.max()); ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        acc = metrics.accuracy_score(y_test, model.predict(Z_test))
        ax.set_title(f"{name} (PCA 2D), test acc = {acc:.3f}")
        ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("decision_boundary.png", dpi=150, bbox_inches="tight")
    plt.show()