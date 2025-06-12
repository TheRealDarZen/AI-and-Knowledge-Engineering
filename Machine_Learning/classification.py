from exp_prep_split_preproc import get_data

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def print_scores(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='weighted'))
    print("Recall:", recall_score(y_true, y_pred, average='weighted'))
    print("F1:", f1_score(y_true, y_pred, average='weighted'))


def naive_bayes(X_train, X_test, y_train, y_test):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    print("NB:")
    print_scores(y_test, y_pred_nb)


def decision_tree(X_train, X_test, y_train, y_test, depth_list):
    for depth in depth_list:
        dtc = DecisionTreeClassifier(max_depth=depth, min_samples_split=10, random_state=42)
        dtc.fit(X_train, y_train)
        y_pred_tree = dtc.predict(X_test)
        print(f"\nDecision Tree (depth={depth}):")
        print_scores(y_test, y_pred_tree)


def random_forest(X_train, X_test, y_train, y_test, estimators_list):
    for estimators in estimators_list:
        rf = RandomForestClassifier(n_estimators=estimators, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        print(f"\nRandom Forest (estimators={estimators}):")
        print_scores(y_test, y_pred_rf)


if __name__ == "__main__":
    data = get_data()
    X_train, X_test = data["original"]
    X_train_std, X_test_std = data["standardized"]
    X_train_norm, X_test_norm = data["normalized"]
    y_train, y_test = data["labels"]

    print("\n")
    print("Original:")
    naive_bayes(X_train, X_test, y_train, y_test)
    decision_tree(X_train, X_test, y_train, y_test, [10, 20, None])
    random_forest(X_train, X_test, y_train, y_test, [50, 100, 500])

    print("Standarized:")
    naive_bayes(X_train, X_test, y_train, y_test)
    decision_tree(X_train, X_test, y_train, y_test, [10, 20, None])
    random_forest(X_train, X_test, y_train, y_test, [50, 100, 500])

    print("Normalized:")
    naive_bayes(X_train, X_test, y_train, y_test)
    decision_tree(X_train, X_test, y_train, y_test, [10, 20, None])
    random_forest(X_train, X_test, y_train, y_test, [50, 100, 500])
