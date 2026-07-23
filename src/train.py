import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier


def train_isolation_forest(X_train, y_train, random_state: int = 42):
    contamination = y_train.mean()

    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def train_random_forest(X_train, y_train, random_state: int = 42):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def isoforest_predict(model, X):
    raw = model.predict(X)
    return (raw == -1).astype(int)


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
