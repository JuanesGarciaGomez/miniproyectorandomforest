import os
import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture(scope="session", autouse=True)
def dummy_model(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("models")
    model_path = tmp_dir / "model.joblib"

    X = np.array([
        [1, 99.9, 499.5, 0, 2],
        [0, 29.85, 29.85, 0, 0],
        [1, 80.0, 400.0, 0, 3],
        [0, 50.0, 600.0, 1, 1],
        [0, 30.0, 1800.0, 2, 3],
        [0, 25.0, 2000.0, 2, 1],
    ] * 15)
    y = np.array([1, 1, 1, 0, 0, 0] * 15)

    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(X, y)
    joblib.dump(clf, model_path)

    os.environ["MODEL_PATH"] = str(model_path)
    return str(model_path)
