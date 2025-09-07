from typing import Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_location_classifier(X_train, y_train, model_type: str = 'logistic'):
    if model_type != 'logistic':
        raise ValueError("Only 'logistic' model_type is implemented in this minimal version.")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_classifier(model, X_test, y_test) -> Tuple[float, str]:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, report
