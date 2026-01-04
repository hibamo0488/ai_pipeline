from src.app.schemas.factory_schema import ModelFactory, Model
from typing import Any, Callable
import optuna

class Objective:
    def __init__(self, model_obj: Model, suggest_params: Callable,
                 accuracy_score: Callable, X_train, y_train, X_valid, y_valid):
        self.model_obj = model_obj
        self.suggest_params = suggest_params
        self.X_train, self.y_train = X_train, y_train
        self.X_valid, self.y_valid = X_valid, y_valid
        self.accuracy_score = accuracy_score

    def __call__(self, trial):
        params = self.suggest_params(trial)
        model = self.model_obj(**params)
        model.train(self.X_train, self.y_train)
        pred = model.predict(self.X_valid)
        return self.accuracy_score(self.y_valid, pred)

def study_execute(model_obj: Model, suggest_params: Callable, 
                  accuracy_score: Callable, X_train, y_train, X_valid, y_valid, n_trials: int):
    obj = Objective(model_obj, suggest_params, accuracy_score, X_train, y_train, X_valid, y_valid)
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials)
