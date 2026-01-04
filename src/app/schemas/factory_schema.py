from typing import Protocol, Any

class Model(Protocol):
    model: Any 
    def train(self, X, y) -> None:
        pass
    def predict(self, X):
        pass

    def __call__(self, **kwargs):
        pass

class ModelFactory(Protocol):
    def create(self) -> Model:
        pass
