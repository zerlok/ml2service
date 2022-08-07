from pathlib import Path

from ml2service.models.base import Model, ModelTrainer


class FooModel(Model[int, int]):

    def __init__(self, k: int) -> None:
        self.__k = k

    def predict(self, input_: int) -> int:
        return self.__k * input_ ** 2


class FooModelTrainer(ModelTrainer[int, int, int]):

    def train(self, input_: int) -> Model[int, int]:
        return FooModel(input_)


def load_foo(path: Path) -> FooModelTrainer:
    # with path.open("r") as f:
    #     int(json.load(f))
    return FooModelTrainer()
