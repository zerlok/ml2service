import json
from pathlib import Path

from ml2service.models.base import Model, ModelTrainer


class FooModel(Model[int, int]):

    def __init__(self, k: int) -> None:
        self.__k = k

    def predict(self, input_: int) -> int:
        return self.__k * input_ ** 2


class FooDynamicModelTrainer(ModelTrainer[int, int, int]):

    def train(self, input_: int) -> Model[int, int]:
        return FooModel(input_)


class FooStaticModelTrainer(ModelTrainer[Path, int, int]):

    def __init__(self) -> None:
        self.__dynamic = FooDynamicModelTrainer()

    def train(self, input_: Path) -> Model[int, int]:
        with input_.open("r") as f:
            data = json.load(f)
            if not isinstance(data, int):
                raise ValueError("int was expected", data, input_)

        return self.__dynamic.train(data)
