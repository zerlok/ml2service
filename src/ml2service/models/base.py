import abc
import typing as t

T_train_input = t.TypeVar("T_train_input", contravariant=True)
T_predict_input = t.TypeVar("T_predict_input", contravariant=True)
T_predict_output = t.TypeVar("T_predict_output", covariant=True)


class Model(t.Generic[T_predict_input, T_predict_output], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def predict(self, input_: T_predict_input) -> T_predict_output:
        raise NotImplementedError


class ModelTrainer(
    t.Generic[T_train_input, T_predict_input, T_predict_output],
    metaclass=abc.ABCMeta,
):
    @abc.abstractmethod
    def train(self, input_: T_train_input) -> Model[T_predict_input, T_predict_output]:
        raise NotImplementedError
