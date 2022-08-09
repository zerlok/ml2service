import abc
import typing as t
from dataclasses import dataclass

K = t.TypeVar("K")
T = t.TypeVar("T")

T_train_input = t.TypeVar("T_train_input", contravariant=True)
T_predict_input = t.TypeVar("T_predict_input", contravariant=True)
T_predict_output = t.TypeVar("T_predict_output", covariant=True)


# noinspection DuplicatedCode
@dataclass(frozen=True)
class TrainRequest(t.Generic[K, T]):
    key: K
    input_: T


@dataclass(frozen=True)
class TrainSuccessResponse(t.Generic[K]):
    key: K


@dataclass(frozen=True)
class TrainInternalErrorResponse(t.Generic[K]):
    key: K
    error: Exception


@dataclass(frozen=True)
class PredictRequest(t.Generic[K, T]):
    key: K
    input_: T


# noinspection DuplicatedCode
@dataclass(frozen=True)
class PredictSuccessResponse(t.Generic[K, T]):
    key: K
    output: T


@dataclass(frozen=True)
class PredictModelNotFoundErrorResponse(t.Generic[K]):
    key: K


@dataclass(frozen=True)
class PredictModelInternalErrorResponse(t.Generic[K]):
    key: K
    error: Exception


@dataclass(frozen=True)
class RemoveRequest(t.Generic[K]):
    key: K


@dataclass(frozen=True)
class RemoveSuccessResponse(t.Generic[K]):
    key: K


@dataclass(frozen=True)
class RemoveModelNotFoundErrorResponse(t.Generic[K]):
    key: K


class ModelService(metaclass=abc.ABCMeta):
    pass


class ModelTrainingService(t.Generic[K, T_train_input], ModelService, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train(
            self,
            request: TrainRequest[K, T_train_input],
    ) -> t.Union[
        TrainSuccessResponse[K],
        TrainInternalErrorResponse[K],
    ]:
        raise NotImplementedError


class ModelPredictionService(t.Generic[K, T_predict_input, T_predict_output], ModelService, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def predict(
            self,
            request: PredictRequest[K, T_predict_input],
    ) -> t.Union[
        PredictSuccessResponse[K, T_predict_output],
        PredictModelNotFoundErrorResponse[K],
        PredictModelInternalErrorResponse[K],
    ]:
        raise NotImplementedError


class ModelRemovingService(t.Generic[K], ModelService, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def remove(
            self,
            request: RemoveRequest[K],
    ) -> t.Union[
        RemoveSuccessResponse[K],
        RemoveModelNotFoundErrorResponse[K],
    ]:
        raise NotImplementedError
