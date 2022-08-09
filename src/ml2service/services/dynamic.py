import typing as t

from ml2service.models.base import Model, ModelTrainer
from ml2service.services.base import (
    ModelPredictionService,
    ModelRemovingService,
    ModelTrainingService,
    PredictModelInternalErrorResponse,
    PredictModelNotFoundErrorResponse,
    PredictRequest,
    PredictSuccessResponse,
    RemoveModelNotFoundErrorResponse,
    RemoveRequest,
    RemoveSuccessResponse,
    TrainInternalErrorResponse,
    TrainRequest,
    TrainSuccessResponse,
)
from ml2service.storages.base import Storage

K = t.TypeVar("K")
T = t.TypeVar("T")

T_train_input = t.TypeVar("T_train_input")
T_predict_input = t.TypeVar("T_predict_input")
T_predict_output = t.TypeVar("T_predict_output")


class DynamicModelService(
    t.Generic[K, T_train_input, T_predict_input, T_predict_output],
    ModelTrainingService[K, T_train_input],
    ModelPredictionService[K, T_predict_input, T_predict_output],
    ModelRemovingService[K],
):
    def __init__(
            self,
            trainer: ModelTrainer[T_train_input, T_predict_input, T_predict_output],
            storage: Storage[K, Model[T_predict_input, T_predict_output]],
    ) -> None:
        self.__trainer = trainer
        self.__storage = storage

    def train(self, request: TrainRequest[K, T_train_input]) -> t.Union[
        TrainSuccessResponse[K],
        TrainInternalErrorResponse[K],
    ]:
        try:
            model = self.__trainer.train(request.input_)

        except Exception as err:
            return TrainInternalErrorResponse(key=request.key, error=err)

        self.__storage.update(request.key, model)

        return TrainSuccessResponse(key=request.key)

    def predict(self, request: PredictRequest[K, T_predict_input]) -> t.Union[
        PredictSuccessResponse[K, T_predict_output],
        PredictModelNotFoundErrorResponse[K],
        PredictModelInternalErrorResponse[K],
    ]:
        model = self.__storage.get(request.key)

        if model is None:
            return PredictModelNotFoundErrorResponse(key=request.key)

        try:
            output = model.predict(request.input_)

        except Exception as err:
            return PredictModelInternalErrorResponse(key=request.key, error=err)

        return PredictSuccessResponse(key=request.key, output=output)

    def remove(self, request: RemoveRequest[K]) -> t.Union[
        RemoveSuccessResponse[K],
        RemoveModelNotFoundErrorResponse[K],
    ]:
        model = self.__storage.remove(request.key)
        if model is None:
            return RemoveModelNotFoundErrorResponse(key=request.key)

        return RemoveSuccessResponse(key=request.key)
