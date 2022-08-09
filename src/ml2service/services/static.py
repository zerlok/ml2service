import typing as t

from ml2service.models.base import Model
from ml2service.services.base import (
    ModelPredictionService,
    PredictModelInternalErrorResponse,
    PredictModelNotFoundErrorResponse,
    PredictRequest,
    PredictSuccessResponse,
)

K = t.TypeVar("K")
T_train_input = t.TypeVar("T_train_input", contravariant=True)
T_predict_input = t.TypeVar("T_predict_input", contravariant=True)
T_predict_output = t.TypeVar("T_predict_output", covariant=True)


class StaticModelService(
    t.Generic[K, T_predict_input, T_predict_output],
    ModelPredictionService[K, T_predict_input, T_predict_output],
):
    def __init__(
            self,
            model: Model[T_predict_input, T_predict_output],
    ) -> None:
        self.__model = model

    def predict(self, request: PredictRequest[K, T_predict_input]) -> t.Union[
        PredictSuccessResponse[K, T_predict_output],
        PredictModelNotFoundErrorResponse[K],
        PredictModelInternalErrorResponse[K],
    ]:
        try:
            output = self.__model.predict(request.input_)

        except Exception as err:
            return PredictModelInternalErrorResponse(key=request.key, error=err)

        return PredictSuccessResponse(key=request.key, output=output)
