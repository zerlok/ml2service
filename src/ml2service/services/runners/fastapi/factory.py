import typing as t

# noinspection PyPackageRequirements
from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException, Path
# noinspection PyPackageRequirements
from starlette import status
# noinspection PyPackageRequirements
from uvicorn import Config, Server

from ml2service.models.loader import ModuleInfo
from ml2service.services.base import (
    ModelPredictionService,
    ModelRemovingService,
    ModelService,
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
from ml2service.services.runners.base import ServiceRunner, ServiceRunnerFactory
from ml2service.services.runners.fastapi.runner import UvicornServiceRunner
from ml2service.services.static import StaticModelService
from ml2service.strict_typing import raise_not_exhaustive

F = t.TypeVar("F")
T_train_input = t.TypeVar("T_train_input")
T_predict_input = t.TypeVar("T_predict_input")
T_predict_output = t.TypeVar("T_predict_output")


class FastAPIServiceRunnerFactory(
    t.Generic[T_train_input, T_predict_input, T_predict_output],
    ServiceRunnerFactory,
):

    def __init__(
            self,
            info: ModuleInfo[T_train_input, T_predict_input, T_predict_output],
            fast_api_factory: t.Optional[t.Callable[[], FastAPI]] = None,
            api_router_factory: t.Optional[t.Callable[[str], APIRouter]] = None,
            server_config_factory: t.Optional[t.Callable[[FastAPI], Config]] = None,
            server_factory: t.Optional[t.Callable[[Config], Server]] = None,
    ) -> None:
        self.__info = info
        self.__fast_api_factory = fast_api_factory
        self.__api_router_factory = api_router_factory
        self.__server_config_factory = server_config_factory
        self.__server_factory = server_factory

    def create_service_runner(
            self,
            service: ModelService,
    ) -> ServiceRunner:
        app = self.__create_fastapi()

        app.router.include_router(self.__create_service_router(service))

        config = self.__create_server_config(app)
        server = self.__create_server(config)

        return UvicornServiceRunner(server)

    def __create_fastapi(self) -> FastAPI:
        if self.__fast_api_factory is not None:
            return self.__fast_api_factory()

        return FastAPI()

    def __create_api_router(self, prefix: str) -> APIRouter:
        return (
            self.__api_router_factory(prefix)
            if self.__api_router_factory is not None
            else APIRouter(prefix=prefix)
        )

    def __create_router_registrator(self, router: APIRouter) -> t.Callable[[str, str, int], t.Callable[[F], F]]:

        def setup(path: str, method: str, success_status_code: int) -> t.Callable[[F], F]:
            def register(func: F) -> F:
                return router.api_route(
                    path=path,
                    response_model=func.__annotations__.get("return"),
                    status_code=success_status_code,
                    methods=[method],
                )(func)  # type: ignore[type-var]

            return register

        return setup

    def __create_service_router(
            self,
            service: ModelService,
    ) -> APIRouter:
        # noinspection PyPep8Naming
        TrainInput, PredictInput, PredictOutput = (
            self.__info.train_input_type,
            self.__info.predict_input_type,
            self.__info.predict_output_type,
        )

        if isinstance(service, StaticModelService):
            router = self.__create_api_router("")
            registrator = self.__create_router_registrator(router)

            def make_constant_key() -> str:
                return ""

            key_dependency = Depends(make_constant_key)

        else:
            router = self.__create_api_router("/{key}")
            registrator = self.__create_router_registrator(router)
            key_dependency = Path()

        if isinstance(service, ModelTrainingService):
            model_training_service: ModelTrainingService[  # type: ignore[valid-type]
                str, TrainInput] = service

            @registrator("/", "PUT", status.HTTP_201_CREATED)
            def handle_train(
                    key: str = key_dependency,
                    input_: TrainInput = Body(alias="input"),  # type: ignore[valid-type]
            ) -> None:
                response = model_training_service.train(TrainRequest(key=key, input_=input_))
                if isinstance(response, TrainSuccessResponse):
                    return None

                elif isinstance(response, TrainInternalErrorResponse):
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                        detail={"error": str(response.error)})

                else:
                    raise_not_exhaustive(response)

        if isinstance(service, ModelPredictionService):
            model_prediction_service: ModelPredictionService[  # type: ignore[valid-type]
                str, PredictInput, PredictOutput] = service

            @registrator("/", "POST", status.HTTP_200_OK)
            def handle_predict(
                    key: str = key_dependency,
                    input_: PredictInput = Body(alias="input"),  # type: ignore[valid-type]
            ) -> PredictOutput:  # type: ignore[valid-type]
                response = model_prediction_service.predict(PredictRequest(key=key, input_=input_))
                if isinstance(response, PredictSuccessResponse):
                    return response.output

                elif isinstance(response, PredictModelNotFoundErrorResponse):
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

                elif isinstance(response, PredictModelInternalErrorResponse):
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                        detail={"error": str(response.error)})

                else:
                    raise_not_exhaustive(response)

        if isinstance(service, ModelRemovingService):
            model_removing_service = service

            @registrator("/", "DELETE", status.HTTP_202_ACCEPTED)
            def handle_remove(key: str) -> None:
                response = model_removing_service.remove(RemoveRequest(key=key))
                if isinstance(response, RemoveSuccessResponse):
                    return None

                elif isinstance(response, RemoveModelNotFoundErrorResponse):
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

                else:
                    raise_not_exhaustive(response)

        return router

    def __create_server_config(self, app: FastAPI) -> Config:
        if self.__server_config_factory is not None:
            return self.__server_config_factory(app)

        return Config(app)

    def __create_server(self, config: Config) -> Server:
        if self.__server_factory is not None:
            return self.__server_factory(config)

        return Server(config)
