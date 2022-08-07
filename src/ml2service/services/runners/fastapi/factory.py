import typing as t

# noinspection PyPackageRequirements
from fastapi import APIRouter, Body, FastAPI, HTTPException
# noinspection PyPackageRequirements
from starlette import status
# noinspection PyPackageRequirements
from uvicorn import Config, Server

from ml2service.models.loader import ModuleInfo
from ml2service.services.base import (
    PredictModelInternalErrorResponse,
    PredictModelNotFoundErrorResponse,
    PredictRequest,
    PredictSuccessResponse,
    RemoveModelNotFoundErrorResponse,
    RemoveRequest,
    RemoveSuccessResponse,
    Service,
    TrainInternalErrorResponse,
    TrainRequest,
    TrainSuccessResponse,
)
from ml2service.services.factories import ServiceRunnerFactory
from ml2service.services.runners.base import ServiceRunner
from ml2service.services.runners.fastapi.runner import UvicornServiceRunner
from ml2service.strict_typing import raise_not_exhaustive

T_train_input = t.TypeVar("T_train_input")
T_predict_input = t.TypeVar("T_predict_input")
T_predict_output = t.TypeVar("T_predict_output")


class FastAPIServiceRunnerFactory(
    t.Generic[T_train_input, T_predict_input, T_predict_output],
    ServiceRunnerFactory[str, T_train_input, T_predict_input, T_predict_output],
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
            service: Service[str, T_train_input, T_predict_input, T_predict_output],
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
        if self.__api_router_factory is not None:
            return self.__api_router_factory(prefix)

        return APIRouter(prefix=prefix)

    def __create_service_router(
            self,
            service: Service[str, T_train_input, T_predict_input, T_predict_output],
    ) -> APIRouter:
        router = self.__create_api_router("/{key}")

        TrainInput, PredictInput, PredictOutput \
            = self.__info.train_input_type, self.__info.predict_input_type, self.__info.predict_output_type

        @router.put("/", status_code=status.HTTP_201_CREATED)
        def handle_train(
                key: str,
                input_: TrainInput = Body(alias="input"),
        ) -> None:
            response = service.train(TrainRequest(key=key, input_=input_))
            if isinstance(response, TrainSuccessResponse):
                return None

            elif isinstance(response, TrainInternalErrorResponse):
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                    detail={"error": str(response.error)})

            else:
                return raise_not_exhaustive(response)

        @router.post("/", response_model=PredictOutput)
        def handle_predict(
                key: str,
                input_: PredictInput = Body(alias="input"),
        ) -> PredictOutput:
            response = service.predict(PredictRequest(key=key, input_=input_))
            if isinstance(response, PredictSuccessResponse):
                return response.output

            elif isinstance(response, PredictModelNotFoundErrorResponse):
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

            elif isinstance(response, PredictModelInternalErrorResponse):
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                    detail={"error": str(response.error)})

            else:
                return raise_not_exhaustive(response)

        @router.delete("/", status_code=status.HTTP_202_ACCEPTED)
        def handle_remove(key: str) -> None:
            response = service.remove(RemoveRequest(key=key))
            if isinstance(response, RemoveSuccessResponse):
                return None

            elif isinstance(response, RemoveModelNotFoundErrorResponse):
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

            else:
                return raise_not_exhaustive(response)

        return router

    def __create_server_config(self, app: FastAPI) -> Config:
        if self.__server_config_factory is not None:
            return self.__server_config_factory(app)

        return Config(app)

    def __create_server(self, config: Config) -> Server:
        if self.__server_factory is not None:
            return self.__server_factory(config)

        return Server(config)
