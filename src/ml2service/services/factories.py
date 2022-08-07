import abc
import typing as t

from ml2service.services.base import Service
from ml2service.services.runners.base import ServiceRunner

K = t.TypeVar("K")
T_train_input = t.TypeVar("T_train_input")
T_predict_input = t.TypeVar("T_predict_input")
T_predict_output = t.TypeVar("T_predict_output")


class ServiceFactory(t.Generic[K, T_train_input, T_predict_input, T_predict_output]):
    @abc.abstractmethod
    def create_service(
            self,
            entrypoint: str,
            args: t.Sequence[str],
    ) -> Service[K, T_train_input, T_predict_input, T_predict_output]:
        raise NotImplementedError


class ServiceRunnerFactory(t.Generic[K, T_train_input, T_predict_input, T_predict_output]):
    @abc.abstractmethod
    def create_service_runner(
            self,
            service: Service[K, T_train_input, T_predict_input, T_predict_output],
    ) -> ServiceRunner:
        raise NotImplementedError
