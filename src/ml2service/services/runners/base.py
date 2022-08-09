import abc

from ml2service.services.base import ModelService


class ServiceRunner(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self) -> None:
        raise NotImplementedError


class ServiceRunnerFactory:
    @abc.abstractmethod
    def create_service_runner(
            self,
            service: ModelService,
    ) -> ServiceRunner:
        raise NotImplementedError
