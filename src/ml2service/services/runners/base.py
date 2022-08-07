import abc


class ServiceRunner(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self) -> None:
        raise NotImplementedError
