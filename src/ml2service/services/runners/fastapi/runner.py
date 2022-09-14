# noinspection PyPackageRequirements
from uvicorn import Server

from ml2service.services.runners.base import ServiceRunner


class UvicornServiceRunner(ServiceRunner):

    def __init__(
            self,
            server: Server,
    ) -> None:
        self.__server = server

    def start(self) -> None:
        self.__server.run()

    def stop(self) -> None:
        self.__server.should_exit = True
