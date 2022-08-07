import abc
import typing as t

T = t.TypeVar("T")


class Serializer(t.Generic[T], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def encode(self, obj: T) -> bytes:
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, data: bytes) -> T:
        raise NotImplementedError
