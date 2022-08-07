import abc
import typing as t

K = t.TypeVar("K")
V = t.TypeVar("V")


class Storage(t.Generic[K, V], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get(self, key: K) -> t.Optional[V]:
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, key: K, value: V) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def remove(self, key: K) -> t.Optional[V]:
        raise NotImplementedError
