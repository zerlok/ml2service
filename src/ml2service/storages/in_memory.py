import typing as t

from ml2service.storages.base import Storage

K = t.TypeVar("K", bound=t.Hashable)
V = t.TypeVar("V")


class InMemoryStorage(t.Generic[K, V], Storage[K, V]):

    def __init__(self, data: t.Optional[t.Dict[K, V]] = None) -> None:
        self.__data: t.Dict[K, V] = data or {}

    def get(self, key: K) -> t.Optional[V]:
        return self.__data.get(key)

    def update(self, key: K, value: V) -> None:
        self.__data[key] = value

    def remove(self, key: K) -> t.Optional[V]:
        return self.__data.pop(key, None)
