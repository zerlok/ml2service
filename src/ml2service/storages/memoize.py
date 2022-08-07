import typing as t

from ml2service.storages.base import Storage

K = t.TypeVar("K")
V = t.TypeVar("V")


class MemoizeStorage(t.Generic[K, V], Storage[K, V]):

    __MISSED: t.Final[object] = object()

    def __init__(self, inner: Storage[K, V]) -> None:
        self.__inner = inner
        self.__cached: t.Dict[K, t.Optional[V]] = {}

    def get(self, key: K) -> t.Optional[V]:
        value = self.__cached.get(key, self.__MISSED)
        if value is self.__MISSED:
            value = self.__cached[key] = self.__inner.get(key)

        return value

    def update(self, key: K, value: V) -> None:
        self.__inner.update(key, value)
        self.__cached.pop(key, None)

    def remove(self, key: K) -> t.Optional[V]:
        value = self.__inner.remove(key)
        self.__cached.pop(key, None)

        return value
