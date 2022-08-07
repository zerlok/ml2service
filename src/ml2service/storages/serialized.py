import typing as t

from ml2service.serializers.base import Serializer
from ml2service.storages.base import Storage

K = t.TypeVar("K")
V = t.TypeVar("V")


class SerializedStorage(Storage[K, V]):
    def __init__(self, inner: Storage[K, bytes], serializer: Serializer[V]) -> None:
        self.__inner = inner
        self.__serializer = serializer

    def get(self, key: K) -> t.Optional[V]:
        serialized_value = self.__inner.get(key)
        return self.__decode(serialized_value)

    def update(self, key: K, value: V) -> None:
        serialized_value = self.__serializer.encode(value)
        self.__inner.update(key, serialized_value)

    def remove(self, key: K) -> t.Optional[V]:
        serialized_value = self.__inner.remove(key)
        return self.__decode(serialized_value)

    def __decode(self, value: t.Optional[bytes]) -> t.Optional[V]:
        return self.__serializer.decode(value) if value is not None else None
