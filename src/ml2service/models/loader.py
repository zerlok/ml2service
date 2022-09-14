import typing as t
from dataclasses import dataclass
# noinspection PyProtectedMember
from pkgutil import resolve_name

from ml2service.models.base import ModelTrainer
from ml2service.strict_typing import get_generic_type_vars

T_train_input = t.TypeVar("T_train_input", contravariant=True)
T_predict_input = t.TypeVar("T_predict_input", contravariant=True)
T_predict_output = t.TypeVar("T_predict_output", covariant=True)


@dataclass(frozen=True)
class ModuleInfo(t.Generic[T_train_input, T_predict_input, T_predict_output]):
    train_input_type: t.Type[T_train_input]
    predict_input_type: t.Type[T_predict_input]
    predict_output_type: t.Type[T_predict_output]

    trainer: ModelTrainer[T_train_input, T_predict_input, T_predict_output]


class EntrypointLoader:

    def load(
            self,
            entrypoint: str,
            args: t.Sequence[str],
    ) -> ModuleInfo[object, object, object]:
        func = self.__resolve_callable(entrypoint)

        obj = func(*args)

        if not isinstance(obj, ModelTrainer):
            raise TypeError("model trainer expected", obj)

        train_input_type, predict_input_type, predict_output_type = get_generic_type_vars(ModelTrainer, obj)

        return ModuleInfo(train_input_type, predict_input_type, predict_output_type, obj)

    def __resolve_callable(self, entrypoint: str) -> t.Callable[..., object]:
        obj = resolve_name(entrypoint)

        if not callable(obj):
            raise TypeError("callable expected", obj)

        return obj
