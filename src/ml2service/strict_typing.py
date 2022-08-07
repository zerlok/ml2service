import typing as t

T1 = t.TypeVar("T1")
T2 = t.TypeVar("T2")


def get_generic_type_vars(type_: t.Type[T1], obj: T2) -> t.Sequence[t.Type[object]]:
    for base in obj.__class__.__orig_bases__:
        origin = t.get_origin(base)
        if origin is type_:
            return t.get_args(base)

    return ()


def raise_not_exhaustive(*args: t.NoReturn) -> t.NoReturn:
    """A helper to make an exhaustiveness check on python expression. See: https://github.com/python/mypy/issues/5818"""
    raise RuntimeError("Not exhaustive expression", *args)
