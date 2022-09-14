import typing as t


def get_generic_type_vars(type_: t.Type[object], obj: object) -> t.Sequence[t.Type[object]]:
    for base in obj.__class__.__orig_bases__:  # type: ignore
        origin = t.get_origin(base)  # type: ignore[misc]
        if origin is type_:  # type: ignore[misc]
            return t.get_args(base)  # type: ignore[misc]

    return ()


def raise_not_exhaustive(*args: t.NoReturn) -> t.NoReturn:  # pragma: no cover
    """A helper to make an exhaustiveness check on python expression. See: https://github.com/python/mypy/issues/5818"""
    raise RuntimeError("Not exhaustive expression", *args)
