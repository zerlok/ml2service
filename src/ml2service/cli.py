import typing as t
from dataclasses import dataclass
from pathlib import Path

import click

from ml2service.models.base import Model
from ml2service.models.loader import EntrypointLoader, ModuleInfo
from ml2service.services.base import ModelService
from ml2service.services.dynamic import DynamicModelService
from ml2service.services.runners.base import ServiceRunner
from ml2service.services.static import StaticModelService
from ml2service.storages.base import Storage
from ml2service.storages.in_memory import InMemoryStorage
from ml2service.storages.memoize import MemoizeStorage


@dataclass()
class CLIContext:
    info: t.Optional[ModuleInfo[object, object, object]] = None
    service: t.Optional[ModelService] = None
    runner: t.Optional[ServiceRunner] = None


@click.group("ml2service")
@click.argument("entrypoint", type=str)
@click.option("-a", "--entrypoint-arg", "args", type=str, multiple=True)
@click.pass_context
def cli(context: click.Context, entrypoint: str, args: t.Sequence[str]) -> None:
    loader = EntrypointLoader()

    try:
        info = loader.load(entrypoint, args)

    except (ImportError, AttributeError, TypeError, ValueError) as err:
        context.fail(f"failed to load '{entrypoint}' entrypoint: {err!r}")
        # FIXME: mypy knows that this statement is unreachable, but PyCharm don't
        return  # type: ignore[unreachable]

    context.obj = CLIContext(info=info)


@cli.command("info")
@click.pass_obj
def show_info(context: CLIContext) -> None:
    info = context.info
    click.echo(f"trainer: {info.trainer if info is not None else None!r}")
    click.echo(f"train input type: {info.train_input_type if info is not None else None!r}")
    click.echo(f"predict input type: {info.predict_input_type if info is not None else None!r}")
    click.echo(f"predict output type: {info.predict_output_type if info is not None else None!r}")


@cli.group("run")
def run() -> None:
    pass


@run.result_callback()
@click.pass_obj
def start_service_runner(context: CLIContext, *_: object, **__: object) -> None:
    if context.runner is not None:
        context.runner.start()


@run.group("static")
@click.argument("input", type=click.Path(exists=True, resolve_path=True, path_type=Path))
@click.pass_obj
@click.pass_context
def run_static(click_context: click.Context, context: CLIContext, input: Path) -> None:
    info = context.info
    assert info is not None

    if not issubclass(info.train_input_type, Path):
        click_context.fail(f"specified entrypoint expects {info.train_input_type}, but must accept {Path}")
        # FIXME: mypy knows that this statement is unreachable, but PyCharm don't
        return  # type: ignore[unreachable]

    model = info.trainer.train(input)
    context.service = StaticModelService(model)


@run.group("dynamic")
@click.option("--memoize/--no-memoize", "memoize_enabled", is_flag=True, default=False)
@click.pass_obj
def run_dynamic(context: CLIContext, memoize_enabled: bool) -> None:
    info = context.info
    assert info is not None

    storage: Storage[object, Model[object, object]] = InMemoryStorage()

    if memoize_enabled:
        storage = MemoizeStorage(storage)

    context.service = DynamicModelService(info.trainer, storage)


@click.command("http")
@click.option("--port", type=int, default=8000)
@click.pass_obj
@click.pass_context
def http(click_context: click.Context, context: CLIContext, port: int) -> None:
    try:
        # noinspection PyPackageRequirements
        from fastapi import FastAPI
        # noinspection PyPackageRequirements
        from uvicorn import Config

    except ImportError as err:
        click_context.fail(f"failed to import http dependencies: {err!r}")
        # FIXME: mypy knows that this statement is unreachable, but PyCharm don't
        return  # type: ignore[unreachable]

    from ml2service.services.runners.fastapi.factory import FastAPIServiceRunnerFactory

    info = context.info
    service = context.service
    assert info is not None
    assert service is not None

    def make_config(app: FastAPI) -> Config:
        return Config(app, port=port)

    service_runner_factory = FastAPIServiceRunnerFactory(
        info=info,
        server_config_factory=make_config,
    )
    context.runner = service_runner_factory.create_service_runner(service)


run_static.add_command(http)
run_dynamic.add_command(http)


@click.command("amqp")
@click.pass_context
def amqp(click_context: click.Context) -> None:
    click_context.fail(f"not implemented yet")


run_static.add_command(amqp)
run_dynamic.add_command(amqp)

if __name__ == "__main__":
    cli()
