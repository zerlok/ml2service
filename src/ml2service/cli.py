import typing as t

import click

from ml2service.models.loader import EntrypointLoader, ModuleInfo
from ml2service.services.base import Service
from ml2service.services.runners.fastapi.factory import FastAPIServiceRunnerFactory
from ml2service.services.stored import StoredModelService
from ml2service.storages.in_memory import InMemoryStorage
from ml2service.storages.memoize import MemoizeStorage


@click.group("ml2service")
@click.argument("entrypoint", type=str)
@click.option("-a", "--entrypoint-arg", "args", type=str, multiple=True)
@click.pass_context
def cli(context: click.Context, entrypoint: str, args: t.Sequence[str]) -> None:
    loader = EntrypointLoader()
    module = loader.load(entrypoint, args)

    storage = MemoizeStorage(InMemoryStorage())
    service = StoredModelService(module.trainer, storage)

    context.obj = (module, service)


@cli.command("http")
@click.option("--port", type=int, default=8080)
@click.pass_obj
def http(module_and_service: t.Tuple[ModuleInfo, Service], port: int) -> None:
    module, service = module_and_service
    service_runner_factory = FastAPIServiceRunnerFactory(module)
    runner = service_runner_factory.create_service_runner(service)

    runner.start()


if __name__ == "__main__":
    cli()
