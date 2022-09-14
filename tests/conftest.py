import re
import typing as t
from threading import Thread
from time import sleep

import click
import pytest
from click.testing import CliRunner

from ml2service.services.runners.base import ServiceRunner

# 'INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)'
UVICORN_SERVER_IS_RUNNING_PATTERN = re.compile(r"uvicorn running on (?P<base_url>[^ ]+)",
                                               flags=re.MULTILINE | re.IGNORECASE)


@pytest.fixture()
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def cli_service_runner(cli_runner: CliRunner) -> t.Iterable[t.Callable[[t.Sequence[str]], str]]:
    from ml2service.cli import CLIContext, cli, run

    thread: t.Optional[Thread] = None
    runner: t.Optional[ServiceRunner] = None

    @run.result_callback(replace=True)
    @click.pass_obj
    def start_service_runner(context: CLIContext, *_: object, **__: object) -> None:
        nonlocal thread, runner

        runner = context.runner
        assert runner is not None

        thread = Thread(target=runner.start)
        thread.start()

        sleep(3.0)
        assert thread.is_alive()

    def invoke(args: t.Sequence[str]) -> str:
        result = cli_runner.invoke(cli, args)

        if result.exception is not None:
            raise result.exception

        if match := UVICORN_SERVER_IS_RUNNING_PATTERN.search(result.output):
            return match.group("base_url")

        raise ValueError("failed to find base url", result.output)

    yield invoke

    if runner is not None:
        runner.stop()

    if thread is not None:
        thread.join(3.0)
