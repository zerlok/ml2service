import typing as t

import pytest
import requests


@pytest.mark.parametrize(("args",), [
    pytest.param(
        ["examples.myproject.models:FooDynamicModelTrainer", "run", "dynamic", "http"],
        id="dynamic model",
    ),
    pytest.param(
        ["examples.myproject.models:FooStaticModelTrainer", "run", "static", "src/examples/myproject/data.json",
         "http"],
        id="static model",
    ),
])
def test_http_docs_available(
        cli_service_runner: t.Callable[[t.Sequence[str]], str],
        args: t.Sequence[str],
) -> None:
    base_url = cli_service_runner(args)

    response = requests.get(f"{base_url}/docs")
    assert response.status_code == 200
