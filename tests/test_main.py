import pytest

from egsis import main

from typer.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


def test_hello_world(runner):
    result = runner.invoke(main.cli, [])
    assert result.exit_code == 0
    assert "Hello world!" in result.stdout
