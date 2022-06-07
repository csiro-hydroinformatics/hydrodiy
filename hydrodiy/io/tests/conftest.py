import pytest
from pytest_allclose import report_rmses


def pytest_terminal_summary(terminalreporter):
    report_rmses(terminalreporter)


def pytest_addoption(parser):
    try:
        parser.addoption("--runslow", action="store_true", default=False, \
                                help="run slow tests")
    except ValueError:
        pass

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow.")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # -- runslow given in cli: do not skip slow tests
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")

    for item in items:
        if "slow" in item.keywords:
            try:
                item.add_marker(skip_slow)
            except ValueError:
                pass
