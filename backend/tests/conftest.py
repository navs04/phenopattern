import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.store import DataStore  # noqa: E402
from tests.fixture_data import make_data  # noqa: E402


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    return make_data(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="session")
def store(data_dir):
    return DataStore.load(data_dir)
