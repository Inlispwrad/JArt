from __future__ import annotations

import pytest
from PIL import Image

from jart.engine import Engine
from jart.models.fake import FakeImages
from jart.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "home")


@pytest.fixture
def fake():
    return FakeImages()


@pytest.fixture
def engine(store, fake):
    return Engine(store, fake)


@pytest.fixture
def make_png(tmp_path):
    def make(name, size=(64, 64), color=(200, 30, 30, 255), mode="RGBA"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(mode, size, color if mode == "RGBA" else color[:3]).save(path)
        return path

    return make
