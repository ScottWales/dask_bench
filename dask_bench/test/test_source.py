from ..source import *


def test_random():
    s = Random((100, 60, 50), chunks={"time": -1, "lat": 20, "lon": 20})
    da = s.load()

    assert da.sizes["lat"] == 60
    assert "time" in da.dims

    assert s.info()["source_type"] == "Random"
