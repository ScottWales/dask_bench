from ..main import *
from ..source import Random
from ..operation import TimeMean


def test_benchmark():
    s = [Random((100, 50))]
    ops = [TimeMean()]

    for r in benchmark(s, ops):
        assert r["time_run"] > 0
        assert r["nbytes"] > 0


def test_run_from_config():
    r = run_from_config("dask_bench/test/test_config.yaml")
    assert len(r) == 2 * 3
