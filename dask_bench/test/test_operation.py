from ..operation import *
from ..source import Random


def test_time_mean():
    s = Random((365, 50))
    op = TimeMean()
    bench = op.benchmark(s)

    assert bench["time_run"] > 0


def test_climatology():
    s = Random((365, 50))
    op = Climatology()
    bench = op.benchmark(s)

    assert bench["time_run"] > 0


def test_climatology_climtas():
    s = Random((365, 50))
    op = ClimatologyClimtas()
    bench = op.benchmark(s)

    assert bench["time_run"] > 0
