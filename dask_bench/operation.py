import xarray
from .source import Source
import time
import climtas
from datetime import datetime
import typing as T
from functools import lru_cache
import tempfile


class Operation:
    """
    A generic operation
    """

    def run(self, da: xarray.DataArray) -> xarray.DataArray:
        pass

    def benchmark(self, source: Source):
        t0 = time.perf_counter()
        da = source.load()

        t1 = time.perf_counter()
        da_out = self.run(da)

        t2 = time.perf_counter()
        self.save(da_out)

        t3 = time.perf_counter()
        return {
            "op_type": self.__class__.__name__,
            "date": datetime.now(),
            "time_open": t1 - t0,
            "time_run": t2 - t1,
            "time_save": t3 - t2,
            "time_total": t3 - t0,
        }

    def save(self, da: xarray.DataArray):
        with tempfile.NamedTemporaryFile() as f:
            da.to_netcdf(f.name)


@lru_cache
def all_operations() -> T.List[Operation]:
    """
    Returns all available operations
    """
    return [op() for op in Operation.__subclasses__()]


class TimeMean(Operation):
    def run(self, da: xarray.DataArray) -> xarray.DataArray:
        return da.mean("time")


class Climatology(Operation):
    def run(self, da: xarray.DataArray) -> xarray.DataArray:
        return da.groupby("time.dayofyear").mean()


class ClimatologyClimtas(Operation):
    def run(self, da: xarray.DataArray) -> xarray.DataArray:
        return climtas.blocked_groupby(da, time="dayofyear").mean()
