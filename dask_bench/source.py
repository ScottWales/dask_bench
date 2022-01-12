import dask.array
import xarray
import numpy
import json
import typing as T

T_chunks = T.Union[T.Dict[str, int], T.Literal["auto"]]


class Source:
    """
    A generic dataset source
    """

    def __init__(
        self,
        config: T.Dict = {},
        chunks: T_chunks = "auto",
        **kwargs,
    ):
        self.name = config.get("name", hash(json.dumps(config)))
        self.config = config
        self.chunks = chunks

    def load(self) -> xarray.DataArray:
        """
        Load the data as a DataArray
        """
        pass

    def info(self) -> T.Dict[str, T.Any]:
        """
        Get information about the held data
        """
        da = self.load()

        return {
            "source_name": self.name,
            "source_type": self.__class__.__name__,
            "chunks": [self.chunks[d] for d in da.dims]
            if self.chunks != "auto"
            else self.chunks,
            "nbytes": da.nbytes,
        }

    def warmup(self):
        """
        Warm up caches
        """
        da = self.load()
        da.mean().load()

    def is_compatible_with(self, op_name: str) -> bool:
        """
        Is this source compatible with an operation?
        """
        only = self.config.get("operations", {}).get("only", None)

        if only is not None:
            return op_name in only

        return True


def source_factory(config) -> T.Iterable[Source]:
    """
    Set up a source from a config
    """
    t = {sc.__name__: sc for sc in Source.__subclasses__()}[config["type"]]

    for chunks in config.get("chunks", ["auto"]):
        s = t(config=config, chunks=chunks, **config.get("args", {}))
        yield s


class Random(Source):
    """
    Some random data
    """

    def __init__(
        self,
        shape: T.Tuple[int],
        *,
        config: T.Dict = {},
        chunks: T_chunks = "auto",
    ):
        self.shape = shape

        super().__init__(config, chunks)

    def load(self) -> xarray.DataArray:

        dims = ["time", "lat", "lon", "level"]
        attrs = {
            "time": {
                "units": "days since 2001-01-01",
                "axis": "T",
                "calendar": "standard",
            },
            "lat": {"units": "degrees_north", "axis": "Y", "standard_name": "latitude"},
            "lon": {"units": "degrees_east", "axis": "X", "standard_name": "longitude"},
            "level": {"units": "m", "axis": "Z"},
        }

        # Convert from xarray-style chunking by dimension to a plain list
        chunks: T.Union[T.Iterable[int], T.Literal["auto"]]
        if self.chunks == "auto":
            chunks = self.chunks
        else:
            chunks = [self.chunks[d] for d in dims[: len(self.shape)]]

        data = dask.array.random.random(self.shape, chunks=chunks)

        coords = [
            (dims[i], numpy.arange(n), attrs[dims[i]]) for i, n in enumerate(self.shape)
        ]

        da = xarray.DataArray(data, coords=coords, name="sample")

        return xarray.decode_cf(da.to_dataset())["sample"]


class MFDataset(Source):
    def __init__(
        self,
        *,
        config: T.Dict = {},
        path: str,
        var: str,
        chunks: T_chunks,
        concat_dim: str = "time",
    ):
        self.path = path
        self.var = var
        self.concat_dim = concat_dim

        super().__init__(config, chunks)

    def load(self) -> xarray.DataArray:
        ds = xarray.open_mfdataset(
            self.path, chunks=self.chunks, combine="nested", concat_dim=self.concat_dim
        )
        return ds[self.var]
