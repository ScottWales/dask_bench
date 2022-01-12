import typing as T
import yaml
import argparse
import io
import jsonschema
import pandas
import sys
import dask.distributed
import socket
from .source import Source, source_factory
from .operation import Operation, all_operations


def benchmark(
    source: T.Iterable[Source], operations: T.Iterable[Operation], repeats: int = 3
):
    """
    Run the benchmark operations for a single source

    Args:
        source: Sources to load
        operations: Operations to run on the sources
        repeats: Number of times to repeat each operation
    """
    for s in source:
        info = s.info()
        s.warmup()

        for op in operations:
            if not s.is_compatible_with(op.__class__.__name__):
                continue

            for _ in range(repeats):
                r = op.benchmark(s)
                yield info | r


def run_from_config(config: T.Union[str, io.IOBase], meta: T.Dict[str, str] = {}):
    """
    Run the operations from a config file

    Args:
        config: Config file to use
        meta: Extra metadata to add to results
    """
    if isinstance(config, io.IOBase):
        c = yaml.safe_load(config)
    else:
        with open(config) as f:
            c = yaml.safe_load(f)

    with open("dask_bench/config_schema.yaml") as f:
        schema = yaml.safe_load(f)

    jsonschema.validate(c, schema)

    all_ops = all_operations()
    results = []

    for s_config in c.get("sources", []):
        s = source_factory(s_config)

        for r in benchmark(s, all_ops):
            results.append(r | meta)
            print(r["source_name"], r["op_type"], r["time_total"])

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        help="benchmark config file",
        required=True,
        type=argparse.FileType("r"),
    )
    parser.add_argument(
        "--output",
        "-o",
        help="output csv file (default: stdout)",
        type=argparse.FileType("a"),
        default=sys.stdout,
    )
    parser.add_argument(
        "--distributed",
        help="use dask.distributed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    meta = {"host": socket.gethostname(), "ncpus": 1, "threads": 1}

    if args.distributed:
        c = dask.distributed.Client()
        meta["ncpus"] = sum([t for t in c.nthreads().values()])
        meta["threads"] = max([t for t in c.nthreads().values()])

    r = run_from_config(args.config, meta)

    if args.distributed:
        c.close()

    df = pandas.DataFrame(r)
    df.to_csv(
        args.output,
        index=False,
        header=(not args.output.seekable()) or (args.output.tell() == 0),
    )


if __name__ == "__main__":
    main()
