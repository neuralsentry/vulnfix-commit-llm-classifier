import os
import itertools

import typer
import pandas as pd


def batch(iterable, size):
    sourceiter = iter(iterable)
    while True:
        batchiter = itertools.islice(sourceiter, size)
        yield list(batchiter)


def read_file(path: str):
    extension = os.path.splitext(path)[1]

    match extension:
        case ".csv":
            return pd.read_csv(path)
        case ".jsonl":
            return pd.read_json(path, lines=True)
        case ".xlsx":
            return pd.read_excel(path)
        case ".parquet":
            return pd.read_parquet(path)
        case _:
            print(f"[!] File extension {extension} not supported")
            raise typer.Exit(1)
