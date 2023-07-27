import os
import itertools

import typer
import pandas as pd
from git import Repo


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


def get_repo_num_commits(repo: Repo, before=None, after=None):
    args = ["--count"]

    if before:
        args.append(f"--before={before}")
    if after:
        args.append(f"--after={after}")

    args.append("HEAD")
    return int(repo.git.rev_list(*args))


def read_lines(path: str, strip=False):
    lines = []
    with open(path, "r") as f:
        lines = f.readlines()
        if strip:
            lines = [line.strip() for line in lines]
    return lines
