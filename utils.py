import os
import itertools
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import typer
import pandas as pd
from git import Repo
from rich.progress import Progress


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


def clone_repos(
    repo_urls: list[str],
    directory: str,
    num_workers: int,
    summary_progress: Progress = None,
    task_progress: Progress = None,
):
    parsed_urls = [urlparse(url) for url in repo_urls]

    clone_task = summary_progress.add_task(
        "[red]Downloading Repos...",
        total=len(repo_urls),
        visible=len(repo_urls) > 0,
    )

    repos = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}

        for url, parsed in zip(repo_urls, parsed_urls):
            future = executor.submit(
                Repo.clone_from,
                url,
                f"{directory}/{parsed.netloc}{parsed.path}",
            )
            task = task_progress.add_task(f"{url}", total=1)
            futures[future] = (clone_task, task)

        for future in as_completed(futures.keys()):
            summary_task, spinner_task = futures[future]
            summary_progress.update(summary_task, advance=1)
            task_progress.update(spinner_task, completed=1)
            repo = future.result()
            if isinstance(repo, Repo):
                repos.append(repo)

    return repos


def pull_repos(
    repos: list[Repo],
    num_workers: int,
    summary_progress: Progress = None,
    task_progress: Progress = None,
):
    pull_task = summary_progress.add_task(
        "[red]Pulling Repos...",
        total=len(repos),
        visible=len(repos) > 0,
    )
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}

        for repo in repos:
            future = executor.submit(repo.remotes.origin.pull)
            repo_name = os.path.basename(repo.working_dir)
            task = task_progress.add_task(f"{repo_name}", total=1)
            futures[future] = (pull_task, task)

        for future in as_completed(futures.keys()):
            summary_task, spinner_task = futures[future]
            summary_progress.update(summary_task, advance=1)
            task_progress.update(spinner_task, completed=1)
            repo = future.result()
            if isinstance(repo, Repo):
                repos.append(repo)

    return repos
