import os
import re
import time
import random
import pandas as pd
from typing import Iterable
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import RenderableType
from rich.live import Live
from rich.console import Group
from rich.rule import Rule

import torch
import typer
from git import Repo
from tqdm import tqdm
from rich import print
from rich.layout import Layout
from rich.progress import (
    track,
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    DownloadColumn,
    TimeRemainingColumn,
)
from datetime import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from optimum.bettertransformer import BetterTransformer


cli = typer.Typer(add_completion=False)


class Model:
    def __init__(self, checkpoint: str, revision: str = None, cache_dir: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, revision=revision, cache_dir=cache_dir
        )
        model.to(self.device)
        self.model = BetterTransformer.transform(model)

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, revision=revision, cache_dir=cache_dir
        )

    def tokenize(self, texts: list[str]):
        return self.tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True, max_length=256
        ).to(self.device)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


@cli.command("")
def main(
    input: list[str] = typer.Option(..., "--input", "-i", help="Input file or URL."),
    output: str = "output.csv",
    bugfix_threshold: float = None,
    batch_size: int = 64,
    after: datetime = None,
    before: datetime = None,
    sample: bool = False,
    data_dir: str = "data/repositories",
    checkpoint: str = "neuralsentry/starencoder-git-commit-bugfix-classification",
    revision: str = None,
    hf_cache_dir: str = None,
    num_workers: int = 4,
    skip_pull: bool = False,
):
    """
    Classify Git commit messasges given a Git repository URL or a file
    containing URLs.
    """
    urls = []
    for e in input:
        if os.path.exists(e):
            with open(e, "r") as f:
                urls.extend([line.strip() for line in f.readlines()])
        else:
            urls.append(e.strip())

    url_regex = "^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$"
    invalid_urls = []
    for url in urls:
        if not re.match(url_regex, url):
            invalid_urls.append(url)
    if len(invalid_urls) > 0:
        print("[!] Invalid URLs:", invalid_urls)
        raise typer.Exit(code=1)

    clone_urls = [
        url
        for url in urls
        if not os.path.exists(
            f"{data_dir.rstrip('/')}/{urlparse(url).netloc}{urlparse(url).path}"
        )
    ]
    existing_paths = [
        f"{data_dir.rstrip('/')}/{urlparse(url).netloc}{urlparse(url).path}"
        for url in urls
        if os.path.exists(
            f"{data_dir.rstrip('/')}/{urlparse(url).netloc}{urlparse(url).path}"
        )
    ]
    repos = [Repo(path) for path in existing_paths]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        summary_progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[cyan]{task.completed}/{task.total} repos"),
            TimeElapsedColumn(),
        )
        task_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        )
        group = Group(summary_progress, task_progress)

        with Live(group):
            clone_task = summary_progress.add_task(
                "[red]Downloading Repos...",
                total=len(clone_urls),
                visible=len(clone_urls) > 0,
            )
            for url in clone_urls:
                future = executor.submit(
                    Repo.clone_from,
                    url,
                    f"{data_dir}/{urlparse(url).netloc}{urlparse(url).path}",
                )
                task = task_progress.add_task(f"Cloning {url}")
                futures[future] = (clone_task, task)

            pull_task = summary_progress.add_task(
                "[red]Pulling Repos...",
                total=len(existing_paths),
                visible=len(existing_paths) > 0 and not skip_pull,
            )
            if not skip_pull:
                for repo in repos:
                    future = executor.submit(repo.remotes.origin.pull)
                    repo_name = os.path.basename(repo.working_dir)
                    task = task_progress.add_task(f"Pulling {repo_name}")
                    futures[future] = (pull_task, task)
            for future in as_completed(futures.keys()):
                summary_task, spinner_task = futures[future]
                task_progress.remove_task(spinner_task)
                summary_progress.update(summary_task, advance=1)
                repo = future.result()
                if isinstance(repo, Repo):
                    repos.append(repo)

    model = Model(checkpoint, revision, hf_cache_dir)

    summary_progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[cyan]{task.completed}/{task.total} {task.fields[unit]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    task_progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[cyan]{task.completed}/{task.total} {task.fields[unit]}"),
    )
    layout = Layout()
    layout.split_column(
        Layout(summary_progress, size=1),
        Layout(task_progress),
    )

    with Live(layout):
        commit_count = 0
        for repo in repos:
            commands = ["HEAD", "--count"]
            if after:
                commands.extend(["--after", after])
            if before:
                commands.extend(["--before", before])
            count = repo.git.rev_list(*commands)
            commit_count += int(count)

        classification_task = summary_progress.add_task(
            "[red]Classifying Commits...", total=commit_count, unit="commits"
        )

        write_header = True
        batch = []
        for repo in repos:
            repo_task = task_progress.add_task(
                f"{os.path.basename(repo.working_dir)}",
                total=int(repo.git.rev_list(*commands)),
                unit="commits",
            )

            commits = repo.iter_commits(after=after, before=before)

            for commit in commits:
                batch.append(commit)
                if len(batch) == batch_size:
                    inputs = model.tokenize([commit.message for commit in batch])
                    outputs = model(**inputs)
                    predictions = torch.softmax(outputs.logits, dim=-1)

                    labels = []
                    for pred in predictions:
                        if bugfix_threshold:
                            if (
                                pred[int(model.model.config.label2id["bugfix"])]
                                > bugfix_threshold
                            ):
                                labels.append("bugfix")
                            else:
                                labels.append("non-bugfix")
                        else:
                            labels.append(
                                model.model.config.id2label[pred.argmax().item()]
                            )

                    df = pd.DataFrame(
                        {
                            "commit_msg": [commit.message for commit in batch],
                            "sha": [commit.hexsha for commit in batch],
                            "remote_url": [
                                f"{commit.repo.remotes.origin.url}/commit/{commit.hexsha}"
                                if "github" in commit.repo.remotes.origin.url
                                else commit.repo.remotes.origin.url
                                for commit in batch
                            ],
                            "date": [
                                commit.authored_datetime.strftime("%Y-%m-%d %H:%M:%S")
                                for commit in batch
                            ],
                            "label": labels,
                            "bugfix": [
                                prediction[
                                    int(model.model.config.label2id["bugfix"])
                                ].item()
                                for prediction in predictions
                            ],
                            "non-bugfix": [
                                prediction[
                                    int(model.model.config.label2id["non-bugfix"])
                                ].item()
                                for prediction in predictions
                            ],
                        }
                    )

                    extension = os.path.splitext(output)[-1]

                    if extension == ".csv":
                        df.to_csv(output, mode="a", header=write_header, index=False)
                    elif extension == ".jsonl":
                        df.to_json(output, lines=True)
                    elif extension == ".xlsx":
                        df.to_excel(output, mode="a", header=write_header, index=False)

                    write_header = False
                    summary_progress.update(classification_task, advance=len(batch))
                    task_progress.update(repo_task, advance=len(batch))
                    batch = []

            if len(batch) > 0:
                inputs = model.tokenize([commit.message for commit in batch])
                outputs = model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=-1)

                labels = []
                for pred in predictions:
                    if bugfix_threshold:
                        if (
                            pred[int(model.model.config.label2id["bugfix"])]
                            > bugfix_threshold
                        ):
                            labels.append("bugfix")
                        else:
                            labels.append("non-bugfix")
                    else:
                        labels.append(model.model.config.id2label[pred.argmax().item()])

                df = pd.DataFrame(
                    {
                        "commit_msg": [commit.message for commit in batch],
                        "sha": [commit.hexsha for commit in batch],
                        "remote_url": [
                            f"{commit.repo.remotes.origin.url}/commit/{commit.hexsha}"
                            if "github" in commit.repo.remotes.origin.url
                            else commit.repo.remotes.origin.url
                            for commit in batch
                        ],
                        "date": [
                            commit.authored_datetime.strftime("%Y-%m-%d %H:%M:%S")
                            for commit in batch
                        ],
                        "label": labels,
                        "bugfix": [
                            prediction[
                                int(model.model.config.label2id["bugfix"])
                            ].item()
                            for prediction in predictions
                        ],
                        "non-bugfix": [
                            prediction[
                                int(model.model.config.label2id["non-bugfix"])
                            ].item()
                            for prediction in predictions
                        ],
                    }
                )

                extension = os.path.splitext(output)[-1]

                if extension == ".csv":
                    df.to_csv(output, mode="a", header=write_header, index=False)
                elif extension == ".jsonl":
                    df.to_json(output, lines=True)
                elif extension == ".xlsx":
                    df.to_excel(output, mode="a", header=write_header, index=False)

                write_header = False
                summary_progress.update(classification_task, advance=len(batch))
                task_progress.update(repo_task, advance=len(batch))
                batch = []


if __name__ == "__main__":
    cli()
