import os
import re
import warnings
from datetime import datetime
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import typer
import clang.cindex
import pandas as pd
from git import Repo, Commit, Diff
from tqdm import tqdm
from rich import print
from rich.live import Live
from rich.rule import Rule
from typing import Iterable
from rich.table import Table
from rich.layout import Layout
from rich.console import Group
from rich.console import RenderableType
from optimum.bettertransformer import BetterTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from function_extraction import (
    get_hunk_headers_function,
    find_function,
    get_function_source,
)
from utils import batch, read_file, get_repo_num_commits

warnings.filterwarnings("ignore", category=UserWarning, module="optimum")

cli = typer.Typer(add_completion=False)


class Model:
    def __init__(self, checkpoint: str, revision: str = None, cache_dir: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            revision=revision,
            cache_dir=cache_dir,
            label2id={"non-bugfix": 0, "bugfix": 1},
            id2label={0: "non-bugfix", 1: "bugfix"},
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


@cli.command("classify")
def main(
    input: list[str] = typer.Option(..., "--input", "-i", help="Input file or URL."),
    output: str = "output.csv",
    bugfix_threshold: float = None,
    non_bugfix_threshold: float = None,
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
                task = task_progress.add_task(f"{url}", total=1)
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
                    task = task_progress.add_task(f"{repo_name}", total=1)
                    futures[future] = (pull_task, task)
            for future in as_completed(futures.keys()):
                summary_task, spinner_task = futures[future]
                summary_progress.update(summary_task, advance=1)
                task_progress.update(spinner_task, completed=1)
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

    print("[red]\nClassifying using the following configuration:")
    print("GPU/CPU:", torch.cuda.get_device_name(0))
    print("Num Workers:", num_workers)
    print("Batch Size:", batch_size)
    print("Bugfix Threshold:", bugfix_threshold)
    print("Non-bugfix Threshold:", non_bugfix_threshold)
    if before:
        print("Before:", before)
    if after:
        print("After:", after)

    data = {}

    print()
    group = Group(summary_progress, task_progress)
    with Live(group):
        commit_count = 0
        for repo in repos:
            commands = ["HEAD", "--count"]
            if after:
                commands.extend(["--after", after])
            if before:
                commands.extend(["--before", before])
            count = repo.git.rev_list(*commands)
            data[os.path.basename(repo.working_dir)] = {
                "bugfix": 0,
                "non-bugfix": 0,
                "outside-threshold": 0,
                "total": count,
            }
            commit_count += int(count)

        classification_task = summary_progress.add_task(
            "[red]Classifying Commits...", total=commit_count, unit="commits"
        )

        write_header = True
        batch = []
        for repo in repos:
            count = int(repo.git.rev_list(*commands))
            repo_task = task_progress.add_task(
                f"{os.path.basename(repo.working_dir)}",
                total=count,
                unit="commits",
            )

            commits = repo.iter_commits(after=after, before=before)
            num_commits = get_repo_num_commits(repo, after=after, before=before)

            for i, commit in enumerate(commits):
                batch.append(commit)

                if len(batch) == batch_size or (
                    # last batch
                    i == num_commits - 1
                    and len(batch) > 0
                ):
                    inputs = model.tokenize([commit.message for commit in batch])
                    outputs = model(**inputs)
                    predictions = torch.softmax(outputs.logits, dim=-1)

                    labels = []
                    for pred in predictions:
                        bugfix_pred = pred[int(model.model.config.label2id["bugfix"])]
                        non_bugfix_pred = pred[
                            int(model.model.config.label2id["non-bugfix"])
                        ]
                        if bugfix_threshold and non_bugfix_threshold:
                            if bugfix_pred >= bugfix_threshold:
                                labels.append("bugfix")
                                data[os.path.basename(repo.working_dir)]["bugfix"] += 1
                            if non_bugfix_pred >= non_bugfix_threshold:
                                labels.append("non-bugfix")
                                data[os.path.basename(repo.working_dir)][
                                    "non-bugfix"
                                ] += 1
                            # add outside-threshold count to data
                            if (
                                bugfix_pred < bugfix_threshold
                                and non_bugfix_pred < non_bugfix_threshold
                            ):
                                labels.append("outside-threshold")
                                data[os.path.basename(repo.working_dir)][
                                    "outside-threshold"
                                ] += 1
                        elif bugfix_threshold:
                            if bugfix_pred >= bugfix_threshold:
                                labels.append("bugfix")
                                data[os.path.basename(repo.working_dir)]["bugfix"] += 1
                            else:
                                labels.append("non-bugfix")
                                data[os.path.basename(repo.working_dir)][
                                    "non-bugfix"
                                ] += 1
                        elif non_bugfix_threshold:
                            if non_bugfix_pred >= non_bugfix_threshold:
                                labels.append("non-bugfix")
                                data[os.path.basename(repo.working_dir)][
                                    "non-bugfix"
                                ] += 1
                            else:
                                labels.append("bugfix")
                                data[os.path.basename(repo.working_dir)]["bugfix"] += 1
                        else:
                            label = model.model.config.id2label[pred.argmax().item()]
                            labels.append(label)
                            data[os.path.basename(repo.working_dir)][label] += 1

                    df = pd.DataFrame(
                        {
                            "is_merge": [len(commit.parents) > 1 for commit in batch],
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
                            "labels": labels,
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

    table = Table(show_header=True, header_style="red")

    table.add_column("Repository")
    table.add_column("Bugfix", justify="right")
    table.add_column("Non-bugfix", justify="right")
    table.add_column("Outside Threshold", justify="right")
    table.add_column("Total", justify="right")

    for repo, values in data.items():
        table.add_row(
            repo,
            str(values["bugfix"]),
            str(values["non-bugfix"]),
            str(values["outside-threshold"]),
            str(values["total"]),
        )

    print()
    print(table)
    print("[green]\nDone!")
    print(f"[green]Output saved to {output}")


@cli.command("extract")
def extract_functions(
    input: str = typer.Option(..., "--input", "-i", help="Input file"),
    output: str = typer.Option(
        "data/functions.csv", "--output", "-o", help="Output file"
    ),
    per_repo_vuln_max: int = (500,),
    per_repo_non_vuln_max: int = (500,),
    batch_size: int = 64,
    assume_all_vulnerable: bool = True,
):
    """
    Extract functions from classified commits.

    **Must be run on `classify` output.**
    """

    df = read_file(input)

    print("[red]\nExtracting functions from commits...")

    df = df[df["is_merge"] == "False"]
    df = df[~df["labels"].str.contains("outside-threshold")]

    urls = [urlparse(url) for url in df["remote_url"]]
    paths = [
        f"data/repositories/{url.netloc}/{'/'.join(url.path.split('/')[1:3])}"
        for url in urls
    ]
    df["path"] = paths
    repos = {}
    commits: list[Commit] = []
    for i, row in df.iterrows():
        path = row["path"]
        if repos.get(path) is None:
            repos[path] = Repo(path)

        repo = repos[path]
        commits.append(repo.commit(row["sha"]))

    index = clang.cindex.Index.create()
    num_functions = 0
    batch = []
    header = True
    for i, commit in enumerate(tqdm(commits)):
        batch.append(commit)

        if len(batch) == batch_size or (i + 1 == len(commits) and len(batch) > 0):
            results = []
            for i, commit in enumerate(batch):
                label = df.iloc[i]["labels"]
                parent_commit = commit.parents[0]
                diff_items = parent_commit.diff(commit, create_patch=True)

                diffs: list[Diff] = [
                    diff
                    for diff in diff_items.iter_change_type("M")
                    if diff.a_path.endswith(".c")
                ]

                if len(diffs) == 0:
                    continue

                if assume_all_vulnerable is False:
                    print("[!] Not implemented yet.")
                    raise typer.Exit(code=1)

                if not len(diffs) == 1:
                    continue

                for diff in diffs:
                    symbols = get_hunk_headers_function(diff)
                    if len(symbols) == 0:
                        print(
                            f"[!] No function headers found in diff {commit.repo.remotes[0].url}/commit{commit.hexsha}"
                        )
                        continue

                    for symbol in symbols:
                        function_name = re.search(r"(\w+)\(", symbol)

                        if function_name is None:
                            print(
                                f"[!] Failed to regex function name {symbol} in {commit.repo.remotes[0].url}/commit/{commit.hexsha}"
                            )
                            continue

                        function_name = function_name.group(1)
                        if label == "bugfix":
                            path = diff.a_path
                            try:
                                code = commit.repo.git.show(f"{commit.hexsha}:{path}")
                            except:
                                print(
                                    f"[!][yellow] Failed to get code for {commit.repo.remotes[0].url}/commit/{commit.hexsha}/{path}"
                                )
                        elif label == "non-bugfix":
                            path = diff.b_path
                            try:
                                code = commit.repo.git.show(f"{commit.hexsha}:{path}")
                            except:
                                print(
                                    f"[!][yellow] Failed to get code for {parent_commit.repo.remotes[0].url}/commit/{parent_commit.hexsha}/{path}"
                                )
                                continue
                        else:
                            print(f"[red][!] Invalid label {label}")
                        temp_file = "data/temp.c"
                        with open(temp_file, "w", encoding="utf-8") as file:
                            file.write(code)

                        translation_unit = index.parse(temp_file)

                        function = find_function(translation_unit.cursor, function_name)

                        if function is None:
                            print(
                                f"[yellow][!] Failed to find function {function_name} in {commit.repo.remotes[0].url}/commit/{commit.hexsha}"
                            )
                            continue
                        else:
                            num_functions += 1
                            print(
                                f"[+] Found ({num_functions}) {df.iloc[i]['labels']} function {function_name} in {commit.repo.remotes[0].url}/commit/{commit.hexsha}"
                            )

                        results.append(
                            {
                                "commit_msg": commit.message,
                                "commit_hash": commit.hexsha,
                                "repo_url": commit.repo.remotes[0].url,
                                "commit_url": f"{commit.repo.remotes[0].url}/commit/{commit.hexsha}",
                                "labels": df.iloc[i]["labels"],
                                "function": get_function_source(temp_file, function),
                            }
                        )

            if len(results) > 0:
                results_df = pd.DataFrame(results)
                results_df.to_csv(
                    output,
                    mode="a",
                    index=False,
                    header=header,
                )
                header = False

            batch = []


if __name__ == "__main__":
    cli()
