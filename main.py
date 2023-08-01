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
from tqdm import tqdm
from rich import print
from rich.live import Live
from rich.rule import Rule
from typing import Iterable
from rich.table import Table
from rich.layout import Layout
from rich.console import Group
from git import Repo, Commit, Diff
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

from utils import (
    batch,
    read_file,
    get_repo_num_commits,
    read_lines,
    clone_repos,
    pull_repos,
)
from function_extraction import (
    get_hunk_headers_function,
    find_function,
    get_function_source,
)

warnings.filterwarnings("ignore", category=UserWarning, module="optimum")

cli = typer.Typer(add_completion=False)


class Model:
    def __init__(self, checkpoint: str, revision: str = None, cache_dir: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            revision=revision,
            cache_dir=cache_dir,
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
    input: list[str] = typer.Option(
        ...,
        "--input",
        "-i",
        help="GitHub repository URL(s) or path to a file containing URLs.",
    ),
    output: str = typer.Option(
        "output.csv",
        "--output",
        "-o",
        help="Output file (.csv|.jsonl|.xlsx).",
    ),
    vulnfix_threshold: float = typer.Option(
        None,
        min=0,
        max=1,
        help="If `--non-vulnfix-threshold` is also set, values outside them will be classified as `outside-threshold`.",
    ),
    non_vulnfix_threshold: float = typer.Option(
        None,
        min=0,
        max=1,
        help="If `--vulnfix-threshold` is also set, values outside them will be classified as `outside-threshold`.",
    ),
    batch_size: int = typer.Option(
        64,
        "-b",
        "--batch-size",
        min=1,
        help="Large size may be faster, but uses more memory.",
    ),
    after: datetime = typer.Option(
        datetime(datetime.now().year, 1, 1),
        help="Only classify commits after this date. Format: YYYY-MM-DD.",
    ),
    before: datetime = typer.Option(
        None,
        help="Only classify commits before this date. Format: YYYY-MM-DD.",
    ),
    data_dir: str = typer.Option(
        "data/repositories",
        help="Directory to clone repositories to.",
    ),
    checkpoint: str = typer.Option(
        "neuralsentry/starencoder-vulnfix-classification-balanced",
        help="Model checkpoint to use.",
    ),
    revision: str = typer.Option(
        None,
        help="Revision of the model to use. Change this if you want to use a different model version than the latest.",
    ),
    hf_cache_dir: str = typer.Option(
        None, help="HuggingFace cache directory. Defaults to your home directory."
    ),
    num_workers: int = typer.Option(
        4,
        "-w",
        "--num-workers",
        min=1,
        help="Number of workers to use for cpu-bound tasks.",
    ),
):
    """
    Classify Git commit messasges given a Git repository URL or a file
    containing URLs.
    """
    # Process `--input`
    urls = [url for url in input if not os.path.exists(url)]
    paths = [path for path in input if os.path.exists(path)]
    for path in paths:
        urls.extend(read_lines(path, strip=True))
    urls = [url.strip() for url in urls]
    urls = list(set(urls))

    # Validate `--input`
    github_repo_url_regex = (
        "^https?:\\/\\/(?:www\\.)?github\\.com\\/[a-zA-Z0-9-_]+\\/[a-zA-Z0-9-_]+$"
    )
    invalid_urls = [url for url in urls if not re.match(github_repo_url_regex, url)]

    if len(invalid_urls) > 0:
        print(
            "[red][!] Invalid GitHub repository URL(s):",
            ", ".join(invalid_urls),
        )
        raise typer.Exit(code=1)

    parsed_urls = [urlparse(url) for url in urls]
    repo_paths = [
        f"{data_dir.rstrip('/')}/{url.netloc}{url.path}" for url in parsed_urls
    ]

    # Clone and/or pull repos
    repo_urls_to_clone = [
        url for url, path in zip(urls, repo_paths) if not os.path.exists(path)
    ]
    repos_to_pull = [Repo(path) for path in repo_paths if os.path.exists(path)]

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

    repos = []
    with Live(group):
        cloned_repos = clone_repos(
            repo_urls_to_clone,
            data_dir,
            num_workers,
            summary_progress=summary_progress,
            task_progress=task_progress,
        )

        pulled_repos = pull_repos(
            repos_to_pull,
            num_workers,
            summary_progress=summary_progress,
            task_progress=task_progress,
        )

        repos.extend(cloned_repos)
        repos.extend(pulled_repos)

    print("[red]\nClassifying using the following configuration:")
    if before:
        print("Before:", before)
    if after:
        print("After:", after)
    print("GPU/CPU:", torch.cuda.get_device_name(0))
    print("Vulnfix Threshold:", vulnfix_threshold)
    print("Non-vulnfix Threshold:", non_vulnfix_threshold)
    print("Num Workers:", num_workers)
    print("Batch Size:", batch_size)
    print()

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

    table_data = {}
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
            table_data[os.path.basename(repo.working_dir)] = {
                "vulnfix": 0,
                "non-vulnfix": 0,
                "undetermined": 0,
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
                        vulnfix_pred = pred[1]
                        non_vulnfix_pred = pred[0]
                        if vulnfix_threshold and non_vulnfix_threshold:
                            if vulnfix_pred >= vulnfix_threshold:
                                labels.append("vulnfix")
                                table_data[os.path.basename(repo.working_dir)][
                                    "vulnfix"
                                ] += 1
                            if non_vulnfix_pred >= non_vulnfix_threshold:
                                labels.append("non-vulnfix")
                                table_data[os.path.basename(repo.working_dir)][
                                    "non-vulnfix"
                                ] += 1
                            if (
                                vulnfix_pred < vulnfix_threshold
                                and non_vulnfix_pred < non_vulnfix_threshold
                            ):
                                labels.append("undetermined")
                                table_data[os.path.basename(repo.working_dir)][
                                    "undetermined"
                                ] += 1
                        elif vulnfix_threshold:
                            if vulnfix_pred >= vulnfix_threshold:
                                labels.append("vulnfix")
                                table_data[os.path.basename(repo.working_dir)][
                                    "vulnfix"
                                ] += 1
                            else:
                                labels.append("non-vulnfix")
                                table_data[os.path.basename(repo.working_dir)][
                                    "non-vulnfix"
                                ] += 1
                        elif non_vulnfix_threshold:
                            if non_vulnfix_pred >= non_vulnfix_threshold:
                                labels.append("non-vulnfix")
                                table_data[os.path.basename(repo.working_dir)][
                                    "non-vulnfix"
                                ] += 1
                            else:
                                labels.append("vulnfix")
                                table_data[os.path.basename(repo.working_dir)][
                                    "vulnfix"
                                ] += 1
                        else:
                            label = (
                                "vulnfix"
                                if pred.argmax().item() == 1
                                else "non-vulnfix"
                            )
                            labels.append(label)
                            table_data[os.path.basename(repo.working_dir)][label] += 1

                    df = pd.DataFrame(
                        {
                            "is_merge": [len(commit.parents) > 1 for commit in batch],
                            "commit_msg": [commit.message for commit in batch],
                            "commit_hash": [commit.hexsha for commit in batch],
                            "repo_url": [
                                commit.repo.remotes.origin.url for commit in batch
                            ],
                            "date": [
                                commit.authored_datetime.strftime("%Y-%m-%d %H:%M:%S")
                                for commit in batch
                            ],
                            "labels": labels,
                            "vulnfix": [
                                prediction[1].item() for prediction in predictions
                            ],
                            "non-vulnfix": [
                                prediction[0].item() for prediction in predictions
                            ],
                        }
                    )
                    df.insert(4, "commit_url", "")
                    df["commit_url"] = df["repo_url"] + "/commit/" + df["commit_hash"]

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
            task_progress.remove_task(repo_task)

    table = Table(show_header=True, header_style="red")

    table.add_column("Repository")
    table.add_column("Vulnfix", justify="right")
    table.add_column("Non-vulnfix", justify="right")
    table.add_column("Undetermined", justify="right")
    table.add_column("Total", justify="right")

    for repo, values in table_data.items():
        table.add_row(
            repo,
            str(values["vulnfix"]),
            str(values["non-vulnfix"]),
            str(values["undetermined"]),
            str(values["total"]),
        )
    table.add_row(
        "Total",
        str(sum([values["vulnfix"] for values in table_data.values()])),
        str(sum([values["non-vulnfix"] for values in table_data.values()])),
        str(sum([values["undetermined"] for values in table_data.values()])),
        str(sum([int(values["total"]) for values in table_data.values()])),
    )

    print()
    print(table)
    print("[green]\nDone!")
    print(f"[green]Output saved to {output}")


@cli.command("extract")
def extract_functions(
    input: str = typer.Option(
        ..., "--input", "-i", help="Path to classified commits file (using `classify`)."
    ),
    output: str = typer.Option(
        "data/functions.csv", "--output", "-o", help="Output file"
    ),
    vulnfix_threshold: float = typer.Option(
        None, help="If not specified, this command will look at the `labels` column"
    ),
    non_vulnfix_threshold: float = typer.Option(
        None, help="If not specified, this command will look at the `labels` column"
    ),
    per_repo_vuln_max: int = typer.Option(
        None, help="Max number of vulnfix functions per repo."
    ),
    per_repo_non_vuln_max: int = typer.Option(
        None, help="Max number of non-vulnfix functions per repo."
    ),
    batch_size: int = 64,
    assume_all_vulnerable: bool = typer.Option(
        True, help="If disabled, will only extract commits with one function modified"
    ),
    include_file: bool = typer.Option(
        False, help="If enabled, will include the file code in the output"
    ),
    shuffle: bool = typer.Option(
        False, help="If enabled, will shuffle the commits before extracting functions"
    ),
    seed: int = typer.Option(
        42, help="Seed to use for shuffling. Only used if `--shuffle` is enabled."
    ),
    resume: bool = typer.Option(
        False, help="If enabled, will resume from the last extracted commit"
    ),
):
    """
    Extract functions from classified commits.

    **Must be run on `classify` output.**
    """
    if (vulnfix_threshold and not non_vulnfix_threshold) or (
        non_vulnfix_threshold and not vulnfix_threshold
    ):
        print(
            "[!] Must specify both `--vulnfix-threshold` and `--non-vulnfix-threshold`"
        )
        raise typer.Exit(code=1)

    df = read_file(input)

    repo_urls = df["repo_url"].unique().tolist()
    parsed_urls = [urlparse(url) for url in repo_urls]
    repo_paths = [f"data/repositories/{url.netloc}{url.path}" for url in parsed_urls]

    repos = {url: Repo(path) for url, path in zip(repo_urls, repo_paths)}

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
    counter = Progress(
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[cyan]({task.completed}/{task.total} {task.fields[unit]})"),
    )
    counter_without_total = Progress(
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[cyan]({task.completed} {task.fields[unit]})"),
    )

    table_data = {}
    group = Group(summary_progress, counter, counter_without_total, task_progress)

    df = df[df["is_merge"] == False]
    if shuffle:
        df = df.sample(frac=1, random_state=seed)

    with Live(group):
        for repo in repos.values():
            table_data[os.path.basename(repo.working_dir)] = {
                "vuln": 0,
                "non-vuln": 0,
            }

        extraction_task = summary_progress.add_task(
            "[red]Extracting Functions...", total=len(df), unit="commits"
        )
        extracted_commits_count = counter.add_task(
            f"[+] Extracted commits",
            total=len(df),
            unit="commits",
        )
        outside_threshold_count = counter.add_task(
            f"[-] Ignored commits outside threshold",
            total=len(df),
            visible=False,
            unit="commits",
        )
        no_c_diffs_count = counter.add_task(
            f"[-] Ignored commits without C/C++ changes",
            total=len(df),
            visible=False,
            unit="commits",
        )
        more_than_one_function_count = counter.add_task(
            f"[-] Ignored commits modifying more than one function",
            total=len(df),
            visible=False,
            unit="commits",
        )
        no_functions_found_count = counter.add_task(
            f"[-] Ignored commits with no functions found",
            total=len(df),
            visible=False,
            unit="commits",
        )
        ignored_commits_from_max_vuln_per_repo_count = counter.add_task(
            f"[-] Ignored commits (max vuln per repo)",
            total=len(df),
            visible=False,
            unit="commits",
        )
        ignored_commits_from_max_non_vuln_per_repo_count = counter.add_task(
            f"[-] Ignored commits (max non-vuln per repo)",
            total=len(df),
            visible=False,
            unit="commits",
        )
        num_repos = df["repo_url"].nunique()
        extracted_vuln_functions_count = counter.add_task(
            f"[+] Extracted functions (vuln)",
            total=0 if per_repo_vuln_max is None else per_repo_vuln_max * num_repos,
            visible=False,
            unit="functions",
        )
        extracted_non_vuln_functions_count = counter.add_task(
            f"[+] Extracted functions (non-vuln)",
            total=0
            if per_repo_non_vuln_max is None
            else per_repo_non_vuln_max * num_repos,
            visible=False,
            unit="functions",
        )
        failed_extract_functions_count = counter.add_task(
            f"[-] Failed to extract functions",
            total=0,
            visible=False,
            unit="functions",
        )

        index = clang.cindex.Index.create()
        batch = pd.DataFrame()
        header = True
        for i, row in df.iterrows():
            if len(batch) >= batch_size or (i + 1 == len(df) and len(batch) > 0):
                functions_df = pd.DataFrame(batch)

                if not include_file:
                    functions_df = functions_df.drop(columns=["file_code"])

                functions_df.to_csv(
                    output,
                    mode="a",
                    index=False,
                    header=header,
                )
                header = False

                batch = pd.DataFrame()

            repo = repos[row["repo_url"]]
            repo_name = os.path.basename(repo.working_dir)
            commit = repo.commit(row["commit_hash"])

            label = ""

            # Determine label
            if vulnfix_threshold and non_vulnfix_threshold:
                preds = [row["non-vulnfix"], row["vulnfix"]]

                if preds[0] >= non_vulnfix_threshold:
                    label = "non-vuln"
                elif preds[1] >= vulnfix_threshold:
                    label = "vuln"
                # ignore commits outside thresholds
                else:
                    counter.update(outside_threshold_count, advance=1, visible=True)
                    summary_progress.update(extraction_task, advance=1)
                    continue
            else:
                print(
                    "[!] Not implemented yet. Must specify both `--vulnfix-threshold` and `--non-vulnfix-threshold`"
                )
                raise typer.Exit(code=1)

            num_vuln_functions = table_data[repo_name]["vuln"]
            num_non_vuln_functions = table_data[repo_name]["non-vuln"]

            if (
                per_repo_vuln_max
                and label == "vuln"
                and num_vuln_functions >= per_repo_vuln_max
            ):
                counter.update(
                    ignored_commits_from_max_vuln_per_repo_count,
                    advance=1,
                    visible=True,
                )
                summary_progress.update(extraction_task, advance=1)
                continue
            if (
                per_repo_non_vuln_max
                and label == "non-vuln"
                and num_non_vuln_functions >= per_repo_non_vuln_max
            ):
                counter.update(
                    ignored_commits_from_max_non_vuln_per_repo_count,
                    advance=1,
                    visible=True,
                )
                summary_progress.update(extraction_task, advance=1)
                continue

            parent_commit = commit.parents[0] if len(commit.parents) > 0 else None

            if parent_commit is None:
                summary_progress.update(extraction_task, advance=1)
                continue

            diffs = parent_commit.diff(commit, create_patch=True)
            diffs = [
                diff
                for diff in diffs.iter_change_type("M")
                if (diff.a_path.endswith(".c") or diff.a_path.endswith(".cpp"))
            ]

            if len(diffs) == 0:
                counter.update(no_c_diffs_count, advance=1, visible=True)
                summary_progress.update(extraction_task, advance=1)
                continue

            # extract function symbols
            functions = []
            for diff in diffs:
                if label == "vuln":
                    path = diff.a_path
                    code = parent_commit.repo.git.show(f"{parent_commit.hexsha}:{path}")
                elif label == "non-vuln":
                    path = diff.b_path
                    code = commit.repo.git.show(f"{commit.hexsha}:{path}")
                else:
                    print("[!] Why isn't the labelled defined 🤔 (BUG!).")
                    raise typer.Exit(code=1)

                symbols = get_hunk_headers_function(diff)
                if len(symbols) == 0:
                    continue

                for symbol in symbols:
                    function_name = re.search(r"(\w+)\(", symbol)
                    if function_name:
                        functions.append(
                            {
                                "labels": label,
                                "vuln": row["vulnfix"],
                                "non-vuln": row["non-vulnfix"],
                                "function_symbol": symbol,
                                "function_name": function_name.group(1),
                                "function_code": None,
                                "file_path": path,
                                "file_code": code,
                                "repo_url": row["repo_url"],
                                "commit_msg": row["commit_msg"],
                                "commit_hash": row["commit_hash"],
                                "commit_url": row["commit_url"],
                                "file_url": row["repo_url"]
                                + "/blob/"
                                + row["commit_hash"]
                                + "/"
                                + path,
                                "date": row["date"],
                            }
                        )

            if not assume_all_vulnerable and not len(functions) == 1:
                counter.update(more_than_one_function_count, advance=1, visible=True)
                summary_progress.update(extraction_task, advance=1)
                continue

            if len(functions) == 0:
                counter.update(no_functions_found_count, advance=1, visible=True)
                summary_progress.update(extraction_task, advance=1)
                continue

            # extract function source
            extracted_functions = []
            for function in functions:
                try:
                    with open("data/temp.c", "w", encoding="utf-8") as file:
                        file.write(function["file_code"])
                except:
                    counter.update(
                        failed_extract_functions_count, advance=1, visible=True
                    )
                    continue

                try:
                    translation_unit = index.parse("data/temp.c")
                except:
                    counter.update(
                        failed_extract_functions_count, advance=1, visible=True
                    )
                    continue

                function_cursor = find_function(
                    translation_unit.cursor, function["function_name"]
                )

                if function_cursor is None:
                    counter.update(
                        failed_extract_functions_count, advance=1, visible=True
                    )
                    continue

                try:
                    function_code = get_function_source("data/temp.c", function_cursor)
                except:
                    counter.update(
                        failed_extract_functions_count, advance=1, visible=True
                    )
                    continue

                function["function_code"] = function_code
                extracted_functions.append(function)

            counter.update(extracted_commits_count, advance=1)

            if label == "vuln":
                counter.update(
                    extracted_vuln_functions_count,
                    advance=len(extracted_functions),
                    visible=True,
                )
                table_data[repo_name]["vuln"] += len(extracted_functions)

            elif label == "non-vuln":
                counter.update(
                    extracted_non_vuln_functions_count,
                    advance=len(extracted_functions),
                    visible=True,
                )
                table_data[repo_name]["non-vuln"] += len(extracted_functions)

            batch = pd.concat([batch, pd.DataFrame(extracted_functions)])
            summary_progress.update(extraction_task, advance=1)

    table = Table(show_header=True, header_style="red")

    table.add_column("Repository")
    table.add_column("Vuln", justify="right")
    table.add_column("Non-vuln", justify="right")

    for repo, values in table_data.items():
        table.add_row(
            repo,
            str(values["vuln"]),
            str(values["non-vuln"]),
        )
    table.add_row(
        "Total",
        str(sum([values["vuln"] for values in table_data.values()])),
        str(sum([values["non-vuln"] for values in table_data.values()])),
    )

    print()
    print(table)
    print("[green]\nDone!")
    print(f"[green]Output saved to {output}")


if __name__ == "__main__":
    cli()
