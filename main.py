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
from pydriller import Repository
from pydriller.domain.commit import Method
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

from collect_commits import get_method_code, changed_methods_both

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
    vulnfix_threshold: float = typer.Option(0.5),
    non_vulnfix_threshold: float = typer.Option(0.5),
    max_vuln_per_repo: int = typer.Option(
        None, help="Max number of vulnfix functions per repo."
    ),
    max_non_vuln_per_repo: int = typer.Option(
        None, help="Max number of non-vulnfix functions per repo."
    ),
    batch_size: int = 64,
    assume_all_vulnerable: bool = typer.Option(
        True,
        help="If disabled, will only extract commits with one function modified. This method is used in Devign.",
    ),
    extract_nonvuln_from_vulnfix: bool = typer.Option(
        False,
        help="If enabled, extract all non-modified functions in vulnfix commits as non-vuln. This is how vulnfix-only datasets extract non-vuln functions, such as BigVul.",
    ),
    include_file: bool = typer.Option(
        False, help="If enabled, will include the file code in the output"
    ),
    shuffle: bool = typer.Option(
        False, help="If enabled, will shuffle the commits before extracting functions"
    ),
    seed: int = typer.Option(
        None, help="Seed to use for shuffling. Only used if `--shuffle` is enabled."
    ),
    resume: bool = typer.Option(
        False, help="If enabled, will resume from the last extracted commit"
    ),
    numeric_labels: bool = typer.Option(
        False, help="If enabled, will use numeric labels instead of strings"
    ),
    num_workers: int = typer.Option(
        0, help="Number of workers to use for extracting functions"
    ),
):
    """
    Extract functions from classified commits.

    **Must be run on `classify` output.**
    """
    if not assume_all_vulnerable and extract_nonvuln_from_vulnfix:
        print(
            "[!] Cannot use `--extract-nonvuln-from-vulnfix` with `--not-assume-all-vulnerable`"
        )
        raise typer.Exit(code=1)

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
            "[red]Extracting Functions from Commits...", total=len(df), unit="commits"
        )
        extracted_commits_count = counter.add_task(
            f"[+] Successfully extracted from commits",
            total=len(df),
            unit="commits",
        )
        failed_extract_commits_count = counter.add_task(
            f"[-] Failed to extract commits",
            total=len(df),
            visible=False,
            unit="commits",
        )
        outside_threshold_count = counter.add_task(
            f"[-] Ignored commits outside threshold",
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
            total=0 if max_vuln_per_repo is None else max_vuln_per_repo * num_repos,
            visible=False,
            unit="functions",
        )
        extracted_non_vuln_functions_count = counter.add_task(
            f"[+] Extracted functions (non-vuln)",
            total=0
            if max_non_vuln_per_repo is None
            else max_non_vuln_per_repo * num_repos,
            visible=False,
            unit="functions",
        )
        failed_extract_functions_count = counter.add_task(
            f"[-] Failed to extract functions",
            total=0,
            visible=False,
            unit="functions",
        )
        failed_extracting_function_file_source_code = counter.add_task(
            f"[-] Failed extracting function's file source code",
            total=0,
            visible=False,
            unit="functions",
        )
        failed_extracting_function_source_code = counter.add_task(
            f"[-] Failed extracting function source code",
            total=0,
            visible=False,
            unit="functions",
        )
        ignored_anonymous_functions_count = counter.add_task(
            f"[-] Ignored anonymous functions",
            total=0,
            visible=False,
            unit="functions",
        )
        commits_grouped_by_repo = df.groupby("repo_url")
        dfs = [group for _, group in commits_grouped_by_repo]

        header = True
        for df in dfs:
            repo_url = df.iloc[0]["repo_url"]
            git_repo = repos[repo_url]
            pydriller_repo = Repository(
                git_repo.working_dir,
                only_commits=df["commit_hash"].tolist(),
                num_workers=num_workers,
            )

            batch = pd.DataFrame()
            for i, commit in enumerate(pydriller_repo.traverse_commits()):
                label = ""
                preds = [df.iloc[i]["non-vulnfix"], df.iloc[i]["vulnfix"]]
                if preds[1] >= vulnfix_threshold:
                    label = "vuln"
                elif preds[0] >= non_vulnfix_threshold:
                    label = "non-vuln"
                else:
                    counter.update(outside_threshold_count, advance=1, visible=True)
                    summary_progress.update(
                        extraction_task,
                        advance=1,
                    )
                    continue

                repo_name = os.path.basename(git_repo.working_dir)
                num_vuln_functions_extracted = table_data[repo_name]["vuln"]
                num_non_vuln_functions_extracted = table_data[repo_name]["non-vuln"]

                if label == "vuln":
                    if extract_nonvuln_from_vulnfix:
                        if (max_vuln_per_repo is not None and num_vuln_functions_extracted >= max_vuln_per_repo) and (
                            max_non_vuln_per_repo is not None and num_non_vuln_functions_extracted >= max_non_vuln_per_repo
                        ):
                            counter.update(
                                ignored_commits_from_max_non_vuln_per_repo_count,
                                advance=1,
                                visible=True,
                            )
                            summary_progress.update(
                                extraction_task,
                                advance=1,
                            )
                            continue
                    elif max_vuln_per_repo is not None and num_vuln_functions_extracted >= max_vuln_per_repo:
                        counter.update(
                            ignored_commits_from_max_vuln_per_repo_count,
                            advance=1,
                            visible=True,
                        )
                        summary_progress.update(
                            extraction_task,
                            advance=1,
                        )
                        continue

                if (
                    max_non_vuln_per_repo is not None
                    and label == "non-vuln"
                    and num_non_vuln_functions_extracted >= max_non_vuln_per_repo
                ):
                    counter.update(
                        ignored_commits_from_max_non_vuln_per_repo_count,
                        advance=1,
                        visible=True,
                    )
                    summary_progress.update(extraction_task, advance=1)
                    continue

                if commit.merge:
                    counter.update(
                        failed_extract_commits_count, advance=1, visible=True
                    )
                    summary_progress.update(extraction_task, advance=1)
                    continue

                functions: list[tuple[Method, int]] = []
                for f in commit.modified_files:
                    ext = os.path.splitext(f.filename)[1]
                    if ext not in [".c", ".cpp"]:
                        continue

                    (
                        modified_functions_after,
                        modified_functions_before,
                    ) = changed_methods_both(f)

                    if label == "vuln":
                        functions.extend(
                            [(function, 1) for function in modified_functions_before]
                        )

                        if (
                            extract_nonvuln_from_vulnfix
                            and len(modified_functions_before) > 0
                        ):
                            unmodified_functions_before = [
                                function
                                for function in f.methods_before
                                if function not in modified_functions_before
                            ]
                            functions.extend(
                                [
                                    (function, 0)
                                    for function in unmodified_functions_before
                                ]
                            )
                    elif label == "non-vuln":
                        functions.extend(
                            [(function, 0) for function in modified_functions_after]
                        )

                if not assume_all_vulnerable and len(functions) > 1:
                    counter.update(
                        more_than_one_function_count, advance=1, visible=True
                    )
                    summary_progress.update(extraction_task, advance=1)
                    continue

                if len(functions) == 0:
                    counter.update(no_functions_found_count, advance=1, visible=True)
                    summary_progress.update(extraction_task, advance=1)
                    continue

                for function, _label in functions:
                    num_vuln_functions_extracted = table_data[repo_name]["vuln"]
                    if (
                        _label == 1
                        and max_vuln_per_repo is not None
                        and num_vuln_functions_extracted + 1 > max_vuln_per_repo
                    ):
                        continue

                    num_non_vuln_functions_extracted = table_data[repo_name]["non-vuln"]
                    if (
                        _label == 0
                        and max_non_vuln_per_repo is not None
                        and num_non_vuln_functions_extracted + 1 > max_non_vuln_per_repo
                    ):
                        continue

                    try:
                        source_code = (
                            f.source_code if _label == 0 else f.source_code_before
                        )
                    except:
                        source_code = None

                    if source_code is None:
                        counter.update(
                            failed_extracting_function_file_source_code,
                            advance=1,
                            visible=True,
                        )
                        continue

                    if function.name == "(anonymous)":
                        counter.update(
                            ignored_anonymous_functions_count,
                            advance=1,
                            visible=True,
                        )
                        continue

                    start_line = function.start_line
                    end_line = function.end_line
                    function_code = get_method_code(source_code, start_line, end_line)

                    if function_code is None:
                        counter.update(
                            failed_extracting_function_source_code,
                            advance=1,
                            visible=True,
                        )
                        continue

                    commit_data = df.iloc[i]
                    path = f.old_path if _label == 1 else f.new_path
                    commit_url = commit_data["commit_url"]
                    function_url = f"{repo_url}/blob/{commit.hash}/{path}#L{start_line}-L{end_line}"

                    batch = pd.concat(
                        [
                            batch,
                            pd.DataFrame(
                                [
                                    {
                                        "labels": _label
                                        if numeric_labels
                                        else ("non-vuln", "vuln")[_label],
                                        "preds": preds,
                                        "repo_url": repo_url,
                                        "commit_url": commit_url,
                                        "function_url": function_url,
                                        "function": function_code,
                                        "commit_msg": commit.msg,
                                        "commit_hash": commit.hash,
                                        "date": commit.author_date,
                                        "name": function.name,
                                        "symbol": function.long_name,
                                        "parameters": function.parameters,
                                        "start_line": start_line,
                                        "end_line": end_line,
                                        "filename": f.filename,
                                        "path": path,
                                        "non_vuln_from_vulnfix": _label == 0
                                        and label == "vuln",
                                    }
                                ]
                            ),
                        ]
                    )

                    counter.update(
                        extracted_vuln_functions_count
                        if _label == 1
                        else extracted_non_vuln_functions_count,
                        advance=1,
                        visible=True,
                    )
                    table_data[repo_name][("non-vuln", "vuln")[_label]] += 1

                    is_batch_full = len(batch) >= batch_size
                    is_last_batch_occupied = i + 1 == len(df) and len(batch) > 0
                    if is_batch_full or is_last_batch_occupied:
                        batch.to_csv(
                            output,
                            mode="a",
                            index=False,
                            header=header,
                        )

                        header = False
                        batch = pd.DataFrame()

                counter.update(extracted_commits_count, advance=1, visible=True)
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
