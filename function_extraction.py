import re

import clang
from git import Diff


def get_hunk_headers_function(diff: Diff):
    # given a diff, read each line containing a hunk header "@@ -a,b +c,d @@ <function>"
    # and return a list of functions
    # if the hunk header has no function, don't include it

    # read the diff
    diff_text = diff.diff.decode("latin-1")
    # split the diff into lines
    diff_lines = diff_text.split("\n")

    # regex to match hunk header
    hunk_header_regex = re.compile(r"^@@ -\d+,\d+ \+\d+,\d+ @@")

    # regex to match function name
    function_name_regex = re.compile(r"@@ -\d+,\d+ \+\d+,\d+ @@ (.+)")

    # list of functions
    functions = []

    # iterate over each line
    for line in diff_lines:
        # if the line matches the hunk header regex
        if hunk_header_regex.match(line):
            # try to match the function name regex
            match = function_name_regex.match(line)
            # if the function name regex matches
            if match:
                # append the function name to the list of functions
                functions.append(match.group(1))

    # return the list of functions
    return functions


def find_function(node, function_name):
    if (
        node.kind == clang.cindex.CursorKind.FUNCTION_DECL
        and node.spelling == function_name
    ):
        return node
    for child in node.get_children():
        result = find_function(child, function_name)
        if result is not None:
            return result
    return None


def get_function_source(file_path, function):
    # Get the starting and ending line numbers of the function
    start_line = function.extent.start.line
    end_line = function.extent.end.line

    # with open(file_path, "r") as file:
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Extract the function's source code
    function_source = "".join(lines[start_line - 1 : end_line])
    return function_source
