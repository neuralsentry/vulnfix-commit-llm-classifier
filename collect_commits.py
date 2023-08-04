from pydriller.domain.commit import Method


def get_method_code(source_code, start_line, end_line):
    try:
        if source_code is not None:
            code = "\n".join(
                source_code.split("\n")[int(start_line) - 1 : int(end_line)]
            )
            return code
        else:
            return None
    except Exception as e:
        return None


def changed_methods_both(file) -> tuple[list[Method], list[Method]]:
    """
    Return the list of methods that were changed.
    :return: list of methods
    """
    new_methods = file.methods
    old_methods = file.methods_before
    added = file.diff_parsed["added"]
    deleted = file.diff_parsed["deleted"]

    methods_changed_new = {
        y for x in added for y in new_methods if y.start_line <= x[0] <= y.end_line
    }
    methods_changed_old = {
        y for x in deleted for y in old_methods if y.start_line <= x[0] <= y.end_line
    }
    return methods_changed_new, methods_changed_old
