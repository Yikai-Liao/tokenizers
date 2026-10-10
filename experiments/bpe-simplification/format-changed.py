#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["tree-sitter==0.25.2", "tree-sitter-rust==0.24.2"]
# ///
"""Temporary BPE formatter: rustfmt plus blank lines between sibling functions.

Only changed Rust files in the BPE directory are eligible. --base defaults to
HEAD (staged/unstaged changes); use a PR base to include committed changes.
Optional paths select a subset of those files. --check never writes files.
"""

import argparse
from pathlib import Path
import shutil
import subprocess

from tree_sitter import Language, Parser
import tree_sitter_rust


SCOPE = "tokenizers/tk-train/src/trainers/bpe"


def separate_functions(source: bytes) -> bytes:
    parser = Parser(Language(tree_sitter_rust.language()))
    tree = parser.parse(source)
    if tree.root_node.has_error:
        raise ValueError("Rust syntax could not be parsed; refusing to rewrite it")
    lines = source.splitlines(keepends=True)
    insertions = set()
    pending = [tree.root_node]
    while pending:
        parent = pending.pop()
        children = parent.named_children
        pending.extend(children)
        for index, node in enumerate(children):
            if node.type not in {"function_item", "function_signature_item"}:
                continue
            # Attributes and comments belong with the following function. AST
            # nodes prevent function-looking text inside raw strings from edits.
            first = index
            while first > 0 and children[first - 1].type in {
                "attribute_item", "line_comment", "block_comment"
            }:
                first -= 1
            if first == 0 or children[first - 1].type not in {"function_item", "function_signature_item"}:
                continue
            previous = children[first - 1]
            start = children[first].start_point.row
            for comment in children[first:index]:
                if comment.start_point.row == previous.end_point.row:
                    start = comment.end_point.row + 1
            gap = lines[previous.end_point.row + 1:start]
            if not any(not line.strip() for line in gap):
                insertions.add(start)
    for row in sorted(insertions, reverse=True):
        lines.insert(row, b"\n")
    return b"".join(lines)


def main() -> int:
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--base", default="HEAD")
    arguments.add_argument("--check", action="store_true")
    arguments.add_argument("paths", nargs="*")
    args = arguments.parse_args()
    root = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip())

    def git_paths(*command: str) -> set[str]:
        output = subprocess.check_output(["git", *command], cwd=root)
        return {entry.decode() for entry in output.split(b"\0") if entry}

    changed = git_paths("diff", "--name-only", "--diff-filter=ACMR", "-z", args.base, "--", SCOPE)
    changed |= git_paths("ls-files", "--others", "--exclude-standard", "-z", "--", SCOPE)
    selected = {name for name in changed if name.endswith(".rs")}
    if args.paths:
        requested = {str(Path(name).resolve().relative_to(root)) for name in args.paths}
        selected &= requested
    rustfmt = shutil.which("rustfmt") or str(Path.home() / ".cargo/bin/rustfmt")
    needs_format = False
    for name in sorted(selected):
        path = root / name
        original = path.read_bytes()
        result = subprocess.run(
            [rustfmt, "--edition", "2024", "--config", "skip_children=true", "--emit", "stdout"],
            input=original, stdout=subprocess.PIPE, check=True, cwd=path.parent,
        )
        formatted = separate_functions(result.stdout)
        if formatted != original:
            needs_format = True
            if not args.check:
                path.write_bytes(formatted)
            print(f"{'Needs formatting' if args.check else 'Formatted'}: {name}")
    if not selected:
        print("No changed BPE Rust files selected.")
    return int(args.check and needs_format)


if __name__ == "__main__":
    raise SystemExit(main())
