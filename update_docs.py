import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def branch_exists(branch: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", branch],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    return result.returncode == 0


def run(args, **kwargs):
    if "check" not in kwargs:
        kwargs["check"] = True

    if "text" not in kwargs:
        kwargs["text"] = True

    return subprocess.run(args, **kwargs)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "version",
        type=str,
        help="A string specifying the documentation version, e.g. '0.1.0' or 'main'",
    )
    parser.add_argument(
        "-c",
        "--commit",
        action="store_true",
        help="Commit the changes, by default False",
    )
    parser.add_argument(
        "-p",
        "--push",
        action="store_true",
        help="Push the changes to the specified branch. If this is set to true, `-c` or `--commit` is automatically set to true, by default False",
    )
    parser.add_argument(
        "-l",
        "--latest",
        action="store_true",
        help="Indicate the version being built is the latest version, by default False",
    )
    parser.add_argument(
        "-b",
        "--branch",
        type=str,
        default="gh-pages",
        help="The branch to deploy the documentation to, by default gh-pages",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    version = args.version
    latest = args.latest
    commit = args.commit
    push = args.push

    if push:
        commit = True

    pages_branch = args.branch

    repo_root = run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True
    ).stdout.strip()

    repo_root = Path(repo_root)
    docs_version_path = repo_root / "docs" / ".docs-version"
    switcher_path = repo_root / "docs" / "_static" / "switcher.json"

    docs_version_path.write_text(version)

    with open(switcher_path) as f:
        switcher: list[dict] = json.load(f)

    switcher_dict = {dct["version"]: dct for dct in switcher}

    switcher_dict[version] = {
        "name": version,
        "version": version,
        "url": f"https://cangyuanli.github.io/checkedframe/{version}",
        "preferred": False,
    }

    if latest:
        for v in switcher_dict.values():
            if v["version"] == version:
                v["preferred"] = True
            else:
                v["preferred"] = False

    switcher_new = list(switcher_dict.values())

    if switcher != switcher_new:
        with open(switcher_path, "w") as f:
            json.dump(list(switcher_dict.values()), f, indent=4)

        if commit:
            run(
                ["git", "commit", switcher_path, "-am", "docs: update switcher.json"],
            )

        if push:
            run(["git", "push"])

    print("building docs")
    run(["sphinx-build", "-E", repo_root / "docs", repo_root / "docs" / "_build"])

    print(f"copying docs to {pages_branch}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        try:
            if not branch_exists(pages_branch):
                run(["git", "checkout", "-b", pages_branch])

            if commit:
                run(["git", "worktree", "add", tmpdir_path, pages_branch])

            version_path = tmpdir_path / version

            if version_path.exists():
                shutil.rmtree(version_path)

            shutil.copytree(repo_root / "docs" / "_build", version_path)

            if latest:
                latest_path = tmpdir_path / "latest"
                if latest_path.exists():
                    shutil.rmtree(latest_path)

                shutil.copytree(repo_root / "docs" / "_build", latest_path)
                run(["git", "add", latest_path])

            nojekyll = tmpdir_path / ".nojekyll"
            nojekyll.touch(exist_ok=True)

            redirect_html = """<!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Redirecting</title>
                <noscript>
                    <meta http-equiv="refresh" content="1; url=latest/" />
                </noscript>
                <script>
                    window.location.replace(
                        "latest/" + window.location.search + window.location.hash
                    );
                </script>
            </head>
            <body>
                Redirecting to <a href="latest/">latest/</a>...
            </body>
            </html>
            """

            index_path = tmpdir_path / "index.html"

            with open(index_path, "w") as f:
                f.write(redirect_html)

            if commit:
                run(["git", "add", index_path], cwd=tmpdir_path)
                run(["git", "add", version_path], cwd=tmpdir_path)

                run(["git", "commit", "-m", version], cwd=tmpdir_path)

            if push:
                print(f"pushing to {pages_branch}")
                run(["git", "push", "-u", "origin", pages_branch], cwd=tmpdir_path)
        finally:
            print(f"removing git worktree at {tmpdir_path}")
            run(["git", "worktree", "remove", "--force", tmpdir_path])
