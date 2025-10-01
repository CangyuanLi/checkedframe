import argparse
import json
import shutil
import subprocess
import sys
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


if __name__ == "__main__":
    latest = False
    commit = True
    push = True
    pages_branch = "gh-pages"

    repo_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    repo_root = Path(repo_root)
    docs_version_path = repo_root / "docs" / ".docs-version"
    switcher_path = repo_root / "docs" / "_static" / "switcher.json"

    version = sys.argv[1] if len(sys.argv) > 1 else "main"

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
            subprocess.run(
                ["git", "commit", switcher_path, "-am", "docs: update switcher.json"],
                check=True,
            )

        if push:
            subprocess.run(["git", "push"], check=True)

    print("building docs")
    subprocess.run(
        ["sphinx-build", repo_root / "docs", repo_root / "docs" / "_build"],
        check=True,
    )

    print(f"copying docs to {pages_branch}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        try:
            if not branch_exists(pages_branch):
                subprocess.run(["git", "checkout", "-b", pages_branch])

            if commit:
                subprocess.run(
                    ["git", "worktree", "add", tmpdir_path, pages_branch], check=True
                )

            version_path = tmpdir_path / version

            if version_path.exists():
                shutil.rmtree(version_path)

            shutil.copytree(repo_root / "docs" / "_build", version_path)

            if latest:
                latest_path = tmpdir_path / "latest"
                if latest_path.exists():
                    shutil.rmtree(latest_path)

                shutil.copytree(repo_root / "docs" / "_build", latest_path)
                subprocess.run(["git", "add", latest_path], check=True)

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
                subprocess.run(["git", "add", index_path], cwd=tmpdir_path, check=True)
                subprocess.run(
                    ["git", "add", version_path], cwd=tmpdir_path, check=True
                )
                subprocess.run(
                    ["git", "commit", "-m", version], cwd=tmpdir_path, check=True
                )

            if push:
                print(f"pushing to {pages_branch}")
                subprocess.run(
                    ["git", "push", "-u", "origin", pages_branch],
                    cwd=tmpdir_path,
                    check=True,
                )
        finally:
            print("removing git worktree")
            subprocess.run(
                ["git", "worktree", "remove", "--force", tmpdir_path], check=True
            )
