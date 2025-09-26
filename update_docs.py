import json
import shutil
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    repo_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    repo_root = Path(repo_root)

    original_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    version = sys.argv[1] if len(sys.argv) > 1 else "main"

    with open(repo_root / "docs" / "_static" / "switcher.json") as f:
        switcher: list[dict] = json.load(f)

    switcher_dict = {dct["version"]: dct for dct in switcher}

    switcher_dict[version] = {
        "name": version,
        "version": version,
        "url": f"https://cangyuanli.github.io/checkedframe/{version}",
    }

    main_is_only_version = switcher_dict.get("main", False) and len(switcher_dict) == 1

    if version != "main" or main_is_only_version:
        for v in switcher_dict.values():
            # set the currently built version to true, and everything else to false
            if v["version"] == version:
                v["preferred"] = True
            else:
                v["preferred"] = False

    with open(repo_root / "docs" / "_static" / "switcher.json", "w") as f:
        json.dump(list(switcher_dict.values()), f, indent=4)

    print("building docs")
    subprocess.run(
        ["sphinx-build", repo_root / "docs", repo_root / "docs" / "_build"],
        check=True,
    )

    print(f"copying docs to {version}")
    subprocess.run(["git", "stash", "push", "-m", "WIP before gh-pages"], check=True)

    try:
        subprocess.run(["git", "checkout", "gh-pages"], check=True)
    except subprocess.CalledProcessError:
        subprocess.run(["git", "checkout", "-b", "gh-pages"], check=True)

    if (repo_root / version).exists():
        shutil.rmtree(repo_root / version)

    shutil.copytree(repo_root / "docs" / "_build", repo_root / version)

    if version != "main" or main_is_only_version:
        if (repo_root / "latest").exists():
            shutil.rmtree(repo_root / "latest")

        shutil.copytree(repo_root / "docs" / "_build", repo_root / "latest")
        subprocess.run(["git", "add", repo_root / "latest"], check=True)

    print("pushing changes")
    subprocess.run(["git", "add", repo_root / version], check=True)
    subprocess.run(["git", "commit", "-m", version], check=True)
    subprocess.run(["git", "push", "-u", "origin", "gh-pages"], check=True)

    subprocess.run(["git", "checkout", original_branch], check=True)
    subprocess.run(["git", "stash", "pop"], check=True)
