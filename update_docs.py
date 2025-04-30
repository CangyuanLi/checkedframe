import subprocess

if __name__ == "__main__":
    print("uploading docs")
    subprocess.run(["sphinx-build", "docs/", "docs/_build/html/"])
    subprocess.run(["ghp-import", "-n", "-p", "-f", "docs/_build/html/"])
