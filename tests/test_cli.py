import subprocess
import sys


def test_initialization_does_not_overwrite_credentials_or_print_them(tmp_path):
    command = [sys.executable, "-m", "hmph.cli", "init"]
    first = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    contents = (tmp_path / ".env").read_text(encoding="utf-8")
    second = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    token = next(
        line.split("=", 1)[1]
        for line in contents.splitlines()
        if line.startswith("HMPH_API_TOKEN=")
    )
    assert second.returncode == 1
    assert (tmp_path / ".env").read_text(encoding="utf-8") == contents
    assert token not in first.stdout + first.stderr + second.stdout + second.stderr
