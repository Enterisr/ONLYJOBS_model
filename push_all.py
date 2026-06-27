"""
Commits and pushes both ONLYJOBS and linkedin-job-filter to GitHub.
Run with VS Code's ▶ button (Run Python File).
"""
import subprocess, os, sys
from pathlib import Path

def run(cmd, cwd):
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.stdout.strip(): print(r.stdout.strip())
    if r.stderr.strip(): print(r.stderr.strip())
    return r.returncode

ONLYJOBS     = Path(r"C:\Users\drunk\ONLYJOBS")
EXT          = Path(r"C:\Users\drunk\linkedin-job-filter")
GH_USER      = "Enterisr"

# ── Remove stale git lock files ───────────────────────────────────────────────
for repo in [ONLYJOBS, EXT]:
    lock = repo / ".git" / "index.lock"
    if lock.exists():
        try:
            lock.unlink()
            print(f"Removed stale lock: {lock}")
        except Exception as e:
            print(f"Warning: could not remove {lock}: {e}")

# ── ONLYJOBS ──────────────────────────────────────────────────────────────────
print("\n=== ONLYJOBS ===")
run(["git", "config", "user.email", "or.p.israeli@gmail.com"], ONLYJOBS)
run(["git", "config", "user.name",  "Or Israeli"],              ONLYJOBS)
run(["git", "add", "-A"],                                        ONLYJOBS)
run(["git", "commit", "-m", "Add SupCon model, evaluate.py, gitignore; update train.py"], ONLYJOBS)
run(["git", "push", "origin", "main"],                          ONLYJOBS)

# ── linkedin-job-filter ───────────────────────────────────────────────────────
print("\n=== linkedin-job-filter ===")
git_dir = EXT / ".git"
if not git_dir.exists():
    print("Initialising git repo...")
    run(["git", "init", "-b", "main"], EXT)

run(["git", "config", "user.email", "or.p.israeli@gmail.com"], EXT)
run(["git", "config", "user.name",  "Or Israeli"],              EXT)
run(["git", "add", "-A"],                                        EXT)
run(["git", "commit", "-m", "Initial commit: LinkedIn Job Filter Chrome extension"], EXT)

# Check if remote exists
r = subprocess.run(["git", "remote"], cwd=EXT, capture_output=True, text=True)
if "origin" not in r.stdout:
    remote = f"https://github.com/{GH_USER}/linkedin-job-filter.git"
    print(f"Adding remote: {remote}")
    run(["git", "remote", "add", "origin", remote], EXT)

run(["git", "push", "-u", "origin", "main"], EXT)

print("\n=== Done ===")
