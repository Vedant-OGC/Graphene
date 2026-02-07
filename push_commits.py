import subprocess
import sys
import time

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing {cmd}:\n{result.stderr}")
        sys.exit(1)
    return result.stdout.strip()

def main():
    print("Fetching commit history...")
    # Get all commit hashes from oldest to newest
    commits_output = run('git log --reverse --format="%H"')
    commits = commits_output.split('\n')
    
    total = len(commits)
    print(f"Found {total} commits. Starting sequential push to 'origin/main'...")

    for i, commit in enumerate(commits, 1):
        print(f"[{i}/{total}] Pushing commit: {commit[:7]}...")
        # Push this specific commit to the remote main branch
        subprocess.run(f"git push -f origin {commit}:refs/heads/main", shell=True)
        # brief pause to be gentle to GitHub APIs
        time.sleep(1)

    print("\n✅ All commits individually pushed!")

if __name__ == "__main__":
    main()
