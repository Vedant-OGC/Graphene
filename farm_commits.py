import os
import subprocess
import random
from datetime import datetime, timedelta

# Configuration
REPO_DIR = r"d:\Graphene"
START_DATE_DAYS_AGO = 45 # How far back to start committing
MAX_COMMITS_PER_DAY = 3

# Files and folders to ignore
IGNORE_DIRS = {".git", "__pycache__", "venv", ".venv", "env", "node_modules", ".pytest_cache", ".idea", ".vscode"}
IGNORE_EXTS = {".pyc", ".pyo", ".pyd", ".log", ".model", ".pt", ".exe", ".zip", ".tar", ".gz"}

def run_cmd(cmd, env=None):
    result = subprocess.run(cmd, cwd=REPO_DIR, env=env, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}\n{result.stderr}")
    return result

def init_git():
    if not os.path.exists(os.path.join(REPO_DIR, ".git")):
        print("Initializing git repository...")
        run_cmd("git init")
    else:
        print("Git repository already exists.")

def get_files_to_commit():
    files_to_commit = []
    for root, dirs, files in os.walk(REPO_DIR):
        # modify dirs in place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in IGNORE_EXTS:
                continue
                
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, REPO_DIR)
            files_to_commit.append(rel_path)
            
    # Sort files to somewhat simulate project growth (framework first, models next, etc.)
    # Basic sort: put README, .env, basic files first, then deep nested files
    files_to_commit.sort(key=lambda x: (x.count(os.sep), x))
    return files_to_commit

def commit_file(file_path, current_date):
    # Formulate a commit message based on the file name/path
    filename = os.path.basename(file_path)
    directory = os.path.dirname(file_path)
    
    if directory:
        msg = f"Add {filename} to {directory} module"
    elif filename.lower() == "readme.md":
        msg = "Initial project setup and documentation"
    else:
        msg = f"Create {filename}"
        
    print(f"Committing {file_path} on {current_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Add file
    # Put quotes around file_path to handle spaces
    run_cmd(f'git add "{file_path}"')
    
    # 2. Assign dates to environment
    env = os.environ.copy()
    date_str = current_date.isoformat()
    env["GIT_AUTHOR_DATE"] = date_str
    env["GIT_COMMITTER_DATE"] = date_str
    
    # 3. Commit
    run_cmd(f'git commit -m "{msg}"', env=env)

def main():
    init_git()
    
    # Get all files
    files = get_files_to_commit()
    print(f"Found {len(files)} files to commit.")
    
    if not files:
        print("No files to commit. Exiting.")
        return
        
    start_date = datetime.now() - timedelta(days=START_DATE_DAYS_AGO)
    
    # Calculate how to spread the commits
    # We want to use roughly START_DATE_DAYS_AGO days, with 1 to MAX_COMMITS_PER_DAY a day
    
    current_date = start_date
    current_day_commits = 0
    max_commits_today = random.randint(1, MAX_COMMITS_PER_DAY)
    
    for file in files:
        # Move to the next day if we've hit our random limit for the day
        if current_day_commits >= max_commits_today:
            # Shift 1 or 2 days forward
            days_to_advance = random.randint(1, 2)
            current_date += timedelta(days=days_to_advance)
            current_day_commits = 0
            max_commits_today = random.randint(1, MAX_COMMITS_PER_DAY)
            
            # Keep the time of day relatively realistic (between 10 AM and 11 PM)
            hour = random.randint(10, 23)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            current_date = current_date.replace(hour=hour, minute=minute, second=second)
            
        commit_file(file, current_date)
        current_day_commits += 1
        
        # Advance slightly within the same day
        current_date += timedelta(minutes=random.randint(5, 120))
        
    print("\nCommit farming complete! You now have a rich git history.")
    print("Next steps:")
    print("1. Create an empty repository on GitHub.")
    print("2. Run: git remote add origin <your-repo-url>")
    print("3. Run: git branch -M main")
    print("4. Run: git push -u origin main")

if __name__ == "__main__":
    main()
