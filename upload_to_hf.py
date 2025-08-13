#!/usr/bin/env python3
"""
Upload local model files (gguf + safetensors + tokenizer) to Hugging Face model repo.

Usage:
  export HF_TOKEN="hf_xxx..."
  python upload_to_hf.py --local-dir ./mrkswe/phi4-model02-endpoint --repo-id mrkswe/phi4-model02-endpoint --use-git-lfs
"""
import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, Repository
from huggingface_hub.utils import RepositoryNotFoundError

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local-dir", "-d", required=True, help="Local directory containing files to upload")
    p.add_argument("--repo-id", "-r", required=True, help="Hugging Face repo id, e.g. username/reponame")
    p.add_argument("--token", "-t", default=None, help="Hugging Face token (optional, can use HF_TOKEN env)")
    p.add_argument("--use-git-lfs", action="store_true", help="If set, perform git-lfs push via huggingface_hub.Repository (recommended for very large files)")
    return p.parse_args()

def main():
    args = parse_args()
    local_dir = Path(args.local_dir)
    if not local_dir.exists() or not local_dir.is_dir():
        print("Local dir not found:", local_dir, file=sys.stderr)
        sys.exit(2)

    token = (args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    if not token:
        print("ERROR: No HF token provided. Set HF_TOKEN env var or pass --token.", file=sys.stderr)
        sys.exit(2)
    token = token.strip()

    api = HfApi()
    # Validate token
    try:
        who = api.whoami(token=token)
        print(f"Authenticated as: {who.get('name') or who.get('user', {}).get('name')}")
    except Exception as e:
        print("ERROR: token validation failed:", e, file=sys.stderr)
        sys.exit(2)

    # Ensure repo exists (create if not)
    try:
        api.repo_info(repo_id=args.repo_id, token=token)
        print(f"Repo exists: {args.repo_id}")
    except Exception as e:
        print(f"Repo '{args.repo_id}' not found or cannot access. Trying to create it...")
        try:
            api.create_repo(repo_id=args.repo_id, token=token, exist_ok=True, private=False)
            print("Repo created (or already exists).")
        except Exception as e2:
            print("ERROR: Could not create repo:", e2, file=sys.stderr)
            sys.exit(2)

    # Collect files to upload (skip hidden)
    files = [p for p in sorted(local_dir.iterdir()) if p.is_file() and not p.name.startswith(".")]
    if not files:
        print("No files found to upload in", local_dir)
        return

    # If user requested git-lfs push for large files, use Repository helper
    # NOTE: argparse turns --use-git-lfs into args.use_git_lfs
    if getattr(args, "use_git_lfs", False):
        print("Using git-lfs Repository method (recommended for large files). This requires 'git' and 'git-lfs' available.")
        try:
            repo_local_dir = Path("./hf_temp_repo")
            if repo_local_dir.exists():
                import shutil
                shutil.rmtree(repo_local_dir)
            # Clone the remote repo into a temp folder
            repo = Repository(local_dir=repo_local_dir, clone_from=args.repo_id, use_auth_token=token)
            # copy files into repo_local_dir
            import shutil
            for f in files:
                dest = repo_local_dir / f.name
                print("Copying", f, "->", dest)
                shutil.copyfile(f, dest)
            # commit & push
            repo.git_add(pattern="*")
            repo.git_commit("Upload gguf + safetensors from local export")
            print("Pushing to HF (this may take long for large files)...")
            repo.git_push()
            print("Push complete.")
            return
        except Exception as e:
            print("Git-lfs push failed, falling back to API upload. Error:", e, file=sys.stderr)

    # Fallback: upload files one-by-one via API
    for f in files:
        try:
            print(f"Uploading {f.name} ...")
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f.name,
                repo_id=args.repo_id,
                token=token,
                commit_message=f"Upload {f.name}"
            )
            print(" Uploaded:", f.name)
        except Exception as e:
            print(f"FAILED to upload {f.name}: {e}", file=sys.stderr)
            if "401" in str(e) or "Unauthorized" in str(e):
                print("Authentication / permission error. Check token scopes (write permission) and repo ownership.", file=sys.stderr)
                sys.exit(1)
    print("All done (attempted uploads).")

if __name__ == "__main__":
    main()
