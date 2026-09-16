"""上傳 hub_stage/<dir> 到 Hugging Face 模型 repo(token 自 ../env/.env 的 HF_TOKEN 讀取,不印出)。
    python -m p2_anchor.wringer.hub.upload_hf --repo wcamon/Agents-A1-4B-Wringer-Q2.6 --folder hub_stage/wringer-p3b_w2
印 HF_UPLOAD_DONE <url> 或拋錯。
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[3]


def token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    for line in (ROOT.parent / "env/.env").read_text().splitlines():
        if line.startswith(("HF_TOKEN_WRITE=", "HF_TOKEN=")):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no HF_TOKEN")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--message", default="Wringer: container, unpack script, bf16 materialization, reports")
    ap.add_argument("--delete", action="append", default=[], help="repo 內要刪除的路徑 glob(如舊容器),可多次")
    args = ap.parse_args()
    api = HfApi(token=token())
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo, repo_type="model", folder_path=str(ROOT / args.folder),
                      commit_message=args.message, delete_patterns=args.delete or None)
    print(f"HF_UPLOAD_DONE https://huggingface.co/{args.repo}", flush=True)


if __name__ == "__main__":
    main()
