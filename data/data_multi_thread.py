import argparse, torch
import json
import os
import shutil
from pathlib import Path
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from huggingface_hub import hf_hub_download
from tqdm import tqdm
import threading

REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
REMOTE_ROOT_PREFIX = os.environ.get("MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", "datasets")
ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
TOKENIZERS_DIR = ROOT / "tokenizers"

# Global lock for thread-safe progress updates
progress_lock = threading.Lock()

def dataset_dir_for_variant(name: str) -> str:
    if name == "byte260":
        return "fineweb10B_byte260"
    if name.startswith("sp") and name[2:].isdigit():
        return f"fineweb10B_{name}"
    raise ValueError(f"unsupported variant {name!r}; expected byte260 or sp<VOCAB_SIZE>")

def local_path_for_remote(relative_path: str) -> Path:
    remote_path = Path(relative_path)
    if REMOTE_ROOT_PREFIX and remote_path.parts[:1] == (REMOTE_ROOT_PREFIX,):
        remote_path = remote_path.relative_to(REMOTE_ROOT_PREFIX)
    if remote_path.parts[:1] == ("datasets",):
        return DATASETS_DIR.joinpath(*remote_path.parts[1:])
    if remote_path.parts[:1] == ("tokenizers",):
        return TOKENIZERS_DIR.joinpath(*remote_path.parts[1:])
    return ROOT / remote_path

def download_single_file(remote_path: str, pbar: tqdm) -> None:
    """
    Download single file with thread-safe progress update.
    """
    destination = local_path_for_remote(remote_path)
    
    # Skip if exists
    if destination.exists():
        pbar.update(1)
        return
    
    if destination.is_symlink():
        destination.unlink()
    
    try:
        # HF cache path
        cached_path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=remote_path,
                subfolder=Path(remote_path).parent.as_posix() if Path(remote_path).parent != Path(".") else None,
                repo_type="dataset",
            )
        )
        cached_source = cached_path.resolve(strict=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Hard link if possible (fastest), else copy
        try:
            os.link(cached_source, destination)
        except OSError:
            shutil.copy2(cached_source, destination)
        
        with progress_lock:
            pbar.set_postfix({"Last": Path(remote_path).name})
            pbar.update(1)
            
    except Exception as e:
        with progress_lock:
            pbar.set_postfix({"Error": f"{remote_path}: {str(e)[:30]}..."})

def download_parallel(files: List[str], max_workers: int = 16, desc: str = "Downloading") -> None:
    """
    Download files in parallel with progress bar.
    """
    with tqdm(total=len(files), desc=desc, unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(download_single_file, f, pbar) for f in files]
            concurrent.futures.wait(futures)

def manifest_path() -> Path:
    return local_path_for_remote(f"{REMOTE_ROOT_PREFIX}/manifest.json")

def load_manifest(*, skip_manifest_download: bool) -> dict:
    path = manifest_path()
    if not path.is_file():
        if skip_manifest_download:
            raise FileNotFoundError(
                f"manifest.json is required but not present at {path}"
            )
        # Download manifest first (single-threaded)
        download_single_file(f"{REMOTE_ROOT_PREFIX}/manifest.json", tqdm(total=1, desc="Manifest"))
    return json.loads(path.read_text(encoding="utf-8"))

def artifact_paths_for_tokenizer(tokenizer_entry: dict) -> List[str]:
    artifacts = []
    for key in ("model_path", "vocab_path", "path"):
        value = tokenizer_entry.get(key)
        if value:
            artifacts.append(str(value))
    if not artifacts:
        raise ValueError(f"tokenizer entry missing artifacts: {tokenizer_entry}")
    return artifacts

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="🚀 Multi-threaded FineWeb dataset download")
    parser.add_argument(
        "train_shards_positional",
        nargs="?",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--train-shards",
        type=int,
        default=80,
        help="Number of training shards to download (default: 80).",
    )
    parser.add_argument(
        "--variant",
        default="sp1024",
        help="Tokenizer: sp1024, sp4096, byte260, etc. (default: sp1024)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Max parallel downloads (default: 16). RTX 3090: 32, RTX 4060: 16, RTX 3050: 8.",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip manifest.json download.",
    )
    parser.add_argument(
        "--with-docs",
        action="store_true",
        help="Download docs_selected.jsonl + source manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist.",
    )
    return parser

def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = dataset_dir_for_variant(args.variant)
    train_shards = (args.train_shards_positional 
                   if args.train_shards_positional is not None 
                   else args.train_shards)
    
    if train_shards < 0:
        raise ValueError("train_shards must be non-negative")

    print(f"🚀 Multi-threaded download: {args.variant}")
    print(f"📊 Train shards: {train_shards}")
    print(f"⚡ Max workers:   {args.max_workers}")
    print(f"💾 Dataset dir:  {DATASETS_DIR / dataset_dir_for_variant(args.variant)}")

    manifest = load_manifest(skip_manifest_download=args.skip_manifest)
    dataset_entry = next((x for x in manifest.get("datasets", []) 
                         if x.get("name") == dataset_dir), None)
    if dataset_entry is None:
        raise ValueError(f"Dataset {dataset_dir} not in manifest")

    max_train = int((dataset_entry.get("stats") or {}).get("files_train", 0))
    val_shards = int((dataset_entry.get("stats") or {}).get("files_val", 0))
    
    if train_shards > max_train:
        print(f"⚠️  Warning: {args.variant} only has {max_train} train shards, requested {train_shards}")
        train_shards = max_train

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = next((x for x in manifest.get("tokenizers", []) 
                           if x.get("name") == tokenizer_name), None)
    if tokenizer_entry is None:
        raise ValueError(f"Tokenizer {tokenizer_name} not in manifest")

    # ── Build download list ──────────────────────────────────────────────────
    files_to_download = []

    if args.with_docs:
        files_to_download.extend([
            f"{REMOTE_ROOT_PREFIX}/docs_selected.jsonl",
            f"{REMOTE_ROOT_PREFIX}/docs_selected.source_manifest.json"
        ])

    # Dataset shards
    dataset_prefix = f"{REMOTE_ROOT_PREFIX}/datasets/{dataset_dir}"
    for i in range(val_shards):
        files_to_download.append(f"{dataset_prefix}/fineweb_val_{i:06d}.bin")
    for i in range(train_shards):
        files_to_download.append(f"{dataset_prefix}/fineweb_train_{i:06d}.bin")

    # Tokenizer artifacts
    tokenizer_artifacts = artifact_paths_for_tokenizer(tokenizer_entry)
    for art in tokenizer_artifacts:
        files_to_download.append(f"{REMOTE_ROOT_PREFIX}/{art}")

    print(f"📥 {len(files_to_download)} files to download "
          f"({train_shards} train + {val_shards} val + {len(tokenizer_artifacts)} tokenizer)")

    if args.dry_run:
        print("\n📋 DRY RUN — files that WOULD be downloaded:")
        for f in files_to_download:
            print(f"  📄 {f}")
        return

    # ── Parallel download ────────────────────────────────────────────────────
    print(f"\n⚡ Starting {args.max_workers}-threaded download...")
    
    # Auto-tune workers based on GPU if detected
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()[0]
        if cap >= 8:  # Ampere+ (3090, 4060)
            args.max_workers = min(args.max_workers, 32)
        elif cap == 7:  # Turing (RTX 20xx)
            args.max_workers = min(args.max_workers, 16)
        else:
            args.max_workers = min(args.max_workers, 8)
        print(f"🎯 Auto-tuned to {args.max_workers} workers (GPU SM{cap}.x)")

    # Filter existing files
    existing = []
    for f in files_to_download:
        dest = local_path_for_remote(f)
        if dest.exists() and not args.force:
            existing.append(f)
        else:
            download_single_file(f, tqdm(total=1, desc="Checking"))  # preload cache
    files_to_download = [f for f in files_to_download if f not in existing]

    if not files_to_download:
        print("✅ All files already exist — nothing to download!")
        return

    print(f"\n📥 Downloading {len(files_to_download)} new files...")
    download_parallel(files_to_download, max_workers=args.max_workers, 
                      desc=f"🚀 {args.variant}")

    print(f"\n🎉 Download complete!")
    print(f"📂 Train: {DATASETS_DIR / dataset_dir_for_variant(args.variant)}/fineweb_train_*.bin")
    print(f"📂 Val:   {DATASETS_DIR / dataset_dir_for_variant(args.variant)}/fineweb_val_*.bin")
    print(f"🔤 Tokenizer: {TOKENIZERS_DIR}")

if __name__ == "__main__":
    main()
