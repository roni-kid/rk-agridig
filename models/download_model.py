
"""
download_model.py — RK AgriDig Phase 1, Task 1.1

Downloads the Phi-3-mini-4k-instruct GGUF (Q4_K_M quantization) from the
official Microsoft Hugging Face repo, verifies its integrity, and reports
timing / size stats.

Model card reference (microsoft/Phi-3-mini-4k-instruct-gguf):
    Phi-3-mini-4k-instruct-q4.gguf | Q4_K_M | ~2.2 GB | "balanced quality - recommended"

Despite the "q4" filename, this file IS the Q4_K_M quant per the model
card's own quant-method column — there is no separately-named
"Q4_K_M"-suffixed file in the official repo (only a much larger fp16
variant also exists, which we deliberately do not download).

Usage:
    python models/download_model.py
    python models/download_model.py --output-dir models --retries 5
    python models/download_model.py --skip-verify   # skip hash check (not recommended)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, HfApi
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print(
        "ERROR: huggingface_hub is not installed.\n"
        "Install it with:  pip install huggingface_hub\n",
        file=sys.stderr,
    )
    sys.exit(1)


REPO_ID = "microsoft/Phi-3-mini-4k-instruct-gguf"
FILENAME = "Phi-3-mini-4k-instruct-q4.gguf"
FINAL_NAME = "phi3_mini_4k_instruct.gguf"  # name expected by rest of the RK AgriDig repo
EXPECTED_SIZE_BYTES_APPROX = 2_200_000_000  # ~2.2 GB per model card; sanity-check only


def human_size(num_bytes: float) -> str:
    """Format bytes as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def human_duration(seconds: float) -> str:
    """Format seconds as a human-readable duration."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def get_remote_sha256(repo_id: str, filename: str) -> str | None:
    """
    Look up the expected SHA256 for a file in an HF repo via the Hub API.

    Returns None if unavailable (e.g. the file isn't tracked by Git LFS/Xet
    with a published sha256, or the API call fails) — the caller should
    treat that as "cannot verify" rather than "verification failed".
    """
    try:
        api = HfApi()
        info = api.model_info(repo_id, files_metadata=True)
        for sibling in info.siblings:
            if sibling.rfilename == filename and sibling.lfs:
                sha256 = sibling.lfs.get("sha256")
                if sha256:
                    return sha256
        return None
    except Exception as exc:  # noqa: BLE001 - we want to degrade gracefully, not crash
        print(f"  (warning: could not fetch remote checksum metadata: {exc})")
        return None


def compute_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute SHA256 of a local file, streaming to avoid loading it all into RAM."""
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_with_retries(
    repo_id: str,
    filename: str,
    local_dir: Path,
    max_retries: int,
) -> Path:
    """
    Download a file from the Hub, retrying on network errors with
    exponential backoff. hf_hub_download already resumes partial
    downloads on retry (it uses the local cache's .incomplete file),
    so retries are cheap after the first attempt.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                wait = min(2 ** (attempt - 1), 30)
                print(f"  Retry {attempt}/{max_retries} in {wait}s...")
                time.sleep(wait)
            path_str = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_dir),
            )
            return Path(path_str)
        except (HfHubHTTPError, ConnectionError, TimeoutError, OSError) as exc:
            last_error = exc
            print(f"  Attempt {attempt}/{max_retries} failed: {exc}")
    raise RuntimeError(
        f"Download failed after {max_retries} attempts. Last error: {last_error}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Phi-3-mini GGUF for RK AgriDig.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to save the model into (default: models/)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Max download attempts on network failure (default: 4)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip SHA256 checksum verification (not recommended)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a verified copy already exists",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / FINAL_NAME

    print("=" * 60)
    print("RK AgriDig — Model Download")
    print("=" * 60)
    print(f"Repo:        {REPO_ID}")
    print(f"File:        {FILENAME}")
    print(f"Quant:       Q4_K_M (balanced quality, ~2.2 GB per model card)")
    print(f"Destination: {final_path}")
    print("=" * 60)

    # --- Idempotency: skip if a good copy is already present ---
    if final_path.exists() and not args.force:
        print(f"\nFound existing file at {final_path} ({human_size(final_path.stat().st_size)}).")
        if args.skip_verify:
            print("Skipping re-download (--skip-verify set, not re-checking hash).")
            print("Use --force to re-download anyway.")
            return 0
        print("Verifying existing file before skipping re-download...")
        remote_sha256 = get_remote_sha256(REPO_ID, FILENAME)
        if remote_sha256:
            local_sha256 = compute_sha256(final_path)
            if local_sha256 == remote_sha256:
                print("✓ Existing file verified — checksum matches. Nothing to do.")
                print(f"  SHA256: {local_sha256}")
                return 0
            else:
                print("✗ Existing file failed checksum verification. Re-downloading...")
        else:
            print("  Could not fetch remote checksum to verify existing file.")
            print("  Use --force to re-download, or --skip-verify to trust it as-is.")
            return 0

    # --- Download ---
    print(f"\nStarting download (up to {args.retries} attempt(s) on failure)...")
    start_time = time.monotonic()
    try:
        downloaded_path = download_with_retries(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=output_dir,
            max_retries=args.retries,
        )
    except RuntimeError as exc:
        print(f"\n✗ FAILED: {exc}", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - start_time

    file_size = downloaded_path.stat().st_size
    avg_speed = file_size / elapsed if elapsed > 0 else 0

    print(f"\n✓ Download complete.")
    print(f"  Size:     {human_size(file_size)}")
    print(f"  Time:     {human_duration(elapsed)}")
    print(f"  Avg rate: {human_size(avg_speed)}/s")

    # Sanity-check size roughly matches what the model card advertises
    if file_size < EXPECTED_SIZE_BYTES_APPROX * 0.8:
        print(
            f"  ⚠ WARNING: file is smaller than expected (~2.2 GB). "
            f"It may be truncated or the wrong file.",
            file=sys.stderr,
        )

    # --- Checksum verification ---
    if args.skip_verify:
        print("\nSkipping checksum verification (--skip-verify).")
    else:
        print("\nVerifying checksum...")
        remote_sha256 = get_remote_sha256(REPO_ID, FILENAME)
        if remote_sha256 is None:
            print(
                "  ⚠ Could not retrieve a remote SHA256 to verify against.\n"
                "    The file was still downloaded via HF Hub's own transfer\n"
                "    integrity checks, but end-to-end checksum verification\n"
                "    was skipped. Re-run with --skip-verify to silence this,\n"
                "    or verify manually against the Hugging Face file page."
            )
        else:
            local_sha256 = compute_sha256(downloaded_path)
            print(f"  Expected: {remote_sha256}")
            print(f"  Actual:   {local_sha256}")
            if local_sha256 != remote_sha256:
                print("\n✗ CHECKSUM MISMATCH — file may be corrupted.", file=sys.stderr)
                print("  Delete the file and re-run this script.", file=sys.stderr)
                return 1
            print("  ✓ Checksum verified.")

    # --- Move/rename into the canonical path expected by the rest of the repo ---
    if downloaded_path != final_path:
        downloaded_path.replace(final_path)
        print(f"\nSaved to: {final_path}")

    print("\n" + "=" * 60)
    print(f"Ready: {final_path} ({human_size(final_path.stat().st_size)})")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())