"""
Download BFCL_v4 dataset from HuggingFace if it doesn't exist.
"""

import argparse
import sys
from pathlib import Path
import os
from huggingface_hub import snapshot_download, hf_hub_download
import json


def download_bfcl_v4(
    data_dir: Path,
    force_download: bool = False,
    token: str = None
) -> Path:
    """
    Download BFCL_v4 dataset from HuggingFace.
    Falls back to BFCL_v3 if v4 is not available.
    
    Args:
        data_dir: Directory to store the dataset
        force_download: Force re-download even if exists
        token: HuggingFace token for authentication
        
    Returns:
        Path to the BFCL directory (v4 or v3)
    """
    bfcl_v4_path = data_dir / "BFCL_v4"
    bfcl_v3_path = data_dir / "BFCL_v3"
    
    # Check if v3 exists and is valid
    if bfcl_v3_path.exists() and not force_download:
        func_doc_path = bfcl_v3_path / "multi_turn_func_doc"
        if func_doc_path.exists() and any(func_doc_path.iterdir()):
            print(f"✅ BFCL_v3 already exists at: {bfcl_v3_path}")
            print(f"   Using existing BFCL_v3 dataset")
            print(f"   Contains {len(list(func_doc_path.iterdir()))} tool definition files")
            
            # Check for multi-turn test files
            multi_turn_files = list(bfcl_v3_path.glob("BFCL_v3_multi_turn_*.json"))
            print(f"   Contains {len(multi_turn_files)} multi-turn test files")
            
            return bfcl_v3_path
    
    # Check if v4 already exists and is valid
    if bfcl_v4_path.exists() and not force_download:
        func_doc_path = bfcl_v4_path / "multi_turn_func_doc"
        if func_doc_path.exists() and any(func_doc_path.iterdir()):
            print(f"✅ BFCL_v4 already exists at: {bfcl_v4_path}")
            print(f"   Contains {len(list(func_doc_path.iterdir()))} tool definition files")
            return bfcl_v4_path
        else:
            print("⚠️ BFCL_v4 exists but appears incomplete.")
    
    # Try to download v4
    print(f"\n{'='*80}")
    print("DOWNLOADING BFCL_v4 FROM HUGGINGFACE")
    print(f"{'='*80}\n")
    
    print(f"Target directory: {bfcl_v4_path}")
    print(f"HuggingFace repo: gorilla-llm/BFCL")
    print("\n⚠️ Note: BFCL_v4 may require HuggingFace authentication.")
    print("   Set HF_TOKEN environment variable or use --token argument.")
    print("   Falling back to BFCL_v3 if download fails...\n")
    
    # Create data directory if needed
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Attempt download with token if available
    if token:
        print("📥 Attempting download with provided token...")
        # Try different possible repo names
        repos_to_try = [
            "gorilla-llm/BFCL",
            "gorilla-llm/berkeley-function-call-leaderboard",
        ]
        
        downloaded = False
        for repo_id in repos_to_try:
            try:
                print(f"\nTrying repo: {repo_id}")
                
                from huggingface_hub import snapshot_download
                # Download the repo
                repo_path = snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    cache_dir=data_dir / "hf_cache",
                    token=token,
                    allow_patterns=["*v4*", "*multi_turn*", "*func_doc*"],
                )
                
                print(f"✅ Downloaded to cache: {repo_path}")
                
                # Find the v4 directory in the downloaded content
                repo_path = Path(repo_path)
                
                # Look for BFCL_v4 or similar
                possible_paths = [
                    repo_path / "BFCL_v4",
                    repo_path / "data" / "BFCL_v4",
                    repo_path / "v4",
                ]
                
                for possible_path in possible_paths:
                    if possible_path.exists():
                        print(f"✅ Found BFCL_v4 at: {possible_path}")
                        
                        # Copy to target location
                        import shutil
                        if bfcl_v4_path.exists():
                            shutil.rmtree(bfcl_v4_path)
                        
                        shutil.copytree(possible_path, bfcl_v4_path)
                        print(f"✅ Copied to: {bfcl_v4_path}")
                        downloaded = True
                        break
                
                if downloaded:
                    return bfcl_v4_path
                    
            except Exception as e:
                print(f"⚠️ Failed with {repo_id}: {e}")
                continue
    
    # If download failed or no token, check for v3
    print(f"\n{'='*80}")
    print("FALLING BACK TO BFCL_v3")
    print(f"{'='*80}\n")
    
    if bfcl_v3_path.exists():
        func_doc_path = bfcl_v3_path / "multi_turn_func_doc"
        if func_doc_path.exists() and any(func_doc_path.iterdir()):
            print(f"✅ Using existing BFCL_v3 at: {bfcl_v3_path}")
            print(f"   Contains {len(list(func_doc_path.iterdir()))} tool definition files")
            
            # Check for multi-turn test files
            multi_turn_files = list(bfcl_v3_path.glob("BFCL_v3_multi_turn_*.json"))
            print(f"   Contains {len(multi_turn_files)} multi-turn test files")
            
            return bfcl_v3_path
    
    # Neither v4 nor v3 available
    print(f"\n{'='*80}")
    print("❌ NO BFCL DATA AVAILABLE")
    print(f"{'='*80}")
    print("\nPlease either:")
    print("1. Provide a HuggingFace token with access to gorilla-llm/BFCL:")
    print("   export HF_TOKEN=your_token_here")
    print("   python download_bfcl_v4.py --token your_token_here")
    print("\n2. Or manually download BFCL_v3 or BFCL_v4 from:")
    print("   https://huggingface.co/datasets/gorilla-llm/BFCL")
    print(f"\nAnd extract to: {data_dir}/BFCL_v3/ or {data_dir}/BFCL_v4/")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download BFCL_v4 dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/ishalyminov/data/magnet_mt/data"),
        help="Directory to store the dataset"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if exists"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)"
    )
    
    args = parser.parse_args()
    
    # Get token from args or environment
    token = args.token or os.environ.get("HF_TOKEN")
    
    # Download
    bfcl_v4_path = download_bfcl_v4(
        data_dir=args.data_dir,
        force_download=args.force,
        token=token
    )
    
    print(f"\n✅ BFCL_v4 ready at: {bfcl_v4_path}")


if __name__ == "__main__":
    main()