#!/usr/bin/env python3
"""
Download and manage BFCL (Berkeley Function Calling Leaderboard) datasets from HuggingFace.

This module provides functionality to:
1. Check if BFCL dataset exists locally
2. Download the latest version if not present
3. Cache the dataset for offline use
"""

import json
import logging
from pathlib import Path
from typing import Optional, List
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.utils import RepositoryNotFoundError

logger = logging.getLogger(__name__)

# BFCL dataset information
BFCL_DATASET_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_DEFAULT_VERSION = "v3"  # Current latest version

# Standard BFCL file patterns
BFCL_TEST_FILES = [
    "BFCL_{version}_simple.json",
    "BFCL_{version}_multiple.json",
    "BFCL_{version}_parallel.json",
    "BFCL_{version}_parallel_multiple.json",
    "BFCL_{version}_chatable.json",
    "BFCL_{version}_irrelevance.json",
    "BFCL_{version}_java.json",
    "BFCL_{version}_javascript.json",
    "BFCL_{version}_rest.json",
    "BFCL_{version}_sql.json",
]

BFCL_LIVE_FILES = [
    "BFCL_{version}_live_simple.json",
    "BFCL_{version}_live_multiple.json",
    "BFCL_{version}_live_parallel.json",
    "BFCL_{version}_live_parallel_multiple.json",
    "BFCL_{version}_live_relevance.json",
    "BFCL_{version}_live_irrelevance.json",
]

BFCL_EXEC_FILES = [
    "BFCL_{version}_exec_simple.json",
    "BFCL_{version}_exec_multiple.json",
    "BFCL_{version}_exec_parallel.json",
    "BFCL_{version}_exec_parallel_multiple.json",
]

BFCL_MULTI_TURN_FILES = [
    "BFCL_{version}_multi_turn_base.json",
    "BFCL_{version}_multi_turn_composite.json",
    "BFCL_{version}_multi_turn_long_context.json",
    "BFCL_{version}_multi_turn_miss_func.json",
    "BFCL_{version}_multi_turn_miss_param.json",
]

BFCL_FUNC_DOC_DIR = "multi_turn_func_doc"


class BFCLDownloader:
    """Download and manage BFCL datasets from HuggingFace."""
    
    def __init__(self, data_dir: str | Path, version: str = BFCL_DEFAULT_VERSION):
        """
        Initialize BFCL downloader.
        
        Args:
            data_dir: Base directory to store BFCL data
            version: BFCL version to download (default: v3)
        """
        self.data_dir = Path(data_dir)
        self.version = version
        self.bfcl_dir = self.data_dir / f"BFCL_{version}"
        self.api = HfApi()
        
    def exists(self) -> bool:
        """Check if BFCL dataset already exists locally."""
        if not self.bfcl_dir.exists():
            return False
        
        # Check for at least some test files
        test_file = self.bfcl_dir / f"BFCL_{self.version}_simple.json"
        return test_file.exists()
    
    def get_available_files(self) -> List[str]:
        """Get list of available files in the HuggingFace dataset."""
        try:
            dataset_info = self.api.dataset_info(BFCL_DATASET_ID)
            return [f.rfilename for f in dataset_info.siblings]
        except RepositoryNotFoundError:
            logger.error(f"Dataset {BFCL_DATASET_ID} not found on HuggingFace")
            return []
    
    def download(self, force: bool = False) -> Path:
        """
        Download BFCL dataset from HuggingFace.
        
        Args:
            force: Force download even if dataset exists
            
        Returns:
            Path to the downloaded dataset directory
        """
        if self.exists() and not force:
            logger.info(f"BFCL {self.version} already exists at {self.bfcl_dir}")
            return self.bfcl_dir
        
        logger.info(f"Downloading BFCL {self.version} from HuggingFace...")
        
        # Create directory
        self.bfcl_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all available files
        available_files = self.get_available_files()
        
        if not available_files:
            raise RuntimeError(f"No files found in dataset {BFCL_DATASET_ID}")
        
        # Download test files
        all_patterns = (
            BFCL_TEST_FILES + BFCL_LIVE_FILES + 
            BFCL_EXEC_FILES + BFCL_MULTI_TURN_FILES
        )
        
        downloaded_count = 0
        for pattern in all_patterns:
            filename = pattern.format(version=self.version)
            if filename in available_files:
                try:
                    local_path = hf_hub_download(
                        repo_id=BFCL_DATASET_ID,
                        filename=filename,
                        repo_type="dataset",
                        local_dir=self.bfcl_dir,
                        local_dir_use_symlinks=False
                    )
                    logger.debug(f"Downloaded: {filename}")
                    downloaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to download {filename}: {e}")
        
        # Download possible_answer files
        answer_dir = self.bfcl_dir / "possible_answer"
        answer_dir.mkdir(exist_ok=True)
        
        answer_files = [f for f in available_files if f.startswith("possible_answer/")]
        for answer_file in answer_files:
            if self.version in answer_file:
                try:
                    hf_hub_download(
                        repo_id=BFCL_DATASET_ID,
                        filename=answer_file,
                        repo_type="dataset",
                        local_dir=self.bfcl_dir,
                        local_dir_use_symlinks=False
                    )
                    logger.debug(f"Downloaded: {answer_file}")
                    downloaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to download {answer_file}: {e}")
        
        # Download multi_turn_func_doc files
        func_doc_files = [f for f in available_files if f.startswith(f"{BFCL_FUNC_DOC_DIR}/")]
        func_doc_dir = self.bfcl_dir / BFCL_FUNC_DOC_DIR
        func_doc_dir.mkdir(exist_ok=True)
        
        for func_doc_file in func_doc_files:
            try:
                hf_hub_download(
                    repo_id=BFCL_DATASET_ID,
                    filename=func_doc_file,
                    repo_type="dataset",
                    local_dir=self.bfcl_dir,
                    local_dir_use_symlinks=False
                )
                logger.debug(f"Downloaded: {func_doc_file}")
                downloaded_count += 1
            except Exception as e:
                logger.warning(f"Failed to download {func_doc_file}: {e}")
        
        logger.info(f"Downloaded {downloaded_count} files to {self.bfcl_dir}")
        return self.bfcl_dir
    
    def get_func_doc_dir(self) -> Path:
        """Get path to multi_turn_func_doc directory."""
        return self.bfcl_dir / BFCL_FUNC_DOC_DIR
    
    def get_test_file_paths(self) -> List[Path]:
        """Get paths to all test files."""
        all_patterns = (
            BFCL_TEST_FILES + BFCL_LIVE_FILES + 
            BFCL_EXEC_FILES + BFCL_MULTI_TURN_FILES
        )
        
        paths = []
        for pattern in all_patterns:
            filename = pattern.format(version=self.version)
            path = self.bfcl_dir / filename
            if path.exists():
                paths.append(path)
        
        return paths
    
    def get_answer_dir(self) -> Path:
        """Get path to possible_answer directory."""
        return self.bfcl_dir / "possible_answer"


def ensure_bfcl_data(
    data_dir: str | Path,
    version: str = BFCL_DEFAULT_VERSION,
    force_download: bool = False
) -> Path:
    """
    Ensure BFCL data exists, downloading if necessary.
    
    Args:
        data_dir: Base directory for data
        version: BFCL version to use
        force_download: Force download even if exists
        
    Returns:
        Path to BFCL dataset directory
    """
    downloader = BFCLDownloader(data_dir, version)
    
    if not downloader.exists() or force_download:
        logger.info(f"BFCL {version} not found, downloading...")
        return downloader.download(force=force_download)
    
    logger.info(f"Using existing BFCL {version} at {downloader.bfcl_dir}")
    return downloader.bfcl_dir


if __name__ == "__main__":
    import sys
    
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/ishalyminov/data/magnet_mt/data"
    version = sys.argv[2] if len(sys.argv) > 2 else BFCL_DEFAULT_VERSION
    
    print(f"Ensuring BFCL {version} data in {data_dir}...")
    bfcl_path = ensure_bfcl_data(data_dir, version)
    print(f"✅ BFCL data available at: {bfcl_path}")
    
    # Print statistics
    downloader = BFCLDownloader(data_dir, version)
    test_files = downloader.get_test_file_paths()
    print(f"\nTest files available: {len(test_files)}")
    for tf in test_files[:5]:
        print(f"  - {tf.name}")
    if len(test_files) > 5:
        print(f"  ... and {len(test_files) - 5} more")
    
    func_doc_dir = downloader.get_func_doc_dir()
    if func_doc_dir.exists():
        func_docs = list(func_doc_dir.glob("*.json"))
        print(f"\nFunction docs available: {len(func_docs)}")
        for fd in func_docs[:5]:
            print(f"  - {fd.name}")