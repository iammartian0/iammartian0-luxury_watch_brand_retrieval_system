"""Utilities for downloading data files from GitHub Releases."""

import requests
import logging
from pathlib import Path
from typing import Optional, Callable
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GitHubReleaseDownloader:
    """Download data files from GitHub Releases."""
    
    def __init__(
        self,
        repo: str = "iammartian0/luxury_watch_brand_retrieval_system",
        tag: str = "v1.0",
        data_dir: Optional[Path] = None
    ):
        self.repo = repo
        self.tag = tag
        self.data_dir = data_dir or Path("data")
        self.base_url = f"https://github.com/{repo}/releases/download/{tag}"
    
    def get_download_url(self, filename: str) -> str:
        """Get full download URL for a file."""
        return f"{self.base_url}/{filename}"
    
    def download_file(
        self,
        filename: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        status_text_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        """
        Download a file from GitHub Releases.
        
        Args:
            filename: Name of the file to download
            progress_callback: Function to call with (progress, message) tuples
            status_text_callback: Function to call with status messages
            
        Returns:
            True if successful, False otherwise
        """
        url = self.get_download_url(filename)
        output_path = self.data_dir / filename
        
        if output_path.exists():
            logger.info(f"File already exists: {filename}")
            if progress_callback:
                progress_callback(1.0, f"✓ {filename} already downloaded")
            return True
        
        try:
            if status_text_callback:
                status_text_callback(f"⏳ Downloading {filename}...")
            
            logger.info(f"Downloading {filename} from {url}")
            
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            if progress_callback:
                progress_callback(0.0, f"Downloading {filename}...")
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0 and progress_callback:
                            progress = downloaded / total_size
                            progress_callback(
                                progress * 0.9,  # Don't show 100% until complete
                                f"Downloading {filename}: {downloaded / (1024 * 1024):.1f}MB"
                            )
            
            if progress_callback:
                progress_callback(1.0, f"✓ {filename} downloaded successfully")
            
            if status_text_callback:
                status_text_callback(f"✓ {filename} downloaded successfully")
            
            logger.info(f"Successfully downloaded {filename}")
            return True
            
        except requests.exceptions.Timeout:
            error_msg = f"Timeout downloading {filename}"
            logger.error(error_msg)
            if status_text_callback:
                status_text_callback(f"⚠️ {error_msg}")
            return False
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to download {filename}: {str(e)}"
            logger.error(error_msg)
            if status_text_callback:
                status_text_callback(f"⚠️ {error_msg}")
            return False
            
        except Exception as e:
            error_msg = f"Unexpected error downloading {filename}: {str(e)}"
            logger.error(error_msg)
            if status_text_callback:
                status_text_callback(f"⚠️ {error_msg}")
            return False
    
    def download_multiple_files(
        self,
        filenames: list[str],
        progress_callback: Optional[Callable[[float, str], None]] = None,
        status_text_callback: Optional[Callable[[str], None]] = None
    ) -> dict[str, bool]:
        """
        Download multiple files from GitHub Releases.
        
        Args:
            filenames: List of filenames to download
            progress_callback: Function to call with progress updates
            status_text_callback: Function to call with status messages
            
        Returns:
            Dictionary mapping filename -> success status
        """
        results = {}
        total_files = len(filenames)
        
        for idx, filename in enumerate(filenames):
            file_progress = None
            if progress_callback:
                def file_progress_callback(progress: float, message: str):
                    overall_progress = (idx + progress) / total_files
                    progress_callback(overall_progress, f"[{idx + 1}/{total_files}] {message}")
                file_progress = file_progress_callback
            
            success = self.download_file(
                filename,
                progress_callback=file_progress,
                status_text_callback=status_text_callback
            )
            results[filename] = success
        
        return results


def check_required_files(
    data_dir: Path,
    required_files: list[str]
) -> dict[str, bool]:
    """
    Check which required files exist.
    
    Args:
        data_dir: Path to data directory
        required_files: List of filenames to check
        
    Returns:
        Dictionary mapping filename -> exists status
    """
    return {filename: (data_dir / filename).exists() for filename in required_files}


def get_missing_files(
    data_dir: Path,
    required_files: list[str]
) -> list[str]:
    """
    Get list of missing required files.
    
    Args:
        data_dir: Path to data directory
        required_files: List of filenames to check
        
    Returns:
        List of missing filenames
    """
    return [f for f in required_files if not (data_dir / f).exists()]


if __name__ == "__main__":
    # Test the downloader
    downloader = GitHubReleaseDownloader()
    
    def progress_callback(progress: float, message: str):
        print(f"\r[{progress:.1%}] {message}", end='', flush=True)
        if progress >= 1.0:
            print()
    
    def status_callback(message: str):
        print(f"\n{message}")
    
    files_to_download = [
        "text_embeddings.npy",
        "text_index.faiss",
        "image_embeddings.npy",
        "image_index.faiss"
    ]
    
    print("Checking for missing files...")
    missing = get_missing_files(downloader.data_dir, files_to_download)
    
    if missing:
        print(f"Missing files: {missing}")
        print("Starting download...")
        results = downloader.download_multiple_files(
            missing,
            progress_callback=progress_callback,
            status_text_callback=status_callback
        )
        print("\nDownload results:")
        for filename, success in results.items():
            status = "✓" if success else "✗"
            print(f"{status} {filename}")
    else:
        print("All required files are present!")

