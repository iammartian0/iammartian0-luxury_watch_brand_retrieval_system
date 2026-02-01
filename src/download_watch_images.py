from ddgs import DDGS
from pathlib import Path
import csv
import logging
import time
import hashlib
from urllib.parse import urlparse
import requests
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BRANDS = [
    "Rolex",
    "Omega",
    "Breitling",
    "Cartier",
    "Seiko",
    "Longines",
    "TAG Heuer",
    "Hublot",
    "Audemars Piguet",
    "Patek Philippe"
]

def get_file_extension(url):
    parsed = urlparse(url)
    path = parsed.path.lower()
    if path.endswith('.jpg') or path.endswith('.jpeg'):
        return '.jpg'
    elif path.endswith('.png'):
        return '.png'
    elif path.endswith('.webp'):
        return '.webp'
    return '.jpg'

def generate_unique_filename(url, index):
    return f"Image_{index}_{hashlib.md5(url.encode()).hexdigest()[:8]}"

def download_image(url, save_path):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=15, stream=True, headers=headers)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        logger.warning(f"Failed to download {url[:50]}...: {e}")
        return False

def download_for_brand(brand, root, target_count=200):
    search_query = f"{brand} watches"
    logger.info(f"📥 Searching for {brand} images...")

    brand_dir = root / brand
    brand_dir.mkdir(parents=True, exist_ok=True)

    downloaded_urls = set()
    downloaded_count = 0
    index = 1

    try:
        ddgs = DDGS()
        results = ddgs.images(
            search_query,
            region='wt-wt',
            safesearch='moderate',
            max_results=400
        )

        for result in results:
            if downloaded_count >= target_count:
                break

            url = result.get('image')
            if url:
                url = url.strip()
                if url not in downloaded_urls and not url.endswith('.gif'):
                    ext = get_file_extension(url)
                    filename = generate_unique_filename(url, index) + ext
                    save_path = brand_dir / filename

                    logger.info(f"  [{downloaded_count + 1:3d}/{target_count}] {filename}")

                    if download_image(url, save_path):
                        downloaded_urls.add(url)
                        downloaded_count += 1
                        index += 1
                        time.sleep(0.3)

        logger.info(f"✅ Downloaded {downloaded_count} images for {brand}")
        return downloaded_count

    except Exception as e:
        logger.error(f"❌ Error searching for {brand}: {e}")
        import traceback
        traceback.print_exc()
        return downloaded_count

def update_metadata(root):
    metadata_path = root.parent / "image_metadata.csv"
    all_metadata = []

    for brand_dir in sorted(root.glob("*")):
        if not brand_dir.is_dir():
            continue

        brand_name = brand_dir.name
        image_files = [f for f in brand_dir.glob("*") if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]

        for img_file in sorted(image_files):
            all_metadata.append({
                "brand": brand_name,
                "image_path": str(img_file),
                "search_term": f"{brand_name} watches"
            })

    with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['brand', 'image_path', 'search_term'])
        writer.writeheader()
        writer.writerows(all_metadata)

    logger.info(f"Metadata updated: {len(all_metadata)} records saved to {metadata_path}")
    return len(all_metadata)

def main():
    parser = argparse.ArgumentParser(description='Download watch images')
    parser.add_argument('--brand', type=str, help='Download specific brand (e.g., "Patek Philippe")')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3],
                        help='Stage to download: 1 (3 brands), 2 (3 brands), 3 (4 brands)')
    args = parser.parse_args()

    if args.brand:
        brands_to_download = [args.brand]
        if args.brand not in BRANDS:
            logger.error(f"Brand '{args.brand}' not in supported brands: {BRANDS}")
            return
    elif args.stage:
        stages = {
            1: BRANDS[:3],
            2: BRANDS[3:6],
            3: BRANDS[6:10]
        }
        brands_to_download = stages[args.stage]
    else:
        logger.error("Please specify --brand or --stage")
        return

    root = Path("data/images")
    root.mkdir(parents=True, exist_ok=True)

    all_metadata = []
    total_downloaded = 0

    logger.info("=" * 60)
    logger.info(f"Downloading {len(brands_to_download)} brand(s)")
    logger.info(f"Brands: {', '.join(brands_to_download)}")
    logger.info("=" * 60)

    for idx, brand in enumerate(brands_to_download, 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {idx}/{len(brands_to_download)}: {brand}")
        logger.info(f"{'='*50}")

        count = download_for_brand(brand, root, target_count=200)
        total_downloaded += count

        if count > 0:
            brand_dir = root / brand
            image_files = [f for f in brand_dir.glob("*") if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]

            for img_file in sorted(image_files):
                all_metadata.append({
                    "brand": brand,
                    "image_path": str(img_file),
                    "search_term": f"{brand} watches"
                })

        time.sleep(1)

    update_metadata(root)

    logger.info(f"\n{'='*50}")
    logger.info(f"✅ COMPLETE!")
    logger.info(f"{'='*50}")
    logger.info(f"Images downloaded: {total_downloaded}")
    logger.info(f"Total images in data/images/: {sum(len(list(f.glob('*'))) for f in root.glob('*') if f.is_dir())}")

if __name__ == "__main__":
    main()