REPO = "https://github.com/google-research/google-research/tree/master/goemotions/data"
EKMAN_MAPPING_FILENAME = "ekman_mapping.json"
DEV_DATA_FILENAME = "dev.tsv"
TEST_DATA_FILENAME = "test.tsv"
TRAIN_DATA_FILENAME = "train.tsv"
EMOTION_MAPPING_FILENAME = "emotions.txt"

import os
import requests


def prepare_env():
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/preprocessed"):
        os.makedirs("data/preprocessed")


def download_and_extract(url, extract_to="data"):
    """Download the file at `url` into `extract_to/`"""
    r = requests.get(url, stream=True)
    r.raise_for_status()
    fn = os.path.join(extract_to, os.path.basename(url))
    with open(fn, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return fn


def fetch_data(force_download=False):
    prepare_env()

    base_raw = REPO.replace("github.com/", "raw.githubusercontent.com/").replace(
        "/tree", ""
    )

    for fname in (
        EKMAN_MAPPING_FILENAME,
        DEV_DATA_FILENAME,
        TEST_DATA_FILENAME,
        TRAIN_DATA_FILENAME,
        EMOTION_MAPPING_FILENAME,
    ):
        url = f"{base_raw}/{fname}"
        if not force_download and os.path.exists(os.path.join("data", fname)):
            print(f"File {fname} already exists, skipping download")
            continue
        out = download_and_extract(url)
        print(f"Downloaded {out}")


if __name__ == "__main__":
    fetch_data()
