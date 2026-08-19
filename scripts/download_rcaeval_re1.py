from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile


ZENODO_RECORD = "14590730"

RAW_ROOT = Path(
    "data/raw/rcaeval_zenodo_v2"
)

ARCHIVES = {
    "re1_ob": {
        "filename": "RE1-OB.zip",
        "directory": "RE1-OB",
        "md5": "47cce26ed24140e8974e68f9db2a5e9c",
    },
    "re1_ss": {
        "filename": "RE1-SS.zip",
        "directory": "RE1-SS",
        "md5": "d2b15cbd3bb3cf6ec5f3cc65f7fac225",
    },
    "re1_tt": {
        "filename": "RE1-TT.zip",
        "directory": "RE1-TT",
        "md5": "48a26925ce47fd4bcfbedbae4f31475b",
    },
}


def calculate_md5(path: Path) -> str:
    digest = hashlib.md5()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def download_archive(
    filename: str,
    destination: Path,
) -> None:
    url = (
        "https://zenodo.org/records/"
        f"{ZENODO_RECORD}/files/"
        f"{filename}?download=1"
    )

    print(f"Downloading {filename}...")
    urlretrieve(url, destination)


def main() -> None:
    RAW_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for dataset, metadata in ARCHIVES.items():
        filename = metadata["filename"]
        archive_path = RAW_ROOT / filename

        if not archive_path.is_file():
            download_archive(
                filename,
                archive_path,
            )

        actual_md5 = calculate_md5(
            archive_path
        )

        if actual_md5 != metadata["md5"]:
            raise ValueError(
                "Checksum mismatch: "
                f"{archive_path}, "
                f"expected={metadata['md5']}, "
                f"actual={actual_md5}"
            )

        extract_root = RAW_ROOT / dataset

        expected_root = (
            extract_root
            / metadata["directory"]
        )

        if expected_root.is_dir():
            print(
                f"Already extracted: "
                f"{expected_root}"
            )
            continue

        extract_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"Extracting {archive_path} "
            f"to {extract_root}..."
        )

        with ZipFile(archive_path) as archive:
            archive.extractall(extract_root)

    print(
        "RCAEval RE1 Zenodo v2 "
        "download completed."
    )


if __name__ == "__main__":
    main()