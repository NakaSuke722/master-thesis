from huggingface_hub import snapshot_download


def main() -> None:
    snapshot_download(
        repo_id="phamquiluan/RCAEval",
        repo_type="dataset",
        allow_patterns=[
            "cases.parquet",
            "re1*",
        ],
        local_dir="data/raw/rcaeval_re1",
    )


if __name__ == "__main__":
    main()