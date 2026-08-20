# Data directory structure

`data/` stores raw and preprocessed benchmark data. Dataset payloads are not
tracked by Git because they are large or generated. The dataset-level scaffold
is tracked with `.gitkeep` files, and the complete case-level naming rules are
documented here.

## Formal RCAEval RE1 Zenodo v2 data

```text
data/
├── raw/
│   └── rcaeval_zenodo_v2/
│       ├── re1_ob/
│       │   └── RE1-OB/
│       │       └── <service>_<fault>/
│       │           └── <run>/
│       │               ├── data.csv
│       │               └── inject_time.txt
│       ├── re1_ss/
│       │   └── RE1-SS/
│       │       └── <service>_<fault>/<run>/...
│       └── re1_tt/
│           └── RE1-TT/
│               └── <service>_<fault>/<run>/...
└── processed/
    └── rcaeval_zenodo_v2/
        └── default/
            └── rcaeval_re1/
                ├── re1_ob/
                ├── re1_ss/
                └── re1_tt/
                    └── <dataset>__<service>_<fault>__<run>/
                        ├── normal_data.csv
                        ├── abnormal_data.csv
                        └── case_info.json
```

- `<fault>` is one of `cpu`, `delay`, `disk`, `loss`, or `mem`.
- `<run>` is an integer from `1` to `5`.
- `re1_ob`, `re1_ss`, and `re1_tt` contain 125 cases each, for 375 cases in
  total.
- Raw data is downloaded from Zenodo record `14590730`, version `v2`.

Use the formal paths configured in
`configs/main/rcaeval_re1_zenodo_v2.yaml`:

```text
data/raw/rcaeval_zenodo_v2
data/processed/rcaeval_zenodo_v2
```

## BARO pilot data

The earlier BARO pilot uses the following dataset-level layout:

```text
data/
├── raw/
│   ├── online_boutique/
│   ├── sock_shop/
│   └── train_ticket/
└── processed/
    └── default/
        ├── online_boutique/
        ├── sock_shop/
        └── train_ticket/
```

These paths remain for pilot reproducibility but are not the formal RCAEval
Zenodo v2 experiment paths.

## Git policy

The scaffold and this document are tracked. Large or generated payloads,
including CSV, JSON, Parquet, ZIP, images, logs, injection-time text files,
and download caches, remain ignored by `.gitignore`.
