# Reproducibility Checklist

Before submission, replace all placeholders in `CITATION.cff` and the manuscript.

Record:

- public repository URL;
- archived DOI;
- commit hash;
- operating system;
- CPU model;
- RAM;
- GPU, if any;
- Python version;
- exact dependency versions;
- experiment configuration;
- seed list;
- start and completion dates;
- raw output checksums; and
- the command used to generate each manuscript table and figure.

The artifact should pass from a clean checkout:

```bash
python -m pip install -e '.[experiments,test]'
pytest
bash scripts/reproduce.sh configs/quick.json results/quick
```

The full journal run is:

```bash
bash scripts/reproduce.sh configs/journal.json results/journal
```
