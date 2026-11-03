# GitHub upload walkthrough

This folder is designed to be the **repository root**. Extract the ZIP and
upload the files inside `GNN-Geometry-Optimization/` (not only the ZIP itself).

1. Create an EMPTY repository named `GNN-Geometry-Optimization`. Keep it
   PRIVATE until redistribution rights and an OSI-approved software license
   have been confirmed.
2. In the author's A100 Python environment, perform:

```
cd GNN-Geometry-Optimization
bash scripts/check_release.sh
bash scripts/run_demo.sh B3D BDim
# Optional: test all six real GNN runners on the packaged synthetic demo:
bash scripts/run_demo.sh all
```

3. Check files staged for upload:

```
git init -b main
git add -A
git diff --cached --name-only
```

The tracked demo should contain `data/demo/manifest.csv`,
`data/demo/descriptors.csv`, and 32 `data/demo/xyz/*.xyz` files. Training
outputs, machine-specific local paths and the optionally generated original
research subset in `data/demo_real/` must NOT be staged.

4. After checking the staged content:

```
git config user.name "YOUR NAME"
git config user.email "YOUR PUBLIC OR NOREPLY EMAIL"
git commit -m "Initial code and executable synthetic demo"
git remote add origin https://github.com/YOUR_USERNAME/GNN-Geometry-Optimization.git
git push -u origin main
```

GitHub authentication requires the account owner's own credentials (for example,
an authenticated `gh` client or SSH setup). Never put credentials or a personal
access token into commands stored in the repository.

## Public release checklist

- KRICT and upstream dataset redistribution rights confirmed.
- Valid `LICENSE` selected and included; `CITATION.cff` release fields updated.
- Figshare dataset DOI and full-data/outer-fold download instructions verified.
- Exact article RMSE and bootstrap results independently matched to archived
  held-out predictions. The synthetic demo cannot verify article scores.
- Dataset/geometry provenance and citation instructions made public.

No tool automatically publishes or pushes files from this package.
