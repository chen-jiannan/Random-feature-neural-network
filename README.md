# RFNN-Based Wave Propagation Benchmarks

This repository contains the code and manuscript source accompanying the study:

**An RFNN-Based Framework for Wave-Propagation Problems with Dirichlet Boundaries, Absorbing Layers, and Heterogeneous Media**

The project investigates random-feature neural network (RFNN)-based solvers for wave-propagation problems with localized source excitations, with comparisons against finite-difference time-domain (FDTD) references and physics-informed neural network (PINN) baselines.

## Recommended hosting

The most suitable public release workflow for this project is:

1. **GitHub** as the main public code repository
2. **Zenodo** linked to GitHub to mint a DOI for each release
3. Optional: **Gitee mirror** if you want a more accessible mirror for users in mainland China

This combination is standard for computational research and works well for journal submission, reproducibility, and long-term citation.

## Repository contents

```text
RFNN_wave_propagation_open_source_kit/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── .gitignore
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
└── notebooks/
    ├── 1D-RNN-PEC.ipynb
    ├── 1D-RNN-absorbing.ipynb
    └── 2D-RNN-update.ipynb
```

## Current scope of the uploaded code

The current repository bundle includes:

- `notebooks/1D-RNN-PEC.ipynb`  
  1D RFNN wave problems with homogeneous Dirichlet (PEC-type) boundaries
- `notebooks/1D-RNN-absorbing.ipynb`  
  1D damping absorbing-layer experiments, including parameter and seed-sensitivity studies
- `notebooks/2D-RNN-update.ipynb`  
  2D RFNN wave experiments, including source variations and damping absorbing-layer cases
- `manuscript/RFNN.tex`  
  LaTeX source of the manuscript

If additional scripts for 3D experiments, heterogeneous-medium tests, figure generation, or preprocessing are released later, they can be added in new subfolders such as `scripts/`, `data/`, or `figures/`.

## Software requirements

The notebooks currently use the following Python packages:

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib
- Pandas
- Numba
- Jupyter

Two ready-to-use environment files are provided:

- `requirements.txt`
- `environment.yml`

## Quick start

### Option 1: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

### Option 2: conda

```bash
conda env create -f environment.yml
conda activate rfnn-wave
jupyter notebook
```

## Reproducing the experiments

A practical order is:

1. Open `notebooks/1D-RNN-PEC.ipynb`
2. Open `notebooks/1D-RNN-absorbing.ipynb`
3. Open `notebooks/2D-RNN-update.ipynb`

Before running the notebooks, please check:

- the random seed settings
- the device configuration (CPU/GPU)
- output paths for figures or cached arrays
- any file paths that may still point to local directories on the original machine

## Suggested public repository name

A concise GitHub repository name would be:

- `rfnn-wave-propagation`
- or `rfnn-wave-benchmarks`

## Suggested release workflow

1. Create a public GitHub repository
2. Upload the contents of this folder
3. Create a `v1.0.0` release on GitHub
4. Link the repository to Zenodo
5. Let Zenodo archive the release and mint a DOI
6. Add the GitHub URL and Zenodo DOI to the manuscript

## Citation

Please cite the accompanying manuscript if you use this repository in academic work. A machine-readable citation file is provided as `CITATION.cff`.

## License

This repository is released under the MIT License. If your institution or coauthors prefer a different open-source license, replace `LICENSE` before publication.

## Contact

- Jian-Nan Chen
- Jun-Jie Zhang
- Yong-Dong Li

For scientific questions, please use the contact information given in the manuscript.
