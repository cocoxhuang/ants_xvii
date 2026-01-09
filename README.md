# Infinite Families of BSD-Verified Quadratic Twists

This repository contains code for verifying the Birch and Swinnerton-Dyer (BSD) conjecture for infinite families of quadratic twists of elliptic curves over Q.

## Directory Structure

```
ants_xvii/
├── infinite_bsd/
│   ├── Algorithm1.py      # Find elliptic curves with BSD-verified twist families
│   ├── Algorithm2.py      # Compute admissible twists for identified curves
│   ├── check_curve.py     # Verify conditions for individual curves
│   └── output/
│       ├── ec_labels.txt  # Output: curve labels from Algorithm 1
│       └── res.json       # Output: admissible twists from Algorithm 2
│
└── rs_conjecture/
    ├── RS_visualization.py                # Generate RS conjecture visualizations
    ├── RS_conjecture_visualization.ipynb  # Interactive notebook version
    └── output/
        ├── frames/                        # PNG animation frames (per-run subfolders)
        │   └── <run_id>/                  # e.g., 46a1_maxd1000_nf5_20260107_120000
        │       └── frame_*.png
        ├── <run_id>.html                  # HTML animations (one per run)
        └── convergence_*.gif              # Legacy animated GIFs
```

## Requirements

- [SageMath](https://www.sagemath.org/) (version 10.0+)
- Python packages: `pandas`, `numpy`, `pytz`, `matplotlib`, `scipy`
- Access to LMFDB database (via the included `lmfdb` module)

## infinite_bsd

This folder contains the main algorithms for finding and verifying BSD for quadratic twist families.

### Algorithm 1: Finding Curves with BSD-Verified Twist Families

Identifies elliptic curves over Q that have an infinite family of quadratic twists satisfying the full BSD conjecture formula. Uses criteria from:

- **[CLZ20]**: Cai-Li-Zhai (2020) for curves with E(Q)[2] = Z/2Z
- **[Zha16]**: Zhai (2016) for curves without 2-torsion

**Usage:**

```bash
cd ants_xvii/infinite_bsd
sage -python Algorithm1.py
```

For testing with a smaller conductor bound:

```bash
sage -python Algorithm1.py --cond_upper_bound 150
```

**Output:** `output/ec_labels.txt` containing Cremona labels, source paper, and LMFDB labels.

### Algorithm 2: Computing Admissible Twists

For each curve identified by Algorithm 1, computes the admissible squarefree integers M (up to a bound) for which the quadratic twist E_M satisfies BSD.

**Usage:**

```bash
cd ants_xvii/infinite_bsd
sage -python Algorithm2.py
```

**Configuration:** Edit the constants at the top of the file:
- `TWIST_BOUND`: Maximum value of M to check (default: 10000)

**Output:** `output/res.json` containing admissible twists for each curve.

### check_curve.py: Verify Individual Curves

Checks all conditions from Algorithm 1 for a specific elliptic curve.

**Usage:**

```bash
cd ants_xvii/infinite_bsd
sage -python check_curve.py <Cremona_label>
```

**Example:**

```bash
sage -python check_curve.py 46a1
```

## rs_conjecture

Visualizations for the Radziwiłł-Soundararajan conjecture, which predicts the distribution of Sha (Shafarevich-Tate group) values for quadratic twists.

### RS_visualization.py

Generates PNG frames and an interactive HTML animation showing how the empirical distribution of normalized Sha values converges to the standard normal distribution N(0,1) as the discriminant bound increases.

Each run creates uniquely named outputs to preserve previous runs, using a run ID of the form: `<curve>_maxd<max_d>_nf<num_frames>_<timestamp>`

**Usage:**

```bash
cd ants_xvii/rs_conjecture
sage -python RS_visualization.py --max_d 10000 --num_frames 10
sage -python RS_visualization.py --curve 11a1 --max_d 5000 --num_frames 5
```

**Arguments:**
- `--curve`: Elliptic curve label (default: 46a1)
- `--max_d`: Maximum absolute discriminant (default: 1000)
- `--num_frames`: Number of animation frames (default: 5)
- `--output_dir`: Output directory (default: output)

**Output:**
- PNG frames: `output/frames/<run_id>/frame_*.png`
- HTML animation: `output/<run_id>.html`

### RS_conjecture_visualization.ipynb

Interactive Jupyter notebook version for exploratory analysis.

**Usage:**

```bash
cd ants_xvii/rs_conjecture
sage -n jupyter RS_conjecture_visualization.ipynb
```

## LMFDB

This repository is forked from the [LMFDB (L-functions and Modular Forms Database)](https://github.com/LMFDB/lmfdb) project. The algorithms in `ants_xvii/infinite_bsd` query the LMFDB database to retrieve precomputed quantities for elliptic curves, including invariants, BSD data, and information about quadratic twists.

## References

- **[CLZ20]** L. Cai, C. Li, and S. Zhai, "On the 2-part of the Birch and Swinnerton-Dyer conjecture for quadratic twists of elliptic curves", J. Lond. Math. Soc. (2) 101 (2020), no. 2, 714–734.

- **[Zha16]** S. Zhai, "Non-vanishing theorems for quadratic twists of elliptic curves", Asian J. Math. 20 (2016), no. 3, 475–502.

- **[RS]** M. Radziwiłł and K. Soundararajan, "Moments and distribution of central L-values of quadratic twists of elliptic curves", Invent. Math. 202 (2015), no. 3, 1029–1068.
