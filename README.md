# MIT-BIH-CNN-Accelerator
arrhythmia-cnn/
│
├── data/
│   ├── raw/                    # Original MIT-BIH files, untouched
│   │   └── mitdb/
│   │       ├── 100.dat
│   │       ├── 100.hea
│   │       ├── 100.atr
│   │       └── ... (all 48 records)
│   ├── processed/              # Preprocessed numpy arrays, ready for training
│   │   ├── X_train.npy
│   │   ├── y_train.npy
│   │   └── ...
│   └── splits/                 # Patient-wise train/val/test split indices
│
├── notebooks/                  # Exploratory work, visualization, one-off analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_dev.ipynb
│   └── 03_model_prototyping.ipynb
│
├── src/                        # All importable Python source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # wfdb loading logic
│   │   ├── preprocessing.py    # filtering, normalization, segmentation
│   │   └── dataset.py          # PyTorch Dataset class
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_cnn.py
│   │   └── variants/           # Architecture experiments go here
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── metrics.py
│   └── utils/
│       ├── __init__.py
│       └── profiler.py         # MAC/param counting, will matter for HW later
│
├── experiments/                # One folder per run, auto-generated
│   └── exp_001_baseline/
│       ├── config.yaml
│       ├── metrics.json
│       └── checkpoints/
│
├── configs/                    # YAML config files for hyperparams and architecture
│   └── baseline.yaml
│
├── tests/                      # Unit tests for preprocessing and data loading
│   └── test_preprocessing.py
│
├── requirements.txt
├── .gitignore
└── README.md


The big picture of the project structure
Think of the project in three phases, and the folder structure maps directly onto them:
Phase 1 — Get the data ready (src/data/)
The raw MIT-BIH files are one long continuous waveform per patient. Your CNN can't train on that directly. You need to chop it into individual beats, filter out noise, normalize the amplitude, and handle the fact that 90%+ of beats are labeled "Normal." That's what loader.py, preprocessing.py, and dataset.py do together. The output of this phase is stored in data/processed/ as numpy arrays ready to feed into training.
Phase 2 — Build and optimize the CNN (src/models/, src/training/, configs/, experiments/)
You train a baseline CNN first, then systematically try different architectures — different numbers of layers, filter sizes, kernel widths — and compare their accuracy, sensitivity, and computational cost. Each experiment gets its own folder under experiments/ so you can always go back and compare. The configs/ folder stores the settings for each run as a YAML file so experiments are reproducible. src/utils/profiler.py logs how many multiply-accumulate operations (MACs) each architecture uses — this number will matter enormously in Phase 3.
Phase 3 — Build hardware to accelerate it (future)
Once you've settled on an architecture that hits your accuracy targets, you design a hardware accelerator — essentially a custom chip or FPGA design that runs that specific CNN as fast and as efficiently as possible. The reason you care about MACs, kernel sizes, and filter counts during Phase 2 is that these directly determine how you design the hardware. A CNN with uniform 3x1 kernel sizes is much easier to build a systolic array for than one with variable sizes. Quantizing weights to INT8 during training means your hardware can use cheap fixed-point multipliers instead of expensive floating-point ones. Every architectural choice you make in software has a hardware cost implication, which is why the profiler lives in the project from day one rather than being an afterthought.