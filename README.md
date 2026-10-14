# Satellite Disaster AI 🛰️🌊

An advanced AI-powered Geospatial Intelligence and Satellite Disaster Assessment platform. This application leverages deep learning (PyTorch, UNet) and computer vision to analyze satellite imagery, segment disaster-affected regions (such as floods), compute impact metrics, and generate actionable disaster assessment reports.

---

## 🌟 Key Features

- **Deep Learning Disaster Segmentation**: UNet architecture tailored for satellite radar (SAR) and optical imagery to detect and segment flooded and affected areas.
- **Geospatial Processing**: Preprocessing pipelines handling multi-spectral bands, geospatial normalization, and coordinate referencing.
- **Interactive Desktop GUI**: Modern, responsive PySide6 (Qt6) interface with real-time image viewer, prediction visualizer, and inspection controls.
- **Damage & Area Quantification**: Automatically computes flooded area in square kilometers, acres, and impact percentage.
- **Severity & Risk Assessment**: Dynamic severity level scoring (Low, Moderate, High, Critical) based on damage extent and heuristics.
- **Automated Report Generation**: Generates comprehensive PDF disaster assessment reports with visualization figures and impact metrics.
- **Historical Analysis Database**: Tracks past assessments and analyses for temporal disaster monitoring.

---

## 📁 Repository Structure

```text
satellitedisasterai/
├── app/                  # Web/CLI application entry points and training runners
│   ├── app.py
│   └── train.py
├── checkpoints/          # Model checkpoints (tracked via Git LFS)
│   └── best_flood_segmentation_model.pth
├── data/                 # Dataset loader and downloader scripts
│   ├── dataset.py
│   └── download_sen1floods11.py
├── gui/                  # PySide6 Desktop GUI application
│   ├── main_window.py
│   └── styles.py
├── inference/            # Prediction, segmentation, and boundary extraction
│   ├── area.py
│   ├── boundary.py
│   ├── predict.py
│   └── segmentation.py
├── models/               # Model architectures and predictors
│   ├── predictor.py
│   ├── segmentation.py
│   ├── train_flood.py
│   └── unet.py
├── preprocessing/        # Geospatial and image preprocessing routines
│   ├── geospatial.py
│   └── preprocess.py
├── training/             # Training loops, loss functions, and validation
│   ├── losses.py
│   ├── train.py
│   └── validate.py
├── utils/                # Helper utilities (reporting, analytics, heatmap, db)
│   ├── analytics.py
│   ├── area_calculator.py
│   ├── database.py
│   ├── heatmap.py
│   ├── image_utils.py
│   ├── report_generator.py
│   └── severity.py
├── visualization/        # Plotting and visualization utilities
│   └── visualize.py
├── app.py                # Main desktop application runner
├── requirements.txt      # Project dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Git & Git LFS (Large File Storage)
- CUDA-enabled GPU (optional, but recommended for faster training and inference)

### 1. Clone the Repository

Ensure you have Git LFS installed before cloning:

```bash
git lfs install
git clone https://github.com/Architpateltech/sattelitefloodai.git
cd sattelitefloodai
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🖥️ Usage

### Run the Desktop GUI Application

Launch the desktop interface:

```bash
python app.py
```

### Run Model Training

To train the flood segmentation model on your dataset:

```bash
python -m training.train
```

### Run Standalone Inference

Perform inference on satellite images:

```bash
python -m inference.predict --input path/to/image.tif
```

---

## 📊 Dataset

The model is designed to work with high-resolution satellite imagery, including:
- **Sen1Floods11**: Sentinel-1 SAR and Sentinel-2 optical flood dataset.
- Use `data/download_sen1floods11.py` to acquire and prepare dataset splits.

---

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, Torchvision
- **Geospatial & Computer Vision**: OpenCV, Rasterio, NumPy, Matplotlib
- **Desktop UI**: PySide6 (Qt6)
- **Reporting & Storage**: ReportLab, SQLite3

---

## 📄 License

This project is licensed under the MIT License.
