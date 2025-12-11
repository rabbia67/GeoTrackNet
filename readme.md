Here is the updated `README.md` with the **Calculate AIS Mean** step added. I have placed it as **Step 3** under the **Data Preparation** section, as this is the logical step after creating trajectories and immediately before starting the training process.

-----

# GeoTrackNet

**Anomaly Detection in Vessel Trajectories using Variational Recurrent Neural Networks**

GeoTrackNet is a deep learning project designed to detect anomalies in maritime vessel trajectories. By leveraging AIS (Automatic Identification System) data, this model constructs probabilistic representations of vessel movement to identify abnormal behaviors using an *a contrario* detection approach.

-----

## Table of Contents

  - [Prerequisites](https://www.google.com/search?q=%23-prerequisites)
  - [Installation](https://www.google.com/search?q=%23-installation)
  - [Project Structure](https://www.google.com/search?q=%23-project-structure)
  - [Data Preparation](https://www.google.com/search?q=%23-data-preparation)
  - [Usage](https://www.google.com/search?q=%23-usage)
      - [1. Training](https://www.google.com/search?q=%231-training-the-embedding-layer)
      - [2. Log Probability](https://www.google.com/search?q=%232-log-probability-calculation)
      - [3. Anomaly Detection](https://www.google.com/search?q=%233-anomaly-detection)
  - [Troubleshooting](https://www.google.com/search?q=%23-troubleshooting)
  - [Acknowledgments](https://www.google.com/search?q=%23-acknowledgments)

-----

## Prerequisites

The instructions assume you are working in a command-line environment (Windows/Linux).

  * **Python:** 3.7
  * **Package Manager:** Anaconda or Miniconda (Highly Recommended)

-----

## Installation

It is critical to use a dedicated environment due to specific legacy dependency requirements (TensorFlow 1.15).

1.  **Create the Environment**

    ```bash
    conda create -n geotracknet python=3.7
    conda activate geotracknet
    ```

2.  **Install Dependencies**
    The project requires TensorFlow 1.15 for compatibility with `tensorflow.compat.v1`.

    ```bash
    pip install tensorflow==1.15.0 numpy matplotlib scipy tqdm dm-sonnet==1.36
    ```

> **⚠️ Note:** Do not upgrade TensorFlow beyond 1.15.x, as the graph construction relies on specific API calls from this version.

-----

## Project Structure

```text
GeoTrackNet/
├── data/
│   ├── datasets.py              # Reader pipelines
│   ├── calculate_AIS_mean.py    # Calculates mean of AIS "four-hot" vectors
│   ├── dataset_preprocessing.py # Main preprocessing script
│   └── csv2pkl.py               # Parses raw AIS (AIVDM/CSV) to Pickle
├── models/
│   └── vrnn.py                  # Variational Recurrent Neural Network implementation
├── results/                     # Output directory
├── chkpt/                       # Checkpoints and summaries
├── geotracknet.py               # Main entry point (Training & Modes)
├── runners.py                   # Graph construction for training/eval
├── bounds.py                    # Code for computing bounds
├── contrario_kde.py             # A contrario detection script
├── contrario_utils.py           # Utilities for detection
└── utils.py                     # General utilities
```

-----

## Data Preparation

This project utilizes two primary datasets. You must preprocess them before training.

### 1\. Sources

  * **MarineC Dataset:** Available at [MarineCadastre.gov](https://marinecadastre.gov/ais/).
  * **Brittany Dataset:** Provided by CLS. A subset is included in `data/ct_2017010203_10_20.zip`.

### 2\. Preprocessing Steps

1.  **Convert to CSV:**

      * *MarineC:* Use [QGIS](https://qgis.org/en/site/) to convert metadata to CSV.
      * *Brittany:* Use [libais](https://github.com/schwehr/libais) to parse raw messages.

2.  **Create Trajectories:**
    Run `csv2pkl.py` to load CSV data, filter by Region of Interest (ROI), and save trajectories keyed by MMSI into `.pkl` format.

    ```bash
    python data/csv2pkl.py [arguments]
    ```

3.  **Calculate AIS Mean:**
    **Crucial:** Before training the model, you must calculate the mean of the AIS "four-hot" vectors. This script is located in the data directory.

    ```bash
    python data/calculate_AIS_mean.py
    ```

-----

## Usage

The analysis pipeline is sequential. You must train the model before running detection.

### 1\. Training the Embedding Layer

Train the VRNN embedding layer on your training set.

```bash
python geotracknet.py \
  --mode=train \
  --dataset_dir=./data \
  --trainingset_name=ct_2017010203_10_20/ct_2017010203_10_20_train.pkl \
  --testset_name=ct_2017010203_10_20/ct_2017010203_10_20_valid.pkl \
  --lat_min=47.5 --lat_max=49.5 \
  --lon_min=-7.0 --lon_max=-4.0 \
  --latent_size=100 \
  --batch_size=32 \
  --num_samples=16 \
  --learning_rate=0.0003
```

### 2\. Log Probability Calculation

After training, calculate the log probabilities. This can be done globally or locally.

**Option A: Save Log Probability (`save_logprob`)**
Calculates and saves log-probs to a `.pkl` file to avoid re-computation.

```bash
python geotracknet.py \
  --mode=save_logprob \
  --dataset_dir=./data \
  --trainingset_name=ct_2017010203_10_20/ct_2017010203_10_20_train.pkl \
  --testset_name=ct_2017010203_10_20/ct_2017010203_10_20_valid.pkl \
  --lat_min=47.5 --lat_max=49.5 \
  --lon_min=-7.0 --lon_max=-4.0 \
  --latent_size=100 \
  --batch_size=32

```
```bash
python geotracknet.py \
  --mode=save_logprob \
  --dataset_dir=./data \
  --trainingset_name=ct_2017010203_10_20/ct_2017010203_10_20_train.pkl \
  --testset_name=ct_2017010203_10_20/ct_2017010203_10_20_test.pkl \
  --lat_min=47.5 --lat_max=49.5 \
  --lon_min=-7.0 --lon_max=-4.0 \
  --latent_size=100 \
  --batch_size=32
```

**Option B: Local Log Probability (`local_logprob`)**
Divides the ROI into cells and generates a log probability map.

```bash
python geotracknet.py --mode=local_logprob [same arguments as above...]
```

### 3\. Anomaly Detection

Finally, run the *a contrario* detection to identify abnormal behaviors.

```bash
python geotracknet.py \
  --mode=contrario_detection \
  --dataset_dir=./data \
  --trainingset_name=ct_2017010203_10_20/ct_2017010203_10_20_train.pkl \
  --testset_name=ct_2017010203_10_20/ct_2017010203_10_20_test.pkl \
  --lat_min=47.5 --lat_max=49.5 \
  --lon_min=-7.0 --lon_max=-4.0 \
  --contrario_eps=1e-10 \
  --latent_size=100
```

-----

## Troubleshooting

### protobuf compatibility issue

**Symptom:** Error related to protobuf versions (common with TF 1.x).
**Solution:** Downgrade protobuf to a compatible version.

```bash
pip uninstall protobuf
pip install protobuf==3.20.1
```

### cudart64\_100.dll not found

**Symptom:** Warning about missing CUDA DLLs.
**Solution:** This indicates a GPU is not detected. If you do not have a dedicated NVIDIA GPU, **you can safely ignore this**. The code will fall back to CPU execution.

-----

## Acknowledgments

We would like to extend our gratitude to:

  * **MarineCadastre, CLS, and Erwan Guegueniat** for data provision.
  * **Kurt Schwehr, The TensorFlow Team, QGIS, and OpenStreetmap** for open-source tools.
  * **Jetze Schuurmans** for assistance in the Python 2 to Python 3 migration.

-----

### Contact

For questions, bugs, or feature requests, please open a GitHub Issue.