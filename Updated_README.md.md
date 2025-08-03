GeoTrackNet Project Guide
This guide provides a step-by-step walkthrough for setting up the environment and running the GeoTrackNet project, which is designed for detecting anomalies in vessel trajectories.

The instructions assume you are working in a command-line environment on a Windows system, as indicated by the file paths in your provided logs.

1. Prerequisites
Before you begin, ensure you have the following installed:

Python: The project uses Python. The recommended way to manage the environment is with Anaconda or Miniconda.

2. Environment Setup
It is highly recommended to create a dedicated Anaconda environment for this project to manage its specific dependencies.

Create the Conda Environment:
Open your terminal or Anaconda Prompt and run the following command to create a new environment named geotracknet with Python 3.7.

conda create -n geotracknet python=3.7

Activate the Environment:
Activate the newly created environment to ensure all subsequent commands are executed within it.

conda activate geotracknet

Install Required Libraries:
With the environment active, install the necessary Python packages using pip. The project's requirements.yml file likely specifies these, but the following are the core dependencies.

pip install tensorflow==1.15.0 numpy matplotlib scipy tqdm dm-sonnet==1.36

Note: The code is explicitly configured to use tensorflow.compat.v1, so installing TensorFlow 1.15.0 is crucial for compatibility.

3. Project File Structure
The elements of the code are organized as follows:

geotracknet.py                   # script to run the model (except the A contrario detection).
runners.py                       # graph construction code for training and evaluation.
bounds.py                        # code for computing each bound.
contrario_kde.py                 # script to run the A contrario detection.
contrario_utils.py
distribution_utils.py
nested_utils.py
utils.py
data
├── datasets.py                  # reader pipelines.
├── calculate_AIS_mean.py        # calculates the mean of the AIS "four-hot" vectors.
├── dataset_preprocessing.py     # preprocesses the AIS datasets.
└── csv2pkl.py                   # parse raw AIS messages from aivdm format to csv files.
└── csv2pkl.py                   # loads AIS data from *.csv files.
models
└── vrnn.py                      # VRNN implementation.
chkpt
└── ...                          # directory to keep checkpoints and summaries in.
results
└── ...                          # directory to save results to.

4. Datasets & Preprocessing
The project utilizes specific datasets, which require preprocessing.

Datasets:
MarineC dataset: Provided by MarineCadastre.gov, Bureau of Ocean Energy Management, and National Oceanic and Atmospheric Administration. Available at (https://marinecadastre.gov/ais/).

Brittany dataset: Provided by CLS-Collecte Localisation Satellites. A processed subset is provided in data/ct_2017010203_10_20.zip for reproducing the paper's results.

Preprocess the Data:
Converting to CSV:

MarineC dataset: Use QGIS (https://qgis.org/en/site/) to convert the original metadata format to CSV files.

Brittany dataset: Use libais (https://github.com/schwehr/libais) to parse raw AIS messages to CSV files.

Creating Trajectories: csv2pkl.py loads the data from CSV files, selects AIS messages in the pre-defined ROI, creates AIS trajectories (keyed by MMSI), and saves them in pickle format (.pkl).

Preprocessing steps: The data are processed as described in the paper by dataset_preprocessing.py.

5. Running the Project
The project is run in different modes. You need to run them sequentially to complete the full analysis.

Step 1: Training the Embedding Layer (train)
First, you must train the Embedding layer using the following command:

python geotracknet.py \
  --mode=train \
  --dataset_dir=./data \
  --trainingset_name=ct_2017010203_10_20/ct_2017010203_10_20_train.pkl \
  --testset_name=ct_2017010203_10_20/ct_2017010203_10_20_valid.pkl \
  --lat_min=47.5 \
  --lat_max=49.5 \
  --lon_min=-7.0 \
  --lon_max=-4.0 \
  --latent_size=100 \
  --batch_size=32 \
  --num_samples=16 \
  --learning_rate=0.0003

Step 2: Running Task-Specific Submodels
After the Embedding layer is trained, you can run the following submodels.

Save Log Probability (save_logprob)
To avoid recalculating log probabilities for each task, this mode calculates them once and saves the results as a .pkl file.

python geotracknet.py \
  --mode=save_logprob \
  --dataset_dir=./data \
  --trainingset_name=ct_2017010203_10_20/ct_2017010203_10_20_train.pkl \
  --testset_name=ct_2017010203_10_20/ct_2017010203_10_20_valid.pkl \
  --lat_min=47.5 \
  --lat_max=49.5 \
  --lon_min=-7.0 \
  --lon_max=-4.0 \
  --latent_size=100 \
  --batch_size=32 \
  --num_samples=16 \
  --learning_rate=0.0003

Local Log Probability (local_logprob)
This mode divides the ROI into small cells and saves the log probabilities of AIS messages in each cell, generating a log probability map.

python geotracknet.py \
  --mode=local_logprob \
  --dataset_dir=./data \
  --trainingset_name=ct_2017010203_10_20/ct_2017010203_10_20_train.pkl \
  --testset_name=ct_2017010203_10_20/ct_2017010203_10_20_valid.pkl \
  --lat_min=47.5 \
  --lat_max=49.5 \
  --lon_min=-7.0 \
  --lon_max=-4.0 \
  --latent_size=100 \
  --batch_size=32 \
  --num_samples=16 \
  --learning_rate=0.0003

Anomaly Detection (contrario_detection)
This mode uses the log probability map to detect abnormal vessel behaviors using an a contrario detection approach and plots the results.

python geotracknet.py \
  --mode=contrario_detection \
  --dataset_dir=./data \
  --trainingset_name=ct_2017010203_10_20/ct_2017010203_10_20_train.pkl \
  --testset_name=ct_2017010203_10_20/ct_2017010203_10_20_test.pkl \
  --lat_min=47.5 \
  --lat_max=49.5 \
  --lon_min=-7.0 \
  --lon_max=-4.0 \
  --contrario_eps=1e-10 \
  --latent_size=100 \
  --batch_size=32 \
  --num_samples=16 \
  --learning_rate=0.0003

6. Troubleshooting Common Issues
A) protobuf compatibility issue
Problem: You may encounter an error related to protobuf versions, often with TensorFlow 1.x. The installed protobuf version may be too new for the TensorFlow version.

Solution: To fix this, you need to first uninstall the current version of protobuf and then install a compatible version.

pip uninstall protobuf
pip install protobuf==3.20.1

B) cudart64_100.dll not found
Problem: You may see a warning about a missing cudart64_100.dll.

Solution: This is a TensorFlow warning indicating that a GPU is not being used. It is safe to ignore this warning if you are running the project on a machine without a dedicated CUDA-compatible GPU. The script will fall back to using the CPU.

7. Acknowledgements & Contact
Acknowledgement
We would like to thank MarineCadastre, CLS and Erwan Guegueniat, Kurt Schwehr, Tensorflow team, QGIS and OpenStreetmap for the data and the open-source codes.

We would also like to thank Jetze Schuurmans for helping convert the code from Python2 to Python3.

Contact
For any questions, please open an issue.