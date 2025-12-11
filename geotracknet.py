# coding: utf-8

# MIT License
#
# Copyright (c) 2018 Duong Nguyen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

"""
A script to run the task-specific blocks of GeoTrackNet.
The code is adapted from
https://github.com/tensorflow/models/tree/master/research/fivo
"""

import os
import glob
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Ensure TensorFlow 1.x compatibility
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.ndimage as ndimage
import pickle
from tqdm import tqdm
import logging
import math
import scipy.special
from scipy import stats
import csv
from datetime import datetime
import utils
import contrario_utils
import runners
from flags_config import config
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LAT_RANGE = config.lat_max - config.lat_min
LON_RANGE = config.lon_max - config.lon_min

FIG_DPI = 150
FIG_W = 960
FIG_H = int(FIG_W * LAT_RANGE / LON_RANGE)

LOGPROB_MEAN_MIN = -10.0
LOGPROB_STD_MAX = 5

## RUN TRAIN
#======================================
def plot_score_distribution(l_dict, save_path):
    scores = []
    for D in l_dict:
        # Calculate average log-likelihood for the track
        scores.append(np.mean(D["log_weights"]))

    plt.figure(figsize=(10, 6), dpi=FIG_DPI)
    sns.histplot(scores, kde=True, bins=50, color='blue', label='All Tracks')
    plt.title("Distribution of Track Log-Likelihoods")
    plt.xlabel("Log-Likelihood (Lower is more anomalous)")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


if config.mode == "train":
    if not os.path.exists(config.trainingset_path):
        logger.error(f"Training set path {config.trainingset_path} does not exist")
        raise FileNotFoundError(f"Training set path {config.trainingset_path} does not exist")
    logger.info(f"Training set path: {config.trainingset_path}")
    fh = logging.FileHandler(os.path.join(config.logdir, config.log_filename + ".log"))
    logger.addHandler(fh)
    runners.run_train(config)

else:
    if not os.path.exists(config.testset_path):
        logger.error(f"Test set path {config.testset_path} does not exist")
        raise FileNotFoundError(f"Test set path {config.testset_path} does not exist")
    with open(config.testset_path, "rb") as f:
        Vs_test = pickle.load(f)
    dataset_size = len(Vs_test)

## RUN TASK-SPECIFIC SUBMODEL
#======================================

step = None
# This block sets up the TensorFlow graph and runs the session for specific modes.
if config.mode in ["save_logprob", "traj_reconstruction"]:
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        inputs, targets, mmsis, time_starts, time_ends, lengths, model = runners.create_dataset_and_model(
            config, shuffle=False, repeat=False
        )

        if config.mode == "traj_reconstruction":
            config.missing_data = True

        track_sample, track_true, log_weights, ll_per_t, ll_acc, _, _, _ = runners.create_eval_graph(
            inputs, targets, lengths, model, config
        )
        saver = tf.train.Saver()
        with tf.train.SingularMonitoredSession() as sess:
            runners.wait_for_checkpoint(saver, sess, config.logdir)
            step = sess.run(global_step)
            
            # All post-training logic that uses the TensorFlow session must be inside this block.
            logger.info(f"Global step: {step}")
            
            # FIX: Use a cleaner and shorter naming convention for paths
            train_name_clean = os.path.basename(config.trainingset_name).replace('.pkl', '')
            test_name_clean = os.path.basename(config.testset_name).replace('.pkl', '')

            outputs_path = os.path.join(
                "results",
                os.path.basename(os.path.dirname(config.trainingset_path)),
                f"logprob-{train_name_clean}-{test_name_clean}-{config.latent_size}-step-{step}.pkl"
            )
            os.makedirs(os.path.dirname(outputs_path), exist_ok=True)
            
            save_dir = os.path.join(
                "results",
                os.path.basename(os.path.dirname(config.trainingset_path)),
                f"local_logprob-{train_name_clean}-{test_name_clean.replace('test', 'valid')}-{config.latent_size}-step-{step}"
            )
            os.makedirs(save_dir, exist_ok=True) # Ensure the main directory exists
            
            # ======================================
            if config.mode == "save_logprob":
                """ save_logprob
                Calculate and save log[p(x_t|h_t)] of each track in the test set.
                """
                l_dict = []
                for d_i in tqdm(range(math.ceil(dataset_size / config.batch_size))):
                    try:
                        inp, tar, mmsi, t_start, t_end, seq_len, log_weights_np, true_np, ll_t = sess.run(
                            [inputs, targets, mmsis, time_starts, time_ends, lengths, log_weights, track_true, ll_per_t]
                        )
                        for d_idx_inbatch in range(inp.shape[1]):
                            seq_len_d = seq_len[d_idx_inbatch]
                            nonzero_indices = np.nonzero(tar[:seq_len_d, d_idx_inbatch, :])[1]
                            if nonzero_indices.size % 4 != 0:
                                logger.warning(f"Invalid shape for seq at batch index {d_idx_inbatch}, size={nonzero_indices.size}, skipping")
                                continue
                            if nonzero_indices.size == 0:
                                logger.warning(f"No non-zero indices for seq at batch index {d_idx_inbatch}, skipping")
                                continue
                            D = {
                                "seq": nonzero_indices.reshape(-1, 4),
                                "t_start": t_start[d_idx_inbatch],
                                "t_end": t_end[d_idx_inbatch],
                                "mmsi": mmsi[d_idx_inbatch],
                                # CORRECT (Slice the Batch dimension, keep all Samples)
                                "log_weights": log_weights_np[:seq_len_d, d_idx_inbatch]
                            }
                            l_dict.append(D)
                    except Exception as e:
                        logger.error(f"Error processing batch {d_i}: {str(e)}")
                        continue
                with open(outputs_path, "wb") as f:
                    pickle.dump(l_dict, f)

                """ LL
                Plot the distribution of log[p(x_t|h_t)] of each track in the test set.
                """
                v_logprob = np.empty((0,))
                for D in tqdm(l_dict):
                    log_weights_np = D["log_weights"]
                    ll_t = np.mean(log_weights_np)
                    v_logprob = np.concatenate((v_logprob, [ll_t]))

                d_mean = np.mean(v_logprob)
                d_std = np.std(v_logprob)
                d_thresh = d_mean - 3 * d_std

                plt.figure(figsize=(1920 / FIG_DPI, 640 / FIG_DPI), dpi=FIG_DPI)
                plt.plot(v_logprob, 'o')
                plt.title(
                    f"Log likelihood {os.path.basename(config.testset_name)}, "
                    f"mean = {d_mean:.2f}, std = {d_std:.2f}, threshold = {d_thresh:.2f}"
                )
                plt.plot([0, len(v_logprob)], [d_thresh, d_thresh], 'r')
                plt.xlim([0, len(v_logprob)])
                fig_name = os.path.join(
                    "results",
                    os.path.basename(os.path.dirname(config.trainingset_path)),
                    f"logprob-{config.bound}-{train_name_clean}-{test_name_clean}-{config.latent_size}-"
                    f"ll_thresh{round(d_thresh, 2)}-missing_data-{config.missing_data}-step-{step}.png"
                )
                plt.savefig(fig_name, dpi=FIG_DPI)
                plt.close()


                # Call it
                dist_plot_path = os.path.join(save_dir, f"score_distribution_step-{step}.png")
                plot_score_distribution(l_dict, dist_plot_path)

# The following variables are now defined globally, after the step is determined.
# This logic is necessary for all non-training modes that require these paths.
else:
    # A single block to determine the checkpoint step and define paths for all subsequent modes.
    checkpoint_files = sorted(glob.glob(os.path.join(config.logdir, "*.index")))
    if not checkpoint_files:
        logger.error(f"No checkpoint files found in {config.logdir}")
        raise FileNotFoundError(f"No checkpoint files found in {config.logdir}")
    index_filename = checkpoint_files[-1]  # the latest step
    step = int(index_filename.split(".index")[0].split("ckpt-")[-1])

    logger.info(f"Global step: {step}")
    
    # FIX: Use a cleaner and shorter naming convention for paths
    train_name_clean = os.path.basename(config.trainingset_name).replace('.pkl', '')
    test_name_clean = os.path.basename(config.testset_name).replace('.pkl', '')
    
    outputs_path = os.path.join(
        "results",
        os.path.basename(os.path.dirname(config.trainingset_path)),
        f"logprob-{train_name_clean}-{test_name_clean}-{config.latent_size}-step-{step}.pkl"
    )
    os.makedirs(os.path.dirname(outputs_path), exist_ok=True)
    
    save_dir = os.path.join(
        "results",
        os.path.basename(os.path.dirname(config.trainingset_path)),
        f"local_logprob-{train_name_clean}-{test_name_clean.replace('test', 'valid')}-{config.latent_size}-step-{step}"
    )
    
    os.makedirs(save_dir, exist_ok=True)

    # All subsequent mode-specific blocks will now have access to 'outputs_path' and 'save_dir'.
    #======================================
    if config.mode == "local_logprob":
        """ LOCAL THRESHOLD
        The ROI is divided into small cells, in each cell, we calculate the mean and
        the std of log[p(x_t|h_t)].
        """
        # Init
        m_map_logprob_std = np.zeros(shape=(config.n_lat_cells, config.n_lon_cells))
        m_map_logprob_mean = np.zeros(shape=(config.n_lat_cells, config.n_lon_cells))
        m_map_density = np.zeros(shape=(config.n_lat_cells, config.n_lon_cells))
        Map_logprob = {f"{row},{col}": [] for row in range(config.n_lat_cells) for col in range(config.n_lon_cells)}

        # Load logprob
        if not os.path.exists(outputs_path):
            logger.error(f"Logprob file {outputs_path} does not exist")
            raise FileNotFoundError(f"Logprob file {outputs_path} does not exist")
        with open(outputs_path, "rb") as f:
            l_dict = pickle.load(f)

        logger.info("Calculating the logprob map...")
        for D in tqdm(l_dict):
            tmp = D["seq"]
            log_weights_np = D["log_weights"]
            for d_timestep in range(2 * 6, len(tmp)):
                try:
                    row = int(tmp[d_timestep, 0] * 0.01 / config.cell_lat_reso)
                    col = int((tmp[d_timestep, 1] - config.onehot_lat_bins) * 0.01 / config.cell_lon_reso)
                    if 0 <= row < config.n_lat_cells and 0 <= col < config.n_lon_cells:
                        Map_logprob[f"{row},{col}"].append(np.mean(log_weights_np[d_timestep, :]))
                    else:
                        logger.warning(f"Invalid cell indices: row={row}, col={col}")
                except Exception as e:
                    logger.warning(f"Error processing timestep {d_timestep}: {str(e)}")
                    continue

        # Remove outliers
        for row in range(config.n_lat_cells):
            for col in range(config.n_lon_cells):
                s_key = f"{row},{col}"
                Map_logprob[s_key] = utils.remove_gaussian_outlier(np.array(Map_logprob[s_key]))
                
                # FIX: Check the length of the array to avoid the ValueError
                if len(Map_logprob[s_key]) > 0:
                    m_map_logprob_mean[row, col] = np.mean(Map_logprob[s_key])
                    m_map_logprob_std[row, col] = np.std(Map_logprob[s_key])
                else:
                    m_map_logprob_mean[row, col] = 0
                    m_map_logprob_std[row, col] = 0
                    
                m_map_density[row, col] = len(Map_logprob[s_key])

        # Save to disk
        # os.makedirs(save_dir, exist_ok=True) # This is now handled before the if/elif blocks
        np.save(os.path.join(save_dir, f"map_density-{config.cell_lat_reso}-{config.cell_lon_reso}"), m_map_density)
        with open(os.path.join(save_dir, f"Map_logprob-{config.cell_lat_reso}-{config.cell_lon_reso}.pkl"), "wb") as f:
            pickle.dump(Map_logprob, f)

        # Show the map
        utils.show_logprob_map(
            m_map_logprob_mean, m_map_logprob_std, save_dir,
            logprob_mean_min=LOGPROB_MEAN_MIN,
            logprob_std_max=LOGPROB_STD_MAX,
            fig_w=FIG_W, fig_h=FIG_H
        )

    #======================================
    elif config.mode == "contrario_detection":
        """ CONTRARIO DETECTION
        Detect abnormal vessels' behavior using a contrario detection.
        An AIS message is considered as abnormal if it does not follow the learned 
        distribution. An AIS track is considered as abnormal if many of its messages
        are abnormal.
        """
        # Load the parameters of the distribution
        map_logprob_path = os.path.join(save_dir, f"Map_logprob-{config.cell_lat_reso}-{config.cell_lon_reso}.pkl")
        if not os.path.exists(map_logprob_path):
            logger.error(f"Map logprob file {map_logprob_path} does not exist")
            raise FileNotFoundError(f"Map logprob file {map_logprob_path} does not exist")
        with open(map_logprob_path, "rb") as f:
            Map_logprob = pickle.load(f)

        # Load the logprob
        if not os.path.exists(outputs_path):
            logger.error(f"Logprob file {outputs_path} does not exist")
            raise FileNotFoundError(f"Logprob file {outputs_path} does not exist")
        with open(outputs_path, "rb") as f:
            l_dict = pickle.load(f)

        # Helper function to extract lat/lon from sequence
        def extract_lat_lon_from_seq(seq, onehot_lat_bins, onehot_lon_bins):
            """Extract start and end lat/lon from sequence data"""
            if len(seq) == 0:
                return None, None, None, None
            
            # Get first and last positions
            start_lat = (seq[0, 0] * 0.01) + config.lat_min  # Convert back to degrees
            start_lon = ((seq[0, 1] - onehot_lat_bins) * 0.01) + config.lon_min
            
            end_lat = (seq[-1, 0] * 0.01) + config.lat_min
            end_lon = ((seq[-1, 1] - onehot_lat_bins) * 0.01) + config.lon_min
            
            return start_lat, start_lon, end_lat, end_lon

        l_dict_normal = []
        l_dict_anomaly = []
        n_error = 0
        
        for D in tqdm(l_dict):
            try:
                tmp = D["seq"]
                m_log_weights_np = D["log_weights"]
                v_A = np.zeros(len(tmp))
                for d_timestep in range(2 * 6, len(tmp)):
                    d_row = int(tmp[d_timestep, 0] * config.onehot_lat_reso / config.cell_lat_reso)
                    d_col = int((tmp[d_timestep, 1] - config.onehot_lat_bins) * config.onehot_lat_reso / config.cell_lon_reso)
                    d_logprob_t = np.mean(m_log_weights_np[d_timestep, :])
                    s_key = f"{d_row},{d_col}"
                    l_local_log_prod = Map_logprob.get(s_key, [])
                    if len(l_local_log_prod) < 2:
                        v_A[d_timestep] = 2
                    else:
                        kernel = stats.gaussian_kde(l_local_log_prod)
                        cdf = kernel.integrate_box_1d(-np.inf, d_logprob_t)
                        if cdf < 0.1:
                            v_A[d_timestep] = 1
                v_A = v_A[12:]
                v_anomalies = np.zeros(len(v_A))
                for d_i_4h in range(0, len(v_A) + 1 - 24):
                    v_A_4h = v_A[d_i_4h:d_i_4h + 24]
                    v_anomalies_i = contrario_utils.contrario_detection(v_A_4h, config.contrario_eps)
                    v_anomalies[d_i_4h:d_i_4h + 24][v_anomalies_i == 1] = 1

                if len(contrario_utils.nonzero_segments(v_anomalies)) > 0:
                    D["anomaly_idx"] = v_anomalies
                    l_dict_anomaly.append(D)
                else:
                    l_dict_normal.append(D)
            except Exception as e:
                logger.error(f"Error processing track {D.get('mmsi', 'unknown')}: {str(e)}")
                n_error += 1
                continue

        logger.info(f"Number of processed tracks: {len(l_dict)}")
        logger.info(f"Number of normal tracks: {len(l_dict_normal)}")
        logger.info(f"Number of abnormal tracks: {len(l_dict_anomaly)}")
        logger.info(f"Number of errors: {n_error}")

        # Save abnormal tracks to CSV
        n_anomalies = len(l_dict_anomaly)
        abnormal_csv_filename = os.path.join(save_dir, 
            f"abnormal_tracks-{train_name_clean}-{test_name_clean.replace('valid', 'test')}-"
            f"latent_size-{config.latent_size}-step-{step}-eps-{config.contrario_eps}-{n_anomalies}.csv"
        )

        with open(abnormal_csv_filename, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "MMSI", "Time_start", "Time_end", "Timestamp_start", "Timestamp_end",
                "Start_Latitude", "Start_Longitude", "End_Latitude", "End_Longitude"
            ])
            
            for D in l_dict_anomaly:
                start_lat, start_lon, end_lat, end_lon = extract_lat_lon_from_seq(
                    D["seq"], config.onehot_lat_bins, config.onehot_lon_bins
                )
                
                writer.writerow([
                    D["mmsi"],
                    datetime.utcfromtimestamp(D["t_start"]).strftime('%Y-%m-%d %H:%M:%SZ'),
                    datetime.utcfromtimestamp(D["t_end"]).strftime('%Y-%m-%d %H:%M:%SZ'),
                    D["t_start"], 
                    D["t_end"],
                    f"{start_lat:.6f}" if start_lat is not None else "",
                    f"{start_lon:.6f}" if start_lon is not None else "",
                    f"{end_lat:.6f}" if end_lat is not None else "",
                    f"{end_lon:.6f}" if end_lon is not None else ""
                ])

        # Save normal tracks to CSV
        n_normal = len(l_dict_normal)
        normal_csv_filename = os.path.join(save_dir, 
            f"normal_tracks-{train_name_clean}-{test_name_clean.replace('valid', 'test')}-"
            f"latent_size-{config.latent_size}-step-{step}-eps-{config.contrario_eps}-{n_normal}.csv"
        )

        with open(normal_csv_filename, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "MMSI", "Time_start", "Time_end", "Timestamp_start", "Timestamp_end",
                "Start_Latitude", "Start_Longitude", "End_Latitude", "End_Longitude"
            ])
            
            for D in l_dict_normal:
                start_lat, start_lon, end_lat, end_lon = extract_lat_lon_from_seq(
                    D["seq"], config.onehot_lat_bins, config.onehot_lon_bins
                )
                
                writer.writerow([
                    D["mmsi"],
                    datetime.utcfromtimestamp(D["t_start"]).strftime('%Y-%m-%d %H:%M:%SZ'),
                    datetime.utcfromtimestamp(D["t_end"]).strftime('%Y-%m-%d %H:%M:%SZ'),
                    D["t_start"], 
                    D["t_end"],
                    f"{start_lat:.6f}" if start_lat is not None else "",
                    f"{start_lon:.6f}" if start_lon is not None else "",
                    f"{end_lat:.6f}" if end_lat is not None else "",
                    f"{end_lon:.6f}" if end_lon is not None else ""
                ])

        logger.info(f"Saved {n_anomalies} abnormal tracks to: {abnormal_csv_filename}")
        logger.info(f"Saved {n_normal} normal tracks to: {normal_csv_filename}")

        interactive_map_name = os.path.join(save_dir, f"interactive_map_step-{step}.html")
        utils.save_interactive_map(
            l_dict_anomaly,
            l_dict_normal,
            interactive_map_name,
            config.lat_min, config.lat_max, config.lon_min, config.lon_max,
            config.onehot_lat_bins, config.onehot_lon_bins
        )

        # Create a subfolder for profiles
        profile_dir = os.path.join(save_dir, "track_profiles")
        os.makedirs(profile_dir, exist_ok=True)

        # Plot top 10 anomalies (or just the first 10 found)
        num_profiles_to_plot = min(10, len(l_dict_anomaly))
        logger.info(f"Plotting {num_profiles_to_plot} track profiles...")

        profiles_created = 0
        for i, D in enumerate(l_dict_anomaly[:num_profiles_to_plot]):
            try:
                profile_name = os.path.join(profile_dir, f"profile_mmsi_{D['mmsi']}_{i}.png")
                success = utils.plot_track_profile(
                    D,
                    profile_name,
                    config.onehot_lat_bins,
                    config.onehot_lon_bins,
                    config.onehot_sog_bins,
                    config.lat_min,
                    config.lat_max,
                    config.lon_min,
                    config.lon_max
                )
                if success:
                    profiles_created += 1
            except Exception as e:
                logger.error(f"Failed to create profile for track {i} (MMSI: {D.get('mmsi', 'unknown')}): {str(e)}")
                continue

        logger.info(f"Successfully saved {profiles_created}/{num_profiles_to_plot} track profiles to {profile_dir}")

        # Save to pickle files as well (keeping original functionality)
        save_filename = (
            f"List_abnormal_tracks-{train_name_clean}-{train_name_clean}-"
            f"{config.latent_size}-missing_data-{config.missing_data}-step-{step}.pkl"
        )
        save_pkl_filename = os.path.join(save_dir, save_filename)
        
        os.makedirs(os.path.dirname(save_pkl_filename), exist_ok=True)
        
        with open(save_pkl_filename, "wb") as f:
            pickle.dump(l_dict_anomaly, f)

        # Also save normal tracks to pickle
        normal_pkl_filename = save_pkl_filename.replace("abnormal", "normal")
        with open(normal_pkl_filename, "wb") as f:
            pickle.dump(l_dict_normal, f)

        # Plot
        if not os.path.exists(config.trainingset_path):
            logger.error(f"Training set path {config.trainingset_path} does not exist")
            raise FileNotFoundError(f"Training set path {config.trainingset_path} does not exist")
        with open(config.trainingset_path, "rb") as f:
            Vs_train = pickle.load(f)
        with open(config.testset_path, "rb") as f:
            Vs_test = pickle.load(f)

        save_filename = (
            f"Abnormal_tracks-{train_name_clean}-{test_name_clean.replace('valid', 'test')}-"
            f"latent_size-{config.latent_size}-step-{step}-eps-{config.contrario_eps}-{n_anomalies}.png"
        )

        # Plot abnormal tracks with training set as background
        utils.plot_abnormal_tracks(
            Vs_train, l_dict_anomaly,
            os.path.join(save_dir, save_filename),
            config.lat_min, config.lat_max, config.lon_min, config.lon_max,
            config.onehot_lat_bins, config.onehot_lon_bins,
            background_cmap="Blues",
            fig_w=FIG_W, fig_h=FIG_H
        )
        plt.close()

        # Plot abnormal tracks with test set as background
        utils.plot_abnormal_tracks(
            Vs_test, l_dict_anomaly,
            os.path.join(save_dir, save_filename.replace("Abnormal_tracks", "Abnormal_tracks2")),
            config.lat_min, config.lat_max, config.lon_min, config.lon_max,
            config.onehot_lat_bins, config.onehot_lon_bins,
            background_cmap="Greens",
            fig_w=FIG_W, fig_h=FIG_H
        )
        plt.close()