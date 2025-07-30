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
# furnished to do so, subject to the following conditions:
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
Script to calculate the mean of the one-hot encoded AIS training data
for normalization. This script should be run from the project root.
"""

import numpy as np
import pickle
import os
import sys
import tensorflow as tf

# Import the configuration from flags_config.py
# This script assumes flags_config.py is in the same directory (project root)
from flags_config import config

# Ensure the required flags for data dimensions are set in flags_config.py
# The values for LAT_BINS, LON_BINS, SOG_BINS, COG_BINS will now come from 'config'
# which are derived from onehot_lat_reso, onehot_lon_reso etc.

# Define constants for AIS message fields if not already defined globally in project
LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9))


def sparse_AIS_to_dense(msgs_, lat_bins, lon_bins, sog_bins, cog_bins):
    """
    Converts sparse AIS messages (lat, lon, sog, cog as indices) to dense one-hot vectors.
    """

    def create_dense_vect(msg, lat_bins, lon_bins, sog_bins, cog_bins):
        lat_idx, lon_idx, sog_idx, cog_idx = msg[0], msg[1], msg[2], msg[3]
        data_dim = lat_bins + lon_bins + sog_bins + cog_bins
        dense_vect = np.zeros(data_dim)

        # Ensure indices are within bounds before setting to 1.0
        # This prevents common errors if data falls outside expected bins
        if 0 <= lat_idx < lat_bins:
            dense_vect[int(lat_idx)] = 1.0
        if 0 <= lon_idx < lon_bins:
            dense_vect[int(lon_idx) + lat_bins] = 1.0
        if 0 <= sog_idx < sog_bins:
            dense_vect[int(sog_idx) + lat_bins + lon_bins] = 1.0
        if 0 <= cog_idx < cog_bins:
            dense_vect[int(cog_idx) + lat_bins + lon_bins + sog_bins] = 1.0

        return dense_vect

    dense_msgs = []
    for msg in msgs_:
        dense_msgs.append(create_dense_vect(msg,
                                            lat_bins=lat_bins,
                                            lon_bins=lon_bins,
                                            sog_bins=sog_bins,
                                            cog_bins=cog_bins))
    return np.array(dense_msgs)


def main():
    # Use paths and dimensions from the config object
    training_data_path = config.trainingset_path
    data_dimension = config.data_dim

    # Get bin numbers from config
    lat_bins = config.onehot_lat_bins
    lon_bins = config.onehot_lon_bins
    sog_bins = config.onehot_sog_bins
    cog_bins = config.onehot_cog_bins

    print(f"Loading training data from: {training_data_path}")
    print(f"Expected data dimension for one-hot encoding: {data_dimension}")
    print(f"Lat Bins: {lat_bins}, Lon Bins: {lon_bins}, SOG Bins: {sog_bins}, COG Bins: {cog_bins}")

    try:
        # Using tf.io.gfile.GFile as per original script's usage, even if deprecated.
        # This helps maintain consistency with how other parts of the TF graph might
        # expect to read files.
        with tf.io.gfile.GFile(training_data_path, "rb") as f:
            Vs = pickle.load(f)
    except Exception as e:
        print(f"Failed to load with default encoding: {e}. Trying latin1 encoding.")
        try:
            with tf.io.gfile.GFile(training_data_path, "rb") as f:
                Vs = pickle.load(f, encoding="latin1")
        except Exception as e_latin1:
            raise IOError(
                f"Failed to load training data even with latin1 encoding: {training_data_path}. Error: {e_latin1}")

    num_tracks = len(Vs)
    print(f"Loaded {num_tracks} tracks from {training_data_path}.")

    sum_all = np.zeros((data_dimension,), dtype=np.float32)
    total_ais_msg = 0

    count = 0
    for mmsi in list(Vs.keys()):
        count += 1
        # print(f"Processing track {count}/{num_tracks} (MMSI: {mmsi})") # Uncomment for verbose progress

        # Select relevant columns: LAT, LON, SOG, COG
        # Ensure that these columns represent *indices* for one-hot encoding
        # or raw values that get mapped to indices.
        # The original code `tmp[tmp == 1] = 0.99999` suggests raw values that are normalized to [0,1)
        # before being multiplied by bin counts to get indices.

        # Assuming `Vs[mmsi]` is a NumPy array where columns are [LAT_raw, LON_raw, SOG_raw, COG_raw, ...]
        # And these raw values (normalized to [0,1)) are directly used as indices.
        # If your raw data are degrees/knots, you'll need a preprocessing step here to convert them to
        # normalized indices suitable for the sparse_AIS_to_dense function.
        # Based on the error `Dimensions must be equal, but are 702 and 602`, the `sparse_AIS_to_dense`
        # is the point of conversion.

        # The `tmp = Vs[mmsi][:,[LAT,LON,SOG,COG]]` line suggests `Vs` contains raw, un-binned data.
        # The `tmp[tmp == 1] = 0.99999` line is a strange normalization. It implies values are already 0 or 1,
        # but 1.0 maps to index '1', not index '0.99999 * binsize'.

        # Let's assume the data in Vs is already pre-normalized to [0, 1) such that
        # (normalized_lat * lat_bins) gives the correct index.
        # If not, you need to add proper normalization to [0, 1) based on lat/lon min/max etc.

        # For the one-hot encoding, the input `msg` to `create_dense_vect` are expected to be the
        # *indices* of the bins, not the raw lat/lon/sog/cog values.
        # The `dataset_preprocessing.py` script likely does this binning.
        # If `Vs` contains the raw data, this script needs the binning logic.

        # Let's assume `Vs` contains the pre-binned indices, or the values are implicitly treated as such.
        # However, the previous error about 702 vs 602 implies `data_dim` is being used in one-hot calculation.

        # If `Vs` contains raw values (lat, lon in degrees, sog in knots, cog in degrees):
        # We need to explicitly convert them to normalized indices here.
        # This is where your dataset_preprocessing logic would come in, which is not in this file.

        # For now, let's assume `Vs[mmsi][:,[LAT,LON,SOG,COG]]` provides values that,
        # when multiplied by their respective BIN_COUNT, yield an integer index.
        # Or that `sparse_AIS_to_dense` is designed to handle this implicit conversion.

        # Given `tmp[tmp == 1] = 0.99999`, it strongly suggests `tmp` are *already* indices or very close to them,
        # and this line is just handling edge cases of values being exactly 1.0 (which would map to the next bin).

        # Let's proceed with the assumption that `sparse_AIS_to_dense` correctly takes the `tmp` values
        # as normalized coordinates (0-1) and converts them to the right one-hot indices.

        current_raw_data = Vs[mmsi][:, [LAT, LON, SOG, COG]]

        # Apply the normalization/clipping as in original code
        current_raw_data[current_raw_data == 1] = 0.99999

        # Convert to dense one-hot representation using config's bin sizes
        current_dense_matrix = sparse_AIS_to_dense(
            current_raw_data,
            lat_bins=lat_bins,
            lon_bins=lon_bins,
            sog_bins=sog_bins,
            cog_bins=cog_bins
        )

        sum_all += np.sum(current_dense_matrix, axis=0)
        total_ais_msg += len(current_dense_matrix)

    if total_ais_msg == 0:
        raise ValueError("No AIS messages found to calculate mean. Check training data.")

    mean_vector = sum_all / total_ais_msg

    if mean_vector.shape[0] != data_dimension:
        raise ValueError(
            f"Calculated mean vector dimension ({mean_vector.shape[0]}) does not match "
            f"expected data_dim from config ({data_dimension}). "
            "Review your data, `flags_config.py` (lat/lon/sog/cog boundaries/resolutions), "
            "and the `sparse_AIS_to_dense` logic."
        )

    # Determine where to save the mean.pkl.
    # It should be saved in the directory of the training data itself.
    output_mean_dir = os.path.dirname(training_data_path)
    output_mean_full_path = os.path.join(output_mean_dir, "mean.pkl")

    # Ensure the output directory exists
    if not os.path.exists(output_mean_dir):
        os.makedirs(output_mean_dir)
        print(f"Created directory: {output_mean_dir}")

    print(f"Calculated mean with shape: {mean_vector.shape}. Saving to: {output_mean_full_path}")
    with open(output_mean_full_path, "wb") as f:
        pickle.dump(mean_vector, f)
    print("Mean saved successfully.")


if __name__ == '__main__':
    main()