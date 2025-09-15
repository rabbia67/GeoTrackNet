# api.py

import os
import pickle
import glob
from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np

# Import the configuration from your project. This is crucial for getting
# the coordinate boundaries (lat_min, lon_min, etc.).
from flags_config import config

# --- Constants for 3D Scene ---
# These should match the constants in your app.js for scaling
WORLD_SIZE = 1000
TRAJECTORY_HEIGHT = 2.0

app = Flask(__name__)
CORS(app) # Allow cross-origin requests

# --- Global cache for the processed data ---
# This avoids reloading and reprocessing files on every single request.
PROCESSED_DATA_CACHE = None

def find_latest_results_file(pattern):
    """Finds the most recently modified file matching a pattern."""
    try:
        # Construct search path within the results directory
        search_dir = os.path.join("results", os.path.basename(os.path.dirname(config.trainingset_path)))
        search_pattern = os.path.join(search_dir, "**", pattern) # Use ** for recursive search
        
        list_of_files = glob.glob(search_pattern, recursive=True)
        if not list_of_files:
            raise FileNotFoundError(f"No files found for pattern: {pattern} in {search_dir}")
        
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"Found latest results file: {latest_file}")
        return latest_file
    except Exception as e:
        print(f"Error finding latest results file: {e}")
        return None

def load_and_process_data():
    """
    Loads the results from the GeoTrackNet pipeline and transforms them
    into a format suitable for the 3D frontend.
    """
    global PROCESSED_DATA_CACHE
    if PROCESSED_DATA_CACHE is not None:
        return PROCESSED_DATA_CACHE

    print("Loading and processing data for the first time...")
    
    # 1. Find and load the list of abnormal tracks
    abnormal_tracks_file = find_latest_results_file("List_abnormal_tracks-*.pkl")
    if not abnormal_tracks_file:
        raise FileNotFoundError("Could not find the abnormal tracks result file. Did you run the pipeline?")

    with open(abnormal_tracks_file, 'rb') as f:
        l_dict_anomaly = pickle.load(f, encoding='latin1')
    
    # Create a set of abnormal MMSIs for fast lookup
    abnormal_mmsi_set = {d['mmsi'] for d in l_dict_anomaly}
    print(f"Loaded {len(abnormal_mmsi_set)} abnormal track MMSIs.")

    # 2. Load the full test set to get ALL tracks (normal and abnormal)
    if not os.path.exists(config.testset_path):
        raise FileNotFoundError(f"Test set file not found: {config.testset_path}")
        
    with open(config.testset_path, 'rb') as f:
        vs_test = pickle.load(f, encoding='latin1')
    print(f"Loaded {len(vs_test)} total tracks from the test set.")

    # 3. Process and transform the data
    all_ships_data = []
    ship_id_counter = 1
    
    # Get coordinate ranges from the config for denormalization
    lat_range = config.lat_max - config.lat_min
    lon_range = config.lon_max - config.lon_min

    for mmsi, track_data in vs_test.items():
        is_abnormal = mmsi in abnormal_mmsi_set
        
        # The track data columns are: [LAT, LON, SOG, COG, ...]
        # We only need LAT (index 0) and LON (index 1)
        # These are normalized coordinates from 0.0 to 1.0
        normalized_coords = track_data[:, [0, 1]]

        waypoints = []
        for point in normalized_coords:
            # --- This is the CRUCIAL data transformation step ---
            # Denormalize lat/lon from [0,1] to their real values
            real_lat = point[0] * lat_range + config.lat_min
            real_lon = point[1] * lon_range + config.lon_min

            # Remap real lat/lon to the 3D world coordinates of your frontend
            # We will map the ROI to the center of the WORLD_SIZE
            # Assuming a simple linear mapping for this example
            pos_x = (real_lon - config.lon_min) / lon_range * WORLD_SIZE - (WORLD_SIZE / 2)
            pos_z = (real_lat - config.lat_min) / lat_range * WORLD_SIZE - (WORLD_SIZE / 2)

            waypoints.append({
                "position": {"x": pos_x, "y": TRAJECTORY_HEIGHT, "z": -pos_z}, # Note: Z is often inverted
                "timestamp": 0 # Timestamp is not used in your frontend animation logic
            })

        ship = {
            "id": ship_id_counter,
            "mmsi": str(mmsi),
            "name": f"Ship {mmsi}", # You can add real names if you have them
            "isAbnormal": is_abnormal,
            "status": "Abnormal" if is_abnormal else "Normal",
            "waypoints": waypoints
        }
        all_ships_data.append(ship)
        ship_id_counter += 1

    # Cache the result
    PROCESSED_DATA_CACHE = all_ships_data
    print("Data processing complete and cached.")
    return PROCESSED_DATA_CACHE


@app.route('/api/trajectories')
def get_trajectories():
    try:
        ship_trajectories = load_and_process_data()
        return jsonify(ship_trajectories)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats')
def get_stats():
    try:
        ship_trajectories = load_and_process_data()
        abnormal_count = sum(1 for ship in ship_trajectories if ship["isAbnormal"])
        normal_count = len(ship_trajectories) - abnormal_count
        
        stats = {
            "totalShips": len(ship_trajectories),
            "normalCount": normal_count,
            "abnormalCount": abnormal_count
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)