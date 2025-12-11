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
Utils for MultitaskAIS. 
"""



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from scipy import interpolate
import scipy.ndimage as ndimage
from math import radians, cos, sin, asin, sqrt
import sys
sys.path.append('..')
sys.path.append('Data')
#import shapefile
import time
from pyproj import Geod
geod = Geod(ellps='WGS84')
#import dataset
import folium

AVG_EARTH_RADIUS = 6378.137  # in km
SPEED_MAX = 30 # knot
FIG_DPI = 150

LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9))

def trackOutlier(A):
    """
    Koyak algorithm to perform outlier identification
    Our approach to outlier detection is to begin by evaluating the expression
    “observation r is anomalous with respect to observation s ” with respect to
    every pair of measurements in a track. We address anomaly criteria below; 
    assume for now that a criterion has been adopted and that the anomaly 
    relationship is symmetric. More precisely, let a(r,s) = 1 if r and s are
    anomalous and a(r,s) = 0 otherwise; symmetry implies that a(r,s) = a(s,r). 
    If a(r,s) = 1 either one or both of observations are potential outliers, 
    but which of the two should be treated as such cannot be resolved using 
    this information alone.
    Let A denote the matrix of anomaly indicators a(r, s) and let b denote 
    the vector of its row sums. Suppose that observation r is an outlier and 
    that is the only one present in the track. Because we expect it to be 
    anomalous with respect to many if not all of the other observations b(r) 
    should be large, while b(s) = 1 for all s ≠ r . Similarly, if there are 
    multiple outliers the values of b(r) should be large for those observations
    and small for the non-outliers. 
    Source: "Predicting vessel trajectories from AIS data using R", Brian L 
    Young, 2017
    INPUT:
        A       : (nxn) symmatic matrix of anomaly indicators
    OUTPUT:
        o       : n-vector outlier indicators
    
    # FOR TEST
    A = np.zeros((5,5))
    idx = np.array([[0,2],[1,2],[1,3],[0,3],[2,4],[3,4]])
    A[idx[:,0], idx[:,1]] = 1
    A[idx[:,1], idx[:,0]] = 1    sampling_track = np.empty((0, 9))
    for t in range(int(v[0,TIMESTAMP]), int(v[-1,TIMESTAMP]), 300): # 5 min
        tmp = utils.interpolate(t,v)
        if tmp is not None:
            sampling_track = np.vstack([sampling_track, tmp])
        else:
            sampling_track = None
            break
    """
    assert (A.transpose() == A).all(), "A must be a symatric matrix"
    assert ((A==0) | (A==1)).all(), "A must be a binary matrix"
    # Initialization
    n = A.shape[0]
    b = np.sum(A, axis = 1)
    o = np.zeros(n)
    while(np.max(b) > 0):
        r = np.argmax(b)
        o[r] = 1
        b[r] = 0
        for j in range(n):
            if (o[j] == 0):
                b[j] -= A[r,j]
    return o.astype(bool)
    
#===============================================================================
#===============================================================================
def detectOutlier(track, speed_max = SPEED_MAX):
    """
    removeOutlier() removes anomalus AIS messages from AIS track.
    An AIS message is considered as beging anomalous if the speed is
    infeasible (> speed_max). There are two types of anomalous messages:
        - The reported speed is infeasible
        - The calculated speed (distance/time) is infeasible
    
    INPUT:
        track       : a (nxd) matrix. Each row is an AIS message. The structure 
                      must follow: [Timestamp, Lat, Lon, Speed]
        speed_max   : knot
    OUTPUT:
        o           : n-vector outlier indicators
    """
    # Remove anomalous reported speed
    o_report = track[:,3] > speed_max # Speed in track is in knot
    if o_report.all():
        return o_report, None
    track = track[np.invert(o_report)]
    # Calculate speed base on (lon, lat) and time
    
    N = track.shape[0]
    # Anomoly indicator matrix
    A = np.zeros(shape = (N,N))
    
    # Anomalous calculated-speed
    for i in range(1,5):
        # the ith diagonal
        _, _, d = geod.inv(track[:N-i,2],track[:N-i,1],
                           track[i:,2],track[i:,1])
        delta_t = track[i:,0] - track[:N-i,0].astype(np.float)  
        cond = np.logical_and(delta_t > 2,d/delta_t > (speed_max*0.514444))
        abnormal_idx = np.nonzero(cond)[0]
        A[abnormal_idx, abnormal_idx + i] = 1
        A[abnormal_idx + i, abnormal_idx] = 1    

    o_calcul = trackOutlier(A)
            
    return o_report, o_calcul
    

#===============================================================================
#===============================================================================   
# Creating shape file
def createShapefile(shp_fname, Vs):
    """
    Creating AIS shape files
    INPUT:
        shp_fname    : name of the shapefile
        Vs          : AIS data, each element of the dictionary is an AIS track
                      whose structure is:
                      [Timestamp, MMSI, Lat, Lon, SOG, COG, Heading, ROT, NAV_STT]
    """
    shp = shapefile.Writer(shapefile.POINT)
    shp.field('MMSI', 'N', 10)
    shp.field('TIMESTAMP', 'N', 12)
    shp.field('DATETIME', 'C', 20)
    shp.field('LAT','N',10,5)
    shp.field('LON','N',10,5)
    shp.field('SOG','N', 10,5)
    shp.field('COG', 'N', 10,5)
    shp.field('HEADING', 'N', 10,5)
    shp.field('ROT', 'N', 5)
    shp.field('NAV_STT', 'N', 2)
    for mmsi in list(Vs.keys()):
        for p in Vs[mmsi]:
            shp.point(p[LON],p[LAT])
            shp.record(p[MMSI],
                       p[TIMESTAMP],
                       time.strftime('%H:%M:%S %d-%m-%Y', time.gmtime(p[TIMESTAMP])),
                       p[LAT],
                       p[LON],
                       p[SOG],
                       p[COG],
                       p[HEADING],
                       p[ROT],
                       p[NAV_STT])
    shp.save(shp_fname)
    
#===============================================================================
#===============================================================================
def interpolate(t, track):
    """
    Interpolating the AIS message of vessel at a specific "t".
    INPUT:
        - t : 
        - track     : AIS track, whose structure is
                     [LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI]
    OUTPUT:
        - [LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI]
                        
    """
    
    before_p = np.nonzero(t >= track[:,TIMESTAMP])[0]
    after_p = np.nonzero(t < track[:,TIMESTAMP])[0]
   
    if (len(before_p) > 0) and (len(after_p) > 0):
        apos = after_p[0]
        bpos = before_p[-1]    
        # Interpolation
        dt_full = float(track[apos,TIMESTAMP] - track[bpos,TIMESTAMP])
        if (abs(dt_full) > 2*3600):
            return None
        dt_interp = float(t - track[bpos,TIMESTAMP])
        try:
            az, _, dist = geod.inv(track[bpos,LON],
                                   track[bpos,LAT],
                                   track[apos,LON],
                                   track[apos,LAT])
            dist_interp = dist*(dt_interp/dt_full)
            lon_interp, lat_interp, _ = geod.fwd(track[bpos,LON], track[bpos,LAT],
                                               az, dist_interp)
            speed_interp = (track[apos,SOG] - track[bpos,SOG])*(dt_interp/dt_full) + track[bpos,SOG]
            course_interp = (track[apos,COG] - track[bpos,COG] )*(dt_interp/dt_full) + track[bpos,COG]
            heading_interp = (track[apos,HEADING] - track[bpos,HEADING])*(dt_interp/dt_full) + track[bpos,HEADING]  
            rot_interp = (track[apos,ROT] - track[bpos,ROT])*(dt_interp/dt_full) + track[bpos,ROT]
            if dt_interp > (dt_full/2):
                nav_interp = track[apos,NAV_STT]
            else:
                nav_interp = track[bpos,NAV_STT]                             
        except:
            return None
        return np.array([lat_interp, lon_interp,
                         speed_interp, course_interp, 
                         heading_interp, rot_interp, 
                         nav_interp,t,
                         track[0,MMSI]])
    else:
        return None

#===============================================================================
#===============================================================================
def remove_gaussian_outlier(v_data,quantile=1.64):
    """
    Remove outliers
    INPUT:
        v_data      : a 1-D array
        quantile    : 
    OUTPUT:
        v_filtered  : filtered array
    """
    d_mean = np.mean(v_data)
    d_std = np.std(v_data)
    idx_normal = np.where(np.abs(v_data-d_mean)<=quantile*d_std)[0] #90%
    return v_data[idx_normal]    
    
#===============================================================================
#===============================================================================
def gaussian_filter_with_nan(U,sigma):
    """
    Apply Gaussian filter when the data contain NaN
    INPUT:
        U           : a 2-D array (matrix)
        sigma       : std for the Gaussian kernel
    OUTPUT:
        Z           : filtered matrix
    """
    V=U.copy()
    V[np.isnan(U)]=0
    VV= ndimage.gaussian_filter(V,sigma=sigma)

    W=0*U.copy()+1
    W[np.isnan(U)]=0
    WW= ndimage.gaussian_filter(W,sigma=sigma)
    Z=VV/WW
    return(Z)

#===============================================================================
#===============================================================================
def show_logprob_map(m_map_logprob_mean, m_map_logprob_std, save_dir, 
                     logprob_mean_min = -10.0, logprob_std_max = 5.0,
                     d_scale = 10, inter_method = "hanning",
                     fig_w = 960, fig_h = 960,
                    ):
    """
    Show the map of the mean and the std of the logprob in each cell.
    INPUT:
        m_map_logprob_mean   : a 2-D array (matrix)
        m_map_logprob_std    : a 2-D array (matrix)
        save_dir             : directory to save the images
    """    
    # Truncate
    m_map_logprob_mean[m_map_logprob_mean<logprob_mean_min] = logprob_mean_min
    m_map_logprob_std[m_map_logprob_std>logprob_std_max] = logprob_std_max
    
    # Improve the resolution
    n_rows, n_cols = m_map_logprob_mean.shape
    m_mean = np.zeros((n_rows*d_scale,n_cols*d_scale))
    m_std = np.zeros((n_rows*d_scale,n_cols*d_scale))
    for i_row in range(m_map_logprob_mean.shape[0]):
        for i_col in range(m_map_logprob_mean.shape[1]):
            m_mean[d_scale*i_row:d_scale*(i_row+1),d_scale*i_col:d_scale*(i_col+1)] = m_map_logprob_mean[i_row,i_col]
            m_std[d_scale*i_row:d_scale*(i_row+1),d_scale*i_col:d_scale*(i_col+1)] = m_map_logprob_std[i_row,i_col]

    # Gaussian filter (with NaN)
    m_nan_idx = np.isnan(m_mean)
    m_mean = gaussian_filter_with_nan(m_mean, sigma=4.0)
    m_mean[m_nan_idx] = np.nan
    m_std = gaussian_filter_with_nan(m_std, sigma=4.0)
    m_nan_idx = np.isnan(m_std)
    m_std[m_nan_idx] = np.nan

    plt.figure(figsize=(fig_w/FIG_DPI, fig_h/FIG_DPI), dpi=FIG_DPI)
    # plt.subplot(1,2,1)
    im = plt.imshow(np.flipud(m_mean),interpolation=inter_method)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"logprob_mean_map.png"))
    plt.close()

    plt.figure(figsize=(fig_w/FIG_DPI, fig_h/FIG_DPI), dpi=FIG_DPI)
    # plt.subplot(1,2,2)
    im = plt.imshow(np.flipud(m_std),interpolation=inter_method)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"logprob_std_map.png"))
    plt.close()
    
#===============================================================================
#===============================================================================
def plot_abnormal_tracks(Vs_background,l_dict_anomaly,
                         filepath,
                         lat_min,lat_max,lon_min,lon_max,
                         onehot_lat_bins,onehot_lon_bins,
                         background_cmap = "Blues",
                         anomaly_cmap = "autumn",
                         l_coastline_poly = None,
                         fig_w = 960, fig_h = 960,
                         fig_dpi = 150,
                        ):
    plt.figure(figsize=(fig_w/FIG_DPI, fig_h/FIG_DPI), dpi=FIG_DPI)
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    ## Plot background
    cmap = plt.cm.get_cmap(background_cmap)
    l_keys = list(Vs_background.keys())
    N = len(Vs_background)
    for d_i in range(N):
        key = l_keys[d_i]
        c = cmap(float(d_i)/(N-1))
        tmp = Vs_background[key]
        v_lat = tmp[:,0]*lat_range + lat_min
        v_lon = tmp[:,1]*lon_range + lon_min
        plt.plot(v_lon,v_lat,color=c,linewidth=0.8)
    plt.xlim([lon_min,lon_max])
    plt.ylim([lat_min,lat_max])
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout() 

    ## Coastlines
    try:
        for point in l_coastline_poly:
            poly = np.array(point)
            plt.plot(poly[:,0],poly[:,1],color="k",linewidth=0.8)
    except:
        pass
    
    ## Plot abnormal tracks
    cmap_anomaly = plt.cm.get_cmap(anomaly_cmap)
    N_anomaly = len(l_dict_anomaly)
    d_i = 0
    for D in l_dict_anomaly:
        try:
            c = cmap_anomaly(float(d_i)/(N_anomaly-1))
        except:
            c = 'r'
        d_i += 1
        tmp = D["seq"]
        m_log_weights_np = D["log_weights"]
        tmp = tmp[12:]
        v_lat = (tmp[:,0]/float(onehot_lat_bins))*lat_range + lat_min
        v_lon = ((tmp[:,1]-onehot_lat_bins)/float(onehot_lon_bins))*lon_range + lon_min
        plt.plot(v_lon,v_lat,color=c,linewidth=1.2) 
    
    plt.savefig(filepath,dpi = fig_dpi)


###################################################################################################
def save_interactive_map(l_dict_anomaly, l_dict_normal, save_path,
                         lat_min, lat_max, lon_min, lon_max,
                         onehot_lat_bins, onehot_lon_bins, max_normal_tracks=200):
    """
    Creates an interactive HTML map with abnormal (Red) and normal (Blue) tracks.
    """
    # Center map on ROI
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles='OpenStreetMap')

    def decode_seq(seq):
        # Convert bin indices back to lat/lon
        lats = (seq[:, 0] * 0.01) + lat_min
        lons = ((seq[:, 1] - onehot_lat_bins) * 0.01) + lon_min
        return list(zip(lats, lons))

    # Plot Normal Tracks (Limit number to prevent browser lag)
    print(f"Plotting {min(len(l_dict_normal), max_normal_tracks)} normal tracks...")
    for i, D in enumerate(l_dict_normal):
        if i >= max_normal_tracks: break
        points = decode_seq(D["seq"])
        folium.PolyLine(points, color='blue', weight=1, opacity=0.3).add_to(m)

    # Plot Abnormal Tracks (Plot ALL of them)
    print(f"Plotting {len(l_dict_anomaly)} abnormal tracks...")
    for D in l_dict_anomaly:
        points = decode_seq(D["seq"])
        folium.PolyLine(
            points,
            color='red',
            weight=3,
            opacity=0.8,
            popup=f"MMSI: {D['mmsi']}<br>Start: {D['t_start']}"
        ).add_to(m)

    m.save(save_path)
    print(f"Interactive map saved to {save_path}")

##################################################################################################
##################################################################################################
def plot_track_profile(D, save_path, onehot_lat_bins, onehot_lon_bins, onehot_sog_bins,
                       lat_min, lat_max, lon_min, lon_max):
    """
    Plots the trajectory, Speed profile, and Log-Likelihood profile for a single track.
    """
    try:
        seq = D["seq"]
        log_weights = D["log_weights"]  # Shape: [Time, Samples]
        anomaly_idx = D.get("anomaly_idx", np.zeros(len(seq)))

        # Skip the first 12 timesteps (2*6) as they are used for context
        seq_plot = seq[12:] if len(seq) > 12 else seq
        log_weights_plot = log_weights[12:] if len(log_weights) > 12 else log_weights

        # Decode latitude and longitude
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        lats = (seq_plot[:, 0] * 0.01) + lat_min
        lons = ((seq_plot[:, 1] - onehot_lat_bins) * 0.01) + lon_min

        # Decode Speed (SOG is the 3rd component, index 2)
        sog_offset = onehot_lat_bins + onehot_lon_bins
        sog_knots = (seq_plot[:, 2] - sog_offset)

        # Decode Course (COG is the 4th component, index 3)
        cog_offset = sog_offset + onehot_sog_bins
        cog_deg = (seq_plot[:, 3] - cog_offset) * 5  # Assuming 5 degree resolution

        # Calculate mean log-prob per timestep
        if len(log_weights_plot.shape) > 1:
            log_prob_t = np.mean(log_weights_plot, axis=1)
        else:
            log_prob_t = log_weights_plot

        # Match lengths
        min_len = min(len(sog_knots), len(log_prob_t), len(cog_deg))

        fig, ax = plt.subplots(4, 1, figsize=(12, 14))

        # 1. Trajectory Map
        ax[0].plot(lons[:min_len], lats[:min_len], 'b-', linewidth=2, alpha=0.6)
        ax[0].scatter(lons[0], lats[0], c='green', s=100, marker='o', label='Start', zorder=5)
        ax[0].scatter(lons[min_len - 1], lats[min_len - 1], c='red', s=100, marker='s', label='End', zorder=5)

        # Highlight anomaly segments
        if len(anomaly_idx) > 0 and np.any(anomaly_idx[:min_len] == 1):
            anomaly_points = anomaly_idx[:min_len] == 1
            ax[0].scatter(lons[:min_len][anomaly_points], lats[:min_len][anomaly_points],
                          c='red', s=50, alpha=0.8, marker='x', label='Anomaly')

        ax[0].set_xlabel("Longitude")
        ax[0].set_ylabel("Latitude")
        ax[0].set_title(f"Track Trajectory - MMSI: {D['mmsi']}")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        # 2. Anomaly Score (Log Prob)
        ax[1].plot(log_prob_t[:min_len], 'k-', linewidth=1.5, label='Log Likelihood')

        # Highlight detected anomaly segments
        if len(anomaly_idx) > 0 and np.any(anomaly_idx[:min_len] == 1):
            y_min, y_max = ax[1].get_ylim()
            ax[1].fill_between(range(min_len), y_min, y_max,
                               where=anomaly_idx[:min_len] == 1,
                               color='red', alpha=0.3, label='Detected Anomaly')

        ax[1].set_ylabel("Log Likelihood")
        ax[1].set_title("Anomaly Detection Score")
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

        # 3. Speed Profile
        ax[2].plot(sog_knots[:min_len], 'b-', linewidth=1.5)
        ax[2].set_ylabel("Speed (Knots)")
        ax[2].set_title("Speed over Ground (SOG)")
        ax[2].grid(True, alpha=0.3)

        # 4. Course Profile
        ax[3].plot(cog_deg[:min_len], 'g-', linewidth=1.5)
        ax[3].set_ylabel("Course (Degrees)")
        ax[3].set_xlabel("Time Steps (10 mins)")
        ax[3].set_title("Course over Ground (COG)")
        ax[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return True

    except Exception as e:
        print(f"Error plotting track profile: {str(e)}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        return False