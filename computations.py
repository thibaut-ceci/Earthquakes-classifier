"""
ESEC computation libraries.

This library contains various functions to perform calculations.
"""

from tqdm.notebook import tqdm

from datetime import datetime
import datetime as dt
import glob
from matplotlib import lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import t
import warnings

tqdm.pandas()


def conversion_temps(dataframe, event_indexX, trace):
    date_string = dataframe.loc[dataframe["numero"] == event_indexX, "time"].values[0]
    date_object = datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S.%fZ')
    start_time = dt.datetime.strptime(str(date_object), '%Y-%m-%d %H:%M:%S.%f')
    start = (start_time - trace.stats.starttime.datetime).total_seconds()

    return start

def create_detection_dataframe(ESEC_avalanches, event_index, trimmed_time_starttime, trimmed_time_endtime, distance_all_trace):
    """
    Creates a dataframe for detection results and visualizes it.

    Parameters:
    -----------
    ESEC_avalanches : pd.DataFrame 
        The ESEC.
    event_index : int
        Index of the event to analyze.
    trimmed_time_starttime : list
        List of detection start times.
    trimmed_time_endtime : list
        List of detection end times.
    distance_all_trace : list
        List of distances for each trace.
    """

    ## Create a dataframe with the start and end time of the detection method and the distance of the stations
    df = pd.DataFrame({'start_time': trimmed_time_starttime, 'end_time': trimmed_time_endtime, 'distance': distance_all_trace})
    df = df.sort_values('distance') ## Sort by distance
    df = df.reset_index(drop=True)  ## Reset index
    df['detection'] = df['start_time'].apply(lambda x: False if np.isnan(x) else True) ## In a new column named "detection", add True if detection is possible. Add False if detection is not possible
    df['duration'] = df['end_time'] - df['start_time'] ## Compute the duration of the event using the detection method

    ## Extract the volume of the event
    volume = [ESEC_avalanches["volume"][event_index]]

    ## Create a figure to see the result
    plt.figure(figsize=(10, 10), dpi=300)

    count_detection = 0
    count_non_detection = 0

    for i in range(len(df["detection"])):
        if df["detection"][i] == True:
            plt.scatter(volume, df["distance"][i], c='blue', marker='o', s=30)
            count_detection += 1
        else:
            plt.scatter(volume, df["distance"][i], c='red', marker='x', s=40)
            count_non_detection += 1

    detection_handle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=3, label='Detection (' + str(count_detection) + ')')
    no_detection_handle = mlines.Line2D([], [], color='red', marker='x', linestyle='None', markersize=3, label='Bad/no detection (' + str(count_non_detection) + ')')

    plt.legend(handles=[detection_handle, no_detection_handle])
    plt.xlabel(r"Volume [$\mathrm{m^3}$]")
    plt.ylabel("Distance from station to avalanche [km]")
    plt.ylim(0, 600)


def moyenne_glissante(f, shanon_index, ax, window_size = 40):
    """
    Calculates and plots a moving average with a specified window size.

    Parameters:
    -----------
    f : np.array
        The frequency values (in that case).
    shanon_index : np.array
        The Shannon index values corresponding to the frequencies (in that case).
    ax : matplotlib.Axes
        The axis on which to plot the moving average.
    window_size : int
        The window size for calculating the moving average.
    """

    ## Calculate the moving average using convolution.
    moving_average = np.convolve(shanon_index, np.ones(window_size)/window_size, mode='valid')

    ## Compute the difference between the original Shannon index length (in that case) and the moving average.
    diff = len(f[:len(shanon_index)]) - len(moving_average)

    ## Pad the moving average with NaN values to match the original length.
    moving_average_padded = np.append(moving_average, [np.nan]*diff)

    ## Plot the result
    ax[2].plot(f[:len(shanon_index)], moving_average_padded, c="red", label="Moving average")


def merge_dataframes(dossier = "features/1_fitting/data", name = "curve_parameters*.csv", area_to_save = 'features/1_fitting/data/fitting_df.csv'):
    """
    Merges multiple CSV files from a specified folder into a single dataframe and saves the result to a CSV file.

    Parameters:
    -----------
    dossier : str
        The directory where the CSV files are located.
    name : str
        The filename pattern for the CSV files to merge.
    area_to_save : str
        The path where the merged dataframe will be saved as a CSV file.

    Returns:
    --------
    dataframe_merged : pd.DataFrame
        The merged dataframe containing data from all the CSV files.
    """

    ## Search for all CSV files in the specified directory matching the pattern 'name'
    fichiers_csv = glob.glob(os.path.join(dossier, name))

    ## Read all the CSV files found into a list of dataframes
    dataframes = [pd.read_csv(fichier) for fichier in fichiers_csv]

    ## Concatenate all the dataframes into a single dataframe
    dataframe_merged = pd.concat(dataframes, ignore_index=True)

    ## Save the merged dataframe
    dataframe_merged.to_csv(area_to_save, index=False)

    return dataframe_merged


def conversion_du_temps_du_catalogue(trace, start_time_string, add_time):
    """
    Converts the catalog start time into seconds relative to the start time of the seismic trace and applies an additional time shift.

    Parameters:
    -----------
    trace : obspy.core.trace.Trace
        The seismic trace containing metadata with start or end time.
    start_time_string : str
        The start time of the event in the format 'YYYY_MM_DD HHMMSS'.
    add_time : float
        Additional time in seconds to be added to the computed time.

    Returns:
    --------
    start_time_event_shifted : float
        The event start time in seconds with the time shift applied.
    """
    
    ## Parse the start_time_string into a datetime object using the specified format
    start_time = dt.datetime.strptime(start_time_string, '%Y_%m_%d %H%M%S')

    ## Compute the difference in seconds between the event start time and the trace start time
    start_time_seconds = (start_time - trace.stats.starttime.datetime).total_seconds()

    ## Add the extra time shift to the computed start time in seconds
    start_time_event_shifted = start_time_seconds + add_time

    return start_time_event_shifted
