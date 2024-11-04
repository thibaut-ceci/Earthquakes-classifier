"""
ESEC computation libraries.

This library contains various functions to perform calculations.
"""

from tqdm.notebook import tqdm

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


def remove_outliers_in_catalog(catalog, catalog_column, subset):
    """
    Removes outliers from the ESEC based on the interquartile range method.

    Parameters:
    ------------
    catalog : pandas.DataFrame
        The ESEC.
    catalog_column : pandas.Series
        The column in the ESEC to be used for outlier detection.
    subset : list
        A list containing the subset columns for which missing values are to be considered.

    Returns:
    ---------
    pandas.DataFrame
        The ESEC without outliers.
    """

    ## Remove a warning
    warnings.filterwarnings("ignore", category=UserWarning)

    ## Calculate the first and third quartiles
    Q1 = catalog_column.quantile(0.25)
    Q3 = catalog_column.quantile(0.75)
    IQR = Q3 - Q1

    ## Determine the bounds for outlier detection
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(f"Lower bound for {subset[1]}: {lower_bound}")
    print(f"Upper bound for {subset[1]}: {upper_bound}")

    ## Identify outliers
    outliers = catalog[(catalog_column < lower_bound) | (catalog_column > upper_bound)]
    print(f"Number of outliers in {subset[1]}:", outliers.shape[0])

    ## Remove rows with missing values
    ESEC_avalanches = catalog.dropna(subset=subset)

    ## Remove outliers with the computed bounds
    ESEC_avalanches_filtered = ESEC_avalanches[(catalog_column > lower_bound) & (catalog_column < upper_bound)]

    return ESEC_avalanches_filtered


def confidence_bounds(X_fit, log_X, popt, pcov):
    """
    Calculates the confidence interval bounds for the fitted curve.

    Parameters:
    ------------
    X_fit : np.ndarray
        X values used to plot the fitted curve (in figures.plot_fitted_curve).
    log_X : np.ndarray
        Log of the data of the X-axis (in figures.plot_fitted_curve).
    popt : np.ndarray
        Optimal parameters for the fitted model (in figures.plot_fitted_curve).
    pcov : np.ndarray
        Covariance matrix of the fitted parameters (in figures.plot_fitted_curve).

    Returns:
    ---------
    Y_fit_lower : np.ndarray
        The lower bound of the fitted curve.
    Y_fit_upper : np.ndarray
        The upper bound of the fitted curve.
    """
    ## Calculate the standard error for each parameter
    perr = np.sqrt(np.diag(pcov))

    ## Calculate the t-statistic for a 90% confidence interval
    t_val = t.ppf(0.95, len(log_X) - len(popt))

    ## Confidence interval (lower and upper bounds) for 'a' in the model
    a_conf_interval = np.exp(popt[0] + t_val * perr[0] * np.array([-1, 1]))

    ## Confidence interval (lower and upper bounds) for 'b' in the model
    b_conf_interval = popt[1] + t_val * perr[1] * np.array([-1, 1])

    ## Calculate the upper and lower confidence bounds of the model
    Y_fit_lower = a_conf_interval[0] * X_fit ** b_conf_interval[0]
    Y_fit_upper = a_conf_interval[1] * X_fit ** b_conf_interval[1]

    return Y_fit_lower, Y_fit_upper


def compute_number_of_stations(ESEC_avalanches, print = False):
    """
    Computes the number of seismic stations for each event in the ESEC.

    Parameters:
    -----------
    ESEC_avalanches : pandas.Dataframe
        The ESEC.
    print_output : bool
        If True, prints the number of stations for each event.

    Returns:
    --------
    ESEC_avalanches : pd.DataFrame
        Updated ESEC with the number of stations for each event.
    """
    
    ## Loop in all events in ESEC
    for numero_event in tqdm(ESEC_avalanches["numero"], total=len(ESEC_avalanches)):

        ## Load the inventory of the event
        inventory = ESEC_avalanches["inventory"][numero_event]

        ## Initializes a list to count the number of stations per event
        num_stations = []

        ## If the inventory is not empty, count the number of stations
        if len(inventory) > 0:
            for i in range(len(inventory.networks)):
                num_stations.append(len(inventory.networks[i].stations))

            ## Print the number of stations if requested
            if print == True:
                print("The number of stations is", sum(num_stations), "for event", str(numero_event))

            ## Add the number of stations in the column 'Number of stations' per event
            ESEC_avalanches.at[numero_event, 'Number of stations'] = sum(num_stations)

        else:
            print("No stations for event", str(numero_event))
            ESEC_avalanches.at[numero_event, 'Number of stations'] = 0

    return ESEC_avalanches


def compute_waveform_with_distance(trace, scale):
    """
    Normalize the waveform and scale it for plotting with distance.

    Parameters
    ----------
    trace : np.ndarray
        The seismic trace.
    scale : float
        Scaling factor for the waveform to control its vertical offset when plotting.

    Returns
    -------
    waveform : np.ndarray
        The scaled and normalized waveform data.
    """
    waveform = trace - np.mean(trace)     ## Center the waveform
    waveform /= np.max(np.abs(waveform))  ## Normalize the waveform
    waveform *= scale                     ## Scale the waveform

    return waveform


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
