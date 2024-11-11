"""
Database computation libraries.

This library contains various functions to perform calculations.
"""

from tqdm.notebook import tqdm

from datetime import datetime
import datetime as dt
import glob
import os
import pandas as pd

tqdm.pandas()


def conversion_temps(database, event_index, trace):
    """
    Converts a timestamp from the database to a time offset in seconds from the start of the seismic trace.

    Parameters
    ----------
    database : pandas.DataFrame
        A DataFrame containing event information, including a 'time' column with timestamps.
    event_index : int
        The unique identifier for the event to be located in the 'numero' column of the database.
    trace : obspy.Trace
        The ObsPy Trace object containing seismic data and metadata, including trace start time.
    
    Returns
    -------
    start : float
        The time offset in seconds from the trace start time to the event start time.
    """

    ## Extract the event timestamp from the database and convert it to a datetime object
    date_string = database.loc[database["numero"] == event_index, "time"].values[0]
    date_object = datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S.%fZ')
    start_time = dt.datetime.strptime(str(date_object), '%Y-%m-%d %H:%M:%S.%f')

    ## Calculate the offset in seconds from the trace start time
    start = (start_time - trace.stats.starttime.datetime).total_seconds()

    return start


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
