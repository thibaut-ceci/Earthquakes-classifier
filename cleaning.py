"""
Cleaning data in the database

This module contains functions to clean data from the database.
"""

from tqdm.notebook import tqdm

import obspy

tqdm.pandas()


def events_without_pickle_file(database):
    """
    Function to remove the events without pickle file.

    Parameters:
    ------------
    database : pandas.DataFrame
        The database.

    Returns:
    ---------
    database : pandas.DataFrame
        The new database without the events without pickle file.
    """

    ## Loop over all events
    for numero_event in tqdm(database["numero"], total=len(database)):

        ## Check if events have pickle file
        try:
            stream = obspy.read(f"sismogrammes/{numero_event:03d}.pickle")

        ## If an event has not pickle file, it is removed
        except FileNotFoundError:

            database = database.drop(numero_event)

            continue

    print("In this catalog, there are now", len(database), "events.")

    return database