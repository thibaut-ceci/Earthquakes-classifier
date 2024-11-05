"""
Cleaning data in ESEC

This module contains functions to clean data from ESEC.
"""

from tqdm.notebook import tqdm

import obspy

tqdm.pandas()


def events_without_pickle_file(ESEC_avalanches):
    """
    Function to remove the events without pickle file.

    Parameters:
    ------------
    ESEC_avalanches : pandas.DataFrame
        The ESEC.

    Returns:
    ---------
    ESEC_avalanches : pandas.DataFrame
        The new ESEC without the events without pickle file.
    """

    ## Loop over all events
    for numero_event in tqdm(ESEC_avalanches["numero"], total=len(ESEC_avalanches)):

        ## Check if events have pickle file
        try:
            stream = obspy.read(f"sismogrammes/{numero_event:03d}.pickle")

        ## If an event has not pickle file, it is removed
        except FileNotFoundError:
            # print("No pickle file in stream " + str(numero_event))

            ESEC_avalanches = ESEC_avalanches.drop(numero_event)

            # print("line " + str(numero_event) + " removed")
            # print("")

            continue

    print("In this catalog, there are now", len(ESEC_avalanches), "avalanches.")

    return ESEC_avalanches