"""
Database management.

This module contains functions to read, filter and display the database.
"""

import pickle
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import obspy
import pandas as pd

tqdm.pandas()


def display_parameters():
    """
    Display all columns and rows of the database. Just call this function for use it.
    """
    pd.set_option("display.max_rows", None)        ## Show all rows
    pd.set_option("display.max_columns", None)     ## Show all columns
    pd.set_option("display.width", None)           ## Adjust width
    pd.set_option("display.max_colwidth", None)    ## Display entire content of each column


def load(path):
    """
    Opens a pickle file and loads its contents.

    Parameters :
    ------------
    path : str
        The path to the pickle file.

    Returns : 
    ---------
    object
        The pickle file (like the database).
    """
    return pickle.load(open(path, "rb"))


def open_plot(database, pos_number=0.260, xlim=100):
    """
    Plot the number of events in each column of the database.

    Parameters :
    ------------
    database : pandas.DataFrame
        The database.
    pos_number : float
        Controls the vertical position of the text labels on the bars.
    xlim : int
        Sets the X-axis limit of the plot.
    """

    ## Count the number of non-null values in each column and sort them
    ax = database.count().sort_values().plot(kind="barh", figsize=(6, 10))
    ax.set_xlabel("Number of events")
    ax.set_xlim(0, xlim)

    ## For counting the number of values in each column
    for index, value in enumerate(database.count().sort_values()):
        ax.text(value + 0.1, index - pos_number, str(value))


def see_number_distribution_plot(database, ylabel, xlim):
    """
    Plots the distribution of column in the database (plot version).

    Parameters:
    ------------
    database : pandas.Series
        The database.
    ylabel : str
        The variable being plotted.
    xlim : str
        The xlim of the plot.
    """

    ## Count the occurrences of each category in the 'type' column
    category_counts = database.value_counts()
    category_counts.plot(kind='barh')

    plt.xlabel("Number of events")
    plt.ylabel(ylabel)

    ## Add text annotations to each bar
    for index, value in enumerate(category_counts):
        plt.text(value+70, index-0.07, str(value), va='center', ha='left')

    plt.xlim(0, xlim)
    plt.tight_layout()
    plt.show()


def see_number_distribution_histogram(database, ylabel):
    """
    Plots the distribution of column in the database (histogram version).

    Parameters:
    ------------
    database : pandas.Series
        The database.
    ax : matplotlib.axes.Axes
        The Axes object on which to plot the distribution.
    ylabel : str
        The variable being plotted.
    """

    plt.hist(database, orientation="horizontal", bins=30)
    plt.xlabel("Number of events")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


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