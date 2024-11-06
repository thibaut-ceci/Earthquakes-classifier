"""
ESEC catalog management.

This module contains functions to read, filter and display the ESEC.
"""

import pickle

import matplotlib.pyplot as plt
import pandas as pd


def display_parameters():
    """
    Display all columns and rows of the ESEC. Just call this function for use it.
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
        The pickle file (like the ESEC).
    """
    return pickle.load(open(path, "rb"))


def open_plot(avalanches, pos_number=0.260, xlim=100):
    """
    Plot the number of events in each column of the ESEC.

    Parameters :
    ------------
    avalanches : pandas.DataFrame
        The ESEC.
    pos_number : float
        Controls the vertical position of the text labels on the bars.
    xlim : int
        Sets the X-axis limit of the plot.
    """

    ## Count the number of non-null values in each column and sort them
    ax = avalanches.count().sort_values().plot(kind="barh", figsize=(6, 10))
    ax.set_xlabel("Number of events")
    ax.set_xlim(0, xlim)

    ## For counting the number of values in each column
    for index, value in enumerate(avalanches.count().sort_values()):
        ax.text(value + 0.1, index - pos_number, str(value))


def see_number_distribution_number(avalanches_select, ylabel, ylim):
    ## Count the occurrences of each category in the 'type' column
    category_counts = avalanches_select.value_counts()
    category_counts.plot(kind='barh')

    plt.xlabel("Number of events")
    plt.ylabel(ylabel)

    ## Add text annotations to each bar
    for index, value in enumerate(category_counts):
        plt.text(value+70, index-0.07, str(value), va='center', ha='left')

    plt.xlim(0, ylim)
    plt.tight_layout()
    plt.show()


def see_number_distribution(avalanches_select, ylabel):
    """
    Plots the distribution of number in the ESEC.

    Parameters:
    ------------
    avalanches_select : pandas.Series
        The ESEC.
    ax : matplotlib.axes.Axes
        The Axes object on which to plot the distribution.
    ylabel : str
        The variable being plotted.
    """

    plt.hist(avalanches_select, orientation="horizontal", bins=30)
    plt.xlabel("Number of events")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

