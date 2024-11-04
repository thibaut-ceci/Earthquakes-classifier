"""
ESEC figures libraries.

This module contains functions to plot data from ESEC.
"""

import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import computations as cp


def save(path, tight_layout=True):
    """
    Saves the current figure to the file path and displays it.

    Parameters:
    ------------
    path : str
        The file path where the plot will be saved.
    tight_layout : bool
        Adjust the layout of the plot to prevent overlap of elements if "True".
    """
    if tight_layout:
        plt.tight_layout()

    plt.savefig(path) ## Save the plot to the path
    plt.show()        ## Display the plot


def linear_model(x, a, b):
    """
    A linear model for the function "plot_fitted_curve"
    """
    return a + b * x


def plot_fitted_curve(catalog, column1, column2, ax, number):
    """
    Plot a fitted curve (a linear model) to the data.

    Parameters:
    ------------
    catalog : pandas.DataFrame
        The ESEC.
    column1 : str
        The data of the X-axis.
    column2 : str
        The data of the Y-axis.
    ax : matplotlib.axes.Axes
        For multiple subplots.
    number : int
        For multiple subplots (the index).

    Returns:
    ---------
    log_X : np.ndarray
        Log of the data of the X-axis.
    log_Y : np.ndarray
        Log of the data of the Y-axis.
    X_fit : np.ndarray
        X values used to plot the fitted curve.
    Y_fit : np.ndarray
        Fitted Y values corresponding to X_fit.
    popt : np.ndarray
        Optimal parameters for the fitted model.
    pcov : np.ndarray
        Covariance matrix of the fitted parameters.
    a : float
        The coefficient of the model.
    b : float
        The exponent of the model.
    """
    
    ## Log of data with a small constant to avoid log(0)
    constant = 0.01
    log_X = np.log(catalog[column1] + constant)
    log_Y = np.log(catalog[column2] + constant)

    ## Fit the linear model to the data
    popt, pcov = curve_fit(linear_model, log_X, log_Y)

    ## Convert the fitted parameters back to the original scale
    a = np.exp(popt[0])
    b = popt[1]

    ## Prepare data for plotting the fitted curve
    X_fit = np.linspace(min(catalog[column1])/300, max(catalog[column1])*500, 400)
    Y_fit = a * X_fit ** b
    
    ax[number].scatter(catalog[column1], catalog[column2], label='Original Data') ## Scatter plot of the original data
    ax[number].plot(X_fit, Y_fit, 'r-', label=f'Fitted Curve: {column2} = {a:.2f} * {column1}^{b:.2f}') ## Plot the model
    ax[number].set_xscale('log')
    ax[number].set_yscale('log')
    ax[number].set_xlabel(f"{column1}" + r" (m$^3$)", fontsize=14)
    ax[number].set_ylabel(f"{column2}" + r" (m$^3$)", fontsize=14)
    ax[number].legend()

    return log_X, log_Y, X_fit, Y_fit, popt, pcov, a, b


def plot_fitted_curve_without_uncertainties(ESEC_with_uncertainties, catalog_without_uncertainties, catalog_filtered, X_fit, Y_fit, Y_fit_lower, Y_fit_upper, Y, a, b, column1, column2, column3, column4, ax, number):
    """
    Plot one fitted curve and the confidence intervals to fill uncertainties.

    Parameters:
    ------------
    ESEC_with_uncertainties : pandas.DataFrame
        The ESEC.
    catalog_without_uncertainties : pandas.DataFrame
        ESEC without uncertainties.
    catalog_filtered : pandas.DataFrame
        ESEC without NaN.
    X_fit : np.ndarray
        X values used to plot the fitted curve.
    Y_fit : np.ndarray
        Fitted Y values corresponding to X_fit.
    Y_fit_lower : np.ndarray
        Lower bound of the confidence interval.
    Y_fit_upper : np.ndarray
        Upper bound of the confidence interval.
    Y : np.ndarray
        The Y values of the data.
    a : float
        Coefficient from the fitted curve.
    b : float
        Exponent from the fitted curve.
    column1 : str
        Name of the column representing the X-axis.
    column2 : str
        Name of the column representing the Y-axis.
    column3 : str
        Column name for the lower bound of the X-error.
    column4 : str
        Column name for the upper bound of the X-error.
    ax : matplotlib.axes.Axes
        For multiple subplots.
    number : int
        For multiple subplots (the index).

    Returns:
    --------
    ESEC_with_uncertainties : pandas.DataFrame
        ESEC with uncertainties.
    """
    ## Plot the fitted curve
    ax[number].plot(X_fit, Y_fit, 'r-', label=f'Curve fitting : DV = {a:.2f} * volume^{b:.2f}')

    ## Plot the data with and without uncertainties
    ax[number].scatter(catalog_without_uncertainties[column1], Y, color='blue', label='Data with computed uncertainties')
    ax[number].errorbar(catalog_filtered[column1], catalog_filtered[column2], xerr=[catalog_filtered[column3], catalog_filtered[column4]], fmt='o', ecolor='gray', color='C0', capsize=2, capthick=0.75, elinewidth=.75, label='Data with uncertainties already in ESEC')

    ## Plot confidence intervals
    ax[number].plot(X_fit, Y_fit_lower, 'r--', alpha=0.5, label='Confidence intervals')
    ax[number].plot(X_fit, Y_fit_upper, 'r--', alpha=0.5)

    #line_values = [] #List to store the first and last values of each line

    ## Plot horizontal lines connecting each point with confidence intervals
    for index, (_, y) in enumerate(zip(catalog_without_uncertainties[column1], Y)):
        h_lower = np.interp(y, Y_fit_lower, X_fit)
        h_upper = np.interp(y, Y_fit_upper, X_fit)
        
        ## Plot the line and add label
        if index == 0:
            ax[number].plot([h_lower, h_upper], [y, y], 'b', alpha=0.2, label="Computed uncertainties")
        else:
            ax[number].plot([h_lower, h_upper], [y, y], 'b', alpha=0.2)
        
        #line_values.append(((h_lower, y), (h_upper, y))) #Store the first and last values of the line
        
        ## Get the original index in the ESEC
        original_index = catalog_without_uncertainties.index[index]
        
        ## Assign the uncertainty values to the corresponding values in the ESEC
        ESEC_with_uncertainties.at[original_index, column3] = h_lower
        ESEC_with_uncertainties.at[original_index, column4] = h_upper

        ## To know which uncertainties are computed or already present in the ESEC
        ESEC_with_uncertainties.at[original_index, 'determined_uncertainty'] = "True"

    ax[number].set_xscale('log')
    ax[number].set_yscale('log')
    ax[number].set_xlabel(f"{column1}" + r" (m$^3$)", fontsize=14)
    ax[number].set_ylabel(f"{column2}" + r" (m$^3$)", fontsize=14)
    ax[number].legend()

    return ESEC_with_uncertainties#, line_values


def scatter_plot(columnx, columny, labelx, labely, logx=False, logy=False):
    """
    Create a scatter plot.

    Parameters:
    ------------
    columnx : np.ndarray
        Data for the x-axis.
    columny : np.ndarray
        Data for the y-axis.
    labelx : str
        Label for the x-axis.
    labely : str
        Label for the y-axis.
    logx : bool
        If True, sets the x-axis to a logarithmic scale.
    logy : bool
        If True, sets the y-axis to a logarithmic scale.
    """
    
    plt.figure(figsize=(10, 6))

    plt.scatter(columnx, columny, edgecolors='black', s=60, c='skyblue')

    plt.xlabel(labelx, fontsize=12)
    plt.ylabel(labely, fontsize=12)
    
    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')

    plt.show()


def plot_detected_signal(time_start_detection, data_start_detection, trimmed_time, trimmed_data, time_raw, data_raw, upper_threshold, lower_threshold):
    """
    Plot detected seismic signals with thresholds.

    Parameters
    ----------
    time_start_detection : np.ndarray
        Time points for detection.
    data_start_detection : np.ndarray
        Data values for detection.
    trimmed_time : np.ndarray
        Time points of the detected signal.
    trimmed_data : np.ndarray
        Trimmed data of the detected signal.
    time_raw : np.ndarray
        Time points of the raw signal.
    data_raw : np.ndarray
        Raw signal of the raw signal.
    upper_threshold : float
        Seismic signal threshold.
    lower_threshold : float
        Noise threshold.
    """

    _, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_raw, data_raw, label = "Signal")
    ax.plot(time_start_detection, data_start_detection, "C1", label="Signal for detection")
    ax.plot(trimmed_time, trimmed_data, "C2", label="Detected signal")

    ax.axhline(upper_threshold, color="g", label="Seismic signal threshold", linestyle="--")
    ax.axhline(lower_threshold, color="r", label="Noise threshold", linestyle="--")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Displacement [m]")
    ax.set_xlim(time_raw[0], time_raw[-1])
    ax.legend()


def plot_waveform_with_distance_signal(ESEC_avalanches, trace_data, event_index, time, distance, ax, color = "k", lw = 0.2):
    """
    Plots the seismic waveform adjusted by the station distance for a given event.

    Parameters:
    -----------
    ESEC_avalanches : pandas.DataFrame 
        The ESEC.
    trace_data : np.ndarray
        Seismic trace data.
    event_index : int
        Index of the event in the ESEC.
    time : list
        Time offsets for the trace.
    distance : float
        Distance of the station.
    ax : matplotlib.axes.Axes
        Matplotlib axis to plot on.
    color : str
        Color of the plot line.
    lw : float
        Line width.
    
    Returns:
    --------
    time_real : list
        Real time
    waveform : np.ndarray
        Waveform data
    """

    ## Convert event start time from string to datetime object
    start_time_real = datetime.datetime.strptime(ESEC_avalanches["starttime"][event_index], "%Y_%m_%d %H%M%S")
    time_real = [start_time_real + timedelta(seconds=t) for t in time]

    ## Compute the waveform based on trace data and a scaling factor
    waveform = cp.compute_waveform_with_distance(trace_data, 10)

    ## Plot the waveform adjusted by distance
    ax.plot(time_real, waveform + distance, color=color, lw=lw, rasterized=True)

    return time_real, waveform


def plot_waveform_with_distance_detected_signal(ESEC_avalanches, trace_data, event_index, time, distance, label1, ax, color = "k", lw = 0.2):
    """
    Plots the seismic waveform detected by the station distance for a given event.

    Parameters:
    -----------
    ESEC_avalanches : pandas.DataFrame 
        The ESEC.
    trace_data : np.ndarray
        Seismic trace data.
    event_index : int
        Index of the event in the ESEC.
    time : list
        Time offsets for the trace.
    distance : float
        Distance of the station.
    label1 : bool
        If True, the figure label will appear
    ax : matplotlib.axes.Axes
        Matplotlib axis to plot on.
    color : str
        Color of the plot line.
    lw : float
        Line width.
    
    Returns:
    --------
    time_real : list
        Real time
    waveform : np.ndarray
        Waveform data
    label1 : bool
        If True, the figure label will appear
    """

    ## Convert event start time from string to datetime object
    start_time_real = datetime.datetime.strptime(ESEC_avalanches["starttime"][event_index], "%Y_%m_%d %H%M%S")
    time_real = [start_time_real + timedelta(seconds=t) for t in time]

    ## Compute the waveform based on trace data and a scaling factor
    waveform = cp.compute_waveform_with_distance(trace_data, 10)

    ## To display the legend one time and plot the detected part of the signal by the detection method
    if label1 == True:
        ax.plot(time_real, waveform + distance, color=color, lw=lw, rasterized=True, label = "Detected signal")
        label1 = False
    else:
        ax.plot(time_real, waveform + distance, color=color, lw=lw, rasterized=True)

    return time_real, waveform, label1
