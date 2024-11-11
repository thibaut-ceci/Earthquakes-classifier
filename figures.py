"""
Database figures libraries.

This module contains functions to plot data from database.
"""

import matplotlib.pyplot as plt


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