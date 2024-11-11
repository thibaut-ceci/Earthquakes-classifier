"""
Database energy calculate

Library with several functions to calculate energy of seismic signals
"""

import numpy as np
import pandas as pd
from scipy.signal import spectrogram
import scipy.stats


def next2pow(x):
    """
    Finds the next power of 2 greater than or equal to x.
    This is used to optimize the FFT size in the spectrogram calculation.

    Write by Francesco Biagioli (biagioli@ipgp.fr)
    
    Parameters:
    ------------
    x : int
        The input number.
    
    Returns:
    --------
    int
        The next power of 2 greater than or equal to x.
    """
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def spectres(trace):
    """
    Computes the spectrogram and energy envelope of a seismic trace over a specific frequency band.

    Write by Francesco Biagioli (biagioli@ipgp.fr) and Thibaut Céci
    
    Parameters:
    ------------
    trace : ObsPy Trace object
        A seismic trace.
    
    Returns:
    --------
    distance : float
        The distance of the seismic station from the source.
    times : np.array
        Time points at which the spectrogram was computed.
    energy : np.array
        Energy values over the selected frequency band.
    time_at_max_energy : float
        The time at which the maximum energy is observed.
    energy_max_index : int
        The index of the maximum energy value.
    """
    
    ## Trace parameters
    distance = trace.stats.distance ## Distance of the station from the event source
    fs = trace.stats.sampling_rate  ## Sampling rate of the trace
    data = trace.data               ## The seismic amplitude data of the trace
    
    ## Set window length for the spectrogram analysis
    window_len = 20
    nperseg = int(window_len * fs)

    ## Compute the spectrogram
    frequencies, times, Sxx = spectrogram(data, fs=fs, nperseg=nperseg, noverlap=nperseg * 0.9, nfft=next2pow(nperseg), scaling="density", mode="psd")

    ## Compute the energy between a frequency band of interest
    frequency_band = (0.1, 2)
    frequency_band_indices = np.where((frequencies > frequency_band[0]) & (frequencies < frequency_band[1]))[0]
    energy = Sxx[frequency_band_indices, :].sum(axis=0)

    ## Find where the maximum energy is (it can be a feature)
    energy_max_index = np.argmax(energy)
    time_at_max_energy = times[energy_max_index]

    return distance, times, energy, time_at_max_energy, energy_max_index


def threshold(times, energy):
    """
    Applies an energy threshold to detect a significant portion of the energy signal.

    Parameters:
    ------------
    times : np.array
        Time of the trace.
    energy : np.array
        Energy values of the trace.

    Returns:
    ---------
    sub_times : np.array
        Time points within the thresholded energy range.
    sub_energy : np.array
        Energy values within the thresholded range.
    first_index : int
        The index of the first time point where the energy exceeds the threshold.
    last_index : int
        The index of the last time point where the energy exceeds the threshold.
    threshold_energy : float
        The energy threshold value used for detection.
    """

    ## Compute the energy threshold
    threshold_energy = energy.mean() / 2

    ## Find the first index where the energy exceeds the threshold
    first_index = np.argmax(energy > threshold_energy)

    ## Boolean array where energy values are above the threshold
    energy_above_threshold_energy = energy > threshold_energy

    ## Find the last index where the energy still exceeds the threshold
    last_index = np.where(energy_above_threshold_energy)[0][-1]

    ## Extract the time points and energy values within the thresholded range
    sub_times = times[first_index:last_index+1]
    sub_energy = energy[first_index:last_index+1]

    return sub_times, sub_energy, first_index, last_index, threshold_energy


def compute(database, trace, event_index):
    """
    Compute energy features of the first seismic trace for each event and save the results in a dataframe.

    Parameters:
    ------------
    database : pandas.DataFrame
        The database.
    trace : obspy.Stream
        Seismic first trace for the event.
    event_index : int
        The index of the event.
    """

    ## Calculate spectrograms and energy for the trace
    distance, times, energy, time_at_max_energy, energy_max_index = spectres(trace)

    ## Apply a threshold to detect the energy envelope
    sub_times, sub_energy, first_index, _, _ = threshold(times, energy)

    ## Extract and save features
    dataframe_event = pd.DataFrame({'event_index': [event_index], 
                                    'numero': [database["numero"][event_index]],
                                    'distance': [distance], 
                                    'skewness': [scipy.stats.skew(sub_energy)], 
                                    'area_of_the_energy_parabola': [np.trapz(sub_energy, sub_times)], 
                                    'energy_max': [np.max(sub_energy)], 
                                    'Impulsion': [(energy[energy_max_index] - energy[first_index]) / (time_at_max_energy - times[first_index])],
                                    'Signal duration': [sub_times[-1] - sub_times[0]]})
    
    dataframe_event.to_csv(f'features/2_energy/dataframe_event_{event_index}_energie.csv', index=False)
