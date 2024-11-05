"""
Analyse seismic waveforms with ObsPy.

Many functions allow to filter and detect seismic signals from ESEC avalanches.
The detection method is here.
"""
from tqdm.notebook import tqdm

from datetime import timedelta
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
from scipy.signal import welch
from sklearn.linear_model import LinearRegression
from obspy import UTCDateTime, Stream

import computations as cp
import figures

tqdm.pandas()


def filter_stream(ESEC_avalanches, event_index, trace_index, freq_HP=9, freq_LP=0.5, max_percentage=0.3):
    """
    Filter a stream.

    Parameters
    ----------
    ESEC_avalanches : pandas.DataFrame
        The ESEC.
    event_index : int
        The index of the event to filter.
    trace_index : int
        The index of the trace to retrieve and filter.
    freq_HP : float
        The high-pass filter frequency.
    freq_LP : float
        The low-pass filter frequency.
    max_percentage : float
        The maximum percentage for tapering. 

    Returns
    -------
    event : pandas.DataFrame
        The event data corresponding to the event_index.
    stream : obspy.Stream
        The filtered seismic stream.
    trace : obspy.Trace
        The specific trace filtered from the stream.
    """

    ## Retrieve the event data
    event = ESEC_avalanches.loc[ESEC_avalanches["numero"] == event_index]

    ## Read and prepare the stream
    stream = obspy.read(f"sismogrammes/{event_index:03d}.pickle")
    stream = stream.sort(keys=["distance"])
    
    ## Remove response, detrend and filter the stream
    stream = stream.select(component="Z")
    stream.remove_response(output="DISP", pre_filt=(0.01, 0.02, 20, 25), water_level=60)
    stream = stream.detrend()
    stream = stream.filter("highpass", freq=freq_LP)                     # High-pass filter
    stream = stream.filter("lowpass", freq=freq_HP)                      # Low-pass filter
    stream = stream.filter("bandpass", freqmin=freq_LP, freqmax=freq_HP) # Band-pass filter
    stream = stream.taper(max_percentage=max_percentage, type="hann")    # Taper

    ## Select one trace
    trace = stream[0]

    return stream, trace


def thresholds(data_start_detection):

    ## Calculate the RMS of the data
    data_rms = np.sqrt(np.mean(data_start_detection**2)) / 2

    ## Calculate the median of the absolute values of the data
    data_abs_median = 4.5 * np.median(np.abs(data_start_detection))

    ## Threshold 1 : the noise
    threshold_1 = data_abs_median / 2
    data_above_threshold_1 = data_start_detection > threshold_1

    ## Threshold 2 : the seismic signal
    data_above_threshold_2 = data_above_threshold_1 & (np.abs(data_start_detection) > data_rms)
    threshold_2 = data_rms + 1.1 * threshold_1 ## Adjusted RMS

    return threshold_1, threshold_2, data_above_threshold_1, data_above_threshold_2


def detection_on_one_trace(trace, dataframe, event_index):

    for event in dataframe["numero"]:
        if event == event_index:
            time_raw = trace.times()
            data_raw = trace.data
            distance = trace.stats.distance
            print("The distance of the trace is " + str(distance))

            start = cp.conversion_temps(dataframe, event, trace)

            ## Trim the trace
            mask = (time_raw >= start) & (time_raw <= start + 500)
            time_start_detection = time_raw[mask]
            data_start_detection = trace.data[mask]


            ### Step 2 : The detection with the thresholds

            ## Compute thresholds
            lower_threshold, upper_threshold, data_above_threshold_1, data_above_threshold_2 = thresholds(data_start_detection)

            try:
                ## Compute start and end times of the trace
                start_time = time_start_detection[data_above_threshold_1][0]
                end_time = time_start_detection[data_above_threshold_2][-1]

                ## Trim the trace
                mask = (time_start_detection >= start_time) & (time_start_detection <= end_time)
                trimmed_time = time_start_detection[mask]
                trimmed_data = data_start_detection[mask]
                

                print("Detection on event", event)

            except IndexError:
                ## If detection is not possible, an error occurs.
                print("No detection on event", event)
                start_time = end_time = trimmed_time = trimmed_data = np.nan

            ### Step 3 : Add conditions to improve the detection method

            ## Compute the duration of the avalanche
            duration = end_time - start_time

            ## Add condition if the duration is too small
            if duration < 10:
                print(f"Event duration too short: {str(duration)} seconds. Bad detection on event {event}.")
                start_time = end_time = trimmed_time = trimmed_data = np.nan


            ## Add condition if the noise threshold is upper the seismic signal threshold
            if lower_threshold > upper_threshold:
                print(f"Noise threshold too high - no detection on event {event}.")
                trimmed_time = trimmed_data = upper_threshold = lower_threshold = start_time = end_time = np.nan

    return time_start_detection, data_start_detection, trimmed_time, trimmed_data, time_raw, data_raw, upper_threshold, lower_threshold


def fit_line(frequencies, values):
    """
    Fits a linear model in the data.

    Parameters:
    ------------
    frequencies : np.ndarray
        The frequency values for the PSD.
    values : np.ndarray
        The PSD values.

    Returns:
    ---------
    model.coef_[0] : float
        The slope of the fitted line.
    model.intercept_ : float
        The intercept of the fitted line.
    """

    model = LinearRegression()     ## The linear model
    f = frequencies.reshape(-1, 1) ## Reshapes the frequencies array into a 2D array
    model.fit(f, values)           ## Adjust the model

    return model.coef_[0], model.intercept_


def find_split_frequency(frequencies, values, min_freq=2.0, max_freq=10.0):
    """
    Finds the split frequency by detecting discontinuities in a specific frequency range.
    
    Parameters:
    ------------
    frequencies : np.ndarray
        The array of frequencies.
    values : np.ndarray
        The array of PSD values corresponding to the frequencies.
    min_freq : float
        The minimum frequency to search for the split.
    max_freq : float
        The maximum frequency to search for the split.

    Returns:
    ---------
    float
        The frequency where a significant discontinuity is detected, which will be used for splitting low and high-frequency ranges.
    """
    
    ## Create a mask to filter frequencies within the range defined by min_freq and max_freq
    freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
    filtered_frequencies = frequencies[freq_mask]
    filtered_values = values[freq_mask]

    ## Calculate the derivative of the PSD values
    derivative = np.diff(filtered_values) / np.diff(filtered_frequencies)

    ## Find the index of the split frequency
    split_index = np.argmax(np.abs(np.diff(derivative)))

    return filtered_frequencies[split_index + 1] if split_index < len(filtered_frequencies) - 1 else filtered_frequencies[-1]


def ajustement_de_segment(mask, frequencies_signal, psd_signal, ax, color='green', label="Ajustement bas", pltplot = True):
    """
    Fits a linear model to a segment of the PSD data and plots the result.

    Parameters:
    ------------
    mask : np.ndarray
        Boolean mask array to filter.
    frequencies_signal : np.ndarray
        The full array of frequency values.
    psd_signal : np.ndarray
        The full array of PSD values corresponding to the frequencies.
    ax : matplotlib axis
        The axis object on which to plot the adjusted line.
    color : str
        The color of the fitted line.
    label : str
        The label for the fitted line.
    pltplot : bool
        If True, uses plt to plot directly, otherwise uses the provided axis.

    Returns:
    ---------
    freq : np.ndarray
        The filtered values.
    slope : float
        The slope of the fitted line.
    intercept : float
        The intercept of the fitted line
    psd : np.ndarray
        The PSD values
    """

    ## Check if there are any valid points in the mask 
    if np.any(mask):

        ## Extract the frequencies and PSD values
        freq = frequencies_signal[mask]
        psd = psd_signal[mask]

        ## Fit the model
        slope, intercept = fit_line(np.log(freq), np.log(psd))

        ## Plot the result
        # if pltplot == True:
        #     plt.loglog(freq, np.exp(slope * np.log(freq) + intercept), color=color, label=label)
        # else:
        #     ax.loglog(freq, np.exp(slope * np.log(freq) + intercept), color=color, label=label)

    return freq, slope, intercept, psd


def plot_spectre(trace, ESEC_avalanches, trimmed_data, trace_index, event_index, conserv_result=False):
    """
    Plots the power spectral density (PSD) of a seismic signal with Welch method and fits two linear models.

    Parameters:
    ------------
    trace : obspy.Trace
        The seismic trace containing the signal data to be analyzed.
    ESEC_avalanches : pandas.DataFrame
        The ESEC.
    trimmed_data : np.ndarray
        The detected signal data.
    trace_index : int
        Index of the seismic trace.
    event_index : int
        Index of the avalanche event.
    conserv_result : bool
        If True, saves the fitting parameters in a CSV file and the plots.
    """

    ## Initialize list to store results
    curve_params = []

    ## Welch method parameters
    segment_duration = 20
    noverlap = 12
    nperseg = int(segment_duration * trace.stats.sampling_rate)

    ## Welch method
    frequencies_signal, psd_signal = welch(trimmed_data, window='hamming', fs=trace.stats.sampling_rate, nperseg=nperseg, noverlap=noverlap)

    ## Cut the spectrum between filter frequencies
    mask = (frequencies_signal > 0.5) & (frequencies_signal < 9)
    frequencies_signal = frequencies_signal[mask]
    psd_signal = psd_signal[mask]

    ## Find the split frequency
    split_freq = find_split_frequency(frequencies_signal, psd_signal, min_freq=1, max_freq=10)
    low_mask = frequencies_signal <= split_freq
    high_mask = frequencies_signal > split_freq

    ## Adjusting two models
    _, low_slope, low_intercept, low_psd = ajustement_de_segment(low_mask, frequencies_signal, psd_signal, plt, color='green', label="Modèle bas", pltplot = False)
    _, high_slope, high_intercept, high_psd = ajustement_de_segment(high_mask, frequencies_signal, psd_signal, plt, color='blue', label="Modèle haut", pltplot = False)

    ## Store results
    if conserv_result == True:
        curve_params.append({
                'Event Index': event_index,
                'Fréquence coin': split_freq,
                'Slope basse frequence': low_slope,
                'Intercept basse frequence': low_intercept,
                'First PSD basse frequence': low_psd[0],
                'PSD requence coin': low_psd[-1],
                'Slope haute frequence': high_slope,
                'Intercept haute frequence': high_intercept,
                'Last PSD haute frequence': high_psd[-1], 
                'numero1': ESEC_avalanches["numero"][event_index],
                'etiquette': ESEC_avalanches["type"][event_index]
            })
        
    ## Save results in a dataframe
    if conserv_result == True:
        df = pd.DataFrame(curve_params)
        df.to_csv(f'features/1_spectre/curve_parameters_{event_index}_spectre.csv', index=False)
        
    # plt.loglog(frequencies_signal, psd_signal, color="C1", label="Spectrum of the detected seismic signal")
    # plt.legend()
    # plt.margins(x=0)
    # plt.xscale("log")
    # plt.xlabel('Fréquences (Hz)')
    # plt.ylabel(r'Power Spectral Density of Displacement($\mathrm{\frac{m^{2}}{Hz}}$)')

    # figures.save(f"fitting_on_trace_{trace_index}_in_event_{event_index}.pdf")

    # plt.show()