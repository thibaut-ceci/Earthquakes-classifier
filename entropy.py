"""
ESEC entropy calculate.

This library contains various functions to compute the entropy of an avalanche.
"""

from tqdm.notebook import tqdm

import covseisnet as csn
import matplotlib.pyplot as plt
import numpy as np
import obspy

import analysis
import computations as cp
import figures

tqdm.pandas()


def filter_stream_with_covseisnet(event_index, trim):
    """
    Filters a seismic stream for a specific event index and applies preprocessing steps.

    Parameters:
    -----------
    event_index : int
        Index of the event to analyze.
    trim : list
        List containing the start and end trim times.

    Returns:
    --------
    stream : csn.arraystream.ArrayStream
        Processed stream of seismic traces.
    """

    ## Load the stream, sort it by distance and keep only the component Z
    stream = obspy.read(f"sismogrammes/{event_index:03d}.pickle")
    stream = stream.sort(keys=["distance"])
    stream = stream.select(component="Z")

    ## Keep only the trace between the source and 100 km distance
    stream = csn.arraystream.ArrayStream([trace for trace in stream if trace.stats.distance < 100])

    ## Trim the stream
    stream.trim(starttime=stream[0].stats.starttime + trim[0], endtime=stream[0].stats.starttime + trim[1])
    min_starttime = max(tr.stats.starttime for tr in stream)
    max_endtime = min(tr.stats.endtime for tr in stream)
    stream.trim(min_starttime, max_endtime)

    ## Resample the stream to a uniform sampling rate of 40 Hz
    stream.resample(40)

    ## Apply detrend, whitening and taper the stream
    stream.detrend("demean")
    stream.preprocess(window_duration_sec=300, epsilon=1e-10)
    stream.taper(max_percentage=0.01)

    ## Trim again after the whitening
    min_starttime = max(tr.stats.starttime for tr in stream)
    max_endtime = min(tr.stats.endtime for tr in stream)
    stream.trim(min_starttime, max_endtime)

    ## Print the stream and the number of traces
    print(stream)
    print("Number of traces :", len(stream))

    return stream


def compute_entropy(stream, window_size=50, average=4):
    """
    Computes the entropy of a stream.

    Parameters:
    -----------
    stream : csn.arraystream.ArrayStream
        The seismic stream.
    window_size : int
        The size of the window for calculating covariance matrices.
    average : int 
        The number of windows to average the covariance matrices.

    Returns:
    --------
    times : np.ndarray
        An array of timestamps.
    frequencies : np.ndarray
        An array of frequency.
    entropy : np.ndarray
        An array representing the entropy.
    """

    ## Compute the covariance matrices from the seismic stream using the specified window size and averaging factor.
    times, frequencies, covariances = csn.covariancematrix.calculate(stream, window_size, average)

    ## Compute entropy
    entropy = covariances.coherence(kind="entropy")

    return times, frequencies, entropy


def compute_shannon_index(stream, ws0, av0, ax):
    """
    Computes the Shannon index from the entropy of a seismic stream and plots the normalized Shannon index against frequency.

    Parameters:
    -----------
    stream : csn.arraystream.ArrayStream
        The seismic data.
    ws0 : int
        The size of the window for calculating covariance matrices.
    av0 : int
        The number of windows to average the covariance matrices.
    ax : list
        A list of matplotlib Axes objects for plotting.

    Returns:
    --------
    frequencies_shannon_index : np.array
        Array of frequencies for which the Shannon index is computed.
    shannon_index_normalised : np.array
        The normalized Shannon index values corresponding to the frequencies.
    """

    ## Compute entropy
    _, frequencies0, entropy0 = compute_entropy(stream, window_size=ws0, average=av0 - 1)
    
    ## Filter out the first frequency (often zero) to avoid issues in analysis.
    frequencies_shannon_index = frequencies0[1:]

    ## Create a mask to select frequencies within the specified range
    frequency_mask = (frequencies_shannon_index > 0.1) & (frequencies_shannon_index < 18)

    ## Compute Shannon index
    frequencies_shannon_index = frequencies_shannon_index[frequency_mask]
    shannon_index = np.exp(entropy0)
    shannon_index = shannon_index[:, frequency_mask].max(axis=0)
    shannon_index_normalised = shannon_index/len(stream)

    ## Plot the result
    ax[2].plot(frequencies_shannon_index, shannon_index_normalised, lw=0.75)

    return frequencies_shannon_index, shannon_index_normalised


def entropy_extract_features(ESEC_avalanches, stream_csn, ws0, av0, ax, curve_params, event_index):
    """
    Extracts features from the Shannon index of a seismic stream and save the results in a list.

    Parameters:
    -----------
    ESEC_avalanches : pd.DataFrame
        The ESEC.
    stream_csn : csn.arraystream.ArrayStream
        The input seismic data stream for analysis.
    ws0 : int
        The size of the window for calculating covariance matrices.
    av0 : int
        The number of windows to average the covariance matrices.
    ax : list
        A list of matplotlib Axes objects for plotting.
    curve_params : list
        A list to append feature extraction results.
    event_index : int
        Index of the event to analyze.
    """
    
    ## Compute the Shannon index
    f, shannon_index = compute_shannon_index(stream_csn, ws0, av0, ax)
    
    ## Apply a moving average in the entropy spectrum
    cp.moyenne_glissante(f, shannon_index, ax, window_size = 40)

    ## Search the split frequency in the entropy spectrum
    split_freq = analysis.find_split_frequency(f, shannon_index, min_freq=4.0, max_freq=10.0)
    print("Value of the split frequency : ", split_freq)

    ## Fit two models (in the high and low frequency) in the spectrum 
    low_mask = f <= split_freq
    high_mask = f > split_freq
    low_freq, low_slope, low_intercept, low_shanon = analysis.ajustement_de_segment(low_mask, f, shannon_index, ax[2], color='green', label="Model low frequency", pltplot = False)
    high_freq, high_slope, high_intercept, high_shanon = analysis.ajustement_de_segment(high_mask, f, shannon_index, ax[2], color='blue', label="Model high frequency", pltplot = False)
        
    ## Extract and save features
    curve_params.append({'Event Index': event_index,
                         'numero3': ESEC_avalanches["numero"][event_index],
                         'FC_ent': split_freq, 
                         'Slope_BF_ent': low_slope,
                         'Intercept_BF_ent': low_intercept,
                         'First Value_BF_ent': low_shanon[0],
                         'FC_value_ent': low_shanon[-1],
                         'Slope_HF_ent': high_slope,
                         'Intercept_HF_ent': high_intercept,
                         'Last Value_HF_ent': high_shanon[-1],
                         'type': ESEC_avalanches["type"][event_index],
                         'volume': ESEC_avalanches["volume"][event_index],
                         'length': ESEC_avalanches["length"][event_index], 
                         'height': ESEC_avalanches["height"][event_index]})


def plot_result(ESEC_avalanches, trim, ws0, av0, curve_params):
    """
    Plot the results after calculating the entropy.

    Parameters:
    -----------
    ESEC_avalanches : pd.DataFrame
        The ESEC.
    trim : list
        List containing start and end trim values for time.
    ws0 : int
        The size of the window for calculating covariance matrices.
    av0 : int
        The number of windows to average the covariance matrices.
    curve_params : list
        A list to append feature extraction results.

    Returns:
    --------
    curve_params : list
        The features extracted from the entropy spectrum
    ESEC_avalanches : pandas.Dataframe
        The ESEC.
    """

    ## Loop over all the events
    for event_index in tqdm(ESEC_avalanches["numero"], total=len(ESEC_avalanches)):
        try:
            print("-------------------------")
            print("Event numéro", event_index)
            print("-------------------------")


            ### Step 1 : Plot the first trace of the event

            ## Load the stream of the event
            stream = obspy.read(f"sismogrammes/{event_index:03d}.pickle")

            ## Sort by distance, keep only the component Z and trim the stream
            stream = stream.sort(keys=["distance"])
            stream = stream.select(component="Z")
            stream.trim(starttime=stream[0].stats.starttime + trim[0], endtime=stream[0].stats.starttime + trim[1])

            ## Preprocessing the stream
            stream = stream.detrend()                                     # Detrend the seismic traces in the stream
            stream = stream.filter("highpass", freq=0.5)                  # High-pass filter
            stream = stream.filter("lowpass", freq=9)                     # Low-pass filter
            stream = stream.filter("bandpass", freqmin=0.5, freqmax=9)    # Band-pass filter
            stream_filter = stream.taper(max_percentage=0.3, type="hann") # Taper

            ## Keep only the first trace
            trace = stream_filter[0]

            if len(stream_filter) == 1:
                raise Exception("There is only one trace in the stream. Calculating entropy is impossible")


            ### Step 2 : Whitening the stream of the event

            stream_csn = filter_stream_with_covseisnet(event_index, trim)

            if len(stream_csn) == 1:
                raise Exception("There is only one trace in the stream. Calculating entropy is impossible")
            

            ### Step 3 : Plot the first trace computed in the step 1 and plot the whitening stream

            fig, ax = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True, gridspec_kw=dict(height_ratios=[1, 1, 2]))

            ## Plot the first trace
            ax[0].plot(trace.times(), trace.data)
            ax[0].set_ylabel("Digital numbers")
            ax[0].set_xmargin(0)
            ax[0].set_xticklabels([])

            ## Plot the whitening stream
            ax[1].plot(stream_csn[0].times(), stream_csn[0].data)
            ax[1].set_xmargin(0)
            ax[1].set_xlabel("Time [s]")
            ax[1].set_ylabel("Digital numbers")


            ### Step 4 : Compute the entropy and display it

            entropy_extract_features(ESEC_avalanches, stream_csn, ws0, av0, ax, curve_params, event_index)

            ax[2].legend()
            ax[2].set_xscale("log")
            ax[2].set_ylabel("Entropy")
            ax[2].set_xlabel("Frequencies [Hz]")
            ax[2].set_xmargin(0)

            # plt.title(ESEC_avalanches["type"][event_index]) ## To see the type of the avalanche

            figures.save(f"features/3_entropie/pictures/entropie_{event_index}.pdf", tight_layout=False)
            plt.show()

        # If an error occured, it's because the stream is too short.
        except Exception as e:
            print("Error :", e)
            
            ## Remove the event
            ESEC_avalanches = ESEC_avalanches.drop(event_index)

    return curve_params, ESEC_avalanches
