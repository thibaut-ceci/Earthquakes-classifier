"""
Manage seismic waveforms with ObsPy.

Write by Léonard Seydoux (seydoux@ipgp.fr) and Thibaut Céci (thi.ceci@gmail.com)
"""

from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees
from tqdm import tqdm

## Connect to the FSDN server, or the IRIS datacenter. Because this variable is defined outside any function, it is visible by all functions of this module.
client = Client("IRIS")


def download_inventory(event, maxradius=1.0, retries=10):
    """
    Downloads the station inventory for a given seismic event.
    
    Parameters:
    -----------
    event : Event
        Seismic event object containing attributes like latitude, longitude, starttime, and endtime.
    maxradius : float
        Maximum radius around the event's location to search for stations (in deg).
    retries : int
        Number of retry attempts in case of failure.
        
    Returns:
    --------
    Inventory
        Inventory object if successful
    """
    
    ## Start time and end time of the event
    start, end = UTCDateTime(event.time), UTCDateTime(event.time) + 300

    ## Download station inventory from client
    for attempt in range(retries):
        try:
            return client.get_stations(
                latitude=event.latitude,
                longitude=event.longitude,
                startbefore=start,
                endafter=end,
                maxradius=maxradius,
                matchtimeseries=True,
                channel="BH*,HH*",
            )
        
        ## If an error occurs, the code restarts
        except Exception as e:
            print(f"Error for download inventory. Attempt {attempt + 1} of {retries}. Error: {e}")
            #time.sleep(delay)
    return []


def download_stream(event, time_margins=150, print_error=False, retries=3):
    """
    Download the stream of waveforms for a given event, focusing only on the nearest station.

    Parameters
    ----------
    event : pd.Series
        A series containing the event information.
    time_margins : float
        The number of seconds to add to the start and end times of the event.
    print_error : bool
        Whether to print error messages.
    retries : int
        Number of times to retry the request in case of an error.

    Returns
    -------
    stream : obspy.Stream
        A stream of waveforms for the event from the nearest station.
    """

    ## Start time and end time of the event
    start, end = UTCDateTime(event.time), UTCDateTime(event.time) + 300
    start, end = start - time_margins, end + time_margins

    ## Initialize stream and minimum distance variables
    stream = Stream()
    min_distance = float("inf")
    nearest_station = None

    ## Determine the nearest station
    for network in event.inventory:
        if any(char.isdigit() for char in network.code):
            continue

        for station in network:
            distance = locations2degrees(
                event.latitude, event.longitude, station.latitude, station.longitude
            )
            if distance < min_distance:
                min_distance = distance
                nearest_station = (network.code, station.code)

    if nearest_station:
        network_code, station_code = nearest_station

        ## Attempt to download the waveform from the nearest station
        for attempt in range(retries):
            try:
                traces = client.get_waveforms(
                    network_code, station_code, "*", "BH*,HH*", start, end, attach_response=True
                )
                traces.merge(method=1, fill_value="interpolate")

                for trace in traces:
                    trace.stats.distance = min_distance * 111.19  # Convert degrees to km

                stream += traces
                break  # Break the retry loop if successful

            except Exception as e:
                if print_error:
                    print(f"Error with station {station_code}. Attempt {attempt + 1} of {retries}. Error: {e}")
                continue  # Retry

    return stream
