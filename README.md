# What is "Earthquakes classifier"?

Earthquakes classifier is a project which aims to create a machine learning algorithm to classify seismic signals from earthquakes or others.
A database based on 10000 seismic signals from earthquakes and 10000 from other sources will be used.

The data comes from USGS : https://earthquake.usgs.gov/earthquakes/search/

# Explanation of the codes :
- 00_read_data --> Read and choose the size of the database
- 01_clean_data --> Remove useless columns
- 02_explore_data --> See the distribution of the database
- 03_download_seismic_waveform --> Download the seismic signals of each events
- 04_clean_pickle --> Remove the events without pickle data
- 05_detection_one_trace --> Test the detection method in one trace
- 06_extract_features --> Extract features from each event
- 07_merge_features --> Merge all features in one database
- 08_train_the_model --> Train and test the model with the features
- 09_test_the_model --> Test the model with the features from the SismoAvalanche project

# A lot of librairies developped in this studies were availables :
- analysis.py : Analyses seismic waveforms with ObsPy. The detection method is here.
- catalog.py : Database catalog management.
- computations.py : Database catalog computation librairies.
- energy.py : Performs energy calculations.
- figures.py : Manages the creation of figures.
- waveform.py : Downloads the seismic data.

# Explanation of the files :
- \data : Contains the many catalogs created during this study and the machine learning model
- \features : Contains the features calculated during this study and the SismoAvalanche features
- \figures : Contains the plots obtained during this study
- \sismogrammes : Contains the seismic signals from each events
