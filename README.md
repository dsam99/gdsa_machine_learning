# GDSA Machine Learning

This repository contains my work at NASA JPL during the summer of 2019. The signal processing and machine learning files were approved for open source access.

This research project will monitor the MSL GDS downlink process to determine whether or not passes are successful. This project will also aim to understand
the characteristics in the dataset and perform anomaly detection

<!-- ## Data Collection

The data is collected from the following APIs:
 - maros
 - tlmweb
 - gdsa elastic search database

The new version of tlmweb allows us to search for relayProductId, which is similar to Overflight ID in the maros and elastic search data. -->

## Signal Processing

The signal processor combines the data from the three data sources, MAROS, Telemetry Data Storage, and the GDSA Elastic Search Database. It computes the relevant features and cleans the dataset for analysis

## Machine Learning

This directory contains the various machine learning algorithms used for our data analysis. It contains a deep neural network for classification of data transmission and multiple different algorithms for anomaly detection.


