# GDSA Machine Learning

This repository contains my work at NASA JPL during the summer of 2019. The signal processing and machine learning files were approved for open source access. This project was in collaboration with the Deep Learning Technologies Group (393K), with my mentors Brian Kahovec and Ryan Alimo. 

The research project monitored the MSL GDS downlink process to determine whether or not data transmission passes are successful. This project also helps us explain the importance of features in the dataset and perform anomaly detection

The data is collected from the following APIs:
 - maros
 - tlmweb
 - gdsa elastic search database
and hand labelled by the GDSA team experts. 

## Signal Processing

The signal processor combines the data from the three data sources, MAROS, Telemetry Data Storage, and the GDSA Elastic Search Database. It computes important features for learning as well as cleaning and normalizing the datasets to pass into our models. 

## Machine Learning

This directory contains the various machine learning algorithms used for our data analysis. It contains a deep neural network for classification of data transmission and multiple different algorithms which obtains ~95% accuracy. For anomaly detection, the repository contains adversarial autoencoders and one-class SVMs.
