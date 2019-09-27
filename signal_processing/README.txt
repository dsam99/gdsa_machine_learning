
merge_csv.py combines the data from multiple GDSA ES csv files into one csv file.

combine_es_maros.py combines data from two csv files, one from GDSA ES and one from MAROS.

merge_datasets.py combines data from two csv files, one is the combined ES/MAROS data and the other is from TLMWEB. Determining which rows in TLMWEB correspond with the other files is difficult. We need to find a better way to do this.

signal_processor.py computes the desired features from the merged dataset, the output of merge_datasets.py.

