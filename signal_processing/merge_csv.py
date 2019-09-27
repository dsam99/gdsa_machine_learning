import pandas as pd
import glob

path = "./es_csv_files"
allFiles = glob.glob(path + "/*.csv")

list_ = []

for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)

frame = pd.concat(list_, axis = 0, ignore_index = True)

frame.to_csv("./sol0000-2500.csv")