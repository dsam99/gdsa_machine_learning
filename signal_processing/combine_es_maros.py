import csv
import pandas as pd
import datetime
import math

maros_df = pd.read_csv("/Users/bkahovec/Desktop/maros_history_full_clean.csv")
maros_cols = list(maros_df)
print(maros_cols)

tds_df = pd.read_csv("/Users/bkahovec/Desktop/NRT_TDS_Records.csv")
tds_cols = list(tds_df)
print(tds_cols)

es_df = pd.read_csv("/Users/bkahovec/Desktop/sol0000-2500.csv")
es_cols = list(es_df)
print(es_cols)
#overflight_id_list = es_df.loc[:,'overflightID']
overflight_id_list = ['MRO_MSL_2019_051_03', 'MRO_MSL_2019_046_01', 'Sol_2320_AM_HGA_DTE']


some_cols = tds_cols + es_cols
df1 = pd.DataFrame(columns=some_cols)
# Combine TDS and ES
for i, item in tds_df.iterrows() :
	beginERT = datetime.datetime.strptime(item[1], "%Y/%j-%H:%M:%S.%f")
	endERT = datetime.datetime.strptime(item[2], "%Y/%j-%H:%M:%S.%f")
	sync = item[3]
	frames = item[4]
	relay = item[5]
	if isinstance(relay,str) :
		relay = relay[:3]
	else :
		relay = ""

	min_diff = datetime.timedelta(0, 100000)
	min_j = 0
	for j, es_item in es_df.iterrows() :
		ert1 = es_item[5]
		ert2 = es_item[6]
		if isinstance(ert1,str):
			es_beginERT = datetime.datetime.strptime(ert1, "%Y-%jT%H:%M:%S.%f")
			es_endERT = datetime.datetime.strptime(ert2, "%Y-%jT%H:%M:%S.%f")
			diff = abs(beginERT - es_beginERT) + abs(endERT - es_endERT)
			if diff < min_diff :
				min_diff = diff
				min_j = j

	x = item.values.tolist() + es_df.loc[min_j].values.tolist()
	df1 = df1.append(pd.Series(x, index=some_cols), ignore_index=True)

df1.to_csv("/Users/bkahovec/Desktop/temp_dataframe.csv")


#all_cols = maros_cols + tds_cols + es_cols
#df = pd.DataFrame(columns=all_cols)
# Combine MAROS and ES
#for ofid in overflight_id_list :

#	x = es_df.loc[lambda es_df: es_df['overflightID'] == ofid].values.tolist()
#	y = maros_df.loc[lambda maros_df: maros_df['overflightID'] == ofid].values.tolist()

#	print(ofid)
#	for i in range(len(x)) :
#		if i < len(y) : 
#			xy = x[i] + y[i]
#			df = df.append(pd.Series(xy, index=all_cols), ignore_index=True)
#		else :
#			df = df.append(pd.Series(x[i], index=es_cols), ignore_index=True) 

#df.to_csv("/Users/bkahovec/Desktop/dummy.csv")


#d = {'col1': [1, 2], 'col2': [3, 4]}
#df = pd.DataFrame(data=d)


#d2 = {'col3': [1, 2], 'col4': [3, 4]}
#df2 = pd.DataFrame(data=d2)

#print(df)
#print(df2)

#x = df.loc[lambda df: df['col1'] == 1]
#print(x)
#y = df2.loc[lambda df2: df2['col3'] == 1]
#print(y)
#print(x.join(y))
#print(y.join(x))