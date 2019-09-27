import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing

def find_orbiter(x) :
	x = str(x)
	orbiter = x[:3]
	if orbiter == "Sol":
		orbiter = "DTE"
	return orbiter

def frames_to_megabits(x,y):
	frames = x + y
	megabits = frames * 1120 / 125000
	return megabits
	
def convert_predicts(x) :
	if "NA" in str(x):
		x = 0
	if "None" in str(x):
		x = 0
	return x
	
def convert_timeformat(x) :
	if x == "" :
		return x
	if "." not in x :
		x = x + ".000"
	if "NA" in x :
		x = ""
	return x

def time_difference(x,y):
	if x == "" or y == "" :
		return 0
	try:
		x = datetime.datetime.strptime(x, '%Y-%jT%H:%M:%S.%f')
		y = datetime.datetime.strptime(y, '%Y-%jT%H:%M:%S.%f')
	except:
		x = datetime.datetime.strptime(x, '%Y-%jT%H:%M:%S')
		y = datetime.datetime.strptime(y, '%Y-%jT%H:%M:%S')
	if x>y:
		delta = x-y
	else:
		delta = y-x
	return delta

def convert_TDS_time(time):
	time = time.replace("-", "T")
	time = time.replace("/", "-")
	return time

def convert_time_to_seconds(time):
	'''
	Converting time string representation to scalar of seconds difference
	'''

	# checking for no difference in time (returned as int)
	if isinstance(time, int):
		return time

	seconds = time.total_seconds()

	return seconds




# read csv file to dataframe
def process_dataframe(filename, save_to_csv=False, outfile=None):
	'''
	Function to process the features of the dataframe

	Keyword Args:
	filename - the filename to process

	Returns:
	a dataframe containing the processed features of the dataframe
	'''

	df = pd.read_csv(filename)

	# compute numerical features
	df['orbiter'] = np.vectorize(find_orbiter)(df['overflightID'])

	# checking if needed to add extra frames columns
	if "TDS_insync_tf_0_frames" not in df.columns:
		df["TDS_insync_tf_0_frames"] = 0

	if "TDS_insync_tf_32_frames" not in df.columns:
		df["TDS_insync_tf_32_frames"] = 0	

	df['TDS_insync_tf_0_frames'] = df['TDS_insync_tf_0_frames'].fillna(0)
	df['TDS_insync_tf_32_frames'] = df['TDS_insync_tf_32_frames'].fillna(0)
	df['TDS_insync_megabits'] = np.vectorize(frames_to_megabits)(df['TDS_insync_tf_0_frames'],df['TDS_insync_tf_32_frames'])

	df['MAROS_lander_return_value'] = df['MAROS_lander_return_value'].fillna(0)
	df['MAROS_orbiter_return_value'] = df['MAROS_orbiter_return_value'].fillna(0)
	df['GDS_dataActual'] = df['GDS_dataActual'].fillna(0)
	df['GDS_dataPredict'] = df['GDS_dataPredict'].fillna(0)
	df['GDS_dataPredict'] = np.vectorize(convert_predicts)(df['GDS_dataPredict'])
	
	# converting to numerical data
	df['GDS_dataPredict'] = pd.to_numeric(df['GDS_dataPredict'])
	df['GDS_dataActual'] = pd.to_numeric(df['GDS_dataActual'])
	df['TDS_insync_tf_0_frames'] = pd.to_numeric(df['TDS_insync_tf_0_frames'])
	df['TDS_insync_tf_32_frames'] = pd.to_numeric(df['TDS_insync_tf_32_frames'])
	df['MAROS_lander_return_value'] = pd.to_numeric(df['MAROS_lander_return_value'])
	df['MAROS_orbiter_return_value'] = pd.to_numeric(df['MAROS_orbiter_return_value'])

	# calculating delta values as absolute values
	df['rover_to_orbiter_delta'] = abs(np.subtract(df['MAROS_orbiter_return_value'],df['MAROS_lander_return_value']))
	df['orbiter_to_TDS_delta'] = abs(np.subtract(df['TDS_insync_megabits'],df['MAROS_orbiter_return_value']))
	df['TDS_to_GDS_delta'] = abs(np.subtract(df['GDS_dataActual'],df['TDS_insync_megabits']))
	df['actual_to_predict_delta'] = abs(np.subtract(df['GDS_dataActual'],df['GDS_dataPredict']))

	df['TDS_insync_startErt'] = df['TDS_insync_startErt'].fillna("")
	df['TDS_outasync_startErt'] = df['TDS_outasync_startErt'].fillna("")
	df['GDS_beginErt'] = df['GDS_beginErt'].fillna("")
	df['TDS_insync_endErt'] = df['TDS_insync_endErt'].fillna("")
	df['TDS_outasync_endErt'] = df['TDS_outasync_endErt'].fillna("")
	df['GDS_endErt'] = df['GDS_endErt'].fillna("")
	df['GDS_predictBeginErt'] = df['GDS_predictBeginErt'].fillna("")
	df['GDS_predictEndErt'] = df['GDS_predictEndErt'].fillna("")

	# converting TDS time to GDS time
	df['TDS_insync_startErt'] = np.vectorize(convert_TDS_time)(df['TDS_insync_startErt'])
	df['TDS_insync_endErt'] = np.vectorize(convert_TDS_time)(df['TDS_insync_endErt'])
	df['TDS_outasync_startErt'] = np.vectorize(convert_TDS_time)(df['TDS_outasync_startErt'])
	df['TDS_outasync_endErt'] = np.vectorize(convert_TDS_time)(df['TDS_outasync_endErt'])

	df['TDS_insync_startErt'] = np.vectorize(convert_timeformat)(df['TDS_insync_startErt'])
	df['GDS_beginErt'] = np.vectorize(convert_timeformat)(df['GDS_beginErt'])
	df['TDS_insync_endErt'] = np.vectorize(convert_timeformat)(df['TDS_insync_endErt'])
	df['GDS_endErt'] = np.vectorize(convert_timeformat)(df['GDS_endErt'])
	df['GDS_predictBeginErt'] = np.vectorize(convert_timeformat)(df['GDS_predictBeginErt'])
	df['GDS_predictEndErt'] = np.vectorize(convert_timeformat)(df['GDS_predictEndErt'])

	df['TDS_GDS_start_timedelta'] = np.vectorize(time_difference)(df["TDS_insync_startErt"],df['GDS_beginErt'])
	df['TDS_GDS_end_timedelta'] = np.vectorize(time_difference)(df['TDS_insync_endErt'],df['GDS_endErt'])
	df['actual_to_predict_begin_timedelta'] = np.vectorize(time_difference)(df['GDS_beginErt'],df['GDS_predictBeginErt'])
	df['actual_to_predict_end_timedelta'] = np.vectorize(time_difference)(df['GDS_endErt'],df['GDS_predictEndErt'])
	
	# converting datetime values into differences in hours
	df['TDS_GDS_start_timedelta'] = np.vectorize(convert_time_to_seconds)(df['TDS_GDS_start_timedelta'])
	df['TDS_GDS_end_timedelta'] = np.vectorize(convert_time_to_seconds)(df['TDS_GDS_end_timedelta'])
	df['actual_to_predict_begin_timedelta'] = np.vectorize(convert_time_to_seconds)(df['actual_to_predict_begin_timedelta'])
	df['actual_to_predict_end_timedelta'] = np.vectorize(convert_time_to_seconds)(df['actual_to_predict_end_timedelta'])

	# create vectorized features
	myFeatList = ['orbiter', 'TDS_dssId']
	df_features_list = list(df)

	for myFeat in myFeatList :

		# find index of myFeat in dataframe
		i = 0
		feat_index = ""
		while i < len(df_features_list) :
			if df_features_list[i] == myFeat :
				feat_index = i
			i = i + 1

		# create dataframe of feature and sort by frequency
		feature = df[myFeat]
		feature = feature.fillna(0)
		vals, counts = np.unique(feature, return_counts=True)
		results = dict(zip(vals, counts))
		data = pd.Series(results, name= 'freq')
		df2 = pd.DataFrame(data)
		df2 = df2.sort_values('freq', ascending='False')

		# choose 20 most frequent
		dd = df2.iloc[-20:]
		mylist = list(dd.index)
		cols = [myFeat+"_"+str(item).replace(".0","") for item in mylist]

		# create one-hot vectors
		eval_string = "[[1 if item." + myFeat + "==x else 0 for x in mylist] for item in df.itertuples()]"
		myarray = eval(eval_string, {"mylist": mylist, "df": df})

		# convert one-hot vectors to dataframe
		new_df = pd.DataFrame(myarray, columns=cols)

		# add dataframe to output dataframe
		df = pd.concat([df,new_df],axis=1)
		

	# remove unnecessary features
	cols_to_drop_MAROS = [
		'MAROS_pass_time',
		'MAROS_hail_duration'
	] # keeping data volumes for percentages in labelling

	cols_to_drop_TDS = [
		'TDS_pub', "TDS_pubShort", "relayProductId",
		"TDS_startRct", "TDS_startScet", 
		"TDS_endScet", "TDS_endRct", 
		"TDS_dssId", "TDS_relayProductID",
		"TDS_dataType", 
		"TDS_insync_startErt", "TDS_insync_endErt", "TDS_outasync_startErt", "TDS_outasync_endErt",
		"TDS_expirationTime", "TDS_creationTime", 
		"TDS_spoolerName", "TDS_jobNumber"
		# 'TDS_endDsntm', 'TDS_startDsntm',
		# 'TDS_endMst', 'TDS_startMst',
		# 'TDS_endSclk','TDS_startSclk',
	]

	cols_to_drop_GDS = [
		'GDS_sol','overflightID', 'GDS_wid',
		"GDS_beginErt", "GDS_endErt", "GDS_predictBeginErt", "GDS_predictEndErt"
	] 

	cols_to_drop_MISC = ['orbiter']


	df = df.drop(columns=cols_to_drop_MAROS)
	df = df.drop(columns=cols_to_drop_TDS)
	df = df.drop(columns=cols_to_drop_GDS)
	df = df.drop(columns=cols_to_drop_MISC)

	# rename and reorder columns
	df = df.rename(index=str, columns={"MAROS_overflightID": "overflightID", "GDS_wid" : "windowID"})
	cols = list(df)
	# cols.remove("windowID")
	# cols.insert(1,"windowID")
	df = df[cols]

	# converting nan to 0
	df.fillna(0, inplace=True)

	if save_to_csv:
		# write dataframe to csv file
		df.to_csv(outfile, index=False)

	return df

def normalize(filename, save_to_csv=False, outfile=None, min_max=True, z_score=False):
	'''
	Function to process the features of the dataframe

	Keyword Args:
	filename - the filename to process

	Returns:
	a dataframe containing the processed features of the dataframe
	'''

	df = pd.read_csv(filename)

	cols_to_normalize = [
		"GDS_dataActual",	"GDS_dataPredict",	
		"MAROS_lander_return_value", "MAROS_max_elevation", "MAROS_orbiter_return_value", "MAROS_rise_elevation",	
		"TDS_insync_tf_0_frames", "TDS_insync_tf_32_frames", "TDS_outasync_tf_frames", "TDS_insync_megabits", 
		"rover_to_orbiter_delta", "orbiter_to_TDS_delta",
		"TDS_to_GDS_delta", "actual_to_predict_delta", 
		"TDS_GDS_start_timedelta", "TDS_GDS_end_timedelta", 
		"actual_to_predict_begin_timedelta", "actual_to_predict_end_timedelta"
	]

	if min_max:
		scaler = preprocessing.MinMaxScaler()
		scaled_data = scaler.fit_transform(df[cols_to_normalize].values)
		df[cols_to_normalize] = scaled_data
		 
	elif z_score:
		scaler = preprocessing.StandardScaler()
		scaled_data = scaler.fit_transform(df[cols_to_normalize].values)
		df[cols_to_normalize] = scaled_data
	
	if save_to_csv:
		df.to_csv(outfile, index=False)







if __name__ == "__main__":
	# process_dataframe("../data/raw_data/full_898_2450.csv", save_to_csv=True, outfile="../data/processed_data/full_processed.csv")
	normalize("../data/processed_data/full_ids_labelled.csv", save_to_csv=True, 
				outfile="../data/processed_data/full_normalized_ids_labelled_zscore.csv", min_max=False, z_score=True)
