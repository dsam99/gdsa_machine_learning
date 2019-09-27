import pandas as pd
import datetime
import numpy as np

tds_df = pd.read_csv("NRT_TDS_Records_chopped.csv")
tds_df.Pub.replace(np.nan,'DTE',inplace=True)

old_df = pd.read_csv("MAROS_ESDB_COMBINED.csv")

old_cols = list(old_df)
new_cols = ['DSS', 'Start ERT', 'End ERT', 'Pub', 'insync_tf_0', 'insync_tf_32', 'outasync_tf']

all_cols = old_cols + new_cols
final_df = pd.DataFrame(columns=all_cols)

count = 0
maximum = 100

# for each row in es, save overflight id.
# loop through doy_A, find which overflight id each row in doyA matches up with
# by looping through a subset of old_df

# for row in old_df:
	# find overflightID
	# frames = 0
	# doy_A = filter tds_df
	# for item in doy_A
		# subset = filter old_df
		# for thing in subset
			# compute min time diff
			# find min overflight
		# if min overflight == overflightID
			# add item frames to overflightID frames

for i, row in old_df.iterrows() :

	#if count >= maximum :
	#	break
	
	# if actual data volume > 0
	es_vol = list(row)[9]
	if es_vol == '' :
		es_vol = 0
	if es_vol > 0 :
	
		# find overflight ID
		overflightID = list(row)[2]
		insync_a_frames = 0
		insync_b_frames = 0
		outasync_frames = 0
	
		# find sol
		es_sol = list(row)[1]
	
		# find orbiter
		es_relay = list(row)[2][:3]
		if es_relay == "Sol" :
			es_relay = "DTE"
		else :
			es_relay = es_relay + "RLY"
		
		# find begin and end doy
		start_doy = row[6][0:8].replace("-","/")
		end_doy = row[7][0:8].replace("-","/")
		
		# filter tds_df on relay and (start doy or end doy)
		doy_df = tds_df.loc[lambda tds_df: (tds_df['Start ERT'].str[0:8] == start_doy) | (tds_df['End ERT'].str[0:8] == end_doy)].loc[lambda tds_df: tds_df['Pub'] == es_relay]
		#print("tds_df")
		#print(tds_df)

		# create placeholders for final times
		final_beginERT = datetime.datetime.strptime("2020/001-00:00:00.000", "%Y/%j-%H:%M:%S.%f")
		final_endERT = datetime.datetime.strptime("2012/001-00:00:00.000", "%Y/%j-%H:%M:%S.%f")
		
		# only loop through doy_df	
		for j, item in doy_df.iterrows() :
		
			tds_beginERT = datetime.datetime.strptime(list(item)[1], "%Y/%j-%H:%M:%S.%f")
			tds_endERT = datetime.datetime.strptime(list(item)[2], "%Y/%j-%H:%M:%S.%f")
			tds_frame_type = list(item)[3]
			tds_frame_count = list(item)[4]
			tds_relay = list(item)[5]
			
			sol_df = old_df.loc[lambda old_df: old_df['sol'] == es_sol]
			
			min_diff = datetime.timedelta(hours=2)
			min_overflight = ""
			
			for k, subrow in sol_df.iterrows() :
			
				sub_vol = list(subrow)[9]
				if sub_vol == "" :
					sub_vol = 0
				
				if sub_vol > 0 :
				
					# find orbiter of subrow
					sub_relay = list(subrow)[2][:3]
					if sub_relay == "Sol" :
						sub_relay = "DTE"
					else :
						sub_relay = sub_relay + "RLY"
						
					if sub_relay == tds_relay :
				
						# find begin and end time
						sub_beginERT = datetime.datetime.strptime(list(subrow)[6], "%Y-%jT%H:%M:%S.%f")
						sub_endERT = datetime.datetime.strptime(list(subrow)[7], "%Y-%jT%H:%M:%S.%f")
					
						diff = abs(tds_beginERT - sub_beginERT) + abs(tds_endERT - sub_endERT)
						if diff < min_diff :
							min_diff = diff
							min_overflight = list(subrow)[2]
			

			if min_overflight == overflightID :
				if tds_frame_type == 'insync_tf_0' :
					insync_a_frames += tds_frame_count
				elif tds_frame_type == 'insync_tf_32' :
					insync_b_frames += tds_frame_count
				elif tds_frame_type == 'outasync_tf' :
					outasync_frames += tds_frame_count
					
				if tds_beginERT < final_beginERT :
					final_beginERT = tds_beginERT
			
				if tds_endERT > final_endERT :
					final_endERT = tds_endERT

					
		final_beginERT = datetime.datetime.strftime(final_beginERT, "%Y-%jT%H:%M:%S.%f")
		final_endERT = datetime.datetime.strftime(final_endERT, "%Y-%jT%H:%M:%S.%f")
					
		#new_cols = ['DSS', 'Start ERT', 'End ERT', 'Pub', 'insync_tf_0', 'insync_tf_32', 'outasync_tf']
		new_list = [item[0], final_beginERT, final_endERT, tds_relay, insync_a_frames, insync_b_frames, outasync_frames]
				
		x = row.values.tolist() + new_list
		
		final_df = final_df.append(pd.Series(x, index=all_cols), ignore_index=True)
	
	else :
	
		# actual data volume in GDS ES database is 0
		# so there is no data for this pass in the TDS
		final_df = final_df.append(pd.Series(row, index=old_cols), ignore_index=True)
	
	count = count + 1
	print(str(count) + " / " + str(old_df.shape[0]), end="\r")
	
final_df.to_csv("final_dataframe.csv")
	