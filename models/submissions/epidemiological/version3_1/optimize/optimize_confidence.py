import pandas as pd 
import numpy as np
import sys
import traceback
from tqdm.auto import tqdm
import datetime
import os
import csv
import json
import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(1, f"{homedir}" + '/models/data_processing')
import loader
sys.path.insert(0, f"{homedir}" + '/models/submissions/processing/')
sys.path.insert(1, f"{homedir}" + '/models/epidemiological/production')
import fit_counties3_1
import formatter2

def get_date(x):
	return '-'.join(x.split('-')[:3])
def get_fips(x):
	return x.split('-')[-1]
def pinball_loss(y_true, y_pred, quantile = 0.5):
	delta = y_true - y_pred
	loss_above = np.sum(delta[delta > 0])*(quantile)
	loss_below = np.sum(-1*delta[delta < 0])*(1-quantile)
	return (loss_above + loss_below) / len(y_true)
def pinball_loss2(y_true, y_pred, size, quantile = 0.5):
	delta = y_true - y_pred
	if delta > 0:
		loss = delta*quantile
	else:
		loss = -1*delta*(1-quantile)
	return loss / size
def evaluate(test_df, user_df):
	join_df = test_df.join(user_df, how = 'inner')
	if(len(join_df) != len(test_df)):
		sys.stderr.write("Submission not right length. \n")
		raise Exception("Submission not right length")
	if(user_df.isna().sum().sum() > 0 ):
		sys.stderr.write("Submission contains NaN. \n")
		raise Exception("Submission Contains NaN.")
	if(join_df.index.equals(test_df.index) == False):
		sys.stderr.write("Incorrect ID format in Submission. \n")
		raise Exception("Incorrect ID format.")
	total_loss = 0

	for column in ['10','20','30','40','50', '60', '70', '80', '90']:
		quantile = int(column) / 100.0
		loss = pinball_loss(join_df['deaths'].values, join_df[column].values, quantile) / 9.0
		total_loss += loss
	return total_loss

def evaluate2(test_df, user_df):
	county_losses = {}

	join_df = test_df.join(user_df, how = 'inner')
	if(len(join_df) != len(test_df)):
		sys.stderr.write("Submission not right length. \n")
		raise Exception("Submission not right length")
	if(user_df.isna().sum().sum() > 0 ):
		sys.stderr.write("Submission contains NaN. \n")
		raise Exception("Submission Contains NaN.")
	if(join_df.index.equals(test_df.index) == False):
		sys.stderr.write("Incorrect ID format in Submission. \n")
		raise Exception("Incorrect ID format.")
	total_loss = 0
	
	size = len(join_df['deaths'].values)
	for index, row in join_df.iterrows():
		county = index.split('-')[-1]
		county_loss = 0
		for column in ['10','20','30','40','50', '60', '70', '80', '90']:
			quantile = int(column) / 100.0
			# if county == '36061':
			#     print(f"{row[column]} versus {row['deaths']}")
			loss = pinball_loss2(row['deaths'], row[column], size, quantile) / 9.0
			county_loss += loss
			total_loss += loss
		if county in county_losses.keys():
			county_losses[county] += county_loss
		else:
			county_losses[county] = county_loss

	return total_loss, county_losses

def evaluator(submission, start_date):
	print(f"scoring {submission}")
	daily_df = pd.read_csv(f"{homedir}" + '/data/us/covid/nyt_us_counties_daily.csv')
	# daily_df = pd.read_csv(f"{homedir}" + '/data/us/covid/nyt_us_counties.csv')
	daily_df.loc[daily_df["county"]=='New York City', "fips"]=36061
	daily_df.dropna(subset=['fips'], inplace=True)
	daily_df['fips'] = daily_df['fips'].astype(int)
	end_date = daily_df['date'].max()
	daily_df['id'] = daily_df['date'] +'-'+ daily_df['fips'].astype(str)
	preperiod_df = daily_df[(daily_df['date'] < start_date)]
	daily_df = daily_df[(daily_df['date'] <= end_date)  & (daily_df['date'] >= start_date)]

	sample_submission = pd.read_csv(f"{homedir}"+ '/sample_submission.csv') # Load the sample submission with all 0's
	sample_submission['date'] = sample_submission['id'].apply(get_date)
	sample_submission['fips'] = sample_submission['id'].apply(get_fips).astype('int')
	sample_submission = sample_submission[(sample_submission['date'] <= end_date)  & (sample_submission['date'] >= start_date)]

	# Disabled FIPS is a set of FIPS to avoid scoring. Covid_active_fips is where there has been reports of covid, 
	# and inactive_fips are fips codes present in sample submission but with no cases reported by the New York Times.
	# New_active_fips are FIPS that were introduced into the dataset during the scoring period. 
	# Active FIPS should be scored against deaths data from NYT if such data is available, 
	# but Inactive FIPS should be scored with a target of 0.
	disabled_fips = set({
	## NEW YORK
    36005, 36047, 36081, 36085, 
    ## Peurto Rico
    72001, 72003, 72005, 72007, 72009, 72011, 72013, 72015, 72017,
    72019, 72021, 72023, 72025, 72027, 72029, 72031, 72033, 72035,
    72037, 72039, 72041, 72043, 72045, 72047, 72049, 72051, 72053,
    72054, 72055, 72057, 72059, 72061, 72063, 72065, 72067, 72069,
    72071, 72073, 72075, 72077, 72079, 72081, 72083, 72085, 72087,
    72089, 72091, 72093, 72095, 72097, 72099, 72101, 72103, 72105,
    72107, 72109, 72111, 72113, 72115, 72117, 72119, 72121, 72123,
    72125, 72127, 72129, 72131, 72133, 72135, 72137, 72139, 72141,
    72143, 72145, 72147, 72149, 72151, 72153,
    ## Virgin Islands
    78010, 78020, 78030})
	prev_active_fips = set(preperiod_df.fips.unique())
	curr_active_fips = set(daily_df.fips.unique())
	all_fips = set(sample_submission.fips.unique())
	covid_active_fips = prev_active_fips.intersection(all_fips).intersection(curr_active_fips) - disabled_fips
	inactive_fips = all_fips - prev_active_fips - curr_active_fips - disabled_fips
	new_active_fips = (curr_active_fips - prev_active_fips).intersection(all_fips) - disabled_fips

	# Create a DataFrame of all 0's for inactive fips by getting those from sample submission.
	inactive_df = sample_submission.set_index('fips')[['id','50']].loc[inactive_fips]
	inactive_df = inactive_df.set_index('id').rename({'50':'deaths'}, axis = 1)
	assert(inactive_df.sum().sum() == 0)
	# Create a DataFrame of active fips from the New York Times data
	active_df = daily_df.set_index('fips')[['id', 'deaths']].loc[covid_active_fips].set_index('id')

	# Create dataframe for new fips
	sample_search = sample_submission.set_index('fips')[['id','50']].rename({'50':'deaths'}, axis = 1)
	daily_search = daily_df.set_index('fips')
	new_df_arr = []
	for fips in new_active_fips:
		tmp_sample = sample_search.loc[[fips]].set_index('id')
		tmp_daily = daily_search.loc[[fips]].set_index('id')
		tmp_sample.update(tmp_daily)
		tmp_sample = tmp_sample[tmp_sample.index <= tmp_daily.index.max()]
		new_df_arr.append(tmp_sample)

	# Join the data frames
	example = None
	if(len(new_active_fips) > 0):
		new_df = pd.concat(new_df_arr)
		example = pd.concat([inactive_df, active_df, new_df]).sort_index()
	else:
		example = pd.concat([inactive_df, active_df]).sort_index()


	# Read some CSV for score
	df = pd.read_csv(submission).set_index('id').sort_index()
	# score = evaluate(example[['deaths']], df)
	score, county_losses = evaluate2(example[['deaths']], df)
	print('Got score of {:.6f}'.format(score))
	mean = np.mean(list(county_losses.values()))
	deviation = np.std(list(county_losses.values()))
	for county in county_losses:
		if county_losses[county] > mean+deviation:
			print(f"{county} score: {county_losses[county]}")

	return county_losses

# hashtable with month and number of days in the month
maxMonth = {1:31, 2:29, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}

# gets next day, needed because the current day could be the last of the month
def next_day(current_day):
	# assumes that everything is in 2020
	if current_day.day < maxMonth[current_day.month]:
		return datetime.datetime(2020, current_day.month, current_day.day + 1)
	else:
		return datetime.datetime(2020, current_day.month + 1, 1)

def previous_day(current_day):
	# assumes that everything is in 2020
	if current_day.day >= 1:
		return datetime.datetime(2020, current_day.month, current_day.day-1)
	else:
		previous_month = current_day.month - 1
		return datetime.datetime(2020, previous_month, maxMonth[previous_month])

# we want formatting in the form 2020-04-01, with 0s before months, days < 10
def formatter(numb):
	if numb < 10:
		return "0" + str(numb)
	else:
		return str(numb)

def format_submission(dates, death_errors, fips, start, transpose=False):
	dates = dates.tolist()
	
	if transpose:
		# swap columns and rows for death_errors
		death_errors = np.array(death_errors)
		death_errors = death_errors.T
		death_errors = death_errors.tolist()

	death_errors = death_errors.tolist()
	
	# trim both lists so they begin with date represented by start
	# assumes the lists begin originally at the same place
	start_index = -1
	for i in range(0, len(dates)):
		current_day = dates[i]
		if current_day.month == start.month and current_day.day == start.day:
			start_index = i
			break

	if start_index == -1: # start doesn't exist in dates
		initial_date = dates[0]
		difference = initial_date.day - start.day
		for i in range(difference):
			dates.insert(0, previous_day(initial_date))
			initial_date = dates[0]
			death_errors = [[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]+death_errors
		start_index = 0

	# adding to dates so lengths match up
	final_date = dates[-1]
	while len(dates) < len(death_errors):
		dates.append(next_day(final_date))
		final_date = dates[-1]
	
		
	death_errors = death_errors[start_index:]
	dates = dates[start_index:]
	# convert dates from datetime to string, add fips code
	for i in range(len(dates)):
		day = dates[i]
		day_format = '{year}-{month}-{day}-{fips}'.format(year = day.year,
														  month = formatter(day.month), 
														  day = formatter(day.day), 
														  fips = fips)
		dates[i] = day_format
		if i < len(death_errors):
			death_errors[i].insert(0, dates[i])
		
	return death_errors

def generate_confidence(combined_parameters, quick=True, error_start=-14, tail=False, fix_nonconvergent=False, sub_id="0"):
	start = datetime.datetime(2020, 4, 1)
	end = datetime.datetime(2020, 6, 30)
	submission = []
	output_dict = fit_counties3_1.multi_generate_confidence(combined_parameters, end, quick=quick, error_start=error_start, tail=tail, fix_nonconvergent=fix_nonconvergent) #do regime next but not ready for fitsQ
	counties_dates = output_dict["counties_dates"]
	counties_death_errors = output_dict["counties_death_errors"]
	print(list(counties_death_errors))
	counties_fips = output_dict["counties_fips"]
	nonconvergent = output_dict["nonconvergent"]
	for i in range(len(counties_fips)):
		county_prediction = format_submission(counties_dates[i], counties_death_errors[i], counties_fips[i], start)
		submission = submission + county_prediction
	# header = "{},{},{},{},{},{},{},{},{},{}\n".format("id", "10", "20", "30", "40", "50", "60", "70", "80", "90")
	output_file = f'{homedir}/models/submissions/epidemiological/version3_1/confidences/predictions{sub_id}.csv'
	header = ["id", "10", "20", "30", "40", "50", "60", "70", "80", "90"]
	with open(output_file, 'w') as submission_file:
		writer = csv.writer(submission_file, delimiter=',')
		writer.writerow(header)
		writer.writerows(submission)

	formatter2.reformat(output_file, save=True, fix=False, id=sub_id)

if __name__ == '__main__':
	# do the code for combine_preidctions first
	# then do the code comparing different weight parameters 
	# Make this into a batch script to automate both 
	start_date = '2020-05-12'
	latest_date = '2020-05-25'
	submissions = [f'{homedir}/models/submissions/epidemiological/version3_1/fits/submission1_1.csv', f'{homedir}/models/submissions/epidemiological/version3_1/fits/submission1_2.csv',\
	f'{homedir}/models/submissions/epidemiological/version3_1/fits/submission2_1.csv', f'{homedir}/models/submissions/epidemiological/version3_1/fits/submission2_2.csv',\
	f'{homedir}/models/submissions/epidemiological/version3_1/fits/submission3_1.csv', f'{homedir}/models/submissions/epidemiological/version3_1/fits/submission3_2.csv',\
	f'{homedir}/models/submissions/epidemiological/version3_1/fits/submission4_1.csv', f'{homedir}/models/submissions/epidemiological/version3_1/fits/submission4_2.csv']
	guesses1 = [1.41578513e-01, 1.61248129e-01, 2.48362028e-01, 3.42978127e-01, 5.79023652e-01, 4.64392758e-02, \
	9.86745420e-06, 4.83700388e-02, 4.85290835e-01, 3.72688900e-02, 4.92398129e-04, 5.20319673e-02, \
	4.16822944e-02, 2.93718207e-02, 2.37765976e-01, 6.38313283e-04, 1.00539865e-04, 7.86113867e-01, \
	3.26287443e-01, 8.18317732e-06, 5.43511913e-10, 1.30387168e-04, 3.58953133e-03, 1.57388153e-05]
	guesses2 = [0.02617736443427591, 0.17255447311461145, 0.15215935309382572, 0.21639011562137145, 0.6814820048990581, \
	0.20502517812934218, 3.3437178707695294e-05, 0.02698465330273812, 0.6410113879774412, 0.0003028925057859545, \
	0.3134893862413215, 0.06970602089626211, 0.42179760229195923, 0.009272596143914662, 0.258962882347026, \
	4.811125145762032e-09, 0.003859238158274466, 0.7716354446714161, 0.23179542329093872, 0.00017236677811295644, \
	0.005038783003615411, 2.683729877737938e-05, 5.3017766786399385e-11, 0.000759771263]
	submissions_args = {0: {"bias":True, "weight":True, "policy_regime":False, "tail_regime":False, "death_metric":"deaths", "adaptive":False}, 1: {"bias":True, "weight":True, "policy_regime":False, "tail_regime":False, "death_metric":"deaths", "adaptive":False},\
	2: {"bias":True, "weight":True, "policy_regime":False, "tail_regime":False, "death_metric":"avg_deaths", "adaptive":False}, 3: {"bias":True, "weight":True, "policy_regime":False, "tail_regime":False, "death_metric":"avg_deaths", "adaptive":False},\
	4: {"bias":True, "weight":True, "policy_regime":False, "tail_regime":True, "death_metric":"deaths", "adaptive":True}, 5: {"bias":True, "weight":True, "policy_regime":False, "tail_regime":True, "death_metric":"deaths", "adaptive":True},\
	6: {"bias":True, "weight":True, "policy_regime":True, "tail_regime":False, "death_metric":"deaths", "adaptive":True}, 7: {"bias":True, "weight":True, "policy_regime":True, "tail_regime":False, "death_metric":"deaths", "adaptive":True}}
	# submissions = [f"{homedir}"+ '/sample_submission.csv', '../epidemiological/version3_1/submission3_1_0.csv', '../epidemiological/version3_1/submission3_1_1.csv', '../epidemiological/version3_1/submission3_1_2.csv']
	# new_submissions = ['../epidemiological/version3_1/submission3_1_0.csv', '../epidemiological/version3_1/submission3_1_1.csv', '../epidemiological/version3_1/submission3_1_2.csv', f'{homedir}/sample_submission.csv']
	
	parameter_files = [f'{homedir}/models/submissions/epidemiological/version3_1/parameters/parameters1_1.csv', f'{homedir}/models/submissions/epidemiological/version3_1/parameters/parameters1_2.csv',\
	f'{homedir}/models/submissions/epidemiological/version3_1/parameters/parameters2_1.csv', f'{homedir}/models/submissions/epidemiological/version3_1/parameters/parameters2_2.csv',\
	f'{homedir}/models/submissions/epidemiological/version3_1/parameters/parameters3_1.csv', f'{homedir}/models/submissions/epidemiological/version3_1/parameters/parameters3_2.csv',\
	f'{homedir}/models/submissions/epidemiological/version3_1/parameters/parameters4_1.csv', f'{homedir}/models/submissions/epidemiological/version3_1/parameters/parameters4_2.csv']

	scores = []
	for submission in submissions:
		score = evaluator(submission, start_date)
		scores.append(score)

	submission_parameters = []
	for parameter_file in parameter_files:
		input_path = parameter_file
		fopen = open(input_path,'r')
		data = fopen.read()
		json_decoded = data.replace("'", "\"")
		params_dict = json.loads(json_decoded)
		submission_parameters.append(params_dict)

	baseline = scores[0]
	combined_args = {}
	for county in list(baseline.keys()):
		best = baseline[county]
		best_index = 0
		for index, score in enumerate(scores):
			if score[county] < best:
				best = score[county]
				best_index = index
		best_parameters = submission_parameters[best_index]
		if county in list(best_parameters.keys()):
			county_args = dict(submissions_args[best_index])
			params = best_parameters[county][-1]
			county_args["params"] = params
		else:
			county_args = None
		combined_args[county] = county_args

	# # Now we have the best args for each county

	# generate the new_submissions (confidence) files using the function defined above, save to the new submissions file paths
	generate_confidence(combined_args, quick=True, error_start=-14, tail=False, fix_nonconvergent=True, sub_id="1")
	generate_confidence(combined_args, quick=True, error_start=-14, tail=-14, fix_nonconvergent=False, sub_id="2")


	new_submissions = [f'{homedir}/models/submissions/epidemiological/version3_1/confidences/submission1.csv', f'{homedir}/models/submissions/epidemiological/version3_1/confidences/submission2.csv', f'{homedir}/sample_submission.csv']

	scores = []
	for submission in new_submissions:
		score = evaluator(submission, start_date)
		scores.append(score)

	baseline = scores[0]
	scored_counties = list(baseline.keys())
	optimal_submission = {}
	for county in scored_counties:
		best = baseline[county]
		best_index = 0
		for index, score in enumerate(scores):
			if score[county] < best:
				best = score[county]
				best_index = index
		print(f"{county} submission {best_index} scores {best} over {baseline[county]}")
		optimal_submission[county] = best_index


	#### 
	baseline_submission = f'{homedir}/models/submissions/epidemiological/version3_1/fits/submission_baseline.csv' #this will have to have errors, not just predictions
	new_submissions.append(baseline_submission)
	submission_files = []
	for submission in new_submissions:
		submission_file = pd.read_csv(submission, index_col=False)
		submission_files.append(submission_file)

	baseline_file = submission_files[-1]
	ultimate_submission = []
	total = len(baseline_file)
	for index, row in baseline_file.iterrows():
		print(f"{index+1} / {total}")
		county = row["id"].split('-')[-1]
		date = row["id"][0:10]
		day = date.split('-')[-1]
		month = date.split('-')[-2]
		if county not in scored_counties or (int(day) <= int(latest_date.split('-')[-1]) and int(month) <= int(latest_date.split('-')[-2])):
			optimal_file_index = -1
			ultimate_submission.append(list(row.values))
			continue
		optimal_file_index = optimal_submission[county]
		optimal_file = submission_files[optimal_file_index]
		ultimate_submission.append(list(optimal_file.iloc[[index]].values[0]))
		

	output_file = f'{homedir}/models/submissions/epidemiological/version3_1/optimize/optimize_confidence.csv'
	header = ["id", "10", "20", "30", "40", "50", "60", "70", "80", "90"]
	with open(output_file, 'w') as submission_file:
		writer = csv.writer(submission_file, delimiter=',')
		writer.writerow(header)
		writer.writerows(ultimate_submission)

	combined = pd.read_csv(output_file, index_col=False)
	combined[["10", "20", "30", "40", "50", "60", "70", "80", "90"]] = combined[["10", "20", "30", "40", "50", "60", "70", "80", "90"]].apply(pd.to_numeric)
	combined.to_csv(output_file, index=False)


	score_date = '2020-05-25'
	evaluator("optimize_confidence.csv", score_date)




	# new_submissions = [f"{homedir}"+ '/sample_submission.csv', '../epidemiological/version3_1/old/submission3_1_0.csv', '../epidemiological/version3_1/old/submission3_1_1.csv', '../epidemiological/version3_1/old/moving_submission3_1_2.csv']
	# submission_files = []
	# for submission in new_submissions:
	# 	submission_file = pd.read_csv(submission, index_col=False)
	# 	submission_files.append(submission_file)

	# baseline_file = submission_files[0]
	# ultimate_submission = []

	# total = len(baseline_file)
	# for index, row in baseline_file.iterrows():
	#     print(f"{index+1} / {total}")
	#     county = row["id"].split('-')[-1]
	#     list1 = list(submission_files[1].iloc[[index]].values[0])
	#     list2 = list(submission_files[2].iloc[[index]].values[0])
	#     list3 = list(submission_files[3].iloc[[index]].values[0])
	#     zipped_list = zip(list1[1:], list2[1:], list3[1:])
	#     mean = [sum(item)/3 for item in zipped_list] 
	#     mean = [list1[0]] + mean
	#     ultimate_submission.append(mean)
		

	# output_file = f'{homedir}/models/submissions/processing/' + 'combined.csv'
	# header = ["id", "10", "20", "30", "40", "50", "60", "70", "80", "90"]
	# with open(output_file, 'w') as submission_file:
	#     writer = csv.writer(submission_file, delimiter=',')
	#     writer.writerow(header)
	#     writer.writerows(ultimate_submission)

	# combined = pd.read_csv(output_file)
	# combined[["10", "20", "30", "40", "50", "60", "70", "80", "90"]] = combined[["10", "20", "30", "40", "50", "60", "70", "80", "90"]].apply(pd.to_numeric)
	# combined.to_csv(output_file)







