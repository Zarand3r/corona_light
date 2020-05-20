import pandas as pd 
import numpy as np
import sys
import traceback
from tqdm.auto import tqdm
import os
import csv
import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(1, f"{homedir}" + '/models/data_processing')
import loader

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
	disabled_fips = set({})
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

if __name__ == '__main__':
	start_date = '2020-05-07'
	latest_date = '2020-05-19'
	submissions = ['../epidemiological/version3_0/old_submissions/submission3_0_0.csv', '../epidemiological/version3_0/old_submissions/submission3_0_1.csv', '../epidemiological/version3_0/old_submissions/submission3_0_2.csv', f'{homedir}/sample_submission.csv']
	# submissions = [f"{homedir}"+ '/sample_submission.csv', '../epidemiological/version3_0/submission3_0_0.csv', '../epidemiological/version3_0/submission3_0_1.csv', '../epidemiological/version3_0/submission3_0_2.csv']
	new_submissions = ['../epidemiological/version3_0/new_submissions/submission3_0_0.csv', '../epidemiological/version3_0/new_submissions/submission3_0_1.csv', '../epidemiological/version3_0/new_submissions/submission3_0_2.csv', f'{homedir}/sample_submission.csv']
	scores = []
	
	for submission in submissions:
		score = evaluator(submission, start_date)
		scores.append(score)

	baseline = scores[0]
	optimal_submission = {}
	for county in list(baseline.keys()):
		best = baseline[county]
		best_index = 0
		for index, score in enumerate(scores):
			if score[county] < best:
				best = score[county]
				best_index = index
			if best_index == 3:
				print(county)
		optimal_submission[county] = best_index


	submission_files = []
	for submission in new_submissions:
		submission_file = pd.read_csv(submission, index_col=False)
		submission_files.append(submission_file)

	baseline_file = submission_files[0]
	ultimate_submission = []

	total = len(baseline_file)
	for index, row in baseline_file.iterrows():
		print(f"{index+1} / {total}")
		county = row["id"].split('-')[-1]
		optimal_file_index = optimal_submission[county]
		if optimal_file_index == 3:
			date = row["id"][0:10]
			day = date.split('-')[-1]
			month = date.split('-')[-2]
			if int(day) <= int(latest_date.split('-')[-1]) or int(month) <= int(latest_date.split('-')[-2]):
				optimal_file_index = 0
		optimal_file = submission_files[optimal_file_index]
		ultimate_submission.append(list(optimal_file.iloc[[index]].values[0]))
		

	output_file = f'{homedir}/models/submissions/epidemiological/version3_0/optimize/combined.csv'
	header = ["id", "10", "20", "30", "40", "50", "60", "70", "80", "90"]
	with open(output_file, 'w') as submission_file:
		writer = csv.writer(submission_file, delimiter=',')
		writer.writerow(header)
		writer.writerows(ultimate_submission)

	combined = pd.read_csv(output_file, index_col=False)
	combined[["10", "20", "30", "40", "50", "60", "70", "80", "90"]] = combined[["10", "20", "30", "40", "50", "60", "70", "80", "90"]].apply(pd.to_numeric)
	combined.to_csv(output_file, index=False)


	evaluator("combined.csv", latest_date)





