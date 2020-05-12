import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
import datetime
from datetime import timedelta
import os
import csv


import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
# FIXES NANs

def fix_nans(submission):
	submission.fillna(method='ffill', inplace=True)
	# Do this stuff in the code that generates the error bounds instead of here
	num = submission._get_numeric_data()
	num[num < 0.1] = 0.00

def fix_submission(submission, file_path):
	pop = pd.read_csv(f"{homedir}/data/us/demographics/county_populations.csv")
	submission['Date'] = submission.id.str.rsplit("-", n = 1, expand = True)[0]
	submission['FIPS'] = submission.id.str.rsplit("-", n = 1, expand = True)[1].astype(int)
	submission = submission.join(pop.set_index('FIPS'), on='FIPS')
	submission['60plus'] = submission['60plus'].fillna(2000)
	submission['60plus'] = submission['60plus'].astype(int)

	temp = submission.copy()
	Large_10 = temp.loc[submission['10'] >= 0.075*submission['total_pop']]
	temp.loc[Large_10.index.values,'10'] = Large_10.total_pop.values*0.075
	temp.loc[Large_10.index.values,'20'] = Large_10.total_pop.values*(1.1)**1*0.075
	temp.loc[Large_10.index.values,'30'] = Large_10.total_pop.values*(1.1)**2*0.075
	temp.loc[Large_10.index.values,'40'] = Large_10.total_pop.values*(1.1)**3*0.075
	temp.loc[Large_10.index.values,'50'] = Large_10.total_pop.values*(1.1)**4*0.075
	temp.loc[Large_10.index.values,'60'] = Large_10.total_pop.values*(1.1)**5*0.075
	temp.loc[Large_10.index.values,'70'] = Large_10.total_pop.values*(1.1)**6*0.075
	temp.loc[Large_10.index.values,'80'] = Large_10.total_pop.values*(1.1)**7*0.075
	temp.loc[Large_10.index.values,'90'] = Large_10.total_pop.values*(1.1)**8*0.075


	Large_Rat = temp.loc[temp['90']/temp['10'] >= 10]
	Meanval = (temp.loc[Large_Rat.index.values,'10'] + temp.loc[Large_Rat.index.values,'20'] + temp.loc[Large_Rat.index.values,'30'])/3
	temp.loc[Large_Rat.index.values,'10'] = Meanval/(1.2)
	temp.loc[Large_Rat.index.values,'20'] = Meanval*(1.2)**0
	temp.loc[Large_Rat.index.values,'30'] = Meanval*(1.2)**1
	temp.loc[Large_Rat.index.values,'40'] = Meanval*(1.2)**2
	temp.loc[Large_Rat.index.values,'50'] = Meanval*(1.2)**3
	temp.loc[Large_Rat.index.values,'60'] = Meanval*(1.2)**4
	temp.loc[Large_Rat.index.values,'70'] = Meanval*(1.2)**5
	temp.loc[Large_Rat.index.values,'80'] = Meanval*(1.2)**6
	temp.loc[Large_Rat.index.values,'90'] = Meanval*(1.2)**7

	temp = temp[['id','10','20','30','40','50','60','70','80','90']]
	temp.to_csv(file_path)


def reformat(file1, file2=f"{homedir}/sample_submission.csv", save=True, fix=False, id=""):
	output_path = os.path.dirname(file1) + f'/submission{id}.csv'
	submission = pd.read_csv(file1, index_col=False)
	sample_submission = pd.read_csv(file2, index_col=False)
	fix_nans(submission)
	# convert to floats
	submission[["10", "20", "30", "40", "50", "60", "70", "80", "90"]] = submission[["10", "20", "30", "40", "50", "60", "70", "80", "90"]].apply(pd.to_numeric)

	final_submission = []

	forecast_dict = {}
	for index, row in submission.iterrows():
		current_id = row["id"]
		forecast = row.values
		forecast_dict[current_id] = row.values

	end = len(sample_submission)
	# for index, row in islice(sample_submission.iterrows(), start, end):
	for index, row in sample_submission.iterrows():
		# print(f"{index} / {end}")
		current_id = row["id"]
		replacement = forecast_dict.pop(current_id, [current_id, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]) 
		final_submission.append(replacement)

	if save:
		header = ["id", "10", "20", "30", "40", "50", "60", "70", "80", "90"]
		with open(output_path, 'w') as submission_file:
			writer = csv.writer(submission_file, delimiter=',')
			writer.writerow(header)
			writer.writerows(final_submission)

	if fix:
		fix_submission(submission, output_path)

	

if __name__ == '__main__':
	reformat("../model1/version2_1/daily_predictions2_1_0.csv", f"{homedir}/sample_submission.csv", id="0")
