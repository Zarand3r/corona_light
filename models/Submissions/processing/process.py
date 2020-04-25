import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
import datetime
from datetime import timedelta
import csv


import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
# FIXES NANs

def fix_nans(submission, save=False):
	submission.fillna(method='ffill', inplace=True)
	# # Do this stuff in the code that generates the error bounds instead of here
	# num = submission._get_numeric_data()
	# num[num < 0.1] = 0.00
	if save:
		submission.to_csv("modified_submission.csv", index=False)

def reformat(file1, file2, save=True):
	submission = pd.read_csv(file1, index_col=False)
	modified_submission = pd.read_csv(file2, index_col=False)
	fix_nans(submission)
	# convert to floats
	submission[["10", "20", "30", "40", "50", "60", "70", "80", "90"]] = submission[["10", "20", "30", "40", "50", "60", "70", "80", "90"]].apply(pd.to_numeric)

	final_submission = []

	forecast_dict = {}
	for index, row in submission.iterrows():
		current_id = row["id"]
		forecast_dict[current_id] = row.values

	end = len(modified_submission)
	# for index, row in islice(modified_submission.iterrows(), start, end):
	for index, row in modified_submission.iterrows():
		print(f"{index} / {end}")
		current_id = row["id"]
		replacement = forecast_dict.pop(current_id, [current_id, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]) 
		final_submission.append(replacement)

	if save:
		header = ["id", "10", "20", "30", "40", "50", "60", "70", "80", "90"]
		with open('formatted_submission.csv', 'w') as submission_file:
			writer = csv.writer(submission_file, delimiter=',')
			writer.writerow(header)
			writer.writerows(final_submission)

	

if __name__ == '__main__':

	reformat("../predictions.csv", f"{homedir}/sample_submission.csv")
