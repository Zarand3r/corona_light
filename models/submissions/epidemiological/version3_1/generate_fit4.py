############################################
# Adapted from code written by August Chen #
############################################
import pandas as pd
import numpy as np
import os
import csv
import datetime
import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(0, f"{homedir}" + '/models/submissions/processing/')
sys.path.insert(1, f"{homedir}" + '/models/epidemiological/production')
import fit_counties3_1
import formatter2

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

def generate_submission(start, end, guesses=None, bias=False, weight=False, policy_regime=False, tail_regime=False, death_metric="deaths", adaptive=False, sub_id="0"):
	submission = []
	output_dict = fit_counties3_1.multi_submission(end, guesses=guesses, bias=bias, weight=weight, policy_regime=policy_regime, tail_regime=tail_regime, death_metric=death_metric, adaptive=adaptive, getbounds=False) 
	counties_dates = output_dict["counties_dates"]
	counties_death_errors = output_dict["counties_death_errors"]
	counties_fips = output_dict["counties_fips"]
	nonconvergent = output_dict["nonconvergent"]
	for i in range(len(counties_fips)):
		county_prediction = format_submission(counties_dates[i], counties_death_errors[i], counties_fips[i], start)
		submission = submission + county_prediction
	# header = "{},{},{},{},{},{},{},{},{},{}\n".format("id", "10", "20", "30", "40", "50", "60", "70", "80", "90")
	output_file = f'{homedir}/models/submissions/epidemiological/version3_1/fits/predictions{sub_id}.csv'
	header = ["id", "10", "20", "30", "40", "50", "60", "70", "80", "90"]
	with open(output_file, 'w') as submission_file:
		writer = csv.writer(submission_file, delimiter=',')
		writer.writerow(header)
		writer.writerows(submission)
	formatter2.reformat(output_file, save=True, fix=False, id=sub_id)

	import json
	parameters= output_dict["parameters"]
	parameter_file = f"{homedir}/models/submissions/epidemiological/version3_1/parameters/parameters{sub_id}.csv"
	with open(parameter_file, 'w') as file:
		file.write(json.dumps(parameters))

if __name__ == '__main__':
	start = datetime.datetime(2020, 4, 1)
	end = datetime.datetime(2020, 6, 30)
	guesses1 = [1.41578513e-01, 1.61248129e-01, 2.48362028e-01, 3.42978127e-01, 5.79023652e-01, 4.64392758e-02, \
	9.86745420e-06, 4.83700388e-02, 4.85290835e-01, 3.72688900e-02, 4.92398129e-04, 5.20319673e-02, \
	4.16822944e-02, 2.93718207e-02, 2.37765976e-01, 6.38313283e-04, 1.00539865e-04, 7.86113867e-01, \
	3.26287443e-01, 8.18317732e-06, 5.43511913e-10, 1.30387168e-04, 3.58953133e-03, 1.57388153e-05]
	guesses2 = [0.02617736443427591, 0.17255447311461145, 0.15215935309382572, 0.21639011562137145, 0.6814820048990581, \
	0.20502517812934218, 3.3437178707695294e-05, 0.02698465330273812, 0.6410113879774412, 0.0003028925057859545, \
	0.3134893862413215, 0.06970602089626211, 0.42179760229195923, 0.009272596143914662, 0.258962882347026, \
	4.811125145762032e-09, 0.003859238158274466, 0.7716354446714161, 0.23179542329093872, 0.00017236677811295644, \
	0.005038783003615411, 2.683729877737938e-05, 5.3017766786399385e-11, 0.000759771263]
	generate_submission(start, end, guesses=guesses1, bias=True, weight=True, policy_regime=True, tail_regime=False, death_metric="deaths", adaptive=True, sub_id="4_1")
	generate_submission(start, end, guesses=guesses2, bias=True, weight=True, policy_regime=True, tail_regime=False, death_metric="deaths", adaptive=True, sub_id="4_2")
