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
import fit_counties
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
			death_errors = [[0,0,0,0,0,0,0,0,0]]+death_errors
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


if __name__ == '__main__':
	start = datetime.datetime(2020, 4, 1)
	end = datetime.datetime(2020, 6, 30)
	submission = []
	counties_dates, counties_death_errors, counties_fips = fit_counties.submission(end, weight=True, strict=True)
	for i in range(len(counties_fips)):
		county_prediction = format_submission(counties_dates[i], counties_death_errors[i], counties_fips[i], start)
		submission = submission + county_prediction
	# header = "{},{},{},{},{},{},{},{},{},{}\n".format("id", "10", "20", "30", "40", "50", "60", "70", "80", "90")
	output_file = 'predictions.csv'
	header = ["id", "10", "20", "30", "40", "50", "60", "70", "80", "90"]
	with open(output_file, 'w') as submission_file:
		writer = csv.writer(submission_file, delimiter=',')
		writer.writerow(header)
		writer.writerows(submission)

	formatter2.reformat(output_file, save=True, fix=True)

	# np.savetxt("submission1.csv", submission, delimiter=",", header=header)





