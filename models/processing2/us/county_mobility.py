import pandas as pd
import numpy as np
import os
import datetime
import git
from pathlib import Path

repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
inputdir = f"{homedir}" + "/data/us/mobility/"
inputdir2 = f"{homedir}" + "/data/google_mobility/"
outputdir = f"{homedir}" + "/models/data/us/mobility/"
Path(outputdir).mkdir(parents=True, exist_ok=True)
outputdir += "county_"

def edit_column_date(frame,index):
    #Edits the date format of columns of dataframes
    #index: index of the first column of dates + 1
    i = 0
    for col in frame:
        i += 1
        if i >= index:
            new_d = date_format(col)
            frame = frame.rename(columns={col : new_d})
    return frame

def sort_dates(frame,index):
    #Sorts the columns by date of a frame with many nonconsecutive dates (several factors per date)
    Beg = list(frame.columns[:index]) #First four entries
    End = list(np.sort(np.array(frame.columns[index:]))) #Every Date Sorted
    cols = list(Beg + End) #Ordered Columns

    frame = frame[cols]
    return frame

def date_format(date):
    d = datetime.datetime.strptime(date, '%Y-%m-%d')
    return datetime.date.strftime(d, "%m/%d/%y")

def main():
    #Loading in mobility data
    DL_us_m50 = pd.read_csv(inputdir+'DL-us-m50.csv', encoding='latin1')
    DL_us_m50_index = pd.read_csv(inputdir+'DL-us-m50_index.csv', encoding='latin1')
    DL_us_samples = pd.read_csv(inputdir+'DL-us-samples.csv')

    #Cleaning the datasets
    DL_us_m50 = edit_column_date(DL_us_m50,6)
    DL_us_m50_index = edit_column_date(DL_us_m50_index,6)
    DL_us_samples = edit_column_date(DL_us_samples,6)
    DL_us_m50 = DL_us_m50.drop(columns=['country_code','admin_level','admin1','admin2'])
    DL_us_m50_index = DL_us_m50_index.drop(columns=['country_code','admin_level','admin1','admin2'])
    DL_us_samples = DL_us_samples.drop(columns=['country_code','admin_level','admin1','admin2'])

    #Separating data into county info
    DL_us_m50_County = DL_us_m50[DL_us_m50.fips >= 1000]
    DL_us_m50_index_County = DL_us_m50_index[DL_us_m50_index.fips >= 1000]
    DL_us_samples_County = DL_us_samples[DL_us_samples.fips >= 1000]

    #merging the 3 datasets together
    Mobility_County = pd.merge(DL_us_m50_County, DL_us_m50_index_County, left_on='fips', right_on='fips', suffixes=('_M_m50', ''), sort=True)
    Mobility_County = pd.merge(Mobility_County, DL_us_samples_County, left_on='fips', right_on='fips', suffixes=('_M_idx', '_M_samples'), sort=True)
    Mobility_County = Mobility_County[Mobility_County.fips >= -1]
    Mobility_County.columns = Mobility_County.columns.str.replace('fips','FIPS')
    #saving datasets with 3 values not consecutive and then consecutive
    Mobility_County_Nonconsecutive = Mobility_County
    Mobility_County_Consecutive = sort_dates(Mobility_County,1)
    #MAking FIPS the main index
    Mobility_County_Consecutive = Mobility_County_Consecutive.set_index('FIPS')
    Mobility_County_Nonconsecutive = Mobility_County_Nonconsecutive.set_index('FIPS')

    Mobility_County_Consecutive.to_csv(outputdir+'consecutive.csv')
    Mobility_County_Nonconsecutive.to_csv(outputdir+'nonconsecutive.csv')



    #New Google Mobility Data, must be processed
    google_mobility = pd.read_csv(inputdir2+'mobility_report_US.csv', encoding='latin1')
    #Taking only county data
    google_mobility_county = google_mobility[google_mobility['Region'] != 'Total']
    #Key to map counties to FIPS, and states to state abbreviations
    Key =  pd.read_csv('county_key.csv').sort_values(by=['FIPS'])
    State_Abv = pd.read_csv('State_Abbrev.csv')
    State_Abv = np.array(State_Abv)
    #Dictionary from state names to state initials
    State_Dict = dict((rows[0],rows[2]) for rows in State_Abv)
    #Changing the state column of google mobility to its abbreviation code
    google_mobility_county = google_mobility_county.replace({'State': State_Dict})
    #Creating a location column, to make the google mobility locations unique
    google_mobility_county['loc'] = google_mobility_county.Region.astype(str).str.cat(google_mobility_county.State.astype(str), sep=', ')
    Key['loc'] = Key.COUNTY.astype(str).str.cat(Key.ST.astype(str), sep=', ')
    #New google county mobility data, with fips codes attached
    google_county = pd.merge(google_mobility_county, Key, left_on='loc', right_on='loc', sort=True)
    #removing unecessary columns
    google_county = google_county.drop(columns=['State','Region','ST','COUNTY','loc'])

    #Splitting up this google county into its components to rejoin it later
    google_residential = google_county.pivot(index='FIPS', columns='Date', values=['Residential'])
    google_residential.to_csv(outputdir+'google_residential.csv')
    #Reading in split up component and the resetting the header values
    google_residential = pd.read_csv(outputdir+'google_residential.csv',header=1).iloc[1:].rename(columns={'Date':'FIPS'})

    google_workplaces = google_county.pivot(index='FIPS', columns='Date', values=['Workplaces'])
    google_workplaces.to_csv(outputdir+'google_workplaces.csv')
    google_workplaces = pd.read_csv(outputdir+'google_workplaces.csv',header=1).iloc[1:].rename(columns={'Date':'FIPS'})

    google_transit = google_county.pivot(index='FIPS', columns='Date', values=['Transit stations'])
    google_transit.to_csv(outputdir+'google_transit.csv')
    google_transit = pd.read_csv(outputdir+'google_transit.csv',header=1).iloc[1:].rename(columns={'Date':'FIPS'})

    google_parks = google_county.pivot(index='FIPS', columns='Date', values=['Parks'])
    google_parks.to_csv(outputdir+'google_parks.csv')
    google_parks = pd.read_csv(outputdir+'google_parks.csv',header=1).iloc[1:].rename(columns={'Date':'FIPS'})

    google_grocery = google_county.pivot(index='FIPS', columns='Date', values=['Grocery & pharmacy'])
    google_grocery.to_csv(outputdir+'google_grocery.csv')
    google_grocery = pd.read_csv(outputdir+'google_grocery.csv',header=1).iloc[1:].rename(columns={'Date':'FIPS'})

    google_retail = google_county.pivot(index='FIPS', columns='Date', values=['Retail & recreation'])
    google_retail.to_csv(outputdir+'google_retail.csv')
    google_retail = pd.read_csv(outputdir+'google_retail.csv',header=1).iloc[1:].rename(columns={'Date':'FIPS'})

    #Merging the data back together
    google_county = pd.merge(google_residential, google_workplaces, left_on='FIPS', right_on='FIPS', suffixes=('_residential', ''))
    google_county = pd.merge(google_county, google_transit, left_on='FIPS', right_on='FIPS', suffixes=('_workplaces', ''))
    google_county = pd.merge(google_county, google_parks, left_on='FIPS', right_on='FIPS', suffixes=('_transit', ''))
    google_county = pd.merge(google_county, google_grocery, left_on='FIPS', right_on='FIPS', suffixes=('_parks', ''))
    google_county = pd.merge(google_county, google_retail, left_on='FIPS', right_on='FIPS', suffixes=('_grocery', 'retail'))

    #saving google dataset with each component either not consecutive and then consecutive
    google_county_Nonconsecutive = google_county
    google_county_Consecutive = sort_dates(google_county,1)

    #MAking FIPS the main index
    google_county_Consecutive = google_county_Consecutive.set_index('FIPS')
    google_county_Nonconsecutive = google_county_Nonconsecutive.set_index('FIPS')

    google_county_Consecutive.to_csv(outputdir+'google_consecutive.csv')
    google_county_Nonconsecutive.to_csv(outputdir+'google_nonconsecutive.csv')

if __name__ == '__main__':
    main()

