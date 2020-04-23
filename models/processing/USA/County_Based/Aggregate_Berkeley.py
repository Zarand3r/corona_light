import numpy as np
import pandas as pd
import os

repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
inputdir = f"{homedir}" + "/data/us/"
outputdir = f"{homedir}" + "/models/data/us/"
Path(outputdir).mkdir(parents=True, exist_ok=True)

"""helper functions for standardizing data format"""
def RatioToFrac(x):
    return x/(x+1)

def PercentToFrac(x):
    return x/100

def main():
    Berkeley = pd.read_csv(inputdir + 'aggregate_berkeley.csv')
    
    """Drop the first column which is meaningless."""
    Berkeley = Berkeley.drop(columns = ['Unnamed: 0'])
    
    """Improve readability."""
    Berkeley = Berkeley.rename(columns = {'Population(Persons)2017': 'Population2017'})
    Berkeley = Berkeley.rename(columns = {'PopTotalMale2017': 'PopMale2017'})
    Berkeley = Berkeley.rename(columns = {'PopTotalFemale2017': 'PopFemale2017'})
    Berkeley = Berkeley.rename(columns = {'MedianAge,Male2010': 'MedianAgeMale2010'})
    Berkeley = Berkeley.rename(columns = {'MedianAge,Female2010': 'MedianAgeFemale2010'})
    Berkeley = Berkeley.rename(columns = {'MedicareEnrollment,AgedTot2017':'MedicareEnrollmentTot2017'})
    Berkeley = Berkeley.rename(columns = {'Smokers_Percentage':'SmokersPercentage'})
    Berkeley = Berkeley.rename(columns = {'#FTEHospitalTotal2017':'#FTEHospital2017'})
    Berkeley = Berkeley.rename(columns = {'TotalM.D.\'s,TotNon-FedandFed2017': '#MDs2017'})
    Berkeley = Berkeley.rename(columns = {'#HospParticipatinginNetwork2017': '#HospitalsInNetwork2017'})
    Berkeley = Berkeley.rename(columns = {'dem_to_rep_ratio':'FracDem'})

    """Changing ratios and percentages to frac. Doing this whenever its relative to a population total."""

    """Editing to get dem_to_rep_ratio into a fraction of Democrats."""
    Berkeley['FracDem'] = Berkeley['FracDem'].apply(RatioToFrac)

    """Changing percentages to frac.""" 
    Berkeley['DiabetesPercentage'] = Berkeley['DiabetesPercentage'].apply(PercentToFrac)
    Berkeley['SmokersPercentage'] = Berkeley['SmokersPercentage'].apply(PercentToFrac)
    
    """Writing to the appropriate .csv file."""
    Berkeley.to_csv(outputdir+'Aggregate_Berkeley.csv')
    
if __name__ == '__main__':
    main()