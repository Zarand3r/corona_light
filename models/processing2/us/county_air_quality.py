import pandas as pd
import numpy as np
import os
import datetime
from scipy.spatial.distance import cdist
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import BallTree
import git
from pathlib import Path

repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
inputdir = f"{homedir}" + "/data/us/air_quality/"
inputdir2 = f"{homedir}" + "/data/us/geolocation/"
outputdir = f"{homedir}" + "/models/data/us/air_quality/"
Path(outputdir).mkdir(parents=True, exist_ok=True)
outputdir += "county_"


def closest_point(point, points):
    """ Find closest point from a list of points. """
    return points[cdist([point], points).argmin()]


def match_value(df, col1, x, col2):
    """ Match value x from col1 row to value in col2. """
    return df[df[col1] == x][col2].values[0]


def main():
    #Fixed Months, no need to run this
    Air_Qual_1 = pd.read_csv(inputdir+'1.tgz', compression='gzip')                 
    Air_Qual_2 = pd.read_csv(inputdir+'2.tgz', compression='gzip')                 
    Air_Qual_3 = pd.read_csv(inputdir+'3.tgz', compression='gzip')                 


    #Air Data that Changes Weekly
    Air_Qual_4 = pd.read_csv(inputdir+'4.tgz', compression='gzip')                 


    #Getting only US Air Data
    Air_Qual_1 = Air_Qual_1[Air_Qual_1['CountryCode'] == 'US']
    Air_Qual_2 = Air_Qual_2[Air_Qual_2['CountryCode'] == 'US']
    Air_Qual_3 = Air_Qual_3[Air_Qual_3['CountryCode'] == 'US']
    Air_Qual_4 = Air_Qual_4[Air_Qual_4['CountryCode'] == 'US']
    #Dropping unneeded columns
    Air_Qual_1 = Air_Qual_1.drop(columns=['out/','SiteName','GMTOffset','CountryCode','StateName', 'Elevation', 'DataSource','ReportingArea_PipeDelimited','Status','EPARegion'])
    Air_Qual_2 = Air_Qual_2.drop(columns=['out/','SiteName','GMTOffset','CountryCode','StateName', 'Elevation', 'DataSource','ReportingArea_PipeDelimited','Status','EPARegion'])
    Air_Qual_3 = Air_Qual_3.drop(columns=['out/','SiteName','GMTOffset','CountryCode','StateName', 'Elevation', 'DataSource','ReportingArea_PipeDelimited','Status','EPARegion'])
    Air_Qual_4 = Air_Qual_4.drop(columns=['out/','SiteName','GMTOffset','CountryCode','StateName', 'Elevation', 'DataSource','ReportingArea_PipeDelimited','Status','EPARegion'])

    Air_Qual_1 = Air_Qual_1.astype({'Latitude': 'float64','Longitude': 'float64'})
    Air_Qual_2 = Air_Qual_2.astype({'Latitude': 'float64','Longitude': 'float64'})
    Air_Qual_3 = Air_Qual_3.astype({'Latitude': 'float64','Longitude': 'float64'})
    Air_Qual_4 = Air_Qual_4.astype({'Latitude': 'float64','Longitude': 'float64'})

    #joining data together
    Air_Qual = Air_Qual_1.append(Air_Qual_2, ignore_index = True) 
    Air_Qual = Air_Qual.append(Air_Qual_3, ignore_index = True) 
    Air_Qual = Air_Qual.append(Air_Qual_4, ignore_index = True) 


    # In[34]:


    #Loading in County_Centers Data to map Air Quality to FIPs code
    County_Centers = pd.read_csv(inputdir2+'county_centers.csv')
    County_Centers = County_Centers.drop(columns=['clon00','clat00','pclon00','pclat00','pclon10','pclat10'])

    County_Centers.to_csv(outputdir+'centers.csv')


    #Creating GeoPandas DataFrames to do fast distance comparison
    C_C = gpd.GeoDataFrame({ #County_center gpd
            'geometry': Point(a, b),
            'x': float(a),
            'y': float(b),
        } for a, b in zip(County_Centers['clat10'], County_Centers['clon10']))
    A_Q = gpd.GeoDataFrame({  #Air_Quality gpd
            'geometry': Point(a, b),
            'x': float(a),
            'y': float(b),
        } for a, b in zip(Air_Qual['Latitude'], Air_Qual['Longitude']))

    tree = BallTree(C_C[['x', 'y']].values, leaf_size=2) #distance tree

    A_Q['distance_nearest'], A_Q['id_nearest'] = tree.query(
        A_Q[['x', 'y']].values, # The input array for the query
        k=1, # The number of nearest neighbors 
    )
        
    #Defining the fips code based on the 'id_nearest' column, the ID of the closest County_Center to each Air_Quality Report 
    Air_Qual['FIPS'] = list(County_Centers.iloc[A_Q['id_nearest']]['fips'])


    #Cleaning Air_Qual to split it up
    Air_Qual = Air_Qual.drop(columns=['ValidTime','OZONE_Measured','PM10_Measured','PM25_Measured','NO2_Measured','PM25', 'PM25_Unit', 'OZONE','OZONE_Unit','NO2','NO2_Unit','PM10','PM10_Unit'])
    # Air_Qual.to_csv(outputdir+'air_qual.csv')


    #Making individual Dataframes for each type of Gas particle collected
    Ozone = Air_Qual.loc[Air_Qual['OZONE_AQI'].astype(float) >= -1]
    Ozone = Ozone.drop(columns=['PM10_AQI','PM25_AQI','NO2_AQI','CO','CO_Unit','SO2','SO2_Unit'])
    #Setting the datapoint to be a float
    Ozone = Ozone.astype({'OZONE_AQI': 'float64'})
    #Taking the average value of 'OZONE_AQI' for each FIPS code and Date
    Ozone = Ozone.groupby(['FIPS','ValidDate']).mean()
    #Isolating only the 'OZONE_AQI' values for each date
    Ozone = Ozone.unstack()['OZONE_AQI']
    Ozone.to_csv(outputdir+'Ozone_AQI.csv')

    PM10 = Air_Qual.loc[Air_Qual['PM10_AQI'].astype(float) >= -1]
    PM10 = PM10.drop(columns=['OZONE_AQI','PM25_AQI','NO2_AQI','CO','CO_Unit','SO2','SO2_Unit'])
    PM10 = PM10.astype({'PM10_AQI': 'float64'})
    #Taking the average value of 'PM10_AQI' for each FIPS code and Date
    PM10 = PM10.groupby(['FIPS','ValidDate']).mean()
    #Isolating only the 'PM10_AQI' values for each date
    PM10 = PM10.unstack()['PM10_AQI']
    PM10.to_csv(outputdir+'PM10_AQI.csv')

    PM25 = Air_Qual.loc[Air_Qual['PM25_AQI'].astype(float) >= -1]
    PM25 = PM25.drop(columns=['OZONE_AQI','PM10_AQI','NO2_AQI','CO','CO_Unit','SO2','SO2_Unit'])
    PM25 = PM25.astype({'PM25_AQI': 'float64'})
    #Taking the average value of 'PM25_AQI' for each FIPS code and Date
    PM25 = PM25.groupby(['FIPS','ValidDate']).mean()
    #Isolating only the 'PM25_AQI' values for each date
    PM25 = PM25.unstack()['PM25_AQI']
    PM25.to_csv(outputdir+'PM25_AQI.csv')

    NO2 = Air_Qual.loc[Air_Qual['NO2_AQI'].astype(float) >= -1]
    NO2 = NO2.drop(columns=['PM10_AQI','PM25_AQI','OZONE_AQI','CO','CO_Unit','SO2','SO2_Unit'])
    NO2 = NO2.astype({'NO2_AQI': 'float64'})
    #Taking the average value of 'NO2_AQI' for each FIPS code and Date
    NO2 = NO2.groupby(['FIPS','ValidDate']).mean()
    #Isolating only the 'NO2_AQI' values for each date
    NO2 = NO2.unstack()['NO2_AQI']
    NO2.to_csv(outputdir+'NO2_AQI.csv')

    CO = Air_Qual.loc[Air_Qual['CO_Unit'].isin(['PPM','PPB'])]
    CO = CO.drop(columns=['PM10_AQI','PM25_AQI','OZONE_AQI', 'NO2_AQI','SO2','SO2_Unit'])
    CO = CO.astype({'CO': 'float64'})
    #Scaling every measurement to be in terms of PPB, so multiply PPM*1000
    CO.loc[CO['CO_Unit'] == 'PPM',['CO']] = CO.loc[CO['CO_Unit'] == 'PPM']['CO'].mul(1000).to_numpy()
    #Drop unit column, everythin is PPB
    CO = CO.drop(columns=['CO_Unit'])
    #Taking the average value of 'CO_Unit' for each FIPS code and Date
    CO = CO.groupby(['FIPS','ValidDate']).mean()
    #Isolating only the 'CO_Unit' values for each date
    CO = CO.unstack()['CO']
    CO.to_csv(outputdir+'CO_PPB.csv')

    SO2 = Air_Qual.loc[Air_Qual['SO2_Unit'].isin(['PPM','PPB'])]
    SO2 = SO2.drop(columns=['PM10_AQI','PM25_AQI','OZONE_AQI', 'NO2_AQI', 'CO', 'CO_Unit'])
    SO2 = SO2.astype({'SO2': 'float64'})
    #Scaling every measurement to be in terms of PPB, so multiply PPM*1000
    SO2.loc[SO2['SO2_Unit'] == 'PPM',['SO2']] = SO2.loc[SO2['SO2_Unit'] == 'PPM']['SO2'].mul(1000).to_numpy()
    SO2 = SO2.drop(columns=['SO2_Unit'])
    #Taking the average value of 'SO2_Unit' for each FIPS code and Date
    SO2 = SO2.groupby(['FIPS','ValidDate']).mean()
    #Isolating only the 'SO2_Unit' values for each date
    SO2 = SO2.unstack()['SO2']
    SO2.to_csv(outputdir+'SO2_PPB.csv')

if __name__ == '__main__':
    main()
