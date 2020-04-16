import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import git
from pathlib import Path

repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
inputcovid = f"{homedir}/data/international/italy/covid/"
outputcovid = f"{homedir}/models/data/international/italy/covid/"
Path(outputcovid).mkdir(parents=True, exist_ok=True)
inputdemo = f"{homedir}/data/international/italy/demographics/"
outputdemo = f"{homedir}/models/data/international/italy/demographics/"
Path(outputdemo).mkdir(parents=True, exist_ok=True)

# translate regional file
dfr = pd.read_csv(inputcovid+"dpc-covid19-ita-regioni.csv")
dfr.columns = ["Date","Country", "Regional Code", "Region", "Latitude","Longitude","HospitalizedWithSymptoms","IntensiveCare","TotalHospitalized","HomeIsolation","TotalCurrentlyPositive","ChangeTotalPositive","NewCurrentlyPositive","DischargedHealed","Deaths","TotalCases","Tested","Note_IT","Note_ENG"]
dfr.to_csv(outputcovid + 'dpc-covid19-ita-regioni.csv', index=False)

# translate provincial file
dfp = pd.read_csv(inputcovid+"dpc-covid19-ita-province.csv")
dfp.columns = ["Date","Country", "Regional Code", "Region", "Province Code","Province","ProvinceInitials","Latitude","Longitude","TotalCases","Note_IT","Note_ENG"]
dfp.to_csv(outputcovid + "dpc-covid19-ita-province.csv", index=False)

population = pd.read_csv(inputdemo+"region-populations.csv")
population.to_csv(outputdemo + "region-populations.csv")