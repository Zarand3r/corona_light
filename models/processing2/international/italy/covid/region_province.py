import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import git

repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
inputdir = f"{homedir}/data/international/italy/covid/"
outputdir = f"{homedir}/models/data/international/italy/covid"

# translate regional file
dfr = pd.read_csv(inputdir + "dpc-covid19-ita-regioni.csv")
dfr.columns = ["Date","Country", "Regional Code", "Region", "Latitude","Longitude","HospitalizedWithSymptoms","IntensiveCare","TotalHospitalized","HomeIsolation","TotalCurrentlyPositive","ChangeTotalPositive","NewCurrentlyPositive","DischargedHealed","Deaths","TotalCases","Tested","Note_IT","Note_ENG"]
dfr.to_csv(outputdir + 'dpc-covid19-ita-regioni.csv', index=False)

# translate provincial file
dfp = pd.read_csv(inputdir + "dpc-covid19-ita-province.csv")
dfp.columns = ["Date","Country", "Regional Code", "Region", "Province Code","Province","ProvinceInitials","Latitude","Longitude","TotalCases","Note_IT","Note_ENG"]
dfp.to_csv(outputdir + "dpc-covid19-ita-province.csv", index=False)

