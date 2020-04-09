import sys
sys.path.insert(1, '../processing')
import loader

italy = loader.load_data("dpc-covid19-ita-regioni.csv", "/models/processing/International/Italy/")
abruzzo = loader.query(italy, "Region", "Abruzzo")
loader.plot_features(abruzzo, "TotalHospitalized","HomeIsolation","TotalCurrentlyPositive","Deaths","TotalCases")