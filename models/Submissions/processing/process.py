import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import git

# FIXES NANs

if __name__ == '__main__':
	submission = pd.read_csv("../submission1.csv")
	submission.fillna(method='ffill', inplace=True)
	num = submission._get_numeric_data()
	num[num < 0.1] = 0
	submission.to_csv("modified_submission.csv", index=False)
	print(submission.isnull().values.any())