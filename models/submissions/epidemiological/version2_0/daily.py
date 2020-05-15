import pandas as pd
import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir


def convert(input_file, directory):
    submission = pd.read_csv(input_file, index_col=False)

    predictions = [[],[],[],[],[],[],[],[],[]]
    previous_row = None
    previous_county = None
    for index, row in submission.iterrows():
        county = row["id"].split('-')[-1]

        if county != previous_county:
            predictions[0].append(0)
            predictions[1].append(0)
            predictions[2].append(0)
            predictions[3].append(0)
            predictions[4].append(0)
            predictions[5].append(0)
            predictions[6].append(0)
            predictions[7].append(0)
            predictions[8].append(0)

        else:
            predictions[0].append(row['10']-previous_row['10'])
            predictions[1].append(row['20']-previous_row['20'])
            predictions[2].append(row['30']-previous_row['30'])
            predictions[3].append(row['40']-previous_row['40'])
            predictions[4].append(row['50']-previous_row['50'])
            predictions[5].append(row['60']-previous_row['60'])
            predictions[6].append(row['70']-previous_row['70'])
            predictions[7].append(row['80']-previous_row['80'])
            predictions[8].append(row['90']-previous_row['90'])

        previous_county = county
        previous_row = row

    submission['10'] = predictions[0]
    submission['20'] = predictions[1]
    submission['30'] = predictions[2]
    submission['40'] = predictions[3]
    submission['50'] = predictions[4]
    submission['60'] = predictions[5]
    submission['70'] = predictions[6]
    submission['80'] = predictions[7]
    submission['90'] = predictions[8]

    # submission.to_csv(f"{homedir}" + "/models/submissions/model1/daily_" + input_file, index=False)
    output_path = directory + "daily_" + input_file
    submission.to_csv(output_path, index=False)
    return output_path

if __name__ == '__main__':
    convert("predictions2.csv")