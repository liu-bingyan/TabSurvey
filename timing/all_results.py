import pandas as pd
import os

folder_path = os.path.dirname(os.path.abspath(__file__)) + '/individual/'
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

if 'all_results.csv' in csv_files:
    csv_files.remove('all_results.csv')


df = pd.DataFrame()
for file in csv_files:
    df =pd.concat([df,pd.read_csv(folder_path + file)],ignore_index=True)
df.to_csv(folder_path + 'all_results.csv', index=False)

#md_file = folder_path + 'all_results.md'
#df.to_markdown(md_file, index=False)
