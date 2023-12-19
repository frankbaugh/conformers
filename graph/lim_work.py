import pandas as pd
import pickle

with open('./benched_models/limforcesplit_ani', 'rb') as f:
    df = pickle.load(f)



diff_col = df['ANI'] - df['FFXML Baseline']
df['Difference'] = diff_col


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)
    
score = (df['Difference'] >= 0).mean()
model_avg = (df['Dihedral Model']).mean()
force_avg = (df['FFXML Baseline']).mean()
ani_avg = (df['ANI']).mean()
print('================================================================')
print(f"ANI vs FF: {score}")
print(f"Model mean correlation: {model_avg}")
print(f"force avg = {force_avg}")
print(f"ani_avg = {ani_avg}")