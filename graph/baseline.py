import pandas as pd

path = '/users/frank/code/conf/37conf8/'
filename = '37Conf8_data.xlsx'

df = pd.read_excel(path + filename, sheet_name='Rel_Energy_OPT', header=2)
df = df.drop(df.index[-3:], axis=0)

forcefield_energies = {}

for index, row in df.iterrows():
    conf = str(row.iloc[0]).strip() + '_' + str((row.iloc[1]))[0]
    energy = row.loc['MMFF94']
    forcefield_energies[conf] = energy


print(forcefield_energies)