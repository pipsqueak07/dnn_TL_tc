from pymatgen.ext.matproj import MPRester
import pandas as pd
from tqdm import tqdm

file2 = pd.read_excel('C:/Users/Administrator/Desktop/Tc.xlsx')
mpr = MPRester('j0zRRK6eTUelWCOdMB')
# get xenonpy descriptors
compositions = []
for i in tqdm(file2['Materials ID']):
    results = mpr.query(str('mp-' + str(i)), ['unit_cell_formula'])
    compositions.append(results)

formulas = []
for i in range(142):
    for j in compositions[i][0].values():
        formulas.append(j)

# get CIF descriptors
mpr.get_data('mp-3161')