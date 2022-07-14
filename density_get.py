import pandas as pd
from tqdm import tqdm
import threading
from pymatgen.ext.matproj import MPRester

mpr = MPRester('j0zRRK6eTUelWCOdMB')


def query_density(input_path, output_path):
    file = pd.read_csv(input_path)
    results = []
    for i in tqdm(file['0']):
        result = mpr.query(i, properties=['density', 'material_id', 'unit_cell_formula'])
        results.append(result)
    results = pd.DataFrame(results)
    results.to_csv(output_path, index=False)

#多线程使用
if __name__ == "__main__":

    t1 = threading.Thread(
        target=query_density, args=(f'D:\pythonProject\summary\XXX_{3}.csv', f'file{3}.csv'))
    t2 = threading.Thread(
        target=query_density, args=(f'D:\pythonProject\summary\XXX_{5}.csv', f'file{5}.csv'))
    '''
    t3 = threading.Thread(
        target=query_density, args=(f'D:\pythonProject\summary\XXX_{5}.csv', f'file{5}.csv'))
    t4 = threading.Thread(
        target=query_density, args=(f'D:\pythonProject\summary\XXX_{0}.csv', f'file{0}.csv'))
    t5 = threading.Thread(
        target=query_density, args=(f'D:\pythonProject\summary\XXX_{1}.csv', f'file{1}.csv'))
    t6 = threading.Thread(
        target=query_density, args=(f'D:\pythonProject\summary\XXX_{2}.csv', f'file{2}.csv'))
    
    '''
    t1.start()
    t2.start()
    '''
    t3.start() 
    t4.start()
    t5.start()
    t6.start()  
    '''
expr="""
    dictt_all{}=[]
    for j in range(file{}.shape[0]):
        dictt=eval(file{}.iloc[j][0])
        dictt_all{}.append(dictt)
    """