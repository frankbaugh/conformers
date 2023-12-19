import os
import msgpack
import numpy as np

direc = "/Users/frank/Downloads"

qm9_file = os.path.join(direc, "qm9_crude.msgpack")
unpacker = msgpack.Unpacker(open(qm9_file, "rb"))

qm9_1k = next(iter(unpacker))


sample_smiles = list(qm9_1k.keys())[10]
sample_sub_dic = qm9_1k[sample_smiles]['conformers'][0]['xyz']

path = '/Users/frank/code/conf/37Conf8/'
def csv_to_dict():
    energies = {}
    filename = path + 'rel_en.csv'
    data = np.genfromtxt(filename, delimiter=',', dtype=None, encoding=None)
    for row in data:
        conf_name = row[0].strip() + "_" + str(row[1])
        energies[conf_name] = row[2]
    
    print(energies['shiepox_depox_5'])


csv_to_dict()