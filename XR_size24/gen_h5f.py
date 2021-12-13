import pickle
import h5py
import numpy as np

file_identity_list = ['t1', 't10', 't13', 't14', 't23', 't24', 't26']
# file_identity_list = ['t1', 't10']

voxel_array_list = list()
target_array_list = list()

for file_identity in file_identity_list:
    with open(f"./Datasets/target_info/INS_{file_identity}_known_unknown.pickle", 'rb') as pickle_file:
        INS_t1_known_unknown = pickle.load(pickle_file)
        print(f'load {file_identity} info file done')

    with open(f"./Datasets/raw_data/INS_21_g3_{file_identity}_bin8.pickle", 'rb') as pickle_file:
        raw_data = pickle.load(pickle_file)
        print(f'load {file_identity} raw data done')

    target_map = {'unknown': 0, 'free': 1, 'bound': 2}
    voxel_array = np.zeros((len(INS_t1_known_unknown), 24, 24, 24), dtype=np.float32)
    target_array = np.zeros(len(INS_t1_known_unknown), dtype=np.int)
    for index, info in enumerate(INS_t1_known_unknown):
        voxel_array[index] = raw_data['vs'][info['uuid'].decode("utf-8")]['v']
        target_array[index] = target_map[info['target']]
    voxel_array = np.expand_dims(voxel_array, -1)

    voxel_array_list.append(voxel_array)
    target_array_list.append(target_array)

# Generate the h5 file
h5f = h5py.File('./Datasets/data_INS2.h5', 'w')
# shape (8868, 24, 24, 24, 1), 8868 = 3268 + 800*7
h5f.create_dataset('dataset_1', data=np.concatenate(voxel_array_list, axis=0))
# shape (8868,)
h5f.create_dataset('target', data=np.concatenate(target_array_list, axis=0))
h5f.close()
