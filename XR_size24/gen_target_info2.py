"""
generating target information for `INS_21_g3_t{1,10,13,14,23,24,26}_bin8_info.pickle` created by Zhu
"""
import pickle
import csv
import random
import numpy as np


def read_csv(filename):
    """read a csv file and convert it to a list of voxel coordinates"""
    voxels = []
    target = filename.split('_')[-1]
    try:
        csv_reader = csv.reader(open(f'./Datasets/target_info/{filename}.csv', 'r'))
        for voxel in csv_reader:
            try:
                voxels.append(list(map(int, voxel[0].split('\t'))))
            except:
                continue
        target = filename.split('_')[-1]
        print(filename, len(voxels))
    except FileNotFoundError:
        print('File Not Found')

    return voxels, target


def min_dis_match(voxel_loc, unknown_voxels, target):
    unknown_voxels_loc = [unknown_voxel['loc'] for unknown_voxel in unknown_voxels]
    unknown_voxels_loc = np.stack(unknown_voxels_loc)     # (5402, 3)
    # exchange x and y
    # unknown_voxels_loc = unknown_voxels_loc[:, [0, 1, 2]]
    dist = np.linalg.norm(voxel_loc - unknown_voxels_loc, axis=1)
    min_index = np.argmin(dist)
    min_dist = np.min(dist)
    if unknown_voxels[min_index]['target'] == 'unknown':
        print(voxel_loc)
        print(unknown_voxels_loc[min_index])
        print(min_dist)
        unknown_voxels[min_index]['target'] = target
        unknown_voxels[min_index]['dist'] = min_dist
        unknown_voxels[min_index]['match'] = voxel_loc
        # print(f'the minimal dist of the index of {min_index} is {min_dist}')
    elif unknown_voxels[min_index]['dist'] > min_dist:
        unknown_voxels[min_index]['target'] = target
        unknown_voxels[min_index]['dist'] = min_dist
        unknown_voxels[min_index]['match'] = voxel_loc
        # print(f'the index of {min_index} is already matched, but update with smaller dist {min_dist}')
    else:
        pass
        # print(f'the index of {min_index} is already matched')


def has_target(voxel):
    return voxel['target'] != 'unknown'


def unknown_target(voxel):
    return voxel['target'] == 'unknown'


def filter_known_list(unknown_list, bound_list, free_list):
    for voxel_loc in bound_list:
        voxel_loc = np.array(voxel_loc)
        min_dis_match(voxel_loc, unknown_list, 'bound')

    for voxel_loc in free_list:
        voxel_loc = np.array(voxel_loc)
        min_dis_match(voxel_loc, unknown_list, 'free')

    known_list = list(filter(has_target, unknown_list))
    unknown_list = list(filter(unknown_target, unknown_list))
    unknown_list = random.sample(unknown_list, 800)

    print(len(known_list))
    dist_list = []
    for voxel in known_list:
        dist_list.append(voxel['dist'])
    dist_list.sort()

    return known_list, dist_list, unknown_list


'''
78586 tuples
(filename, uuid, [x, y, z])
(b'INS_21_g3_t1.rec',
 b'552f1298-c785-4310-ae30-5aa173c7778b',
 [240, 80, 40])
'''
filename_list = ['INS_21_g3_t1', 'INS_21_g3_t10', 'INS_21_g3_t13',
                 'INS_21_g3_t14', 'INS_21_g3_t23', 'INS_21_g3_t24', 'INS_21_g3_t26']
# filename_list = ['INS_21_g3_t10', 'INS_21_g3_t13',
#                  'INS_21_g3_t14', 'INS_21_g3_t24', 'INS_21_g3_t26']
for filename in filename_list:
    with open(f'./Datasets/target_info/{filename}_bin8_info.pickle', 'rb') as pickle_file:
        INS_unknown = pickle.load(pickle_file, encoding='bytes')

    # convert from list to dict
    keys = ['filename', 'uuid', 'loc']
    for i in range(len(INS_unknown)):
        INS_unknown[i] = dict(zip(keys, INS_unknown[i]))
        INS_unknown[i]['loc'] = np.array(INS_unknown[i]['loc']) * 8
        INS_unknown[i]['match'] = np.zeros(3)
        INS_unknown[i]['dist'] = float('inf')
        INS_unknown[i]['target'] = 'unknown'

    INS_bound, _ = read_csv(f'{filename}_bound')    # 150
    INS_free, _ = read_csv(f'{filename}_free')      # 308

    INS_known, t_dist, INS_unknown = filter_known_list(INS_unknown, INS_bound, INS_free)
    INS_known_unknown = INS_known + INS_unknown

    file_identity = filename.split('_')[-1]     # t10

    np.save(f"./Datasets/target_info/{file_identity}_dist.npy", np.array(t_dist))

    with open(f"./Datasets/target_info/{file_identity}_dist.txt", 'w') as f:
        for dist in t_dist:
            f.write(str(dist))
            f.write('\n')

    with open(f"./Datasets/target_info/INS_{file_identity}_known.pickle", 'wb') as pickle_file:
        pickle.dump(INS_known, pickle_file)

    with open(f"./Datasets/target_info/INS_{file_identity}_known_unknown.pickle", 'wb') as pickle_file:
        pickle.dump(INS_known_unknown, pickle_file)

    '''
    a list of dicts like:
    {'filename': b'INS_21_g3_t1_bin8.rec',
     'uuid': b'5e694074-72b0-4502-8055-d85e30a43f93',
     'loc': array([2976, 2712,  288]),
     'match': array([2960, 2712,  272]),
     'dist': 22.627416997969522,
     'target': 'free'}
    '''
    # with open(f"./Datasets/target_info/INS_{file_identity}_known.pickle", 'rb') as pickle_file:
    #     tmp = pickle.load(pickle_file)

with open("./Datasets/target_info/INS_all_known_unknown.pickle", 'wb') as pickle_file:
    pickle.dump(INS_all, pickle_file)   # 8868 = 3268 + 800*7

    a = 1
