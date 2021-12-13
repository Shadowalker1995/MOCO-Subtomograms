"""
generating target information for `info_INS.pickle` provided by Xiangrui
"""
import pickle
import csv
import numpy as np


def read_csv(filename):
    """read a csv file and convert it to a list of voxel coordinates"""
    csv_reader = csv.reader(open(f'./Datasets/target_info/{filename}.csv', 'r'))
    voxels = []
    for voxel in csv_reader:
        try:
            voxels.append(list(map(int, voxel[0].split('\t'))))
        except:
            continue
    target = filename.split('_')[-1]
    print(filename, len(voxels))
    return voxels, target


# 35502 tuples
# (filename, uuid, [x, y, z])
# (b'/data/jin7/Ribosome_July2021/INS_21_g3_t1/INS_21_g3_t1.rec',
#  b'6b7bffbb-65f4-4d61-877d-312d23ac19e3',
#  [40, 124, 66])
with open('./Datasets/target_info/info_INS.pickle', 'rb') as pickle_file:
    no_targets = pickle.load(pickle_file, encoding='bytes')

# convert from list to dict
keys = ['filename', 'uuid', 'loc']
for i in range(len(no_targets)):
    no_targets[i] = dict(zip(keys, no_targets[i]))
    no_targets[i]['loc'] = np.array(no_targets[i]['loc']) * 4
    no_targets[i]['dist'] = float('inf')
    no_targets[i]['target'] = 'unknown'

# split the no_targets into several file
# INS_21_g3_t23, INS_21_g3_t24, INS_21_g3_t10, INS_21_g3_t26, INS_21_g3_t1, INS_21_g3_t14
INS_t1_unknown = list()
INS_t10_unknown = list()
INS_t14_unknown = list()
INS_t23_unknown = list()
INS_t24_unknown = list()
INS_t26_unknown = list()
for voxel in no_targets:
    if b'INS_21_g3_t1.rec' in voxel['filename']:
        INS_t1_unknown.append(voxel)         # 5402
    elif b'INS_21_g3_t10.rec' in voxel['filename']:
        INS_t10_unknown.append(voxel)        # 5790
    elif b'INS_21_g3_t14.rec' in voxel['filename']:
        INS_t14_unknown.append(voxel)        # 5418
    elif b'INS_21_g3_t23.rec' in voxel['filename']:
        INS_t23_unknown.append(voxel)        # 5874
    elif b'INS_21_g3_t24.rec' in voxel['filename']:
        INS_t24_unknown.append(voxel)        # 6245
    elif b'INS_21_g3_t26.rec' in voxel['filename']:
        INS_t26_unknown.append(voxel)        # 6773
# print('INS_t1_unknown', len(INS_t1_unknown))
# print('INS_t10_unknown', len(INS_t10_unknown))
# print('INS_t14_unknown', len(INS_t14_unknown))
# print('INS_t23_unknown', len(INS_t23_unknown))
# print('INS_t24_unknown', len(INS_t24_unknown))
# print('INS_t26_unknown', len(INS_t26_unknown))

# sum is 3512
INS_t1_bound, _ = read_csv('INS_21_g3_t1_bound')    # 150
INS_t1_free, _ = read_csv('INS_21_g3_t1_free')      # 308
INS_t10_bound, _ = read_csv('INS_21_g3_t10_bound')  # 100
INS_t10_free, _ = read_csv('INS_21_g3_t10_free')    # 728
INS_t13_bound, _ = read_csv('INS_21_g3_t13_bound')  # 60
INS_t13_free, _ = read_csv('INS_21_g3_t13_free')    # 379
INS_t14_bound, _ = read_csv('INS_21_g3_t14_bound')  # 6
INS_t14_free, _ = read_csv('INS_21_g3_t14_free')    # 349
INS_t23_free, _ = read_csv('INS_21_g3_t23_free')    # 497
INS_t24_bound, _ = read_csv('INS_21_g3_t24_bound')  # 303
INS_t24_free, _ = read_csv('INS_21_g3_t24_free')    # 363
INS_t26_bound, _ = read_csv('INS_21_g3_t26_bound')  # 1
INS_t26_free, _ = read_csv('INS_21_g3_t26_free')    # 268


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
        print(dist[min_index])
        print(min_dist)
        unknown_voxels[min_index]['target'] = target
        unknown_voxels[min_index]['dist'] = min_dist
        # print(f'the minimal dist of the index of {min_index} is {min_dist}')
    elif unknown_voxels[min_index]['dist'] > min_dist:
        unknown_voxels[min_index]['target'] = target
        unknown_voxels[min_index]['dist'] = min_dist
        # print(f'the index of {min_index} is already matched, but update with smaller dist {min_dist}')
    else:
        pass
        # print(f'the index of {min_index} is already matched')


# INS_t1_unknown
# INS_t1_bound
# INS_t1_free
def has_target(voxel):
    return voxel['target'] != 'unknown'


def filter_known_list(unknow_list, bound_list=None, free_list=None):
    if free_list is None:
        free_list = []
    if bound_list is None:
        bound_list = []

    for voxel_loc in bound_list:
        voxel_loc = np.array(voxel_loc)
        min_dis_match(voxel_loc, unknow_list, 'bound')

    for voxel_loc in free_list:
        voxel_loc = np.array(voxel_loc)
        min_dis_match(voxel_loc, unknow_list, 'free')

    known_list = list(filter(has_target, unknow_list))
    print(len(known_list))
    dist_list = []
    for voxel in known_list:
        dist_list.append(voxel['dist'])
    dist_list.sort()

    return known_list, dist_list


INS_t1_known, t1_dist = filter_known_list(INS_t1_unknown, INS_t1_bound, INS_t1_free)
INS_t10_known, t10_dist = filter_known_list(INS_t10_unknown, INS_t10_bound, INS_t10_free)
INS_t14_known, t14_dist = filter_known_list(INS_t14_unknown, INS_t14_bound, INS_t14_free)
INS_t23_known, t23_dist = filter_known_list(INS_t23_unknown, None, INS_t23_free)
INS_t24_known, t24_dist = filter_known_list(INS_t24_unknown, INS_t24_bound, INS_t24_free)
INS_t26_known, t26_dist = filter_known_list(INS_t26_unknown, INS_t26_bound, INS_t26_free)

# np.save('t1_dist.npy', np.array(t1_dist))

# with open('t1_dist.txt', 'w') as f:
#     for dist in t1_dist:
#         f.write(str(dist))
#         f.write('\n')

with open('./Datasets/target_info/INS_t1_known.pickle', 'wb') as pickle_file:
    pickle.dump(INS_t1_known, pickle_file)
with open('./Datasets/target_info/INS_t10_known.pickle', 'wb') as pickle_file:
    pickle.dump(INS_t10_known, pickle_file)
with open('./Datasets/target_info/INS_t14_known.pickle', 'wb') as pickle_file:
    pickle.dump(INS_t14_known, pickle_file)
with open('./Datasets/target_info/INS_t23_known.pickle', 'wb') as pickle_file:
    pickle.dump(INS_t23_known, pickle_file)
with open('./Datasets/target_info/INS_t24_known.pickle', 'wb') as pickle_file:
    pickle.dump(INS_t24_known, pickle_file)
with open('./Datasets/target_info/INS_t26_known.pickle', 'wb') as pickle_file:
    pickle.dump(INS_t26_known, pickle_file)

# with open('./Datasets/target_info/INS_t1_known.pickle', 'rb') as pickle_file:
#     tmp = pickle.load(pickle_file)
