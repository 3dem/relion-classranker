import numpy as np
import os
import sys
import numpy as np
import torch


def load_star(filename):
    from collections import OrderedDict
    # This is not a very serious parser; should be token-based.
    datasets = OrderedDict()
    current_data = None
    current_colnames = None
    in_loop = 0  # 0: outside 1: reading colnames 2: reading data

    for line in open(filename):
        line = line.strip()

        # remove comments
        comment_pos = line.find('#')
        if comment_pos > 0:
            line = line[:comment_pos]

        if line == "":
            continue

        if line.startswith("data_"):
            in_loop = 0

            data_name = line[5:]
            current_data = OrderedDict()
            datasets[data_name] = current_data

        elif line.startswith("loop_"):
            current_colnames = []
            in_loop = 1

        elif line.startswith("_"):
            if in_loop == 2:
                in_loop = 0

            elems = line[1:].split()
            if in_loop == 1:
                current_colnames.append(elems[0])
                current_data[elems[0]] = []
            else:
                current_data[elems[0]] = elems[1]

        elif in_loop > 0:
            in_loop = 2
            elems = line.split()
            assert len(elems) == len(current_colnames)
            for idx, e in enumerate(elems):
                current_data[current_colnames[idx]].append(e)

    return datasets


def load_mrc(filename, maxz=-1):
    import numpy as np

    inmrc = open(filename, "rb")
    header_int = np.fromfile(inmrc, dtype=np.uint32, count=256)
    inmrc.seek(0, 0)
    header_float = np.fromfile(inmrc, dtype=np.float32, count=256)

    nx, ny, nz = header_int[0:3]
    eheader = header_int[23]
    mrc_type = None
    if header_int[3] == 2:
        mrc_type = np.float32
    elif header_int[3] == 6:
        mrc_type = np.uint16
    if maxz > 0:
        nz = np.min([maxz, nz])
    # print "NX = %d, NY = %d, NZ = %d" % (nx, ny, nz), mrc_type

    inmrc.seek(1024 + eheader, 0)
    map_slice = np.fromfile(inmrc, mrc_type, nx * ny * nz).reshape(nz, ny, nx).astype(np.float32)

    return nx, ny, nz, map_slice


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('data_root', type=str)
    parser.add_argument('--nr_valid', type=int, defult=6048)
    args = parser.parse_args()

    print('Generating tensors from the RELION STAR file...')
    dataset = load_star(args.data_root + '/combined_features_normalized.star')['normalized_features']
    nr_entries = len(dataset['rlnClassScore'])
    fn_subimage = dataset['rlnSubImageStack'][0]

    nr_train = nr_entries - args.nr_valid
    nr_valid = args.nr_valid

    nx, ny, nz, testsubimage = load_mrc(fn_subimage)

    my_x_train = np.zeros(shape=(nr_train * nz, 1, nx, ny), dtype=np.single)
    my_x_valid = np.zeros(shape=(nr_valid * nz, 1, nx, ny), dtype=np.single)
    for i, x in enumerate(dataset['rlnSubImageStack']):
        if i < nr_train + nr_valid:
            if (i % 1000 == 0):
                print(i)
            nx, ny, nz, subimage = load_mrc(x)
            for z in range(nz):
                if i < nr_train:
                    my_x_train[i * nz + z, 0] = subimage[z, :, :]
                else:
                    my_x_valid[(i - nr_train) * nz + z, 0] = subimage[z, :, :]
        else:
            break

    my_xp_train = np.zeros(shape=(nr_train * nz, 24))
    my_xp_valid = np.zeros(shape=(nr_valid * nz, 24))
    for i, x in enumerate(dataset['rlnNormalizedFeatureVector']):
        stringarray = x.replace('[', '').replace(']', '').split(',')
        if i < nr_train + nr_valid:
            for z in range(nz):
                for j, y in enumerate(stringarray):
                    if (i < nr_train):
                        my_xp_train[i * nz + z, j] = float(y)
                    else:
                        my_xp_valid[(i - nr_train) * nz + z, j] = float(y)

    my_y_train = np.zeros(shape=(nr_train * nz, 1), dtype=np.single)
    my_y_test = np.zeros(shape=(nr_valid * nz, 1), dtype=np.single)

    for i, x in enumerate(dataset['rlnClassScore']):
        if i < nr_train + nr_valid:
            for z in range(nz):
                score = float(x)
                if (i < nr_train):
                    my_y_train[i * nz + z, 0] = score
                else:
                    my_y_test[(i - nr_train) * nz + z, 0] = score
        else:
            break

    print('y_train_shape= ', my_y_train.shape)
    print('x_train_shape= ', my_x_train.shape)
    print('xp_train_shape= ', my_xp_train.shape)
    print('y_valid_shape= ', my_y_test.shape)
    print('x_valid_shape= ', my_x_valid.shape)
    print('xp_valid_shape= ', my_xp_valid.shape)

    torch.save({
        "x": torch.Tensor(my_x_train),
        "xp": torch.Tensor(my_xp_train),
        "y": torch.Tensor(my_y_train)
    }, args.data_root + '/dataset_train.pt')

    torch.save({
        "x": torch.Tensor(my_x_valid),
        "xp": torch.Tensor(my_xp_valid),
        "y": torch.Tensor(my_y_test)
    }, args.data_root + '/dataset_valid.pt')
