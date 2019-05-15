import argparse
import datetime
import numpy as np


def load_code_and_label(path):
    database_code = np.sign(np.load(path + "train_code.npy"))
    database_labels = np.load(path + "train_labels.npy")
    test_codes = np.sign(np.load(path + "test_code.npy"))
    test_labels = np.load(path + "test_labels.npy")

    return {"database_code": database_code, "database_labels": database_labels, \
            "test_code": test_codes, "test_labels": test_labels}


def mean_average_precision(params, R):
    database_code = params['database_code']
    validation_code = params['test_code']
    database_labels = params['database_labels']
    validation_labels = params['test_labels']
    query_num = validation_code.shape[0]

    print(database_code.shape, validation_code.shape, database_labels.shape, validation_labels.shape)

    sim = np.dot(database_code, validation_code.T)
    ids = np.argsort(-sim, axis=0)
    APx = []

    for i in range(query_num):
        label = validation_labels[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)
        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)

    return np.mean(np.array(APx))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HashNet')
    parser.add_argument('--dataset_dir', type=str, default='/tmp', help="device id to run")
    parser.add_argument('--hash_bit', type=int, default=16, help="number of hash code bits")
    args = parser.parse_args()
    bit = args.hash_bit
    path = args.dataset_dir + '-models/' + str(bit) + '/'

    print(path)
    print('Under computer the mAP.......')
    R = 1000

    code_and_label = load_code_and_label(path=path)
    # print('code_and_label:', code_and_label)

    mAP = mean_average_precision(code_and_label, R)
    print(str(bit), "MAP: " + str(mAP))

    with open(path + 'eval_data.txt', 'a+') as f:
        f.write("%s of %s bit MAP: %s \n" % (
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(bit), str(mAP)))
