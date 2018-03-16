import os
import random
import shutil

import hyperparams as hp


def create_train_val(root, ratio=0.1, seed=1337):
    random.seed(seed)
    data = os.listdir(root)
    train_dir = os.path.join(root, 'train')
    val_dir = os.path.join(root, 'val')

    # copy everything to train dir
    for sample in data:
        src = os.path.join(root, sample)
        dst = os.path.join(train_dir, sample)
        shutil.copytree(src, dst)

    # move sampled data to val dir
    num_val = int(len(data) * ratio)
    val_samples = random.sample(data, k=num_val)
    for sample in val_samples:
        src = os.path.join(train_dir, sample)
        dst = os.path.join(val_dir, sample)
        shutil.move(src, dst)


if __name__ == '__main__':
    create_train_val(hp.ROOT, hp.validation_size)
