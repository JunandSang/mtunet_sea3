import argparse
import os
import csv
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='path to the data', required=True)
parser.add_argument('--split', type=str, help='path to the split folder', required=True)
args = parser.parse_args()

def get_name(root, mode_folder=True):
    for root, dirs, file in os.walk(root):
        if mode_folder:
            return dirs
        else:
            return file

def make_csv(data, name):
    f_val = open(name + ".csv", "w", encoding="utf-8")
    csv_writer = csv.writer(f_val)
    csv_writer.writerow(["filename", "label"])
    for i in range(len(data)):
        csv_writer.writerow(data[i])
    f_val.close()

def move(cls, phase):
    src_dir = os.path.join(data_root, cls)
    dest_dir = os.path.join(save_root, "sea", phase, cls)
    os.makedirs(dest_dir, exist_ok=True)  # Ensure the destination directory exists

    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dest_path = os.path.join(dest_dir, item)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dest_path, dirs_exist_ok=True)  # Copy subdirectories
        else:
            shutil.copy2(src_path, dest_path)  # Copy files

def read_csv(name):
    with open(name, 'r') as f:
        split = [x.strip() for x in f.readlines() if x.strip() != '']
    return split

def get_split(cls, phase):
    record = []
    for cl in cls:
        name_imgs = get_name(os.path.join(data_root, cl), mode_folder=False)
        for name in name_imgs:
            record.append([cl + "/" + name, cl])
        move(cl, phase)
    make_csv(record, f"/home/work/aim_lab/simsingae/MTUNet/data/data_split/seas/{phase}")

if __name__ == '__main__':
    if args.data is None or args.split is None:
        raise ValueError("Both --data and --split arguments are required. Use --data <path> and --split <path>.")

    data_root = args.data
    save_root = args.split
    r_root = os.path.join(save_root, "sea")
    os.makedirs(r_root, exist_ok=True)

    name_train = os.path.join(data_root, "splits", "train.txt")
    name_val = os.path.join(data_root, "splits", "val.txt")
    name_test = os.path.join(data_root, "splits", "test.txt")

    train = read_csv(name_train)
    val = read_csv(name_val)
    test = read_csv(name_test)

    os.makedirs(os.path.join(save_root, "sea", "train"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "sea", "val"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "sea", "test"), exist_ok=True)

    get_split(train, "train")
    get_split(val, "val")
    get_split(test, "test")
