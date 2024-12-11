import os
import pandas as pd
from sklearn.model_selection import train_test_split

__all__ = ['DatasetFolder']


class DatasetFolder(object):
    def __init__(self, root, set_name, split_type, transform, out_name=False, cls_selction=None, mode=None):
        assert split_type in ['train', 'test', 'val']
        split_file = os.path.join("data/data_split", set_name, split_type + '.csv')
        print(f"Split file path: {split_file}")
        assert os.path.isfile(split_file), f"Split file does not exist: {split_file}"

        data = self.read_csv(split_file)
        cls = list(data.keys())
        print(f"Total classes in CSV: {len(cls)}")

        # 선택된 클래스를 확인 및 데이터 필터링
        data_new, cls_new = self.select_class(data, cls, cls_selction)
        print(f"Selected classes: {cls_new}")
        print(f"Total samples after selection: {len(data_new)}")

        # 데이터가 비어있으면 에러를 발생
        assert len(data_new) > 0, "No data available after class selection."

        # 데이터 루트 확인 및 경로 체크
        for sample in data_new[:10]:  # 처음 10개 샘플 경로 확인
            file_path = os.path.join(root, sample[0])
            print(f"Checking file path: {file_path}, Exists: {os.path.isfile(file_path)}")
            assert os.path.isfile(file_path), f"File does not exist: {file_path}"

        if mode is not None:
            train, val = train_test_split(data_new, random_state=1, train_size=0.9)
            data_new = train if mode == "train" else val

        self.data = [x[0] for x in data_new]
        self.labels = [cls_new.index(x[1]) for x in data_new]
        self.root = root
        self.transform = transform
        self.out_name = out_name
        self.length = len(self.data)

        print(f"Dataset initialized with {self.length} samples.")

    def select_class(self, data, cls, selection):
        """선택된 클래스와 해당 데이터를 필터링"""
        current_data = []
        cls_new = cls if selection is None else [cls[i] for i in selection if i < len(cls)]
        for cl in cls_new:
            current_data.extend(data[cl])
        return current_data, cls_new

    def read_csv(self, name):
        data_all = pd.read_csv(name)
        data = data_all[["filename", "label"]]
        data = data.values
        cls = {}
        for img in data:
            cls.setdefault(img[1], []).append([img[0], img[1]])
        return cls

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        folder_name, img_name = self.data[index].split("/", 1)
        img_root = os.path.join(self.root, folder_name, img_name)
        assert os.path.isfile(img_root), f"Image file not found: {img_root}"
        img = Image.open(img_root).convert('RGB')
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label
