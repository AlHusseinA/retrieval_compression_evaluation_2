import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url
from PIL import Image
from torch.utils.data import Dataset


class Cub2011(Dataset):
    base_folder = "CUB_200_2011/images"
    root = "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/cub200/"
    # "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/cub200/CUB_200_2011"
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    filename = "CUB_200_2011.tgz"
    tgz_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.train_ids = []  # Initialize as class attribute
        self.test_ids = []   # Initialize as class attribute

        if download:
            self._download()

        self.data, self.targets, self.classes = self._load_metadata()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        # self.data, self.targets, self.classes = self._load_metadata()


    def _load_metadata(self):
        data = []
        targets = []
        classes = []


        with open(os.path.join(self.root, 'CUB_200_2011', 'images.txt')) as f:
            img_filenames = [x.split() for x in f.readlines()]

        with open(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt')) as f:
            img_labels = [int(x.split()[1]) - 1 for x in f.readlines()]

        with open(os.path.join(self.root, 'CUB_200_2011', 'classes.txt')) as f:
            class_names = [x.split(' ', 1)[1].strip() for x in f.readlines()]

        with open(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt')) as f:
            splits = [int(x.split()[1]) for x in f.readlines()]

        for i, (file_info, label) in enumerate(zip(img_filenames, img_labels)):
            img_id, img_filename = file_info
            if (self.train and splits[i]) or (not self.train and not splits[i]):
                data.append(img_filename)
                targets.append(label)
                if splits[i]:
                    self.train_ids.append((int(img_id), label))
                else:
                    self.test_ids.append((int(img_id), label))

        return data, targets, class_names


    def get_train_image_ids(self):
        if self.train:
            return self.train_ids  # Now returns tuples of (ID, label)
        else:
            raise ValueError("This instance is using the test dataset, not the training dataset.")

    def get_test_image_ids(self):
        if not self.train:
            return self.test_ids  # Now returns tuples of (ID, label)
        else:
            raise ValueError("This instance is using the training dataset, not the test dataset.")

    

    

    def get_class_names(self):
        """
        Returns a list of all class names (label names) in the dataset.
        """
        return self.classes

    def get_number_of_classes(self):
        return len(self.classes)

    def _check_integrity(self):
        for filename in self.data:
            filepath = os.path.join(self.root, self.base_folder, filename)
            if not os.path.isfile(filepath):
                return False
        return True
    # def _check_integrity(self):
    #     for filename in self.data:
    #         filepath = os.path.join(self.root, self.base_folder, filename)
    #         if not os.path.isfile(filepath):
    #             print("%%%"*35)
    #             print(f"File not found: {filepath}")  # Add this line to print the missing file path
    #             print("%%%"*35)
    #             return False
    #     return True

    def get_label_to_name_mapping(self, single_label):
        """
        Returns the class name for a given numeric label.
        Raises an error if the label is out of range.
        Assumes labels in the dataset start from 1.
        """

        # Adjusting label to 0-indexed to match Python list indexing
        adjusted_label = single_label - 1

        # return self.classes[adjusted_label]
        return self.classes[single_label]


    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.base_folder, self.data[idx])
        target = self.targets[idx]

        img = Image.open(path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target


# if __name__ == '__main__':
#     train_dataset = Cub2011('./cub2011', train=True, download=False)
#     test_dataset = Cub2011('./cub2011', train=False, download=False)






### this is _load_metadata() function before the modficiations 
    

    # def _load_metadata(self):
    #     data = []
    #     targets = []
    #     classes = []

    #     with open(os.path.join(self.root, 'CUB_200_2011', 'images.txt')) as f:
    #         img_filenames = [x.split()[1] for x in f.readlines()]

    #     with open(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt')) as f:
    #         img_labels = [int(x.split()[1]) - 1 for x in f.readlines()]

    #     with open(os.path.join(self.root, 'CUB_200_2011', 'classes.txt')) as f:
    #         class_names = [x.split(' ', 1)[1].strip() for x in f.readlines()]

    #     with open(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt')) as f:
    #         splits = [int(x.split()[1]) for x in f.readlines()]

    #     for i in range(len(img_filenames)):
    #         if (self.train and splits[i]) or (not self.train and not splits[i]):
    #             data.append(img_filenames[i])
    #             targets.append(img_labels[i])

    #     return data, targets, class_names