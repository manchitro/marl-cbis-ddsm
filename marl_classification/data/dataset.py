import pickle as pkl
from os import listdir
from os.path import exists, isdir, join
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as fun
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as tr


#RES_PATH = abspath(join(dirname(abspath(__file__)), "..", "..", "resources"))


def my_pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    f = open(path, 'rb')
    img = Image.open(f)
    return img.convert('RGB')
	
def cbis_pil_loader(path: str) -> Image.Image:
    f = open(path, 'rb')
    img = Image.open(f)
    return img.convert('L')

class CBISDatasetSimplified(ImageFolder):
    def __init__(self, res_path: str, img_transform: Any) -> None:
        # cbis_root_path = join(res_path, "downloaded", "cbis")
        cbis_root_path = join(res_path, "downloaded", "cbis_debug")

        assert exists(cbis_root_path) and isdir(cbis_root_path), \
            f"{cbis_root_path} does not exist or is not a directory"

        super().__init__(cbis_root_path, transform=img_transform,
                         target_transform=None, loader=cbis_pil_loader,
                         is_valid_file=None)

class CBISDataset(Dataset):
    def __init__(self, resource_path: str, img_transform: Any, dataset_to_train: str = "mass"):
        super().__init__()

        self.__cbis_root_path = join(resource_path, "downloaded", "cbis")

        # get csv files
        self.dicom_info_csv = pd.read_csv(
            join(self.__cbis_root_path, "csv", "dicom_info.csv"), sep=","
        )
        self.mass_train_csv = pd.read_csv(
            join(self.__cbis_root_path, "csv", "mass_case_description_train_set.csv"), sep=","
        )
        self.mass_test_csv = pd.read_csv(
            join(self.__cbis_root_path, "csv", "mass_case_description_test_set.csv"), sep=","
        )
        self.calc_train_csv = pd.read_csv(
            join(self.__cbis_root_path, "csv", "calc_case_description_train_set.csv"), sep=","
        )
        self.calc_test_csv = pd.read_csv(
            join(self.__cbis_root_path, "csv", "calc_case_description_test_set.csv"), sep=","
        )

        tqdm.tqdm.pandas()

        # read mass train dataset
        self.mass_dataset_train = [
            (str(path), label)
            for path, label in zip(
                self.mass_train_csv["cropped image file path"].tolist(),
                self.mass_train_csv["pathology"].tolist()
            )
        ]

        # # add mass to label
        # for i in range(len(self.mass_dataset_train)):
        #     path = self.mass_dataset_train[i][0]
        #     label = self.mass_dataset_train[i][1]
        #     new_tuple = (path, "MASS_" + label)
        #     self.mass_dataset_train[i] = new_tuple

		# read mass test dataset
        self.mass_dataset_test = [
            (str(path), label)
            for path, label in zip(
                self.mass_test_csv["cropped image file path"].tolist(),
                self.mass_test_csv["pathology"].tolist()
            )
        ]

        # # add mass to label
        # for i in range(len(self.mass_dataset_test)):
        #     path = self.mass_dataset_test[i][0]
        #     label = self.mass_dataset_test[i][1]
        #     new_tuple = (path, "MASS_" + label)
        #     self.mass_dataset_test[i] = new_tuple

		# read calc train dataset
        self.calc_dataset_train = [
            (str(path), label)
            for path, label in zip(
                self.calc_train_csv["cropped image file path"].tolist(),
                self.calc_train_csv["pathology"].tolist()
            )
        ]

        # # add calc to label
        # for i in range(len(self.calc_dataset_train)):
        #     path = self.calc_dataset_train[i][0]
        #     label = self.calc_dataset_train[i][1]
        #     new_tuple = (path, "CALC_" + label)
        #     self.calc_dataset_train[i] = new_tuple

		# read calc test dataset
        self.calc_dataset_test = [
            (str(path), label)
            for path, label in zip(
                self.calc_test_csv["cropped image file path"].tolist(),
                self.calc_test_csv["pathology"].tolist()
            )
        ]

        # # add calc to label
        # for i in range(len(self.calc_dataset_test)):
        #     path = self.calc_dataset_test[i][0]
        #     label = self.calc_dataset_test[i][1]
        #     new_tuple = (path, "CALC_" + label)
        #     self.calc_dataset_test[i] = new_tuple

        # count benign and malignant images
        train_benign_count = 0
        train_malign_count = 0
        for path, label in self.mass_dataset_train:
            if label.startswith("BENIGN"):
                train_benign_count += 1

            if label.startswith("MALIGN"):
                    train_malign_count += 1

        print("train malignant count: ", train_malign_count)
        print("train benign count: ", train_benign_count)

        test_benign_count = 0
        test_malign_count = 0
        for path, label in self.mass_dataset_test:
            if label.startswith("BENIGN"):
                test_benign_count += 1

            if label.startswith("MALIGN"):
                    test_malign_count += 1

        print("test malignant count: ", test_malign_count)
        print("test benign count: ", test_benign_count)

        # mass only
        self.dataset_train = self.mass_dataset_train
        self.dataset_test = self.mass_dataset_test

        # calc only
        # self.dataset_train = self.calc_dataset_train
        # self.dataset_test = self.calc_dataset_test

        print("train dataset length: ",  len(self.dataset_train))
        print("test dataset length: ",  len(self.dataset_test))

        # augmented datasets

        # originals only
        # self.augments = ["original"]

        # with augments
        self.augments = ["original", "h_flip", "v_flip", "90", "180", "270", "h_flip_90", "v_flip_90", "h_flip_180", "v_flip_180", "h_flip_270", "v_flip_270"]
        self.augments_indices = []
        self.aug_dataset_train = []

        i = 0
        j = 0
        for augment in self.augments:
            self.aug_dataset_train = self.aug_dataset_train + self.dataset_train
            j = len(self.aug_dataset_train)
            self.augments_indices.append([i, j])
            i = j + 1

        for i in range(len(self.augments)):
            print(self.augments[i], self.augments_indices[i])
        
        print("augmented training dataset length: ", len(self.aug_dataset_train))

        self.__dataset = self.aug_dataset_train + self.dataset_test

        self.class_to_idx = {
            "benign": 0,
            "malignant": 1,
        }

    def getCroppedImagePathFromCSVPath(self, path: str):
        components = path.split('/')
        folder_name = components[2]
        images = listdir(join(self.__cbis_root_path, "jpeg", folder_name))

        for image in images:
            index = self.dicom_info_csv.index[self.dicom_info_csv['image_path'].str.contains(
                folder_name+'/'+image)][0]
            if self.dicom_info_csv['SeriesDescription'][index] == 'cropped images':
                image_path = join('jpeg', folder_name, image)
                return image_path

    def cbis_pil_loader(self, path: str) -> Image.Image:
        f = open(path, 'rb')
        img = Image.open(f)
        resized_img = img.resize((224, 224))
        f.close()
        return resized_img

    def __open_img(self, path: str, index: int) -> th.Tensor:
        file = self.cbis_pil_loader(
            join(self.__cbis_root_path, self.getCroppedImagePathFromCSVPath(path)))

        transforms = tr.ToTensor()

        augment = "original"
        for i in range(len(self.augments_indices)):
            if self.augments_indices[i][0] <= index <= self.augments_indices[i][1]:
                augment = self.augments[i]

        if augment == "original":
            transforms = tr.Compose([
                # tr.Grayscale(num_output_channels=3),
                tr.ToTensor()
            ])
        if augment == "h_flip":
            transforms = tr.Compose([
                tr.RandomHorizontalFlip(p=1),
                tr.ToTensor()
            ])
        if augment == "v_flip":
            transforms = tr.Compose([
                tr.RandomVerticalFlip(p=1),
                tr.ToTensor()
            ])
        if augment == "90":
            transforms = tr.Compose([
                tr.RandomRotation(degrees=(90, 90)),
                tr.ToTensor()
            ])
        if augment == "180":
            transforms = tr.Compose([
                tr.RandomRotation(degrees=(180, 180)),
                tr.ToTensor()
            ])
        if augment == "270":
            transforms = tr.Compose([
                tr.RandomRotation(degrees=(270, 270)),
                tr.ToTensor()
            ])
        if augment == "h_flip_90":
            transforms = tr.Compose([
                tr.RandomHorizontalFlip(p=1),
                tr.RandomRotation(degrees=(90, 90)),
                tr.ToTensor()
            ])
        if augment == "h_flip_180":
            transforms = tr.Compose([
                tr.RandomHorizontalFlip(p=1),
                tr.RandomRotation(degrees=(180, 180)),
                tr.ToTensor()
            ])
        if augment == "h_flip_270":
            transforms = tr.Compose([
                tr.RandomHorizontalFlip(p=1),
                tr.RandomRotation(degrees=(270, 270)),
                tr.ToTensor()
            ])
        if augment == "v_flip_90":
            transforms = tr.Compose([
                tr.RandomVerticalFlip(p=1),
                tr.RandomRotation(degrees=(90, 90)),
                tr.ToTensor()
            ])
        if augment == "v_flip_180":
            transforms = tr.Compose([
                tr.RandomVerticalFlip(p=1),
                tr.RandomRotation(degrees=(180, 180)),
                tr.ToTensor()
            ])
        if augment == "v_flip_270":
            transforms = tr.Compose([
                tr.RandomVerticalFlip(p=1),
                tr.RandomRotation(degrees=(270, 270)),
                tr.ToTensor()
            ])

        augmented_image = transforms(file)

        return augmented_image

    def __getitem__(self, index) -> Tuple[th.Tensor, th.Tensor]:
        img_path_csv = self.__dataset[index][0]

        label = self.__dataset[index][1]
        label_to_index = 0
        if label.startswith('BENIGN'):
            label_to_index = 0
        elif label.startswith('MALIGNANT'):
            label_to_index = 1
        img = self.__open_img(img_path_csv, index)

        return img, th.tensor(label_to_index)

    def __len__(self) -> int:
        return len(self.__dataset)

class MNISTDataset(ImageFolder):
    def __init__(self, res_path: str, img_transform: Any) -> None:
        mnist_root_path = join(res_path, "downloaded", "mnist_png", "all_png")

        assert exists(mnist_root_path) and isdir(mnist_root_path), \
            f"{mnist_root_path} does not exist or is not a directory"

        super().__init__(mnist_root_path, transform=img_transform,
                         target_transform=None, loader=my_pil_loader,
                         is_valid_file=None)


class RESISC45Dataset(ImageFolder):
    def __init__(self, res_path: str, img_transform: Any) -> None:
        resisc_root_path = join(res_path, "downloaded", "NWPU-RESISC45")

        assert exists(resisc_root_path) and isdir(resisc_root_path), \
            f"{resisc_root_path} does not exist or is not a directory"

        super().__init__(resisc_root_path, transform=img_transform,
                         target_transform=None, loader=my_pil_loader,
                         is_valid_file=None)


class AIDDataset(ImageFolder):
    def __init__(self, res_path: str, img_transform: Any) -> None:
        aid_root_path = join(res_path, "downloaded", "AID")

        assert exists(aid_root_path) and isdir(aid_root_path), \
            f"{aid_root_path} does not exist or is not a directory"

        super().__init__(aid_root_path, transform=img_transform,
                         target_transform=None, loader=my_pil_loader,
                         is_valid_file=None)


class KneeMRIDataset(Dataset):
    def __init__(self, res_path: str, img_transform: Any):
        super().__init__()

        self.__knee_mri_root_path = join(res_path, "downloaded", "knee_mri")

        self.__img_transform = img_transform

        metadata_csv = pd.read_csv(
            join(self.__knee_mri_root_path, "metadata.csv"), sep=","
        )

        tqdm.tqdm.pandas()

        self.__max_depth = -1
        self.__max_width = -1
        self.__max_height = -1
        self.__nb_img = 0

        def __open_pickle_size(fn: str) -> None:
            f = open(join(self.__knee_mri_root_path, "extracted", fn), "rb")
            x = pkl.load(f)
            f.close()
            self.__max_depth = max(self.__max_depth, x.shape[0])
            self.__max_width = max(self.__max_width, x.shape[1])
            self.__max_height = max(self.__max_height, x.shape[2])
            self.__nb_img += 1

        metadata_csv["volumeFilename"].progress_map(__open_pickle_size)

        self.__dataset = [
            (str(fn), lbl)
            for fn, lbl in zip(
                metadata_csv["volumeFilename"].tolist(),
                metadata_csv["aclDiagnosis"].tolist()
            )
        ]

        self.class_to_idx = {
            "healthy": 0,
            "partially injured": 1,
            "completely ruptured": 2
        }

    def __open_img(self, fn: str) -> th.Tensor:
        f = open(join(self.__knee_mri_root_path, "extracted", fn), "rb")
        x = pkl.load(f)
        f.close()

        x = th.from_numpy(x.astype(np.float)).to(th.float)

        # depth
        curr_depth = x.size(0)

        to_pad = self.__max_depth - curr_depth
        pad_1 = to_pad // 2 + to_pad % 2
        pad_2 = to_pad // 2

        # width
        curr_width = x.size(1)

        to_pad = self.__max_width - curr_width
        pad_3 = to_pad // 2 + to_pad % 2
        pad_4 = to_pad // 2

        # height
        curr_height = x.size(2)

        to_pad = self.__max_height - curr_height
        pad_5 = to_pad // 2 + to_pad % 2
        pad_6 = to_pad // 2

        return fun.pad(
            x, [pad_6, pad_5, pad_4, pad_3, pad_2, pad_1], value=0
        )

    def __getitem__(self, index) -> Tuple[th.Tensor, th.Tensor]:
        fn = self.__dataset[index][0]

        label = self.__dataset[index][1]
        img = self.__open_img(fn).unsqueeze(0)

        return img, th.tensor(label)

    def __len__(self) -> int:
        return self.__nb_img
