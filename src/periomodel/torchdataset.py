"""
Create PyTorch data set from data frames
Andreas Werdich
Center for Computational Biomedicine
"""

import numpy as np
import logging
import torch
import torchvision
from torch.utils.data import Dataset
import albumentations as alb
import cv2
from albumentations.augmentations.geometric.resize import LongestMaxSize
from albumentations.augmentations.geometric.transforms import PadIfNeeded

# Imports from this module
from periomodel.imageproc import ImageData, validate_image_data

logger = logging.getLogger(name=__name__)


def augmentations(im_size, image_mean, image_std, augment=None):
    if augment == 'augment_1':
        aug_transform = alb.Compose([
            alb.Resize(im_size + 32, im_size + 32),
            alb.RandomCrop(im_size, im_size),
            alb.HorizontalFlip(),
            alb.ShiftScaleRotate(),
            alb.Blur(),
            alb.RandomGamma(),
            alb.Sharpen(),
            alb.GaussNoise(),
            alb.CoarseDropout(max_holes=16, max_width=16, max_height=16),
            alb.CLAHE(),
            alb.Normalize(mean=image_mean, std=image_std)])

    elif augment == 'augment_2':
        aug_transform = alb.Compose([
            alb.Resize(im_size + 32, im_size + 32),
            alb.RandomCrop(im_size, im_size, p=0.5, always_apply=True),
            alb.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=25,
                                 p=0.5, always_apply=True),
            alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,
                                         p=0.5, always_apply=True),
            alb.RandomGamma(gamma_limit=(80, 120), p=0.5, always_apply=True),
            alb.GaussianBlur(blur_limit=(3, 7), p=0.5, always_apply=True),
            alb.MotionBlur(blur_limit=7, p=0.7, always_apply=True),
            alb.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.5, always_apply=True),
            alb.CLAHE(p=0.5, always_apply=True),
            alb.HorizontalFlip(p=0.5, always_apply=True),
            alb.CoarseDropout(max_holes=16, max_width=8, max_height=8, p=0.5, always_apply=False),
            alb.PixelDropout(dropout_prob=0.01, per_channel=False, drop_value=0, mask_drop_value=None, p=0.5,
                             always_apply=False),
            alb.Normalize(mean=image_mean, std=image_std)])

    elif augment is None:
        aug_transform = alb.Compose([alb.Resize(im_size, im_size),
                                     alb.Normalize(mean=image_mean, std=image_std)])

    else:
        error_msg = f'Unknown augmentation. Parameter "augment" must be "augment_1", "augment_2" or "None".'
        logger.error(error_msg)
        raise ValueError(error_msg)
    return aug_transform


def load_and_process_image(image_file_path, max_im_size=512, hist_eq=True, gray=True):
    """
    Parameters:
        image_file_path: Path to the image file.
        max_im_size: Maximum size of the image. Default is 512.
        hist_eq: Whether to apply histogram equalization. Default is True.
        gray: Whether to convert the image to grayscale. Default is True.

    Returns:
        Processed image as per the specified parameters.
    """
    transform = alb.Compose([LongestMaxSize(max_size=max_im_size),
                             PadIfNeeded(min_height=max_im_size,
                                         min_width=max_im_size,
                                         border_mode=cv2.BORDER_CONSTANT,
                                         value=0)])
    im = ImageData().load_image(image_file_path)
    if hist_eq:
        im = ImageData().hist_eq(im)
    if gray:
        im = ImageData().np2color(im, color_scheme='GRAY')
    im_output = transform(image=im)['image']
    return im_output


class DatasetFromDF(Dataset):
    """
    This class represents a dataset for loading images from a pandas DataFrame object.
    Parameters:
        data (pandas DataFrame): The input DataFrame object containing image data.
        file_col (str): The name of the column in the DataFrame that contains the image file paths.
        label_col (str): The name of the column in the DataFrame that contains the corresponding image labels.
        max_im_size (int): The maximum size of the loaded images.
        transform (optional, callable): A callable function for image transformation, such as color conversion.
        validate (optional, bool): If set to True, the dataset will perform image data validation upon initialization.
    Methods:
        __len__():
            Returns the length of the dataset.
        __getitem__(idx):
            Returns the image and label at the specified index as a tuple.
    """

    def __init__(self,
                 data,
                 file_col,
                 label_col,
                 max_im_size,
                 transform=None,
                 hist_eq=True,
                 gray=True,
                 validate=False):
        self.df = data
        self.file_col = file_col
        self.label_col = label_col
        self.max_im_size = max_im_size
        self.transform = transform
        self.hist_eq = hist_eq
        self.gray = gray
        self.validate = validate
        if self.validate:
            validate_image_data(data_df=self.df, file_path_col=self.file_col)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        idx = idx % self.__len__()
        df_idx = self.df.iloc[idx]
        file, label = df_idx[self.file_col], df_idx[self.label_col]
        assert isinstance(label, np.int64), f'Label must be type np.int64.'
        # Image preprocessing, e.g., color conversion
        img = load_and_process_image(image_file_path=file,
                                     max_im_size=self.max_im_size,
                                     hist_eq=self.hist_eq,
                                     gray=self.gray)
        if self.transform:
            img = self.transform(image=img)['image']
        img_tensor = torch.from_numpy(img)
        # For loading gray scale images, we need to add a color channel
        if self.gray:
            img_tensor = torch.unsqueeze(img_tensor, dim=0)
        else:
            img_tensor = img_tensor.permute(2, 0, 1)
        label_tensor = torch.from_numpy(np.array(label))
        output = tuple([img_tensor, label_tensor])
        return output


class TorchDataset:
    """
    A class representing a Torch Dataset.
    Parameters:
        data (pd.DataFrame): The input data.
        file_col (str): The column name containing the file paths.
        label_col (str): The column name containing the labels.
        max_im_size (int, optional): The maximum size of the images. Default is 2500.
        im_size (int, optional): The desired size of the images. Default is 512.
    Methods:
        __init__(data, file_col, label_col, max_im_size=2500, im_size=512)
            Initializes the TorchDataset instance.
        max_shape()
            Returns the maximum dimension (width or height) of the images in the dataset.
        create_dataset_from_index(index_set, augment)
            Creates a PyTorch Dataset from a subset of the data frame.
    """

    def __init__(self,
                 data,
                 file_col,
                 label_col,
                 im_mean,
                 im_std,
                 max_im_size=2500,
                 im_size=512,
                 hist_eq=True,
                 gray=True):
        self.file_col = file_col
        self.label_col = label_col
        self.im_mean = im_mean
        self.im_std = im_std
        self.data = validate_image_data(data, self.file_col)
        self.max_im_size = max_im_size
        self.im_size = im_size
        self.hist_eq = hist_eq
        self.gray = gray

    def max_shape(self):
        """
        Method: max_shape
        Description:
        This method calculates the maximum dimension (either width or height)
        of the images in the dataset and returns the value.
        Returns:
            max_dim (int): The maximum dimension of the images in the dataset.
        """
        file_list = list(self.data.get(self.file_col).unique())
        logger.info(f'opening {len(file_list)} files')
        shape_list = [ImageData().load_image(file).shape for file in file_list]
        width_list = [sh[1] for sh in shape_list]
        height_list = [sh[0] for sh in shape_list]
        self.data = self.data.assign(width=width_list, height=height_list)
        max_dim = np.max((np.max(width_list), np.max(height_list)))
        return max_dim

    def create_dataset_from_index(self, index_set, augment):
        """
        Creates a dataset from the given index set.
        Parameters:
        -   index_set (set): A set containing the indexes of the data to include in the dataset.
        -   augment (bool): Flag indicating whether data augmentation should be applied.
        -   image_mean (float, optional): Mean value of the input image.
        -   image_std (float, optional): Standard deviation of the input image.
        Returns:
        -   DatasetFromDF: The created dataset.
        Raises:
        -   TypeError: If the index_set parameter is not a set.
        Warning:
        -   The number of rows in the output data frame may be smaller than the length of the index_set.
        """
        aug_transform = augmentations(im_size=self.im_size,
                                      augment=augment,
                                      image_mean=self.im_mean,
                                      image_std=self.im_std)

        if not isinstance(index_set, set):
            raise TypeError(f'Parameter "index_set" must be a set.')
        df = self.data.loc[self.data.index.isin(index_set)]
        if not df.shape[0] == len(index_set):
            logger.warning(f'Warning: The number of rows in the output '
                           'data frame is smaller than the length of the index_set: '
                           f'{df.shape[0]} != {len(index_set)}')

        dataset = DatasetFromDF(data=df,
                                file_col=self.file_col,
                                label_col=self.label_col,
                                max_im_size=self.max_im_size,
                                transform=aug_transform,
                                hist_eq=self.hist_eq,
                                gray=self.gray,
                                validate=True)
        return dataset


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Class to perform inverse normalization on a given tensor.
    Uses torchvision.transforms.Normalize as base class.

    Attributes:
        mean (torch.Tensor): The mean values for normalization.
        std (torch.Tensor): The standard deviation values for normalization.

    Methods:
        __init__(mean, std):
            Initializes the NormalizeInverse object.

        __call__(tensor):
            Applies the inverse normalization transformation on the input tensor.
    """
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
