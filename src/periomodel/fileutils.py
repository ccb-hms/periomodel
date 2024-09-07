"""
File operations including downloading and extracting
Andreas Werdich
Center for Computational Biomedicine
"""

import os
import shutil
import logging
import contextlib
import traceback
import pydicom
import glob
from pathlib import Path
import pandas as pd
from urllib import request
from urllib.error import HTTPError
import gzip
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class DownloadProgressBar(tqdm):
    """ Small helper class to make a download bar """

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def chunks(input_list, n):
    """
    Args:
        input_list: The list or iterable to be divided into chunks.
        n: The size of each chunk.
    Returns:
        A generator that yields chunks of size n from the input list or iterable.
    """
    for i in range(0, len(input_list), n):
        yield input_list[i:i + n]


class FileOP:
    """
    Class for file operations including downloading, extracting and file size checking.

    Parameters:
    - data_output_dir (Optional[str]): The directory to store the downloaded files.

    Attributes:
    - data_output_dir (str): The directory to store the downloaded files.
    - url (None or str): The URL of the file to download.

    Methods:
    - unzip(in_file: str, out_file: str) -> int: Unzips a .gz file and returns the file size.
    - file_size_from_url(url: str) -> int: Gets the size of a file without downloading it.
    - download_from_url(url: str, download_dir: str, extract: bool = True, delete_after_extract: bool = False, ext_list: Optional[List[str]] = None) -> str: Downloads a file from a URL and
    * returns the file path.

    """

    def __init__(self, data_output_dir=None):
        self.data_output_dir = data_output_dir
        self.url = None

    def unzip(self, in_file, out_file):
        """
        Unzip .gz file and return file size
        :param in_file: complete file path of compressed .gz file
        :param out_file: complete file path of output file
        :return: os.path.getsize(out_file) in bytes
        """
        if not os.path.isfile(out_file):
            try:
                with gzip.open(in_file, 'rb') as f_in, open(out_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            except Exception as e:
                logger.error(f'gzip failed on file: {in_file}: {e}')
                print(f'gzip failed on file: {in_file}: {e}')
                file_size = None
            else:
                file_size = os.path.getsize(out_file)
        else:
            print(f'Uncompressed output file exists: {out_file}. Skipping.')
            file_size = os.path.getsize(out_file)
        return file_size

    def file_size_from_url(self, url):
        """
        Method to acquire size of a file without download
        :param: url
        :returns: size in bytes (int)
        """
        url_size = np.nan
        try:
            with contextlib.closing(request.urlopen(url)) as ul:
                url_size = ul.length
        except HTTPError as http_err:
            logger.error(f'ERROR: {http_err}: URL: {url}')
        except Exception as e:
            logger.error(f'ERROR {e}: URL: {url}')
        return url_size

    def download_from_url(self, url, download_dir, extract=True, delete_after_extract=False, ext_list=None):
        """
        :param url: cloud storage location URL
        :param download_dir: path-like object representing file path.
        :param extract: extract file if compressed
        :param delete_after_extract: if file is an archive, delete file after extraction.
        :param ext_list: list of allowed extensions, for example '.json.gz' or '.zip'
        :return: file path of output file
        """
        output_file_name = os.path.basename(url)
        if ext_list is not None:
            ext_in_url = [xt for xt in ext_list if xt in url]
            if len(ext_in_url) > 0:
                xt = ext_in_url[0]
                output_file_name = f'{output_file_name.split(xt, maxsplit=1)[0]}{xt}'
        output_file = os.path.join(download_dir, output_file_name)
        if os.path.exists(download_dir):
            if not os.path.exists(output_file):
                try:
                    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_file_name) as t:
                        request.urlretrieve(url, filename=output_file, reporthook=t.update_to)
                except HTTPError as http_err:
                    print(http_err)
                    logger.error(f'Download failed for URL: {url}'
                                 f' {http_err}')
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f'Download failed for URL: {url}'
                                 f' {e}')
                else:
                    logger.info(f'Download complete: {output_file}.')
            else:
                logger.info(f'File exists: {output_file}')
            # Unpacking
            output_file_path = output_file
            if os.path.exists(output_file) and extract:
                file_parts = os.path.splitext(output_file)
                xt = file_parts[-1]
                if xt in ['.gz']:
                    print(f'Extracting from {xt} archive.')
                    out_file = file_parts[0]
                    file_size = self.unzip(in_file=output_file, out_file=out_file)
                    if file_size is not None:
                        output_file_path = out_file
                        if delete_after_extract:
                            os.unlink(output_file)
                            logger.info(f'Deleted compressed file {output_file}')
                elif xt in ['.json', '.csv', '.pickle', '.parquet', '.ckpt', '.pth']:
                    print(f'Created {xt} file.')
                else:
                    print(f'File: {xt} loaded.')
                    logger.warning(f'File extension is unexpected {xt}.')
            elif os.path.exists(output_file) and not extract:
                output_file_path = output_file
        else:
            logger.error(f'Output directory {download_dir} does not exist.')
            output_file_path = None
        return output_file_path

    def load_file(self, file, kind=None):
        """
        Load file into memory with context manager and error handling
        Parameters:
            file, str, file path
            kind, str, one in ['dicom', 'pandas', 'txt']
        Returns:
            output, type of kind
        """
        msg = f'Unable to open {kind} file: {file}.'
        output = None
        if os.path.exists(file):
            if kind == 'dicom':
                try:
                    with open(file, 'rb') as fl:
                        ds = pydicom.dcmread(fl)
                except Exception as ex:
                    logger.error(msg=f'{msg} \n {ex}')
                else:
                    output = ds
            else:
                logger.error(msg=f'{msg} \n Unknown file type.')
                raise NotImplementedError
        else:
            logger.error(msg=f'{msg} \n File not found.')
        return output

    def search_file_tree(self, top_dir, file_pattern):
        """
        Search directory recursively for file pattern
        Parameters
        ----------
        top_dir: str, top file directory
        file_pattern: str, file pattern e.g. '*part.jpg'
        Returns
        -------
        output_df: pd.DataFrame with all files
        """
        output_df = None
        file_list = []
        print(f'Searching for files from top-dir "{top_dir}"')
        if os.path.exists(top_dir):
            file_list = glob.glob(os.path.join(top_dir, '**', f'{file_pattern}'), recursive=True)
        else:
            print(f'top_dir: {top_dir} does not exist.')
        if len(file_list) > 0:
            file_name_list = [os.path.basename(file) for file in file_list]
            file_dir_list = [os.path.split(file)[0] for file in file_list]
            file_dict = {'file_name': file_name_list,
                         'file_dir': file_dir_list,
                         'file': file_list}
            output_df = pd.DataFrame(file_dict)
        return output_df


class Flag:
    def __init__(self, flag_dir):
        self.flag_dir = flag_dir
        self.flag_option_list = ['started', 'success', 'failed']

    def set_flag_file(self, flag_base, flag, clean_flags=True):
        """
        Set a new flag file
        :param: flag_base: str, file basename of flag file without extension
        :param: flag: str, file extension of flag file must be in flag_option_list
        :param: clean_flags: remove all flags with flag_base and set new flag
        :return: dict{flag_option: flag file path}
        """
        flag_file_list = [os.path.join(self.flag_dir, f'{flag_base}.{flg}') for flg in self.flag_option_list]
        flag_file_dict = dict(zip(self.flag_option_list, flag_file_list))

        assert flag in self.flag_option_list, f'Wrong flag parameter. Must be from list {self.flag_option_list}'

        if Path(self.flag_dir).exists():
            if clean_flags:
                for fl in flag_file_dict.values():
                    Path(fl).unlink(missing_ok=True)
            with open(flag_file_dict.get(flag), 'w') as fl:
                fl.write('')
        else:
            print(f'Folder flag_dir={self.flag_dir} does not exist.')

        return flag_file_dict.get(flag)

    def find_flags(self):
        """
        Find flag files in flag_dir
        :return: list of flag_base names
        """
        flag_pat_list = [os.path.join(self.flag_dir, f'*.{flg}') for flg in self.flag_option_list]
        flag_file_dict = dict(zip(self.flag_option_list, flag_pat_list))

        flag_exist_list = []
        for file_pat in flag_file_dict.values():
            flag_exist_list.extend(glob.glob(file_pat))

        flag_base_exist_list = [os.path.splitext(os.path.basename(file))[0] for file in flag_exist_list]

        return flag_base_exist_list
