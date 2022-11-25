# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import tarfile
import requests
from tqdm import tqdm
import shutil
from pathlib import Path


from paddleocr.ppocr.utils.logging import get_logger


def download_with_progressbar(url, save_path):
    logger = get_logger()
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size_in_bytes = int(response.headers.get('content-length', 1))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
    else:
        logger.error("Something went wrong while downloading models")
        sys.exit(0)

from pdb import set_trace
def maybe_download(model_storage_directory, url, force_redownload=False):
    # using custom model
    model_file_names = {
        'inference.pdiparams', 'inference.pdiparams.info', 'inference.pdmodel'
    }
    
    # If all files exist and force_redownload is False, return without doing anything
    if os.path.exists( os.path.join(model_storage_directory, 'inference.pdiparams')) and os.path.exists(
            os.path.join(model_storage_directory, 'inference.pdmodel')) and not force_redownload:
        return
    
    # Make the directory to store the models
    os.makedirs(model_storage_directory, exist_ok=True)
    # If the url is a web url and ends with .tar, download
    if url.endswith('.tar'):
        if url.startswith("http"):
            tmp_tarfile_path = os.path.join(model_storage_directory, url.split('/')[-1])
            download_with_progressbar(url, tmp_tarfile_path)
            print('Downloading {} to {}'.format(url, tmp_tarfile_path))
        else:
            tmp_tarfile_path = url
        # Extract file to temporary folder
        extract_tar(model_storage_directory, model_file_names, tmp_tarfile_path)
    # if url is a path
    elif os.path.isdir(url):
        for tar_file_name in model_file_names:
            file = os.path.join(url,tar_file_name)
            if os.path.isfile(file):
                # print("copying {} to {}".format(file,model_storage_directory))
                shutil.copy(file, model_storage_directory)
                print(f"{file} Reloaded.")
    elif not os.path.exists(url):
        raise Exception(f"Library path not found: {url}")
    else:
        raise Exception(f"Unknown library path provided: {url}")

def extract_tar(model_storage_directory, model_file_names : set , tmp_tarfile_path):
    
    with tarfile.open(tmp_tarfile_path, 'r') as tarObj:
        for member in tarObj.getmembers():
            if os.path.basename(member.name) in model_file_names:
                # This is a required file, extract it
                # dest_filepath = os.path.join(model_storage_directory, member.name)
                # Change the member.name to it's basename, there will no longer be nested directory in output.
                # Be careful of duplicated names!
                member.name = os.path.basename(member.name)
                tarObj.extract(member, model_storage_directory)
                print(f"{member.name} Reloaded.")

def is_link(s):
    return s is not None and s.startswith('http')

def confirm_model_dir_url(model_dir, default_model_dir, default_url):
    url = default_url
    if model_dir is None or is_link(model_dir):
        if is_link(model_dir):
            url = model_dir
        # file_name = url.split('/')[-1][:-4]
        
        file_name = Path(url).stem if url.endswith(".tar") else Path(url).name
        model_dir = default_model_dir
        model_dir = os.path.join(model_dir, file_name)
    return model_dir, url
