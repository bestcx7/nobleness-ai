import gzip
import logging
import os
import random as rnd
import tarfile
import zipfile

import decord
import webdataset as wds
import numpy as np
import torch
from torch.utils.data.dataset import IterableDataset, ChainDataset
from decord import VideoReader
from common.registry import registry
from datasets.datasets.base_dataset import ConcatDataset
from tqdm import tqdm

decord.bridge.set_bridge("torch")
MAX_INT = registry.get("MAX_INT")

def load_video(video_path, n_frms=MAX_INT, height=-1, width=-1, sampling="uniform"):
    """
    参数：
       video_path: 视频文件的路径
       n_frms: 采样帧数
       height: 采样高度
       width: 采样宽度
       sampling: 采样方法
    """
    vr = VideoReader(uri=video_path, height=height, width=width)

    vlen = len(vr)
    start, end = 0, vlen

    n_frms = min(n_frms, vlen)

    if sampling == "uniform":
        indices = np.arange(start, end, vlen / n_frms).astype(int)
    elif sampling == "headtail":
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplemented

    # get_batch -> T, H, W, C
    frms = vr.get_batch(indices).permute(3, 0, 1, 2).float()

    return frms

def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x
        
    return _apply(sample)

def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)

def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        samples = move_to_cuda(samples)

    # TODO fp16 support

    return samples

def reorg_datasets_by_split(datasets):
    """
    划分数据集

    参数：
       datasets: 其中每个键代表一个数据集的名称，每个值是一个torch.utils.data.Dataset对象

    返回：
       输出是一个字典，每个键表示数据集的一个部分，每个值是一个包含所有该部分数据集的列表。
    """
    reorg_datasets = dict()

    for _, dataset in datasets.items():
        for split_name, dataset_split in dataset.items():
            if split_name not in reorg_datasets:
                reorg_datasets[split_name] = [dataset_split]
            else:
                reorg_datasets[split_name].append(dataset_split)

    return reorg_datasets

def concat_datasets(datasets):
    """
    将多个数据集合并为一个数据集。

    该函数支持多种类型的数据集，包括 PyTorch 的 map-style 数据集和 WebDataset 的 DataPipeline。
    但是它目前不支持通用的 IterableDataset 类型数据集，因为需要创建单独的样本采样器。
    该函数仅支持合并训练数据集，而不支持合并验证集和测试集，因为对于这些数据集来说，指标不应该在合并后计算。

    参数：
        datasets: 被划分的数据集字典
    
    返回：
        函数返回值是一个字典，其中包含了合并后的数据集，
        "train" 是由多个数据集合并而成的，
        "val" 和 "test" 保持不变。
        如果输入的训练数据集既包含 map-style 数据集又包含 DataPipeline 数据集，则返回一个元组，其中第一个元素是合并后的 map-style 数据集，第二个元素是链式连接的 DataPipeline 数据集。
    """
    for split_name in datasets:
        if split_name != "train":
            assert(
                len(datasets[split_name]) == 1
            ), "Do not support multiple {} datasets.".format(split_name)
            datasets[split_name] = datasets[split_name][0]
        else:
            iterable_datasets, map_datasets = [], []
            for dataset in datasets[split_name]:
                if isinstance(dataset, wds.DataPipeline):
                    logging.info(
                        "Dataset {} is IterableDataset, can't be concatenated.".format(
                            dataset
                        )
                    )
                    iterable_datasets.append(dataset)
                elif isinstance(dataset, IterableDataset):
                    raise NotImplementedError(
                        "Do not support concatenation of generic IterableDataset."
                    )
                else:
                    map_datasets.append(dataset)

            # if len(iterable_datasets) > 0:
            # concatenate map-style datasets and iterable-style datasets separately
            chained_datasets = (
                ChainDataset(iterable_datasets) if len(iterable_datasets) > 0 else None
            )
            concat_datasets = (
                ConcatDataset(map_datasets) if len(map_datasets) > 0 else None
            )

            train_datasets = concat_datasets, chained_datasets
            train_datasets = tuple([x for x in train_datasets if x is not None])
            train_datasets = (
                train_datasets[0] if len(train_datasets) == 1 else train_datasets
            )

            datasets[split_name] = train_datasets

    return datasets


def extract_archive(from_path, to_path=None, overwrite=False):
    """解压文件.
    参数：

    from_path：压缩文件的路径。
    to_path：解压缩文件的根路径（即from_path的目录）。
    overwrite：是否覆盖已有文件（False为不覆盖）。

    Examples:
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> from_path = './validation.tar.gz'
        >>> to_path = './'
        >>> torchtext.utils.download_from_url(url, from_path)
        >>> torchtext.utils.extract_archive(from_path, to_path)
        >>> ['.data/val.de', '.data/val.en']
        >>> torchtext.utils.download_from_url(url, from_path)
        >>> torchtext.utils.extract_archive(from_path, to_path)
        >>> ['.data/val.de', '.data/val.en']

    """

    if to_path is None:
        to_path = os.path.dirname(from_path)

    if from_path.endswith((".tar.gz", ".tgz")):
        logging.info("Opening tar file {} to {}.".format(from_path, to_path))
        with tarfile.open(from_path, "r") as tar:
            files = []
            for file_ in tqdm(tar):
                file_path = os.path.join(to_path, file_.name)
                if file_.isfile():
                    files.append(file_path)
                    if os.path.exists(file_path):
                        logging.info("{} already extracted.".format(file_path))
                        if not overwrite:
                            continue
                tar.extract(file_, to_path)
            logging.info("Finished extracting tar file {}.".format(from_path))
            return files

    elif from_path.endswith(".zip"):
        assert zipfile.is_zipfile(from_path), from_path
        logging.info("Opening zip file {} to {}.".format(from_path, to_path))
        with zipfile.ZipFile(from_path, "r") as zfile:
            files = []
            for file_ in tqdm(zfile.namelist()):
                file_path = os.path.join(to_path, file_)
                files.append(file_path)
                if os.path.exists(file_path):
                    logging.info("{} already extracted.".format(file_path))
                    if not overwrite:
                        continue
                zfile.extract(file_, to_path)
        files = [f for f in files if os.path.isfile(f)]
        logging.info("Finished extracting zip file {}.".format(from_path))
        return files

    elif from_path.endswith(".gz"):
        logging.info("Opening gz file {} to {}.".format(from_path, to_path))
        default_block_size = 65536
        filename = from_path[:-3]
        files = [filename]
        with gzip.open(from_path, "rb") as gzfile, open(filename, "wb") as d_file:
            while True:
                block = gzfile.read(default_block_size)
                if not block:
                    break
                else:
                    d_file.write(block)
            d_file.write(block)
        logging.info("Finished extracting gz file {}.".format(from_path))
        return files

    else:
        raise NotImplementedError(
            "We currently only support tar.gz, .tgz, .gz and zip achives."
        )


def save_frames_grid(img_array, out_path):
    import torch
    from PIL import Image
    from torchvision.utils import make_grid

    if len(img_array.shape) == 3:
        img_array = img_array.unsqueeze(0)
    elif len(img_array.shape) == 5:
        b, t, c, h, w = img_array.shape
        img_array = img_array.view(-1, c, h, w)
    elif len(img_array.shape) == 4:
        pass
    else:
        raise NotImplementedError(
            "Supports only (b,t,c,h,w)-shaped inputs. First two dimensions can be ignored."
        )

    assert img_array.shape[1] == 3, "Exepcting input shape of (H, W, 3), i.e. RGB-only."

    grid = make_grid(img_array)
    ndarr = grid.permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    img = Image.fromarray(ndarr)

    img.save(out_path)
