import json
from typing import Iterable

from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate

class BaseDataset(Dataset):
    def __init__(
        self,
        vis_processor=None,
        text_processor=None,
        vis_root=None,
        ann_paths=[]
    ):
        """
        参数：
            vis_root(string): 图像数据集的根目录
            ann_root(string): 标注文件的存储目录
        """
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r")))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_procesor):
        self.vis_processor = vis_processor
        self.text_processor = text_procesor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

class ConcatDataset(ConcatDataset):
    """
    接受一个数据集列表作为输入，用于将多个数据集合并为一个数据集
    """
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        # ToDo目前仅支持相同底层的数据整合

        all_keys = set()
        for s in samples:
            all_keys.update(s)
        
        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.key())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)