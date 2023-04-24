import logging
import os 

import numpy as np
import torch
import torch.nn as nn
from common.dist_utils import download_cached_file, is_dist_avail_and_initialized
from common.utils import get_abs_path, is_url
from omegaconf import OmegaConf

class BaseModel(nn.Module):
    """
    模型的基本类别
    """

    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def load_checkpoint(self, url_or_filename):
        """
        从一个微调好的checkpoint里加载

        希望模型的keys和checkpoint的keys不会错配
        """
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")
        
        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint %s" % url_or_filename)

        return msg
    
    @classmethod
    def from_pretrained(cls, model_type):