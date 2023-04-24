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
        """
        从默认的配置文件中搭建一个预训练模型，由model_type定义

        参数：
           - model_type (str) : 模型类型， 由模型的结构和激活点定义

        返回值：
          - model (nn.module) : 预训练或者经过微调的模型， 取决于配置文件 
        """
        model_cfg = OmegaConf.load(cls.default_config_path(model_type)).model
        model = cls.from_pretrained(model_cfg)

        return model
    
    @classmethod
    def default_config_path(cls, model_type):
        assert (
            model_type in cls.PRETRAINED_MODEL_CONFIG_DICT
        ), "Unknow model type {}".format(model_type)
        return get_abs_path(cls.PRETRAINED_MODEL_CONFIG_DICT[model_type])
    
    def load_checkpoint_from_config(self, cfg, **kwargs):
        """
        加载被定义在配置文件里的权重文件

        如果 load_finetuned 为True, 加载微调好的模型， 否则， 加载预训练的模型
        当加载预训练好的模型时，每个 task-specific 结构可能会定义他们自己的load_from pretrained方法
        """
        load_finetuned = cfg.get("load_finetuned", True)
        if load_finetuned:
            finetune_path = cfg.get("finetuned", None)
            assert(
                finetune_path is not None
            ), "Found load_finetuned is True, but finetun_path is None."
            self.load_checkpoint(url_or_filename=finetune_path)
        else:
            # 加载预训练的权重
            pretrain_path = cfg.get("pretrained", None)
            assert"Found load_finetuned is False, but pretrain_path is None."
            self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)

    def before_evalution(self, **kwargs):
        pass

    def show_n_params(self, return_str=True):
        tot = 0
        for p in self.parameters():
            w = 1
            for x in p.shape:
                w *= x
            tot += w
        if return_str:
            if tot >= 1e6:
                return "{:.1f}M".format(tot / 1e6)
            else:
                return "{:.1f}K".format(tot / 1e3)
            
        else:
            return tot
        
class BaseEncoder(nn.Module):
    """
    基本class for primitive encoders, 例如 ViT, TimeSformer, etc.
    """

    def __init__(self):
        super().__init__()

    def forward_feature(self, samples, **kwargs):
        raise NotImplementedError
    
    @property
    def device(self):
        return list(self.parameters())[0].device