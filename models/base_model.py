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
    
class sharedQueueMixin:
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idxs=None):
        # 在更新队列之前，汇聚所有的key
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0 # 为了简单一点

        # 在prt中替换keys(出队和入队)
        self.image_queue[:, ptr : ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr : ptr + batch_size] = text_feats.T

        if idxs is not None:
            idxs = concat_all_gather(idxs)
            self.idx_queue[:, ptr : ptr + batch_size] = idxs.T

        ptr = (ptr + batch_size) % self.queue_size # 移动指针
        self.queue_ptr[0] = ptr


class MonentumDistilationMixin:
    # 将源模型(model_pair[0])的参数值拷贝到目标模型(model_pair[1])的参数值上
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters, model_pair[1].parameters()
            ):
                param_m.data = param_m.data * self.momentum + param.data * (
                    1.0 - self.momentum
                )

class GatherLayer(torch.autograd.Function):
    """
    从所有的进程中收集支持反向传播的的张量
    通过调用torch的分布式库在不截断梯度上实现
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributions.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)
    
    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]
    
    
def all_gather_with_grad(tensors):
    """
    对输入的张量进行all_gather操作并进行图连接以便反向传播
    """
    world_size = torch.distributed.get_world_size()
    # 如果是单进程，不需要all_gather
    if world_size == 1:
        return tensors
    
    # tensor_all = GatherLayer.apply(tensor)
    tensor_all = GatherLayer.apply(tensors)
    
    return torch.cat(tensor_all, dim=0)

@torch.no_grad
def concat_all_gather(tensor):
    """
    在提供的张量上进行all_gather操作
    ***警告***:torch.distributed.all_gather没有梯度
    """
    # 如果使用分布式训练
    if not is_dist_avail_and_initialized():
        return tensor
    
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
    )
    return torch.index_select(x, dim, order_index.to(x.device))