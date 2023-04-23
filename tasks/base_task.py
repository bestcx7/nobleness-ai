import logging
import os

import torch
import torch.distributed as dist
from common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from common.logger import MetricLogger, SmoothedValue
from common.registry import registry
from datasets.data_utils import prepare_sample

class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)