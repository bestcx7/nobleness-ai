import contextlib
import logging
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import common.dist_utils as dist_utils
from common.dist_utils import download_cached_file
from common.utils import is_url
from common.logger import MetricLogger
from models.base_model import BaseModel
