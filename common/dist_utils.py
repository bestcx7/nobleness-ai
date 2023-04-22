import datetime
import functools
import os

import torch
import torch.distributed as dist
import timm.models.hub as timm_hub

def setup_for_distribution(is_master):
    """
    如果不是在主进程就不会打印
    This functional disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)
        
    __builtin__.print = print

def is_dist_avail_and_initialized():
    """
    如果分布式不可用或者分布式没有被初始化都会返回False
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    """
    获取全局的gpu数，如果分布式没有初始化成功则返回1
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    """
    获取当前的进程，如果分布式没有被初始化则返回0
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0

def init_distributed_mode(args):
    """
    1.如果环境变量里有RANK,WORLD_SIZE等信息，就从环境变量获取
    2.如果环境变量里有SLURM_PROCID,就从SLURM调度器获取位于集群中的进程ID
    """
    if "RANK" in os.environ and "WORLDA_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode") 
        args.distributed = False

    args.distributed = True 

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}, world {}): {}".format(
            args.rank, args.world_size, args.dist_url
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        worldsize=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(
            days=365
        ),
    )
    torch.distributed.barrier()
    setup_for_distribution(args.rank == 0)  


def get_dist_info():
    if torch.__version__ < "1.0":
        initialized = dist.initialized
    else:
        initialized = dist.is_initialized
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def main_process(func):
    # 获取分布式的信息，并在rank=0上打印
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)
        
    return wrapper


def download_cached_file(url, check_hash=True, progress=False):
    """
    从url上下载文件，并在本地缓存，如果文件已经存在，就不会再次下载
    如果是分布式的话，只在主进程上下载文件，其他进程等待文件被下载
    Download a file from a URL and cache it locally. If the file already exists, it is not downloaded again.
    If distributed, only the main process downloads the file, and the other processes wait for the file to be downloaded.    
    """
    def get_cached_file_path():
        parts = torch.hub.urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(timm_hub.get_cache_dir(), filename)

        return cached_file
    
    if is_main_process():
        timm_hub.download_cached_file(url, check_hash, progress)

    if is_dist_avail_and_initialized():
        dist.barrier()

    return get_cached_file_path()