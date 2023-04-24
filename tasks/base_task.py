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
    
    def build_datasets(self, cfg):
        """
        构建一个数据集的字典， 使用"train", "valid", "test"划分
        如果不存在的话自动下载数据集和标注

        参数：
            cfg(commom.config.Config):_description_

        返回：
            字典：torch.utils.data.Dataset 被划分
        """

        datasets = dict()

        datasets_config = cfg.dataset_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            datasets_config = datasets_config[name]

            builder = registry.get_builder_class(name)(datasets_config)
            dataset = builder.build_datasets()

            datasets[name] = dataset

        return datasets
    
    def train_steps(self, model, samples):
        loss = model(samples)["loss"]
        return loss
    
    def valid_step(self, model, samples):
        raise NotImplementedError
    
    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evalation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError
    
    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results
    
    def train_epoch(
            self, 
            epoch,
            model,
            data_loader,
            optimizer, 
            lr_scheduler,
            scaler=None,
            cuda_enabled=False,
            log_freq=50,
            accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch,
            iter_per_epoch=len(data_loader),
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )
    
    def train_iters(
            self,
            epoch,
            start_iter,
            iters_per_inner_epoch,
            model,
            data_loader,
            optimizer,
            lr_scheduler,
            scaler=None,
            cuda_enabled=False,
            log_freq=50,
            accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iter=start_iter,
            iter_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )
    
    def _train_inner_loop(
            self,
            epoch,
            iters_per_epoch,
            model,
            data_loader,
            optimizer,
            lr_scheduler,
            scaler=None,
            start_iters=None,
            log_freq=50,
            cuda_enabled=False,
            accum_grad_iters=1,
    ):
        """
        内部训练循环可以兼容基于epoch和iter的训练。

        在基于epoch的训练中，训练会在一个epoch（即所有训练数据都被用于训练一次）之后停止
        而在基于iter的训练中，训练会在完成指定数量的迭代后（例如每个epoch需要迭代的次数）停止。
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1,fmt="{value:.4f}"))

        # 如果基于iter训练， schedule lr 基于 inner epoch
        logging.info(
            "Start training epoch {}, {}, iters per inner epoch.".format(
            epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # 基于epoch训练
            inner_epoch = epoch
        else:
            # 基于iter-based训练， 我们根据iterations计划学习率
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # 如果使用iter_based 训练，我们在iters_per_epoch后停止迭代
            if i >= iters_per_epoch:
                break
            
            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                       "epoch": inner_epoch,
                       "num_iter_per_epoch" : iters_per_epoch,
                       "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_epoch(model=model, samples=samples)

            # _after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 在每个accum_grad_iter iterations之后更行梯度
            if(i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_group[0]["lr"])

            # 在train_epoch()之后
            # 汇聚所有进程的状态
            metric_logger.synchronize_between_processes()
            logging.info("Averaged stats: " + str(metric_logger.global_avg()))
            return {
                k: "{:.3f}".format(meter.global_avg)
                for k, meter in metric_logger.meters.items()
            }
        
    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # 比较所有进程的结果
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file), "r")
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new
                
            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file