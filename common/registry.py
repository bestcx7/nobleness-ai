class Registry:
    mapping = {
        "builder_name_mapping": {},
        "task_name_mapping": {},
        "processer_name_mapping": {},
        "model_name_mapping": {},
        "lr_scheduler_name_mapping": {},
        "runner_name_mapping": {},
        "state": {},
        "paths": {},
    }
    @classmethod
    def register_builder(cls, name):
        r"""注册一个数据集生成器到以'name'作为关键字的注册表中

        参数：
           name: 生成器会被注册到的key

        使用方法：
           from common.registry import registry
           from datasets.base_dataset_builder import BaseDataserBuilder
        """

        def wrap(builder_cls):
            from datasets.builders.base_dataset_builder import BaseDatasetBuilder

            assert issubclass(
                builder_cls, BaseDatasetBuilder
            ), "All builders must inherit BaseDatasetBuilder class, found {}".format(
                builder_cls
            )
            if name in cls.mapping["builder_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["builder_name_mapping"][name]
                    )
                )
            cls.mapping["builder_name_mapping"][name] = builder_cls
            return builder_cls

        return wrap

    @classmethod
    def register_task(cls, name):
        r"""注册一个使用关键词'key'

        参数：
            name: 被注册的任务的key

        使用方法：
            from common.registry import registor
        """

        def wrap(task_cls):
            from tasks.base_task import BaseTask

            assert issubclass(
                task_cls, BaseTask
            ), "All tasks must inherit BaseTask class"
            if name in cls.mapping["task_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["task_name_mapping"][name]
                    )
                )
            cls.mapping["task_name_mapping"][name] = task_cls
            return task_cls
        
        return wrap
    
    @classmethod
    def register_model(cls, name):
        r"""
        使用关键词key注册一个任务

        参数：
            name: 被注册任务的key

        使用方法：
           from common.registry import registry
        """

        def wrap(model_cls):
            from models import BaseModel