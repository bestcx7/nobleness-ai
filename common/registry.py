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

            assert issubclass(
                model_cls, BaseModel
            ), "All models inherit BaseModel class"
            if name in cls.mapping["model_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["model_name_mapping"][name]
                    )
                )
            cls.mapping["model_name_mapping"][name] = model_cls
            return model_cls
        
        return wrap
    

    @classmethod
    def register_processor(cls, name):
        r"""
        将一个处理器（processor）注册到名为“registry”的注册表中，并用键名“name”进行标识。

        参数:
            是一个字符串类型的“name”，用于标识该处理器在注册表中的位置。

        用法:
            from common.registry import registry
        """

        def wrap(processor_cls):
            from processors import BaseProcessor

            assert issubclass(
                processor_cls, BaseProcessor
            ), "All processors must inherit BaseProcessor class"
            if name in cls.mapping["processor_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["processor_name_mapping"][name]
                    )
                )
            cls.mapping["processor_name_mapping"][name] = processor_cls
            return processor_cls

        return wrap
    

    @classmethod
    def register_lr_scheduler(cls, name):
        r"""
        将一个模型（model）注册到名为“registry”的注册表中，并用键名“name”进行标识。

        参数：
           是一个字符串类型的“name”，用于标识该任务在注册表中的位置。

        用法：
           from common.registry import registry
        """

        def wrap(lr_sched_cls):
            if name in cls.mapping["lr_scheduler_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["lr_scheduler_name_mapping"][name]
                    )
                )
            cls.mapping["lr_scheduler_name_mapping"][name] = lr_sched_cls
            return lr_sched_cls

        return wrap
    
    @classmethod
    def register_runner(cls, name):
        r"""
        将一个模型（model）注册到名为“registry”的注册表中，并用键名“name”进行标识。

        参数：
           是一个字符串类型的“name”，用于标识该任务在注册表中的位置。

        用法：
           from common.registry import registry
        """

        def wrap(runner_cls):
            if name in cls.mapping["runner_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["runner_name_mapping"][name]
                    )
                )
            cls.mapping["runner_name_mapping"][name] = runner_cls
            return runner_cls

        return wrap
    

    @classmethod
    def register_path(cls, name, path):
        r"""
        将一个路径（path）注册到名为“registry”的注册表中，并用键名“name”进行标识。

        参数：
           是一个字符串类型的“name”，用于标识该路径在注册表中的位置。

        用法：
           from common.registry import registry
        """
        assert isinstance(path, str), "All path must be str."
        if name in cls.mapping["paths"]:
            raise KeyError("Name '{}' already registered.".format(name))
        cls.mapping["paths"][name] = path


    @classmethod
    def register(cls, name, obj):
        r"""
        将一个item注册到名为“registry”的注册表中，并用键名“name”进行标识。

        参数：
           是一个字符串类型的“name”，用于标识item在注册表中的位置。

        用法：
           from common.registry import registry

           registry.register("config", {})
        """
        path = name.split(".")
        current = cls.mapping["state"]

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj     

# @classmethod
# def get_trainer_class(cls, name):
#     return cls.mapping["trainer_name_mapping"].get(name, None)  

    @classmethod
    def get_builder_class(cls, name):
        return cls.mapping["builder_name_mapping"].get(name, None)

    @classmethod
    def get_model_class(cls, name):
        return cls.mapping["model_name_mapping"].get(name, None)

    @classmethod
    def get_task_class(cls, name):
        return cls.mapping["task_name_mapping"].get(name, None)

    @classmethod
    def get_processor_class(cls, name):
        return cls.mapping["processor_name_mapping"].get(name, None)

    @classmethod
    def get_lr_scheduler_class(cls, name):
        return cls.mapping["lr_scheduler_name_mapping"].get(name, None)

    @classmethod
    def get_runner_class(cls, name):
        return cls.mapping["runner_name_mapping"].get(name, None)

    @classmethod
    def list_runners(cls):
        return sorted(cls.mapping["runner_name_mapping"].keys())

    @classmethod
    def list_models(cls):
        return sorted(cls.mapping["model_name_mapping"].keys())

    @classmethod
    def list_tasks(cls):
        return sorted(cls.mapping["task_name_mapping"].keys())

    @classmethod
    def list_processors(cls):
        return sorted(cls.mapping["processor_name_mapping"].keys())

    @classmethod
    def list_lr_schedulers(cls):
        return sorted(cls.mapping["lr_scheduler_name_mapping"].keys())

    @classmethod
    def list_datasets(cls):
        return sorted(cls.mapping["builder_name_mapping"].keys())

    @classmethod
    def get_path(cls, name):
        return cls.mapping["paths"].get(name, None)
    
    @classmethod
    def get(cls, name, default=None, no_warning=False):
        r"""
        从注册表中获取item和key 'name'

        参数：
            name：要检索的键名，字符串类型。
            default：可选参数，当键名不存在于注册表中时，返回该默认值，并产生一个警告。默认值为 None。
            no_warning：可选参数，当设置为 True 时，键名不存在于注册表中不会产生警告，通常用于 MMF 的内部操作。默认值为 False。 
        """
        original_name = name
        name = name.split(".")
        value = cls.mapping["state"]
        for subname in name:
            value = value.get(subname, default)
            if value is default:
                break

        if (
            "writer" in cls.mapping["state"]
            and value == default
            and no_warning is False
        ):
            cls.mapping["state"]["writer"].warning(
                "Key {} is not present in registry, returning default value"
                "of {}".format(original_name, default)
            )
        return value
    
    @classmethod
    def unregister(cls, name):
        r"""
        从注册表移除item和key "name"

        参数：
           name： 需要被移除的key

        使用方法：
           
           from common.registry import registry

           config = registry.unregistry("config")
        """
        return cls.mapping["state"].pop(name)
    

registry = Registry()