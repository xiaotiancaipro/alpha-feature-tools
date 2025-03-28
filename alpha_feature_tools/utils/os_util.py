import json
import os
import pickle
from typing import Any, List

from sklearn2pmml import PMMLPipeline, sklearn2pmml
from sklearn_pandas import DataFrameMapper


class Path(object):

    @classmethod
    def mkdir(cls, path: str | list):
        """创建路径"""
        if isinstance(path, list):
            for _ in path:
                cls.mkdir(_)
            return
        if os.path.exists(path):
            return
        os.makedirs(path)
        return


class File(object):

    @staticmethod
    def save_json(path: str, any_: Any) -> None:
        """保存为 Json 文件"""
        with open(path, "w") as f:
            json.dump(any_, f, indent=4)
            f.close()
        return None

    @staticmethod
    def read_json(path: str) -> Any:
        """读取 Json 文件"""
        with open(path, "r") as f:
            any_ = json.load(f)
            f.close()
        return any_

    @staticmethod
    def save_pkl(path: str, any_: Any) -> None:
        """保存为 PKL 包"""
        with open(path, "wb+") as f_:
            pickle.dump(any_, f_)
            f_.close()
        return

    @staticmethod
    def read_pkl(path: str) -> Any:
        """读取 PKL 包"""
        with open(path, "rb") as f_:
            any_ = pickle.load(f_)
            f_.close()
        return any_

    @staticmethod
    def save_text(path: str, text: str) -> None:
        with open(path, "w") as f:
            f.write(text)
            f.close()
        return None

    @staticmethod
    def save_pmml(path: str, model: Any, feature_list: List[str]) -> None:
        """保存为 Pmml 文件"""
        pipeline = PMMLPipeline([
            ("mapper", DataFrameMapper([(feature_list, None)])),
            ("model", model)
        ])
        sklearn2pmml(pipeline, path, with_repr=True)
        return None
