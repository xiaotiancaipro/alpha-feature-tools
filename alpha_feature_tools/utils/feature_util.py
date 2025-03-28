import datetime
import random
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

DATE_INFO = "date_info"
DATE_MONTH = "date_month"
LABEL = "label"
FLAG = "flag"
SAMPLE_USE = "sample_use"
RANDOM_SEED = 211


class Feature(object):

    @staticmethod
    def df_sample(df: pd.DataFrame, p: float = 0.20, date_month: str = DATE_MONTH, label: str = LABEL):
        """
        样本抽样: 将不同月份的坏占比控制为 p

        :param df: 数据 DataFrame
        :param p: 需要控制的坏人占比
        :param date_month: 月份
        :param label: 标签
        :return 抽样后坏占比为 p 的样本
        """
        sampled_data = list()
        grouped = df.groupby(date_month)
        for month, group in grouped:
            label_0_data = group[group[label] == 0]
            label_1_data = group[group[label] == 1]
            l_0 = len(label_0_data)
            l_1 = len(label_1_data)
            if l_1 / (l_0 + l_1) <= p:
                n_0 = int(((1 - p) / p) * l_1)
                sampled_label_0 = label_0_data.sample(n=n_0, random_state=42)
                sampled_group = pd.concat([sampled_label_0, label_1_data])
            else:
                n_1 = int((p / (1 - p)) * l_0)
                sampled_label_1 = label_1_data.sample(n=n_1, random_state=42)
                sampled_group = pd.concat([label_0_data, sampled_label_1])
            sampled_data.append(sampled_group)
        sampled_df = pd.concat(sampled_data)
        return sampled_df

    @staticmethod
    def data_split(
            df: pd.DataFrame,
            year: int,
            month: int,
            day: int,
            date_info: str = DATE_INFO,
            sample_use: str = SAMPLE_USE,
            split_class: str = "train_test-oot",
            is_split: bool = True
    ) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame] | Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        根据指定日期切分数据集为 train, test 和 oot

        :param df: 原始数据
        :param year: 年
        :param month: 月
        :param day: 日
        :param date_info: 时间信息列
        :param sample_use: 切分数据集后的标记列, 0 表示训练测试样本, 1 表示 OOT 样本
        :param split_class: 切分数据集的种类, 可选参数为 "train_test-oot" 或 "train-test-oot"
        :param is_split: 是否直接返回切分后的数据集
        :return:
        """
        df_ = df.copy()
        if split_class == "train_test-oot":
            df_[sample_use] = df_[date_info].map(
                lambda x: "train_test" if x < datetime.datetime(year, month, day) else "oot"
            )
            if is_split:
                return (
                    df_[df_[sample_use] == "train_test"].drop(sample_use, axis=1),
                    df_[df_[sample_use] == "oot"].drop(sample_use, axis=1)
                )
            return df_
        df_[sample_use] = df_[date_info].map(
            lambda x: "train" if x < datetime.datetime(year, month, day) else "oot"
        )
        df_test_index = df_[df_[sample_use] == "train"].sample(frac=0.25, random_state=RANDOM_SEED).index
        df_.loc[df_test_index, sample_use] = "test"
        if is_split:
            return (
                df_[df_[sample_use] == "train"].drop(sample_use, axis=1),
                df_[df_[sample_use] == "test"].drop(sample_use, axis=1),
                df_[df_[sample_use] == "oot"].drop(sample_use, axis=1)
            )
        return df_

    @staticmethod
    def retrieval_rate(df: pd.DataFrame, feature_list: List[str], return_str: bool = False) -> str | None:
        """
        计算数据查得率

        :param df: 数据样本
        :param feature_list: 特征列表
        :param return_str: 是否直接返回结果字符串
        :return:
        """
        df_ = df.dropna(subset=feature_list, how="all")
        detail_str = (
            f"原始数据共有 {df.shape[0]} 行 {df.shape[1]} 列\n"
            f"查得数据共有 {df_.shape[0]} 行 {df_.shape[1]} 列\n"
            f"查得率为 {df_.shape[0] / df.shape[0] * 100}%"
        )
        if return_str:
            return detail_str
        print(detail_str)
        return None

    @staticmethod
    def generate_id(df: pd.DataFrame, select_columns: List[str], id_name: str = "__GEN_ID__") -> pd.DataFrame:
        """
        根据指定列生成一个新的 id 列

        :param df: 数据 DataFrame
        :param select_columns: 按照这些列生成 id 列
        :param id_name: 指定生成 id 列的列名
        :return: DataFrame
        """
        code_list = list()
        for column in select_columns:
            df[column] = df[column].astype(str)
            code_list.append(f"df[\"{column}\"]")
        df[id_name] = eval(" + ".join(code_list))
        return df

    @staticmethod
    def generate_retrieval_missing(
            df: pd.DataFrame,
            feature_list: List[str],
            retrieval_missing_name: str = "__RETRIEVAL_MISSING__"
    ) -> pd.DataFrame:
        """
        查看指定列是否都有值, 并生成一个新的 missing 列

        :param df: 数据样本
        :param feature_list: 特征列表
        :param retrieval_missing_name: 指定生成 missing 列的列名
        :return:
        """
        df[retrieval_missing_name] = np.where(df[feature_list].isnull().all(axis=1), 1, 0)
        return df

    @staticmethod
    def data_detail(
            data_all: pd.DataFrame,
            data_train: pd.DataFrame,
            data_oot: pd.DataFrame,
            date_info: str = DATE_INFO,
            label: str = LABEL,
            return_str: bool = False
    ) -> None | str:
        """
        数据详情

        :param data_all: 全量样本
        :param data_train: 训练样本
        :param data_oot: OOT样本
        :param date_info: 时间列
        :param label: 标签列
        :param return_str:
        :return:
        """
        data_detail_list = [
            f"总数据量: {len(data_all)}",
            f"Train数据数量: {len(data_train)}",
            f"Train与总样本比例: {len(data_train) / len(data_all)}",
            f"Train坏人个数: {data_train[label].sum()}",
            f"Train坏人率: {data_train[label].mean()}",
            f"Train日期范围: {data_train[date_info].min()} - {data_train[date_info].max()}",
            f"OOT数据数量: {len(data_oot)}",
            f"OOT与总样本比例: {len(data_oot) / len(data_all)}",
            f"OOT坏人个数: {data_oot[label].sum()}",
            f"OOT坏人率: {data_oot[label].mean()}",
            f"OOT日期范围: {data_oot[date_info].min()} - {data_oot[date_info].max()}"
        ]
        data_detail_str = "\n".join(data_detail_list)
        if return_str:
            return data_detail_str
        print(data_detail_str)
        return None

    @staticmethod
    def data_detail_2(
            data_all: pd.DataFrame,
            data_train: pd.DataFrame,
            data_test: pd.DataFrame,
            data_oot: pd.DataFrame,
            date_info: str = DATE_INFO,
            label: str = LABEL,
            return_str: bool = False
    ) -> None | str:
        """
        数据详情

        :param data_all: 全量样本
        :param data_train: 训练样本
        :param data_test: 测试样本
        :param data_oot: OOT样本
        :param date_info: 时间列
        :param label: 标签列
        :param return_str:
        :return:
        """
        data_detail_list = [
            f"总数据量: {len(data_all)}",
            f"Train数据数量: {len(data_train)}",
            f"Train与总样本比例: {len(data_train) / len(data_all)}",
            f"Train坏人个数: {data_train[label].sum()}",
            f"Train坏人率: {data_train[label].mean()}",
            f"Train日期范围: {data_train[date_info].min()} - {data_train[date_info].max()}",
            f"Test数据数量: {len(data_test)}",
            f"Test与总样本比例: {len(data_test) / len(data_all)}",
            f"Test坏人个数: {data_test[label].sum()}",
            f"Test坏人率: {data_test[label].mean()}",
            f"Test日期范围: {data_test[date_info].min()} - {data_test[date_info].max()}",
            f"OOT数据数量: {len(data_oot)}",
            f"OOT与总样本比例: {len(data_oot) / len(data_all)}",
            f"OOT坏人个数: {data_oot[label].sum()}",
            f"OOT坏人率: {data_oot[label].mean()}",
            f"OOT日期范围: {data_oot[date_info].min()} - {data_oot[date_info].max()}"
        ]
        data_detail_str = "\n".join(data_detail_list)
        if return_str:
            return data_detail_str
        print(data_detail_str)
        return None

    @staticmethod
    def month_info(df: pd.DataFrame, date_month: str = DATE_MONTH, label: str = LABEL) -> pd.DataFrame:
        """
        数据按月份详情

        :param df: 数据 DataFrame
        :param date_month: 月份列
        :param label: 标签列
        :return:
        """
        objs = [
            df[[date_month, label]].groupby([date_month]).count(),
            df[[date_month, label]].groupby([date_month]).sum()[label],
            df[[date_month, label]].groupby([date_month]).mean()[label]
        ]
        month_detail = pd.concat(objs, axis=1)
        month_detail.columns = ["Total", "Bad", "Bad_Per"]
        month_detail["Good"] = month_detail["Total"] - month_detail["Bad"]
        month_detail = month_detail[["Total", "Good", "Bad", "Bad_Per"]]
        month_detail = month_detail.reset_index()
        month_detail.columns = ["Month", "Total", "Good", "Bad", "Bad_Per"]
        month_detail["Total"] = month_detail["Total"].map(int)
        month_detail["Good"] = month_detail["Good"].map(int)
        month_detail["Bad"] = month_detail["Bad"].map(int)
        month_detail["Bad_Per"] = month_detail["Bad_Per"].map(lambda x: f"{x:.6f}")
        return month_detail

    @classmethod
    def object_to_number(
            cls,
            df: pd.DataFrame,
            select_columns: List[str] | None = None
    ) -> Dict[str, int | dict]:
        """
        将非数值型特征进行数字编码

        :param df: 数据 DataFrame
        :param select_columns: 需要进行数字编码的特征列表
        :return: 返回数字编码后的数据、已经进行数字编码特征的数量和编码字典

        Examples 1:
        data = pd.DataFrame({"col_1": [1, 2, 3], "col_2": ["a", "b", "b"]})
        data
            col_1  col_2
        0      1      a
        1      2      b
        2      3      b
        label_encoding_dict = Feature.object_to_number(data, ["col_2"])
        data
            col_1  col_2
        0      1      1
        1      2      2
        2      3      2
        label_encoding_dict
        {"size": 1, "value": {"col_2": {"a": 1, "b": 2}}}

        Examples 2:
        data = pd.DataFrame({"col_1": [1, 2, 3], "col_2": ["a", "b", "b"]})
        data
            col_1  col_2
        0      1      a
        1      2      b
        2      3      b
        label_encoding_dict = Feature.object_to_number(data)
        data
            col_1  col_2
        0      1      1
        1      2      2
        2      3      2
        label_encoding_dict
        {"size": 1, "value": {"col_2": {"a": 1, "b": 2}}}
        """
        save_dict = {"size": 0, "value": dict()}
        if not select_columns:
            select_columns = df.columns.tolist()
        for _column in select_columns:
            if df[_column].dtype != object:
                continue
            str_to_num_dict = {value: num for num, value in enumerate(np.unique(df[_column].astype("str")), start=1)}
            df[_column] = df[_column].map(lambda x: str_to_num_dict[x])
            save_dict["size"] += 1
            save_dict["value"][_column] = str_to_num_dict
        return save_dict

    @classmethod
    def fill_nan(
            cls,
            df: pd.DataFrame,
            select_columns: List[str],
            stratepy: str | int | float = -999
    ) -> Dict[str, str] | Dict[str, dict]:
        """
        缺失值填补

        :param df: 数据 DataFrame
        :param select_columns: 需要进行缺失值填补的特征
        :param stratepy: 填补策略, 当类型为 str 时: 1. mode: 众数; 2. median: 中位数; 3. mean: 平均数
        :return: {"type": "number; str; error", "value": {"feature_1": "missing_value_1", ...}}
        """

        if isinstance(stratepy, int) or isinstance(stratepy, float):
            df[select_columns] = df[select_columns].fillna(stratepy)
            return {"type": "number", "value": {_column: stratepy for _column in select_columns}}

        if stratepy not in ["mode", "median", "mean"]:
            return {"type": "error", "value": "stratepy is one of mode, median or mean"}

        save_dict = {"type": "str", "value": dict()}
        value_list = "df[_column].mode().tolist()" if stratepy == "mode" else f"[df[_column].{stratepy}()]"
        for _column in select_columns:
            random.seed(1)
            missing_value = random.choice(eval(value_list))
            save_dict["value"][_column] = missing_value
            df[_column] = df[_column].fillna(missing_value)
        return save_dict

    @classmethod
    def get_bins_list(cls, df: pd.DataFrame, column: str) -> list:
        """
        获取特征分箱列表

        :param df: 数据 DataFrame
        :param column: 需要进行分箱的特征
        :return: 返回分箱列表
        """
        quant_point = list(range(10, 100, 10))
        all_score = np.array(df[column], dtype=np.float64)
        arr_non_miss = all_score[~(np.isnan(all_score) | np.isin(all_score, [np.nan]))]
        if len(arr_non_miss) == 0:
            return [np.nan] * len(quant_point)
        bins_list = np.percentile(arr_non_miss, quant_point, method="lower")
        return sorted(list(set(bins_list)))

    @classmethod
    def assign_bin(
            cls,
            x: float,
            bins_list: List[float],
            missing_value: int | float | None = None
    ) -> str:
        """
        特征分箱后进行编码

        :param x:
        :param bins_list:
        :param missing_value:
        :return:
        """

        if (x == missing_value) or (str(x) == "nan"):
            return "缺失"

        bins_list = sorted(bins_list)
        bins_min, bins_max = bins_list[0], bins_list[-1]

        len_list = str(bins_max).split(".")
        len_begin, len_end = len(len_list[0]), len(len_list[-1])

        low = "-" * (len_begin + len_end + 1)
        top = "*" * (len_begin + len_end + 1)
        width = len_begin + len_end + 1

        if x < bins_min:
            return f"[{low} - {bins_min:0{width}.{len_end}f})"

        if x >= bins_max:
            return f"[{bins_max:0{width}.{len_end}f} - {top}]"

        for index in range(len(bins_list) - 1):
            left, right = bins_list[index], bins_list[index + 1]
            if left <= x < right:
                return f"[{left:0{width}.{len_end}f} - {right:0{width}.{len_end}f})"

        return f"Error: 当前 x 为 {x}, 此值可能为缺失值, 请设置 missing_value 参数"

    @classmethod
    def calculate_iv(
            cls,
            df: pd.DataFrame,
            column: str,
            column_bin: str,
            label: str,
            return_df: bool = False
    ) -> float | Tuple[float, pd.DataFrame]:
        """
        计算IV, 请先进行分箱

        :param df:
        :param column:
        :param column_bin:
        :param label:
        :param return_df:
        :return:
        """

        sum_counts = df.groupby(column_bin)[label].agg(["count", "sum"])
        sum_counts["good"] = sum_counts["count"] - sum_counts["sum"]

        sum_good = sum_counts["good"].sum()
        sum_bad = sum_counts["sum"].sum()

        result_list = list()
        for _bin, _row in sum_counts.iterrows():
            _bad = _row["good"] / sum_good if sum_good != 0 else 0
            _good = _row["sum"] / sum_bad if sum_bad != 0 else 0
            _iv = _bad + _good if (_bad == 0) or (_good == 0) else (_bad - _good) * np.log(_bad / _good)
            result_list.append({
                "特征": column,
                "区间": _bin,
                "组内总人数": _row["count"],
                "组内坏客户数": _row["sum"],
                "组内好客户数": _row["good"],
                "组内坏客户率": np.around(_row["sum"] / _row["count"] if _row["count"] != 0 else 0, 16),
                "iv": np.around(_iv, 16)
            })

        iv_df = pd.DataFrame(result_list)
        iv = iv_df["iv"].sum()

        if return_df:
            return iv, iv_df

        return iv

    @classmethod
    def calculate_psi(
            cls,
            df: pd.DataFrame,
            column: str,
            column_bin: str,
            by: str,
            return_df: bool = False
    ) -> float | Tuple[float, pd.DataFrame]:
        """
        计算PSI, 请先进行分箱

        :param df:
        :param column:
        :param column_bin:
        :param by:
        :param return_df:
        :return:
        """

        psi_temp_df = pd.DataFrame()
        for _month, _df in df.groupby(by, sort=True):
            _bin_df = _df[column_bin].value_counts(normalize=True).sort_index().to_frame(name="percent")
            psi_temp_df[_month] = _bin_df["percent"]
        psi_temp_df = psi_temp_df.reset_index(col_level=0)

        by_list = sorted(list(set(df[by].tolist())))
        psi_df = pd.DataFrame({"column": [column] * psi_temp_df.shape[0]})
        psi_df["bin"] = psi_temp_df[psi_temp_df.columns.tolist()[0]]
        psi_df[by_list[0]] = 0
        base = psi_temp_df[by_list[0]]
        for _actual in by_list[1:]:
            actual = psi_temp_df[_actual]
            psi_df[_actual] = (actual - base) * np.log(actual / base)
        psi_list = [psi_df[_].sum() for _ in by_list]
        psi_df.loc[-1] = [column, "PSI"] + psi_list
        for _column in by_list:
            psi_df[_column] = np.around(psi_df[_column], 16)
            psi_df[_column] = psi_df[_column].map(lambda x: f"{x:.6f}")
        psi_df = psi_df.reset_index(drop=True)
        psi = max(psi_list)

        if return_df:
            return psi, psi_df

        return psi

    @classmethod
    def calculate_psi_by_column(
            cls,
            df: pd.DataFrame,
            column: str,
            return_value: bool = True,
            return_df: bool = True
    ):
        bins_list = cls.get_bins_list(df, column)
        df[f"{column}_bin__"] = df[column].apply(lambda x: cls.assign_bin(x, bins_list))
        psi, psi_df = cls.calculate_psi(df, column, f"{column}_bin__", by=DATE_MONTH, return_df=True)
        if return_value and return_df:
            return psi, psi_df
        if return_value:
            return psi
        if return_df:
            return psi_df

    @classmethod
    def calculate_iv_psi_by(
            cls,
            df: pd.DataFrame,
            columns: List[str],
            label: str,
            by: str,
            missing_value: int | float | None = None,
            return_detail_df: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | Tuple[pd.DataFrame, pd.DataFrame]:
        """
        计算特征 IV 和 PSI

        :param df: 数据 pd.DataFrame
        :param columns: 数据特征列表
        :param label: 标签列名称
        :param by: 通过该参数进行计算 PSI
        :param missing_value: 数据填补的缺失值
        :param return_detail_df: 是否需要返回 IV 详细 pd.DataFrame 和 PSI 详细 pd.DataFrame
        :return: IV 和 PSI
        """

        iv_list, iv_df_list, psi_list, psi_df_list = list(), list(), list(), list()

        for _column in tqdm(columns):

            if df[_column].dtype != float:
                df[_column] = df[_column].astype(float)

            # 特征分箱
            _bins_list = sorted(df[_column].unique().tolist())
            if len(_bins_list) > 10:
                _bins_list = cls.get_bins_list(df, _column)

            # 编码
            _column_bin = "__bin__" + _column
            df[_column_bin] = df[_column].map(lambda x: cls.assign_bin(x, _bins_list, missing_value))

            # 编码后传入iv值计算函数
            _iv, _iv_df = cls.calculate_iv(df, _column, _column_bin, label, return_df=True)
            iv_list.append({"column": _column, "iv": _iv})
            iv_df_list.append(_iv_df)

            # 编码后传入psi计算函数
            _psi, _psi_df = cls.calculate_psi(df, _column, _column_bin, by, return_df=True)
            psi_list.append({"column": _column, "psi": _psi})
            psi_df_list.append(_psi_df)

            df.drop(_column_bin, axis=1, inplace=True)

        iv_df = pd.DataFrame(iv_list).sort_values(by="iv", ascending=False).reset_index(drop=True)
        iv_detail_df = pd.concat(iv_df_list)
        psi_df = pd.DataFrame(psi_list).sort_values(by="psi", ascending=True).reset_index(drop=True)
        psi_detail_df = pd.concat(psi_df_list)

        if return_detail_df:
            return iv_df, iv_detail_df, psi_df, psi_detail_df

        return iv_df, psi_df

    @classmethod
    def corr_filter(
            cls,
            df: pd.DataFrame,
            importance_df_: pd.DataFrame,
            column_list: List[str],
            limit: float = 0.80
    ) -> List[str]:
        """
        通过特征重要性和相关性筛选特征, 获取需要删除的特征列表

        :param df: 数据 DataFrame
        :param importance_df_: 特征重要性 DataFrame
        :param column_list: 需要进行筛选的特征列表
        :param limit: 相关性阈值, 默认为 0.80
        :return: 返回需要删除的特征列表
        """
        corr = abs(df.corr())
        remove_list = list()
        for _column in column_list:
            for __column in column_list:
                if (
                        (_column != __column) and
                        (corr.loc[_column, __column] >= limit) and
                        (importance_df_.loc[_column]["importance"] > importance_df_.loc[__column]["importance"]) and
                        (__column not in remove_list)
                ):
                    remove_list.append(__column)
        return remove_list

    @staticmethod
    def proba_distribution(df: pd.DataFrame, bins: int = 100) -> None:
        """
        数据分布图

        :param df: 数据
        :param bins: 分箱数量
        :return: None
        """
        df.plot(kind="hist", bins=bins, edgecolor="black")
        plt.grid(True)
        plt.show()
        return None

    @staticmethod
    def score_distribution(df, label, score_column_name, begin=0.0, end=1.0, bins=20) -> pd.DataFrame:
        """分数分布"""

        begin = np.float64(begin)
        end = np.float64(end)
        bins = np.float64(bins)

        sep = (end - begin) / bins
        sep_str = f"{sep:.16f}".rstrip("0")
        decimals = np.float64(0)
        if "." in sep_str:
            decimals = len(sep_str.split(".")[1])

        df_ = df.sort_values(by=score_column_name).reset_index(drop=True)
        colum_bin = f"{score_column_name}_bin"

        bins_ = np.arange(begin, end + sep, sep)
        labels = [
            f"{np.round(bins_[_], decimals):.{decimals}f} - {np.round(bins_[_ + 1], decimals):.{decimals}f}"
            for _ in range(len(bins_) - 1)
        ]
        df_[colum_bin] = pd.cut(df_[score_column_name], bins_, False, labels, include_lowest=True)

        df_distribution = df_.groupby(colum_bin).agg(
            Bin=(colum_bin, "first"),
            Total=(score_column_name, "size"),
            Bad=(label, lambda x: (x == 1).sum()),
        ).reset_index(drop=True)
        df_distribution["Bad_Per"] = df_distribution["Bad"] / df_distribution["Total"]

        df_distribution["Total"] = df_distribution["Total"].map(lambda x: f"{x:.0f}")
        df_distribution["Bad"] = df_distribution["Bad"].map(lambda x: f"{x:.0f}")
        df_distribution["Bad_Per"] = df_distribution["Bad_Per"].map(lambda x: f"{x:.6f}")

        return df_distribution

    @staticmethod
    def proba_score_detail(df, proba_column, score_column) -> pd.DataFrame:
        """概率和分值详细"""
        r_1_1 = f"{np.round(df[proba_column].min(), 6)} - {np.round(df[proba_column].max(), 6)}"
        r_1_2 = f"{df[score_column].min()} - {df[score_column].max()}"
        r_2_1 = np.round(df[proba_column].max() - df[proba_column].min(), 6)
        r_2_2 = df[score_column].max() - df[score_column].min()
        r_3_1 = ", ".join([str(np.round(_, 6)) for _ in df[proba_column].mode().to_list()])
        r_3_2 = ", ".join(list(map(str, df[score_column].mode().to_list())))
        r_4_1 = np.round(df[proba_column].median(), 6)
        r_4_2 = int(df[score_column].median())
        r_5_1 = np.round(df[proba_column].mean(), 6)
        r_5_2 = np.round(df[score_column].mean(), 6)
        data_list = [[r_1_1, r_1_2], [r_2_1, r_2_2], [r_3_1, r_3_2], [r_4_1, r_4_2], [r_5_1, r_5_2]]
        return pd.DataFrame(data_list, columns=["概率", "分值"], index=["范围", "极差", "众数", "中位数", "平均值"])
