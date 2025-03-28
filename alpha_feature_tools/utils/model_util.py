import datetime
import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc


class Model(object):

    @staticmethod
    def fusion_mean(df, column_list, proba_list):
        """模型加权平均融合"""
        _1 = " + ".join([f"df[\"{name}\"] * {proba}" for name, proba in zip(column_list, proba_list)])
        _2 = " + ".join([str(_) for _ in proba_list])
        return eval(_1) / eval(_2)

    @classmethod
    def evaluate_ks(cls, y_true, y_score) -> float:
        """计算KS"""
        return ks_2samp(y_score[y_true == 1], y_score[y_true != 1]).statistic

    @classmethod
    def evaluate_lift(cls, y_true, y_pred_prob, is_qcut=True, ascending=False) -> pd.DataFrame:
        """
        计算 Lift 提升度

        ascending=False 越大越坏
        """

        df = pd.DataFrame({"actual": y_true, "predicted_proba": y_pred_prob})
        df = df.sort_values(by="predicted_proba", ascending=ascending).reset_index(drop=True)

        df["__bin__"] = pd.cut(df["predicted_proba"], bins=10, labels=range(10, 0, -1))  # 等距
        if is_qcut:
            df["__bin__"] = pd.qcut(df["predicted_proba"], q=10, labels=range(10, 0, -1))  # 等频

        df_lift = df.groupby("__bin__").agg(
            Bin=("predicted_proba", lambda x: f"{x.min():.16f} - {x.max():.16f}"),
            Total=pd.NamedAgg(column="actual", aggfunc="size"),
            Bad=pd.NamedAgg(column="actual", aggfunc="sum")
        )
        df_lift["__sort__"] = df_lift.index
        df_lift = df_lift.sort_values(by="__sort__", ascending=ascending).reset_index(drop=True)
        df_lift = df_lift.drop("__sort__", axis=1)

        # df_lift["Bad_Model(%)"] = df_lift["Bad"] / df_lift["Bad"].sum() * 100
        # df_lift["Bad_Random(%)"] = df_lift["Total"] / df_lift["Total"].sum() * 100
        # df_lift["Cumulative_Bad_Model(%)"] = df_lift["Bad_Model(%)"].cumsum()
        # df_lift["Cumulative_Bad_Random(%)"] = df_lift["Bad_Random(%)"].cumsum()
        # df_lift["Lift"] = df_lift["Cumulative_Bad_Model(%)"] / df_lift["Cumulative_Bad_Random(%)"]

        # df_lift["Total"] = df_lift["Total"].map(lambda x: f"{x:.0f}")
        # df_lift["Bad"] = df_lift["Bad"].map(lambda x: f"{x:.0f}")
        # df_lift["Bad_Model(%)"] = df_lift["Bad_Model(%)"].map(lambda x: f"{x:.6f}")
        # df_lift["Bad_Random(%)"] = df_lift["Bad_Random(%)"].map(lambda x: f"{x:.6f}")
        # df_lift["Cumulative_Bad_Model(%)"] = df_lift["Cumulative_Bad_Model(%)"].map(lambda x: f"{x:.6f}")
        # df_lift["Cumulative_Bad_Random(%)"] = df_lift["Cumulative_Bad_Random(%)"].map(lambda x: f"{x:.6f}")
        # df_lift["Lift"] = df_lift["Lift"].map(lambda x: f"{x:.6f}")

        df_lift["Bad_Bin(%)"] = df_lift["Bad"] / df_lift["Total"] * 100
        df_lift["Bad_Total(%)"] = df_lift["Bad"].sum() / df_lift["Total"].sum() * 100
        df_lift["Lift"] = df_lift["Bad_Bin(%)"] / df_lift["Bad_Total(%)"]

        df_lift["Total"] = df_lift["Total"].map(lambda x: f"{x:.0f}")
        df_lift["Bad"] = df_lift["Bad"].map(lambda x: f"{x:.0f}")
        df_lift["Bad_Bin(%)"] = df_lift["Bad_Bin(%)"].map(lambda x: f"{x:.6f}")
        df_lift["Bad_Total(%)"] = df_lift["Bad_Total(%)"].map(lambda x: f"{x:.6f}")
        df_lift["Lift"] = df_lift["Lift"].map(lambda x: f"{x:.6f}")

        return df_lift

    @classmethod
    def evaluate(
            cls,
            y_true,
            y_pred_prob,
            name: str | None = None,
            ascending: bool = False,
            is_print: bool = True,
            return_type: int | None = None
    ) -> list | dict | None:
        """
        模型评价

        ascending=False 越大越坏
        """

        df = pd.DataFrame({"y_pred_prob": y_pred_prob, "label": y_true})
        df["y_pred"] = 0
        df["y_pred"][df["y_pred_prob"] > df["label"].mean()] = 1
        y_pred = df["y_pred"]

        accuracy = accuracy_score(y_true, y_pred)
        auc = 1 - roc_auc_score(y_true, y_pred_prob) if ascending else roc_auc_score(y_true, y_pred_prob)
        ks = cls.evaluate_ks(y_true, y_pred_prob)

        if is_print:
            model_name = f"{name}_" if name else ""
            print(f"{model_name}Accuracy: {accuracy}\n{model_name}AUC: {auc}\n{model_name}KS: {ks}")

        if return_type == 1:
            return [accuracy, auc, ks]

        if return_type == 2:
            return {"accuracy": accuracy, "auc": auc, "ks": ks}

        return None

    @staticmethod
    def evaluate_to_csv(
            save_file: str,
            model_name: str,
            train_evaluate: Any,
            test_evaluate: Any,
            oot_evaluate: Any,
            best_params: Dict[str, Any],
            feature_version: str = "base",
            dataset_name: str = "",
            model_version: str = ""
    ) -> None:
        """模型评估结果及模型参数保存"""
        write_dict = {
            "KS_Train": "{:.6f}".format(train_evaluate["ks"]),
            "KS_Test": "{:.6f}".format(test_evaluate["ks"]),
            "KS_OOT": "{:.6f}".format(oot_evaluate["ks"]),
            # "KS_TT": "{:.6f}".format(train_evaluate["ks"] - test_evaluate["ks"]),
            # "KS_TO": "{:.6f}".format(test_evaluate["ks"] - oot_evaluate["ks"]),
            "AUC_Train": "{:.6f}".format(train_evaluate["auc"]),
            "AUC_Test": "{:.6f}".format(test_evaluate["auc"]),
            "AUC_OOT": "{:.6f}".format(oot_evaluate["auc"]),
            # "AUC_TT": "{:.6f}".format(train_evaluate["auc"] - test_evaluate["auc"]),
            # "AUC_TO": "{:.6f}".format(test_evaluate["auc"] - oot_evaluate["auc"]),
            "Dataset_Name": dataset_name,
            "Model_Version": model_version,
            "Feature_Version": feature_version,
            "Model_Type": model_name,
            "Model_Params": "\"{}\"".format(best_params),
            "Time": datetime.datetime.now().isoformat()
        }
        if not os.path.exists(save_file):
            with open(save_file, "a+") as f:
                f.write(",".join(list(write_dict.keys())))
                f.write("\n")
                f.close()
        with open(save_file, "a+") as f:
            f.write(",".join(list(write_dict.values())))
            f.write("\n")
            f.close()
        return None

    @staticmethod
    def prob_to_score(prob: float, a: int | float, b: int | float, begin=300, end=1000):
        """模型概率转换分值"""
        score = int(a - b * np.log(prob / (1 - prob)) / np.log(2))
        if score < begin:
            return begin
        if score > end:
            return end
        return score

    @staticmethod
    def plot_ks(y_true, y_pred_proba):
        fpr, tpr, threshold = roc_curve(y_true, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(threshold[1:], tpr[1:])
        plt.plot(threshold[1:], fpr[1:])
        plt.plot(threshold[1:], tpr[1:] - fpr[1:])
        plt.legend(["tpr", "fpr", "tpr - fpr"])
        plt.title("KS Curve")
        plt.gca().invert_xaxis()
        plt.show()
        return None

    @staticmethod
    def plot_roc(y_true, y_pred_proba):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.show()
        return None

    @classmethod
    def plot_lift(cls, y_true, y_pred_proba, ascending: bool = False):
        """

        :param y_true:
        :param y_pred_proba:
        :param ascending: False 越大越坏
        :return:
        """
        y_values = cls.evaluate_lift(y_true, y_pred_proba, is_qcut=True, ascending=ascending)["Lift"]
        x_values = range(len(y_values))
        plt.figure(figsize=(8, 6))
        plt.plot(x_values, y_values, marker="o", linestyle="-", color="b")
        plt.title("Lift Curve")
        plt.xticks(x_values)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return None
