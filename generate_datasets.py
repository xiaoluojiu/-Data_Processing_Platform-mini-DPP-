"""
生成常用测试数据集的 CSV 文件，保存在项目根目录下的 data/ 目录。
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    fetch_california_housing,
    make_blobs,
)


def main() -> None:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # 1. Iris 鸢尾花
    iris = load_iris(as_frame=True)
    iris.frame.to_csv(data_dir / "iris.csv", index=False)

    # 2. Wine 葡萄酒
    wine = load_wine(as_frame=True)
    wine.frame.to_csv(data_dir / "wine.csv", index=False)

    # 3. Breast Cancer 乳腺癌
    bc = load_breast_cancer(as_frame=True)
    bc.frame.to_csv(data_dir / "breast_cancer.csv", index=False)

    # 4. California Housing 加州房价
    cal = fetch_california_housing(as_frame=True)
    cal.frame.to_csv(data_dir / "california_housing.csv", index=False)

    # 5. 聚类用的合成数据（blobs）
    X, y = make_blobs(
        n_samples=500,
        centers=4,
        cluster_std=1.2,
        random_state=42,
    )
    blobs_df = pd.DataFrame(X, columns=["feature1", "feature2"])
    blobs_df["label"] = y
    blobs_df.to_csv(data_dir / "blobs_clustering.csv", index=False)

    print("已生成以下数据集（CSV 文件）保存在:", data_dir.resolve())
    for p in sorted(data_dir.glob("*.csv")):
        print(" -", p.name)


if __name__ == "__main__":
    main()


