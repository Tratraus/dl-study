# 运行一次即可，生成 iris.csv
# 你可以单独运行这段，也可以放在 day1 脚本最开头

import os
import pandas as pd
from sklearn.datasets import load_iris

os.makedirs("data", exist_ok=True)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
df["species"] = iris.target

df.to_csv("data/iris.csv", index=False)

print("iris.csv saved.")
print(df.head())
print("Shape:", df.shape)
