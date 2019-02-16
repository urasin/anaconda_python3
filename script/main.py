import pandas as pd

## pandas
#- DataFrame型: 2次元テーブル
#- Serias: 一次元構造

train = pd.read_csv('train.csv')
print(train.head()) # pandasではhead(), tail(), columns()カラム取得, index() 行の数 が使えます!
print('平均値(mean)', train['Age'].mean())  # 平均値(mean), NaN欠損値を除外して計算してくれる
print('中央値(median)', train['Age'].median())  # 中央値(median), データーを昇順に並べたときに中央の値、個数が偶数の場合足して2で割った数

# 平均値は一番大きい値と小さい値にバイアスがかかりやすい => あまりデーター分析では平均値は使われない => 中央値が使われやすい
# 平均値がバイアスが大きいか、小さいかの判断 => 中央値との差がなければ使われる
smoking_df = pd.DataFrame(
  {"member": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
  "is_smoking": [1, 1, 0, 0, 0, 0, 1, 0, 0, 1]}, # 喫煙者フラグ
  columns=["member", "is_smoking"]
)

print(smoking_df)

print(smoking_df["is_smoking"].sum())  # 喫煙者の数
print(smoking_df["is_smoking"].sum()/len(smoking_df)) # 喫煙者の数の全体に占める割合

# 分散と標準偏差 => どれだけデーターが散らばっているか？
# 分散 => 平均から引いて二乗する => 負の数を出さないために二乗する => あまり使われない
# 標準偏差 => 分散をルートに入れる => よく使われる
# sklearnのdatasetを使う
# あやめ https://horti.jp/5294
from sklearn.datasets import load_iris
iris = load_iris()
iris_df = pd.DataFrame(iris["data"], columns=iris["feature_names"])

# print(iris_df.columns)
## がく片長(Sepal Length)  
print('分散(var): ',iris_df['sepal length (cm)'].var())
print('標準偏差(std)', iris_df['sepal length (cm)'].std())

## 花びら長(Petal Length)
print('分散(var): ',iris_df['petal length (cm)'].var())
print('標準偏差(std)', iris_df['petal length (cm)'].std())

## pandasの使い方
print(train.describe())  # 基礎データー量を全部計算してくれる

## 25%(中央値を基準に計算)箱ひげ図

## 文法的に気持ち悪い(indexで範囲指定)
train.index = range(100, 991)
print(train.index)

## filter 文字列比較 (==), かつ&
print(train["Age"] > 50)
print(train[train["Age"] > 50].head(10))

## ソート
print(train.sort_values(by=["Pclass", "Age"]).head(20)) # 昇順
print(train.sort_values(by=["Pclass", "Age"],ascending=False).head(20)) # 降順(ascending=False)

## pandasでのデーターの結合 concat
smoking1_df = pd.DataFrame(
    {"member": ["A", "B", "C", "D", "E"],
     "is_smoking": [1, 1, 0, 0, 0]},
    columns=["member", "is_smoking"]
)

smoking2_df = pd.DataFrame(
    {"member": ["F", "G", "H", "I", "J"],
     "is_smoking": [0, 1, 0, 0, 1]}, 
   columns=["member", "is_smoking"]
)

print(pd.concat([smoking1_df, smoking2_df]).reset_index(drop=True))  # indexを貼り直す

smoking3_df = pd.DataFrame(
    {"gender": ["male", "female", "male", "male", "female"], 
    "age": [24, 55, 30, 42, 28]},
    columns=["gender", "age"]
)

# 横に結合する, indexの数字が合わないとエラー失敗(concat, axis=1)
print(pd.concat([smoking1_df, smoking3_df], axis=1))

# pandasでinner join系もできる

# データー集約処理
print(train.groupby("Pclass").size())  # max, minも使える
print(train.groupby("Pclass").count())

## NaN(欠損値 => データーが取れないよ)
print(train['Age'].fillna(0)) # 0を入れる、中央値の28を採用する場合もある
## NaNを消す
print(train['Age'].dropna())

## csvに変換(いろいろoptions有り)
train.to_csv(
  "converted_train.csv"
)