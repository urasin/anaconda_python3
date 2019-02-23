import pandas as pd

# 第二回目 2019/02/23
## pandas
#- DataFrame型: 2次元テーブル
#- Serias: 一次元構造

# データーの可視化
# 1変数だけ見たいとき => ヒストグラムを見る => matplotlibが必要

import matplotlib.pyplot as plt
train = pd.read_csv('train.csv')
# import matplotlib.pyplot as plt
# %matplotlib inline # jupyterの場合しよう

age_sex = train[['Age', 'Sex']]
age_sex = age_sex.dropna()
age_sex.hist(by="Sex", sharex=True, sharey=True)

# 2変数 => 棒グラフを使う
# 棒グラフ 種類と終了の関係性を見たい場合つかう
# jupyterでshift + tabを使うとライブラリのドキュメントがでる

train.groupby('Pclass')["Age"].mean().plot(kind="bar")

age_fare_sex = train[['Age', 'Fare', 'Sex']]
age_fare_sex.groupby("Sex").mean().plot(kind="bar")
# 画像出力
plt.savefig('age_fare_sex.png')

# 散布図 (直線的な関係を見るときに使う)
train.plot(kind='scatter', x="Age", y="Fare")
plt.savefig('sanpuzu_age_fare_sex.png') # 直線ではないので相関なさそう

# 散布図をの作成(2) 性別別に書く
female = train[train["Sex"] == "female"]
male = train[train["Sex"] == "male"]

# axを次のplot時に渡せば同じグラフに重ねて掛ける
ax = female.plot(kind="scatter", x="Age", y="Fare", color="red")
male.plot(kind="scatter", x="Age", y="Fare", color="blue", ax=ax)
plt.savefig('sanpuzu_separelate_gendar_age_fare_sex.png')  # 性別別に見ても相関なし

# 折れ線グラフ => 時間と数量の関係を見たいときに使う
import numpy as np

# 擬似的な株価データー(=乱打m樹を～区を生成してみる)
ts_a = 1.0 * pd.Series(np.random.randn(1000), index=pd.date_range('2010-01-01', periods=1000))

# 初期値を100として、ある日までの変化分をcumsum()メソッドで足し上げる(累積和)
stock_a_price = 100 + ts_a.cumsum()

# プロットしてみる
stock_a_price.plot(kind="line")

plt.savefig('stock_price.png')  # 性別別に見ても相関なし


# 折れ線グラフ2 
# 変化量と初期値が違うものを幾つか生成してみる
# 何故か動かないのでコメントアウト
# ts_b = 2 * pd.Series(np.random.randn(1000), index=pd.date_range('2010-01-01', periods=1000))
# ts_c = 0.5 * pd.Series(np.random.randn(1000), index=pd.date_range('2010-01-01', periods=1000))
# ts_d = 0.1 * pd.Series(np.random.randn(1000), index=pd.date_range('2010-01-01', periods=1000))
# stock_b_price = 120 + ts_b.cumsum()
# stock_c_price = 85 + ts_c.cumsum()
# stock_d_price = 70 + ts_d.cumsum()
# # DataFrame にする
# stock_df = pd.DataFrame(
#     {"a": stock_a_price, "b": stock_b_price, "c": stock_c_price, "d": stock_d_price}, 
#    columns=["a", "b", "c", "d"]
# )
# # プロット
# stock_df.plot(kind="line")

# plt.savefig('oresen2.png')

# データーの相関と回帰
## 相関分析 2つの変数間の直線的な関係を分析する
## 強い負の相関, 無相関, 強い正の相関がある
## 共分散とは => 計算式がある => 正の値、負の値で分析する => ばらつきの大きさ、正の相関か負の相関がわかる?
## 相関係数 => 計算式がある => 共分散を使う => 値が出る => 相関係数のほうが共分散より使われる
## 相関係数 => -1 < x < 1 => 0.7より大きければ、一概には言えないが強い相関 相関関係がわかる

# print(age_fare_sex.corr())  # 0.09のため相関がないことがわかる

# 自己相関係数の算出
## 時系列データーのとき自己相関をつかう
## 周期性の分析、有る無しに使う
## 時系列分野 => 株価, IOTの異常値検査などがおおい

# [1, 1, 1, 1, 1, 1, 5] を 50 個分結合した Series (350 要素) 作成
s = pd.Series([1, 1, 1, 1, 1, 1, 5] * 50, index=pd.date_range('2010-01-01', periods=350))

# 作成した Series に乱数追加
s += pd.Series(np.random.randn(350), index=pd.date_range('2010-01-01', periods=350))

# プロット
s.plot(kind="line")
# 自己相関係数の算出
autocorr_series = pd.Series([s.autocorr(lag=i) for i in range(25)])

# プロット
# print(autocorr_series.plot("bar"))
plt.savefig('self_soukan.png')

## 疑似相関に注意
## 身長と算数の関係 => 身長が高いほど、点数が高い => 年齢が高ければ年齢がたかいので相関なし
## 疑似相関見分け方 => 文脈から見分ける？ => 気をつけるしか無い。 => データーのみのときは => ランダムフォレストで重要度が高そうなデーターを抽出する
## 疑似相関を集めたサイト => https://twitter.com/podoron/status/889475502209064960

## どの変数を使うか => 様々な図や値を出して人間が判断する + ランダムフォレストで重要度が高いものを抽出してつかう。

# 回帰分析
## 線形回帰
## 最小二乗法 => シンプルな回帰

from sklearn.linear_model import LinearRegression
x = train[["Age"]].fillna(28)# DataFrame として取り出す
y = train["Fare"]  # Series として取り出す

# 必要なライブラリ読み込み
from sklearn.linear_model import LinearRegression

# 線形回帰モデルを構築
model = LinearRegression()
model.fit(x, y)
# print("a:", model.coef_, "b:", model.intercept_) # 係数と切片
# print("R-squared:", model.score(x, y))  # 決定係数 0-1 1に近づくほど当てはまりが良い 
# 結果の0.009めちゃ低いので相関なさそう、一概にはいえないが0.7ぐらいだと信用できそう
# 数学値 Y ハット(Y + ^)が予測値

# 線形回帰の 直線を引く
linear_df = pd.DataFrame(
    {"X": x["Age"],"Y": y, "predict_Y": model.predict(x)},
    columns=["X", "Y", "predict_Y"]
)

# ax を次の plot 時に渡せば同じグラフに重ねて描ける
ax = linear_df.plot(kind="scatter", x="X", y="Y", color="black")
linear_df.plot(kind="line", x="X", y="predict_Y", color="blue", xlim=(0.5, 7.0), ax=ax)
plt.savefig('linear-regression.png')

# 重回帰 => 線形回帰の説明変数が2つ以上になったもの。 => 性別とかのカテゴリーデーターは微妙 => 量的データーが良い
# 説明変数同士は独立(無相関)を仮定する
## 広告で使われることが多い => a1, b1 => 偏回帰係数
## 主成分分析 => 4次元を2次元にする  => 情報の希釈化
## 影響度の高い低いを測定するのにも使われる => マゼランの広告の例
## コードはなし

# ロジステック回帰
## 2値分析
## 目的変数が0,1の2値
train = pd.read_csv('train.csv')

from sklearn.linear_model import LogisticRegression

train["Sex"] = train["Sex"].map({"female":0,"male":1})
X = train[["Sex","Pclass"]]
y = train["Survived"]

# ロジスティック回帰モデルを読み込みモデリング

model = LogisticRegression()
model.fit(X, y)
# print("a:", model.coef_, "b:", model.intercept_)
# print("R^2:", model.score(X, y))  # R square 0.78がでた, 生存をある程度予測できているかも。

# classification reportをimportする
from sklearn.metrics import classification_report

# 予測値を出す
y_pred = model.predict_proba(X)
print(y_pred)

# 精度を算出する
# print(classification_report(y, y_pred))