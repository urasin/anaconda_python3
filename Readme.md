# Anaconda3 + python3 in Docker

## python version
- 3.7.1 (2019/02/15時点) imageの更新によりupdateされる可能性有り

## How to use

### Install
- make build

### Execte code endpoint
- script/main.py

### Execute
- make run

### Pip install
- make pip-install # 一時的にInstall、containerを消すとライブラリも消える
- edit requirements.py and make build # 永続化、containerを消しても残したいとき使う

## Misc
- できればpipライブラリをボリュームコンテナに入れたい
- jupyter対応したい。portの開放のみでいけるか？