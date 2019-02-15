## scriptの実行
run:
	docker-compose run python python script/main.py
## docker環境内にbashで入る
bash:
	docker-compose run python bash
## dockerfileをbuild
build:
	docker-compose build

pip-install:
	docker-compose run python pip install -r ./requirements.txt

#E TODO: 削除系を整理したい
#E 不要なイメージと使われていないvolumeを削除
clean:
	docker image prune
	docker volume prune
	docker rmi -f `docker images -f "dangling=true" -q`
