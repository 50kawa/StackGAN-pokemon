# ポケモンのデータ成型
makedata.pyを実行するとpokemondata.pklに学習に必要なデータが成型できる。


## DLしたデータ
- 英語のポケモン名をキーとした日本語のポケモン名とタイプ:[pokemon_20191125.csv](https://empoh.com/download/1478/)
- ポケモンの種族値:[pokemon-with-stats-generation-8](https://www.kaggle.com/tlegrand/pokemon-with-stats-generation-8)
- ポケモンの画像:[pokemon-images-and-types](https://www.kaggle.com/vishalsubbiah/pokemon-images-and-types)
   - 形式がバラバラなのでpngtojpeg.pyでjpegに変換する