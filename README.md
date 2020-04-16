# StackGAN-pokemon
fork元:[StackGAN-v2](https://github.com/hanzhanggit/StackGAN-v2)

StackGANでポケモンの画像を生成します。現状はうまく画像が生成できていないので適宜修正していきます。

## 実行方法
### GAN
##### GANの学習
code/main.pyを実行

```python main.py --cfg cfg/pokemon.yml --gpu 0```

設定はyamlファイルを適宜変更する
### auto-encoder
##### auto-encoderの学習
code/auto-encoder-pokemon/main.pyを実行
```
python main.py --cfg ../cfg/pokemon.yml 
               --gpu 0
               --data_dir  path_to_dir_with_pokemondata_and_img_dir
               --model_dir path_to_save_model 
```
##### auto-encoderの出力確認
code/auto-encoder-pokemon/interpreter.pyを実行

```
 python interpreter.py --model_path saved_model_path
                       --cfg ../cfg/pokemon.yml  
                       --gpu 0 
                       --char_dict path_to_pokemondata.pkl  
```
現状cfg/pokemon.ymlでTRAIN.BATCH_SIZEを1にしないと動かないので変更してから動かす。

実行したらキーボード入力を求められるので、ポケモンの名前やタイプ、種族値を指示通りに入れていく

## 結果例
### auto-encoder
train,valにある第7世代までのポケモンの名前、タイプ、種族値を入れたときの出力
```
入力
pokemon name:チョボマキ
type split by 、 :むし
habcds split by , :50,40,85,40,65,25
出力
pokemon name:チョボマキ
typeむし
49 37 81 39 64 26

入力
pokemon name:ナマコブシ
type split by 、 :みず
habcds split by , :55,60,130,30,130,5
出力
pokemon name:ナマコブシ
typeみず むし
54 58 129 29 129 6

入力
pokemon name:ガブリアス
type split by 、 :ドラゴン、じめん
habcds split by , :108,130,95,80,85,102
出力
pokemon name:ガブリアス
typeじめん ドラゴン
103 126 95 79 82 103
```
train,valにない第8世代のポケモンの名前、タイプ、種族値を入れたときの出力
```
入力
pokemon name:ドラパルト
type split by 、 :ドラゴン、ゴースト
habcds split by , :88,120,75,100,75,142
出力
pokemon name:ドルララー
typeみず むし
83 100 68 82 66 129

入力
pokemon name:エーズバーン
type split by 、 :ほのお
habcds split by , :80,116,75,65,75,119
出力
pokemon name:エーバドー
typeノーマル ひこう
71 114 70 59 72 118

入力
pokemon name:ダイオウドウ
type split by 、 :はがね
habcds split by , :122,130,69,80,69,30
出力
pokemon name:ドウツロイ
typeむし じめん
126 130 77 82 78 49

入力
pokemon name:サッチムシ
type split by 、 :むし
habcds split by , :25,20,20,25,45,45
出力
pokemon name:オッチッ
typeむし でんき
33 29 29 35 50 55
```

やはりタイプが当たらないのでロスの計算方法を変えて試してみる。