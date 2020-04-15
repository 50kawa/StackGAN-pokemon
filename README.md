# StackGAN-pokemon
fork元:[StackGAN-v2](https://github.com/hanzhanggit/StackGAN-v2)

StackGANでポケモンの画像を生成します。現状はうまく画像が生成できていないので適宜修正していきます。

## 実行方法
##### GANの学習
code/main.pyを実行

```python main.py --cfg cfg/pokemon.yml --gpu 0```

設定はyamlファイルを適宜変更する
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