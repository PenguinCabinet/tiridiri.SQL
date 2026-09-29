# 散り散り.SQL
> [!IMPORTANT]
> 本リポジトリは[ボカロ曲「散り散り」](https://www.youtube.com/watch?v=Xn-JobZlsQo)の二次創作です     
> ボカロ曲「散り散り」は、[いくつかの条件の下で二次利用が許可されています](https://yomitanakane.myportfolio.com/contact)。    
> 映像を解析し、SQLで記述・それの動画を作成するという点で、本リポジトリに創意性があるという私の認識です     
> ※本リポジトリは作業途中です。     

キャラクターが表示・非表示を繰り返す[ボカロ曲「散り散り」の映像](https://www.youtube.com/watch?v=Xn-JobZlsQo)を、SQLで記述したリポジトリです。

[original.mp4](./video/original.mp4)は、加工されていないオリジナルのボカロ曲「散り散り」の映像です(ただし、ダウンロードする過程で画質は変化しているみたいです)。

# 成果物の動画
* 「散り散り.SQL」 - [Youtube](https://www.youtube.com/watch?v=yEDGXC1o6I8),[ニコニコ動画](https://www.nicovideo.jp/watch/sm46862151)
* 「散り散りの画像認識」 - [Youtube](https://www.youtube.com/watch?v=xRp4AJ_pzlY),[ニコニコ動画](https://www.nicovideo.jp/watch/sm46862181)

# 説明
[database.db](./database.db)のcharactersテーブルに登場するキャラクターのデータが含まれています。

```
sqlite> SELECT * FROM characters;
╭─────────┬──────────╮
│  name   │  status  │
╞═════════╪══════════╡
│ teacher │ real     │
│ teacher │ portrait │
╰─────────┴──────────╯
```

opencvでパターンマッチングした過程の動画ファイルが[OpenCV_processing_process.mp4](./video/OpenCV_processing_process.mp4)です。

そして、どのフレームでどのSQLを実行すれば、オリジナルの結果と一致するかは[SQL.yaml](./SQL.yaml)に記述されています。

これらのSQLをターミナル画像にして、動画にまとめ、音楽と合成したのが、[SQL.mp4](./video/SQL.mp4)です。


# Usage

## make_SQL_yaml_by_video.py

[オリジナルの動画](./video/OpenCV_processing_process.mp4)から、各フレームのSQL文が書かれた[SQL.yaml](./SQL.yaml)とopencvでパターンマッチングした過程の動画ファイル[OpenCV_processing_process.mp4](./video/OpenCV_processing_process.mp4)を出力します。
```
python make_SQL_yaml_by_video.py
```

実行には、ffmpegが必要です。

標準の認識方式は、この動画の座席配置に合わせた `layout` です。
`recognition_profile/` に保存した動画由来の顔画像と基準画像を使用し、SIFTの特徴点と
RANSACでカメラの拡大・移動を推定してから、各人物の座席付近を照合します。
遺影は別の固定レイヤーとして照合するため、遺影を実体として誤検出しにくくなります。
白い歌詞による遮蔽と画面端の顔の切れを考慮した相関を求め、最後に動画全体の状態遷移を
Viterbi法で最適化します。時刻に応じた正解状態を認識処理に埋め込んではいません。

依存パッケージは `python -m pip install -r requirements.txt` でインストールできます。
出力は `SQL.yaml`、音声付き確認動画 `video/OpenCV_processing_process.mp4`、
全人物・全フレームの認識ログ `video/recognition.csv` です。
ログには状態、実体・遺影それぞれの一致度、検出位置、位置推定のインライア数を記録します。
インライア数が0の場合は固定の広角構図で照合します。

```
python make_SQL_yaml_by_video.py --threshold 0.72 --switch-cost 1
```

確認動画が不要な試行では `--no-debug-video` を指定できます。
旧方式との比較には `--detector templates --scales 0.9 1.0 1.1` を指定します。
`--scales` と `--templates` は旧方式にだけ適用されます。
別の動画や配置に対しては、顔の切り出し・座席座標・基準画像の更新が必要です。
この動画には意図的な短い表示があるため、`layout` の `--switch-cost` の既定値は1です。
値を大きくすると、正しい短時間の表示まで消える場合があります。旧方式の既定値は5です。

精度検証では、元動画を目視して作成した `evaluation/labels.json` を使用します。
テンプレート抽出に使ったフレームは評価から除外しています。
正解ラベルがあるフレームでの一致率であり、動画全フレームの正解率ではありません。
実測値・改善前後の画像・確認内容は [検証レポート](evaluation/REPORT.md) にまとめています。

```
python make_SQL_yaml_by_video.py --scores-json evaluation/improved_full.json
python evaluate_detection.py evaluation/improved_full.json --output evaluation/improved_metrics.json
python -m unittest -v
```

## make_SQL_video.py
各フレームのSQL文が書かれた[SQL.yaml](./SQL.yaml)から、SQLをターミナル画像にして、動画にまとめ、音楽と合成した[SQL.mp4](./video/SQL.mp4)を出力します。
```
python make_SQL_yaml_by_video.py
```
