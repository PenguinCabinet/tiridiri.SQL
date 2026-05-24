# 散り散り.SQL
> [!IMPORTANT]
> 本リポジトリは[ボカロ曲「散り散り」](https://www.youtube.com/watch?v=Xn-JobZlsQo)の二次創作です     
> ボカロ曲「散り散り」は、[いくつかの条件の下で二次利用が許可されています](https://yomitanakane.myportfolio.com/contact)。    
> 映像を解析し、SQLで記述・それの動画を作成するという点で、本リポジトリに創意性があるという私の認識です     
> ※本リポジトリは作業途中です。     

キャラクターが表示・非表示を繰り返す[ボカロ曲「散り散り」の映像](https://www.youtube.com/watch?v=Xn-JobZlsQo)を、SQLで記述したリポジトリです。

[original.mp4](./video/original.mp4)は、加工されていないオリジナルのボカロ曲「散り散り」の映像です(ただし、ダウンロードする過程で画質は変化しているみたいです)。

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

## make_SQL_video.py
各フレームのSQL文が書かれた[SQL.yaml](./SQL.yaml)から、SQLをターミナル画像にして、動画にまとめ、音楽と合成した[SQL.mp4](./video/SQL.mp4)を出力します。
```
python make_SQL_yaml_by_video.py
```
