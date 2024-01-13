# Object Detection

物体検出AIのネットワークモデルを構築するライブラリ。
### 現在使用可能なモデル
- RetinaNet
- FCOS
- EfficientDet

### 現在使用可能なバックボーン
- ResNet50
- EfficientNet b0 ~ b7
- PoolFormer S24, S36
- EfficientNetV2 S, M, L
### 現在使用可能なピラミッドネットワーク
- FPN
- BiFPN

## 事前準備

### Windows

[ここ](https://sourceforge.net/projects/libjpeg-turbo/files/])
から最新版のlibjpeg-turboをインストール。

### CentOS
下記コマンドでlibjpeg-turboをインストール。
```bash
sudo yum install turbojpeg
```
## 環境構築
Python 3.11をインストール。

pipenv をインストール
```bash
pip install pipenv
```

ライブラリのインストール
```bash
pipenv sync
```

## 使用方法




### 推論
下記のコマンドで推論を行うことができます。

```bash
python -m object_detection.predict --image_dir /path/to/image/directory
```
コマンドライン引数

--image_dir 推論対象の画像が入っているディレクトリへのパス\
--ext 画像ファイルの拡張子 (.jpgなど)\
--batch_size バッチサイズ \
--model_path 学習済みモデルへのパス\
--arg_file 学習に用いたパラメータなどの設定ファイルへのパス\
--device 使用するデバイス (cuda or cpu)\
--save_dir 推論結果の画像を保存するディレクトリへのパス\
--img_save 推論結果の画像を保存するかどうか(保存する場合は --img_save　オプションを付ける。しない場合は付けない)


### 学習
下記のコマンドで物体検出AIの学習を行うことができます。 configファイルの中でAIのハイパーパラメータなどの指定を行います。

```sh
train --train_dir /path/to/dataset --config /path/to/config/file
```

コマンドライン引数

--train_dir トレーニング用のデータセットのディレクトリへのパス\
--config トレーニングのパラメータなどの設定ファイルへのパス


### 設定ファイルの書き方
```python
args = dict(
    epoch=100,  # エポック数
    batch_size=16,  # バッチサイズ
    height=1024,  # 画像の高さ
    width=1024,  # 画像の幅
    arch='retinanet',  # 使用する領域検出AIのモデル
    backbone='efficientnet_b4',   # バックボーンのモデル
    learning_rate=1e-4,   # 学習率
    bg_iou_thresh=0.5,  # 背景かどうかの閾値
    fg_iou_thresh=0.6,  # 検出対称の物体かどうかの閾値
    detections_per_img=500,  # nms後に、一画像あたりで検出する物体の最大数
    topk_candidates=1000,  # nms前に、一画像あたりで候補として残す推定領域の最大数
    nms_thresh=0.3,  # nmsの閾値
    score_thresh=0.5,  # スコアの閾値
    multi=False,  # 複数型番まとめて学習するときはTrue, 単数型番のみを学習する場合はFalse
    use_amp=True,  # ampを利用する場合はTrue, 利用しない場合はFalse
    returned_layers=[2, 3, 4, 5, 6, 7],  # backboneが返す特徴マップの層
    out_channels=64,  # 特徴マップのチャンネル数
    baseline='bifpn', # バックボーンのピラミッドネットワーク(fpn or bifpn)
    num_layers=7, # ピラミッドネットワークの層数(fpnを使用する際は使用しない)
)

```