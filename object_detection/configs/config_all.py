
args = dict(
    epoch=10,  # エポック数
    batch_size=8,  # バッチサイズ
    height=1024,  # 画像の高さ
    width=1024,  # 画像の幅
    arch='retinanet',  # 使用する領域検出AIのモデル
    backbone='efficientnetv2_m',   # バックボーンのモデル
    learning_rate=1e-4,   # 学習率
    bg_iou_thresh=0.5,  # 背景かどうかの閾値
    fg_iou_thresh=0.6,  # 検出対称の物体かどうかの閾値
    detections_per_img=300,  # nms後に、一画像あたりで検出する物体の最大数
    topk_candidates=1000,  # nms前に、一画像あたりで候補として残す推定領域の最大数
    nms_thresh=0.3,  # nmsの閾値
    score_thresh=0.5,  # スコアの閾値
    multi=True,  # 複数型番まとめて学習するかどうか
    use_amp=True,  # ampを利用するかどうか
    returned_layers=[2, 3, 4, 5, 6, 7],  # backboneが返す特徴マップの層
    out_channels=64,  # 特徴マップのチャンネル数
    baseline='bifpn',
    num_layers=7,
    classes=['aeroplane', 'bicycle', 'bird'],  # 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
    colors=((0, 0, 0), (0, 255, 0)),
    is_torch=True,
)
