import argparse
import datetime
import os
import sys
import warnings

import joblib
import numpy as np
import torch
import tqdm
from sklearn.model_selection import KFold
from torch.optim import lr_scheduler
from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection import MeanAveragePrecision

from . import configs
from .configs.CC import Config
from .datasets import mydataset
from .retinanet import retinanet_model

warnings.simplefilter('ignore')
sys.modules['configs'] = configs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a RetinaNet')
    parser.add_argument('--train_dir', default='/home/f1004/Documents/ObjectDetection/Yamaguchi/train', help='training dataset')
    parser.add_argument('--config', '-c', type=str, default='/home/f1004/Documents/ObjectDetection/object_detection/object_detection/configs/config_all.py', help='path to config file')
    parser.add_argument('--ext', type=str, default='.jpg')
    parser.add_argument('--model_save_dir', type=str, default='./models')
    args = parser.parse_args()
    return args


def collate_fn(batch):
    return tuple(zip(*batch))


def get_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    i = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    return ap


def train(model, train_loader, optimizer, scheduler, epoch, global_step, use_amp):
    loss_epo = []
    pbar = tqdm.tqdm(train_loader)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    model.train()
    for i, batch in enumerate(pbar):
        global_step += 1
        images, targets = batch
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()}
                   for t in targets]
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scheduler.step()
        loss_epo.append(loss_value)
        pbar.set_postfix(loss=str(loss_value))
        scaler.update()
    return loss_epo, global_step


def validation(model, val_loader, fg_iou_thresh):
    metric = MeanAveragePrecision(iou_type="bbox", backend='faster_coco_eval', class_metrics=True)
    model.eval()
    for batch in tqdm.tqdm(val_loader):
        images, targets = batch
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()}
                   for t in targets]
        with torch.no_grad():
            prediction = model(images)
        metric.update(prediction, targets)
    return metric.compute()


def cross_validation(args, model, dataset, optimizer, model_save_dir, ewc=None, lam=10):
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    best_loss = float('inf')
    global_step = 0
    writer = SummaryWriter()

    joblib.dump(args, f'{model_save_dir}/args.pkl')
    with open(f'{model_save_dir}/args.txt', 'w') as f:
        for key in args.keys():
            print('%s: %s' % (key, args[key]), file=f)
    for _fold, (train_index, val_index) in enumerate(kf.split(dataset)):
        train_dataset = Subset(dataset, train_index)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                                                   num_workers=2, pin_memory=True)
        val_dataset = Subset(dataset, val_index)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
                                                 num_workers=2, pin_memory=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=1e-8)
        for epoch in tqdm.tqdm(range(args.epoch)):
            num_iter = _fold * args.epoch + epoch + 1
            loss_epo, global_step = train(model, train_loader, optimizer, scheduler, epoch, global_step, args.use_amp)

            result = validation(model, val_loader, args.fg_iou_thresh)
            loss = np.mean(loss_epo)
            print(f"mean_average precision: {result['map']}, loss: {loss}")
            writer.add_scalar('mAP', result['map'], num_iter)
            writer.add_scalar('mAP50', result['map50'], num_iter)
            writer.add_scalar('training loss', loss, num_iter)
            if num_iter % 10 == 0:
                torch.save({'weights': model.state_dict(), 'height': args.height, 'width': args.width}, f'{model_save_dir}/model{str(num_iter)}.pt')
        if np.mean(loss_epo) < best_loss:
            best_loss = np.mean(loss_epo)
            torch.save({'weights': model.state_dict(), 'height': args.height, 'width': args.width}, f'{model_save_dir}/best.pt')
    torch.save({'weights': model.state_dict(), 'height': args.height, 'width': args.width}, f'{model_save_dir}/model.pt')


def train_model(args, model, classes, optimizer, model_path):
    torch.backends.cudnn.benchmark = True
    loss_list = []
    print('loading')
    train_loader = mydataset.dataloader(args.train_dir, classes, args.batch_size, args.scale, args.multi)
    print('complete')
    pbar = tqdm.tqdm(range(args.epoch))
    for epoch in pbar:
        pbar.set_postfix(epoch=str(epoch + 1))
        loss_epo = train(model, train_loader, optimizer)
        loss_list.append(np.mean(loss_epo))
        torch.save(model.state_dict(), model_path)


def get_model(args, cfg, dataset_class):
    backbone = cfg.args.backbone
    arch = cfg.args.arch
    size = min(cfg.args.height, cfg.args.width)
    returned_layers = cfg.args.returned_layers
    model = retinanet_model.get_model(len(dataset_class) + 1, min_size=size, max_size=size, pretrained=True, arch=arch, backbone=backbone, is_torch=cfg.args.is_torch, out_channels=cfg.args.out_channels, baseline=cfg.args.baseline, num_layers=cfg.args.num_layers, score_thresh=cfg.args.score_thresh, nms_thres=cfg.args.nms_thresh, fg_iou_thresh=cfg.args.fg_iou_thresh, bg_iou_thresh=cfg.args.bg_iou_thresh, topk_candidates=cfg.args.topk_candidates, detections_per_img=cfg.args.detections_per_img, layers=returned_layers)
    model.to(device)
    return model


def main():
    args = parse_args()
    train_dir = args.train_dir
    cfg = Config.fromfile(args.config)
    dataset_class = cfg.args.classes
    backbone = cfg.args.backbone
    model = get_model(args, cfg, dataset_class)
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = SAM(params, base_optimizer, lr=cfg.args.learning_rate, betas=(0.9, 0.99), eps=1e-6, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(params, lr=cfg.args.learning_rate, betas=(0.9, 0.99), eps=1e-6, weight_decay=1e-4)
    model_save_dir = args.model_save_dir
    model_save_dir = os.path.join(model_save_dir, backbone)
    now = datetime.datetime.now().strftime('%Y%m%d%H%M')
    now = f'{os.path.basename(train_dir)}_{now}'
    model_save_dir = os.path.join(model_save_dir, now)
    os.makedirs(model_save_dir, exist_ok=True)
    dataset = mydataset.MyDataset(train_dir, dataset_class, ext=args.ext)
    cross_validation(cfg.args, model, dataset, optimizer, model_save_dir, ewc=None, lam=308.945075261017)


if __name__ == '__main__':
    main()
