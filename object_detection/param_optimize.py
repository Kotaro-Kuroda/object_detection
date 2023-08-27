import torch
import os
import optuna

import warnings
import datetime
from .configs.CC import Config
from . import config_chip
from .datasets import mydataset
from .train import parse_args, get_model, cross_validation
optuna.logging.disable_default_handler()

warnings.simplefilter('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = parse_args()
train_dir = args.train_dir
cfg = Config.fromfile(args.config)
dataset_class = config_chip.dataset_class
backbone = cfg.args.backbone
model, ewc = get_model(args)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=cfg.args.learning_rate, betas=(0.9, 0.99), eps=1e-6, weight_decay=1e-4)
model_save_dir = args.model_save_dir
model_save_dir = os.path.join(model_save_dir, backbone)
now = datetime.datetime.now().strftime('%Y%m%d%H%M')
now = f'{os.path.basename(train_dir)}_{now}'
model_save_dir = os.path.join(model_save_dir, now)
os.makedirs(model_save_dir, exist_ok=True)
dataset = mydataset.MyDataset(train_dir, cfg.args.height, cfg.args.width, dataset_class, cfg.args.multi, ext=args.ext)
mean = dataset.mean
std = dataset.std


def get_lambda(trial):
    lam = trial.suggest_float('lambda', 0, 1000)
    return lam


def objective(trial):
    lam = get_lambda(trial)
    obj_value = cross_validation(cfg.args, model, dataset, optimizer, model_save_dir, mean, std, ewc=ewc, lam=lam)
    print(f'目的関数地:{obj_value}, lambda:{lam}')
    return obj_value


def main():
    TRIAL_SIZE = 100
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=TRIAL_SIZE)
    print(study.best_params)
    print()


if __name__ == "__main__":
    main()
