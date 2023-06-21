"""Basic script to train CIFAR10 and ImageNet close to SOTA with ResNets"""

"""
CIFAR10
accelerate launch --mixed_precision fp16 main.py --model $model --cifar-resize $size --batch-size 128 --seed 0
model     |         32         |         52         |
resnet20  | 97.97% ( 90% 2h30) | 98.66% (100% 3h38) |
resnet56  | 98.27% ( 90% 5h22) | 
resnet18  | 97.77% (100% 2h28) | 
resnet50  | 98.17% ( 90% 5h18) | 
resnet20, width 16, 32x32: 94.65% 
resnet56, width 16, 32x32: 96.76%

CIFAR100
accelerate launch --mixed_precision fp16 main.py --model $model --dataset cifar100 --cifar-resize $size --batch-size 128 --seed 0
model     |         32         |         52         |
resnet20  | 83.69% ( 90% 2h32) | 83.88% ( 70% 3h42) | 
resnet56  | 85.64% ( 70% 5h27) | 
resnet18  | 82.96% ( 70% 2h16) |
resnet50  | 84.81% ( 60% 5h39) |
resnet20, width 16, 32x32: 71.83%
resnet56, width 16, 32x32: 

ImageNet
accelerate launch --mixed_precision fp16 main.py --model resnet18 --dataset imagenet --seed 0
accelerate launch --mixed_precision fp16 main.py --model resnet50 --dataset imagenet --seed 0
"""

import torch
import torch.nn as nn

import os
import torchvision
import torchvision.transforms as transforms
import argparse
import random
import time
import math
import numpy as np
from accelerate import Accelerator

from resnet import *

from utils import ExponentialMovingAverage, RandomMixup, RandomCutmix
from torch.utils.data.dataloader import default_collate

accelerator = Accelerator()

parser = argparse.ArgumentParser(description="Vincent's Training Routine")
parser.add_argument('--dataset', type=str, default="CIFAR10", help="CIFAR10, CIFAR100 or ImageNet")
parser.add_argument('--steps', type=int, default=750000)
parser.add_argument('--batch-size', type = int, default=1024)
parser.add_argument('--seed', type = int, default = random.randint(0, 1000000000))
parser.add_argument('--model', type=str, default="resnet50")
parser.add_argument('--width', type=int, default=64, help="number of feature maps for first layers")
parser.add_argument('--dataset-path', type=str, default=os.getenv("DATASETS"))
parser.add_argument('--cifar-resize', type=int, default=32)
parser.add_argument('--label-smoothing', type=float, default=0.1)
parser.add_argument('--test-steps', type=int, default=15)
parser.add_argument('--adam', action="store_true")
parser.add_argument('--mixup-alpha', type=float, default=0.2)
parser.add_argument('--cutmix-alpha', type=float, default=1.)
parser.add_argument('--eras', type=int, default=1)
args = parser.parse_args()

# deterministic mode for reproducibility
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.use_deterministic_algorithms(True)
print("random seed is", args.seed)

# prepare dataloaders
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
if args.dataset.lower() == "imagenet":
    train = torchvision.datasets.ImageNet(
        root=args.dataset_path,
        split="train",
        transform=transforms.Compose([
            transforms.RandomResizedCrop(176, antialias=True),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            normalize,
            torchvision.transforms.RandomErasing(0.1)
        ]))
    test = torchvision.datasets.ImageNet(
        root=args.dataset_path,
        split="val",
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(236, antialias=True),
            transforms.CenterCrop(224),
            normalize]))
    num_classes, large_input = 1000, True
if args.dataset.lower() == "cifar10" or args.dataset.lower() == "cifar100":
    if args.dataset.lower() == "cifar10":
        tvdset = torchvision.datasets.CIFAR10
        num_classes = 10
    else:
        tvdset = torchvision.datasets.CIFAR100
        num_classes = 100
    train = tvdset(
        root=args.dataset_path,
        train=True,
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            normalize,
            transforms.Resize(args.cifar_resize, antialias=True),
            transforms.RandomErasing(0.1)
        ]))
    test = tvdset(
        root=args.dataset_path,
        train=False,
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            transforms.Resize(args.cifar_resize, antialias=True)
        ]))
    large_input = False
mixupcutmix = torchvision.transforms.RandomChoice([RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha), RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha)])
def collate_fn(batch):
    return mixupcutmix(*default_collate(batch))

train_loader = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        num_workers=min(30, os.cpu_count()), drop_last=True, pin_memory=True, collate_fn=collate_fn, persistent_workers=True)

test_loader = torch.utils.data.DataLoader(
        test, batch_size=args.batch_size, shuffle=False,
        num_workers=min(30, os.cpu_count()), pin_memory=True, persistent_workers=True)

# Prepare model, EMA and parameter sets
net = eval(args.model)(num_classes, large_input, args.width)
net, train_loader, test_loader = accelerator.prepare(net, train_loader, test_loader)
net.to(non_blocking=True, memory_format=torch.channels_last)
num_parameters = int(torch.tensor([x.numel() for x in net.parameters()]).sum().item())
accelerator.print("{:d} parameters".format(num_parameters))

ema = ExponentialMovingAverage(net, decay=0.999)
ema = accelerator.prepare(ema)
ema.eval()

modules = [x for x in net.modules()]
wd = []
nowd = []
trained_parameters = 0
for x in modules:
    if isinstance(x, nn.BatchNorm2d) or isinstance(x, nn.Linear):
        if isinstance(x, nn.BatchNorm2d):
            nowd.append(x.weight)
            trained_parameters += x.weight.numel()
        else:
            wd.append(x.weight)
            trained_parameters += x.weight.numel()
        nowd.append(x.bias)
        trained_parameters += x.bias.numel()
    elif isinstance(x, nn.Conv2d):
        wd.append(x.weight)
        trained_parameters += x.weight.numel()
assert(num_parameters == trained_parameters)

# define criterion and aggregators
criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
train_losses, test_scores, test_scores_ema, test_card = [], [], [], []
peak, peak_step, peak_ema, peak_step_ema = 0, 0, 0, 0

# test function
def test():
    net.eval()
    correct = 0
    total = 0
    correct_ema = 0
    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(non_blocking=True, memory_format=torch.channels_last), targets.to(non_blocking=True)
            outputs = net(inputs)
            outputs_ema = ema(inputs)
            _, predicted = outputs.max(1)
            _, predicted_ema = outputs_ema.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum()
            correct_ema += predicted_ema.eq(targets).sum()
    accelerator.print("{:6.2f}% (ema: {:6.2f}%)".format(100.*correct/total, 100.*correct_ema/total))
    net.train()
    return correct_ema/total

start_time = time.time()
epoch = 0

test_enum = list(test_loader)
index_test = 0

net.train()

for era in range(1 if args.adam else 0, args.eras + 1):
    step = 0
    print("{:s}".format(str(era + 1) if era > 0 else "Warming up"))

    # define optimizers/schedulers
    if era == 0:
        optimizer = torch.optim.SGD([{"params":wd, "weight_decay":2e-5}, {"params":nowd, "weight_decay":0}], lr=0.5, momentum=0.9, nesterov=True)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.01, total_iters = len(train_loader) * 5)
    else:
        if args.adam:
            optimizer = torch.optim.AdamW([{"params":wd, "weight_decay":0.05}, {"params":nowd, "weight_decay":0}], lr = 1e-3 * (0.9 ** era))
        else:
            optimizer = torch.optim.SGD([{"params":wd, "weight_decay":2e-5}, {"params":nowd, "weight_decay":0}], lr=0.5 * (0.9 ** era), momentum=0.9, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps - step)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    total_steps_for_era = args.steps if era > 0 else 5 * len(train_loader)
    
    while step < total_steps_for_era:
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            step += 1
            inputs, targets = inputs.to(non_blocking=True, memory_format=torch.channels_last), targets.to(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = net(inputs)

            loss = criterion(outputs, targets)

            accelerator.backward(loss)
            optimizer.step()
            if step % 32 == 0:
                ema.update_parameters(net)
                if epoch < 5:
                    ema.n_averaged.fill_(0)
            train_losses.append(loss.item())
            train_losses = train_losses[-len(train_loader):]

            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            accelerator.print("\r{:6.2f}% loss:{:.4e} lr:{:.3e}".format(100 * step / total_steps_for_era, torch.mean(torch.tensor(train_losses)).item(), lr), end="")

            step_time = (time.time() - start_time) / (args.steps * (era-1) + step + (5 * len(train_loader) if not args.adam else 0))
            remaining_time = (total_steps_for_era - step + (args.eras - era - 1) * args.steps) * step_time
            
            if accelerator.is_main_process:
                score = 100 * accelerator.gather_for_metrics(torch.tensor(test_scores)).sum() / accelerator.gather_for_metrics(torch.tensor(test_card)).sum()
                score_ema = 100 * accelerator.gather_for_metrics(torch.tensor(test_scores_ema)).sum() / accelerator.gather_for_metrics(torch.tensor(test_card)).sum()
                if score > peak:
                    peak = score
                    peak_step = step
                if score_ema > peak_ema:
                    peak_ema = score_ema
                    peak_step_ema = step
                accelerator.print(" {:6.2f}% (ema {:6.2f}%) {:4d}h{:02d}m {:d} epochs".format(score, score_ema, int(remaining_time / 3600), (int(remaining_time) % 3600) // 60, epoch + 1), end='')

            if batch_idx % args.test_steps == 0:
                net.eval()
                try:
                    inputs, targets = test_enum[index_test]
                    index_test = (index_test + 1) % len(test_enum)
                except StopIteration:
                    test_enum = enumerate(test_loader)
                    _, (inputs, targets) = next(test_enum)
                inputs, targets = inputs.to(non_blocking=True, memory_format=torch.channels_last), targets.to(non_blocking=True)
                with torch.inference_mode():
                    outputs = net(inputs)
                    outputs_ema = ema(inputs)
                    _, predicted = outputs.max(1)
                    _, predicted_ema = outputs_ema.max(1)
                    correct = predicted.eq(targets).sum()
                    correct_ema = predicted_ema.eq(targets).sum()                    
                    test_scores.append(correct.item())
                    test_scores_ema.append(correct_ema.item())
                    test_card.append(inputs.shape[0])
                    test_scores = test_scores[-len(test_enum):]
                    test_scores_ema = test_scores_ema[-len(test_enum):]
                    test_card = test_card[-len(test_enum):]
                net.train()
            if step % (args.steps // 10) == 0 and step > 1:
                accelerator.print("\r{:3d}% steps score:                    ".format(round(100 * step / args.steps)), end='')
                res = test()
                if step == (10 * (args.steps // 10)):
                    break
        epoch += 1

total_time = time.time() - start_time
accelerator.print()
accelerator.print("total time is {:4d}h{:02d}m".format(int(total_time / 3600), (int(total_time) % 3600) // 60))
accelerator.print("Peak perf is {:6.2f}% at step {:d} ({:6.2f}% at step {:d})".format(peak, peak_step, peak_ema, peak_step_ema))
accelerator.print()
accelerator.print()

