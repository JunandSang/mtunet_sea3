import os
import torch
from pathlib import Path
from loaders.base_loader import get_dataloader
import model.extractor as base_bone
from tools.calculate_tool import MetricLogFew
from engine_base import train_one_epoch, evaluate
from model.model_tools import print_param
import tools.prepare_things as prt
import time
import datetime
import argparse


def main(args):
    device = torch.device(args.device)

    # CSV 파일 경로 설정
    args.data_root = "/content/mtunet_sea3/FSL_data/seadata"
    if not args.dataset:
        print("Warning: args.dataset is not set. Defaulting to 'seas'.")
        args.dataset = "seas"

    print(f"Dataset name: {args.dataset}")
    print("Train CSV Path: ", os.path.join(args.data_root, "train.csv"))
    print("Validation CSV Path: ", os.path.join(args.data_root, "val.csv"))
    print("Test CSV Path: ", os.path.join(args.data_root, "test.csv"))

    # 데이터 로더 초기화
    selection = None  # 모든 클래스를 선택
    loaders_train = get_dataloader(args, "train", selection=selection)
    loaders_val = get_dataloader(args, "val", selection=selection)

    # 모델 초기화
    model = base_bone.__dict__[args.base_model](num_classes=args.num_classes, drop_dim=True, extract=True)
    model.to(device)
    print_param(model)

    # 최적화 및 스케줄러 설정
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop)

    print("Start training")
    start_time = time.time()
    log = MetricLogFew(args)
    record = log.record
    output_dir = Path(args.output_dir)

    max_acc1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, loaders_train, device, record, epoch, optimizer, torch.nn.CrossEntropyLoss())
        evaluate(args, model, loaders_val, device, record, epoch)
        lr_scheduler.step()

        # 모델 저장
        if args.output_dir:
            checkpoint_path = output_dir / f"{args.dataset}_{args.base_model}_checkpoint.pth"
            if record["val"]["accm"][-1] > max_acc1:
                print("Higher accuracy found, saving model...")
                max_acc1 = record["val"]["accm"][-1]
                prt.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

    total_time = time.time() - start_time
    print('Training time:', str(datetime.timedelta(seconds=int(total_time))))


def get_args_parser():
    parser = argparse.ArgumentParser(description="Arguments for model training")
    parser.add_argument('--dataset', type=str, default="", help="Dataset name")
    parser.add_argument('--base_model', type=str, default="resnet18", help="Base model name")
    parser.add_argument('--channel', type=int, default=512, help="Channel size")
    parser.add_argument('--num_classes', type=int, default=10, help="Number of classes")
    parser.add_argument('--data_root', type=str, default="", help="Root path of dataset")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
    parser.add_argument('--lr_drop', type=int, default=20, help="Step size for learning rate decay")
    parser.add_argument('--output_dir', type=str, default='saved_model', help="Directory to save model checkpoints")
    parser.add_argument('--start_epoch', type=int, default=0, help="Start epoch")
    parser.add_argument('--fsl', action='store_true', help="Enable Few-Shot Learning mode")
    parser.add_argument('--img_size', type=int, default=224, help="Image size for resizing")
    parser.add_argument('--aug', type=bool, default=True, help="Enable data augmentation")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for DataLoader")
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

