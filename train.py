import logging
import argparse
import os
import random
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models.modeling import VisionTransformer,CONFIGS
from transformers import get_cosine_schedule_with_warmup
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, f"{args.name}_checkpoint.bin")
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def setup(args,config):
    num_classes = 10 if args.dataset == "cifar10" else 100
    model = VisionTransformer(config=config, zero_head=True, num_classes=num_classes,vis=False)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)
    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_dataloaders(args,config):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.Resize(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    ])
    dataset_dict = {
        "cifar10":datasets.CIFAR10,
        "cifar100":datasets.CIFAR100
    }
    dataset_class = dataset_dict[args.dataset]
    train_dataset = dataset_class(root=args.data_dir,train=True,transform=train_transform,download=True)
    test_dataset = dataset_class(root=args.data_dir,train=False,transform=test_transform,download=True)
    train_loader = DataLoader(train_dataset,shuffle=True,batch_size=args.train_batch_size)
    test_loader = DataLoader(test_dataset,shuffle=False,batch_size=args.eval_batch_size)
    return train_loader,test_loader

def valid(args, model, writer, test_loader, global_step):
    eval_losses = AverageMeter()
    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                      desc="Validating... (loss=X.X)",
                      bar_format="{l_bar}{r_bar}",
                      dynamic_ncols=True)
    loss_fct = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            logits,attn_weights = model(x)
            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item(),n=x.size(0))
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_label.append(y.cpu().numpy())
            epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
    all_preds = np.concatenate(all_preds, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    accuracy = simple_accuracy(all_preds, all_label)
    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)
    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy

def train(args,config, model):
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    train_loader,test_loader = get_dataloaders(args,config)
    optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate,momentum=args.momentum,weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=args.warmup_steps,num_training_steps=args.num_steps)
    model.zero_grad()
    set_seed(args)
    losses = AverageMeter()
    global_step = 0
    micro_step = 0
    best_accuracy = float("-inf")
    while global_step<args.num_steps:
        for x,y in train_loader:
            micro_step+=1
            x = x.to(args.device)
            y = y.to(args.device)
            loss = model(x,y)
            losses.update(loss.item(),x.size(0))
            loss = loss/args.gradient_accumulation_steps
            loss.backward()
            if micro_step%args.gradient_accumulation_steps==0:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),max_norm=args.max_grad_norm)
                current_lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step +=1
                writer.add_scalar("train/loss",scalar_value=losses.avg,global_step=global_step)
                writer.add_scalar("train/lr",scalar_value=current_lr,global_step=global_step)
                losses.reset()
                if global_step%args.eval_every==0:
                    accuracy = valid(args,model,writer,test_loader,global_step)
                    model.train()
                    if accuracy > best_accuracy:
                        save_model(args,model)
                        best_accuracy = accuracy
            if global_step>=args.num_steps:
                break
    if global_step%args.eval_every!=0:
        accuracy = valid(args, model, writer, test_loader, global_step)
        model.train()
        if accuracy > best_accuracy:
            save_model(args, model)
            best_accuracy = accuracy
    writer.close()
    logger.info(best_accuracy)
    logger.info("End Training!")

def get_args():
    parser = argparse.ArgumentParser(description="Train Vision Transformer")
    parser.add_argument("--name",type=str,default="vit_exp",help="实验名称")
    parser.add_argument("--dataset",type=str,default="cifar10",choices=["cifar10", "cifar100"],help="训练数据集")
    parser.add_argument(
        "--model_type",
        type=str,
        default="VIT-B_16",
        choices=[
            "VIT-B_16",
            "VIT-B_32",
            "VIT-L_16",
            "VIT-L_32",
            "VIT-H_14",
            "VIT-testing",
            "VIT-r50_b16"
        ],
        help="ViT模型类型"
    )
    parser.add_argument("--pretrained_dir",type=str,default=str(BASE_DIR/"checkpoint"/"ViT-B_16.npz"),help="预训练npz权重路径")
    parser.add_argument("--output_dir",type=str,default=str(BASE_DIR / "output"),help="模型保存目录")
    parser.add_argument("--train_batch_size",type=int,default=32,help="训练batch size")
    parser.add_argument("--eval_batch_size",type=int,default=64,help="验证batch size")
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,help="梯度累计步数")
    parser.add_argument("--learning_rate",type=float,default=3e-2,help="初始学习率")
    parser.add_argument("--weight_decay",type=float,default=0.0,help="权重衰减")
    parser.add_argument("--momentum",type=float,default=0.9,help="SGD momentum")
    parser.add_argument("--max_grad_norm",type=float,default=1.0,help="梯度裁剪最大范数")
    parser.add_argument("--num_steps",type=int,default=10000,help="最大optimizer更新步数")
    parser.add_argument("--warmup_steps",type=int,default=500,help="warmup步数")
    parser.add_argument("--eval_every",type=int,default=100,help="每多少个global step验证一次")
    parser.add_argument("--seed",type=int,default=42,help="随机种子")
    parser.add_argument("--data_dir", type=str, default=str(BASE_DIR / "data"), help="模型保存目录")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO )
    logger.warning("device: %s, n_gpu: %s" %( args.device, args.n_gpu))
    set_seed(args)
    config = CONFIGS[args.model_type]
    model = setup(args,config)
    train(args,config,model)
