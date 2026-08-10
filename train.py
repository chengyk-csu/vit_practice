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

def setup(args):
    config = CONFIGS[args.model_type]
    num_classes = 10 if args.dataset == "cifar10" else 100
    model = VisionTransformer(config=config, zero_head=True, num_classes=num_classes,vis=False)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)
    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

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
                      dynamic_ncols=True,
                      disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            logits = model(x)[0]
            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())
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

def train(args, model):
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

