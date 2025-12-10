import torch
from Metis import Metis
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data.dataset import ConcatDataset
import argparse
from torch.utils.tensorboard import SummaryWriter
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.set_device(0)
from dataset import *
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def train(args):
    logger = SummaryWriter(os.path.join("/data/chenmingzhi/MambaBrain/logger", 'Metis_V8'))
    model = Metis(n_layers=12, dim=512)
    model = model.cuda()
    dataset_0 = SleepEDF(folder_path='')
    dataset_1 = SHHSPretrain(root='')
    dataset_2 = HMC(folder_path='')
    dataset_3 = ShuDatasetPretrain(folder_path='')
    dataset_4 = AlzheimerDataset()
    dataset_5 = ADauditory(folder_path='')
    dataset_6 = RestEyesOpen()
    dataset_7 = BrainLat(folder_path='')
    dataset_8 = TDBrain(folder_path='', label_dict_path='')
    dataset_9 = CHBMITPretrain(root='')
    dataset_10 = SeizelT2Pretrain(root='')
    dataset_11 = HUP()
    dataset_12 = SWEC_ETHZ(folder_path='')
    dataset_13 = FNUSA(folder_path='')
    dataset_14 = KaggleIEEGEpilepsy()
    dataset_15 = TUABPretrain(root='')
    dataset_16 = TUEVPretrain(root='')
    dataset_17 = PhysionetMI()
    dataset_18 = TUEPPretrain(root='')
    dataset_19 = TUSZPretrain(folder_path='', label_dict_path='')
    dataset = ConcatDataset(
        [dataset_0, dataset_1, dataset_2, dataset_3, dataset_4, dataset_5, dataset_6, dataset_7, dataset_8, dataset_9,
         dataset_10, dataset_11, dataset_12, dataset_13, dataset_14, dataset_15, dataset_16, dataset_17, dataset_18, dataset_19
         ])
    train_size = [len(dataset_0), len(dataset_1), len(dataset_2), len(dataset_3), len(dataset_4), len(dataset_5),
                  len(dataset_6), len(dataset_7), len(dataset_8), len(dataset_9), len(dataset_10), len(dataset_11),
                  len(dataset_12), len(dataset_13), len(dataset_14), len(dataset_15), len(dataset_16), len(dataset_17),
                  len(dataset_18), len(dataset_19)]
    sequential_datasets_idx = None
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                        sampler=SchedulerSampler(dataset=dataset,
                                                 largest_dataset_size=max(train_size), batch_size=args.batch_size,
                                                 sequential_datasets_idx=sequential_datasets_idx), shuffle=False,
                        drop_last=True, num_workers=args.num_workers)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=30000)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    l = len(loader)
    for epoch in range(args.epochs):
        print("epoch:", epoch)
        pbar = tqdm(loader)
        total_loss = 0
        total_ntp_loss = 0
        total_aux_loss = 0
        for i, (signal, ids, label, y) in enumerate(pbar):
            signal = augmentation_new(signal)
            signal = signal.to(args.device)
            ids = ids.to(args.device)
            label = label.to(args.device)
            with torch.cuda.amp.autocast():
                logits, aux_loss = model(signal, ids[:, :-1])
                loss = F.cross_entropy(logits.contiguous().view(-1, logits.size(-1)), label[:, 1:].contiguous().view(-1), ignore_index=-100, reduction='mean')
            total_loss = total_loss + loss.item() + aux_loss.item()
            total_ntp_loss = total_ntp_loss + loss.item()
            total_aux_loss = total_aux_loss + aux_loss.item()
            scaler.scale(loss+aux_loss).backward()
            scheduler.step()
            if (i + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix(total_loss=total_loss, total_ntp_loss=total_ntp_loss, total_aux_loss=total_aux_loss)
                logger.add_scalar("total_loss", total_loss, global_step=epoch * l + i)
                logger.add_scalar("total_ntp_loss", total_ntp_loss, global_step=epoch * l + i)
                logger.add_scalar("total_aux_loss", total_aux_loss, global_step=epoch * l + i)
                total_loss = 0
                total_ntp_loss = 0
                total_aux_loss = 0

            if i % 100000 == 0:
                torch.save(model.state_dict(), os.path.join("", f"Metis{epoch}_{i}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metis Train")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--accumulation_steps", type=int, default=20, help="Gradient accumulation steps")
    args = parser.parse_args()
    train(args)