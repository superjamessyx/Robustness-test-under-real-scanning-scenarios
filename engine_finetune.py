
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import confusion_matrix
import numpy as np

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():

            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



import torch.distributed as dist


def gather_tensor(tensor, world_size):
    gather_list = [torch.empty_like(tensor).cuda() for _ in range(world_size)]
    dist.all_gather(tensor_list=gather_list, tensor=tensor.cuda())
    return gather_list

def board_tensor(tensor, world_size):
    gather_list = [torch.empty_like(tensor).cuda() for _ in range(world_size)]
    dist.broadcast(tensor_list=gather_list, tensor=tensor.cuda())
    return gather_list
#
# def reduce_list(ls, world_size):
#     tensor = torch.tensor(ls).cuda()
#     reduced_tensor = reduce_tensor(tensor, world_size)
#     return reduced_tensor.mean().cpu().item()

def gather_list(ls, world_size):
    tensor = torch.tensor(ls).cuda()
    gathered_tensor = gather_tensor(tensor, world_size)
    gathered_list = []

    for tensor in gathered_tensor:
        gathered_list += list(tensor.cpu().numpy())
    return gathered_list


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    total_gt = []
    total_pred = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-2]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # print(target[0])
        # compute output
        img_names = np.array(batch[-1])
        with torch.cuda.amp.autocast():
            output = model(images)
            _, pred = torch.topk(output, 1, dim=-1)
            pred = pred.data.cpu().numpy()[:, 0]
            gt = target.data.cpu().numpy()
            total_pred.extend(pred)
            total_gt.extend(gt)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    # import pdb
    # pdb.set_trace()

    metric_logger.synchronize_between_processes()

    test_con_mat = confusion_matrix(total_gt, total_pred)
    # dist_eval
    train_pred = gather_list(total_pred, 4)
    train_gt = gather_list(total_gt, 4)
    test_con_mat = confusion_matrix(train_gt, train_pred)
    print(test_con_mat)
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

