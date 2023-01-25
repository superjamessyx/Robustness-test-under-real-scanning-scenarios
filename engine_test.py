import torch
import util.misc as misc
import numpy as np
import torch.distributed as dist


def gather_tensor(tensor, world_size):
    gather_list = [torch.empty_like(tensor).cuda() for _ in range(world_size)]
    dist.all_gather(tensor_list=gather_list, tensor=tensor.cuda())
    return gather_list


def board_tensor(tensor, world_size):
    gather_list = [torch.empty_like(tensor).cuda() for _ in range(world_size)]
    dist.broadcast(tensor_list=gather_list, tensor=tensor.cuda())
    return gather_list

def gather_list(ls, world_size):
    tensor = torch.tensor(ls).cuda()
    gathered_tensor = gather_tensor(tensor, world_size)
    gathered_list = []

    for tensor in gathered_tensor:
        gathered_list += list(tensor.cpu().numpy())
    return gathered_list


@torch.no_grad()
def evaluate(args, data_loader, model, model_name, device):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    total_pred_dict = []
    total_pred = []
    total_names = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        print(len(images))
        images = images.to(device, non_blocking=True)
        img_names = np.array(batch[-1])
        with torch.cuda.amp.autocast():
            output = model(images)
            _, pred = torch.topk(output, 1, dim=-1)
            pred = pred.data.cpu().numpy()[:, 0]
            pred[pred > 0] = 1
            total_pred.extend(pred)
        total_names.extend(img_names)
    for idx, pred_ in enumerate(total_pred):
        total_pred_dict.append((total_names[idx], pred_))
    np.save('./predict_result/total_pred-' + model_name + '-' + str(args.fold) + '.npy', total_pred_dict)
