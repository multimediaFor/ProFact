import torch
import os
import glob
from collections import OrderedDict

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

# save models
def save_models(pth_dir, epoch, best_iou, model, optimizer, scheduler, tag = 'best'):
    torch.save({
        'epoch': epoch,
        'best_iou': best_iou,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, os.path.join(pth_dir, f'{tag}_epoch_weights.pth'))        

def find_checkpoint(pth_dir, device, load_tag = 'best'):
    pth_zoo = "/*" + load_tag + "*.pth"
    special_pth = glob.glob(pth_dir + pth_zoo)
    if len(special_pth) != 0:
        checkpoint = torch.load(special_pth[0], map_location = device)
        return checkpoint
    else:
        print("[WARNING]: No pretrained weights in path `{}`.".format(pth_dir))
        return None

