import os
import os.path as osp
import shutil
import torch


def save_checkpoint(obj,path,name):
    torch.save(obj, osp.join(path, name + '.pred'))

def l2_loss(a, b):
    return ((a - b)**2).sum() / (len(a) * 2)

def mask_l2_loss(a, b, mask):
    return l2_loss(a[mask], b[mask])

def ensure_path(path):
    if osp.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)

def set_gpu(gpu):
    if torch.cuda.is_available():
        device = torch.device('cuda:'+ gpu)
        torch.cuda.set_rng_state(torch.cuda.get_rng_state())
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device('cpu')

    print('using gpu {}'.format(gpu))
    return device


def pick_vectors(dic, wnids, is_tensor=False):
    o = next(iter(dic.values()))
    dim = len(o)
    ret = []
    for wnid in wnids:
        v = dic.get(wnid)
        if v is None:
            if not is_tensor:
                v = [0] * dim
            else:
                v = torch.zeros(dim)
        ret.append(v)
    if not is_tensor:
        return torch.FloatTensor(ret)
    else:
        return torch.stack(ret)