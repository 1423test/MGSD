import argparse
import json
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from visual.resnet import make_resnet50_base
from loader.imagenet import ImageNet
from tools import set_gpu, pick_vectors
import scipy.io as scio


def test_on_subset(dataset, cnn, n, pred_vectors, all_label,
                   consider_trains):
    top = [1, 2, 5, 10, 20]
    hits = torch.zeros(len(top)).cuda()
    tot = 0

    loader = DataLoader(dataset=dataset, batch_size=32,
                        shuffle=False, num_workers=2)

    for batch_id, batch in enumerate(loader, 1):
        data, label = batch
        data = data.cuda()

        feat = cnn(data)  # (batch_size, d)
        feat = torch.cat([feat, torch.ones(len(feat)).view(-1, 1).cuda()], dim=1)

        fcs = pred_vectors.t()

        table = torch.matmul(feat, fcs)
        if not consider_trains:
            table[:, :n] = -1e18

        gth_score = table[:, all_label].repeat(table.shape[1], 1).t()
        rks = (table >= gth_score).sum(dim=1)

        assert (table[:, all_label] == gth_score[:, all_label]).min() == 1

        for i, k in enumerate(top):
            hits[i] += (rks <= k).sum().item()
        tot += len(data)

    return hits, tot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn',default='./visual/resnet50-base.pth')
    parser.add_argument('--pred', default='./models/f.pred')

    parser.add_argument('--test-set', default='hierarchy', help='hierarchy, balance')
    parser.add_argument('split',default='2-hops',help='2-hops, 3-hops, all,'
                                                      'lp-500, lp-1000, lp-5000,'
                                                      'mp-500, mp-1000, mp-5000')
    parser.add_argument('--output', default=None)

    parser.add_argument('--gpu', default='0')

    parser.add_argument('--keep-ratio', type=float, default=0.1)
    parser.add_argument('--consider-trains', default=True)


    args = parser.parse_args()

    set_gpu(args.gpu)

    test_sets = json.load(open('./data/imagenet-testsets.json', 'r'))
    train_wnids = test_sets['train']
    test  = test_sets['all']

    dic = dict(zip(range(1, len(train_wnids) + len(test) + 1), train_wnids + test))
    if args.test_set == 'hierarchy':
        train_wnids = test_sets['train']
        test_wnids = test_sets[args.split]
    elif args.test_set == 'balance':
        data_path = "./ImageNet_splits.mat"
        data = scio.loadmat(data_path)
        split = data.get(args.split).tolist()  
        test_wnids = []
        for split_label in split[0]:
            test_wnids.append(dic.get(split_label))

    print('test set: {}, {} classes, ratio={}'
          .format(args.test_set, len(test_wnids), args.keep_ratio))
    print('consider train classifiers: {}'.format(args.consider_trains))

    pred_file = torch.load(args.pred)
    pred_wnids = pred_file['wnids']
    pred_vectors = pred_file['pred']
    pred_dic = dict(zip(pred_wnids, pred_vectors))
    pred_vectors = pick_vectors(pred_dic, train_wnids + test_wnids, is_tensor=True).cuda()

    pred_vectors = pred_vectors.cuda()

    n = len(train_wnids)
    m = len(test_wnids)

    cnn = make_resnet50_base()
    cnn.load_state_dict(torch.load(args.cnn))
    cnn = cnn.cuda()
    cnn.eval()

    TEST_TRAIN = args.test_train

    imagenet_path = './data/imagenet'
    dataset = ImageNet(imagenet_path)
    dataset.set_keep_ratio(args.keep_ratio)

    s_hits = torch.FloatTensor([0, 0, 0, 0, 0]).cuda()  # top 1 2 5 10 20
    s_tot = 0

    results = {}


    for i, wnid in enumerate(train_wnids, 1):
        subset = dataset.get_subset(wnid)
        hits, tot = test_on_subset(subset, cnn, n, pred_vectors, i - 1,
                                       consider_trains=args.consider_trains)
        results[wnid] = (hits / tot).tolist()

        s_hits += hits
        s_tot += tot

        print('{}/{}, {}:'.format(i, len(train_wnids), wnid), end=' ')
        for i in range(len(hits)):
                print('{:.0f}%({:.2f}%)'
                      .format(hits[i] / tot * 100, s_hits[i] / s_tot * 100), end=' ')
        print('x{}({})'.format(tot, s_tot))


    print('summary:', end=' ')
    for s_hit in s_hits:
        print('{:.2f}%'.format(s_hit / s_tot * 100), end=' ')
    print('total {}'.format(s_tot))

    if args.output is not None:
        json.dump(results, open(args.output, 'w'))