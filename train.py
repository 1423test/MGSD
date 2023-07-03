import json
import random
import torch.nn.functional as F
import os
import os.path as osp
import shutil
import torch_geometric.transforms as T
import argparse
import json
from nltk.corpus import wordnet as wn
import torch
from ordered_set import OrderedSet
from torch_geometric.data import HeteroData,Data
from tools import ensure_path,save_checkpoint,mask_l2_loss,set_gpu,pick_vectors
from models import heto,diffusion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-epoch', type=int, default=3000)
    parser.add_argument('--trainval', default='10,0')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight-decay', type=float, default=0.00001)
    parser.add_argument('--save-epoch', type=int, default=3000)
    parser.add_argument('--save-path', default='models')

    parser.add_argument('--gpu', default='0')

    parser.add_argument('--no-pred', action='store_true')
    parser.add_argument('--use-gdc', default = True)
    parser.add_argument('--use-multi-aggregators', default=True)
    args = parser.parse_args()


    device = set_gpu(args.gpu)
    save_path = args.save_path
    ensure_path(save_path)

    graph = json.load(open('./graph/mulit_view_graph', 'r'))
    edges = torch.tensor(graph['edges']).to(device)
    wnids = graph['wnids']
    word_vectors = torch.tensor(graph['vectors']).to(device)
    word_vectors = F.normalize(word_vectors)


    ent2id = {ent: idx for idx, ent in enumerate(wnids)}  
    data = {}
    with open('./dataset/relation.txt') as f:
        for line in f.readlines():
            sub, rel, obj = map(str.lower, line.strip().split('\t'))
            sub, obj = ent2id[sub], ent2id[obj]
            data.setdefault(rel, []).append(torch.LongTensor([sub, obj]))

    graph =HeteroData()
    graph['nodes'].x = word_vectors
    graph['nodes', 'hy', 'nodes'].edge_index = edges
    transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])
    graph = transform(graph)
    for rel in data:
        graph['nodes', rel, 'nodes'].edge_index = torch.stack(data.get(rel), 1).to(device)
    print(graph)


    fcfile = json.load(open('./visual/fc-weights.json', 'r'))
    train_wnids = [x[0] for x in fcfile]
    fc_vectors = [x[1] for x in fcfile]
    fc_vectors = torch.tensor(fc_vectors).to(device)
    fc_vectors = F.normalize(fc_vectors)

    hagnn = heto(graph.metadata(),300,2048,2049).to(device)
    optimizer = torch.optim.Adam(hgt.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    v_train, v_val = map(float, args.trainval.split(','))
    n_trainval = len(fc_vectors)
    n_train = round(n_trainval * (v_train / (v_train + v_val)))
    print('num train: {}, num val: {}'.format(n_train, n_trainval - n_train))
    tlist = list(range(len(fc_vectors)))
    random.shuffle(tlist)

    for epoch in range(1, args.max_epoch + 1):
        hagnn.train()
        z1 = hagnn(graph.x_dict,graph.edge_index_dict)
        loss  = mask_l2_loss(z1, fc_vectors, tlist[:n_train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        hgt.eval()
        with torch.no_grad():
            h = hagnn(graph.x_dict,graph.edge_index_dict)
            train_loss = mask_l2_loss(z1, fc_vectors, tlist[:n_train]).item()

        print('epoch {}, loss={:.4f}'.format(epoch, train_loss))

        if (epoch % args.save_epoch == 0):
            if args.no_pred:
                pred_obj = None
            else:
                pred_obj = {
                    'wnids': wnids,
                    'pred': h
                }
            save_checkpoint(pred_obj,save_path,'hagnn'.format(epoch))

    with torch.no_grad():
        h = hagnn(graph.x_dict, graph.edge_index_dict)

    test_sets = json.load(open('./data/imagenet-testsets.json', 'r'))
    train_wnids = test_sets['train']
    test_wnids = test_sets['all']
    pred_vectors = h['pred'].clone()
    pred_wnids = h['wnids']
    pred_dic = dict(zip(pred_wnids, pred_vectors))
    h = pick_vectors(pred_dic, train_wnids + test_wnids, is_tensor=True)

    ed = json.load(open('./graph/similar-10.json', 'r'))
    ed = torch.tensor(ed).to(device)
    graph_simi = Data()
    graph_simi.x = h
    graph_simi.edge_index = ed
    transform = T.Compose([T.ToUndirected(),T.AddSelfLoops()])
    graph_simi = transform(graph_simi).to(device)
    print(graph_simi)

    lambda_ = 0.28
    beta_ = 0.44
    dif = diffusion( diffusion_layer = 20, lambda_ = lambda_,
                     noise=True,device=device)

    f = dif.augment(graph_simi.x, fc_vectors, tlist, graph_simi.edge_index,beta_)

    hidden_agnn = {'wnids':train_wnids+test_wnids,'pred': f}
    torch.save(hidden_agnn, osp.join(save_path,  "{}.pred".format('F')))