import json
import torch
from glove import GloVe
from nltk.corpus import wordnet as wn
import numpy as np
import time
from evaluate import*

def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))


if torch.cuda.is_available():
    device = torch.device('cuda:1')
    torch.cuda.set_rng_state(torch.cuda.get_rng_state())
    torch.backends.cudnn.deterministic = True
else:
    device = torch.device('cpu')

edges = []
count = []

test_sets = json.load(open('./data/imagenet-testsets.json', 'r'))
train_wnids = test_sets['train']
test_wnids = test_sets['all']
wnids = train_wnids + test_wnids
print('making glove embedding ...')
glove = GloVe('glove.6B.300d.txt')

vectors = []
for e in wnids:
    vectors.append(glove[getnode(e).lemma_names()])
vectors = torch.stack(vectors)
dic = dict(zip(range(len(wnids)),wnids))

since = time.time()

for i, feat in enumerate(vectors):
    print(i)
    table = torch.cosine_similarity(feat, vectors)
    value, idx = torch.sort(table, descending=True)
    idx = idx[0:30].tolist()
    value  = value[0:30]
    print(idx)
    id_rank = dict(zip(range(len(idx)),idx))

    wn_similar = []
    visual = []
    for k,rank in enumerate(idx):
        wnid = dic.get(i)
        syn = getnode(wnid)
        cos =  syn.wup_similarity(getnode(dic.get(rank)))
        wn_similar.append(cos)
    wn_similar = np.array(wn_similar)
    value = np.array(value)
    cos = torch.tensor(wn_similar* 0.1 + value)
    v,d = torch.sort(torch.tensor(wn_similar), descending=True)
    d = d.tolist()

    for dst in d[:11]:
        dst = id_rank.get(dst)
        ed = (i, dst)
        re_ed = (dst, i)
        if ed in count or re_ed in count or i == dst:
            continue
        edges.append(torch.LongTensor([i, dst]))
        count.append(ed)
        count.append(re_ed)

edges = torch.stack(edges, 1)
print(edges.size())
print(edges)
json.dump(edges.tolist(), open('similar-10.json', 'w'))

time_elapsed = time.time() - since
print('time {:0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))



