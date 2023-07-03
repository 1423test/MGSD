# MGSD
Multi-view Graph Representation with Similarity Diffusion for General Zero-Shot Learning

### Data Preparation
#### ImageNet and part visual features 

1. Download ImageNet.

An ImageNet root directory should contain image folders, each folder with the wordnet id of the class.

#### Glove Word Embedding
1. Download: http://nlp.stanford.edu/data/glove.6B.zip
2. Unzip it, find and put `glove.6B.300d.txt` to `graph/`.

#### Multi-view Graph and Similarity Graph
1. `cd graph/`
2. Run `graph.py`, and get `multi_view_graph.json
3. Run `similar.py`, and get `similar-10.json

#### Pretrained ResNet50
1. Download: https://figshare.com/articles/dataset/data_rar/20342646, get `fc-weights.json` and `resnet50-base.pth`
2. Put files :`cd visual`
3. Run 'python resnet_process.py' get visual features of imagenet in `datasets/imagenet`

#### Train Graph Networks
Run `python train.py`, and get `F.pred` results for testing.

### Testing （General and detailed test）
Run `python evaluate.py` with the args:

* `--pred`: the `.pred` file for testing. 
* `--test-set`: choose test set, choices: `[Hierarchy,  Most Populated, Least Populated, All]`
* `--split`: choose test set, choices: `['2-hops, 3-hops, all,'
                                        'lp-500, lp-1000, lp-5000,'
                                        'mp-500, mp-1000, mp-5000']`
* (optional) `--keep-ratio` for the ratio of testing data, `--consider-trains` to include training classes' classifiers, `--test-train` for testing with train class images only.

