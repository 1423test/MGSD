## Requirements

* python 3.8
* pytorch 1.11.0
* nltk 3.4
* PyG 2.0.4

### Data Preparation
#### ImageNet and part visual features 

1. Download ImageNet, and put the dataset in the dictionary `data/imagenet`
An ImageNet root directory should contain image folders, each folder with the wordnet id of the class.

#### Glove Word Embedding
1. Download: http://nlp.stanford.edu/data/glove.6B.zip
2. Unzip it, find and put `glove.6B.300d.txt` to `graph/`.

#### Multi-view Graph
1. `cd graph/`
2. Run `graph.py`, get `mulit_view_graph.json`

#### Similarity Diffusion Matrix
1. `cd graph/`
2. Run `similar.py`, get `similar-10.json`

#### Pretrained ResNet50
1. Download: https://download.pytorch.org/models/resnet50-19c8e357.pth, get `fc-weights.json` and `resnet50-base.pth`
2. Put files in the dictionary `visual`

#### Train Graph Networks
Run `python train.py`, get results in `models/f.pred`. Else download pretrained model from https://drive.google.com/drive/folders/1xq6gUnonGjdG9Qr4F-rp5ld056LKCNrT?usp=sharing for testing.

### Testing （General and detailed test）
Run `python evalute.py` with the args:

* `--pred`: the `.pred` file for testing. 
* `--cnn`: path to resnet50 weights: `materials/resnet50-base.pth`
* `--test-set`: choose test set, choices: `[hierarchy, balance]`
* `--split`: choose test set, choices: `[2-hops, 3-hops, all, '
                                       'lp-500, lp-1000, lp-5000,'
                                       'mp-500, mp-1000, mp-5000']`
* (optional) `--keep-ratio` for the ratio of testing data, `--consider-trains` to include training classes' classifiers.
