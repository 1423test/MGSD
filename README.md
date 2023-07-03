# MGSD
Multi-view Graph Representation with Similarity Diffusion for General Zero-Shot Learning

## Requirements

* python 3.7
* pytorch 1.6.0
* nltk 3.4

### Data Preparation
#### ImageNet and part visual features 

1. Download ImageNet.

An ImageNet root directory should contain image folders, each folder with the wordnet id of the class.

#### Glove Word Embedding
1. Download: http://nlp.stanford.edu/data/glove.6B.zip
2. Unzip it, find and put `glove.6B.300d.txt` to `graph/`.

#### Semantic-Visual Shared Knowledge Graph
1. `cd graph/`
2. Run `svkg.py`, get `svkg.json`
3. Else download `svkg.json` from https://figshare.com/articles/dataset/data_rar/20342646

#### Pretrained ResNet50
1. Download: https://figshare.com/articles/dataset/data_rar/20342646, get `fc-weights.json` and `resnet50-base.pth`
2. Put files :`cd visual`
3. Run 'python resnet_process.py' get visual features of imagenet in `datasets/imagenet`

#### Train Graph Networks
Run `python train.py`, and get `baseline/svkg-1000.pred` results. Else download pre-trained model from https://figshare.com/articles/dataset/data_rar/20342646 for testing.

### Testing （General and detailed test）
Run `python test.py` with the args:

* `--pred`: the `.pred` file for testing. 
* `--test-set`: choose test set, choices: `[general, detailed]`
* `--split`: choose test set, choices: `[2-hops, 3-hops, all, bird, dog, snake, monkey]`
* (optional) `--keep-ratio` for the ratio of testing data, `--consider-trains` to include training classes' classifiers, `--test-train` for testing with train class images only.

