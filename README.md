# Control, Generate and Augment: A Scalabel Framework for Multi-Attributes Controlled Text Generation
## Introduction
PyTorch code for the EMNLP 2020 paper "Control, Generate and Augment: A Scalabel Framework for Multi-Attributes Controlled Text Generation". Slides of our EMNLP 2020 talk are avialable [here](). 



## Data Download

Please download the YELP restaurants review data from [here ](https://github.com/shentianxiao/language-style-transfer (edited)) and the IMDB 50K movie review from [here](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). The preprocessing of the data can be executed following the procedure explained in the supplementary materials of the paper "Control, Generate and Augment: A Scalabel Framework for Multi-Attributes Controlled Text Generation"
#### Data Preprocessing

To obtain the multi-attributes dataset used please run first

```bash
python TenseLabeling.py
```

and second

```bash
python PronounLabeling.py
```

## Model

#### Training

to train the model please run 

```bash
python Analysis.py
```

All the parameters to obtain the results reported in the paper are set as default values.
The model trained is saved in the bin folder. The name used is the date and the time the experiment is started

#### Generation

To generate new sentences simply run 

```bash
python generation.py
```

The default parameters for this script let generate sentences with all possible combinations of attributes. For specifically attributes, please specify the examples desired.

## Evaluation

All these scripts are in the Evalution folder

#### Data Augmentation 

For the Data Augmentation Evaluation please run 
```bash
python AugmentData.py
```
to generate all the combinations of augmented data for each of the starting training size in the paper. Afterwards run 
```bash
python GPU_DAE.py
```
to obtain the validation and test results for the data augmentation experiment.

#### Attribute Matching

please run the script
```bash
python AttrMatch.py
```
to obtain all the different attribute matching accuracy for the generated sentences

#### Sentence Embedding Similarity

```bash
python UniversalSentenceEvaluator.py
```

### Model Checkpoints
In the folder Generated you will find examples of our generated sentences, running both single and multi-attribute controls. In addition, the model checkpoints for each of these experiments are provided alongside with the parameters used for the experiments

## Reproducibility Information

#### Description of computing infrastructure used: 
All models presented in this work were implemented in PyTorch, and trained and tested on single Titan XP GPUs with 12GB memory.
#### Average runtime for each approach: 
The average runtime was 07:26:14 for the model trained with YELP.  The average runtime was 04:09:54 for the model trained with IMDB. 

#### Number of Parameters for each model



| Dataset | S-VAE (Generator) | Discriminator |
|---------|:-----------------:|:-------------:|
| YELP    |     3.417.176     |      4452     |
| IMDB    |     4.433.176     |      4470     |



