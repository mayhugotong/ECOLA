# Enhanced Temporal Knowledge Embeddings with Contextualized Language Representations

This repository is the official implementation of **Enhancing Temporal Knowledge Embeddings with Contextualized Language Representations**. 

## Requirements

To install environment:

```setup
conda env create --file ecola_env.yml
```

## Training
The training commands of ECOLA with different TKG embedding models on different datasets can be seen in train.sh.
To specify dataset and TKG embedding model:

    --dataset: name of loaded dataset, choices=['GDELT', 'DuEE', 'Wiki']
    --tkg_type: name of enhanced tkg embedding model', choices=['DE', 'UTEE','DyERNIE']
    --data_dir: specify the location of dataset
    --entity_dic_file: specify the location of entity dictionary file
    --relation_dic_file: specify the location of relation dictionary file

We adapt three existing datasets for training ECOLA:

GDELT: https://www.gdeltproject.org/data.html#googlebigquery

DuEE: https://ai.baidu.com/broad/download

Wiki: https://www.wikidata.org/wiki/Wikidata:MainPage 


The uploaded datasets are orgnized in following structure: 

Dataset (DuEE/WiKi_short/GDELT_short)
 - entities.txt (indexed entity ids)
 - relations.txt  (indexed relation ids)
 - val.txt  (quadruples for validation)
 - test.txt  (quadruples for test)
 - training_data.json  (quadruples for training with aligned tokenized textual decriptions.)

In the github repository, DuEE, partial Wiki and partial Gdelt (short version with 1000 samples) are uploaded for fast preview.
Besides, we also provide the plain textual descriptions before tokeniztion in DuEE/quadruple_with_text_train.txt and GDELT/quadruple_with_text_train(corpus_day_01).json to give the readers a more clear understanding of the datasets.
Full datasets are avaible in the following link https://drive.google.com/file/d/1gu1ElWtK8ObnrlGhqlFs0ng9vue_2xra/view?usp=drive_link.

The training_data.json file has the format as the following example:

    {"token_ids": [101, 2006, 9317, 1010, 1996, 2343, 2097, 4088, 2010, 2880, 5704, 2012, 4830, 19862, 1010, 5288, 1010, 2073, 2002, 2097, 2907, 6295, 2007, 3010, 3539, 2704, 6796, 19817, 27627, 1010, 3035, 20446, 2050, 1062, 2953, 2890, 25698, 12928, 1998, 1996, 3539, 2704, 2928, 21766, 4674, 1997, 4549, 1010, 1998, 5364, 2343, 15654, 2022, 22573, 2102, 1012, 102, 2, 163, 19], "tuple": [2, 19, 163, 31695]}







## Results

Our model (ECOLA) and baselines achieve the following results on Temporal Knowledge Graph Completion task:
| Dataset                | GDELT       |             |             |             |  Wiki       |             |             |             | DuEE        |             |             |             |
|------------------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| Model                  | MRR         | Hits@1      | Hits@3      | Hits@10     | MRR         | Hits@1      | Hits@3      | Hits@10     | MRR         | Hits@1      | Hits@3      | Hits@10     |
| TransE                 | 8.08        | 0.00        | 8.33        | 25.33       | 27.25       | 16.09       | 33.06       | 48.24       | 34.25       | 4.45        | 60.73       | 80.97       |
| SimplE                 | 10.98       | 4.76        | 10.49       | 23.67       | 20.75       | 16.77       | 23.23       | 27.62       | 51.13       | 40.69       | 58.30       | 68.62       |
| DistMult               | 11.27       | 4.86        | 10.87       | 24.47       | 21.40       | 17.54       | 23.86       | 28.15       | 48.58       | 38.26       | 55.26       | 65.58       |
| TeRO                   | 6.59        | 1.75        | 5.86        | 15.58       | 32.92       | 21.74       | 39.12       | 53.45       | 54.29       | 39.27       | 63.16       | 85.02       |
| ATiSE                  | 7.00        | 2.48        | 6.26        | 14.61       | 35.36       | 24.07       | 41.69       | 54.74       | 53.79       | 42.31       | 59.92       | 75.91       |
| TNTComplEx             | 8.93        | 3.60        | 8.52        | 19.01       | 34.36       | 22.38       | 40.64       | 56.03       | 57.56       | 43.52       | 65.99       | 83.60       |
| TTransE                | 11.48       | 4.72        | 11.18       | 25.25       | 30.88       | 20.16       | 35.27       | 53.08       | 61.63       | 48.58       | 69.64       | 85.63       |
| DE-SimplE              | 12.25       | 5.33        | 12.29       | 26.64       | 42.12       | 34.03       | 45.23       | 58.86       | 58.86       | 44.74       | 68.62       | 86.84       |
| ECOLA-SF               | 14.44       | 5.11        | 20.32       | 26.40       | 42.28       | 35.22       | 44.88       | 56.27       | 60.64       | 46.96       | 69.64       | 87.45       |
| ECOLA-DE               | 19.67 &pm;  | 16.04 &pm;  | 19.50 &pm;  | 25.58 &pm;  | 43.53 &pm;  | 35.78 &pm;  | 46.42 &pm;  | 60.26 &pm;  | 60.78 &pm;  | 47.43 &pm;  | 69.43 &pm;  | 86.70 &pm;  |
|                        | 00.11       | 00.19       | 00.04       | 00.03       | 00.08       | 00.17       | 00.02       | 00.04       | 00.16       | 00.13       | 00.64       | 00.17       |
| UTEE                   | 9.76        | 4.23        | 9.77        | 21.29       | 26.96       | 20.98       | 30.39       | 37.57       | 53.36       | 43.92       | 60.52       | 68.62       |
| ECOLA-UTEE             | 19.11 &pm;  | 15.29 &pm;  | 19.46 &pm;  | 25.59 &pm;  | 38.35 &pm;  | 30.56 &pm;  | 42.11 &pm;  | 53.02 &pm;  | 60.36 &pm;  | 46.55 &pm;  | 69.22 &pm;  | 87.11 &pm;  |
|                        | 00.16       | 00.38       | 00.05       | 00.09       | 00.22       | 00.18       | 00.14       | 00.41       | 00.36       | 00.51       | 00.93       | 00.07       |
| DyERNIE                | 10.72       | 4.24        | 10.81       | 24.00       | 23.51       | 14.53       | 25.21       | 41.67       | 57.58       | 41.49       | 70.24       | 86.23       |
| ECOLA-DyERNIE          | 19.99 &pm;  | 16.40 &pm;  | 19.78 &pm;  | 25.67 &pm;  | 41.22 &pm;  | 33.02 &pm;  | 45.00 &pm;  | 57.17 &pm;  | 59.64 &pm;  | 46.35 &pm;  | 67.87 &pm;  | 85.48 &pm;  |
|                        | 00.05       | 00.09       | 00.03       | 00.04       | 00.06       | 00.27       | 00.20       | 00.32       | 00.18       | 00.53       | 00.29       | 00.35       |


