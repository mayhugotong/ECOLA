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

In the github repository, we only upload DuEE, partial Wiki and partial Gdelt (short version with 1000 samples).
Full datasets are avaible in the following link https://drive.google.com/file/d/1gu1ElWtK8ObnrlGhqlFs0ng9vue_2xra/view?usp=drive_link.

The training_data.json file has the format as the following example:

    {"token_ids": [101, 2006, 9317, 1010, 1996, 2343, 2097, 4088, 2010, 2880, 5704, 2012, 4830, 19862, 1010, 5288, 1010, 2073, 2002, 2097, 2907, 6295, 2007, 3010, 3539, 2704, 6796, 19817, 27627, 1010, 3035, 20446, 2050, 1062, 2953, 2890, 25698, 12928, 1998, 1996, 3539, 2704, 2928, 21766, 4674, 1997, 4549, 1010, 1998, 5364, 2343, 15654, 2022, 22573, 2102, 1012, 102, 2, 163, 19], "tuple": [2, 19, 163, 31695]}


Besides, we also provide the plain textual descriptions before tokeniztion in DuEE/quadruple_with_text_train.txt and GDELT/quadruple_with_text_train(corpus_day_01).json to give the readers a more clear understanding of the datasets.


