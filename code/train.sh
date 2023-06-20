# sh config
####### GDELT #######
# DE
python3 run_e2e_train.py --tkg_type DE --dataset 'GDELT' --data_dir '../data/GDELT' --entity_dic_file '../data/GDELT/entities2id.txt' --relation_dic_file '../data/GDELT/relations2id.txt'
# UTEE
python3 run_e2e_train.py --tkg_type UTEE --num_train_epochs 5 --warm_up 0.5 --dataset 'GDELT' --data_dir '../data/GDELT' --entity_dic_file '../data/GDELT/entities2id.txt' --relation_dic_file '../data/GDELT/relations2id.txt'
# DyERNIE
python3 run_e2e_train.py --tkg_type DyERNIE --num_train_epochs 4 --dataset 'GDELT' --data_dir '../data/GDELT' --entity_dic_file '../data/GDELT/entities2id.txt' --relation_dic_file '../data/GDELT/relations2id.txt'

####### WIKI #######
# DE
python3 run_e2e_train.py --learning_rate 0.00001 --tkg_type DE --dataset 'Wiki' --data_dir '../data/Wiki' --entity_dic_file '../data/Wiki/entities2id.txt' --relation_dic_file '../data/Wiki/relations2id.txt' --num_train_epoch 10
# UTEE
python3 run_e2e_train.py --learning_rate 0.0002  --tkg_type UTEE --dataset 'Wiki' --data_dir '../data/Wiki' --entity_dic_file '../data/Wiki/entities2id.txt' --relation_dic_file '../data/Wiki/relations2id.txt' --num_train_epoch 20
# DyERNIE
python3 run_e2e_train.py --learning_rate 0.0002  --tkg_type DyERNIE --dataset 'Wiki' --data_dir '../data/Wiki' --entity_dic_file '../data/Wiki/entities2id.txt' --relation_dic_file '../data/Wiki/relations2id.txt' --num_train_epoch 25

####### DUEE ######
# DE
python3 run_e2e_train.py --tkg_type DE --warm_up 0.3 --warm_up 0.2 --learning_rate 0.0005 --train_batch_size 8 --num_train_epoch 1000
# UTEE
python3 run_e2e_train.py --tkg_type UTEE --train_batch_size 8 --num_train_epoch 1000 
# DyERNIE
python3 run_e2e_train.py --tkg_type DyERNIE --train_batch_size 8 --num_train_epoch 1000 