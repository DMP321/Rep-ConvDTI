# Rep_ConvDTI

The drug-protein interaction prediction model, We were inspired by this [work](http://arxiv.org/abs/2311.15599).

![Model network](https://github.com/DMP321/Rep_ConvDTI/blob/main/Figure1.jpg?raw=true)

## Requirements env
Bio——1.7.1

biopython——1.84

matplotlib——3.7.1

numpy——2.1.3

pandas——2.0.1

prefetch_generator——1.0.3

rdkit——2024.3.5

scikit_learn——1.5.2

tensorboardX——2.6.2.2

torch——2.4.0+cu118

tqdm——4.66.5

xgboost——2.1.3

## Usage
First go to the folder.

 `cd ./Rep_ConvDTI/` 
 
###	1. Training
 We provide a complete model and a convenient model training method, you can use the command to quickly start training the model：
 
 `python train.py --train_dataset your_datapath`  
 
 -----Note that the training data should be in **.csv** format and use **["protein_id", "drug_id", "AAS", "SMILE", "Label"]** as columns.
 We also provide a variety of training methods, you can use the **-h** command to learn more.
 
  `python train.py -h` 
  
 ### 2. Data process
We also provide two classic public data sets, davis and kiba, as the baseline data set, and three sample proposal methods to reduce data redundancy, namely random deletion, distance-based method and MULGA, which you can use with the following command.

 `python data_generation.py --dataset ./dataset/DAVIS --method random`
  
 The distance-based method uses a [pre-trained model](https://github.com/IBM/molformer) to measure compound data.
The MULGA referred to this [article](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btad524/7248910) .
### 3.Predicting
We provide a benchmark model that does not perform well in a single training set, but has good generalization.You can use it quickly with the following command.

 `python predict.py --predict_target ./your/target.fasta --predict_ligands ./your/ligands.sdf`
