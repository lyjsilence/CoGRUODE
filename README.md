# CoGRUODE
Dependence Learning of Multivariate Asynchronous Time Series

### Requirements
Python == 3.8.   
Pytorch: 1.8.1+cu102, Sklearn:0.23.2, Numpy: 1.19.2, Pandas: 1.1.3, Matplotlib: 3.3.2   
All the codes are run on GPUs by default. 

### LSST Experiments
LSST dataset can be downloaded from 

http://www.timeseriesclassification.com/Downloads/LSST.zip

To run LSST experiment, first run LSST_pre to prepare the dataset with missing rate 0.25, 0.5, and 0.75.

Train the LSST dataset
```
python3 LSST.py --model_name CoGRUODE-HV --num_exp 5 --n_dim 20 --batch_size 256 --dt 0.02 --missing_rate 0.5
python3 LSST.py --model_name CoGRUODE-HM --num_exp 5 --n_dim 20 --batch_size 256 --dt 0.02 --missing_rate 0.5
```
### Human Activity Experiments
Human Activity dataset can be downloaded from 

https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt

Train the Activity experiment
```
python3 Activity.py --model_name CoGRUODE-HV --num_exp 5 --n_dim 20 --batch_size 64 --dropout 0.3 --dt 0.01
python3 Activity.py --model_name CoGRUODE-HM --num_exp 5 --n_dim 20 --batch_size 64 --dropout 0.3 --dt 0.01
```
### Physionet Experiments
Physionet dataset contains three files, and each one consists of data from 4,000 patients.   
This three files can be downloaded from 

https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz

https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz

https://physionet.org/files/challenge-2012/1.0.0/set-c.tar.gz

Train the Physionet experiment
```
python3 Physionet.py --model_name CoGRUODE-HV --num_exp 5 --n_dim 20 --batch_size 500 --dt 0.1
python3 Physionet.py --model_name CoGRUODE-HM --num_exp 5 --n_dim 20 --batch_size 500 --dt 0.1
```

