# CoGRUODE
Learning Multivariate Asynchronous Time Series for High-Frequency Option Price Forecasting

![Image text](https://github.com/lyjsilence/CoGRUODE/tree/main/img/eval_example.png)

### Requirements
Python == 3.8.   
Pytorch: 1.8.1+cu102, torchdiffeq: 0.2.2, Sklearn:0.23.2, Numpy: 1.19.2, Pandas: 1.1.3, Matplotlib: 3.3.2   
All the codes are run on GPUs by default. 

The data is stored in the folder "data", including 3-min transaction records for the Call option and Put option of Tencent security from 01/Dec./2014 to 31/Dec/2017.
   
### Financial Transactions (Tencent Stock Options) Experiments

Training
```
python3 TCH.py --dataset Call --model_name RFN --num_exp 5 
```

The CoGRUODE in model_name can be replaced by one of the baselines: GRUODE, mGRUODE, ODELSTM, ODERNN, GRU-D, GRU-delta-t.
The dataset can be selected from the Call option (Call) or Put option (Put).


