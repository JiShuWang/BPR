## BPR：Blockchain-Enabled Efficient and Secure Parking Reservation Framework with Block Size Dynamic Adjustment Method

This repository contains the author's implementation in PyTorch of **dynamic adjustment method of block size** for the paper "BPR：Blockchain-Enabled Efficient and Secure Parking Reservation Framework with Block Size Dynamic Adjustment Method".

## Dependencies

- Python (>=3.6)
- numpy (>=1.20.3)
- pandas (>=1.2.4)
- matplotlib (>=3.4.2)
- seaborn (>=0.11.2)
- torch (>=1.7.1)
- scikit-learn (>=0.24.2)

## Implementation

Here, we provide an implementation of the **Dynamic Adjustment Method of Block Size** in BPR. The repository is organized as follows:

- **The Prediction of Transaction Send Rates**

  - `data/` contains transaction send rates dataset and block performance dataset;

  - `model/dataset.py` contains procedures for data preprocessing for transaction send rates dataset ;

  - `model/RNN.py` contains the implementation of LSTM;

  - `model/train.py` puts all of the above together and may be used to execute a full training run on transaction send rates dataset.

- **The Blockchain Performance Scoring Model**

  - `model/Blockchain Performance Model.ipynb` contains the implementation for Blockchain Performance Model.