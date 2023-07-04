# BPR：Blockchain-Enabled Efficient and Secure Parking Reservation Framework with Block Size Dynamic Adjustment Method

This repository contains the author's implementation in PyTorch of **dynamic adjustment method of block size** for the paper "BPR：Blockchain-Enabled Efficient and Secure Parking Reservation Framework with Block Size Dynamic Adjustment Method".

## Address
https://ieeexplore.ieee.org/document/9961087
## Cite
J. Wang et al., "BPR: Blockchain-Enabled Efficient and Secure Parking Reservation Framework With Block Size Dynamic Adjustment Method," in IEEE Transactions on Intelligent Transportation Systems, vol. 24, no. 3, pp. 3555-3570, March 2023, doi: 10.1109/TITS.2022.3222960.
## Or
@ARTICLE{9961087,
  author={Wang, Jishu and Zhu, Chao and Miao, Chen and Zhu, Rui and Zhang, Xuan and Tang, Yahui and Huang, Hexiang and Gao, Chen},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={BPR: Blockchain-Enabled Efficient and Secure Parking Reservation Framework With Block Size Dynamic Adjustment Method}, 
  year={2023},
  volume={24},
  number={3},
  pages={3555-3570},
  doi={10.1109/TITS.2022.3222960}}

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
