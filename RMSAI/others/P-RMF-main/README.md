# Proxy-Driven Robust Multimodal Sentiment Analysis with Incomplete Data
Code for the ACL 2025.

> This is a reorganized code, if you find any bugs please contact me. Thanks.

![The framework of P-RMF](imgs/Framework.png)

This paper proposes a data-centric robust multimodal sentiment analysis method to address the robustness problem under uncertain missing data. This approach learns stable unimodal representations from missing data and incorporates quantization uncertainty to guide the learning of dynamic and robust multimodal representation.

## Content
- [Note](#Note)
- [Data Preparation](#Data-preparation)
- [Environment](#Environment)
- [Training](#Training)
- [Evaluation](#Evaluation)
- [Citation](#Citation)

## Note
1Due to the regression metrics (such as MAE and Corr) and classification metrics (such as acc2 and F1) focus on different aspects of model performance. A model that achieves the lowest error in sentiment intensity prediction does not necessarily perform best in classification tasks. To comprehensively demonstrate the capabilities of the models, all the results of all models in the comparisons are selected as the best-performing checkpoint for each type of metric. This means that the classification metrics (such as acc2 and F1) and regression metrics (such as MAE and Corr) correspond to different epochs of the same training process. If you wish to compare the performance of models across different metrics at the same epoch, we recommend you rerun this code.


## Data Preparation
MOSI/MOSEI/CH-SIMS Download: Please see [MMSA](https://github.com/thuiar/MMSA)

## Environment
The basic training environment for the results in the paper is Pytorch  1.8.2, Python 3.9.19 with NVIDIA A40. 

## Training
You can quickly run the code with the following command:
```
bash train.sh
```

## Evaluation
After the training is completed, the checkpoints corresponding to the three random seeds (1111,1112,1113) can be used for evaluation. For example, evaluate the the model's binary classification accuracy in MOSI:
```
CUDA_VISIBLE_DEVICES=0 python robust_evaluation.py --config_file configs/eval_mosi.yaml --key_eval Has0_acc_2
```

## Citation

Please cite our paper if you find our work useful for your research:

```
@inproceedings{zhu-etal-2025-proxy,
    title = "Proxy-Driven Robust Multimodal Sentiment Analysis with Incomplete Data",
    author = "Zhu, Aoqiang  and Hu, Min  and Wang, Xiaohua  and Yang, Jiaoyun  and Tang, Yiming  and An, Ning",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    pages = "22123--22138",
    ISBN = "979-8-89176-251-0",
}
```

### Acknowledgement
Thanks to [LNLN](https://github.com/Haoyu-ha/LNLN) for their great help to our codes and research. 

