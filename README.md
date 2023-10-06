# Structure-free Graph Condensation (SFGC): From Large-scale Graphs to Condensed Graph-free Data

This is the Pytorch implementation of NeurIPS-23 work: "Structure-free Graph Condensation (SFGC): From Large-scale Graphs to Condensed Graph-free Data".

The overall framework: 

![Fig_all-6](https://github.com/Amanda-Zheng/SFGC/assets/61812981/156af96a-4709-4f60-9d31-a9d1e1bfbbb0)


### Requirements

torch==1.7.1+cu110

torch-geometric==1.6.3

torch-sparse==0.6.9

torch-scatter==2.0.7

tensorboardx==2.6

deeprobust==0.2.4

matplotlib==3.5.3

scikit-learn==1.0.2


## Instructions

Following is the step-by-step instruction to reproduce our proposed method SFGC. 
Directly run the "4_script_test.sh" with our condensed data could reproduce our report results.

All condensed graph-free data can be found in the "logs/Distill" folder.

The large buffer files that we create as:

(1) Run to generate the buffer for keeping the model's training parameter distribution (training trajectory) in "scripts/1_shell_buffer.sh"

For examples:

Dataset: Citeseer

```
python buffer_transduct.py --lr_teacher 0.001 \
--teacher_epochs 800 --dataset citeseer   \
--num_experts=200 --wd_teacher 5e-4 --optim Adam 
```

(2) Use the coreset method to initialize the synthesized small-scale graph node features in "scripts/2_shell_coreset.sh"

For examples:

Dataset: Citeseer

```
python train_coreset.py --dataset citeseer --device cuda:0 --epochs 800 --lr 0.001 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.5 --load_npy ''
```

(3) Distill under training trajectory and coreset initialization to generate synthesized small-scale structure-free graph data in "scripts/3_script_distill.sh"

For examples:

Dataset: Citeseer

```
python distill_transduct_adj_identity2.py --config configs/config_distill.json --section citeseer-r025
```

(4) Training with the small-scale structure free graph data and test on the large-scale graph test set in "scripts/4_script_test.sh":

For example:

Dataset: Citeseer

```
python test_condg.py --config configs/config_test.json --section citeseer-r025
```

Please take note of these critical considerations that could impact your results when condensing your own datasets:

1. It's important to recognize that your buffer files, initialized differently, can potentially influence the condensation process due to the learning behavior imitation schema. To ensure an accurate trajectory imitation with well-parameterized results, please diligently monitor the condensation process with careful p/q setting.

2. The GNTK evaluation section incorporates a validation graph sampling approach to conserve memory. Alternatively, you have the option to employ a fixed full validation graph if your memory resources are sufficient.


Welcome to kindly cite our work and discuss with xin.zheng@monash.edu:
```
@inproceedings{zheng2023structure,
  title={Structure-free Graph Condensation: From Large-scale Graphs to Condensed Graph-free Data},
  author={Zheng, Xin and Zhang, Miao and Chen, Chunyang and Nguyen, Quoc Viet Hung and Zhu, Xingquan and Pan, Shirui},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
  }
```
