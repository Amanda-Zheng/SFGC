#@citeseer-buff
CUDA_VISIBLE_DEVICES=0 python buffer_transduct.py --lr_teacher 0.001 \
--teacher_epochs 800 --dataset citeseer   \
--num_experts=200 --wd_teacher 5e-4 --optim Adam 

#@cora-buff
CUDA_VISIBLE_DEVICES=2 python buffer_transduct.py --lr_teacher 0.4 \
--teacher_epochs 2500 --dataset cora   \
--num_experts=200 --wd_teacher 0 --optim SGD

#@ogbn-buff
CUDA_VISIBLE_DEVICES=2 python buffer_transduct.py --lr_teacher 1 \
--teacher_epochs 2000 --dataset ogbn-arxiv   \
--num_experts=200 --wd_teacher 0 --optim SGD


#@flickr-buff
CUDA_VISIBLE_DEVICES=2 python buffer_inductive.py --lr_teacher 0.001 \
--teacher_epochs 1000 --dataset flickr   \
--num_experts=200 --wd_teacher 5e-4 --optim Adam 

#@reddit-buff
CUDA_VISIBLE_DEVICES=2 python buffer_inductive.py --lr_teacher 0.001 \
--teacher_epochs 1000 --dataset reddit   \
--num_experts=200 --wd_teacher 5e-4 --optim Adam 




