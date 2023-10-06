#@cora-r025
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset cora --device cuda:0 --epochs 1200 --lr 0.01 --lr_coreset 0.01 \
--weight_decay 5e-4 --wd_coreset 1e-4  --save 1 --method kcenter --reduction_rate 0.25 \
--load_npy '' --runs 1

#@cora-r05
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset cora --device cuda:0 --epochs 1 --lr 0.01 --lr_coreset 0.005 \
--weight_decay 5e-4 --wd_coreset 5e-4 --save 1 --method kcenter --reduction_rate 0.5

#@cora-r1
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset cora --device cuda:0 --epochs 600 --lr 0.01 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 1 --load_npy ''

#@obgn-r0001
CUDA_VISIBLE_DEVICES=2 python train_coreset.py --dataset ogbn-arxiv --device cuda:0 --epochs 1000 --lr 0.01 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.001 --load_npy ''

#@obgn-r0005
CUDA_VISIBLE_DEVICES=2 python train_coreset.py --dataset ogbn-arxiv --device cuda:0 --epochs 1000 --lr 0.01 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.005 --load_npy ''

#@obgn-r001
CUDA_VISIBLE_DEVICES=3 python train_coreset.py --dataset ogbn-arxiv --device cuda:0 --epochs 1000 --lr 0.01 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.01 --load_npy ''

#@citeseer-r05
CUDA_VISIBLE_DEVICES=3 python train_coreset.py --dataset citeseer --device cuda:0 --epochs 800 --lr 0.001 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.5 --load_npy ''


#@citeseer-r025
CUDA_VISIBLE_DEVICES=3 python train_coreset.py --dataset citeseer --device cuda:0 --epochs 800 --lr 0.001 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.25 --load_npy ''

#@citeseer-r1
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset citeseer --device cuda:0 --epochs 800 --lr 0.001 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 1 --load_npy ''

#@flickr-r0001
CUDA_VISIBLE_DEVICES=1 python train_coreset_inductive.py --dataset flickr --device cuda:0 --epochs 1000 --lr 0.001 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.001 --load_npy ''

#@flickr-r0005
CUDA_VISIBLE_DEVICES=1 python train_coreset_inductive.py --dataset flickr --device cuda:0 --epochs 800 --lr 0.001 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.005 --load_npy ''

#@flickr-r001
CUDA_VISIBLE_DEVICES=1 python train_coreset_inductive.py --dataset flickr --device cuda:0 --epochs 800 --lr 0.001 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.01 --load_npy ''


#@reddit-r00005
CUDA_VISIBLE_DEVICES=3 python train_coreset_inductive.py --dataset reddit --device cuda:0 --epochs 1000 --lr 0.001 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.0005 --load_npy ''

#@reddit-r0001
CUDA_VISIBLE_DEVICES=3 python train_coreset_inductive.py --dataset reddit --device cuda:0 --epochs 800 --lr 0.001 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.001 --load_npy ''

#@reddit-r0002
CUDA_VISIBLE_DEVICES=3 python train_coreset_inductive.py --dataset reddit --device cuda:0 --epochs 1000 --lr 0.001 \
--weight_decay 0.0005  --save 1 --method kcenter --reduction_rate 0.002 --load_npy ''

#@reddit-r0005
CUDA_VISIBLE_DEVICES=3 python train_coreset_inductive.py --dataset reddit --device cuda:0 --epochs 1000 --lr 0.001 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.005 --load_npy ''







