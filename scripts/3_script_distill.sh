CUDA_VISIBLE_DEVICES=1 python distill_transduct_adj_identity2.py --config config_distill.json --section citeseer-r025
CUDA_VISIBLE_DEVICES=1 python distill_transduct_adj_identity2.py --config config_distill.json --section citeseer-r05
CUDA_VISIBLE_DEVICES=1 python distill_transduct_adj_identity2.py --config config_distill.json --section citeseer-r1.0
CUDA_VISIBLE_DEVICES=1 python distill_transduct_adj_identity.py --config config_distill.json --section cora-r025
CUDA_VISIBLE_DEVICES=1 python distill_transduct_adj_identity.py --config config_distill.json --section cora-r05
CUDA_VISIBLE_DEVICES=1 python distill_transduct_adj_identity.py --config config_distill.json --section cora-r1.0
CUDA_VISIBLE_DEVICES=1 python distill_transduct_adj_identity2.py --config config_distill.json --section ogbn-arxiv-r0001
CUDA_VISIBLE_DEVICES=0 python distill_transduct_adj_identity2.py --config config_distill.json --section ogbn-arxiv-r0005
CUDA_VISIBLE_DEVICES=0 python distill_transduct_adj_identity2.py --config config_distill.json --section ogbn-arxiv-r001
CUDA_VISIBLE_DEVICES=0 python distill_inductive_adj_identity.py --config config_distill.json --section flickr-r0001
CUDA_VISIBLE_DEVICES=0 python distill_inductive_adj_identity.py --config config_distill.json --section flickr-r0005
CUDA_VISIBLE_DEVICES=0 python distill_inductive_adj_identity.py --config config_distill.json --section flickr-r001
CUDA_VISIBLE_DEVICES=0 python distill_inductive_adj_identity.py --config config_distill.json --section reddit-r00005
CUDA_VISIBLE_DEVICES=0 python distill_inductive_adj_identity.py --config config_distill.json --section reddit-r0001
CUDA_VISIBLE_DEVICES=0 python distill_inductive_adj_identity.py --config config_distill.json --section reddit-r0005

