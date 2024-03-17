# Codes for Knowledge Graph Completion
The codes are based on [MuRP](https://github.com/ibalazevic/multirelational-poincare) repo. 

 ```run
32-dimension

python main.py --dataset WN18RR --num_epochs 1000 --batch_size 1000 --nneg 100 --lr 0.005 --margin 8.0 --max_norm 1.5 --max_scale 3.5 --max_grad_norm 1.0 --dim 32 --valid_steps 10 --early_stop 10 --cuda True --real_neg --optimizer radam
python main.py --dataset FB15k-237 --num_epochs 500 --batch_size 500 --nneg 50 --lr 0.005 --margin 8.0 --max_norm 1.5 --max_scale 2.5 --max_grad_norm 1.0 --dim 32 --valid_steps 10 --early_stop 10 --cuda True --optimizer radam --real_neg
python main.py --dataset Nations --num_epochs 500 --batch_size 128 --nneg 10 --lr 0.005 --margin 8.0 --max_norm 1.5 --max_scale 3.5 --max_grad_norm 1.0 --dim 32 --valid_steps 10 --early_stop 10 --cuda True --real_neg --optimizer radam
python main.py --dataset CoDEx-s --num_epochs 500 --batch_size 128 --nneg 10 --lr 0.005 --margin 8.0 --max_norm 1.5 --max_scale 3.5 --max_grad_norm 1.0 --dim 64 --valid_steps 10 --early_stop 10 --cuda True --real_neg --optimizer radam
python main.py --dataset CoDEx-m --num_epochs 500 --batch_size 128 --nneg 10 --lr 0.005 --margin 8.0 --max_norm 1.5 --max_scale 3.5 --max_grad_norm 1.0 --dim 64 --valid_steps 10 --early_stop 10 --cuda True --real_neg --optimizer radam

500-dimension
python main.py --dataset WN18RR --num_epochs 1000 --batch_size 1000 --nneg 150 --lr 0.005 --margin 8.0 --max_norm 1.5 --max_scale 3.5 --max_grad_norm 1.0 --dim 500 --valid_steps 10 --early_stop 10 --cuda True --real_neg --optimizer radam 
python main.py --dataset FB15k-237 --num_epochs 500 --batch_size 500 --nneg 50 --lr 0.005 --margin 8.0 --max_norm 1.5 --max_scale 2.5 --max_grad_norm 1.0 --dim 500 --valid_steps 10 --early_stop 10 --cuda True --optimizer radam --real_neg
 ```
