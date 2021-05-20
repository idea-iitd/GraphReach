rm -r __pycache__/ datasets/cache/* runs/ results/
python main.py --layer_num 2  --dataset All --epoch_num 2000  --repeat_num 5   --remove_link_ratio 0.2 --deleteFedges 0.1 --model PGNN --rm_feature --AdverserialAttack



