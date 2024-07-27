# !/bin/bash

python main.py --cfg configs/GPS/fi-GPS.yaml &
sleep 10
python main.py --cfg configs/GPS/wm-GPS.yaml &
sleep 10
python main.py --cfg configs/GPS/age-GPS.yaml &

wait

# python main.py --cfg configs/GatedGCN/fi-Sage.yaml &
# sleep 10
# python main.py --cfg configs/GatedGCN/wm-Sage.yaml &
# sleep 10
# python main.py --cfg configs/GatedGCN/age-Sage.yaml &

# wait 1

