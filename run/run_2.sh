for cfg in 'configs/GatedGCN/wm-GCN.yaml' 'configs/GatedGCN/fi-GCN.yaml' 
do
    for base_lr in 0.05
    do  
        for wd in 0.0001 
        do
            IFS='/'
            read -ra array <<< "$cfg"
            IFS='-'
            read -ra array <<< "${array[-1]}"
            IFS=''
            info="${array[-2]}_${array[-1]}_base_lr_${base_lr}"


            echo "Start ${info}"
            output_file="log/${info}.log"
            echo $cfg
            nohup python main.py \
                --cfg $cfg \
                --seed 11000 \
                --optim.max_epoch 100 > $output_file 2>&1 &
            # pid=$!
            # wait $pid
            sleep 10
        done
    done
done




