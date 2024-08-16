for seed in 0 1000
do
    for num in 6
    do  
        for cfg in 'configs/GatedGCN/brain-Sage.yaml' 'configs/GatedGCN/brain-GCN.yaml' 
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
                --seed $seed \
                # --model.num_experts $num \
                --optim.max_epoch 100 > $output_file 2>&1 &
            sleep 30
        done
    done
done