for seed in 4000 5000
do
    for base_lr in 0.05
    do  
        for cfg in 'configs/GPS/brain-GPS.yaml' 'configs/Mamba/HCP/brain-mamba.yaml' 'configs/Mamba/HCP/age-mamba.yaml' 'configs/Mamba/HCP/wm-mamba.yaml' 
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
                --optim.max_epoch 100 > $output_file 2>&1 &
            sleep 60
        done
        pid=$!
        wait $pid
    done
done

