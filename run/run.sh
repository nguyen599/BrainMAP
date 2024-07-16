for cfg in 'configs/Mamba/HCP/age-mamba.yaml'
do
    for base_lr in 0.1 0.05 0.01 0.001
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
                --optim.weight_decay 0.001\
                --optim.base_lr $base_lr > $output_file 2>&1 &
                
            # pid=$!
            # wait $pid
            sleep 60
        done
    done
done




