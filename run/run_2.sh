for seed in 2000 3000
do
    for num in 2 4 5 6
    do  
        for cfg in 'configs/Mamba/HCP/age-mamba.yaml' 'configs/Mamba/HCP/gender-mamba.yaml' 'configs/Mamba/HCP/wm-mamba.yaml'
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
                --model.num_experts $num \
                --optim.max_epoch 100 > $output_file 2>&1 &
<<<<<<< HEAD
            sleep 240
=======
            sleep 150
>>>>>>> 7a4f0bef31ae730c38edbe85fe27fd8e6186af78
        done
        # pid=$!
        # wait $pid
    done
done