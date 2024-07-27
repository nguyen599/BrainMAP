# !/bin/bash


# node
for cfg in configs/Exphormer/brain_Exp.yaml configs/GPS/brain_1.yaml configs/Mamba_ablation/brain_mamba_1.yaml
do
    for b in Nan
    do
        for c in Nan 
        do
            for d in Nan
            do 
                for e in Nan
                do
                    # Function to get GPU utilization for a given GPU ID
                    get_gpu_load() {
                        gpu_id=$1
                        load=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu_id)
                        echo "$load"
                    }

                    # Function to choose the GPU with the least load
                    choose_gpu_with_least_load() {
                        gpu_count=$(nvidia-smi --list-gpus | wc -l)
                        if [ $gpu_count -eq 0 ]; then
                            echo "No GPUs available."
                            exit 1
                        fi

                        # Initialize variables
                        min_load=100
                        chosen_gpu=""

                        # Loop through available GPUs
                        for ((gpu_id = 0; gpu_id < $gpu_count; gpu_id++)); do
                            load=$(get_gpu_load $gpu_id)
                            if [ -z "$load" ]; then
                                continue
                            fi

                            if ((load < min_load)); then
                                min_load=$load
                                chosen_gpu=$gpu_id
                            fi
                        done

                        echo "$chosen_gpu"
                    }

                    # Choose GPU with the least load
                    chosen_gpu=$(choose_gpu_with_least_load)

                    if [ -z "$chosen_gpu" ]; then
                        echo "No available GPUs or unable to determine GPU load."
                        exit 1
                    fi


                    echo "Selected GPU: $chosen_gpu"

                    # Set the CUDA_VISIBLE_DEVICES environment variable to restrict execution to the chosen GPU
                    # export CUDA_VISIBLE_DEVICES=$chosen_gpu

                    IFS=$'/' read -ra foo <<< "$cfg"

                    info="Config: ${cfg}"

                    echo "Start ${info}"
                    output_file=""logs/${foo[-2]}_${foo[-1]}.txt""
                    declare -p foo

                    nohup python main.py \
                        --cfg $cfg > $output_file 2>&1 &
                    pid=$!
                    sleep 400
                done
            done
        done
    done
done