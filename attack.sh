#!/bin/bash

#SBATCH --job-name=attack
#SBATCH --time=10-00:00:00  # days-hh:mm:ss
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=40000  # megabytes
#SBATCH -o attack.log

eval "$(conda shell.bash hook)"
conda activate MedViT

:'
# Run the full attack for all convolutional models and datasets with epsilon 2 and 4.
for epsilon in 2 4; do
    for model_name in MedVit densenet121 resnet50 vgg16 convnext_base; do
        for data_info in "dermamnist_224 7" "octmnist_224 4" "pneumoniamnist_224 2"; do
            read -r data_name n_classes <<< "$data_info"
            
            for alg_info in "IntegratedGradients 6" "Saliency 2" "DeepLift 2" "InputXGradient 6" "GradCAM 1" "SmoothGrad 1"; do
                read -r algorithm batch <<< "$alg_info"
                
                echo "Running attack for $model_name on $data_name with algorithm $algorithm and ε=$epsilon..."

                # Conditionally set the NEW_PROCESS flag
                if [[ "$algorithm" == "GradCAM" || "$algorithm" == "Saliency" ]]; then
                    NEW_PROCESS="--new_process"
                else
                    NEW_PROCESS=""
                fi

                python3 run_attack.py \
                    --model_name "$model_name" \
                    --train_data_name "$data_name" \
                    --n_classes "$n_classes" \
                    --data_name "$data_name" \
                    --algorithm "$algorithm" \
                    --epsilon "$epsilon" \
                    --batch_size "$batch" \
                    --n_steps 100 \
                    $NEW_PROCESS
            done
        done
    done
done
# Run the full attack for all transformers models and datasets with epsilon 2 and 4.
for epsilon in 2 4; do
    for model_name in MedVit; do
        for data_info in "dermamnist_224 7" "octmnist_224 4" "pneumoniamnist_224 2"; do
            read -r data_name n_classes <<< "$data_info"
            
            for alg_info in "lxt 1"; do
                read -r algorithm batch <<< "$alg_info"
                
                echo "Running attack for $model_name on $data_name with algorithm $algorithm and ε=$epsilon..."
                    
                python3 run_attack.py \
                    --model_name "$model_name" \
                    --train_data_name "$data_name" \
                    --n_classes "$n_classes" \
                    --data_name "$data_name" \
                    --algorithm "$algorithm" \
                    --epsilon "$epsilon" \
                    --batch_size "$batch" \
                    --n_steps 100
            done
        done
    done
done
'

