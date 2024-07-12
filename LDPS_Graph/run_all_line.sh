if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

random_seed=2023

seq_len=8

root_path_name=./datas/C2TM/
model_id_name=C2TM
data_name=C2TM

for model_name in 'DecomLinear' 'STID' 'DyDcrnn' 'DyDgcrn' 'Gwnet' 'Periodformer' 'DLinear' 'PatchTST' 'FEDformer' 'Autoformer' 'Informer' 'Transformer'
do 
    echo "------------- pre-len --------------" $model_name
    for pred_len in 4 5 6 8
    do
        echo "------------- pre-len --------------" $pred_len
        python -u run_longExp.py \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --model_id $model_id_name_$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 552 \
        --d_model 512 \
        --d_ff 2048 \
        --dropout 0.05\
        --des 'Exp' \
        --train_epochs 10\
        --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/$data_name'_'LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
    done
done