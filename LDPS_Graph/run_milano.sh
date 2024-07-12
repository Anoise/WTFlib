random_seed=2023
root_path_name=../data_trfc/Milano/
model_id_name=Milano
data_name=Milano
seq_len=36

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/$data_name'_'LongForecasting" ]; then
    mkdir ./logs/$data_name'_'LongForecasting
fi


# Models=('DecomLinear' 'STID' 'Periodformer' 'DLinear' 'PatchTST' 
# 'FEDformer' 'Autoformer' 'Informer' 'Transformer' 'Mvstgn' 'DyDgcrn' 'Gwnet' 'DyDcrnn')


model_name=DecomLinearV2
echo "------------- model --------------" $model_name
for pred_len in 24 36 48 72
do
    echo "------------- pre-len --------------" $pred_len
    python -u run.py \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 416 \
    --train_epochs 10 \
    --lradj='type1' \
    --learning_rate 0.0001 \
    >logs/$data_name'_'LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

exit 0

model_name=DecomLinear
echo "------------- model --------------" $model_name
for pred_len in 48 # 24 36 48 72
do
    echo "------------- pre-len --------------" $pred_len
    python -u run.py \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 416 \
    --train_epochs 10 \
    --lradj='type1' \
    --learning_rate 0.0001 \
    #>logs/$data_name'_'LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

exit 0

# ## Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
# ## Link: https://arxiv.org/abs/2208.05233
# ## Official Code: https://github.com/zezhishao/STID

# model_name=STID
# echo "------------- model --------------" $model_name
# for pred_len in 24 36 48 72
# do
#     echo "------------- pre-len --------------" $pred_len
#     python -u run.py \
#     --random_seed $random_seed \
#     --root_path $root_path_name \
#     --model_id $model_id_name'_'$seq_len'_'$pred_len \
#     --model $model_name \
#     --data $data_name \
#     --seq_len $seq_len \
#     --pred_len $pred_len \
#     --enc_in 416 \
#     --train_epochs 10 \
#     --lradj='type1' \
#     --learning_rate 0.002 \
#     --weight_decay 0.0001 \
#     >logs/$data_name'_'LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
# done


# ## Paper: Does Long-Term Series Forecasting Need Complex Attention and Extra Long Inputs?
# ## Link: https://arxiv.org/abs/2306.05035
# ## Official Code: https://github.com/Anoise/Periodformer

# model_name=Periodformer
# echo "------------- model --------------" $model_name
# for pred_len in 24 36 48 72
# do
#     echo "------------- pre-len --------------" $pred_len
#     python -u run.py \
#     --random_seed $random_seed \
#     --root_path $root_path_name \
#     --model_id $model_id_name'_'$seq_len'_'$pred_len \
#     --model $model_name \
#     --data $data_name \
#     --seq_len $seq_len \
#     --pred_len $pred_len \
#     --enc_in 416 \
#     --train_epochs 10 \
#     --loss mae \
#     --lradj='type1' \
#     --learning_rate 0.0001 \
#     >logs/$data_name'_'LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
# done

# ## Paper: Does Long-Term Series Forecasting Need Complex Attention and Extra Long Inputs?
# ## Link: https://arxiv.org/abs/2205.13504
# ## Official Code: https://github.com/cure-lab/LTSF-Linear

# model_name=DLinear
# echo "------------- model --------------" $model_name
# for pred_len in 24 36 48 72
# do
#     echo "------------- pre-len --------------" $pred_len
#     python -u run.py \
#     --random_seed $random_seed \
#     --root_path $root_path_name \
#     --model_id $model_id_name'_'$seq_len'_'$pred_len \
#     --model $model_name \
#     --data $data_name \
#     --seq_len $seq_len \
#     --pred_len $pred_len \
#     --enc_in 416 \
#     --train_epochs 10 \
#     --lradj='type1' \
#     --learning_rate 0.0001 \
#     >logs/$data_name'_'LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
# done


## Paper: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
## Link: https://arxiv.org/abs/2211.14730
## Official Code: https://github.com/yuqinie98/PatchTST

model_name=PatchTST
echo "------------- model --------------" $model_name
for pred_len in 24 36 48 72
do
    echo "------------- pre-len --------------" $pred_len
    python -u run.py \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 416 \
    --train_epochs 10 \
    --lradj='TST' \
    --learning_rate 0.0001 \
    >logs/$data_name'_'LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

exit 0

## Paper: FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting
## Link: https://arxiv.org/abs/2201.12740
## Official Code: https://github.com/MAZiqing/FEDformer

model_name=FEDformer
echo "------------- model --------------" $model_name
for pred_len in 24 36 48 72
do
    echo "------------- pre-len --------------" $pred_len
    python -u run.py \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 416 \
    --train_epochs 10 \
    --lradj='type1' \
    --learning_rate 0.0001 \
    >logs/$data_name'_'LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

## Paper: Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
## Link: https://arxiv.org/abs/2106.13008
## Official Code: https://github.com/thuml/Autoformer

model_name=Autoformer
echo "------------- model --------------" $model_name
for pred_len in 24 36 48 72
do
    echo "------------- pre-len --------------" $pred_len
    python -u run.py \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 416 \
    --train_epochs 10 \
    --lradj='type1' \
    --learning_rate 0.0001 \
    >logs/$data_name'_'LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

## Paper: Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
## Link: https://arxiv.org/abs/2012.07436
## Official Code: https://github.com/thuml/Autoformer

model_name=Informer
echo "------------- model --------------" $model_name
for pred_len in 24 36 48 72
do
    echo "------------- pre-len --------------" $pred_len
    python -u run.py \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 416 \
    --train_epochs 10 \
    --lradj='type1' \
    --learning_rate 0.0001 \
    >logs/$data_name'_'LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done


## Paper: Attention Is All You Need
## Link: https://arxiv.org/abs/1706.03762
## Official Code: https://github.com/tensorflow/tensor2tensor

model_name=Transformer
echo "------------- model --------------" $model_name
for pred_len in 24 36 48 72
do
    echo "------------- pre-len --------------" $pred_len
    python -u run.py \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 416 \
    --train_epochs 10 \
    --lradj='type1' \
    --learning_rate 0.0001 \
    >logs/$data_name'_'LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done


## Paper: MVSTGN: A Multi-View Spatial-Temporal Graph Network for Cellular Traffic Prediction
## Link: https://ieeexplore.ieee.org/document/9625773
## Official Code: https://github.com/glab2019/MVSTGN

model_name=Mvstgn
echo "------------- model --------------" $model_name
for pred_len in 24 36 48 72
do
    echo "------------- pre-len --------------" $pred_len
    python -u run.py \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 416 \
    --train_epochs 10 \
    --lradj='Mvstgn' \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    >logs/$data_name'_'LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done


## Paper: Graph WaveNet for Deep Spatial-Temporal Graph Modeling
## Link: https://arxiv.org/abs/1906.00121
## Official Code: https://github.com/nnzhan/Graph-WaveNet

model_name=Gwnet
echo "------------- model --------------" $model_name
for pred_len in 24 36 48 72
do
    echo "------------- pre-len --------------" $pred_len
    python -u run.py \
    --random_seed $random_seed \
    --root_path $root_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 416 \
    --train_epochs 10 \
    --lradj='default' \
    --dropout 0.3 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    >logs/$data_name'_'LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done


### Diffcult to Training ...

## Paper: Dynamic Graph Convolutional Recurrent Network for Traffic Prediction: Benchmark and Solution
## Link: https://arxiv.org/abs/2104.14917
## Official Code: https://github.com/tsinghua-fib-lab/Traffic-Benchmark

# model_name=DyDgcrn
# echo "------------- model --------------" $model_name
# for pred_len in 24 36 48 72
# do
#     echo "------------- pre-len --------------" $pred_len
#     python -u run.py \
#     --random_seed $random_seed \
#     --root_path $root_path_name \
#     --model_id $model_id_name'_'$seq_len'_'$pred_len \
#     --model $model_name \
#     --data $data_name \
#     --seq_len $seq_len \
#     --pred_len $pred_len \
#     --enc_in 416 \
#     --train_epochs 10 \
#     --lradj='default' \
#     --learning_rate 0.001 \
#     --weight_decay 0.0001 \
#     >logs/$data_name'_'LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
# done


## Paper: Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting
## Link: https://arxiv.org/abs/1707.01926
## Official Code: https://github.com/liyaguang/DCRNN

# model_name=DyDcrnn
# echo "------------- model --------------" $model_name
# for pred_len in 24 36 48 72
# do
#     echo "------------- pre-len --------------" $pred_len
#     python -u run.py \
#     --random_seed $random_seed \
#     --root_path $root_path_name \
#     --model_id $model_id_name'_'$seq_len'_'$pred_len \
#     --model $model_name \
#     --data $data_name \
#     --seq_len $seq_len \
#     --pred_len $pred_len \
#     --enc_in 416 \
#     --train_epochs 10 \
#     --dropout 0.0 \
#     --lradj='default' \
#     --learning_rate 0.001 \
#     >logs/$data_name'_'LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
# done
