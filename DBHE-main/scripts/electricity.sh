if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi


if [ ! -d "./logs/LongForecasting/electricity" ]; then
    mkdir ./logs/LongForecasting/electricity
fi


# seq_len=96
model_name=PatchTST_MoE_cluster

root_path_name=/data/coding
data_path_name=electricity.csv
model_id_name=electricity
data_name=electricity

# random_seed=2021

   

for seq_len in 96
do
for pred_len in 96
do
for random_seed in 2021
do
for learning_rate in 0.0005
do
for T_num_expert in 16
do
for T_top_k in 1
do
for F_num_expert in 16
do
for F_top_k in 1
do
for rank in 32
do
for d_embed in 128
do
for ortho_loss_weight in 0.01
do
for k in None
do
    if [ "$k" = "None" ]; then
        K_FLAG=""
    else
        K_FLAG="--k $k"
    fi
    python -u /data/coding/DBHE-main/run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      $K_FLAG \
      --rank $rank \
      --d_embed $d_embed \
      --ortho_loss_weight $ortho_loss_weight \
      --enc_in 321 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --T_num_expert $T_num_expert\
      --T_top_k $T_top_k\
      --F_num_expert $F_num_expert\
      --F_top_k $F_top_k\
      --beta 0.1 \
      --des 'Exp' \
      --train_epochs 100\
      --gpu 0\
      --itr 1 --batch_size 4 --learning_rate $learning_rate >logs/LongForecasting/electricity/$model_name'_'$pred_len'_'$T_num_expert'_'$T_top_k'_'$F_num_expert'_'$F_top_k'_'$learning_rate'_'$rank'_'$d_embed'_'$ortho_loss_weight'_'$k.log
done
done
done
done
done
done
done
done
done
done
done
done



for seq_len in 96
do
for pred_len in 192
do
for random_seed in 2021
do
for learning_rate in 0.0005
do
for T_num_expert in 16
do
for T_top_k in 1
do
for F_num_expert in 16
do
for F_top_k in 1
do
for rank in 32
do
for d_embed in 128
do
for ortho_loss_weight in 0.01
do
for k in None
do
    if [ "$k" = "None" ]; then
        K_FLAG=""
    else
        K_FLAG="--k $k"
    fi
    python -u /data/coding/DBHE-main/run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      $K_FLAG \
      --rank $rank \
      --d_embed $d_embed \
      --ortho_loss_weight $ortho_loss_weight \
      --enc_in 321 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --T_num_expert $T_num_expert\
      --T_top_k $T_top_k\
      --F_num_expert $F_num_expert\
      --F_top_k $F_top_k\
      --beta 0.1 \
      --des 'Exp' \
      --train_epochs 100\
      --gpu 0\
      --itr 1 --batch_size 4 --learning_rate $learning_rate >logs/LongForecasting/electricity/$model_name'_'$pred_len'_'$T_num_expert'_'$T_top_k'_'$F_num_expert'_'$F_top_k'_'$learning_rate'_'$rank'_'$d_embed'_'$ortho_loss_weight'_'$k.log
done
done
done
done
done
done
done
done
done
done
done
done


for seq_len in 96
do
for pred_len in 336
do
for random_seed in 2021
do
for learning_rate in 0.0005
do
for T_num_expert in 16
do
for T_top_k in 1
do
for F_num_expert in 16
do
for F_top_k in 1
do
for rank in 32
do
for d_embed in 128
do
for ortho_loss_weight in 0.01
do
for k in None
do
    if [ "$k" = "None" ]; then
        K_FLAG=""
    else
        K_FLAG="--k $k"
    fi
    python -u /data/coding/DBHE-main/run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      $K_FLAG \
      --rank $rank \
      --d_embed $d_embed \
      --ortho_loss_weight $ortho_loss_weight \
      --enc_in 321 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --T_num_expert $T_num_expert\
      --T_top_k $T_top_k\
      --F_num_expert $F_num_expert\
      --F_top_k $F_top_k\
      --beta 0.1 \
      --des 'Exp' \
      --train_epochs 100\
      --gpu 0\
      --itr 1 --batch_size 4 --learning_rate $learning_rate >logs/LongForecasting/electricity/$model_name'_'$pred_len'_'$T_num_expert'_'$T_top_k'_'$F_num_expert'_'$F_top_k'_'$learning_rate'_'$rank'_'$d_embed'_'$ortho_loss_weight'_'$k.log
done
done
done
done
done
done
done
done
done
done
done
done


for seq_len in 96
do
for pred_len in 720
do
for random_seed in 2021
do
for learning_rate in 0.0005
do
for T_num_expert in 16
do
for T_top_k in 1
do
for F_num_expert in 16
do
for F_top_k in 1
do
for rank in 32
do
for d_embed in 128
do
for ortho_loss_weight in 0.01
do
for k in None
do
    if [ "$k" = "None" ]; then
        K_FLAG=""
    else
        K_FLAG="--k $k"
    fi
    python -u /data/coding/DBHE-main/run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      $K_FLAG \
      --rank $rank \
      --d_embed $d_embed \
      --ortho_loss_weight $ortho_loss_weight \
      --enc_in 321 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --T_num_expert $T_num_expert\
      --T_top_k $T_top_k\
      --F_num_expert $F_num_expert\
      --F_top_k $F_top_k\
      --beta 0.1 \
      --des 'Exp' \
      --train_epochs 100\
      --gpu 0\
      --itr 1 --batch_size 4 --learning_rate $learning_rate >logs/LongForecasting/electricity/$model_name'_'$pred_len'_'$T_num_expert'_'$T_top_k'_'$F_num_expert'_'$F_top_k'_'$learning_rate'_'$rank'_'$d_embed'_'$ortho_loss_weight'_'$k.log
done
done
done
done
done
done
done
done
done
done
done
done
