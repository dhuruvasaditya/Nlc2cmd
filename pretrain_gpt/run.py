import os

if __name__ == "__main__":
    os.system("python clm.py \
    --model_name_or_path gpt2 \
    --train_file 'G:/Work Related/Nlc2cmd/pretrain_gpt/pretrain_data/Cmd.txt' \
    --validation_file '' \
    --do_train \
    --do_eval \
    --output_dir /bashgpt_model/")