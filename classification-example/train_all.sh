sbatch --export=MODEL=distilbert-base-uncased,BATCH_SIZE=256 train_model_on_rockfish.sh
sbatch --export=MODEL=bert-base-uncased,BATCH_SIZE=128 train_model_on_rockfish.sh
sbatch --export=MODEL=bert-large-uncased,BATCH_SIZE=64 train_model_on_rockfish.sh
sbatch --export=MODEL=roberta-base-uncased,BATCH_SIZE=128 train_model_on_rockfish.sh
sbatch --export=MODEL=roberta-large,BATCH_SIZE=64 train_model_on_rockfish.sh
