model_type=$1
data=$2
seed=42
block_size=128
OUTPUT_DIR=../preprocess/$seed/$model_type

rm -r $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

python -u ../src/preprocess.py --input ../data/$data \
                        --stereotypes ../data/stereotype.txt \
                        --attributes ../data/female.txt,../data/male.txt \
                        --output $OUTPUT_DIR \
                        --seed $seed \
                        --block_size $block_size \
                        --model_type $model_type

