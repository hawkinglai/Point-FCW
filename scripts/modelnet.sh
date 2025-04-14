GPU_ID=0

####################  ModelNet dataset  #######################
DATASET='mn40'
SPLIT=1    #  whatever you want to input, not split in this dataset
BZ=64
NPOINT=1024
KNBRS=120
METRIC='euclidean'
FC=0.2

args=(--dataset "${DATASET}" --split $SPLIT 
      --bz $BZ --points $NPOINT --k $KNBRS 
      --metric "${METRIC}"
      --factor $FC)

CUDA_VISIBLE_DEVICES=$GPU_ID python cls.py "${args[@]}"