GPU_ID=0

####################  ScanNet_aug dataset  #######################
DATASET='scan'
SPLIT=3 
BZ=32
NPOINT=512
KNBRS=120
METRIC='seuclidean'
FC=0.2

args=(--dataset "${DATASET}" --split $SPLIT 
      --bz $BZ --points $NPOINT --k $KNBRS 
      --metric "${METRIC}"
      --factor $FC)

CUDA_VISIBLE_DEVICES=$GPU_ID python cls.py "${args[@]}"