GPU_ID=0

####################  ScanNet_orn dataset  #######################
DATASET='scan'
SPLIT=1 
BZ=16
NPOINT=512
KNBRS=120
METRIC='seuclidean'
FC=0.3

args=(--dataset "${DATASET}" --split $SPLIT 
      --bz $BZ --points $NPOINT --k $KNBRS 
      --metric "${METRIC}"
      --factor $FC)

CUDA_VISIBLE_DEVICES=$GPU_ID python cls.py "${args[@]}"