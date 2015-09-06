#!/usr/bin/env sh
# Create the imagenet lmdb inputs

TRANSFORMATIONS=('_0.047_0.671_0.092_1.898')
TRAIN_TEST=('source_train' 'target_train' 'source_val' 'target_val' 'source_test' 'target_test')
RESIZE_SIZE=(227 216 227 216 227 216)
TARGET_ROOT="/scr/r6/tgebru/inverting_conv/caffe_invert_alexnet/data/pascal/"
DATA_ROOT="/"
TOOLS=/scr/r6/tgebru/timnit_caffe/build/tools
RESIZE=true

i=0
for sw in ${TRANSFORMATIONS[@]}
do
  for t in ${TRAIN_TEST[@]}
    do
      echo $i
      IMLIST_FILE="${TARGET_ROOT}${sw}_${t}.txt"
      TARGET_FILE="${TARGET_ROOT}${sw}_${t}_lmdb"
      echo $IMLIST_FILE

      # Set RESIZE=true to resize the images to 256x256. 
      if $RESIZE; then
        RESIZE_HEIGHT=${RESIZE_SIZE[$i]}
        RESIZE_WIDTH=${RESIZE_SIZE[$i]}
    else
      RESIZE_HEIGHT=0
      RESIZE_WIDTH=0
    fi

    if [ ! -d "$DATA_ROOT" ]; then
  echo "Error: DATA_ROOT is not a path to a directory: $DATA_ROOT"
    echo "Set the DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

   echo "Creating ${sw}_${t} lmdb..."

   GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $DATA_ROOT \
    $IMLIST_FILE \
    $TARGET_FILE

    echo "Done."
    ((i++))
  done
done

