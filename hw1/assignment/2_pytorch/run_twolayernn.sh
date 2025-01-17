#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model twolayernn \
    --hidden-dim 512 \
    --epochs 5 \
    --weight-decay 0.0 \
    --momentum 0.0 \
    --batch-size 256 \
    --lr 0.01 | tee twolayernn.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
