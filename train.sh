#!/bin/sh

analysis=$1
resnet="resnet18"
device='cpu'

##### Training Hyper-Parameter #####
start_epochs=0
num_epochs=10
batch_size=100
num_workers=2

##### Optimizer Specifications #####
optimizer='sgd'
learning_rate=1e-5
momentum=0.9
weight_decay=0.0001

##### EA Blocks Specifications #####
ae_rate=1e-3
### ae_blocks_mode : 'add', 'mult', 'zeros'
ae_conv1='zeros'
ae_layer1='zeros,zeros'
ae_layer2='zeros,zeros'
ae_layer3='zeros,zeros'
ae_layer4='zeros,zeros'
ae_layers=$ae_conv1';'$ae_layer1';'$ae_layer2';'$ae_layer3';'$ae_layer4
echo $ae_layers




python ae_resnet_train.py   --analysis $analysis            \
                            --resnet $resnet                \
                            --start-epoch $start_epochs     \
                            --epochs $num_epochs            \
                            --batch-size $batch_size        \
                            --worker $num_workers           \
                            --optimization $optimizer       \
                            --learning-rate $learning_rate  \
                            --momentum $momentum            \
                            --weight_decay $weight_decay    \
                            --ae-layers $ae_layers          \
                            --ae-rate $ae_rate              \
                            --gpu                           \
                            --pretrained                    \
                            --device $device                  
