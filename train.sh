export CUDA_VISIBLE_DEVICES=0,1,2,3
# model_list = ['resnet50', 'convnext_base','resnet34', 'vgg16','resnet18', \
# 'resnet101', 'swin_base_patch4_window7_224', 'swin_large_patch4_window7_224','vit_base_patch16_224'\
# 'vit_large_patch16_224', 'convnext_large', 'vit_small_patch16_224','swin_small_patch4_window7_224', 'deit_base_patch16_224', 'efficientnet_b7']
export model=resnet18
export fold=0
OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=4 --master_port 6249 train.py \
    --accum_iter 1 \
    --batch_size 128 \
    --model $model \
    --epochs 100 \
    --blr 5e-4 \
    --output_dir ./output_source \
    --fold $fold \
    --dist_eval