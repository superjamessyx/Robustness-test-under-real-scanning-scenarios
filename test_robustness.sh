export CUDA_VISIBLE_DEVICES=0,1,2,3
# model_list = ['resnet50', 'convnext_base', 'efficientnet_b7','resnet34', 'vgg16','resnet18', \
# 'resnet101', 'swin_base_patch4_window7_224', 'swin_large_patch4_window7_224','vit_base_patch16_224'\
# 'vit_large_patch16_224', 'convnext_large', 'vit_small_patch16_224','swin_large_patch4_window7_224', 'deit_base_patch16_224']
export model=resnet18
for fold in 0 1 2 3 4
do
  export CUDA_VISIBLE_DEVICES=1
  port=6283
  OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port $port test_robustness.py \
    --accum_iter 1 \
    --batch_size 128 \
    --model $model \
    --epochs 100 \
    --blr 5e-4 \
    --resume './output_finetuned/'$model'/'$fold'/checkpoint_best.pth' \
    --eval \
    --fold $fold \
done

