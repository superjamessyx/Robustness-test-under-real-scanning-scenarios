export CUDA_VISIBLE_DEVICES=0,1,2,3
# model_list = ['resnet50', 'convnext_base','resnet34', 'vgg16','resnet18', \
# 'resnet101', 'swin_base_patch4_window7_224', 'swin_large_patch4_window7_224','vit_base_patch16_224'\
# 'vit_large_patch16_224', 'convnext_large', 'vit_small_patch16_224','swin_small_patch4_window7_224', 'deit_base_patch16_224', 'efficientnet_b7']
export model=resnet18
for fold in 0 1 2 3 4
do
  port=$[$fold+6225]
  OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=4 --master_port $port finetune.py \
      --accum_iter 1 \
      --batch_size 128 \
      --model $model \
      --epochs 30 \
      --blr 1e-4 \
      --resume './output_source/'$model'/'$fold'/checkpoint_best.pth' \
      --fold $fold \
      --output_dir ./output_finetuned \
      --dist_eval
done