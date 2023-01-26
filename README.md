## A tool for robustness assessment of DNN under practical scanning scenarios.

This is a PyTorch/GPU implementation of the paper **Assessing the Robustness of Deep Learning-Assisted Pathological Image Analysis under Practical Variables of Imaging System**. The code is run on 4 * A100 40G GPUs.

## Usage:



### Step1: Transfer the model pre-tained on ImageNet to TCT source dataset.

```bash
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
```

Or you can run

```bash
bash train.sh
```

- Here the effective batch size is 64 (`batch_size` per gpu)  * 4(gpus) * 4(`accum_iter`) = 1024.
- `blr` is the base learning rate. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective batch size / 256.
- The `fold` is the fold index of cross-validation. 



### Step2: Fine-tune the model on images scanned by default scanner parameters.

```bash
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
```

Or you can run

```
bash finetune.sh
```

- Here the effective batch size is 64 (`batch_size` per gpu)  * 4(gpus) * 4(`accum_iter`) = 1024.
- `blr` is the base learning rate. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective batch size / 256.
- The `fold` is the fold index of cross-validation. 



### Step3: Test the robustness of the model under multiple scanner parameters using proposed indicators.

```bash
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
    --batch_size 128 \
    --model $model \
    --resume './output_finetuned/'$model'/'$fold'/checkpoint_best.pth' \
    --eval \
    --fold $fold \
done
```

This code will save the test results of the model under different scanner parameters, then you can use the following code to calculate the robustness results of the model, which will give you latex format results as well as excel format results.

```bash
python ./performance_eval/show_robustness_results.py
```

