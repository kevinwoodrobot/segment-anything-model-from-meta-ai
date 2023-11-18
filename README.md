# Segment Anything Model from Meta AI 

## Classical Segmentation Methods
- HSV Color Space Segmentation 
https://www.youtube.com/watch?v=G3PW5ysKDxc&list=PLSK7NtBWwmpQwSUi53XUK5o6-b9H3ABrO&index=9&t=21s
- Histogram Backprojection: 
https://www.youtube.com/watch?v=aOHStBqEFlQ&list=PLSK7NtBWwmpQwSUi53XUK5o6-b9H3ABrO&index=33&t=513s
- Watershed Segmentation: 
https://www.youtube.com/watch?v=3MUxPn3uKSk&list=PLSK7NtBWwmpQwSUi53XUK5o6-b9H3ABrO&index=42
- Graph Cut Segmentation: 
https://www.youtube.com/watch?v=hEdQAhuYO3A&list=PLSK7NtBWwmpQwSUi53XUK5o6-b9H3ABrO&index=43&t=518s

## Segment Anything Repo Overview 
https://github.com/facebookresearch/segment-anything

Dataset: 
https://segment-anything.com/dataset/index.html

Requirements: 
`python >= 3.8`, `pytorch>=1.7` and `torchvision>=0.8`

## SAM Website GUI 
https://segment-anything.com/demo


## Environment Setup  
1. Create virtual environment 
```bash
py -3.11 -m venv env311 
.\env311\Scripts\activate
```

2. Install modules 
```bash 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```
Pytorch and torchvision install reference: 
https://pytorch.org/get-started/locally/

3. Download model ViT-H (large model)
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

4. Download model ViT-B (small model)
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

## Coding  
Topics covered in the coding portion of this video. 
### testSmallModel() 
Using the vit-b model. 
### testLargeModel() 
Using the vit-h model. 
### testLargeModelTuned() 
Using the vit-h model with custom parameters. 
### testSmallModelSingleMask([300,700]) 
Using the vit-b model to find mask of specific object. 

## Reference
https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/notebooks/automatic_mask_generator_example.ipynb




