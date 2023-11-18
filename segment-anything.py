from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
from enum import Enum

class ModelType(Enum): 
    vit_b = 'sam_vit_b_01ec64.pth' # smallest model 
    vit_h = 'sam_vit_h_4b8939.pth' # largest model 

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    return img 

def is_point_inside_bbox(point_x, point_y, bbox):
    bbox_x, bbox_y, bbox_width, bbox_height = bbox
    is_inside_x = bbox_x <= point_x < bbox_x + bbox_width
    is_inside_y = bbox_y <= point_y < bbox_y + bbox_height
    return is_inside_x and is_inside_y

class SegmentAnything():
    def __init__(self,model:ModelType): 
        print('Loading Model...')
        sam_checkpoint = model.value
        self.model_name = model.name
        self.sam = sam_model_registry[self.model_name](checkpoint=sam_checkpoint)

        # Only small model works for my gpu, large is too big 
        if model == ModelType.vit_b: 
            print('  Using Cuda...')
            device = "cuda"
            self.sam.to(device=device)

    def loadImage(self,imgPath): 
        print('Loading Image...')
        img = cv2.imread(imgPath) 
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure() 
        plt.imshow(self.img)
        plt.show() 

    def saveResults(self,imgMask,tuned=False): 
        print('Saving Results...')
        imgMask = (imgMask * 255).astype(np.uint8)
        rgb = imgMask[:, :, :3]
        alpha = imgMask[:, :, 3] / 255.0 
        combined = np.zeros_like(self.img)
        for c in range(3): 
            combined[:, :, c] = alpha * rgb[:, :, c] + (1 - alpha) * self.img[:, :, c]
        combined = cv2.cvtColor(combined,cv2.COLOR_RGB2BGR)

        if tuned:
            cv2.imwrite('result-mask-' + self.model_name +'-tuned.png',imgMask)
            cv2.imwrite('result-combined-' + self.model_name +'-tuned.png',combined)
        else: 
            cv2.imwrite('result-mask-' + self.model_name +'.png',imgMask)
            cv2.imwrite('result-combined-' + self.model_name +'.png',combined)

    def generateMasks(self): 
        print('Generating Masks...')
        mask_generator = SamAutomaticMaskGenerator(self.sam)
        maskList = mask_generator.generate(self.img)
        plt.figure() 
        plt.imshow(self.img)
        imgMask = show_anns(maskList)
        plt.show() 

        plt.figure() 
        plt.imshow(imgMask)
        plt.show() 
        self.saveResults(imgMask)

    def generateTunedMasks(self): 
        print('Generating Tuned Masks...')
        mask_generator_tuned = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  
        )
        maskList = mask_generator_tuned.generate(self.img)
        plt.figure() 
        plt.imshow(self.img)
        imgMask = show_anns(maskList)
        plt.show() 

        plt.figure() 
        plt.imshow(imgMask)
        plt.show() 
        self.saveResults(imgMask,tuned=True)

    def generateSingleMask(self,coord): 
        print('Generating Masks...')
        mask_generator = SamAutomaticMaskGenerator(self.sam)
        masks = mask_generator.generate(self.img)

        for mask in masks: 
            if is_point_inside_bbox(coord[0],coord[1],mask['bbox']): 
                plt.figure() 
                plt.imshow(mask['segmentation'])
                plt.plot(coord[0],coord[1])
                plt.show() 
                color_true = [255, 255, 255] 
                color_false = [0, 0, 0]  
                mask_rgb = np.zeros((mask['segmentation'].shape[0], mask['segmentation'].shape[1], 3), dtype=np.uint8)
                mask_rgb[mask['segmentation']] = color_true
                mask_rgb[~mask['segmentation']] = color_false
                print('Saving Results...')
                cv2.imwrite('result-mask-single-' + self.model_name +'.png',mask_rgb)

def testSmallModel(): 
    sa = SegmentAnything(ModelType.vit_b)
    sa.loadImage('nature.png')
    sa.generateMasks() 

def testLargeModel(): 
    sa = SegmentAnything(ModelType.vit_h)
    sa.loadImage('nature.png')
    sa.generateMasks() 

def testLargeModelTuned(): 
    sa = SegmentAnything(ModelType.vit_h)
    sa.loadImage('nature.png')
    sa.generateTunedMasks() 

def testSmallModelSingleMask(coord): 
    sa = SegmentAnything(ModelType.vit_b)
    sa.loadImage('nature.png')
    sa.generateSingleMask(coord) 

if __name__ == '__main__': 
    # testSmallModel() 
    # testLargeModel() 
    # testLargeModelTuned() 
    testSmallModelSingleMask([300,700]) 