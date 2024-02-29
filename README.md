# Robust 3D Gaussian Splatting
Armin Ettenhofer & Paul Ungermann (Technical University of Munich)

---

Abstract: <i>Neural radiance fields (NeRFs) are vulnerable to dynamic objects polluting the input data of an otherwise static scene that should not be part of the learned scene representation called distractors. In our project, we adapt 3D Gaussian Splatting, a novel method to generate NeRFs, to be robust against such distractors. We achieve this by implementing a robust loss function. We create segmentation masks of distractor to ignore them during training. Using this robust loss and our adaptations, we obtain significantly better quantitative and qualitative results on distractor-polluted scenes than the original implementation of 3D Gaussian Splatting and RobustNeRF. </i>

## Quantitative Results

|Scene|Gaussian Splat.|RobustNeRF|+Neural|+Segment.|+Both|
|---|---|---|---|---|---|
|Statue|0.1443|0.1374|0.1376|0.1526|<b>0.1214</b>|
|Yoda|0.2001|0.2177|0.1746|0.1891|<b>0.1684</b>|
|And-bot|0.1638|0.1622|<b>0.1308</b>|0.1778|0.1314|
|Crab|0.1414|0.1919|0.1184|0.1765|<b>0.1151</b>|
|Mean|0.1624|0.1773|0.1404|0.1740|<b>0.1341</b>|

LPIPS score for each scene, lower better. We can see that our method significantly performs better.
        

        

## Qualitative Results
We conduct five different versions to evaluate our method:
- Gaussian Splat. (Baseline) 
The unmodified implementation of 3D Gaussian Splatting
- RobustNeRF: 
A version of RobustNeRF implemented for 3D Gaussian Splatting.
- RobustNeRF + Segmentation (+Segment.): 
Sequentially adds the segmentation overlap improvements after calculating the mask using RobustNeRF but doesn’t use logistic regression.
- RobustNeRF + Linear Layer (+Neural): 
Sequentially adds a trainable linear layer after calculating the mask using RobustNeRF but doesn’t use segmentation.
- RobustNeRF + Linear Layer + Segmentation (+Both):
Our method.

|Gaussian Splat.|RobustNeRF|+Neural|+Segment.|+Both|
|---|---|---|---|---|
| <img src="/assets/images/and_bot/baseline.png" width="150"/>| <img src="/assets/images/and_bot/robust.png" width="150"/> | <img src="/assets/images/and_bot/neural.png" width="150"/> | <img src="/assets/images/and_bot/seg.png" width="150"/> | <img src="/assets/images/and_bot/both.png" width="150"/> |
| <img src="/assets/images/and_bot_2/baseline.png" width="150"/>| <img src="/assets/images/and_bot_2/robust.png" width="150"/> | <img src="/assets/images/and_bot_2/neural.png" width="150"/> | <img src="/assets/images/and_bot_2/seg.png" width="150"/> | <img src="/assets/images/and_bot_2/both.png" width="150"/> |
| <img src="/assets/images/balloon/baseline.png" width="150"/>| <img src="/assets/images/balloon/robust.png" width="150"/> | <img src="/assets/images/balloon/neural.png" width="150"/> | <img src="/assets/images/balloon/seg.png" width="150"/> | <img src="/assets/images/balloon/both.png" width="150"/> |
| <img src="/assets/images/balloon_2/baseline.png" width="150"/>| <img src="/assets/images/balloon_2/robust.png" width="150"/> | <img src="/assets/images/balloon_2/neural.png" width="150"/> | <img src="/assets/images/balloon_2/seg.png" width="150"/> | <img src="/assets/images/balloon_2/both.png" width="150"/> |
| <img src="/assets/images/crab/baseline.png" width="150"/>| <img src="/assets/images/crab/robust.png" width="150"/> | <img src="/assets/images/crab/neural.png" width="150"/> | <img src="/assets/images/crab/seg.png" width="150"/> | <img src="/assets/images/crab/both.png" width="150"/> |
| <img src="/assets/images/crab_2/baseline.png" width="150"/>| <img src="/assets/images/crab_2/robust.png" width="150"/> | <img src="/assets/images/crab_2/neural.png" width="150"/> | <img src="/assets/images/crab_2/seg.png" width="150"/> | <img src="/assets/images/crab_2/both.png" width="150"/> |
| <img src="/assets/images/yoda/baseline.png" width="150"/>| <img src="/assets/images/yoda/robust.png" width="150"/> | <img src="/assets/images/yoda/neural.png" width="150"/> | <img src="/assets/images/yoda/seg.png" width="150"/> | <img src="/assets/images/yoda/both.png" width="150"/> |
| <img src="/assets/images/yoda_2/baseline.png" width="150"/>| <img src="/assets/images/yoda_2/robust.png" width="150"/> | <img src="/assets/images/yoda_2/neural.png" width="150"/> | <img src="/assets/images/yoda_2/seg.png" width="150"/> | <img src="/assets/images/yoda_2/both.png" width="150"/> |


## Technical Tutorial
How to run 3D Gaussian Splatting in general see [here](https://github.com/graphdeco-inria/gaussian-splatting). In addition to these features, we provide a config file to set parameters for robust training. 

config.json:
|  key | value  | 
|---|---|
| save_mask_interval | interval (in number of interations) after a the mask should be saved (in /output/<model>/masks)  | 
|  test_size |  number of test images | 
|  mask_start_epoch |  number of epochs to start using the distractor masks |  
| use_segmentation | flag to intersect the masks (true) or just using the raw masks (false) |
| seg_overlap | indicates the minimum intersection ratio to classify an object as a distractor |
| n_residuals | number of residuals from the last n iterations used to compute the mask |
| per_channel | indicates if the mask is computed using all three image channel (true) |
| use_neural | indicates if the neural approach will be used |
| lambda_reg | regularization parameter for the mask loss |

## Data Preprocessing
Before training the model you need to generate the object segmentation masks using SegmentAnything. You can use ```python3 segment.py --input <input path (use correct scale images!)> --output <path to dataset> --model-type vit_b --checkpoint <path>.ckpt``` to generate the object segmentation mask. Note that you need to install SegmentAnything and use the latest vit_b checkpoint. See [here](https://github.com/facebookresearch/segment-anything) for a tutorial.
