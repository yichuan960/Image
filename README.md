# Robust 3D Gaussian Splatting
Armin Ettenhofer & Paul Ungermann (Technical University of Munich)

---

Abstract: <i>Neural radiance fields (NeRFs) are vulnerable to dynamic objects polluting the input data of an otherwise static scene that should not be part of the learned scene representation called distractors. In our project, we adapt 3D Gaussian Splatting, a novel method to generate NeRFs, to be robust against such distractors. We achieve this by implementing a robust loss function. We create segmentation masks of distractor to ignore them during training. Using this robust loss and our adaptations, we obtain significantly better quantitative and qualitative results on distractor-polluted scenes than the original implementation of 3D Gaussian Splatting and RobustNeRF. </i>

## Technical Tutorial
How to run 3D Gaussian Splatting in general see [here]([kh](https://github.com/graphdeco-inria/gaussian-splatting)). In addition to these features, we provide a config file to set parameters for robust training.

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
