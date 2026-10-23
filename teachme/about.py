"""Text for the 'Under the hood' tab: which syllabus module each feature demonstrates."""

ABOUT_MD = """
## 📚 What each feature demonstrates

| Feature in the app | What happens under the hood | Syllabus module |
|---|---|---|
| Choosing classes (incl. **Nothing**) | Framing the task as image **classification**, one label per frame. A closed-set classifier *must* pick a class, so we add "Nothing" for no gesture | **M1** Understanding the Vision Problem |
| Frozen MobileNetV3 + new head | Pretrained **classifier** backbone, **transfer learning**: only a tiny head is trained | **M2** Building Your First Vision Model |
| Model card, Linear vs MLP head | **Tensor shapes** `[N,3,224,224] → [N,960,7,7] → [N,960] → [N,K]`, **parameter counts** (≈3 M frozen vs 3.8 k trainable), MobileNetV3 building blocks | **M3** Neural Network Design |
| Augmentation checkboxes, ×3 copies | **Data augmentation** (crop, flip, colour jitter, rotation, blur) turns ~25 frames per class into ~100 training views. "Wrong? It was ✌️" adds hard examples | **M4** Improving the Data |
| Training curves, confusion matrix, optimiser & activation choice | **Cross-entropy loss**, **Adam vs SGD**, **accuracy + confusion matrix** on a held-out split, **ReLU / Hardswish / Tanh / Sigmoid** | **M5** Training Better Models |
| 🎯 Remove background, "What the AI sees" | **Semantic segmentation**: LR-ASPP MobileNetV3 labels every pixel; we keep the "person" pixels and grey out the room | **M6** Beyond Classification |
| int8 backbone, int8 head, feature caching | **Quantisation** (int8 trunk ~4× smaller and much faster), **compute optimisation** (backbone runs once per image, not once per epoch) | **M7** Efficient Edge AI |
| Normalisation layer, feature standardisation, weight decay, dropout, init choice, occlusion map, steady throws | **Image normalisation** (ImageNet mean/std as a layer), **feature normalisation**, **regularisation**, **weight initialisation** (zeros → symmetry problem), **shortcut learning** caught by the occlusion map and fixed by background removal, **temporal smoothing** | **M8** Making Models Robust |

### Tensor shapes
```
webcam frame      any size → centre crop → 256 x 256
segmentation      [1, 3, 320, 320] → LR-ASPP → [1, 21, 320, 320] → p(person) → mask [1, 1, 256, 256]
background-free   [1, 3, 256, 256]    room painted grey, then resized to 224
input             [1, 3, 224, 224]    raw pixels in [0, 1]
normalise         [1, 3, 224, 224]    (x - mean) / std, done inside the model
trunk (frozen)    [1, 960, 7, 7]      int8 by default
avg-pool          [1, 960]            the feature vector we cache
head              [1, K]              Standardise → Dropout → Linear 960→K   (3,844 params for K=4)
softmax           [1, K]              class probabilities
```

### Things to try live
* **Untick 🎯 Remove background**, capture each gesture from a *different* camera angle, and train. Validation still says
  ~100%, but the occlusion map is red on the room, not your hand. That's shortcut learning. Tick it again and retrain.
* **Untick all augmentations** and train again, then test with different lighting or distance.
* Set the head to **MLP** and init to **Zeros**: accuracy is stuck at chance, because symmetry is never broken.
* Switch the backbone to **fp32** and watch feature extraction get roughly 7× slower.
"""
