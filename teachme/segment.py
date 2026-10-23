"""Background removal with person segmentation (syllabus Module 6: segmentation).

Why this exists: with a busy room behind you, a classifier trained on ~25
frames per gesture happily learns the *background* ("this view of the window =
scissors") instead of your hand, which is shortcut learning. We saw exactly that
in testing: the occlusion map was cold on the hand and hot on the room.

Fix: LR-ASPP MobileNetV3-Large (torchvision, trained on COCO with the 21 Pascal
VOC labels) labels every pixel. We keep the pixels it calls "person" (your
hand, arm and body) and paint everything else flat grey, so the classifier
only ever sees *you*. The same step runs on training samples, live frames,
throws and explanations, so train and test inputs match.

    frame 256x256 → resize 320x320 → LR-ASPP → p(person) [1, 1, 320, 320]
          → upsample to 256x256 → threshold 0.5 → mask → grey background
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from teachme.config import IMAGENET_MEAN, IMAGENET_STD

SEG_SIZE = 320        # segmentation resolution: 320 keeps fingertips that 224/256 lose
THRESHOLD = 0.5       # p(person) above this = keep the pixel
FILL = 0.5            # background colour (mid grey, in [0, 1] pixel units)


class PersonSegmenter:
    """Wraps a torchvision segmentation model and keeps only the 'person' pixels."""

    def __init__(self, model: nn.Module, person_index: int) -> None:
        self.model = model.eval()
        self.person_index = person_index
        self.mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    @torch.inference_mode()
    def person_mask(self, x: torch.Tensor) -> torch.Tensor:
        """[N, 3, H, W] pixels in [0, 1] -> [N, 1, H, W] mask (1 = person)."""
        small = F.interpolate(x, size=(SEG_SIZE, SEG_SIZE), mode="bilinear", align_corners=False)
        logits = self.model((small - self.mean) / self.std)["out"]          # [N, 21, S, S]
        p_person = logits.softmax(1)[:, self.person_index:self.person_index + 1]
        p_person = F.interpolate(p_person, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return (p_person > THRESHOLD).float()

    def remove_background(self, x: torch.Tensor) -> torch.Tensor:
        """Keep person pixels, paint the rest grey. Shapes: [N, 3, H, W] -> same."""
        mask = self.person_mask(x)
        return x * mask + FILL * (1 - mask)

    def focus_pil(self, img: Image.Image) -> Image.Image:
        """PIL in -> PIL out (same size) with the background greyed out."""
        x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None]
        out = self.remove_background(x)[0].permute(1, 2, 0).numpy()
        return Image.fromarray((out * 255).round().astype(np.uint8))


def load_person_segmenter() -> PersonSegmenter:
    """torchvision LR-ASPP MobileNetV3-Large, ~3.2 M params, 12.5 MB, 57.9 mIoU on COCO-VOC."""
    from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights, lraspp_mobilenet_v3_large

    weights = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
    model = lraspp_mobilenet_v3_large(weights=weights).eval()
    model.requires_grad_(False)
    return PersonSegmenter(model, weights.meta["categories"].index("person"))
