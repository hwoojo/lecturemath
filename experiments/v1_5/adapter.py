"""Optional V1 CPU adapter; importing postprocessing never imports Torch."""
import sys
from pathlib import Path
import cv2
import numpy as np


def load_mask(path, shape, polarity):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != shape:
        raise ValueError(f'missing or misaligned binary mask: {path}')
    if not np.isin(image, [0, 255]).all():
        raise ValueError(f'expected strictly binary 0/255 PNG: {path}')
    return image == (0 if polarity == 'dark' else 255)


def validate_probability(p, shape):
    if p.shape != shape or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError('probability must be finite [0,1] and match original coordinates')
    if np.isin(p, [0, 1]).all():
        raise ValueError('B4 requires continuous probabilities, not binary masks')
    return p


class LectureNetAdapter:
    def __init__(self, config, checkpoint, threads=2):
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ACCESS2021_release'))
        from AM_CommonTools.configuration.configuration import Configuration
        from AccessMath.lecturenet_v1.FCN_lecturenet import FCN_LectureNet
        torch.set_num_threads(threads)
        self.model = FCN_LectureNet.CreateFromConfig(Configuration.from_file(str(config)), 3, False)
        self.model.load_state_dict(torch.load(str(checkpoint), map_location='cpu', weights_only=True))
        self.model.eval()

    def infer(self, image):
        import torch
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        w, h = pil.size
        # Exact V1 resize policy. Native 720p and 1080p are unaffected.
        while pil.width * pil.height > 2500000:
            pil = pil.resize((pil.width//2, pil.height//2), Image.Resampling.LANCZOS)
        with torch.inference_mode():
            logits, _, _ = self.model(self.model.prepare_image(pil))
            p = torch.sigmoid(logits)[0, 0].cpu().numpy()
        # V1 binary output is black writing / white background. Its sigmoid
        # is background probability (GT is TF.to_tensor of black-writing PNG).
        binary = (p*255).astype(np.uint8) < 128
        if p.shape != (h, w):
            binary = cv2.resize(binary.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            p = cv2.resize(p, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
        return binary, 1.0 - p
