from PIL import Image
import cv2
from .upstream import activate, research_config


class LectureNet:
    def __init__(self, settings):
        settings.require_checkpoint()
        activate()
        import torch
        from AccessMath.lecturenet_v1.FCN_lecturenet import FCN_LectureNet
        torch.set_num_threads(settings.threads)
        device = torch.device(settings.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Supported devices: cpu, cuda, cuda:N")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.net = FCN_LectureNet.CreateFromConfig(research_config(settings), 3, False)
        # V34 is a state_dict; never execute arbitrary pickle objects.
        state = torch.load(str(settings.checkpoint), map_location="cpu", weights_only=True)
        self.net.load_state_dict(state, strict=True)
        self.net.to(device).eval()

    def extract(self, bgr_frame):
        image = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        binary = self.net.binarize(image, force_binary=True)
        # The upstream worker also inverts: ST3D expects white ink on black.
        return 255 - binary
