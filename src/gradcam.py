import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
import cv2

from model import WasteClassifier
from dataset import DEFAULT_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM explanation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/gradcam.png")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--image_size", type=int, default=224)
    return parser.parse_args()


def get_preprocess(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def apply_colormap_on_image(org_im, activation, colormap_name="jet"):
    heatmap = cv2.applyColorMap(np.uint8(255 * activation), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]
    cam = heatmap + np.float32(org_im)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax().item()

        target = output[0, class_idx]
        target.backward(retain_graph=True)

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=0)
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
        return cam, class_idx

    def close(self):
        for handle in self.hook_handles:
            handle.remove()


def run_gradcam(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model = WasteClassifier(num_classes=len(DEFAULT_CLASSES), backbone_name=args.backbone, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if hasattr(model.backbone, "layer4"):
        target_layer = model.backbone.layer4[-1]
    elif hasattr(model.backbone, "features"):
        target_layer = model.backbone.features[-1]
    else:
        raise RuntimeError("Cannot find convolutional block for target layer")

    gradcam = GradCAM(model, target_layer)

    preprocess = get_preprocess(args.image_size)
    image = Image.open(args.image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)

    cam, predicted_class = gradcam(input_tensor)
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)[0]
    top_prob = probs[predicted_class].item()

    orig = np.array(image.resize((args.image_size, args.image_size))).astype(np.float32) / 255.0
    vis = apply_colormap_on_image(orig, cam)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(orig)
    ax[0].axis("off")
    ax[0].set_title("Original image")
    ax[1].imshow(vis)
    ax[1].axis("off")
    ax[1].set_title(f"Grad-CAM: {DEFAULT_CLASSES[predicted_class]} ({top_prob:.2f})")
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved Grad-CAM to {args.output}")
    gradcam.close()


if __name__ == "__main__":
    args = parse_args()
    run_gradcam(args)
