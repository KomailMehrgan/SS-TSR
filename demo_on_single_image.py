import argparse
import torch
import numpy as np
import cv2
import time
import os
from torch.serialization import add_safe_globals
from PIL import Image
import torchvision.transforms as transforms

# Import model definitions
from Network.srresnet import _NetG
from model.tsrn import TSRN
from model.rdn import RDN
from model.srcnn import SRCNN

# Trust saved architectures
add_safe_globals([_NetG, TSRN])

# Model checkpoint paths
MODEL_PATHS = {
    'ss-srresnet': 'back_up_models/SR/SS-srresnet.pth',
    'ss-tsrn': 'back_up_models/SR/SS-tsnr.pth',
    'tpgsr': 'back_up_models/SR/TPGSR.pth',
    'ss-rdn': 'back_up_models/SR/SS-rdn.pth',
    'ss-srcnn': 'back_up_models/SR/SS-srcnn.pth'
}

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess(image_path, model_name):
    if model_name in ['ss-tsrn', 'tpgsr']:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((64, 16), Image.BICUBIC)
        img_tensor = transforms.ToTensor()(img)

        mask = img.convert('L')
        thres = np.array(mask).mean()
        mask = mask.point(lambda x: 0 if x > thres else 255)
        mask = transforms.ToTensor()(mask)

        im_input = torch.cat((img_tensor, mask), 0).unsqueeze(0)
        im_input_org = np.array(img)
    else:
        im_input_org = cv2.imread(image_path)
        im_input = im_input_org.transpose(2, 0, 1)
        im_input = im_input.reshape(1, im_input.shape[0], im_input.shape[1], im_input.shape[2])
        im_input = torch.from_numpy(im_input / 255.).float()

    return im_input, im_input_org

def postprocess(tensor):
    output = tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    output = (output * 255).astype(np.uint8)
    return output

def load_model(name):
    path = MODEL_PATHS[name]

    if name == 'tpgsr':
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = TSRN(scale_factor=2, width=128, height=32, STN=True, srb_nums=5, mask=True, hidden_units=32)
        model.load_state_dict(checkpoint['state_dict_G'])

    elif name == 'ss-srresnet':
        model = torch.load(path, map_location='cpu', weights_only=False)["model"]

    elif name == 'ss-tsrn':
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = TSRN(scale_factor=2, width=128, height=32, STN=True, srb_nums=12, mask=True, hidden_units=64)
        model.load_state_dict(checkpoint['model'].state_dict())

    elif name == 'ss-rdn':
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = RDN()
        model.load_state_dict(checkpoint['model'].state_dict())

    elif name == 'ss-srcnn':
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = SRCNN()
        model.load_state_dict(checkpoint['model'].state_dict())

    else:
        raise ValueError(f"Unknown model: {name}")

    model.eval()
    return model

def infer_and_save(model, im_input, im_input_org, name):
    print(f"Running inference with {name}...")
    start_time = time.time()

    with torch.no_grad():
        output = model(im_input)

    elapsed = time.time() - start_time
    print(f"{name} Inference time: {elapsed:.3f} seconds")

    result = postprocess(output)
    h, w, _ = result.shape
    print(f"{name} Output size: {w}x{h}")

    out_path = os.path.join(OUTPUT_DIR, f"output_{name}.png")
    cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR
    print(f"Saved: {out_path}")

    # Resize only for display
    result_resized = cv2.resize(result, (512, 128))
    cv2.imshow(f"{name.upper()} Output ({w}x{h})", result_resized)

def main(selected_models, image_path):
    for name in selected_models:
        im_input, im_input_org = preprocess(image_path, name)
        model = load_model(name)
        model.cpu()
        infer_and_save(model, im_input, im_input_org, name)

    input_resized = cv2.resize(im_input_org, (512, 128))
    input_out_path = os.path.join(OUTPUT_DIR, "input.png")
    cv2.imwrite(input_out_path, cv2.cvtColor(im_input_org, cv2.COLOR_RGB2BGR))
    cv2.imshow("Input", input_resized)
    print(f"Saved input image to {input_out_path}")
    cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['ss-rdn'],
                        help="Which models to run: ss-srresnet, ss-tsrn, tpgsr, ss-rdn, ss-srcnn")
    parser.add_argument('--image', default='tets_images/lrTZtesthard_63.png',
                        help="Path to input image")
    args = parser.parse_args()

    main(args.models, args.image)
