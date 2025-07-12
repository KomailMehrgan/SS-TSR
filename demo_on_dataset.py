import argparse
import os
import time
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.serialization import add_safe_globals

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
    'ss-rdn' : 'back_up_models/SR/SS-rdn.pth',
    'ss-srcnn' : 'back_up_models/SR/SS-srcnn.pth'
}

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

    model.eval()
    return model

def infer_and_save(model, im_input, save_path):
    with torch.no_grad():
        output = model(im_input)
    result = postprocess(output)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result_resized = cv2.resize(result, (512, 128))
    cv2.imwrite(save_path, cv2.cvtColor(result_resized, cv2.COLOR_RGB2BGR))  # 🔧 RGB to BGR
    print(f"Saved: {save_path}")

def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))

def run_on_dataset(dataset_folder, selected_models):
    lr_folder_name = os.path.basename(os.path.normpath(dataset_folder))
    image_paths = []

    for root, _, files in os.walk(dataset_folder):
        for file in files:
            if is_image_file(file):
                image_paths.append(os.path.join(root, file))

    for model_name in selected_models:
        print(f"\n==> Running model: {model_name}")
        model = load_model(model_name)
        model.cpu()

        for img_path in image_paths:
            try:
                im_input, _ = preprocess(img_path, model_name)
                img_name = os.path.basename(img_path)
                save_path = os.path.join("output", "dataset", model_name, lr_folder_name, img_name)
                infer_and_save(model, im_input, save_path)
            except Exception as e:
                print(f"Failed on {img_path} with {model_name}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['ss-srcnn'],
                        help="Models to run: ss-tsrn, ss-srresnet, tpgsr,ss-rdn, ss-srcnn" )
    parser.add_argument('--dataset', type=str, default=R"D:\Researches\SR\FirstPaperCode\OCR\Benchmark_ocr\dataset\TextZoom_testeasy\LR_images",
                        help="Path to dataset folder containing LR images")
    args = parser.parse_args()

    run_on_dataset(args.dataset, args.models)
