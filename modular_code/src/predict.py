import cv2
import torch
import albumentations
from utils import load_obj
from source.network import ConvRNN
from argparse import ArgumentParser


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--test_img", default="D:/Megha/PPro/textdetectioninimage/data/data/images/00000017.jpg", help="path to test image")
    parser.add_argument("--model_path", default="D:/Megha/PPro/textdetectioninimage/modular_code/modular_code/output/models/model.pth", help="path to the saved model")
    parser.add_argument("--int2char_path", default="D:/Megha/PPro/textdetectioninimage/modular_code/modular_code/input/data/int2char.pkl", help="path to int2char")
    opt = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    int2char = load_obj(opt.int2char_path)
    n_classes = len(int2char)

    model = ConvRNN(n_classes)
    model.load_state_dict(torch.load(opt.model_path, map_location=device))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    img = cv2.imread(opt.test_img)
    img_aug = albumentations.Compose(
            [albumentations.Normalize(mean, std,
                                      max_pixel_value=255.0,
                                      always_apply=True)]
        )
    augmented = img_aug(image=img)
    img = augmented["image"]
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)
    img = img.to(device)

    out = model(img)
    out = torch.squeeze(out, 0)
    out = out.softmax(1)
    pred = torch.argmax(out, 1)
    pred = pred.tolist()
    int2char[0] = "ph"
    out = [int2char[i] for i in pred]

    res = list()
    res.append(out[0])
    for i in range(1, len(out)):
        if out[i] != out[i - 1]:
            res.append(out[i])
    res = [i for i in res if i != "ph"]
    res = "".join(res)
    print(res)
