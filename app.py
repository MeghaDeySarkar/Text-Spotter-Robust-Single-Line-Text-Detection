from flask import Flask, render_template, request
import cv2
import torch
import albumentations
from modular_code.src.source.network import ConvRNN
from modular_code.src.utils import load_obj

app = Flask(__name__)

int2char = load_obj(r"D:\Megha\PPro\NLP-projects-completed\textdetectioninimage\modular_code\input\data\int2char.pkl")

model = ConvRNN(len(int2char))

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    
    img_aug = albumentations.Compose([
        albumentations.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True)
    ])
    
    augmented = img_aug(image=img)
    img = augmented["image"]
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)

    return img


def perform_text_detection(image_path):
    img = preprocess_image(image_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(r"D:\Megha\PPro\NLP-projects-completed\textdetectioninimage\modular_code\output\models\model.pth", map_location=device))

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    with torch.no_grad():
        out = model(img)

    out = torch.squeeze(out, 0)
    out = out.softmax(1)
    pred = torch.argmax(out, 1)
    pred = pred.tolist()
    int2char[0] = "ph"
    out = [int2char[i] for i in pred]
    res = [out[0]]
    for i in range(1, len(out)):
        if out[i] != out[i - 1]:
            res.append(out[i])
    res = [i for i in res if i != "ph"]
    res = "".join(res)

    return res


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect_text', methods=['POST'])
def handle_text_detection():
    
    image_file = request.files['image']
    
    temp_image_path = 'temp_image.jpg'
    image_file.save(temp_image_path)
    
    prediction = perform_text_detection(temp_image_path)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
