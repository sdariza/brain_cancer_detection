from flask import Flask, render_template, request
from models import *
from utils.utils import * 
from utils.datasets import *
import cv2
import torch
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet('config/yolov3-custom.cfg', img_size=512).to(device)
model.load_state_dict(torch.load('checkpoints/yolov3_ckpt_99.pth', map_location=torch.device('cpu')))
model.eval()
classes = load_classes('data/custom/classes.name')
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

def Convertir_RGB(img):
    # Convertir Blue, green, red a Red, green, blue
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img

def Convertir_BGR(img):
    # Convertir red, blue, green a Blue, green, red
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img




app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./static/imgs/" + imagefile.filename
    imagefile.save(image_path)
    frame = cv2.imread(image_path)
    RGBimg=Convertir_RGB(frame)
    imgTensor = transforms.ToTensor()(RGBimg)
    imgTensor, _ = pad_to_square(imgTensor, 0)
    imgTensor = resize(imgTensor, 512)
    imgTensor = imgTensor.unsqueeze(0)
    imgTensor = Variable(imgTensor.type(Tensor))
    with torch.no_grad():
        detections = model(imgTensor)
        detections = non_max_suppression(detections,0.98, 0.4)
    for detection in detections:
        if detection is not None:
            detection = rescale_boxes(detection, 512, RGBimg.shape[:2])
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                box_w = x2 - x1
                box_h = y2 - y1
                color = [int(c) for c in colors[int(cls_pred)]]
                print("Se detectó {} en X1: {}, Y1: {}, X2: {}, Y2: {}".format(classes[int(cls_pred)], x1, y1, x2, y2))
                frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color, 3)
                cv2.putText(frame, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)# Nombre de la clase detectada
                cv2.putText(frame, str("%.2f" % float(conf)), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2) # Certeza de prediccion de la clase
    cv2.imwrite(filename=image_path,img=frame)
    
    return render_template('index.html', img_url = image_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)