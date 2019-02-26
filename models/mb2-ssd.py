import torch
from torchsummary import summary
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

model_path = './models/mb2-ssd-lite-mp-0_686.pth'
label_path = './models/voc-model-labels.txt'

class_names = [name.strip() for name in open(label_path).readlines()]

net = create_mobilenetv2_ssd_lite(len(class_names), is_test=False)
net.load(model_path)


# Freezer the SSD
for param in net.parameters():
    param.requires_grad = False


orig_image = cv2.imread(image_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
boxes, labels, probs = predictor.predict(image, 10, 0.4)
#summary(net, (3,300,300))
#print(net)
# summary(model.cuda(), (INPUT_SHAPE))
# predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
