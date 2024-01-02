import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets

#pretrained model is model that have been trained from imagenet 

# load the model

model = torch.load(r'C:\Users\Safwan\Downloads\Sem 1 2324\Machine Vision\amir.pt')
model.eval()


model.to('cpu')

class_labels = ['1', '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']

# no 2(peace),17(shaka),19(thumbs up)

def preprocess_image(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]) # (mean, standard deviation) -> get average
    ])

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(frame)
    frame = frame.unsqueeze(0) # add batch dimension

    return frame

# reading the image
capture = cv2.VideoCapture(0)

while True:

    isTrue, frame = capture.read()
    
    # frame_tensor = preprocess_image(frame)

    # # feedforward/ inference
    # with torch.no_grad():
    #     output = model(frame_tensor)

    # # postprocess output/ label
    # _,predicted_class = output.max(1)
    # predicted_class = predicted_class.item()

    # predicted_class_name = class_labels[predicted_class]

    # # label = f"Class: {predicted_class}"
    cv2.putText(frame,"Door: Closed",(10,10),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
    results = model(frame, stream = True)
    # cv2.putText(frame,predicted_class_name,(10,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
    for r in results:
        boxes = r.boxes
        for bbox in boxes:
            x1,y1,x2,y2 = bbox.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            
            cls_idx = int(bbox.cls[0])
            cls_name = model.names[cls_idx]

            conf = round(float(bbox.conf[0]),2)
            
            if cls_name == "2" or cls_name == "3" or cls_name == "4":
                cv2.rectangle(frame, (x1,y1), (x2,y2),(255,0,0),4)
                cv2.putText(frame,f'{cls_name} {conf}',(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
                cv2.putText(frame,"Door: Opened",(10,10),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)

    cv2.imshow('Hand Gesture', frame)

    cv2.waitKey(1)


capture.release()

cv2.destroyAllWindows()

