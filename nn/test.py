import torchvision
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
import torch
import os
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

########### start configs
is_save_img_box = False
box_images_folder_path = "/Users/iliabenkovitch/Documents/Computer_Vision/git/git_orign_cv_project/nn/boxes"

if not os.path.exists(box_images_folder_path):
    os.makedirs(box_images_folder_path)

test_path = "/Users/iliabenkovitch/Documents/Computer_Vision/git/git_orign_cv_project/nn/all_images/test"
output_path = os.path.join(test_path, "predictions")


if not os.path.exists(output_path):
    os.makedirs(output_path)

model_pt_path = "/Users/iliabenkovitch/Documents/Computer_Vision/git/git_orign_cv_project/nn/nn_busses_2.pt"

# Defines the labels for the model, change to colors if you want.
COCO_INSTANCE_CATEGORY_NAMES = ['background','bus']
# COCO_INSTANCE_CATEGORY_NAMES = ['background', 'green', 'yellow', 'white', 'grey', 'blue', 'red']


########### end configs

# create model - same function
def create_model():
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = len(COCO_INSTANCE_CATEGORY_NAMES)  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
# load model
model = create_model()
model.load_state_dict(torch.load(model_pt_path))
model.eval()
# path for the test images

file_list = [file for file in os.listdir(test_path) if '.JPG' in file]

# COCO_INSTANCE_CATEGORY_NAMES = [
#     '_background_', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
#     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#     'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
#     'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#     'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#     'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#     'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
#     'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
#     'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# predict - different from the one a "fine_tune_model" (same functionality different inputs for saving)
def get_prediction(img_path, threshold):
  img = Image.open(img_path) # Load the image
  transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
  img = transform(img) # Apply the transform to the image
  pred = model([img]) # Pass the image to the model
  pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
  pred_score = list(pred[0]['scores'].detach().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x > threshold] # Get list of index with score greater than threshold.
  print(img_path + "    scores: " + str(pred_score))

  if pred_t:
      pred_t = pred_t[-1]
      pred_boxes = pred_boxes[:pred_t+1]
      pred_class = pred_class[:pred_t+1]

  return pred_boxes, pred_class

# different from the one a "fine_tune_model" (same functionality different inputs for saving)
def object_detection_api(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    boxes, pred_cls = get_prediction(img_path, threshold)  # Get predictions
    img = cv2.imread(img_path)  # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    if is_save_img_box is True:
        img_for_box_save = cv2.imread(img_path)  # Read image with cv2

    for i in range(len(boxes)):
        if is_save_img_box is True:
            save_img_box(img_for_box_save, boxes[i][0], boxes[i][1])

        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0),
                      thickness=rect_th)  # Draw Rectangle with the coordinates
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
                    thickness=text_th)  # Write the prediction class

    plt.figure(figsize=(20, 30))  # display the output image
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    #plt.show()
    plt.savefig(os.path.join(output_path, os.path.basename(img_path).replace('.JPG','_prid.JPG')))
    #plt.savefig('/Users/omriefroni/PycharmProjects/comp_vision_ex2/Train/busesTrain/pred_DSCF1016.JPG')
#file_list = ['/Users/omriefroni/PycharmProjects/comp_vision_ex2/Train/busesTrain/DSCF1016.JPG']

def save_img_box(img, min_coor, max_coor):
    save_img_box.counter += 1
    x_min = int(min_coor[0])
    y_min = int(min_coor[1])
    x_max = int(max_coor[0])
    y_max = int(max_coor[1])

    box_img = img[y_min : (y_max + 1), x_min : (x_max + 1)]

    cv2.imwrite(os.path.join(box_images_folder_path, str(save_img_box.counter) + ".JPG"), box_img)

save_img_box.counter = 0

# loop on images
for file in file_list:
    curr_path = os.path.join(test_path, file)
    #curr_path = path
    object_detection_api(curr_path, threshold=0.9)