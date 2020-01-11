import torchvision
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
import torch
import os
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time
import pickle
########### start configs
is_save_img_box = False
box_images_folder_path = "/Users/iliabenkovitch/Documents/Computer_Vision/git/git_orign_cv_project/nn/boxes"

if not os.path.exists(box_images_folder_path):
    os.makedirs(box_images_folder_path)

test_path = "/Users/iliabenkovitch/Documents/Computer_Vision/git/git_orign_cv_project/nn/all_images"
output_path = os.path.join(test_path, "predictions")


if not os.path.exists(output_path):
    os.makedirs(output_path)

model_pt_path = "/Users/iliabenkovitch/Documents/Computer_Vision/git/git_orign_cv_project/nn/nn_busses_omri.pt"

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
def get_features(img):
    surf = cv2.KAZE_create()

    kp, des = surf.detectAndCompute(img, None)

    return kp, des

def get_amount_good_matching_points(des1, des2, ratio=0.75, k=2):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = 0

    for m,n in matches:
        if m.distance < ratio * n.distance:
            good_matches += 1

    return good_matches

def get_best_label_candidate(score):
        best_score_label = 'none'
        best_score = 0

        for label, label_score in score.items():
            if label_score > best_score:
                best_score = label_score
                best_score_label = label

        return best_score_label

def predict_label(img, des_label_train, file_path):
    _, des_test = get_features(img)

    ratio = 0.75

    score_d = {'blue': 0, 'red': 0, 'white': 0, 'green': 0, 'orange': 0, 'grey': 0}
    amount_train_per_label_d = {'blue': 0, 'red': 0, 'white': 0, 'green': 0, 'orange': 0, 'grey': 0}
    is_skip_decision_by_score = False

    for train_des_label in des_label_train:
        label_train = train_des_label[0]
        des_train = train_des_label[1]
        train_file_name = train_des_label[2]

        if os.path.basename(file_path).replace('.JPG', '') in train_file_name:
            continue

        amount_train_per_label_d[label_train] += 1 #TODO only for testing, in submission these values are already known

        amount_good_matching_points = get_amount_good_matching_points(des_test, des_train, ratio=ratio, k=2)
        score_d[label_train] += amount_good_matching_points

        if amount_good_matching_points >= 250:
            best_score_label = label_train
            is_skip_decision_by_score = True
            break


    if is_skip_decision_by_score is False:
        score_d = {label:(value / amount_train_per_label_d[label]) for (label, value) in score_d.items()}

        best_score_label = get_best_label_candidate(score_d)

    return best_score_label

def get_prediction(img_path, threshold, des_label_train=[]):
  # img = Image.open(img_path) # Load the image
  # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) # Load the image
  img = cv2.imread(img_path, 0) # Load the image
  transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
  img2 = transform(img) # Apply the transform to the image
  pred = model([img2]) # Pass the image to the model
  # pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
  pred_score = list(pred[0]['scores'].detach().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x > threshold] # Get list of index with score greater than threshold.
  print(img_path + "    scores: " + str(pred_score))

  pred_class = []

  if pred_t:
      pred_t = pred_t[-1]
      pred_boxes = pred_boxes[:pred_t+1]
      pred_class = pred_class[:pred_t+1]

      for i in range(len(pred_boxes)):
          min_coor = pred_boxes[i][0]
          max_coor = pred_boxes[i][1]
          x_min = int(min_coor[0])
          y_min = int(min_coor[1])
          x_max = int(max_coor[0])
          y_max = int(max_coor[1])

          pred_class.append(predict_label(img[y_min : (y_max + 1), x_min : (x_max + 1)], des_label_train, img_path))

  return pred_boxes, pred_class

# different from the one a "fine_tune_model" (same functionality different inputs for saving)
def object_detection_api(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3, train_des_label=[]):
    boxes, pred_cls = get_prediction(img_path, threshold, train_des_label)  # Get predictions
    img = cv2.imread(img_path)  # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    if is_save_img_box is True:
        img_for_box_save = cv2.imread(img_path)  # Read image with cv2

    for i in range(len(boxes)):
        if is_save_img_box is True:
            save_img_box(img_for_box_save, img_path, boxes[i][0], boxes[i][1])

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

def save_img_box(img, img_path, min_coor, max_coor):
    save_img_box.counter += 1
    x_min = int(min_coor[0])
    y_min = int(min_coor[1])
    x_max = int(max_coor[0])
    y_max = int(max_coor[1])

    box_img = img[y_min : (y_max + 1), x_min : (x_max + 1)]

    cv2.imwrite(os.path.join(box_images_folder_path, os.path.basename(img_path).replace('.JPG', '') + '_' + str(save_img_box.counter) + "_.JPG"), box_img)
save_img_box.counter = 0

# loop on images
t = time.time()

with open('/Users/iliabenkovitch/Documents/Computer_Vision/git/git_orign_cv_project/nn/save_data.pickle', 'rb') as handle:
        des_label_list_from_file = pickle.load(handle)

for file in file_list:
    curr_path = os.path.join(test_path, file)
    #curr_path = path
    object_detection_api(curr_path, threshold=0.9, train_des_label=des_label_list_from_file)

print(str(time.time() - t) + ' secs for test')