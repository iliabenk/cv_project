import numpy as np
import os
import torchvision
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset

#import matplotlib.pyplot as plt
import cv2
import torch
import os
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time
import pickle
import torch.utils
from busProjectTest import runTest


class my_time:

    def tic(self):
        self.t = time.time()

    def toc(self):
        self.elapsed = float(time.time()) - float(self.t)
        s = "elapsed time is %0.3f seconds" % self.elapsed
        print(s)
        return self.elapsed

def create_model():
    INSTANCE_CATEGORY_NAMES = ['background','bus']

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = len(INSTANCE_CATEGORY_NAMES)  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


class bassesDataset(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = [os.path.join(root, file) for file in os.listdir(os.path.join(root)) if '.JPG' in file]
        d = {}
        self.transforms = transforms

    def __getitem__(self, idx):
        # load images ad masks

        img_path = self.imgs[idx]
        img = cv2.imread(img_path, 0)  # Read image with cv2

        # img = Image.open(img_path).convert("RGB")
        img_name = os.path.basename(img_path)
        # get bounding box coordinates for each mask


        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        return len(self.imgs)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
with torch.no_grad():
    model = create_model()
    model.to(device)
    model.load_state_dict(torch.load('nn_busses.pt'))
    model.eval()
print('Model was loaded')




def create_basic_prediction(estimatedAnnFileName, busDir, batch_size = 30):
    transform = T.Compose([T.ToTensor()])
    dataset = bassesDataset(busDir, transform)


    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    start = True
    t = my_time()

    with torch.no_grad():
        t2 = my_time()
        t2.tic()
        for curr_sample in data_loader:
            print('start forward')
            t.tic()
            curr_sample = curr_sample.to(device)
            curr_pred = model(curr_sample)
            t.toc()
            if start:
                pred = curr_pred
                start = False
            else:
                pred = pred + curr_pred
            print(len(pred))
    t2.toc()
    print('finished pred')
    files_path_list = [os.path.join(busDir, file) for file in os.listdir(busDir) if '.JPG' in file]

    with open('KAZE_trained_features.pickle', 'rb') as handle:
        des_label_list = pickle.load(handle)

    dict_color = {'red': 6, 'blue': 5, 'white': 3, 'grey': 4, 'orange': 2, 'green': 1}
    t = my_time()
    t.tic()
    with open (estimatedAnnFileName, 'w') as fp_anns:
        for indx, file_path in enumerate(files_path_list):

            boxes, pred_cls = object_detection_api(pred[indx],file_path, threshold=0.9, train_des_label=des_label_list)
            # boxes, pred_cls = object_detection_api([],file_path, threshold=0.9, train_des_label=des_label_list)

            strToWrite = os.path.basename(file_path) + ":"

            for i in range(len(boxes)):
                min_coor = boxes[i][0]
                max_coor = boxes[i][1]
                x_min = int(min_coor[0])
                y_min = int(min_coor[1])
                x_max = int(max_coor[0])
                y_max = int(max_coor[1])
                width = x_max - x_min
                height = y_max - y_min

                # ann = [x_min, y_min, width, height, convert_label_name_to_label_num(pred_cls[i])]
                ann = [x_min, y_min, width, height, dict_color[pred_cls[i]]]

                posStr = [str(x) for x in ann]
                posStr = ','.join(posStr)
                strToWrite += '[' + posStr + ']'
                if (i == int(len(boxes)) - 1):
                    strToWrite += '\n'
                else:
                    strToWrite += ','

            fp_anns.write(strToWrite)
    t.toc()
            #print(strToWrite)

def run_gpu_faster(estimatedAnnFileName, busDir):
    transform = T.Compose([T.ToTensor()])
    dataset = bassesDataset(busDir, transform)
    files_path_list = [os.path.join(busDir, file) for file in os.listdir(busDir) if '.JPG' in file]
    with open('KAZE_trained_features.pickle', 'rb') as handle:
        des_label_list = pickle.load(handle)
    global_indx = 0
    t = my_time()
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=30, shuffle=False, num_workers=1)
    start = True

    with open (estimatedAnnFileName, 'w') as fp_anns:

      with torch.no_grad():
        # t2 = my_time()
        # t2.tic()
        for curr_sample in data_loader:
            print('start forward')
            t.tic()
            curr_sample = curr_sample.to(device)
            curr_pred = model(curr_sample)
            #return curr_sample
            t.toc()
            for i in range(len(curr_sample)):
              file_path = files_path_list[global_indx]
              global_indx +=1
              boxes, pred_cls = object_detection_api_faster(curr_pred[i], curr_sample[i].cpu().numpy()[0,:,:] ,file_path, threshold=0.9, train_des_label=des_label_list)
              strToWrite = os.path.basename(file_path) + ":"

              for i in range(len(boxes)):
                  min_coor = boxes[i][0]
                  max_coor = boxes[i][1]
                  x_min = int(min_coor[0])
                  y_min = int(min_coor[1])
                  x_max = int(max_coor[0])
                  y_max = int(max_coor[1])
                  width = x_max - x_min
                  height = y_max - y_min
                  dict_color = {'red':6, 'blue':5 ,'white':3, 'grey':4, 'orange':2, 'green':1}
                  ann = [x_min, y_min, width, height, dict_color[pred_cls[i]]]

                  posStr = [str(x) for x in ann]
                  posStr = ','.join(posStr)
                  strToWrite += '[' + posStr + ']'
                  if (i == int(len(boxes)) - 1):
                      strToWrite += '\n'
                  else:
                      strToWrite += ','

              fp_anns.write(strToWrite)


def object_detection_api_pr(pred, img, img_path, threshold=0.9, rect_th=3, text_size=3, text_th=3, train_des_label=[]):
    #img = cv2.imread(img_path, 0)  # Read image with cv2
    boxes, pred_cls = get_prediction(pred, img, img_path, threshold, train_des_label)  # Get predictions --- #img_path is only for testing, not needed later

    return boxes, pred_cls

def object_detection_api(pred, img_path, threshold=0.9, rect_th=3, text_size=3, text_th=3, train_des_label=[]):
    img = cv2.imread(img_path, 0)  # Read image with cv2
    boxes, pred_cls = get_prediction(pred, img, img_path, threshold, train_des_label)  # Get predictions --- #img_path is only for testing, not needed later

    return boxes, pred_cls

def convert_label_name_to_label_num(label):
    if 'red' ==  label:
        return 6
    elif 'blue' == label:
        return 5
    elif 'white' == label:
        return 3
    elif 'grey' == label:
        return 4
    elif 'green' == label:
        return 1
    elif 'orange' == label:
        return 2

def get_prediction(pred, img, img_path, threshold, des_label_train=[]): #img_path is only for testing, not needed later
  # img = Image.open(img_path) # Load the image
  # transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
  # img2 = transform(img) # Apply the transform to the image
  # pred = model([img2]) # Pass the image to the model
  # pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
  # pred_score = list(pred[0]['scores'].detach().numpy())

  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred['boxes'].detach().cpu().numpy())] # Bounding boxes
  pred_score = list(pred['scores'].detach().cpu().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x > threshold] # Get list of index with score greater than threshold.

  pred_class = []
  pred_img = []
  if pred_t:
      pred_t = pred_t[-1]
      pred_boxes = pred_boxes[:pred_t+1]

      for i in range(len(pred_boxes)):
          min_coor = pred_boxes[i][0]
          max_coor = pred_boxes[i][1]
          x_min = int(min_coor[0])
          y_min = int(min_coor[1])
          x_max = int(max_coor[0])
          y_max = int(max_coor[1])
          curr_img = img[y_min : (y_max + 1), x_min : (x_max + 1)]
          pred_class.append(predict_label(curr_img, des_label_train, img_path))
          pred_img.append(curr_img)
  return pred_img, pred_boxes, pred_class


def predict_label(img, des_label_train, file_path):
    _, des_test = get_features(img)

    ratio = 0.75
    good_matches_limit = 250

    score_d = {'blue': 0, 'red': 0, 'white': 0, 'green': 0, 'orange': 0, 'grey': 0}
    amount_train_per_label_d = {'blue': 0, 'red': 0, 'white': 0, 'green': 0, 'orange': 0, 'grey': 0}
    is_skip_decision_by_score = False

    for train_des_label in des_label_train:
        label_train = train_des_label[0]
        des_train = train_des_label[1]
        train_file_name = train_des_label[2]

        if os.path.basename(file_path).replace('.JPG', '') in train_file_name: #TODO only for testing, not needed in real run
            continue

        amount_train_per_label_d[label_train] += 1 #TODO only for testing, in submission these values are already known

        amount_good_matching_points = get_amount_good_matching_points(des_test, des_train, ratio=ratio, k=2, good_matches_limit=good_matches_limit)
        score_d[label_train] += amount_good_matching_points

        if amount_good_matching_points >= good_matches_limit:
            best_score_label = label_train
            is_skip_decision_by_score = True
            #print(best_score_label)
            #print(amount_good_matching_points)
            break


    if is_skip_decision_by_score is False:
        score_d = {label:(value / amount_train_per_label_d[label]) for (label, value) in score_d.items()}

        best_score_label = get_best_label_candidate(score_d)

    return best_score_label


def get_best_label_candidate(score):
        best_score_label = 'red'
        best_score = 0

        for label, label_score in score.items():
            if label_score > best_score:
                best_score = label_score
                best_score_label = label

        return best_score_label

def get_amount_good_matching_points(des1, des2, ratio=0.75, k=2, good_matches_limit=10000):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = 0

    for m,n in matches:
        if m.distance < ratio * n.distance:
            good_matches += 1

        if good_matches >= good_matches_limit:
            break

    return good_matches


def get_features(img):
    #print(img.shape)
    img = cv2.resize(img, (300,225))
    surf = cv2.AKAZE_create()

    _, des = surf.detectAndCompute(img, None)

    return _, des

if __name__ == "__main__":
    t = my_time()

    t.tic()



    run_gpu('annotationsTrain_test.txt', \
         '/Users/omriefroni/PycharmProjects/comp_vision_ex2/cv_project/nn/final_dir/buses')
    # runTest("annotationsTrain.txt", "annotationsTrain_test.txt", 'buses/', 'result/', 10)

    elapsed = t.toc()
    print('elapsed')
