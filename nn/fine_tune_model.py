import os
import numpy as np
import torch
from PIL import Image
import torchvision
from torch.utils.data import Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch.utils
import ast
from engine import train_one_epoch, evaluate
import utils
import torchvision.transforms as T
import copy
import cv2
import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import shutil
import operator
import time
import pickle

########### start configs
#data_path = "buses"
train_perc = 0.8
#model_file_name = "nn_busses_3.pt"

#train_color_folder_path = "/Users/iliabenkovitch/Documents/Computer_Vision/git/git_orign_cv_project/nn/boxes"
#test_color_folder_path = "/Users/iliabenkovitch/Documents/Computer_Vision/git/git_orign_cv_project/nn/boxes_test"

# Defines the labels for the model, change to colors if you want.
COCO_INSTANCE_CATEGORY_NAMES = ['background','bus']
# COCO_INSTANCE_CATEGORY_NAMES = ['background', 'green', 'yellow', 'white', 'grey', 'blue', 'red']
num_epochs = 30

test_threshold = 0.9

num_train_per_label = 12
is_train_surf_on_all_images = True
is_random_amount_surf_train_per_label = False
is_get_statistics = True
min_num_imgs_for_label = 5
surf_data_path = "/Users/iliabenkovitch/Documents/Computer_Vision/git/git_orign_cv_project/nn/all_boxes_with_names"
surf_train_data = "/Users/iliabenkovitch/Documents/Computer_Vision/git/git_orign_cv_project/nn/train_boxes"
surf_test_data = "/Users/iliabenkovitch/Documents/Computer_Vision/git/git_orign_cv_project/nn/test_boxes"

########### end configs

# get prediction from model (was taken from an RCNN tutorial)
def get_prediction(model, img_path, threshold):
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  img = cv2.imread(img_path) # Load the image


  transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
  img = transform(img) # Apply the transform to the image
  img = img.to(device)
  pred = model([img]) # Pass the image to the model

  pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())] # Get the Prediction Score
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].cpu().detach().numpy())] # Bounding boxes
  pred_score = list(pred[0]['scores'].detach().cpu().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x > threshold]
  if len(pred_t) != 0:
      pred_t =pred_t[-1] # Get list of index with score greater than threshold.
      pred_boxes = pred_boxes[:pred_t + 1]
      pred_class = pred_class[:pred_t + 1]
  else:
      pred_boxes = []
      pred_class =[]

  return pred_boxes, pred_class

# get prediction from model (was taken from an RCNN tutorial)
def object_detection_api(model, output_dir,img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    boxes, pred_cls = get_prediction(model, img_path, threshold)  # Get predictions
    img = cv2.imread(img_path)  # Read image with cv2
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    for i in range(len(boxes)):
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0),
                      thickness=rect_th)  # Draw Rectangle with the coordinates
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
                    thickness=text_th)  # Write the prediction class
    #plt.figure(figsize=(20, 30))  # display the output image
    #plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    #plt.show()
    cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path).replace('.JPG','_prid.JPG')),img)

# Test function - FIXME need to add aoc score as defined, Changes at DataLoarder has to be done
def test(model, epoch, output_path ):
    file_list = [os.path.join(output_path, file) for file in os.listdir(output_path) if '.JPG' in file]
    model.eval()
    output_fol = os.path.join(output_path, 'ep' + str(epoch))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if not os.path.exists(output_fol):
        os.makedirs(output_fol)
    for img in file_list:
        object_detection_api(model,output_fol,img, threshold=test_threshold )

# convert from gt format to NN format
def convert_BB_to_net(x_min,y_min,width,hight):
    x_max = x_min + width
    y_max = y_min + hight
    return [x_min, y_min, x_max, y_max]
# FIXME - need to add invers function for creating output text file

# function for NN
def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    #if train:
    #    transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# dataset for NN

class bassesDataset(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = [path for path in os.listdir(os.path.join(root)) if '.JPG' in path]
        d = {}
        with open(root + "/annotationsTrain.txt") as f:
            for line in f:
                (key, val) = line.split(':')
                val = val.replace('\n', '')
                d[key] = ast.literal_eval(val)
        self.gt_dict = d
        self.transforms = transforms

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img_name = os.path.basename(img_path)
        # get bounding box coordinates for each mask
        boxes_list = []
        labels_list = []
        labeled_data_raw = np.array(self.gt_dict[img_name])

        if len(labeled_data_raw.shape)==1:
            num_objs = 1
            xmin = labeled_data_raw[0]
            ymin = labeled_data_raw[1]
            width = labeled_data_raw[2]
            hight = labeled_data_raw[3]

            if len(COCO_INSTANCE_CATEGORY_NAMES) is 2:
                label = 1
            else:
                label = labeled_data_raw[4]

            boxes_list.append(convert_BB_to_net(xmin, ymin, width, hight))
            labels_list.append(label)
        else:
            num_objs = len(labeled_data_raw)
            for i in range(num_objs):
                pos = labeled_data_raw[i]
                xmin = pos[0]
                ymin = pos[1]
                width = pos[2]
                hight = pos[3]

                if len(COCO_INSTANCE_CATEGORY_NAMES) is 2:
                    label = 1
                else:
                    label = pos[4]

                boxes_list.append(convert_BB_to_net(xmin,ymin,width,hight))
                labels_list.append(label)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes_list, dtype=torch.float32)
        # there is only one class
        #FIXME - change to correct label
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        labels = torch.tensor(labels_list, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = torch.tensor([0])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

# Create the model - load pre-trained model
def create_model():
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = len(COCO_INSTANCE_CATEGORY_NAMES)  # 1 class (person) + background #FIXME change to 7 label::: Changed to length of labels list
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def train_k_means(train_folder_path, is_hsv=True):
    files_path_list = [os.path.join(train_folder_path, file) for file in os.listdir(train_folder_path) if ".JPG" in file]

    if is_hsv is True:
        h_all = np.array([0])
        s_all = np.array([0])
        v_all = np.array([0])

        for file_path in files_path_list:
            img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(img)

            h_all = np.append(h_all, h.flatten())
            s_all = np.append(s_all, s.flatten())
            v_all = np.append(v_all, v.flatten())

        h_all = np.delete(h_all, 0)
        s_all = np.delete(s_all, 0)
        v_all = np.delete(v_all, 0)
        hsv_all = np.transpose(np.array([h_all, s_all, v_all]))

        kmeans = KMeans(n_clusters=6)
        kmeans.fit(hsv_all)

        print(kmeans.cluster_centers_.astype(int))

    else:
        r_all = np.array([0])
        g_all = np.array([0])
        b_all = np.array([0])

        for file_path in files_path_list:
            img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

            r, g, b = cv2.split(img)
            r_all = np.append(r_all, r.flatten())
            g_all = np.append(g_all, g.flatten())
            b_all = np.append(b_all, b.flatten())

        r_all = np.delete(r_all, 0)
        g_all = np.delete(g_all, 0)
        b_all = np.delete(b_all, 0)

        rgb_all = np.transpose(np.array([r_all, g_all, b_all]))

        kmeans = KMeans(n_clusters=6)
        kmeans.fit(rgb_all)

        print(kmeans.cluster_centers_.astype(int))
        # fig = plt.figure()
        # ax = Axes3D(fig)

        # for color in kmeans.cluster_centers_.astype(int):
        #     print(color)
        #     ax.scatter(color[0], color[1], color[2], color=rgb_to_hex(color))
        #
        # plt.show()

    return kmeans.cluster_centers_.astype(int)

def rgb_to_hex(rgb):
        return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))

# def test_k_means(train_folder_path): #TODO: write the function
#     files_path_list = [os.path.join(train_folder_path, file) for file in os.listdir(train_folder_path) if ".JPG" in file]
#
#     for file_path in files_path_list:
#         img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

def get_surf_features(img, hessian_thr=400):
    surf = cv2.AKAZE_create()

    img = cv2.resize(img, (300,225))

    _, des = surf.detectAndCompute(img, None)

    return _, des

def train_SURF(train_folder_path):
    files_path_list = [os.path.join(train_folder_path, file) for file in os.listdir(train_folder_path) if ".JPG" in file]
    des_label_list = []
    amount_labels = {'blue': 0, 'red': 0, 'white': 0, 'green': 0, 'orange': 0, 'grey': 0}

    for file_path in files_path_list:
        img = cv2.imread(file_path, 0)

        _, des = get_surf_features(img)

        label = get_label_from_basename(os.path.basename(file_path))
        amount_labels[label] += 1

        des_label_list.append((label, des, os.path.basename(file_path))) #file name is only for testing, not needed

    print(amount_labels)
    return des_label_list

def test_SURF(test_surf_folder_path, des_label_list, is_get_statistics=False):
    files_path_list = [os.path.join(test_surf_folder_path, file) for file in os.listdir(test_surf_folder_path) if ".JPG" in file]
    num_correct_pred = 0
    num_test_imgs = len(files_path_list)

    if is_get_statistics:
        correct_preds_prob = []
        wrong_preds_prob = []
        wrong_pred_images = []

    for file_path in files_path_list:
        img = cv2.imread(file_path, 0)

        _, des_test = get_surf_features(img)

        ratio = 0.75
        is_found_good_candidate = False

        score_d = {'blue': 0, 'red': 0, 'white': 0, 'green': 0, 'orange': 0, 'grey': 0}
        amount_train_per_label_d = {'blue': 0, 'red': 0, 'white': 0, 'green': 0, 'orange': 0, 'grey': 0}
        is_skip_decision_by_score = False
        max_score = 0

        for train_des_label in des_label_list:
            label_train = train_des_label[0]
            des_train = train_des_label[1]
            train_file_name = train_des_label[2]

            if train_file_name == os.path.basename(file_path):
                continue

            amount_train_per_label_d[label_train] += 1 #TODO only for testing, in submission these values are already known

            amount_good_matching_points = get_amount_good_matching_points(des_test, des_train, ratio=ratio, k=2)

            # print(label_train + '    ' + str(amount_good_matching_points))

            score_d[label_train] += amount_good_matching_points

            if amount_good_matching_points >= 80:
                best_score_label = label_train
                is_skip_decision_by_score = True
                print(label_train + '  ' + str(amount_good_matching_points))
                break

            if amount_good_matching_points > max_score:
                max_score = amount_good_matching_points

        if is_skip_decision_by_score is False:
            score_d = {label:(value / amount_train_per_label_d[label]) for (label, value) in score_d.items()}

            prob_d = convert_score_to_probability(score_d)

            best_score_label = get_best_label_candidate(prob_d)

        is_correct_pred = (best_score_label in os.path.basename(file_path))

        print(score_d)

        if is_correct_pred:
            num_correct_pred += 1

        # print(prob_d)
        print(os.path.basename(file_path) + ':  ' + best_score_label + '    Is correct prediction:  ' + str(is_correct_pred))
        # print()

        if is_get_statistics:
            if is_correct_pred:
                if is_skip_decision_by_score is False:
                    correct_preds_prob.append(prob_d[best_score_label])
            else:
                wrong_preds_prob.append(prob_d[best_score_label])
                wrong_pred_images.append(os.path.basename(file_path))


    print('Total correct predictions: ' + str(num_correct_pred) + ' out of: ' + str(num_test_imgs) + ' test images')

    if is_get_statistics:
        total_correct_preds = len(correct_preds_prob)
        total_wrong_preds = len(wrong_preds_prob)

        mean_correct_prob = np.mean(correct_preds_prob)
        mean_wrong_prob = np.mean(wrong_preds_prob)

        std_correct_prob = np.std(correct_preds_prob)
        std_wrong_prob = np.std(wrong_preds_prob)

        lowest_correct_prob = np.min(correct_preds_prob)
        highest_wrong_prob = np.max(wrong_preds_prob)

        num_correct_pred_below_30_perc_prob = sum(prob < 0.25 for prob in correct_preds_prob)
        num_wrong_pred_above_30_perc_prob = sum(prob >= 0.25 for prob in wrong_preds_prob)

        print('Amount of training data:')
        print(amount_train_per_label_d)
        print()

        print('********** Correct predictions statistics ***********')
        print('Total correct predictions: ' + str(total_correct_preds))
        print('Mean probability of correct predictions: ' + str(mean_correct_prob))
        print('Std probability of correct predictions: ' + str(std_correct_prob))
        print('Lowest probability of correct predictions: ' + str(lowest_correct_prob))
        print('Number of correct preds with probability below 0.25: ' + str(num_correct_pred_below_30_perc_prob))

        print()
        print('********** Wrong predictions statistics ***********')
        print('Total probability of wrong predictions: ' + str(total_wrong_preds))
        print('Mean probability of wrong predictions: ' + str(mean_wrong_prob))
        print('Std probability of wrong predictions: ' + str(std_wrong_prob))
        print('Highest probability of wrong predictions: ' + str(highest_wrong_prob))
        print('Number of wrong preds with probability above 0.25: ' + str(num_wrong_pred_above_30_perc_prob))

        print()
        print('Wrong predcition on images:')

        for file in wrong_pred_images:
            print(file)


def get_best_label_candidate(score):
        best_score_label = 'none'
        best_score = 0

        for label, label_score in score.items():
            if label_score > best_score:
                best_score = label_score
                best_score_label = label

        second_best_score = 0

        for label, label_score in score.items():
            if (label_score > second_best_score) and (label != best_score_label):
                second_best_score = label_score

        if second_best_score < 0.75 * best_score:
            is_found_good_candidate = True #TODO: decide on measure for good candidate

        return best_score_label

def convert_score_to_probability(score):
    total_scores = sum(score.values())
    score_prob = {key:(score_value / total_scores) for (key,score_value) in score.items()}

    return score_prob

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

def get_label_from_basename(basename):
    if 'red' in  basename:
        return 'red'
    elif 'blue' in basename:
        return 'blue'
    elif 'white' in basename:
        return 'white'
    elif 'grey' in basename:
        return 'grey'
    elif 'green' in basename:
        return 'green'
    elif 'orange' in basename:
        return 'orange'
    else:
        assert False, 'Unknown label for image: ' + basename

def create_train_test_folders_surf(surf_data_path, num_train_per_label, min_num_imgs_for_label, is_train_surf_on_all_images=False, is_random_amount_surf_train_per_label=False):
    files_list = [file for file in os.listdir((surf_data_path)) if '.JPG' in file]

    files_dict_list = {}

    labels = ['red', 'blue', 'green', 'orange', 'white', 'grey']

    for label in labels:
        files_dict_list[label] = [file for file in files_list if label in file]

        num_imgs = len(files_dict_list[label])

        if is_train_surf_on_all_images is True: # -1 indicates to train on all images
            train_imgs = files_dict_list[label]
            test_imgs = files_dict_list[label]
        elif is_random_amount_surf_train_per_label is True:
            assert min_num_imgs_for_label >= 5, 'Need at least 5 surf train images for good prediction'

            random_num_of_train_imgs = np.random.choice(range(5, num_train_per_label + 1), 1)
            train_imgs_indices = np.random.choice(num_imgs, random_num_of_train_imgs, replace=False)
            train_imgs = [files_dict_list[label][i] for i in train_imgs_indices]
            test_imgs = list(set(files_dict_list[label]) - set(train_imgs))
        else:
            train_imgs_indices = np.random.choice(num_imgs, num_train_per_label, replace=False)
            train_imgs = [files_dict_list[label][i] for i in train_imgs_indices]
            test_imgs = list(set(files_dict_list[label]) - set(train_imgs))

        create_surf_train_test_dir(surf_data_path, os.path.join(surf_data_path, 'train'), train_imgs, label == labels[0])
        create_surf_train_test_dir(surf_data_path, os.path.join(surf_data_path, 'test'), test_imgs, label == labels[0])


def create_surf_train_test_dir(orig_data_path, new_data_path, data, is_create_new_dir):
    if is_create_new_dir is True:
        if os.path.exists(new_data_path):
            is_delete_dir = input(new_data_path + ' already exists. I need to delete it to create new one. Can I?')

            if is_delete_dir.lower() == 'y':
                shutil.rmtree(new_data_path)
            else:
                assert False, 'MEAN PERSON DOES NOT ALLOW ME TO DELETE THE FOLDER!! :('

        os.makedirs(new_data_path)

    copy_data(orig_data_path, new_data_path, data)


def sep_data_to_train_test(data_path, train_perc):
    files_list = os.listdir(data_path)

    images_list = []

    for file in files_list:
        if '.JPG' in file:
            images_list.append(file)
        elif 'annotationsTrain.txt' in file:
            annotations_file = file

    train_imgs, test_imgs = get_train_test_imgs(images_list, train_perc)

    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')

    create_specific_data_dir(data_path, train_path, train_imgs, os.path.join(data_path, annotations_file))
    create_specific_data_dir(data_path, test_path, test_imgs, os.path.join(data_path, annotations_file))

    return train_path, test_path

def create_specific_data_dir(orig_data_path, new_data_path, data, annotations_file_path):

    if os.path.exists(new_data_path):
        is_delete_dir = input(new_data_path + ' already exists. I need to delete it to create new one. Can I?')

        if is_delete_dir.lower() == 'y':
            shutil.rmtree(new_data_path)
        else:
            assert False, 'MEAN PERSON DOES NOT ALLOW ME TO DELETE THE FOLDER!! :('

    os.makedirs(new_data_path)

    create_new_annotation_file(new_data_path, data, annotations_file_path)

    copy_data(orig_data_path, new_data_path, data)

def copy_data(orig_data_path, new_data_path, data):
    for file in data:
        shutil.copy2(os.path.join(orig_data_path, file), os.path.join(new_data_path, file))

def create_new_annotation_file(data_path, data, annotations_file_path):
    new_annotations_file_path = os.path.join(data_path, os.path.basename(annotations_file_path))

    with open (annotations_file_path, 'r') as annotations_fp:
        with open (new_annotations_file_path, 'w') as new_annotations_fp:
            for line in annotations_fp:
                (img_name, _) = line.split(':')

                if img_name in data:
                    new_annotations_fp.write(line)

def get_train_test_imgs(images_list, train_perc):
    num_imgs = len(images_list)

    train_imgs_indices = np.random.choice(num_imgs, round(train_perc * num_imgs), replace=False)

    train_imgs = [images_list[i] for i in train_imgs_indices if '1013' not in images_list[i]]

    test_imgs = list(set(images_list) - set(train_imgs))
    print(test_imgs)
    print(len(train_imgs))

    return train_imgs, test_imgs

def main(model_file_name = 'nn_buses_3.pt', data_path='final_dir/buses', batch_size=1):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_path, test_path = sep_data_to_train_test(data_path, train_perc)

    # use our dataset and defined transformations
    dataset = bassesDataset(train_path, get_transform())
    dataset_test = bassesDataset(test_path, get_transform())

    # split the dataset in train and test set
    #FIXME - an option for split train test from one folder
    #indices = torch.randperm(len(dataset)).tolist()
    #dataset = torch.utils.data.Subset(dataset, indices[:-50])
    #dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)
    # currently not in use
    #FIXME can change for use at testing (the train punction recives the image path instead of the current output of the loader which is loaded image)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = create_model()

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        # # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device) # old funcrion from the tutorial - i didnt use
        # Run the model on the test images and save predicted image.
        if (epoch)%10 ==0 or epoch == num_epochs-1:
            test(model, epoch, test_path)
    print("That's it!")

    # save the model for reloading after
    torch.save(model.state_dict(), model_file_name)


if __name__ == "__main__":
    #### create train & test automatically from all images #########

    # create_train_test_folders_surf(surf_data_path, num_train_per_label, min_num_imgs_for_label, is_train_surf_on_all_images=is_train_surf_on_all_images,\
    #                                is_random_amount_surf_train_per_label=is_random_amount_surf_train_per_label)
    #
    # des_label_list = train_SURF(os.path.join(surf_data_path, 'train'))
    #
    # t = time.time()
    #
    # with open('/Users/iliabenkovitch/Documents/Computer_Vision/git/git_orign_cv_project/nn/KAZE_trained_features.pickle', 'wb') as handle:
    #     pickle.dump(des_label_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open('/Users/iliabenkovitch/Documents/Computer_Vision/git/git_orign_cv_project/nn/KAZE_trained_features.pickle', 'rb') as handle:
    #     des_label_list_from_file = pickle.load(handle)
    #
    # test_SURF(os.path.join(surf_data_path, 'train'), des_label_list_from_file, is_get_statistics=is_get_statistics)
    # # test_SURF(os.path.join(surf_data_path), des_label_list_from_file, is_get_statistics=True)
    #
    # # save / load data testing
    #
    # print(str(time.time() - t) + ' secs for test_SURF')


    ########## Run on pre-made train & test surf images ############

    # create_train_test_folders_surf(surf_train_data, num_train_per_label, min_num_imgs_for_label, is_train_surf_on_all_images=is_train_surf_on_all_images,\
    #                                is_random_amount_surf_train_per_label=is_random_amount_surf_train_per_label)
    #
    # des_label_list = train_SURF(os.path.join(surf_train_data, 'train'))
    #
    # t = time.time()
    #
    # test_SURF(surf_test_data, des_label_list, is_get_statistics=is_get_statistics)
    # # test_SURF(os.path.join(surf_data_path), des_label_list, is_get_statistics=True)
    #
    # print(str(time.time() - t) + ' secs for test_SURF')

    main()


