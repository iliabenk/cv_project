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


# Defines the labels for the model, change to colors if you want.
COCO_INSTANCE_CATEGORY_NAMES = ['background','buss']

# get prediction from model (was taken from an RCNN tutorial)
def get_prediction(model, img_path, threshold):
  img = Image.open(img_path) # Load the image
  transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
  img = transform(img) # Apply the transform to the image
  pred = model([img]) # Pass the image to the model

  pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
  pred_score = list(pred[0]['scores'].detach().numpy())
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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    for i in range(len(boxes)):
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0),
                      thickness=rect_th)  # Draw Rectangle with the coordinates
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
                    thickness=text_th)  # Write the prediction class
    plt.figure(figsize=(20, 30))  # display the output image
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    #plt.show()
    plt.savefig(output_dir +'/' + os.path.basename(img_path).replace('.JPG','_prid.JPG'))

# Test function - FIXME need to add aoc score as defined, Changes at DataLoarder has to be done
def test(model,  data_loader, epoch, output_path ):
    file_list = [output_path + '/busesTrain/' +path for path in os.listdir(output_path + '/busesTrain/') if '.JPG' in path]
    model.eval()
    output_fol = output_path + '/' + 'ep'+str(epoch)
    if not os.path.exists(output_fol):
        os.makedirs(output_fol)
    for img in file_list:

        object_detection_api(model,output_fol,img )

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
    def __init__(self, root,transforms):
        self.root = root
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = [path for path in os.listdir(os.path.join(root, "busesTrain")) if '.JPG' in path]
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
        img_path = os.path.join(self.root, "busesTrain", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img_name = os.path.basename(img_path)
        # get bounding box coordinates for each mask
        boxes = []
        boxes_raw = np.array(self.gt_dict[img_name])
        if len(boxes_raw.shape)==1:
            num_objs =1
            xmin = boxes_raw[0]
            ymin = boxes_raw[1]
            width = boxes_raw[2]
            hight = boxes_raw[3]
            boxes.append(convert_BB_to_net(xmin, ymin, width, hight))
        else:
            num_objs = len(boxes_raw)
            for i in range(num_objs):
                pos = boxes_raw[i]
                xmin = pos[0]
                ymin = pos[1]
                width = pos[2]
                hight = pos[3]
                boxes.append(convert_BB_to_net(xmin,ymin,width,hight))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        #FIXME - change to correct label
        labels = torch.ones((num_objs,), dtype=torch.int64)

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
    num_classes = 2  # 1 class (person) + background #FIXME change to 7 label
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model



def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = bassesDataset('train/',get_transform())
    dataset_test = bassesDataset('test/',get_transform())

    # split the dataset in train and test set
    #FIXME - an option for split train test from one folder
    #indices = torch.randperm(len(dataset)).tolist()
    #dataset = torch.utils.data.Subset(dataset, indices[:-50])
    #dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1,
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

    # let's train it for 10 epochs #FIXME - can set for less
    num_epochs = 10
    # path for the test function - to read the images inside "bussesTrian" and to save test images:
    output_path = "/Users/omriefroni/PycharmProjects/comp_vision_ex2/test"
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        #evaluate(model, data_loader_test, device=device) # old funcrion from the tutorial - i didnt use
        # Run the model on the test images and save predicted image.
        test(model, data_loader, epoch, output_path)
    print("That's it!")
    # save the model for reloading after
    torch.save(model.state_dict(), 'nn_busses_2.pt')



main()

