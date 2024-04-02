from skimage import io, transform
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import monai
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from rembg import remove
import pandas as pd
from PIL import Image



def get_model(num_classes):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model

# set seeds
torch.manual_seed(2023)
np.random.seed(2023)


# prepare SAM model
model_type = 'vit_b'
checkpoint = 'work_dir/demo2D/sam_model_best.pth'
device = 'cuda:0'
sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
sam_model.train()
# Set up the optimizer, hyperparameter tuning will improve performance here
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')


#create csv file
if not os.path.exists('data.csv'):
    with open('data.csv', 'w') as file:
        pass

# Initialize headers in a csv file if it is empty
headers = ["ID", "Calculated Percentage"]
empty_df = pd.DataFrame(columns=headers)
csv_file_path = "data.csv"
empty_df.to_csv(csv_file_path, index=False)

data = {}
data['ID'] = []
data['Calculated Percentage'] = []
data['Labeled Percentage'] = []
data['Error'] = []

excel = pd.read_excel('cortex_perc.xlsx')
id_list = excel.iloc[:, 0].tolist()
labeled_perc_list = excel.iloc[:,1].tolist()


labeled_data = {}
for i in range(len(id_list)):
    labeled_data[id_list[i]] = labeled_perc_list[id_list.index(id_list[i])] 

ts_img_path = 'data/MedSAMDemo_2D/test/images'
ts_gt_path = 'data/MedSAMDemo_2D/test/labels'
test_names = sorted(os.listdir(ts_img_path))

for img_idx in range(len(test_names)):
#for img_idx in range(3):
    image_data = io.imread(join(ts_img_path, test_names[img_idx]))
    print(test_names[img_idx])

    image_data = remove(image_data)
    # Resize the image_data to 256x256 pixels
    new_size = (256, 256)
    resized_image_data = transform.resize(image_data, new_size, anti_aliasing=True)

    # If the image_data is of type float, convert it back to uint8 in the range [0, 255]
    if resized_image_data.dtype == float:
        resized_image_data = (resized_image_data * 255).astype('uint8')

    image_data = resized_image_data

    if image_data.shape[-1]>3 and len(image_data.shape)==3:
        image_data = image_data[:,:,:3]
    if len(image_data.shape)==2:
        image_data = np.repeat(image_data[:,:,None], 3, axis=-1)

    # read ground truth (gt should have the same name as the image) and simulate a bounding box
    def get_bbox_from_mask(imgs):
        model_path = 'best_bbox_seg_model.pth'
        model = get_model(num_classes=2)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model = model.cuda()

        imgs = [imgs.to(device)]

        predictions = model(imgs)

        best_preds = []
        for pred in predictions:
            max_conf_index = pred["scores"].argmax()  # Index of highest confidence score
            highest_conf_box = pred["boxes"][max_conf_index]  # Bounding box with highest confidence
            best_preds.append(highest_conf_box)

        best_preds_cpu = best_preds[0].cpu()

        return best_preds_cpu.detach().numpy()
        

    gt_data_path = os.path.join(ts_img_path, test_names[img_idx])
    ts_img = Image.open(gt_data_path).convert("RGB")

    # Transformations
    def trans(img):
        return F.to_tensor(img)

    ts_img = trans(ts_img)
    bbox_raw = get_bbox_from_mask(ts_img)

    # preprocess: cut-off and max-min normalization
    lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
    image_data_pre[image_data==0] = 0
    image_data_pre = np.uint8(image_data_pre)
    H, W, _ = image_data_pre.shape

    sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    resize_img = sam_transform.apply_image(image_data_pre)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
    input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
    assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'

    with torch.no_grad():
        # pre-compute the image embedding
        ts_img_embedding = sam_model.image_encoder(input_image)
        # convert box to 1024x1024 grid
        bbox = sam_transform.apply_boxes(bbox_raw, (H, W))
        # print(f'{bbox_raw=} -> {bbox=}')    
        box_torch = torch.as_tensor(bbox, dtype=torch.float, device=device)

        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] # (B, 4) -> (B, 1, 4) #My code for not showing the bbox

        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        medsam_seg_prob, _ = sam_model.mask_decoder(
            image_embeddings=ts_img_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            )
        
        medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
        print(medsam_seg.shape)

    threshold = 50  # Adjust this value based on your preference

    # Remove black or very dark gray pixels
    image_without_dark = image_data.copy()
    gray_intensity = np.sum(image_data, axis=-1) // 3  # Convert RGB to grayscale
    image_without_dark[gray_intensity <= threshold] = [255, 255, 255]
    plt.imshow(image_without_dark)

    grayscale_image = np.sum(image_without_dark, axis=-1) // 3

    # Count non-white pixels
    non_white_pixels = np.sum(grayscale_image < 255)

    overlay = image_data.copy()
    overlay[(medsam_seg > 0) & (grayscale_image < 255)] = [255, 255, 255]

    # Save the overlaid image as a JPEG file
    output_folder = 'annotated_images/'

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    output_file_name = 'output_' + test_names[img_idx]
    cv2.imwrite(output_folder + output_file_name, overlay)

    segmented = np.sum((medsam_seg > 0) & (grayscale_image < 255))

    cortex_percentage = (segmented / non_white_pixels) * 100
    formatted_percentage = "{:.2f}\n".format(cortex_percentage)    
    id = test_names[img_idx][:-4]

    data['ID'].append(int(id))
    data['Calculated Percentage'].append(cortex_percentage)
    data['Labeled Percentage'].append(labeled_data[int(id)])
    data['Error'].append(abs(cortex_percentage - labeled_data[int(id)]))
    

csv = pd.read_csv('data.csv')
df = pd.DataFrame(data)
updated_data = pd.concat([csv, df], ignore_index=True)
updated_data.to_csv('data.csv', index=False)