# Kidney_Tissue_Segmentation

**1. Clone repository**
  - git clone https://github.com/Khangxlei/Kidney_Tissue_Segmentation/

**2. Ensure data files**
  - Place the source and target data files in the correct directory within the project.

**3. Preprocessing**
  - Run preprocess.py twice, each time changing its parameters to process different datasets.

**4. Get Bounding Box Model**
  - Open and run get_bbox_model.ipynb in Jupyter Notebook. Execute all cells.

**5. Inference and Evaluation**
  - Run inference.py to perform inference on the testing dataset using the trained model.
  - Run IoU.py to calculate the Intersection over Union (IoU) scores for the testing dataset and record the values into an Excel spreadsheet.
    
**Results**
  - The Excel spreadsheets in the results folder contain the IoU scores calculated using 5-fold cross-validation across 100 annotated kidney tissue images.
  - There are four files:
      - '_DSC with RCNN UPDATED.xlsx_': DSC results with Faster R-CNN trained over the MedSAM and kidney tissue datasets. 
      - '_IoU with RCNN UPDATED.xlsx_': IoU results with Faster R-CNN trained over the MedSAM and kidney tissue datasets.
      - '_IoU with RCNN and DANN.xlsx_': IoU results with Faster R-CNN as well as DANN.
      - '_IoU with RCNN without DANN.xlsx_': IoU results with Faster R-CNN and without DANN. 
