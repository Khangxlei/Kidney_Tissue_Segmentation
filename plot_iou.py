import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = 'results/IoU.xlsx'  # Update with your file path
df = pd.read_excel(file_path)

# Define the label IDs to remove
label_ids_to_remove = [1, 26, 26, 27, 28, 29, 30, 31, 79]

# Filter out rows with label IDs to remove
df_filtered = df[~df['Label ID'].isin(label_ids_to_remove)]

# Extract Label ID and IoU columns from the filtered DataFrame
label_ids = df_filtered['Label ID']
iou_values = df_filtered['IoU']

# Calculate the average IoU score
average_iou = df_filtered['IoU'].mean()

print("Average IoU score after excluding specified label IDs:", average_iou)


# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(label_ids, iou_values)
plt.title('IoU Values vs. Label IDs')
plt.xlabel('Label ID')
plt.ylabel('IoU')
plt.grid(True)
plt.show()
