import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = 'results/DSC.xlsx'  # Update with your file path
df = pd.read_excel(file_path)

# Define the label IDs to remove
label_ids_to_remove = [1, 26, 26, 27, 28, 29, 30, 31, 79]

# Filter out rows with label IDs to remove
df_filtered = df[~df['Label ID'].isin(label_ids_to_remove)]

# Extract Label ID and DSC columns from the filtered DataFrame
label_ids = df_filtered['Label ID']
dsc_values = df_filtered['DSC']

# Calculate the average DSC score
average_dsc = df_filtered['DSC'].mean()

print("Average DSC score after excluding specified label IDs:", average_dsc)


# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(label_ids, dsc_values)
plt.title('DSC Values vs. Label IDs')
plt.xlabel('Label ID')
plt.ylabel('DSC')
plt.grid(True)
plt.show()
