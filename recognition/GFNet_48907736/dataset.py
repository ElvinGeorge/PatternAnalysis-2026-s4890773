"""
Goal is have a processed dataset with 70% allocated to training, 15% to validation and 15% to testing.

First, the data is manually put moved out from the pre-allocated test and train folders.

Next, the data will be split on patient ID. For each image, the name is patientID_sliceNumber meaning
this process will sort by the first half of the image name.

After this is done, the test/train/val split will be made randomly while ensuring patients are kept together.

Dataset info: 20 pts with 
"""


from pathlib import Path
import shutil

AD_files_by_pt = {}
CN_files_by_pt = {}

AD_path = Path(r"C:\Users\elvin\OneDrive\Desktop\UQ\Year 3, Semester 2\COMP3710\Final Project\ADNI_Unsplit/AD")
CN_path = Path(r"C:\Users\elvin\OneDrive\Desktop\UQ\Year 3, Semester 2\COMP3710\Final Project\ADNI_Unsplit/CN")

def pt_id_from_path(path) -> str:
    #'1003730_100.jpeg' -> '1003730' (everything before the last underscore)
    return int(path.stem.rsplit("_", 1)[0])

def pic_id_from_path(path) -> str:
    #'1003730_100.jpeg' -> '100' (everything after the last underscore)
    return int(path.stem.rsplit("_", 1)[1])

# Build dictionarys by patient 
for file_path in AD_path.iterdir():
    if file_path.is_file():
        pt_id = pt_id_from_path(file_path)
        pic_id = pic_id_from_path(file_path)
        AD_files_by_pt.setdefault(pt_id, []).append(pic_id)
        
for file_path in CN_path.iterdir():
    if file_path.is_file():
        pt_id = pt_id_from_path(file_path)
        pic_id = pic_id_from_path(file_path)
        CN_files_by_pt.setdefault(pt_id, []).append(pic_id)
     
# Check to see if any one pt is a significant proportion of the dataset   
"""
print("\n--- ADNI Dataset Statistics ---")

print("AD:")
print("Total images:", sum(len(v) for v in AD_files_by_pt.values()))
print("Unique patients:", len(AD_files_by_pt))
print("Average images per patient:", sum(map(len, AD_files_by_pt.values())) / len(AD_files_by_pt))
print("Largest patient image count:", max(map(len, AD_files_by_pt.values())))

print("\nCN:")
print("Total images:", sum(len(v) for v in CN_files_by_pt.values()))
print("Unique patients:", len(CN_files_by_pt))
print("Average images per patient:", sum(map(len, CN_files_by_pt.values())) / len(CN_files_by_pt))
print("Largest patient image count:", max(map(len, CN_files_by_pt.values())))

Output
--- ADNI Dataset Statistics ---
AD:
Total images: 14860
Unique patients: 743
Average images per patient: 20.0
Largest patient image count: 20

CN:
Total images: 15660
Unique patients: 783
Average images per patient: 20.0
Largest patient image count: 20

"""








destination = Path(r"C:\Users\elvin\OneDrive\Desktop\UQ\Year 3, Semester 2\COMP3710\Final Project\ADNI_Processed\val\AD")

