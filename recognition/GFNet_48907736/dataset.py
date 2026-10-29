"""
Goal is have a processed dataset with 70% allocated to training, 15% to validation and 15% to testing.

First, the data is manually put moved out from the pre-allocated test and train folders.

Next, the data will be split on patient ID. For each image, the name is patientID_sliceNumber meaning
this process will sort by the first half of the image name.

After this is done, the test/train/val split will be made randomly while ensuring patients are kept together.
"""


from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

AD_path = Path(r"C:\Users\elvin\OneDrive\Desktop\UQ\Year 3, Semester 2\COMP3710\Final Project\ADNI_Unsplit/AD")
CN_path = Path(r"C:\Users\elvin\OneDrive\Desktop\UQ\Year 3, Semester 2\COMP3710\Final Project\ADNI_Unsplit/CN")

def pt_id_from_path(path) -> str:
    #'1003730_100.jpeg' -> '1003730' (everything before the last underscore)
    return int(path.stem.rsplit("_", 1)[0])

def pic_id_from_path(path) -> str:
    #'1003730_100.jpeg' -> '100' (everything after the last underscore)
    return int(path.stem.rsplit("_", 1)[1])

# Build dictionarys by patient
def dict_build(path): 
    dict = {}
    for file_path in path.iterdir():
        if file_path.is_file():
            pt_id = pt_id_from_path(file_path)
            pic_id = pic_id_from_path(file_path)
            dict.setdefault(pt_id, []).append(file_path)
    return dict

     
# Check to see if any one pt is a significant proportion of the dataset
def print_stats(AD, CN):
    print("\n--- ADNI Dataset Statistics ---")

    print("AD:")
    print("Total images:", sum(len(v) for v in AD.values()))
    print("Unique patients:", len(AD))
    print("Average images per patient:", sum(map(len, AD.values())) / len(AD))
    print("Largest patient image count:", max(map(len, AD.values())))

    print("\nCN:")
    print("Total images:", sum(len(v) for v in CN.values()))
    print("Unique patients:", len(CN))
    print("Average images per patient:", sum(map(len, CN.values())) / len(CN))
    print("Largest patient image count:", max(map(len, CN.values())))

    overlap = set(AD.keys()) & set(CN.keys())

    print("\n--- Patient Overlap ---")
    print("Patients in both AD and CN:", len(overlap))


# test train split val
def split_patients(patient_ids):

    # 70% training 30% remaining
    train, remaining = train_test_split(
        patient_ids,
        test_size=0.30,
        random_state=42
    )

    # Split remaining equally into validation and testing
    val, test = train_test_split(
        remaining,
        test_size=0.50,
        random_state=42
    )

    return {
        "train": train,
        "val": val,
        "test": test
    }
    
# moving the files
def move_files(AD, CN, AD_split, CN_split):
    output_path = Path(r"C:\Users\elvin\OneDrive\Desktop\UQ\Year 3, Semester 2\COMP3710\Final Project\ADNI_Processed")

    # Move images into existing folders
    for category, files_by_pt, split in [("AD", AD, AD_split), ("CN", CN, CN_split)]:
        for dataset, patients in split.items():
            
            destination = output_path / dataset / category
            for patient in patients:
                for file_path in files_by_pt[patient]:
                    shutil.copy2(str(file_path), str(destination / file_path.name))
                    
    print("Files moved successfully")


AD_files_by_pt = dict_build(AD_path)
CN_files_by_pt = dict_build(CN_path)

print_stats(AD_files_by_pt, CN_files_by_pt)

AD_split = split_patients(list(AD_files_by_pt.keys()))
CN_split = split_patients(list(CN_files_by_pt.keys()))

move_files(AD_files_by_pt, CN_files_by_pt, AD_split, CN_split)