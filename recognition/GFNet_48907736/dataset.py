"""
Goal is have a processed dataset with 70% allocated to training, 15% to validation and 15% to testing.

First, the data is manually put moved out from the pre-allocated test and train folders.

Next, the data will be split on patient ID. For each image, the name is patientID_sliceNumber meaning
this process will sort by the first half of the image name.

After this is done, the test/train/val split will be made randomly while ensuring patients are kept together.
"""


from pathlib import Path

dir_path = Path(r"C:\Users\elvin\OneDrive\Desktop\UQ\Year 3, Semester 2\COMP3710\Final Project\ADNI_Unsplit\AD")

def patient_id_from_path(path) -> str:
    """'1003730_100.jpeg' -> '1003730' (everything before the last underscore)."""
    return path.stem.rsplit("_", 1)[0]

for file_path in dir_path.iterdir():
    if file_path.is_file():
        print(patient_id_from_path(file_path))
        
