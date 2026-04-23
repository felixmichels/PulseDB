# /// script
# dependencies = [
#   "numpy",
#   "h5py",
#   "mat73",
#   "tqdm"
# ]
# ///

import mat73
import h5py
from pathlib import Path
import numpy as np
from tqdm import tqdm

COLUMNS = [
    "ECG_F",
    "PPG_F",
    "ABP_Raw",
    "SegSBP",
    "SegDBP",
    "Age",
    "Gender",
    "Height",
    "Weight",
    "BMI"
]

def dt_str(length):
    return h5py.string_dtype(encoding='ascii', length=length)

def loadmat(file: Path, segment=False):
    data = mat73.loadmat(file.with_suffix('.mat'))
    assert len(data) == 1
    data = next(iter(data.values()))
    if segment:
        data = {k: v for k, v in data.items() if k in COLUMNS}
    return {k: np.vstack(v).squeeze() for k, v in data.items()}


def generate_subset(mimic_path, vital_path, info_file_path, save_name):
    mimic_path = Path(mimic_path)
    vital_path = Path(vital_path)
    info_file_path = Path(info_file_path)
    save_name = Path(save_name).with_suffix('.h5')
    
    lock_file = save_name.with_suffix('.tmp')
    if lock_file.is_file() or save_name.is_file():
        print(f"File for {save_name} exists, skipping")
        return
    lock_file.touch()

    print(f"Generating subset {save_name}")

    info = loadmat(info_file_path)
    length = len(info['Subj_Name'])
    subjects = np.unique(info['Subj_Name'])

    with h5py.File(save_name, 'w') as file:
        file.create_dataset("Subject", shape=(length,), dtype=dt_str(7))
        file.create_dataset("Signals", shape=(length, 3, 1250), dtype="float64", fillvalue=0)
        file.create_dataset("Gender", shape=(length,), dtype=dt_str(1))
        for key in "SBP", "DBP", "Age", "Height", "Weight", "BMI":
            file.create_dataset(key, shape=(length,), dtype="float64", fillvalue=np.nan)

        pos = 0
        for subj_name in tqdm(subjects):
            subj_id = subj_name[:7]
            source = int(subj_name[-1])
            segment_path = mimic_path if source == 0 else vital_path

            subj_segments = loadmat(segment_path / subj_id, segment=True)
            # -1 because matlab is 1-indexed
            selected_idx = info['Subj_SegIDX'][info['Subj_Name'] == subj_name].astype(np.long) - 1

            for j in selected_idx:
                segment = {k: v[j] for k, v in subj_segments.items()}
                file['Subject'][pos] = subj_name
                file['Signals'][pos] = np.stack([
                    segment['ECG_F'],
                    segment['PPG_F'],
                    segment['ABP_Raw']
                ])

                file['SBP'][pos] = segment['SegSBP']
                file['DBP'][pos] = segment['SegDBP']
                file['Age'][pos] = segment['Age']
                file['Gender'][pos] = segment['Gender']
                if source == 1: # Record information for VitalDB subjects
                    file['Height'][pos] = segment['Height']
                    file['Weight'][pos] = segment['Weight']
                    file['BMI'][pos] = segment['BMI']
                pos += 1
    lock_file.unlink()
    print(f"Finished generating subset {save_name}")


def main():
    # Generating training, calibration, and testing subsets of PulseDB
    # Note: Height, weight and BMI field for data will only be valid for
    # segements from the VitalDB dataset, and wil be NaN for segments from
    # the MIMIC-III matched subset, since these infroamtion are not included in
    # the original MIMIC-III matched subset.
    # Locate segment files
    MIMIC_Path='Segment_Files/PulseDB_MIMIC'
    Vital_Path='Segment_Files/PulseDB_Vital'

    # Locate info files
    Train_Info='Info_Files/Train_Info'
    CalBased_Test_Info='Info_Files/CalBased_Test_Info'
    CalFree_Test_Info='Info_Files/CalFree_Test_Info'
    AAMI_Test_Info='Info_Files/AAMI_Test_Info'
    AAMI_Cal_Info='Info_Files/AAMI_Cal_Info'

    # Generate training set
    generate_subset(MIMIC_Path,Vital_Path,Train_Info,'Subset_Files/Train_Subset')
    # Generate calibration-based testing set
    generate_subset(MIMIC_Path,Vital_Path,CalBased_Test_Info,'Subset_Files/CalBased_Test_Subset')
    # Generate calibration-free testing set
    generate_subset(MIMIC_Path,Vital_Path,CalFree_Test_Info,'Subset_Files/CalFree_Test_Subset')
    # Generate AAMI testing set
    generate_subset(MIMIC_Path,Vital_Path,AAMI_Test_Info,'Subset_Files/AAMI_Test_Subset')
    # Generate AAMI calibration set
    generate_subset(MIMIC_Path,Vital_Path,AAMI_Cal_Info,'Subset_Files/AAMI_Cal_Subset')
    ## Generating supplementary trainining, calibration, and testing subsets from only VitalDB subjects

    # Locate info files
    VitalDB_Train_Info='Supplementary_Info_Files/VitalDB_Train_Info'
    VitalDB_CalBased_Test_Info='Supplementary_Info_Files/VitalDB_CalBased_Test_Info'
    VitalDB_CalFree_Test_Info='Supplementary_Info_Files/VitalDB_CalFree_Test_Info'
    VitalDB_AAMI_Test_Info='Supplementary_Info_Files/VitalDB_AAMI_Test_Info'
    VitalDB_AAMI_Cal_Info='Supplementary_Info_Files/VitalDB_AAMI_Cal_Info'

    # Generate training set
    generate_subset(MIMIC_Path,Vital_Path,VitalDB_Train_Info,'Supplementary_Subset_Files/VitalDB_Train_Subset')
    # Generate calibration-based testing set
    generate_subset(MIMIC_Path,Vital_Path,VitalDB_CalBased_Test_Info,'Supplementary_Subset_Files/VitalDB_CalBased_Test_Subset')
    # Generate calibration-free testing set
    generate_subset(MIMIC_Path,Vital_Path,VitalDB_CalFree_Test_Info,'Supplementary_Subset_Files/VitalDB_CalFree_Test_Subset')
    # Generate AAMI testing set
    generate_subset(MIMIC_Path,Vital_Path,VitalDB_AAMI_Test_Info,'Supplementary_Subset_Files/VitalDB_AAMI_Test_Subset')
    # Generate AAMI calibration set
    generate_subset(MIMIC_Path,Vital_Path,VitalDB_AAMI_Cal_Info,'Supplementary_Subset_Files/VitalDB_AAMI_Cal_Subset')


if __name__ == '__main__':
    main()
