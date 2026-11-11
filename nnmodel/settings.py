CLASS_NAMES = ['Benign', 'Indeterminate', 'Malignant']

MALIGNANT_CLASS_NAMES = ['Adrenocortical Carcinoma', 'Pheochromocytoma']
BENIGN_LEVEL1_CLASS_NAMES = ['Solid', 'Cystic/Lipomatous']
BENIGN_LEVEL2_SOLID_CLASS_NAMES = ['Adenoma', 'Nodular Hyperplasia']
BENIGN_LEVEL2_CYSTIC_CLASS_NAMES = ['Cyst', 'Myelolipoma']

# Phenotype (bagging) + segmentation weights
settings = {
    'segmentation': {'all': './model_files/best.pt'},
    'classification': {'all': './model_files/'},
}

# Tencent MedicalNet pretrained backbone (ResNet-18, shortcut A)
# Download: https://github.com/Tencent/MedicalNet
# or HF: resnet_18_23dataset.pth
MEDICALNET_WEIGHTS_PATH = './MedicalNet/resnet_18_23dataset.pth'

# Second-stage diagnosis (MedicalNet 3D ResNet-18 fine-tunes)
MALIGNANT_MODEL_PATH = './model_files/stage2_malignant.pth'
BENIGN_LEVEL1_PATH = './model_files/stage1_best.pth'
BENIGN_LEVEL2_SOLID_PATH = './model_files/stage2_solid_best.pth'
BENIGN_LEVEL2_CYSTIC_PATH = './model_files/stage2_cystic_best.pth'
