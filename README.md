# Vision For Social Good

* **Project description**
This project aims to develop a multimodal approach for detecting diseases in cash crops using various types of imagery, including RGB, multispectral, and aerial images. The goal is to leverage different data sources to improve the accuracy and robustness of disease detection models.
 
## **Project objective**
The main objective of this project is to develop a multimodal approach for detecting diseases in cash crops using various machine learning models and data sources. The project aims to leverage different types of imagery and data to improve the accuracy and robustness of disease detection.

  * Utilize ground RGB photos, aerial UAV photos, and multispectral imagery for data collection and preprocessing.
  * Implement and compare various machine learning models, including InceptionV3, ResNet152, VGG19, ViT, and attention-augmented models.
  * Perform feature extraction and classification using different fusion methods, such as early, intermediate, and late fusion.
  * Evaluate the performance of the models using metrics like accuracy, precision, recall, and F1-score.
  * Generate visualizations and reports to analyze the results and identify the best-performing models and fusion methods.

The project involves multiple stages, including data preprocessing, model training, evaluation, and result analysis, to achieve the goal of accurate and efficient cash crop disease detection.

## **Table of contents**
* [Project description](#project-description)
* [Table of contents](#table-of-contents)
* [Project structure](#Project-structure)
* [Project structure overview](#Project-structure-overview)
* [Installation](#installation)
* [Usage](#usage)
* [Contributing](#contributing)
* [License](#license)
* [Acknowledgements](#acknowledgements)

## **Project structure**
An overview of the project's directory structure, including key files and directories.
```
├── CODE/
│   ├── MULTIMODAL/
│   │   ├── AACN_Layer.py
│   │   ├── AACN_Model.py
│   │   ├── Feature Extraction - Classification.ipynb
│   │   ├── Feature Extraction - Classification.py
│   │   ├── concat_multimodal_files.ipynb
│   │   ├── dataset_summary.csv
│   │   ├── early_fusion.ipynb
│   │   ├── multimodal_feature_extraction.ipynb
│   │   ├── multimodal_fusion_backup.ipynb
│   │   ├── multimodal_training copy.ipynb
│   │   ├── multimodal_training.ipynb
│   │   ├── multimodal_training_CNN.ipynb
│   │   ├── multimodal_training_no_early_stop_bs128.ipynb
│   │   ├── multimodal_training_no_early_stop_bs16.ipynb
│   │   ├── multimodal_training_no_early_stop_bs32.ipynb
│   │   ├── multimodal_training_no_early_stop_bs64.ipynb
│   │   └── multimodal_training_no_early_stop_bs8.ipynb
│   ├── SINGLE MODAL/
│   │   ├── AACN_Layer.py
│   │   ├── AACN_Model.py
│   │   ├── ground-imagry-backup copy.ipynb
│   │   ├── ground-imagry-backup-wuth-tensorflow.ipynb
│   │   ├── ground-imagry-backup.ipynb
│   │   ├── ground_RGB_imagry.ipynb
│   │   ├── ground_imagry pretrained.ipynb
│   │   ├── ground_imagry training from scratch.ipynb
│   │   ├── ground_imagry_with_tensorflow.ipynb
│   │   ├── ground_multispectral_imagry.ipynb
│   │   ├── ground_uav_csv.ipynb
│   │   ├── move_part_of_images.ipynb
│   │   ├── read_multispectral_imagery.ipynb
│   │   └── uav_imagry training from scratch.ipynb
│   └── accessing_tif_image_data.ipynb
├── RESULTS/
│   ├── Multimodal/
│   │   ├── Peach/
│   │   │   ├── CNN+BB/
│   │   │   │   ├── T1/
│   │   │   │   ├── T2/
│   │   │   │   │   └──  model_metrics.csv
│   │   │   │   └── T2+NoES+BS008/
│   │   │   │       └── model_metrics.csv
│   ├── Single/
│   │   ├── Peach/
│   │   │   ├── CSV/
│   │   │   │   ├── T1/
│   │   │   │   │   └── model_metrics.csv
│   │   │   │   ├── T2/
│   │   │   │   │   └── model_metrics.csv
│   │   │   ├── Multispectral/
│   │   │   │   ├── T4/
│   │   │   │   │   └── model_metrics.csv
│   │   │   ├── RGB/
│   │   │   │   ├── T1/
│   │   │   │   │   ├── model_metrics.csv
│   │   │   │   ├── T2/
│   │   │   │   │   ├── model_metrics.csv
│   │   │   │   ├── T3/
│   │   │   │       └── model_metrics.csv
├── SAMPLE DATA/
│   ├── Peach/
│   │   ├── 04_11_21/
│   │   │   ├── Aerial_UAV_Photos/
│   │   │   │   └── uav_converted_with_new_names_and_labels.csv
│   │   │   ├── Ground_Multispectral_Photos_bounding_box/
│   │   │   │   ├── Anarsia lineatella/
│   │   │   │   │   ├── 1 - 12/
│   │   │   │   │   └── 9 - 5/
│   │   │   │   ├── Dead Trees/
│   │   │   │   │   ├── 13 - 19/
│   │   │   │   │   └── 14 - 1/
│   │   │   │   ├── Grapholita molesta/
│   │   │   │   │   ├── 15 - 1/
│   │   │   │   │   └── 15 - 19/
│   │   │   │   ├── Healthy/
│   │   │   │   │   ├── 3 - 5/
│   │   │   │   │   └── 3 - 8/
│   │   │   ├── Ground_RGB_Photos_bounding_box/
│   │   │   │   ├── Anarsia lineatella/
│   │   │   │   │   ├── annotation.zip
│   │   │   │   ├── Dead Trees/
│   │   │   │   ├── Grapholita molesta/
│   │   │   │   │   ├── annotation.zip
│   │   │   │   └── Healthy/
│   │   │   ├── Orchard_Mapping/
│   │   │   │   └── Orthomosaic_converted_with_new_names_and_labels.csv
│   │   │   ├── 04_11_21.csv
│   │   │   ├── multimodal_data.csv
│   │   │   └── tree_ndvi_values.csv
│   │   ├── Tree Mapping.xlsx
│   │   └── combined_multimodal_data.csv
├── .gitattributes
├── LICENSE
├── README.md
└── Vision_for_Social_Good___Multimodal_Crop_Plant_Disease_Detection_compressed.pdf
```

## Project structure overview

The project is organized into several key directories and files, each serving a specific purpose:

### CODE/
This directory contains all the code related to the project, divided into subdirectories for different types of models and tasks:
- **MULTIMODAL/**: Contains scripts and notebooks for multimodal data processing and model training.
  - `AACN_Layer.py`, `AACN_Model.py`: Python scripts defining model layers and architecture.
  - Various Jupyter notebooks (`.ipynb`) for feature extraction, classification, and training.
  - `dataset_summary.csv`: A summary of the dataset used.
- **SINGLE MODAL/**: Contains scripts and notebooks for single modal data processing and model training.
  - Similar structure to the MULTIMODAL directory with specific scripts and notebooks for single modal tasks.
- `accessing_tif_image_data.ipynb`: A notebook for accessing and processing `.tif` image data.

### RESULTS/
This directory stores the results of the model training and evaluation:
- **Multimodal/**: Contains results for multimodal models, organized by crop type (e.g., Peach) and further by model configurations and trials.
  - Each trial directory contains `model_metrics.csv` files with performance metrics.
- **Single/**: Contains results for single modal models, similarly organized by crop type and trials.

### SAMPLE DATA/
This directory contains sample data used for training and evaluation:
- **Peach/**: Contains subdirectories for different dates and types of data (e.g., Aerial UAV Photos, Ground Multispectral Photos).
  - Each subdirectory contains images and annotations organized by categories (e.g., Anarsia lineatella, Dead Trees, Healthy).
  - CSV files with metadata and combined multimodal data.

### Other Files
- `.gitattributes`: Git configuration file.
- `LICENSE`: The project's license file.
- `README.md`: The main README file with project information.
- `Vision_for_Social_Good___Multimodal_Crop_Plant_Disease_Detection_compressed.pdf`: A compressed PDF document related to the project.

This structure ensures that the project is well-organized, with clear separation of code, results, and data, making it easier to navigate and maintain.


## **Installation**
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/IsraaMohamedHamid/Vision-for-Social-Good-Multimodal-Cash-Crop-Disease-Detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Vision-for-Social-Good-Multimodal-Cash-Crop-Disease-Detection
   ```
3. Install the required dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

## **Usage**
To use the project, follow these steps:

1. Preprocess the dataset:
   * Use the provided Jupyter notebooks in the `CODE/DATA COLLECTION` directory to preprocess the RGB, multispectral, and aerial images. For example, you can use `CODE/DATA COLLECTION/cherry_tree_preprocessing.ipynb` to preprocess cherry tree images.

2. Train the models:
   * Use the provided Jupyter notebooks in the `CODE/SINGLE MODAL` and `CODE/MULTIMODAL` directories to train single-modal and multimodal models. For example, you can use `CODE/SINGLE MODAL/ground_imagry training from scratch.ipynb` to train a single-modal model on ground imagery and `CODE/MULTIMODAL/multimodal_training.ipynb` to train a multimodal model.

3. Evaluate the models:
   * Use the provided Jupyter notebooks in the `CODE/SINGLE MODAL` and `CODE/MULTIMODAL` directories to evaluate the trained models. For example, you can use `CODE/SINGLE MODAL/ground_imagry training from scratch.ipynb` to evaluate a single-modal model and `CODE/MULTIMODAL/multimodal_training.ipynb` to evaluate a multimodal model.


## **Contributing**
We welcome contributions to the project. To contribute, follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature-name
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Description of your changes"
   ```
4. Push your changes to your fork:
   ```bash
   git push origin feature-name
   ```
5. Create a pull request to the main repository.

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## **Acknowledgements**
We would like to thank all contributors and collaborators who have supported this project. Special thanks to the authors of the various libraries and tools used in this project.