# DINO Object Detection on IIT Delhi Pedestrian Dataset
An assignment submission by Aniruddh Mantrala, student of MSRIT, Bangalore.

This repository contains the code and methodology used to train and evaluate the DINO object detection model on the IIT Delhi Pedestrian Dataset. The project was conducted as part of an internship assignment focused on computer vision tasks.

## Table of Contents
- [Instructions](#instructions)
- [Project Overview](#project-overview)
- [Dataset Preparation](#dataset-preparation)
- [Model Setup](#model-setup)
- [Evaluation](#evaluation)
- [Fine-Tuning Process](#fine-tuning-process)
- [Challenges Faced](#challenges-faced)
- [Conclusion](#conclusion)
- [Model Weights](#model-weights)
- [Original Paper](#original-paper)

## Instructions

1. **Clone the repository**  
   Start by cloning this repository to your local machine:
   ```bash
   git clone https://github.com/DarkSamurai1907/IITDVision-CV.git
   cd your-repo-name
   ```

2. **Download the dataset**  
   Obtain the IIT Delhi Pedestrian Dataset and run the preprocessing notebook:
   - Open `IITDVision_Data_Preprocessing.ipynb`.
   - Set up the dataset paths as required for the training and validation directories.
   - Execute the notebook to prepare the dataset.

3. **Download the model weights**  
   Download the fine-tuned model weights from the following link:  
   [Fine-Tuned Model Weights](https://drive.google.com/file/d/126M-yYpeDdtS6olqdihLukdMcMa-SE04/view?usp=sharing)

4. **Update model path**  
   Open the `IITVision_Inference_and_Visualization.ipynb` notebook and update the `model_checkpoint_path` variable with the downloaded weights.

5. **Run the inference notebook**  
   Run `IITVision_Inference_and_Visualization.ipynb` to:
   - Visualize the model's predictions on the validation set.
   - Obtain the Average Precision (AP) score for pedestrian detection.
   - Generate the Precision-Recall graph.

## Project Overview
The goal of this project was to train and fine-tune the DINO object detection model on a custom pedestrian dataset collected from the IIT Delhi campus. The model was evaluated on its ability to detect pedestrians and other objects. Initial testing with a pre-trained ResNet-50 model showed strong results, achieving an Average Precision (AP) score of 0.7726 for the "person" class. Fine-tuning, however, resulted in a lower AP score of 0.4546, mainly due to dataset challenges such as mislabeled and overlapping bounding boxes.

## Dataset Preparation
The IIT Delhi Pedestrian Dataset consists of 200 images in COCO format, divided into training and validation sets. The dataset preparation involved:
1. Creating the required directory structure.
2. Randomly shuffling and splitting the dataset into training (160 images) and validation (40 images).
3. Filtering annotations and organizing them into JSON files for training and validation.

The dataset was processed using the `IITDVision-Data-Preprocessing.ipynb` notebook.

## Model Setup
The DINO object detection model was set up using a pre-trained checkpoint (`checkpoint0011_4scale.pth`) based on a ResNet-50 backbone. The environment was configured in Google Colab, with the dataset stored in Google Drive for easy access. The setup allowed for both evaluation and fine-tuning on the custom pedestrian dataset.

## Evaluation
The pre-trained DINO model was evaluated on the validation set of 40 images, achieving an AP score of 0.7726 for pedestrian detection. The model performed well under challenging conditions, such as motion blur and distant objects, but also encountered some misclassifications.

## Fine-Tuning Process
The fine-tuning was conducted using 160 training images. Key configuration details included:
- Learning rate: 0.0001 (1e-05 for the ResNet-50 backbone)
- Batch size: 2
- Epochs: 12
- Loss coefficients: Bounding Box Loss (5.0), GIoU Loss (2.0), Focal Loss Alpha (0.25)

Despite some challenges, the fine-tuned model demonstrated improvements in detecting pedestrians in blurry images, but overall accuracy dropped due to issues in the dataset.
The training logs (train and test loss values) are available on `training_logs.txt`.

## Challenges Faced
Some of the key challenges encountered were:
- Environment setup on Google Colab, particularly dependency issues.
- Organizing data and annotations to meet the model's requirements.
- Misclassifications and labeling inaccuracies in the dataset affected the fine-tuning process.

## Conclusion
The DINO model achieved respectable performance in detecting pedestrians in various conditions. However, dataset challenges led to lower fine-tuning accuracy. Future improvements could include data augmentation, refined labeling, and additional hyperparameter tuning.

## Model Weights
You can download the fine-tuned model weights from [here](https://drive.google.com/file/d/126M-yYpeDdtS6olqdihLukdMcMa-SE04/view?usp=sharing).

## Original Paper

### DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection

We present DINO (**D**ETR with **I**mproved de**N**oising anch**O**r boxes), a state-of-the-art end-to-end object detector. DINO enhances the performance and efficiency of previous DETR-like models by employing a contrastive approach for denoising training, a mixed query selection method for anchor initialization, and a "look forward twice" scheme for box prediction. DINO achieves 49.4 AP in 12 epochs and 51.3 AP in 24 epochs on COCO with a ResNet-50 backbone and multi-scale features, yielding significant improvements of **+6.0 AP** and **+2.7 AP**, respectively, compared to DN-DETR, the previous best DETR-like model. DINO scales effectively with both model size and data size. Without additional enhancements, after pre-training on the Objects365 dataset with a SwinL backbone, DINO achieves the best results on both COCO _val2017_ (**63.2 AP**) and _test-dev_ (**63.3 AP**). Compared to other models on the leaderboard, DINO significantly reduces model size and pre-training data size while delivering better performance. Our code will be available [here](https://github.com/IDEACVR/DINO).

```
@article{Zhang2022DINODW,
  title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection},
  author={Hao Zhang and Feng Li and Shilong Liu and Lei Zhang and Hang Su and Jun-Juan Zhu and Lionel Ming-shuan Ni and Heung-yeung Shum},
  journal={ArXiv},
  year={2022},
  volume={abs/2203.03605},
  url={https://api.semanticscholar.org/CorpusID:247292561}
}
```
