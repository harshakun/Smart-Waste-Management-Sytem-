
# ♻️ Smart Waste Management System (SWMS)

A practical and scalable solution for transforming urban sanitation using AI-driven waste classification and IoT-integrated smart bins.

>  **Published Article**  
>  *Transforming Urban Sanitation: Enhancing Sustainability through Machine Learning-Driven Waste Processing*  
>  [MDPI Sustainability Journal (Open Access)](https://www.mdpi.com/2940500)  
>  *Impact Factor: 3.3 | CiteScore: 7.7*

---

## 🔍 Project Overview

We propose a **Smart Waste Management System (SWMS)** that leverages **Machine Learning** and **Edge Computing** to automate the waste processing pipeline. At the core of this system lies a **Smart Bin**, equipped with ultrasonic sensors and intelligent sorting mechanisms to monitor, classify, and segregate waste in real-time.

The trained model, based on **MobileNetV2**, is capable of classifying waste into 12 distinct categories with high accuracy. The solution is further optimized for real-world deployment using **TensorFlow Lite**, making it suitable for embedded systems like Raspberry Pi or microcontrollers. Data collected from the smart bins is processed at fog nodes, reducing latency and bandwidth usage before being sent to the cloud for deeper analytics.

This system addresses key urban challenges by:

- Reducing manual intervention in waste sorting
- Improving recycling efficiency
- Enabling real-time monitoring of bin fill levels
- Supporting data-driven decisions for waste collection logistics
- Promoting environmental sustainability through automation


---

## 📦 Features

-  Machine Learning-based waste classification  
-  Built using MobileNetV2 and TensorFlow/Keras  
-  Supports 12 categories of waste  
-  Includes dataset preparation, model training, evaluation & visualization  
-  TensorFlow Lite export for deployment on edge devices  
-  Automatic train/test split and preprocessing

---

##  Model Architecture

- Base: `MobileNetV2` (pretrained on ImageNet)  
- Layers:
  - `GlobalAveragePooling2D`
  - `Dense(128, activation='relu')`
  - `Dense(12, activation='softmax')`

---

##  Dataset

- Structured in folders (each folder = waste class)  
- Image augmentation (rotation, zoom, shift, flip) for better generalization  
- Automatic 80/20 split into training and testing CSVs
  
---

## 📁 Folder Structure

```text
📦 SWMS
 ┣ 📂 archive (4)
 ┃ ┗ 📂 garbage_classification
 ┃   ┣ 📂 Plastic
 ┃   ┣ 📂 Organic
 ┃   ┣ 📂 Metal
 ┃   ┣ ...
 ┣ 📜 SWMS.py
 ┣ 📜 train_dataset.csv
 ┣ 📜 test_dataset.csv
 ┗ 📜 waste_management.tflite

```
## Tech Stack

- Python  
- TensorFlow / Keras  
- MobileNetV2 (Transfer Learning)  
- NumPy, Pandas, Matplotlib, Seaborn  
- TensorFlow Lite for edge deployment  

---

##  Model Evaluation

-  Training vs Validation Accuracy & Loss Graphs  
-  Confusion Matrix  
-  Classification Report (Precision, Recall, F1-score)  
-  Final Accuracy printed and plotted  

---

##  Setup Instructions

1. **Clone the repository**

    ```bash
    git clone https://github.com/yourusername/smart-waste-management.git
    cd smart-waste-management
    ```

2. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare the dataset**

    - Download or use your own garbage classification dataset  
    - Structure it as:

    ```text
    garbage_classification/
     ├── Plastic/
     ├── Organic/
     ├── Metal/
     └── ...
    ```

    - Update the dataset path in `SWMS.py` accordingly

4. **Run the script**

    ```bash
    python SWMS.py
    ```

---

##  Authors

This project is part of an academic research initiative aimed at sustainable urban development.

---

##  Citation

If you use this project in your research, please cite the original paper:

```bibtex
[Authors]. "Transforming Urban Sanitation: Enhancing Sustainability through Machine Learning-Driven Waste Processing." Sustainability 2024. https://www.mdpi.com/2940500

