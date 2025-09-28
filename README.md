# 🏠 AI House Drawing Psychological Analysis Model (YOLOv8 Based)

## 🎯 Project Overview and Theoretical Background

This project utilizes the **YOLOv8** deep learning object detection model to analyze user-drawn house sketches and diagnose psychological characteristics. It quantifies the presence, size, and location of objects (doors, windows, sun, chimney, etc.), converts this data into psychological score indicators, and provides a customized analysis script.

### 📚 Key References and Theory

| Item | Description |
| :--- | :--- |
| **Data (AI Hub)** | Utilized house drawing datasets from the Korea Institute of Information and Communications Technology Planning and Evaluation (AI Hub) for AI model training. |
| **Jolles, I. (1964), A Guide to the House-Tree-Person Test** | Referred to the HTP (House-Tree-Person) test guidebook, a projective technique used for psychological analysis, to establish the theoretical foundation for house drawing analysis. |
| **Rubin, J. (1984), The Art of Art Therapy** | Referenced this book on art therapy to gain a broad understanding of psychological analysis through drawing. |

### ⚙️ Technology Stack

| Category | Technology / Framework |
| :--- | :--- |
| **Model** | YOLOv8s (Ultralytics) |
| **Frameworks** | PyTorch, NumPy |
| **Environment** | Google Colab, Python 3.8+ |

-----

## 🗂️ Project Folder Structure

For successful execution, your project folder must adhere to the following structure:

```
/Your_Project_Root/
├── data/
│   ├── Training/
│   │   ├── Images/  (Original JPG/PNG image files)
│   │   └── Labels/  (Converted YOLO TXT label files)
│   └── Validation/
│       ├── Images/
│       └── Labels/
├── detections/      (Trained model weights storage)
│   └── yolov8s-final/
│       └── weights/best.pt
├── main.py          (Final execution and analysis pipeline for local run)
├── scoring_system.py  (4 Psychological index scoring logic)
├── case_classifier.py (Score-based 24 case classification logic)
├── convert_json_to_txt.py (JSON → TXT conversion script)
├── scripts.json     (Analysis script data for 24 cases)
└── data.yaml        (YOLO training configuration file)
```

-----

## 🚀 Execution Guide (Colab Environment Focus)

### 1\. Colab Execution (Recommended)

Uploading the `YOLO_Prj.ipynb` file to Colab and running the cells sequentially will automate the entire process from environment setup to final analysis.

| Step | File/Configuration | Description |
| :--- | :--- | :--- |
| **1. Environment Setup** | `YOLO_Prj.ipynb` | Mounts Google Drive, installs `ultralytics`, and adds the project root path (`ROOT_DIR`) to the system path. |
| **2. Data Preprocessing** | `convert_json_to_txt.py` logic | Executes the conversion of original JSON labels into YOLO-compatible `.txt` files. **(User execution required)** |
| **3. YOLOv8 Model Training** | `train_model.py` logic | Starts YOLOv8 model training based on the `data.yaml` configuration. **(User execution required)** |
| **4. Psychological Analysis** | `main.py` logic | Loads the trained `best.pt` weights, performs object detection on the test image, and runs the custom analysis pipeline. |

#### Augmented Batch Example
<img src="https://github.com/user-attachments/assets/76cf6d83-b473-4aff-92bf-7d2dce358454" width="200" alt="Augmented Batch Example"/>

-----

## 🧠 2. Local Environment Execution

To run the project locally, ensure you have secured the trained weights from Colab and follow these steps.

1.  **Secure Weights:** Download the `best.pt` file from the Colab training folder (`detections/yolov8s-final/weights/`) and place it in your local project structure.
2.  **Path Configuration:** Modify the `ROOT_DIR` variable within `data.yaml` and `main.py` to your local absolute path.
3.  **Execution:** Run the following command from the directory containing `main.py`.

<!-- end list -->

```bash
python main.py
```

## Expected Output

The console will display the analysis results. 

<img width="1934" height="1239" alt="KakaoTalk_20250928_154013979" src="https://github.com/user-attachments/assets/2714dc14-b485-4ba8-8859-f6d93faf227b" />


