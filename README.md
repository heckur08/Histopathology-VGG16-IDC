# 🔬 Histopathology-VGG16-IDC: IDC Image Classification

This repository presents a deep learning implementation for classifying **Invasive Ductal Carcinoma ($\text{IDC}$)** in breast histopathology images. It uses a modified **VGG16** Convolutional Neural Network architecture incorporating **Batch Normalization ($\text{BN}$)** for enhanced performance.

---

## 🚀 Setup and Dependencies

All necessary dependencies can be installed easily using `pip` or `pip3` with the provided requirements file:

```bash
pip install -r requirements.txt
```

## 💻 Repository Structure and Usage

The network implementation is contained within two primary files, catering to both development and deployment phases:

* **`VGG_Histopathology.ipynb`**: The primary Jupyter Notebook for **development**.
    * Details the network configuration and architecture.
    * Includes the complete **training loop** and logic for generating test statistics.

* **`VGG16_BN.py`**: The pure Python file containing the **VGG-16 ($\text{BN}$) network class**.
    * Designed for **deployment** after the model has been trained and saved.
    * This file, along with the saved model state dictionary, is sufficient for running the model in production.

### **Deployment Snippet**

To instantiate and load the trained model in your application, ensure `VGG16_BN.py` is in the same directory, then use:

```python
# Import Network
from VGG16_BN import VGG16
# Model Instantiation
model = VGG16()
# Model loading Statement (Load saved weights here)
...
```
## 🧬 Dataset

This project addresses a crucial problem: developing accurate and robust algorithms for identifying and categorizing **Invasive Ductal Carcinoma ($\text{IDC}$)**, the most common subtype of breast cancer in women. Developing automated methods is highly beneficial due to their ability to process large amounts of data quickly with minimum user involvement.

To assign an aggressiveness grade to a whole mount sample, pathologists typically focus on the regions which contain the $\text{IDC}$. As a result, one of the common pre-processing steps for automatic aggressiveness grading is to delineate the exact regions of $\text{IDC}$ inside of a whole mount slide. This step is already pre-implemented in the context of this dataset.

* **Task**: Binary classification of image patches ($\mathbf{50} \times \mathbf{50}$ size) to detect $\text{IDC}$ regions within whole-mount slides.
* **Patches**: The patches are $\mathbf{50} \times \mathbf{50}$ size.
* **Source**: For more details, please refer to the Kaggle page hosting the dataset.

### **Original Data Links**

* Kaggle Dataset Page: [Link](https://www.kaggle.com/paultimothymooney/breast-histopathology-images)
* Original Dataset Download: [Link](http://gleason.case.edu/webdata/jpi-dl-tutorial/IDC_regular_ps50_idx5.zip)

---

## 📝 Citation

Please cite the following works if you use this implementation or the related methodology:

1.  Janowczyk A, Madabhushi A. Deep learning for digital pathology image analysis: A comprehensive tutorial with selected use cases. J Pathol Inform. 2016;7:29. Published 2016 Jul 26. [doi:10.4103/2153-3539.186902](https://pubmed.ncbi.nlm.nih.gov/27563488/)
2.  Cruz-Roa, Angel, et al. Automatic detection of invasive ductal carcinoma in whole slide images with Convolutional Neural Networks. Progress in Biomedical Optics and Imaging - Proceedings of SPIE. 9041. (2014). [10.1117/12.2043872.](https://spie.org/Publications/Proceedings/Paper/10.1117/12.2043872)
