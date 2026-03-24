# SpinalSAM-R1

## Overview

**SpinalSAM-R1** is a novel multimodal interactive system that combines a fine-tuned **Segment Anything Model (SAM)** enhanced with anatomical attention and **Low-Rank Adaptation (LoRA)** techniques, integrated with the **DeepSeek-R1** large language model for natural language-driven spine CT image segmentation. 

This system empowers clinicians with precise, efficient, and intuitive spinal segmentation through points, bounding boxes, or natural language commands.

---

## Features

* **Feature-Enhanced SAM Backbone**: Incorporates a Convolutional Block Attention Module (**CBAM**) and **LoRA** fine-tuning on the ViT-H architecture to improve segmentation accuracy on complex spinal CT images while avoiding overfitting.
* **Multimodal Interaction**: Supports point, box, and text-based prompts for flexible and natural segmentation refinement.
* **Natural Language Commands**: Integrates **DeepSeek-R1** for parsing clinical instructions into structured segmentation prompts, supporting 11 clinical operation types with a 94.3% parsing accuracy.
* **Real-Time Performance**: Optimized with **ONNX Runtime**, achieving 250–300 ms inference latency per slice on consumer-grade hardware (RTX 4060 Laptop), with overall system response times under 800 ms.
* **Cross-Platform GUI**: Developed with **PyQt5**, providing a user-friendly interface for high-resolution visualization, manual annotation, and interactive command parsing.
* **Robust Generalization**: Rigorously validated on both an internal clinical dataset (120 lumbar CT scans) and the multi-center external **VerSe** dataset, outperforming state-of-the-art networks and generalist LVLM pipelines (e.g., Qwen3-VL + SAM).

---

## Installation

### Requirements

* Python 3.10
* torch 1.13.0+cu116
* OpenCV 4.11.0
* PyQt5 5.15.2
* numpy 1.26.4
* matplotlib 3.7.0
* OpenAI 1.65.4

---

## Usage

### Clone the repository

```bash
git clone [https://github.com/6jm233333/spinalsam-r1.git](https://github.com/6jm233333/spinalsam-r1.git)
cd spinalsam-r1
```

### Run the GUI application

```bash
python main.py
```

### Interface Capabilities

* **Load Data**: Open spine CT images (`png`, `jpg`, `jpeg`).
* **Manual Annotation**: Add coordinate points or bounding boxes interactively.
* **Generate Masks**: Create segmentation masks from provided points or boxes.
* **AI Assistant**: Use natural language commands (e.g., *"Add three points"*, *"Generate point mask"*) via the built-in DeepSeek-R1 parser.
* **Real-Time Feedback**: View segmentation results and quantitative evaluation metrics directly inline.

---

## Model Architecture

Our framework is a slice-based 2D segmentation system prioritizing computational efficiency. It fine-tunes the original SAM (ViT-H backbone) using:

1. **CBAM Attention Modules**: Sequentially applies channel and spatial attention to 2D-reshaped feature maps to enhance vertebral boundaries and salient anatomical regions.
2. **Low-Rank Adaptation (LoRA)**: Parameter-efficient fine-tuning that updates only a fraction of parameters, retaining generalization while adapting to spinal CT characteristics.

The system dynamically integrates **DeepSeek-R1** in the business logic layer for semantic command parsing, enabling an intuitive, closed-loop workflow driven by natural language.

---

## Evaluation

### Internal Clinical Dataset

Performance on 120 lumbar CT scans (31,454 slices) from Shandong University Qilu Hospital:

| Method                  | Dice (DC)↑          | IoU↑                | MSD↓            | HD95↓           |
| ----------------------- | ------------------- | ------------------- | --------------- | --------------- |
| U-Net                   | 0.8701 ± 0.0152*    | 0.7847 ± 0.0261*    | 3.35 ± 1.65*    | 23.18 ± 12.81*  |
| TransUNet               | 0.9327 ± 0.0002*    | 0.9097 ± 0.0006*    | 1.88 ± 0.06*    | 5.66 ± 2.27*    |
| Swin-UNet               | 0.8847 ± 0.0019*    | 0.9104 ± 0.0013*    | 3.75 ± 1.58*    | 5.81 ± 0.02*    |
| SAM-Med2D (Box)         | 0.9329 ± 0.0013*    | 0.8717 ± 0.0033*    | 2.37 ± 0.62*    | 6.25 ± 1.63*    |
| SAM-Med2D (Point)       | 0.9325 ± 0.0014*    | 0.8750 ± 0.0034*    | 2.33 ± 0.61*    | 6.20 ± 5.63*    |
| **SpinalSAM-R1 (Ours)** | **0.9535 ± 0.0005** | **0.9098 ± 0.0017** | **1.78 ± 0.54** | **5.44 ± 0.80** |

** Shows statistically significant improvements over baselines (p < 0.05).*

> **Note on Generalist LVLMs**: When benchmarked against a pipeline combining Qwen3-VL with vanilla SAM, our method significantly outperformed the generalist approach (Qwen3-VL + SAM achieved DC: 0.9459 ± 0.0050, IoU: 0.8974 ± 0.0090, MSD: 2.15 ± 0.85, HD95: 5.56 ± 0.92).

### External Validation (VerSe Dataset)

Zero-shot generalization performance on the multi-center VerSe dataset:

| Method                  | DC↑                 | IoU↑                | MSD↓             | HD95↓             |
| ----------------------- | ------------------- | ------------------- | ---------------- | ----------------- |
| U-Net                   | 0.5804 ± 0.0386*    | 0.4099 ± 0.0392*    | 39.08 ± 3.69*    | 78.53 ± 7.55*     |
| TransUNet               | 0.7012 ± 0.0514*    | 0.6288 ± 0.0411*    | 21.33 ± 4.12*    | 62.45 ± 5.18*     |
| Swin-UNet               | 0.6940 ± 0.0822*    | 0.6380 ± 0.0469*    | 24.59 ± 3.54*    | 68.29 ± 4.83*     |
| SAM-Med2D (Box)         | 0.7125 ± 0.1245*    | 0.6110 ± 0.1320*    | 18.25 ± 6.44*    | 60.14 ± 12.41*    |
| SAM-Med2D (Point)       | 0.7088 ± 0.1311*    | 0.6045 ± 0.1385*    | 19.01 ± 7.12*    | 61.08 ± 13.95*    |
| Qwen3-VL + SAM          | 0.7314 ± 0.1050*    | 0.6274 ± 0.1190*    | 16.56 ± 7.92*    | 58.45 ± 15.22*    |
| **SpinalSAM-R1 (Ours)** | **0.7650 ± 0.1641** | **0.6415 ± 0.1688** | **14.41 ± 5.79** | **55.72 ± 21.40** |

---

## Commands & Interaction

The system supports the following commands via AI Assistant or text input:

* Open/Select Image
* Add X points (e.g., "Add three points")
* Add bounding box / segmentation box
* Generate point mask
* Generate box mask
* Clear data / points / boxes
* Previous/Next image
* Check GPU status
* Verify parameter matching

*Natural language commands are parsed with ~94.3% accuracy with response times under 800ms.*

---

## Citation

If you find SpinalSAM-R1 helpful, please cite our paper:

```bibtex
@misc{liu2025spinalsamr1visionlanguagemultimodalinteractive,
      title={SpinalSAM-R1: A Vision-Language Multimodal Interactive System for Spine CT Segmentation}, 
      author={Jiaming Liu and Dingwei Fan and Junyong Zhao and Chunlin Li and Haipeng Si and Liang Sun},
      year={2025},
      eprint={2511.00095},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={[https://arxiv.org/abs/2511.00095](https://arxiv.org/abs/2511.00095)}, 
}

```

---

## Contact

For questions or contributions, please open an issue or contact us via email:

* Liang Sun (sunl@nuaa.edu.cn)

---

## Acknowledgments

Thanks to the developers of SAM, DeepSeek-R1, PyQt5, and open-source contributors. This work was supported in part by the National Natural Science Foundation of China.

---

*Enjoy using SpinalSAM-R1!*

