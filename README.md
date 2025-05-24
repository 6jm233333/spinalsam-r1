```markdown
# SpinalSAM-R1



## Overview

**SpinalSAM-R1** is a novel multimodal interactive system that combines a fine-tuned Segment Anything Model (SAM) enhanced with anatomical attention and Low-Rank Adaptation (LoRA) techniques, integrated with the DeepSeek-R1 large language model for natural language-driven spine CT image segmentation. This system empowers clinicians with precise, efficient, and intuitive spinal segmentation through points, bounding boxes, or natural language commands.

---

## Features

- **Feature-enhanced SAM backbone**: Incorporates Convolutional Block Attention Module (CBAM) and LoRA fine-tuning to improve segmentation accuracy on challenging spinal CT images.
- **Multimodal interaction**: Supports point, box, and text-based prompts for flexible and natural segmentation refinement.
- **Natural language commands**: Integrates DeepSeek-R1 for parsing clinical instructions into segmentation prompts with 94.3% parsing accuracy.
- **Real-time performance**: Achieves sub-200 ms rendering latency and overall sub-800 ms response times for seamless clinical workflows.
- **Cross-platform GUI**: Developed with PyQt5 providing user-friendly interface for visualization, annotation, and interactive command parsing.
- **Comprehensive evaluation**: Outperforms state-of-the-art methods (U-Net, TransUNet, SAM-Med2D) on multiple segmentation metrics including Dice, IoU, MSD, and HD95.

---

## Installation

### Requirements
Python 3.10
torch 1.13.0+cu116
OpenCV 4.11.0
PyQt5 5.15.2
numpy 1.26.4
matplotlib 3.7.0
OpenAI 1.65.4



---

## Usage

### Clone the repository

```bash
git clone https://github.com/6jm233333/spinalsam-r1.git
cd spinalsam-r1
```

### Download pretrained models

Place the fine-tuned SAM checkpoint (e.g., `Medsam_best.pth`) into the `/models` directory.

SAM checkpoint: https://pan.baidu.com/s/13vUpCgdVEtXdiCntuGIV6A?pwd=2hek


### Run the GUI application

```bash
python main.py
```

### Features

- Load spine CT images (`png`, `jpg`, `jpeg`).
- Add coordinate points or boxes interactively.
- Generate segmentation masks from points or boxes.
- Use natural language commands (e.g., "Add three points", "Generate point mask") via built-in AI assistant.
- View segmentation results and evaluation metrics inline.

---

## Model Architecture

Our model fine-tunes the original SAM (ViT-H backbone) with:

- **CBAM Attention Modules**: Enhance feature extraction focused on vertebral edges and anatomical regions.
- **Low-Rank Adaptation (LoRA)**: Parameter-efficient fine-tuning enabling effective domain adaptation with fewer trainable parameters.

The system dynamically integrates DeepSeek-R1 for semantic command parsing, enabling interactive segmentation driven by natural language.

---

## Evaluation

| Method              | Dice Coefficient (DC) | Intersection over Union (IoU) | Mean Surface Distance (MSD) mm | 95% Hausdorff Distance (HD95) mm |
|---------------------|-----------------------|------------------------------|-------------------------------|----------------------------------|
| U-Net               | 0.8675 ± 0.0434       | 0.7725 ± 0.0576              | 3.25 ± 2.68                   | 10.81 ± 12.21                    |
| TransUNet           | 0.8814 ± 0.0434       | 0.7932 ± 0.0650              | 4.00 ± 4.47                   | 11.70 ± 15.21                    |
| Swin-UNet           | 0.8930 ± 0.0261       | 0.8106 ± 0.0395              | 2.55 ± 1.03                   | 7.55 ± 4.37                     |
| SAM-Med2D box-tight | 0.9045 ± 0.0271       | 0.8299 ± 0.0420              | 1.72 ± 0.54                   | 4.46 ± 1.67                     |
| SAM-Med2D point5     | 0.9013 ± 0.0251       | 0.8239 ± 0.0381              | 2.29 ± 1.24                   | 7.04 ± 4.95                     |
| SAM-Med2D mask       | 0.8996 ± 0.0258       | 0.8211 ± 0.0388              | 1.84 ± 0.68                   | 5.49 ± 3.33                     |
| **SpinalSAM-R1 (Ours)**   | **0.9225 ± 0.0674**       | **0.8949 ± 0.0420**              | **1.69 ± 0.41**                   | **4.21 ± 2.73**                     |

*Shows statistically significant improvements over baselines.*

---

## Commands & Interaction

The system supports the following commands via AI Assistant or text input:

- Open/Select Image
- Add X points (e.g., "Add three points")
- Add bounding box / segmentation box
- Generate point mask
- Generate box mask
- Clear data / points / boxes
- Previous/Next image
- Check GPU status
- Verify parameter matching

*Natural language commands are parsed with ~94.3% accuracy with response times under 800ms.*

---

## Citation

If you find SpinalSAM-R1 helpful, please cite our paper:

```bibtex

```

---

## Contact

For questions or contributions, please open an issue or contact us via email:  
- Haipeng Si (sihaipeng1978@email.sdu.edu.cn)  
- Liang Sun (sunl@nuaa.edu.cn)

---

## Acknowledgments

Thanks to the developers of SAM, DeepSeek-R1, PyQt5, and open-source contributors.

---

*Enjoy using SpinalSAM-R1!*  
```
