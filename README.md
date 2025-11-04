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


| Method              | Dice Coefficient (DC)↑ | Intersection over Union (IoU)↑ | Mean Surface Distance (MSD)↓ mm | 95% Hausdorff Distance (HD95)↓ mm |
|---------------------|-----------------------|------------------------------|-------------------------------|----------------------------------|
| U-Net               | 0.8700 ± 0.0144\*       | 0.7861 ± 0.0238              | 3.25 ± 1.43                   | 23.05 ± 12.05                    |
| TransUNet           | 0.9335 ± 0.0002\*       | 0.9113 ± 0.0005\*            | 1.92 ± 0.06\*                 | 5.58 ± 2.01\*                    |
| Swin-UNet           | 0.8863 ± 0.0016\*       | 0.9097 ± 0.0012\*            | 3.64 ± 1.37\*                 | 4.79 ± 0.02\*                    |
| SAM-Med2D(Box)      | 0.9316 ± 0.0012\*       | 0.8738 ± 0.0031\*            | 2.25 ± 0.54\*                 | 6.14 ± 1.41\*                    |
| SAM-Med2D(Point)    | 0.9329 ± 0.0011\*       | 0.8760 ± 0.0029\*            | 2.21 ± 0.53\*                 | 6.08 ± 4.95\*                    |
| **SpinalSAM-R1 (Ours)**   | **0.9532 ± 0.0005**       | **0.9114 ± 0.0015**              | **1.81 ± 0.50**                   | **5.47 ± 0.73**                    |

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
@misc{liu2025spinalsamr1visionlanguagemultimodalinteractive,
      title={SpinalSAM-R1: A Vision-Language Multimodal Interactive System for Spine CT Segmentation}, 
      author={Jiaming Liu and Dingwei Fan and Junyong Zhao and Chunlin Li and Haipeng Si and Liang Sun},
      year={2025},
      eprint={2511.00095},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.00095}, 
}
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
