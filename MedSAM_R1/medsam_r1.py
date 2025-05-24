import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, \
    QMessageBox, QLineEdit, QHBoxLayout, QDialog, QMenuBar, QAction, QTextEdit, QStyle, QDockWidget, QTextBrowser, \
    QFrame, QActionGroup
from PyQt5.QtGui import QPixmap, QImage, QIcon, QPainter, QPen, QColor, QTextCursor, QMovie, QFont
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QEvent
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import os
import time
from PyQt5.QtGui import QIcon
from openai import OpenAI
import re
from functools import partial

# ------------------ 高端扁平化QSS方案（推荐！） ------------------
highend_qss = """

2
QMainWindow, QWidget {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #23262930, stop:1 #23262930),
        url(resources/bg.jpg) center/cover no-repeat;
    color: #e7eaf0;
    font-family: 'Segoe UI', 'PigFang SC', 'Microsoft YaHei', 'Arial';
    font-size: 15px;
}


QTabWidget::pane {
    border-top: 2px solid #3A91E5;
    background: #181A1B;
    padding: 8px;
    border-radius: 10px;
}
QTabBar::tab {
    background: #29313A;
    color: #A5B8C8;
    border: none;
    padding:10px 30px;
    min-width:100px;
    min-height:30px;
    border-radius:10px 10px 0 0;
    margin-right: 3px;
    font-weight: bold;
}
QTabBar::tab:selected, QTabBar::tab:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3A91E5 stop:1 #5B9EF4);
    color: white;
}

QGroupBox {
    border:1.5px solid #39598F;
    border-radius:11px;
    margin-top: 15px;
    font-size:16px;
    font-weight:bold;
    background: #23282C;
    padding:10px;
}
QGroupBox:title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left:32px;
    top:-12px;
    background-color:#181A1B;
    color: #7FC5FF;
    padding: 2px 12px;
}

QLabel {
    color: #B691DB;
    font-size:15px;
    font-weight: 500;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {
    background: #18202A;
    border: 1.5px solid #39598F;
    border-radius:9px;
    color: #ECF2FD;
    padding: 6px 12px;
    font-size:15px;
}
QTableWidget {
    background:#15191C;
    color:#D1E4FF;
    border-radius: 7px;
    font-size:14px;
    gridline-color: #384556;
}
QHeaderView::section {
    background: #39598F;
    color: #FFFFFF;
    font-weight:bold;
    border: none;
    border-radius:7px;
    padding: 5px;
}
QTableWidget QTableCornerButton::section {
    background: #39598F;
}

QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #5275F6, stop:1 #3A91E5);
    color: #FFFFFF;
    border:none;
    border-radius: 11px;
    font-size: 16px;
    font-weight: bold;
    padding: 8px 26px;
    margin: 5px;
    min-width: 96px;
    min-height: 36px;
    transition: all .3s;
}
QPushButton:hover {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #4F7AE5, stop:1 #3988EA);
}
QPushButton:pressed {
    background: #274E92;
}

QTextEdit[readOnly="true"] {
    background: #1B2227;
    color: #7FC5FF;
    border:none;
    border-radius:10px;
    font-size:15px;
    padding: 8px;
}
QStatusBar {
    background: #181A1B;
    font-size:14px;
    color: #90B8FC;
    border-top: 2px solid #39598F;
    min-height: 28px;
}
"""


def set_highend_style(app):
    app.setStyle('Fusion')
    app.setStyleSheet(highend_qss)


# 图像处理函数（保持不变）
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def calculate_dice(pred_mask, true_mask):
    """计算 Dice 系数"""
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = pred_mask.sum() + true_mask.sum()
    dice = (2 * intersection) / (union + 1e-7)  # 避免除零
    return dice


def generate_image_with_points(image_path, input_points, input_labels, mask_folder, ground_truth_mask=None):
    """根据坐标点生成掩码"""
    start_time = time.time()

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model_type = "vit_h"
    sam_checkpoint = "./models/Medsam_best.pth"
    # sam_checkpoint = "./models/sam_vit_h_4b8939.pth"
    # sam_checkpoint = "./models/best.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )

    end_time = time.time()
    inference_time = (end_time - start_time) * 10  # 转换为毫秒

    dice_scores = []
    if ground_truth_mask is not None:
        for mask in masks:
            dice = calculate_dice(mask, ground_truth_mask)
            dice_scores.append(dice)
        dice_mean = np.mean(dice_scores)
        dice_std = np.std(dice_scores)
    else:
        dice_mean, dice_std = None, None

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())  # 修正为 mask
        show_points(input_points, input_labels, plt.gca())
        plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        mask_filename = f"mask_{i + 1}.png"
        plt.savefig(os.path.join(mask_folder, mask_filename), bbox_inches='tight', pad_inches=0)
        plt.close()

    return masks, scores, dice_mean, dice_std, inference_time


def generate_image_with_box(image_path, input_box, mask_folder, ground_truth_mask=None):
    """根据坐标框生成掩码"""
    start_time = time.time()

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model_type = "vit_h"
    sam_checkpoint = "./models/Medsam_best.pth"
    # sam_checkpoint = "./models/sam_vit_h_4b8939.pth"
    # sam_checkpoint = "./models/best.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=True,
    )

    end_time = time.time()
    inference_time = (end_time - start_time) * 10  # 转换为毫秒

    dice_scores = []
    if ground_truth_mask is not None:
        for mask in masks:
            dice = calculate_dice(mask, ground_truth_mask)
            dice_scores.append(dice)
        dice_mean = np.mean(dice_scores)
        dice_std = np.std(dice_scores)
    else:
        dice_mean, dice_std = None, None

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_box(input_box, plt.gca())
        plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        mask_filename = f"box_mask_{i + 1}.png"
        plt.savefig(os.path.join(mask_folder, mask_filename), bbox_inches='tight', pad_inches=0)
        plt.close()

    return masks, scores, dice_mean, dice_std, inference_time


# 新增自定义事件类型
class CommandEvent(QEvent):
    _type = QEvent.Type(QEvent.registerEventType())

    def __init__(self, callback):
        super().__init__(self._type)
        self.callback = callback

    def execute(self):
        try:
            self.callback()
        except Exception as e:
            QMessageBox.critical(QApplication.activeWindow(), "Run Error!", str(e))

ENGLISH_NUMBERS = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20
        } #英文对齐
class AIWorker(QObject):
    process_requested = pyqtSignal(str)
    response_received = pyqtSignal(str, bool)
    error_occurred = pyqtSignal(str)

    def __init__(self, api_key):
        super().__init__()
        self.client = OpenAI(api_key="sk-1b0f815baa9d4f499542fbc36ef2e37b", base_url="https://api.deepseek.com")



        # 字典结构
        self.command_map = {
            r'打开图片|选择'
            r'图片|openpicture|choosepicture|OpenPicture|Ope'
            r'npictures|openpictures|Choosepicture|ChooseP'
            r'icture|Choosepictures|ChoosePictures': {
                'command': 'open_image'
            },
            r'添加([一二三四五六七八九十百千万两\d]+)[个]?坐标点': {'handler': self.handle_add_points},
            r'add\s*([0-9]+)\s*points': {'handler': self.handle_add_points},
            r'add\s*([1]+)\s*point': {'handler': self.handle_add_points},
            r'Add\s*([1]+)\s*point': {'handler': self.handle_add_points},
            r'Add\s*([0-9]+)\s*points': {'handler': self.handle_add_points},
            r'add([a-z]+)point[s]?': {'handler': self.handle_add_points},
            r'add([a-z]+)point?': {'handler': self.handle_add_points},
            r'Add([a-z]+)point[s]?': {'handler': self.handle_add_points},
            r'Add([a-z]+)point?': {'handler': self.handle_add_points},
            #  'convert': self.chinese_to_number
            r'生成点掩码|创建点掩码|Maskgeneration|maskgeneration|Maskgenerate|maskgenerate|pointmask|Pointmask|pointsmask|Pointmas'
            r'k|Pointmasks|pointmasks|pointsmasks|Pointsmasks|generatepointmask|generatepointmasks|Generatepointmask|Generatepointmasks': {
                'command': 'generate_point_mask'
            },
            r'生成框掩码|创建框掩码|Maskgeneration|maskgeneration|Maskgenerate|maskgenerate|boxmask|Boxmask|Boxmasks|boxmasks|generateboxmask|generateboxmasks|Generateboxmasks|Generateboxmask': {
                'command': 'generate_box_mask'
            },
            r'验证参数|parameter|parameters|Parameter|Parameters|VerifyingParameter|VerifyingParameters|Verifyingparameter|Verifyingparamete'
            r'rs|verifyparameter|verifyparameters|Verifyparameter|Verifyparameters': {
                'command': 'check_parameters'
            },
            r'GPU状态|GPU|gpu|GPUstatus|GPUStatus|verifyGPU|VerifyGPU|CheckGPU|Checkgpu|CheckingGPU|Checkinggpu': {
                'command': 'check_gpu'
            },
            r'清空|清空数据|clear|reset|empty|clean|clearall|resetdata|emptydata|cleardata|resetpicture|emptypicture|clearpicture|Clear|Reset|Empty|Clearpicture|Cleardata': {
                'command': 'clear_images'
            },
            r'添加坐标框|选择分割框|addbox|Addbox|AddBox|addboundingbox|Addboundingbox|addrectangle|Addrectangle|selectbox|Selectbox|selectboundingbox|Selectboundingbox|selectsegmentationbox|Selectsegmentationbox|newbox|Newbox|createbox|Createbox': {
                'command': 'add_box'
            },
            r'上一张|上一张图片|previousimage|Previousimage|previmage|Previmage|lastimage|Lastimage|previouspicture|Previouspicture|prevpicture|Prevpicture|showprevious|Showprevious|backimage|Backimage|showlast|Showlast|goback|Goback|goprevious|Goprevious': {
                'command': 'show_prev_image'
            },
            r'下一张|下一张图片|nextimage|Nextimage|nextpicture|Nextpicture|shownext|Shownext|forwardimage|Forwardimage|gonext|Gonext|shownextimage|Shownextimage|next|Next|skipimage|Skipimage|nextslide|Nextslide|forwardpicture|Forwardpicture': {
                'command': 'show_next_image'
            }
        }
        self.process_requested.connect(self.process_message)

    def markdown_to_html(self, text):
        """将Markdown格式的文本转换为HTML"""
        # 替换换行符为<br>
        text = text.replace('\n', '<br>')

        # 处理Markdown标题 (# 和 ##)
        lines = text.split('<br>')
        processed_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith('##'):
                # 二级标题
                processed_lines.append(f'<h2>{line[2:].strip()}</h2>')
            elif line.startswith('#'):
                # 一级标题
                processed_lines.append(f'<h1>{line[1:].strip()}</h1>')
            else:
                # 普通文本
                processed_lines.append(line)

        # 合并处理后的行
        html_text = '<br>'.join(processed_lines)

        # 确保文本被<p>标签包裹以保持一致的样式
        html_text = f'<p style="margin: 0; padding: 0;">{html_text}</p>'

        return html_text

    # 消息处理
    def process_message(self, message):
        """处理用户输入的核心方法"""
        try:
            # 先尝试本地指令解析
            if self.detect_commands(message):
                return

            # 本地指令未匹配时调用API
            completion = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": message}],
                temperature=0.1,
                stream=True
            )

            # 处理流式响应
            full_response = []
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    full_response.append(chunk.choices[0].delta.content)

            # 将响应合并并转换为HTML
            raw_response = ''.join(full_response)
            html_response = self.markdown_to_html(raw_response)

            self.response_received.emit(html_response, False)

        except Exception as e:
            self.error_occurred.emit(f"请求失败: {str(e)}")
            self.detect_commands(message)  # API失败时再次尝试本地解析
            return False

        except Exception as e:
            self.error_occurred.emit(f"请求失败: {str(e)}")
            self.detect_commands(message)  # API失败时再次尝试本地解析
            return False  # 添加明确的返回值

    def handle_add_points(self, match, num_override=None):
        try:
            if num_override is not None:
                num = num_override
            else:
                num_str = None
                if match and match.lastindex:
                    num_str = match.group(1)
                if num_str is None:
                    num = 1
                else:
                    # 英文数字优先映射，否则尝试数字，否则中文
                    num_lower = num_str.lower()
                    if num_lower in ENGLISH_NUMBERS:
                        num = ENGLISH_NUMBERS[num_lower]
                    elif num_str.isdigit():
                        num = int(num_str)
                    else:
                        num = self.chinese_to_number(num_str)
            QApplication.postEvent(
                QApplication.instance().main_window,
                CommandEvent(lambda: self.execute_add_points(num))
            )
        except Exception as e:
            self.error_occurred.emit(f"参数解析失败: {str(e)}")

    def execute_add_points(self, num):
        main_window = QApplication.instance().main_window
        for _ in range(num):
            main_window.add_point()

    def chinese_to_number(self, s):
        # 支持从1-99的简单实现
        map_digit = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
                     '六': 6, '七': 7, '八': 8, '九': 9}
        if s.isdigit():
            return int(s)
        if '十' in s:
            parts = s.split('十')
            if parts[0] == '':  # "十"开头
                left = 1
            else:
                left = map_digit.get(parts[0], 0)
            if len(parts) == 1 or parts[1] == '':
                right = 0
            else:
                right = map_digit.get(parts[1], 0)
            return left * 10 + right
        else:
            res = 0
            for c in s:
                if c in map_digit:
                    res = res * 10 + map_digit[c]
            return res if res > 0 else 1  # 默认1

    def detect_commands(self, text):
        try:
            cleaned_text = re.sub(r'[请帮可以吗？\s]', '', text).strip()
            for pattern, config in self.command_map.items():
                match = re.search(pattern, cleaned_text, re.IGNORECASE)
                if match:
                    if 'handler' in config:
                        # 直接用handler处理（比如处理带数量的点）
                        config['handler'](match)
                    else:
                        # 正常处理
                        QApplication.postEvent(QApplication.instance().main_window,
                                               CommandEvent(
                                                   lambda cmd=config['command']:
                                                   QApplication.instance().main_window.execute_command(cmd)
                                               ))
                    return True
            self.show_command_list()
            return False
        except Exception as e:
            self.error_occurred.emit(f"指令解析错误: {str(e)}")
            return False

    def show_command_list(self):
        """显示可用指令列表"""
        help_text = """<div style="font-size: 16px; line-height: 1.6;">
        <h3>📚 Available Command List (Text Supported):</h3>

        <h4>🖼️ Image Operations:</h4>
        <ul>
          <li>Open/Select Image</li>
          <li>Clear Image/Clear</li>
        </ul>

        <h4>📍 Point Operations:</h4>
        <ul>
          <li>Add [number] points (e.g., Add three points)</li>
          <li>Generate Point Mask</li>
          <li>Clear Points</li>
        </ul>

        <h4>🟦 Bounding Box Operations:</h4>
        <ul>
          <li>Add Bounding Box/Select Segmentation Box</li>
          <li>Generate Box Mask</li>
          <li>Clear Bounding Boxes</li>
        </ul>

        <h4>🔍 Other Functions:</h4>
        <ul>
          <li>Previous Image/Next Image</li>
          <li>GPU Status/GPU</li>
          <li>Verify Parameter Matching</li>
        </ul>
        </div>"""
        self.response_received.emit(help_text, False)

    def trigger_command(self, command_name):
        QApplication.postEvent(QApplication.instance().main_window,
                               CommandEvent(
                                   lambda: QApplication.instance().main_window.execute_command(command_name)
                               ))


class ChatDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MEDSAM AI Assistant")
        self.setWindowIcon(QIcon(r"I:\MedSAM_R1\MedSAM_R1\UI\snow.ico"))
        self.resize(800, 600)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # 标题区域
        title_label = QLabel("MEDSAM Intelligent Assistant")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #9A7EEB;
                padding: 8px 0;
            }
        """)
        main_layout.addWidget(title_label)

        # 分割线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: #E2E8F0;")
        main_layout.addWidget(separator)

        # 聊天记录区域
        self.chat_history = QTextBrowser()
        self.chat_history.setStyleSheet("""
            QTextBrowser {
                background-color: #FFF8E1; /* 米黄色背景 */
                border-radius: 8px;
                padding: 16px;
                font-size: 18px; /* 增大字体 */
                border: 1px solid #E4E7EB;
                min-height: 400px;
            }
        """)

        # 输入区域
        input_layout = QHBoxLayout()
        input_layout.setSpacing(12)

        # 输入框
        self.input_field = QTextEdit()
        self.input_field.setMaximumHeight(100)
        self.input_field.setPlaceholderText("Input message...")
        self.input_field.setStyleSheet("""
            QTextEdit {
                border: 1px solid #E4E7EB;
                border-radius: 8px;
                padding: 12px;
                font-size: 16px;
                background-color: white;
                color: #000000; /* 确保文字是黑色 */
            }
        """)

        # 按钮区域
        button_layout = QVBoxLayout()
        button_layout.setSpacing(8)

        # 各种按钮
        send_btn = QPushButton("Send")
        send_btn.setFixedSize(80, 40)
        send_btn.setStyleSheet("""
            QPushButton {
                background-color: #007BFF;
                color: white;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0056B3;
            }
        """)

        help_btn = QPushButton("Help")
        help_btn.setFixedSize(80, 40)
        help_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedSize(80, 40)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #6C757D;
                color: white;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5A6268;
            }
        """)

        # 添加按钮到布局
        button_layout.addWidget(send_btn)
        button_layout.addWidget(help_btn)
        button_layout.addWidget(clear_btn)

        # 组装输入区域
        input_layout.addWidget(self.input_field)
        input_layout.addLayout(button_layout)

        # 组装主布局
        main_layout.addWidget(self.chat_history)
        main_layout.addLayout(input_layout)
        self.setLayout(main_layout)

        # 初始化AI助手
        self.worker = AIWorker("sk-1b0f815baa9d4f499542fbc36ef2e37b")
        self.worker.response_received.connect(self.append_assistant_message)
        self.worker.error_occurred.connect(self.append_error_message)

        # 连接信号
        send_btn.clicked.connect(self.send_message)
        help_btn.clicked.connect(self.show_help)
        clear_btn.clicked.connect(self.clear_input)

        self.input_field.installEventFilter(self)#事件过滤器方式

        # 加载动画
        self.loading_movie = QMovie("loading.gif")
        self.loading_label = QLabel()
        self.loading_label.setMovie(self.loading_movie)
        self.loading_label.hide()

        # 添加初始问候
        self.append_system_message(
            "Hello! I am the MED-SAM AI Assistant, here to assist you with image segmentation operations anytime!")

    def eventFilter(self, obj, event):
        if obj == self.input_field and event.type() == QEvent.KeyPress:
            # 只拦截需要特殊处理的回车键
            if event.key() == Qt.Key_Return and not event.modifiers() & Qt.ShiftModifier:
                self.send_message()
                return True

            # 其他所有按键交给默认处理
            return False

        return super().eventFilter(obj, event)



    def show_help(self):
        """显示帮助信息"""
        self.worker.show_command_list()

    def clear_input(self):
        """清空输入框"""
        self.input_field.clear()
        self.input_field.setFocus()  # 清空后将焦点放回输入框

    def append_message(self, html, class_name):
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(f'<div class="{class_name}">{html}</div><br>')
        self.chat_history.ensureCursorVisible()

    def append_user_message(self, text):
        self.append_message(f'<div style="float:right">{text}</div>', "user-message")

    def append_assistant_message(self, text):
        self.append_message(f'<div style="float:left">{text}</div>', "assistant-message")

    def append_system_message(self, text):
        self.append_message(text, "system-message")

    def append_error_message(self, text):
        self.append_message(text, "error-message")

    def handle_key_press(self, event):
        """修复键盘事件处理"""
        if event.key() == Qt.Key_Return and not event.modifiers() & Qt.ShiftModifier:
            self.send_message()
            event.accept()
        else:
            # 允许正常键盘输入（包括删除键和数字键）
            super(QTextEdit, self.input_field).keyPressEvent(event)
            event.accept()

    def showEvent(self, event):
        # 窗口显示时加载动画
        self.loading_movie.start()
        super().showEvent(event)

    def show_welcome_message(self):
        """显示欢迎信息"""
        welcome_msg = """欢迎使用智能图像处理助手！您可以语音或文字指令以下功能：
           \n🔍 选择图片 | 📍 添加X个坐标点 | 🖼 生成点掩码
           \n🟦 添加坐标框 | 📦 生成框掩码 | 🧹 清空
           \n⚙️ 验证参数 | 💻 GPU | ◀️▶️ 上一张，下一张 切换图片
           \n\n直接输入指令如："添加3个坐标点 并生成掩码\""""
        self.append_system_message(welcome_msg)

    def send_message(self):
        text = self.input_field.toPlainText().strip()
        if not text:
            return

        self.append_user_message(text)
        self.input_field.clear()

        # 显示加载状态
        self.append_system_message('<i class="fa fa-spinner fa-spin"></i> 正在处理请求...')
        self.worker.process_requested.emit(text)

    def update_chat(self, message, is_command=False):
        if is_command:
            self.chat_history.setTextColor(QColor(255, 165, 0))  # 橙色显示命令
            self.chat_history.append(f"检测到指令：{message}")
            self.chat_history.append("请按照提示完成操作。")
        else:
            self.chat_history.setTextColor(QColor(0, 0, 255))  # 蓝色显示回复
            self.chat_history.append(message)
        self.chat_history.moveCursor(QTextCursor.End)

    def show_error(self, error_msg):
        self.chat_history.setTextColor(QColor(255, 0, 0))
        self.chat_history.append(error_msg)


class MainWindow(QMainWindow):
    command_signal = pyqtSignal(str)
    theme_changed = pyqtSignal(str)  # 主题变更信号

    def __init__(self):
        super().__init__()
        icon_path = r"I:\Gra_paper_SAM\segment-anything-main\snow.ico"  # 使用 ICO 文件
        self.setWindowIcon(QIcon(icon_path))  # 设置自定义图标
        self.setWindowTitle("SpinalSAM-R1")
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel(self)
        self.resize(1500, 900)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        central_widget = QWidget()
        central_widget.setLayout(self.layout)

        self.setCentralWidget(central_widget)
        self.add_debug_menu()  # 在初始化时增加调试菜单
        self.create_menu()
        self.initUI()

        # 在MainWindow初始化时添加样式表
        # self.setStyleSheet("""
        #     QMainWindow {
        #         background-color: #2D2D30;
        #         font-family: 'Microsoft YaHei';
        #     }
        #     QLabel {
        #      color: #363F4B;  /* 冷色调蓝白色（符合现代UI趋势） */
        #     font-size: 16px;
        #     text-shadow: 0 1px 3px rgba(0, 92, 175, 0.3);  /* 匹配按钮主题的阴影 */
        #     }
        #     QMenuBar {
        #         background-color: #252526;
        #         color: #DCDCDC;
        #     }
        #     QMenuBar::item:selected {
        #         background-color: #094771;
        #     }
        #     QPushButton {
        #         background-color: #005B9F;
        #         color: yellow;
        #         border-radius: 4px;
        #         padding: 5px 10px;
        #         min-width: 80px;
        #     }
        #     QPushButton:hover {
        #         background-color: #007ACC;
        #     }
        # """)
        self.command_handlers = {
            'open_image': self.open_image,
            'add_point': self.add_point,
            'generate_point_mask': self.generate_point_mask,
            'generate_box_mask': self.generate_box_mask,
            'check_parameters': self.check_parameters,
            'check_gpu': self.check_gpu,
            'clear_images': self.clear_images,
            'add_box': self.add_box,
            'show_prev_image': self.show_prev_image,
            'show_next_image': self.show_next_image
        }
        self.command_signal.connect(self.handle_command)

    def event(self, event):
        if event.type() == CommandEvent._type:
            event.execute()
            return True
        return super().event(event)

    def execute_command(self, command_name):
        handler = self.command_handlers.get(command_name)
        if handler:
            try:
                handler()
                self.chat_dialog.update_chat(f"✅ {command_name} 执行成功", False)
            except Exception as e:
                self.chat_dialog.show_error(f"执行失败: {str(e)}")
        else:
            self.chat_dialog.show_error(f"未知指令: {command_name}")

    def update_metrics(self, dice, time):
        """更新分析数据面板的内容"""
        # 获取当前图片尺寸
        if self.image_path:
            image = cv2.imread(self.image_path)
            if image is not None:
                self.img_height, self.img_width = image.shape[:2]
            else:
                self.img_width, self.img_height = 0, 0
        else:
            self.img_width, self.img_height = 0, 0

        # 处理 Dice 系数的显示
        dice_str = f"{dice:.3f}" if dice is not None else "N/A"
        # 处理时间（确保传入的是数值）
        time_val = time if time is not None else 0.0
        # <b>Dice系数:</b> <span style='color:#CE9178;'>{dice}</span><br>
        # 使用HTML模板更新显示内容
        html_template = """
        <h3 style='color:#4EC9B0;'>Analysis of segmentation results</h3>
        <p style='font-size:16px;'>

            <b>Inference Time:</b> {time:.2f} ms<br>
            <b>Image size:</b> {width}x{height}
        </p>
        """.format(
            dice=dice_str,
            time=time_val,
            width=self.img_width,
            height=self.img_height
        )
        self.info_text.setHtml(html_template)

    def add_debug_menu(self):
        """添加调试工具菜单"""
        debug_menu = self.menuBar().addMenu("Debugging tools")

        # 添加参数校验动作
        param_action = QAction("Verifying parameter matching", self)
        param_action.triggered.connect(self.check_parameters)
        debug_menu.addAction(param_action)

        # 添加设备检查动作
        device_action = QAction("Checking GPU status", self)
        device_action.triggered.connect(self.check_gpu)
        debug_menu.addAction(device_action)

    def check_parameters(self):
        """检查参数匹配状态"""
        try:
            from segment_anything import sam_model_registry

            model_type = "vit_h"
            # checkpoint_path = "./models/best.pth"
            # checkpoint_path = "./models/sam_vit_h_4b8939.pth"
            checkpoint_path = "./models/Medsam_best.pth"

            # 加载原始模型结构
            sam = sam_model_registry[model_type](checkpoint=None)

            # 加载检查点
            checkpoint = torch.load(checkpoint_path)

            # 尝试参数匹配
            missing, unexpected = sam.load_state_dict(checkpoint, strict=False)

            # 构建信息消息
            info_msg = [
                "Parameter verification results:",
                f"※ Model Type: {model_type}",
                f"※ Checkpoint path:{checkpoint_path}",
                f"※ Total number of parameters:{len(checkpoint)}",
                f"※ Matching parameters: {len(checkpoint) - len(missing) - len(unexpected)}",
                f"※ Missing parameters:{len(missing)} (Includes adapter layer)",
                f"※ Redundant parameters:{len(unexpected)}"
            ]

            # 显示详细结果
            QMessageBox.information(self, "Parameter validation", "\n".join(info_msg), QMessageBox.Ok)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed parameter validation:{str(e)}", QMessageBox.Ok)

    def check_gpu(self):
        """检查GPU可用性"""
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0

        msg = [
            "Hardware status:",
            f"※ PyTorch version: {torch.__version__}",
            f"※ CUDA is available: {'True' if cuda_available else 'False'}",
            f"※ Number of GPU: {device_count}",
            f"※ Current device: {'GPU' if torch.cuda.is_available() else 'CPU'}"
        ]

        QMessageBox.information(self, "Hardware inspection", "\n".join(msg), QMessageBox.Ok)

    def create_menu(self):
        menu_bar = self.menuBar()

        # 基本功能菜单
        basic_func_menu = menu_bar.addMenu("Basic functions")
        open_action = QAction("Selecting images", self)
        open_action.triggered.connect(self.open_image)
        clear_action = QAction("Clear the picture", self)
        clear_action.triggered.connect(self.clear_images)
        basic_func_menu.addAction(open_action)
        basic_func_menu.addAction(clear_action)

        # 坐标点分割菜单
        point_split_menu = menu_bar.addMenu("Coordinate point segmentation")
        add_point_action = QAction("Adding coordinate points", self)
        add_point_action.triggered.connect(self.confirm_points)
        generate_point_action = QAction("Mask generation", self)
        generate_point_action.triggered.connect(self.generate_point_mask)
        clear_points_action = QAction("Clear coordinate points", self)
        clear_points_action.triggered.connect(self.clear_points)
        point_split_menu.addAction(add_point_action)
        point_split_menu.addAction(generate_point_action)
        point_split_menu.addAction(clear_points_action)

        # 坐标框分割菜单
        box_split_menu = menu_bar.addMenu("Coordinate box segmentation")
        add_box_action = QAction("Select split box", self)
        add_box_action.triggered.connect(self.add_box)
        generate_box_action = QAction("Mask generation", self)
        generate_box_action.triggered.connect(self.generate_box_mask)
        clear_box_action = QAction("Clear the segmentation box", self)
        clear_box_action.triggered.connect(self.clear_box)
        box_split_menu.addAction(add_box_action)
        box_split_menu.addAction(generate_box_action)
        box_split_menu.addAction(clear_box_action)

        # 观察图片菜单
        view_images_menu = menu_bar.addMenu("Observe the picture")
        prev_image_action = QAction("Previous slide", self)
        prev_image_action.triggered.connect(self.show_prev_image)
        next_image_action = QAction("Next slide", self)
        next_image_action.triggered.connect(self.show_next_image)
        view_images_menu.addAction(prev_image_action)
        view_images_menu.addAction(next_image_action)

        # ======== 新增主题切换菜单 ======== ↓↓↓
        # 在菜单栏添加新的"主题"子菜单
        theme_menu = menu_bar.addMenu("Style")

        light_action = QAction("Bright mode", self, checkable=True)
        light_action.triggered.connect(partial(self.set_theme, "light"))

        dark_action = QAction("Dark mode", self, checkable=True)
        dark_action.triggered.connect(partial(self.set_theme, "dark"))

        # 添加到菜单并设置单选效果
        theme_group = QActionGroup(self)
        theme_group.addAction(light_action)
        theme_group.addAction(dark_action)
        theme_group.setExclusive(True)

        theme_menu.addAction(light_action)
        theme_menu.addAction(dark_action)

        # 默认选中当前主题
        if hasattr(self, 'current_theme'):
            light_action.setChecked(self.current_theme == "light")
            dark_action.setChecked(self.current_theme == "dark")

        # 新增AI助手菜单
        ai_menu = self.menuBar().addMenu("AI Assistant")
        chat_action = QAction("Open a chat window", self)
        chat_action.triggered.connect(self.show_chat)
        ai_menu.addAction(chat_action)

    def show_chat(self):
        if not hasattr(self, 'chat_dialog') or self.chat_dialog is None:
            self.chat_dialog = ChatDialog(self)
            # Set default theme if not already set
            if not hasattr(self, 'current_theme'):
                self.current_theme = "dark"  # or "light"
            self.update_chat_dialog_theme(self.current_theme)
        self.chat_dialog.show()

    def execute_command(self, command_name):
        try:
            method = getattr(self, command_name)
            method()
        except AttributeError:
            self.chat_dialog.show_error(f"Instruction not found:{command_name}")

    def handle_command(self, command):
        self.execute_command(command)

    def initUI(self):
        self.image_path = None
        self.input_points = np.empty((0, 2), dtype=np.float32)
        self.input_labels = np.empty((0,), dtype=np.float32)
        self.input_box = None
        self.mask_folder = "masks"
        self.original_folder = "originals"
        self.mask_images = []
        self.current_image_index = 0

        if not os.path.exists(self.mask_folder):
            os.makedirs(self.mask_folder)
        if not os.path.exists(self.original_folder):
            os.makedirs(self.original_folder)

        self.size_label = QLabel(self)
        self.size_label.setAlignment(Qt.AlignCenter)
        self.size_label.setStyleSheet("font-size: 24px; color: #666;")
        self.size_label.setText("Current image size: No image selected")

        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 24px; color: #333;")
        self.result_label.setText("Inference Time: N/A ms")

        self.layout.addWidget(self.size_label)
        self.layout.addWidget(self.result_label)

        # ================== 新增代码：创建分析数据面板 ==================
        # 创建右侧分析数据面板
        self.info_panel = QDockWidget("Data Analysis", self)
        self.info_text = QTextEdit()
        self.info_text.setStyleSheet("""
               background-color: #1E1E1E;
               color: #DCDCDC;
               border: 1px solid #3F3F46;
               padding: 10px;
           """)
        self.info_panel.setWidget(self.info_text)
        self.addDockWidget(Qt.RightDockWidgetArea, self.info_panel)

        # 初始化默认分析数据
        self.update_metrics(dice=0.0, time=0.0)

    def set_theme(self, theme):
        """完整的主题切换函数（支持亮/暗模式）
        Args:
            theme (str): 'light' 或 'dark'
        """
        self.current_theme = theme  # 记录当前主题状态

        # ===================== 暗色模式样式 =====================
        DARK_THEME = """
        /* 全局应用 */
        * {
            font-family: 'Microsoft YaHei', 'Segoe UI';
        }

        /* 主窗口及所有子部件 */
        QMainWindow, QWidget, QDialog {
            background-color: #2D2D30;
            color: #DCDCDC;
        }

        /* 中央部件 */
        QWidget#centralWidget {
            background-color: #252526;
        }

        /* 菜单栏 */
        QMenuBar {
            background-color: #252526;
            color: #DCDCDC;
            padding: 2px;
            border-bottom: 1px solid #1E1E1E;
        }
        QMenuBar::item {
            padding: 5px 10px;
            background: transparent;
            border-radius: 4px;
        }
        QMenuBar::item:selected {
            background-color: #094771;
        }
        QMenuBar::item:pressed {
            background-color: #073052;
        }

        /* 子菜单 */
        QMenu {
            background-color: #1E1E1E;
            color: #DCDCDC;
            border: 1px solid #3F3F46;
            padding: 5px;
        }
        QMenu::item {
            padding: 5px 30px 5px 20px;
            border-radius: 4px;
        }
        QMenu::item:selected {
            background-color: #094771;
        }
        QMenu::separator {
            height: 1px;
            background-color: #3F3F46;
            margin: 5px 0px;
        }

        /* 按钮 */
        QPushButton {
            background-color: #005B9F;
            color: #FFFFFF;
            border: none;
            border-radius: 4px;
            padding: 5px 10px;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #007ACC;
        }
        QPushButton:pressed {
            background-color: #003D66;
        }

        /* 标签 */
        QLabel {
            color: #B6CEE0;
            font-size: 15px;
        }
        QLabel#titleLabel {
            font-size: 24px;
            font-weight: bold;
            color: #7FC5FF;
        }

        /* 输入框 */
        QLineEdit, QTextEdit {
            background-color: #1E1E1E;
            color: #DCDCDC;
            border: 1px solid #3F3F46;
            border-radius: 4px;
            padding: 5px;
        }

        /* 停靠面板 */
        QDockWidget {
            background: #252526;
            color: #DCDCDC;
            border: 1px solid #3F3F46;
            titlebar-close-icon: url(close_light.png);
            titlebar-normal-icon: url(maximize_light.png);
        }
        QDockWidget::title {
            background: #252526;
            padding: 3px;
            text-align: left;
        }

        /* 对话框 */
        QDialog {
            background-color: #2D2D30;
            color: #DCDCDC;
        }

        /* 文本浏览器 */
        QTextBrowser {
            background-color: #1E1E1E;
            color: #DCDCDC;
            border: 1px solid #3F3F46;
            border-radius: 4px;
        }
        """

        # ===================== 明亮模式样式 =====================
        LIGHT_THEME = """
        /* 全局应用 */
        * {
            font-family: 'Microsoft YaHei', 'Segoe UI';
        }

        /* 主窗口及所有子部件 */
        QMainWindow, QWidget, QDialog {
            background-color: #F5F7FA;
            color: #333333;
        }

        /* 中央部件 */
        QWidget#centralWidget {
            background-color: #FFFFFF;
        }

        /* 菜单栏 */
        QMenuBar {
            background-color: #F0F0F0;
            color: #333333;
            padding: 2px;
            border-bottom: 1px solid #D3D3D3;
        }
        QMenuBar::item {
            padding: 5px 10px;
            background: transparent;
            border-radius: 4px;
        }
        QMenuBar::item:selected {
            background-color: #D3E3FD;
        }
        QMenuBar::item:pressed {
            background-color: #B0D0FF;
        }

        /* 子菜单 */
        QMenu {
            background-color: #FFFFFF;
            color: #333333;
            border: 1px solid #D3D3D3;
            padding: 5px;
        }
        QMenu::item {
            padding: 5px 30px 5px 20px;
            border-radius: 4px;
        }
        QMenu::item:selected {
            background-color: #D3E3FD;
        }
        QMenu::separator {
            height: 1px;
            background-color: #D3D3D3;
            margin: 5px 0px;
        }

        /* 按钮 */
        QPushButton {
            background-color: #4A90E2;
            color: #FFFFFF;
            border: none;
            border-radius: 4px;
            padding: 5px 10px;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #6AA8F4;
        }
        QPushButton:pressed {
            background-color: #2A70C0;
        }

        /* 标签 */
        QLabel {
            color: #363F4B;
            font-size: 15px;
        }
        QLabel#titleLabel {
            font-size: 24px;
            font-weight: bold;
            color: #2A70C0;
        }

        /* 输入框 */
        QLineEdit, QTextEdit {
            background-color: #FFFFFF;
            color: #333333;
            border: 1px solid #D3D3D3;
            border-radius: 4px;
            padding: 5px;
        }

        /* 停靠面板 */
        QDockWidget {
            background: #FFFFFF;
            color: #333333;
            border: 1px solid #D3D3D3;
            titlebar-close-icon: url(close_dark.png);
            titlebar-normal-icon: url(maximize_dark.png);
        }
        QDockWidget::title {
            background: #F0F0F0;
            padding: 3px;
            text-align: left;
        }

        /* 对话框 */
        QDialog {
            background-color: #F5F7FA;
            color: #333333;
        }

        /* 文本浏览器 */
        QTextBrowser {
            background-color: #FFFFFF;
            color: #333333;
            border: 1px solid #D3D3D3;
            border-radius: 4px;
        }
        """

        # ===================== 应用主题 =====================
        # 先清空样式
        self.setStyleSheet("")

        # 给中央组件设置对象名称以便直接定位
        self.centralWidget().setObjectName("centralWidget")

        # 应用全局样式
        app = QApplication.instance()
        app.setStyleSheet(LIGHT_THEME if theme == "light" else DARK_THEME)

        # 特殊控件单独设置
        self.info_text.setStyleSheet("""
            background-color: %s;
            color: %s;
            border: 2px solid %s;
            padding: 10px;
            font-size: 16px;
        """ % (
            "#FFFFFF" if theme == "light" else "#1E1E1E",
            "#333333" if theme == "light" else "#DCDCDC",
            "#D3D3D3" if theme == "light" else "#3F3F46"
        ))

        # 更新聊天对话框样式（如果存在）
        if hasattr(self, 'chat_dialog') and self.chat_dialog is not None:
            self.update_chat_dialog_theme(theme)

        # 发送主题变更信号
        self.theme_changed.emit(theme)
        QApplication.processEvents()  # 强制刷新界面

    def update_chat_dialog_theme(self, theme):
        """更新聊天对话框的主题样式
        Args:
            theme (str): 'light' 或 'dark'
        """
        if not hasattr(self, 'chat_dialog') or self.chat_dialog is None:
            return

        if theme == "light":
            # 明亮主题
            self.chat_dialog.setStyleSheet("""
                QDialog {
                    background-color: #F5F7FA;
                    font-family: 'Segoe UI', 'Microsoft YaHei';
                }
                QLabel {
                    color: #2D3748;
                    font-size: 16px;
                    font-weight: bold;
                }
                QTextBrowser {
                    background-color: #FFF8E1; /* 米黄色背景 */
                    border-radius: 8px;
                    padding: 16px;
                    font-size: 16px; /* 增大字体 */
                    border: 1px solid #E4E7EB;
                    color: #333333; /* 黑色文字 */
                }
                QTextEdit {
                    background-color: #FFFFFF;
                    border-radius: 8px;
                    padding: 12px;
                    font-size: 16px; /* 增大字体 */
                    border: 1px solid #E4E7EB;
                    color: #000000; /* 黑色文字 */
                }
                QPushButton {
                    background-color: #4A90E2;
                    color: white;
                    border-radius: 6px;
                    padding: 10px 20px;
                    min-width: 90px;
                    border: none;
                    font-size: 15px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #3A80D2;
                }
            """)
            assistant_bg = "#F1F3F5"
            assistant_text = "#212529"
            system_text = "#6C757D"
            h1_color = "#2D3748"
            h2_color = "#4A5568"
        else:
            # 暗黑主题
            self.chat_dialog.setStyleSheet("""
                QDialog {
                    background-color: #2D2D30;
                    font-family: 'Segoe UI', 'Microsoft YaHei';
                }
                QLabel {
                    color: #B1A4EC; /* 紫色标题 */
                    font-size: 16px;
                    font-weight: bold;
                }
                QTextBrowser {
                    background-color: #3A3D41; /* 深色但稍微亮一点的背景 */
                    border-radius: 8px;
                    padding: 16px;
                    font-size: 16px; /* 增大字体 */
                    border: 1px solid #4D4D4D;
                    color: #E0E0E0; /* 浅色文字 */
                }
                QTextEdit {
                    background-color: #252526;
                    border-radius: 8px;
                    padding: 12px;
                    font-size: 16px; /* 增大字体 */
                    border: 1px solid #3F3F46;
                    color: #FFFFFF; /* 白色文字 */
                }
                QPushButton {
                    background-color: #0E639C;
                    color: white;
                    border-radius: 6px;
                    padding: 10px 20px;
                    min-width: 90px;
                    border: none;
                    font-size: 15px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #1177BB;
                }
            """)
            assistant_bg = "#2D2D30"
            assistant_text = "#B691DB"
            system_text = "#A0A0A0"
            h1_color = "#B1A4EC"
            h2_color = "#A0A0A0"

        # 更新CSS样式，使用 %% 转义 % 字符
        css = """
            .user-message {
                background-color: #007BFF;
                color: white;
                border-radius: 12px;
                padding: 12px;
                margin: 8px 0;
                max-width: 70%%;
                float: right;
                clear: both;
            }
            .assistant-message {
                background-color: %s;
                color: %s;
                border-radius: 12px;
                padding: 12px;
                margin: 8px 0;
                max-width: 70%%;
                float: left;
                clear: both;
            }
            .assistant-message h1 {
                font-size: 20px;
                font-weight: bold;
                margin: 8px 0;
                color: %s;
            }
            .assistant-message h2 {
                font-size: 18px;
                font-weight: bold;
                margin: 6px 0;
                color: %s;
            }
            .system-message {
                color: %s;
                text-align: center;
                margin: 8px 0;
                clear: both;
            }
            .error-message {
                color: #DC3545;
                border: 1px solid #DC3545;
                border-radius: 12px;
                padding: 12px;
                margin: 8px 0;
                max-width: 70%%;
                float: left;
                clear: both;
            }
        """ % (assistant_bg, assistant_text, h1_color, h2_color, system_text)

        self.chat_dialog.chat_history.document().setDefaultStyleSheet(css)

    def confirm_points(self):
        num_points_dialog = QDialog(self)
        num_points_dialog.setWindowTitle("Enter the number of coordinate points")
        num_points_dialog_layout = QVBoxLayout()

        point_label = QLabel("The number of coordinate points to enter:")
        num_points_dialog_layout.addWidget(point_label)

        num_input = QLineEdit()
        num_points_dialog_layout.addWidget(num_input)

        confirm_button = QPushButton("Confirmation")
        confirm_button.clicked.connect(lambda: self.confirm_num_points(int(num_input.text()), num_points_dialog))
        num_points_dialog_layout.addWidget(confirm_button)

        num_points_dialog.setLayout(num_points_dialog_layout)
        num_points_dialog.exec_()

    def confirm_num_points(self, num, dialog):
        for _ in range(num):
            self.add_point()
        dialog.accept()

    def add_point(self):
        point_dialog = QDialog(self)
        point_dialog.setWindowTitle("Input coordinate points")
        point_dialog_layout = QVBoxLayout()

        point_label = QLabel("X-coordinate:")
        point_dialog_layout.addWidget(point_label)

        x_input = QLineEdit()
        point_dialog_layout.addWidget(x_input)

        point_label = QLabel("Y-coordinate:")
        point_dialog_layout.addWidget(point_label)

        y_input = QLineEdit()
        point_dialog_layout.addWidget(y_input)

        point_label = QLabel("Label (0 or 1):")
        point_dialog_layout.addWidget(point_label)

        label_input = QLineEdit()
        point_dialog_layout.addWidget(label_input)

        confirm_button = QPushButton("Confirmation")
        confirm_button.clicked.connect(
            lambda: self.confirm_point(x_input.text(), y_input.text(), label_input.text(), point_dialog))
        point_dialog_layout.addWidget(confirm_button)

        point_dialog.setLayout(point_dialog_layout)
        point_dialog.exec_()

    def confirm_point(self, x, y, label, dialog):
        try:
            x, y = float(x), float(y)
            label = int(label)
            if label not in [0, 1]:
                raise ValueError("The label must be either 0 or 1")

            if self.input_points.size == 0:
                self.input_points = np.array([[x, y]])
            else:
                self.input_points = np.vstack((self.input_points, [[x, y]]))

            if self.input_labels.size == 0:
                self.input_labels = np.array([label])
            else:
                self.input_labels = np.append(self.input_labels, label)

            dialog.accept()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Mistyped: {e}", QMessageBox.Ok)

    def clear_points(self):
        self.input_points = np.array([])
        self.input_labels = np.array([])
        QMessageBox.information(self, "Information", "All coordinates have been cleared", QMessageBox.Ok)

    def add_box(self):
        box_dialog = QDialog(self)
        box_dialog.setWindowTitle("Enter the coordinates of the split box")
        box_dialog_layout = QVBoxLayout()

        coords = ["X Left", "Y Top", "X Right", "Y Bottom"]
        self.box_inputs = []
        for coord in coords:
            label = QLabel(f"{coord}:")
            box_dialog_layout.addWidget(label)
            input_field = QLineEdit()
            self.box_inputs.append(input_field)
            box_dialog_layout.addWidget(input_field)

        confirm_button = QPushButton("Confirmation")
        confirm_button.clicked.connect(self.confirm_box)
        box_dialog_layout.addWidget(confirm_button)

        box_dialog.setLayout(box_dialog_layout)
        box_dialog.exec_()

    def confirm_box(self):
        try:
            x1, y1, x2, y2 = [int(input.text()) for input in self.box_inputs]
            self.input_box = np.array([x1, y1, x2, y2])
            QMessageBox.information(self, "Information", "Segmentation box confirmed", QMessageBox.Ok)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Mistyped: {e}", QMessageBox.Ok)

    def generate_box_mask(self):
        if not self.image_path:
            QMessageBox.warning(self, "Warnings", "Please select the image!", QMessageBox.Ok)
            return
        if self.input_box is None:
            QMessageBox.warning(self, "Warnings", "Please enter the split box first!", QMessageBox.Ok)
            return
        try:
            ground_truth_mask = None
            masks, scores, dice_mean, dice_std, inference_time = generate_image_with_box(
                self.image_path, self.input_box, self.mask_folder, ground_truth_mask
            )

            # 更新分析数据面板
            self.update_metrics(dice_mean, inference_time)

            if dice_mean is not None and dice_std is not None:
                result_text = f"Dice Score: {dice_mean:.3f} ± {dice_std:.3f}\nInference time: {inference_time:.2f} ms"
            else:
                result_text = f"Inference time: {inference_time:.2f} ms"
            self.result_label.setText(result_text)

            self.mask_images.clear()
            for i in range(len(masks)):
                mask_filename = f"box_mask_{i + 1}.png"
                self.mask_images.append(mask_filename)
            self.current_image_index = 0
            self.update_image(self.mask_images[self.current_image_index])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred while generating the image: {e}", QMessageBox.Ok)

    def clear_box(self):
        self.input_box = None
        QMessageBox.information(self, "Information", "The split box has been cleared", QMessageBox.Ok)

    def generate_point_mask(self):
        if not self.image_path:
            QMessageBox.warning(self, "Warnings", "Please select the image!", QMessageBox.Ok)
            return
        if self.input_points.size == 0:
            QMessageBox.warning(self, "Warnings", "Please add the coordinates first!", QMessageBox.Ok)
            return
        try:
            ground_truth_mask = None
            masks, scores, dice_mean, dice_std, inference_time = generate_image_with_points(
                self.image_path, self.input_points, self.input_labels, self.mask_folder, ground_truth_mask
            )
            # 更新分析数据面板
            self.update_metrics(dice_mean, inference_time)

            if dice_mean is not None and dice_std is not None:
                result_text = f"IoU: {dice_mean:.3f} ± {dice_std:.3f}\nInference time: {inference_time:.2f} ms"
            else:
                result_text = f"Inference time: {inference_time:.2f} ms"
            self.result_label.setText(result_text)

            self.mask_images.clear()
            for i in range(len(masks)):
                mask_filename = f"mask_{i + 1}.png"
                self.mask_images.append(mask_filename)
            self.current_image_index = 0
            self.update_image(self.mask_images[self.current_image_index])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred while generating the image:{e}", QMessageBox.Ok)

    def show_prev_image(self):
        if self.mask_images and self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_image(self.mask_images[self.current_image_index])

    def show_next_image(self):
        if self.mask_images and self.current_image_index < len(self.mask_images) - 1:
            self.current_image_index += 1
            self.update_image(self.mask_images[self.current_image_index])

    def clear_images(self):
        for filename in os.listdir(self.mask_folder):
            os.remove(os.path.join(self.mask_folder, filename))
        for filename in os.listdir(self.original_folder):
            os.remove(os.path.join(self.original_folder, filename))

        self.mask_images = []
        self.current_image_index = 0
        self.image_label.clear()
        self.image_path = None
        self.size_label.setText("No image is currently selected, please select the spine image!")
        self.size_label.setStyleSheet("font-size: 24px; color: #666;")
        self.result_label.setText("IoU: N/A\nInference Time: N/A ms")
        self.input_points = np.empty((0, 2), dtype=np.float32)
        self.input_labels = np.empty((0,), dtype=np.float32)
        self.input_box = None

        QMessageBox.information(self, "Tips", "All image data has been cleared!", QMessageBox.Ok)

    def clear_pictures(self):  # 添加别名方法
        self.clear_images()

    def open_image(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Choose Picture", "", "Image files (*.png *.jpg *.jpeg)")
        if self.image_path:
            image = cv2.imread(self.image_path)
            if image is not None:
                h, w = image.shape[:2]
                self.size_label.setText(f"Current image size:Width {w}px，Height {h}px")
                self.size_label.setStyleSheet("font-size: 24px; color: #333;")
            self.update_image()
            self.save_image(cv2.imread(self.image_path), self.original_folder, os.path.basename(self.image_path))

    def update_image(self, filename=None):
        if filename:
            image_path = os.path.join(self.mask_folder, filename)
        else:
            image_path = self.image_path

        if self.image_path:
            image = cv2.imread(self.image_path)
            if image is not None:
                h, w = image.shape[:2]
                self.size_label.setText(f"Current image size:Width {w}px，Height {h}px")

        image = cv2.imread(image_path)
        if image is None:
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation
        )

        pixmap = QPixmap(p.size())
        pixmap.fill(Qt.white)
        painter = QPainter(pixmap)
        painter.drawImage(0, 0, p)

        if self.image_path:
            self.draw_scale(painter, p.width(), p.height(), w, h)

        painter.end()
        self.image_label.setPixmap(pixmap)

    def draw_scale(self, painter, pix_width, pix_height, img_width, img_height):  # 255分割框标识
        pass
        pen = QPen(QColor(255, 0, 255), 1)
        painter.setPen(pen)
        font=QFont()
        font.setPointSize(6)  # 标注刻   度的字体大小
        painter.setFont(font)
        scale_x = pix_width / img_width
        scale_y = pix_height / img_height

        max_scale = 512
        num_ticks = 10
        tick_interval = max_scale / num_ticks

        for i in range(num_ticks + 1):
            value = max_scale - i * tick_interval
            x = int((value / max_scale) * img_width * scale_x)
            painter.drawLine(x, pix_height - 10, x, pix_height)
            painter.drawText(x - 10, pix_height - 15, f"{int(value)}")

        for i in range(num_ticks + 1):
            value = max_scale - i * tick_interval
            y = int((value / max_scale) * img_height * scale_y)
            painter.drawLine(0, y, 10, y)
            painter.drawText(15, y + 5, f"{int(value)}")

    def save_image(self, image, folder, filename):
        if image is not None and filename is not None:
            if not os.path.exists(folder):
                os.makedirs(folder)
            cv2.imwrite(os.path.join(folder, filename), image)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    set_highend_style(app)
    QApplication.instance().main_window = main_window
    main_window.show()
    sys.exit(app.exec_())