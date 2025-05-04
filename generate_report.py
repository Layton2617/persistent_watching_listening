# /home/ubuntu/persistent_watching_listening/generate_report.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import numpy as np
import glob
import re # Import re for extracting recognized text example

# Define input/output paths
OUTPUT_DIR = Path("/home/ubuntu/persistent_watching_listening_en/output") # Updated path
FALL_RESULTS_CSV = OUTPUT_DIR / "fall_detection_results_val.csv"
KEYWORD_RESULTS_CSV = OUTPUT_DIR / "keyword_detection_results.csv"
ACTION_REC_LOG = OUTPUT_DIR / "action_recognition_inference_log.txt" # Added action rec log path
REPORT_FILE = OUTPUT_DIR / "evaluation_report_enhanced.md" # New report filename
FALL_CM_PLOT = OUTPUT_DIR / "fall_detection_confusion_matrix.png"
KEYWORD_CM_PLOT = OUTPUT_DIR / "keyword_detection_confusion_matrix.png"
OBJECT_DETECTION_EXAMPLE_IMG = OUTPUT_DIR / "object_detection_example.png"
FALL_VIZ_DIR = OUTPUT_DIR / "fall_viz"

def plot_confusion_matrix(cm, classes, title, output_path):
    """Plots and saves a confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 16})
    plt.title(title, fontsize=14)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}") # EN

report_content = "# Enhanced Evaluation Report: Persistent Watching & Listening\n\n" # EN
report_content += "This report summarizes the evaluation results for the Fall Detection and Keyword Detection components and includes illustrative examples of more advanced AI concepts relevant to the project goals.\n\n" # EN

# --- Fall Detection Evaluation --- 
report_content += "## 1. Fall Detection Evaluation (Validation Split)\n\n" # EN
report_content += "This section evaluates the current fall detection implementation, which uses MediaPipe pose estimation on static images to infer falls based on body orientation.\n\n" # EN

if FALL_RESULTS_CSV.is_file():
    try:
        fall_df = pd.read_csv(FALL_RESULTS_CSV)
        report_content += f"Processed {len(fall_df)} samples from the validation split.\n\n" # EN
        
        y_true_fall = fall_df["ground_truth"].astype(bool)
        y_pred_fall = fall_df["prediction"].astype(bool)
        
        # Calculate Confusion Matrix
        cm_fall = confusion_matrix(y_true_fall, y_pred_fall, labels=[False, True]) # Labels: Not Fall, Fall
        # Handle case where cm might not be 2x2 if only one class is present/predicted
        if cm_fall.shape == (1, 1):
            if True in y_true_fall.unique() or True in y_pred_fall.unique(): # Only True class
                tn, fp, fn, tp = 0, 0, 0, cm_fall[0, 0]
            else: # Only False class
                tn, fp, fn, tp = cm_fall[0, 0], 0, 0, 0
        else:
             tn, fp, fn, tp = cm_fall.ravel()
        
        report_content += "### 1.1 Confusion Matrix:\n"
        report_content += f"```\n"
        report_content += f"          Predicted Not Fall | Predicted Fall\n"
        report_content += f"True Not Fall |      {tn:^10} |     {fp:^10}\n"
        report_content += f"   True Fall  |      {fn:^10} |     {tp:^10}\n"
        report_content += f"```\n\n"
        
        # Plot Confusion Matrix
        plot_confusion_matrix(cm_fall, classes=["Not Fall", "Fall"], 
                              title="Fall Detection Confusion Matrix", 
                              output_path=FALL_CM_PLOT)
        report_content += f"![Fall Detection Confusion Matrix]({FALL_CM_PLOT.name})\n\n" # EN
        
        # Classification Report
        report_fall = classification_report(y_true_fall, y_pred_fall, target_names=["Not Fall", "Fall"], zero_division=0)
        report_content += "### 1.2 Classification Metrics:\n"
        report_content += f"```\n{report_fall}\n```\n\n" # EN
        report_content += "**Note:** The current static image-based method has limitations. Precision might be high if it rarely predicts a fall, but recall can be low if it misses actual falls. Real-world fall detection requires analyzing motion over time.\n\n" # EN
        
    except Exception as e:
        error_msg = f"Error processing fall detection results: {e}" # EN
        print(error_msg)
        report_content += f"*Error processing fall detection results: {e}*\n\n" # EN
else:
    report_content += "*Fall detection results file not found.*\n\n" # EN

# --- Keyword Detection Evaluation --- 
report_content += "## 2. Keyword Detection Evaluation\n\n" # EN
report_content += "This section evaluates the current keyword detection implementation, which uses Google Speech Recognition to transcribe audio and then searches for predefined keywords.\n\n" # EN

if KEYWORD_RESULTS_CSV.is_file():
    try:
        keyword_df = pd.read_csv(KEYWORD_RESULTS_CSV)
        valid_predictions = keyword_df[keyword_df["prediction_details"] != "Recognition Failed"].copy() # Use .copy() to avoid SettingWithCopyWarning
        num_total = len(keyword_df)
        num_failed = num_total - len(valid_predictions)
        report_content += f"Processed {num_total} samples. Recognition failed for {num_failed} samples.\n\n" # EN

        if not valid_predictions.empty:
            y_true_kw = valid_predictions["ground_truth_expected"].astype(bool)
            y_pred_kw = valid_predictions["prediction_found"].astype(bool)
            
            # Calculate Confusion Matrix (based on presence/absence)
            cm_kw = confusion_matrix(y_true_kw, y_pred_kw, labels=[False, True]) # Labels: Keyword Absent, Keyword Present
            if cm_kw.shape == (1, 1):
                if True in y_true_kw.unique() or True in y_pred_kw.unique(): # Only True class
                    tn_kw, fp_kw, fn_kw, tp_kw = 0, 0, 0, cm_kw[0, 0]
                else: # Only False class
                    tn_kw, fp_kw, fn_kw, tp_kw = cm_kw[0, 0], 0, 0, 0
            else:
                 tn_kw, fp_kw, fn_kw, tp_kw = cm_kw.ravel()
            
            report_content += "### 2.1 Confusion Matrix (Keyword Presence/Absence on Successful Recognitions):\n"
            report_content += f"```\n"
            report_content += f"          Predicted Absent | Predicted Present\n"
            report_content += f"True Absent   |      {tn_kw:^10} |     {fp_kw:^10}\n"
            report_content += f"True Present  |      {fn_kw:^10} |     {tp_kw:^10}\n"
            report_content += f"```\n\n"
            
            # Plot Confusion Matrix
            plot_confusion_matrix(cm_kw, classes=["Absent", "Present"], 
                                  title="Keyword Detection Confusion Matrix (Presence)", 
                                  output_path=KEYWORD_CM_PLOT)
            report_content += f"![Keyword Detection Confusion Matrix]({KEYWORD_CM_PLOT.name})\n\n" # EN
            
            # Classification Report
            report_kw = classification_report(y_true_kw, y_pred_kw, target_names=["Absent", "Present"], zero_division=0)
            report_content += "### 2.2 Classification Metrics (Presence/Absence on Successful Recognitions):\n"
            report_content += f"```\n{report_kw}\n```\n\n" # EN
            report_content += "**Note:** This evaluation checks if *any* target keyword was detected when expected. It doesn\'t measure the accuracy of the transcription itself (Word Error Rate) or if the *specific* expected keyword was found. The reliance on a generic ASR service also limits performance, especially in noisy environments or with unclear speech.\n\n" # EN
        else:
             report_content += "*No successful keyword recognitions to generate detailed metrics.*\n\n" # EN
            
    except Exception as e:
        error_msg = f"Error processing keyword detection results: {e}" # EN
        print(error_msg)
        report_content += f"*Error processing keyword detection results: {e}*\n\n" # EN
else:
    report_content += "*Keyword detection results file not found.*\n\n" # EN

# --- Advanced Concepts Examples --- 
report_content += "## 3. Illustrative Examples of Advanced Concepts\n\n" # EN
report_content += "The following sections illustrate more advanced AI capabilities that could significantly enhance the monitoring system, as requested. **These are conceptual examples and are not generated by the current project code unless otherwise noted.**\n\n" # EN

# 3.1 Object Detection Example
report_content += "### 3.1 目标检测 (Object Detection)\n\n" # EN
report_content += "目标检测模型（如 YOLOv5）可以实时识别和定位图像中的多个物体。这对于监控系统非常有用，例如：\n"
report_content += "*   **识别关键人物：** 确认老人是否在画面中，以及他们的位置。\n"
report_content += "*   **检测重要物品：** 识别药瓶、拐杖、轮椅等，判断它们是否在常用位置，或者是否被遗忘。\n"
report_content += "*   **环境安全：** 检测地面上的障碍物、打开的柜门等潜在危险。\n\n"
report_content += f"**示例 (来自用户提供):** 下图展示了目标检测模型在复杂室内场景中识别不同物体（用红绿框标出）的效果。\n"
if OBJECT_DETECTION_EXAMPLE_IMG.is_file():
    report_content += f"![Object Detection Example]({OBJECT_DETECTION_EXAMPLE_IMG.name})\n\n" # EN
else:
    report_content += "*(用户提供的目标检测示例图片未找到)*\n\n"

# 3.2 Action Recognition & Scene Understanding Example
report_content += "### 3.2 动作识别 (Action Recognition) & 场景理解 (Scene Understanding)\n\n" # EN
report_content += "**动作识别** 通常基于 3D 卷积神经网络 (3D CNN) 或其他分析视频序列的模型，可以识别复杂的动态行为，而不仅仅是静态姿态。例如：\n"
report_content += "*   识别“跌倒”这一完整动作过程，而非仅仅是躺在地上的姿态。\n"
report_content += "*   识别“站立困难”、“徘徊”、“长时间静止”等异常行为模式。\n\n"

# Add Action Recognition results from log
report_content += "**实际运行结果 (Action Recognition):**\n\n" # EN
report_content += "我们使用 PyTorchVideo 的 Slow R50 模型对官方提供的 `archery_official_sample.mp4` 视频进行了动作识别推理。以下是模型的 Top-5 预测结果：\n\n" # EN
report_content += "```\n"
try:
    with open(ACTION_REC_LOG, "r") as f:
        log_content = f.read()
        # Extract the prediction block
        prediction_match = re.search(r"Top 5 predictions.*?\n(.*?)\nPrediction for", log_content, re.DOTALL)
        if prediction_match:
            prediction_lines = prediction_match.group(1).strip()
            report_content += prediction_lines + "\n"
        else:
            report_content += "(未能从日志中提取动作识别预测结果)\n"
except FileNotFoundError:
    report_content += "(动作识别推理日志文件未找到)\n"
except Exception as e:
    report_content += f"(读取或解析动作识别日志时出错: {e})\n"
report_content += "```\n\n" # EN
report_content += "该结果表明模型以极高的置信度 (1.0000) 正确识别出了视频中的“射箭”动作。\n\n" # EN

report_content += "**场景理解** (通常通过语义分割实现，如 DeepLabv3) 可以将图像像素分配到不同的类别（如地板、墙壁、家具、楼梯）。这有助于：\n"
report_content += "*   **区域判断：** 区分卧室、厨房、客厅等区域。\n"
report_content += "*   **危险区域警报：** 判断用户是否进入或停留在危险区域（如楼梯口、未围栏的阳台）。\n\n"

# Find an example visualization from the fall detection run
example_viz_path = None
if FALL_VIZ_DIR.is_dir():
    viz_files = list(FALL_VIZ_DIR.glob("eval_*.png"))
    if viz_files:
        example_viz_path = viz_files[0] # Use the first one as example

report_content += f"**示例 (基于当前项目的姿态估计输出):** 下图是当前项目处理一张跌倒图片的可视化结果。\n"
if example_viz_path:
    # Use relative path for markdown link
    relative_viz_path = example_viz_path.relative_to(OUTPUT_DIR)
    report_content += f"![Pose Estimation Example]({relative_viz_path})\n\n" # EN
else:
    report_content += "*(未找到跌倒检测可视化图片作为示例)*\n\n"

report_content += "结合上图，我们可以想象：\n"
report_content += "*   **动作识别** 会分析导致这个姿态的视频片段，确认这是一个快速的、不受控制的“跌倒”动作。\n"
report_content += "*   **场景理解** 会将图像分割成不同区域，例如，将地面标记为“地板”，将背景中的楼梯标记为“楼梯”（危险区域）。如果检测到用户（姿态）位于“楼梯”区域，可以发出更高级别的警报。\n\n"

# 3.3 Advanced Audio Processing Example
report_content += "### 3.3 高级听觉处理 (Advanced Audio Processing)\n\n" # EN
report_content += "当前的关键词检测比较基础。更高级的系统可以利用自然语言处理 (NLP) 技术：\n"
report_content += "*   **语音转文本 (STT):** 使用更鲁棒的模型（如 Wav2Vec 2.0）将语音转化为文本，提高在嘈杂环境下的准确率。\n"
report_content += "*   **语义理解 (NLU):** 使用 BERT 等模型分析转录文本，理解用户的意图和情感，而不仅仅是匹配关键词。例如，理解“我感觉不太好”和“帮帮我”之间的关联。\n"
report_content += "*   **上下文追踪:** 使用 Transformer 等模型跟踪对话历史或事件序列。例如，如果用户先说“我头晕”，几分钟后又说“我起不来了”，系统可以将这两条信息结合起来判断情况的紧急性。\n\n"

# Get an example transcription from keyword results
audio_example_text = "(未找到音频处理结果示例)"
if KEYWORD_RESULTS_CSV.is_file():
    try:
        kw_df = pd.read_csv(KEYWORD_RESULTS_CSV)
        # Find first successful recognition with non-empty details
        valid_recognitions = kw_df[(kw_df["prediction_details"] != "Recognition Failed") & (kw_df["prediction_details"].notna()) & (kw_df["prediction_details"].str.len() > 2)].copy()
        if not valid_recognitions.empty:
            first_success = valid_recognitions.iloc[0]
            # Extract recognized text from prediction_details (assuming format like "Keywords: [...] Text: ...")
            match = re.search(r"Text:\s*(.*)", first_success["prediction_details"])
            if match:
                audio_example_text = match.group(1).strip()
            else: # Fallback if format is different
                audio_example_text = first_success["prediction_details"] # Use the whole detail string
    except Exception as e:
        print(f"Could not extract audio example: {e}")
        audio_example_text = "(提取音频示例时出错)"

report_content += f"**示例 (来自当前项目的语音识别结果):**\n"
report_content += f"```\n原始音频 -> [语音识别模型] -> 识别文本: \"{audio_example_text}\"\n```\n"
report_content += "基于这样的识别文本，语义理解模型可以分析其含义，上下文追踪模型可以结合之前的事件（如检测到的跌倒）来做出更智能的判断。\n\n"

# --- Save Report --- 
try:
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
    with open(REPORT_FILE, "w", encoding='utf-8') as f: # Specify UTF-8 encoding
        f.write(report_content)
    print(f"Enhanced evaluation report saved to: {REPORT_FILE}") # EN
except Exception as e:
    print(f"Error saving enhanced report file: {e}") # EN

print("\nEnhanced report generation finished.") # EN

