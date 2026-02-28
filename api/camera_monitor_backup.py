"""
摄像头实时监测模块 - USB摄像头完整修复版
用于控制USB摄像头拍照和保存结果
"""

import sys
import os
import time
import threading
import json
from datetime import datetime
import traceback

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ================ 检查 OpenCV ================
try:
    import cv2

    CV2_AVAILABLE = True
    print(f"✅ OpenCV 版本: {cv2.__version__}")
except ImportError as e:
    CV2_AVAILABLE = False
    print(f"❌ 无法导入 OpenCV: {e}")
    print("请运行: pip install opencv-python")

# ================ 检查模型 ================
try:
    from models.emotion_model import load_model, EmotionRecognitionModel

    MODEL_AVAILABLE = True
    print("✅ 成功导入表情识别模型")
except ImportError as e:
    MODEL_AVAILABLE = False
    print(f"❌ 导入表情识别模型失败: {e}")


    # 创建虚拟模型类
    class EmotionRecognitionModel:
        def __init__(self, *args, **kwargs):
            pass

        def eval(self):
            pass

        def to(self, device):
            return self


    def load_model(*args, **kwargs):
        return EmotionRecognitionModel()

# ================ 导入其他模块 ================
from PIL import Image
import torch
import torchvision.transforms as transforms


class CameraMonitor:
    """USB摄像头监测器"""

    def __init__(self, model_path=None, capture_interval=5, save_dir="data/monitor_results"):
        """
        初始化摄像头监测器

        Args:
            model_path: 模型文件路径
            capture_interval: 抓拍间隔（秒）
            save_dir: 保存目录
        """
        self.capture_interval = capture_interval
        self.save_dir = save_dir

        # 检查OpenCV可用性
        if not CV2_AVAILABLE:
            print("⚠️  OpenCV未安装，摄像头功能不可用")
            print("请运行: pip install opencv-python")
            self.camera_available = False
        else:
            self.camera_available = True

        # 监测状态
        self.is_monitoring = False
        self.is_paused = False
        self.total_captures = 0
        self.successful_analyses = 0
        self.camera = None
        self.monitor_thread = None

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "results"), exist_ok=True)

        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 监测器使用设备: {self.device}")

        # 尝试加载模型
        self.model = None
        if model_path and os.path.exists(model_path) and MODEL_AVAILABLE:
            try:
                self.model = load_model(model_path, model_name='resnet18', num_classes=7, device=self.device)
                print("✅ 监测模型加载成功")
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                self.model = None

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 情绪映射
        self.emotion_zh = {
            'anger': '愤怒',
            'disgust': '厌恶',
            'fear': '恐惧',
            'happy': '快乐',
            'sad': '悲伤',
            'surprised': '惊讶',
            'neutral': '平静'
        }

        # 情绪标签列表
        self.emotions = list(self.emotion_zh.keys())

        print(f"📂 监测数据保存到: {os.path.abspath(save_dir)}")

    def check_camera(self, camera_index=0):
        """
        检查摄像头连接

        Args:
            camera_index: 摄像头索引

        Returns:
            tuple: (success, message)
        """
        if not self.camera_available:
            return False, "OpenCV未安装"

        try:
            print(f"检查摄像头 {camera_index}...")

            # 尝试用DirectShow后端（Windows最稳定）
            cap = cv2.VideoCapture(camera_index + cv2.CAP_DSHOW)

            if not cap.isOpened():
                return False, f"摄像头 {camera_index} 无法打开"

            # 测试读取一帧
            ret, frame = cap.read()
            cap.release()

            if ret:
                return True, f"摄像头 {camera_index} 可用"
            else:
                return False, f"摄像头 {camera_index} 无法读取图像"

        except Exception as e:
            return False, f"摄像头检查失败: {str(e)}"

    def start_monitoring(self, camera_index=0):
        """
        开始监测

        Args:
            camera_index: 摄像头索引

        Returns:
            bool: 是否成功启动
        """
        if not self.camera_available:
            print("❌ OpenCV未安装，无法使用摄像头")
            return False

        if self.is_monitoring:
            print("⚠️  监测已在运行中")
            return True

        try:
            # 检查摄像头连接
            success, message = self.check_camera(camera_index)
            if not success:
                print(f"❌ {message}")
                return False

            print(f"🚀 正在打开摄像头 {camera_index}...")

            # 尝试用DirectShow后端（Windows）
            self.camera = cv2.VideoCapture(camera_index + cv2.CAP_DSHOW)

            if not self.camera.isOpened():
                print(f"❌ 无法打开摄像头 {camera_index}")
                return False

            # 设置摄像头参数
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 15)

            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.camera.get(cv2.CAP_PROP_FPS)

            print(f"✅ 摄像头 {camera_index} 已打开")
            print(f"📊 分辨率: {width}x{height}, FPS: {fps:.1f}")
            print(f"📸 抓拍间隔: {self.capture_interval}秒")

            self.is_monitoring = True
            self.is_paused = False

            # 启动监测线程
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(camera_index,),
                daemon=True
            )
            self.monitor_thread.start()

            print("🎬 监测线程已启动")
            return True

        except Exception as e:
            print(f"❌ 启动监测失败: {e}")
            traceback.print_exc()
            if self.camera:
                self.camera.release()
                self.camera = None
            return False

    def _monitoring_loop(self, camera_index):
        """监测循环"""
        print(f"🔍 监测循环开始 (摄像头: {camera_index}, 间隔: {self.capture_interval}s)")

        while self.is_monitoring:
            try:
                current_time = time.time()

                # 检查暂停状态
                if self.is_paused:
                    time.sleep(0.5)
                    continue

                # 读取摄像头帧
                ret, frame = self.camera.read()
                if not ret:
                    print("⚠️  读取摄像头帧失败")
                    time.sleep(1)
                    continue

                # 保存图像文件
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                image_filename = f"capture_{timestamp}.jpg"
                image_path = os.path.join(self.save_dir, "images", image_filename)

                # 保存图像
                try:
                    # 调整图像质量
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                    success = cv2.imwrite(image_path, frame, encode_param)

                    if success:
                        self.total_captures += 1
                        print(f"📸 保存第 {self.total_captures} 张图片: {image_filename}")

                        # 分析图像
                        result = self._analyze_frame(frame, timestamp, image_filename)

                        if result:
                            self.successful_analyses += 1
                            self._save_result(result, timestamp, image_filename)

                            emotion = result.get('emotion_zh', '未知')
                            confidence = result.get('confidence', 0)
                            print(f"✅ 分析完成: {emotion} ({confidence:.1%})")

                    else:
                        print(f"❌ 保存图片失败: {image_path}")

                except Exception as e:
                    print(f"❌ 保存图片异常: {e}")

                # 等待指定的间隔时间
                time.sleep(self.capture_interval)

            except Exception as e:
                print(f"❌ 监测循环错误: {e}")
                traceback.print_exc()
                time.sleep(1)

    def _analyze_frame(self, frame, timestamp, image_filename):
        """
        分析摄像头帧

        Args:
            frame: OpenCV图像帧
            timestamp: 时间戳
            image_filename: 图像文件名

        Returns:
            dict: 分析结果或None
        """
        if self.model is None:
            # 没有模型，生成模拟数据
            print("⚠️  使用模拟分析结果")
            return self._simulate_analysis(frame, timestamp, image_filename)

        try:
            # 转换OpenCV BGR到RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # 显示图像信息
            print(f"🔍 分析图像: {pil_image.size}像素, 模式: {pil_image.mode}")

            # 预处理
            img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            # 推理
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_idx].item()

            emotion = self.emotions[predicted_idx]

            # 构造概率字典
            prob_dict = {}
            for i, emotion_name in enumerate(self.emotions):
                prob_dict[emotion_name] = float(probabilities[i])

            result = {
                'timestamp': datetime.now().isoformat(),
                'emotion': emotion,
                'emotion_zh': self.emotion_zh.get(emotion, emotion),
                'confidence': float(confidence),
                'probabilities': prob_dict,
                'image_filename': image_filename,
                'image_path': f"images/{image_filename}"
            }

            return result

        except Exception as e:
            print(f"❌ 分析帧失败: {e}")
            traceback.print_exc()
            return None

    def _simulate_analysis(self, frame, timestamp, image_filename):
        """模拟情绪分析（用于测试）"""
        try:
            import random

            # 模拟随机情绪
            emotions = self.emotions
            main_emotion = random.choice(emotions)
            confidence = random.uniform(0.6, 0.9)

            # 生成模拟的概率分布
            probabilities = {}
            for emotion in emotions:
                if emotion == main_emotion:
                    probabilities[emotion] = confidence
                else:
                    probabilities[emotion] = (1 - confidence) / (len(emotions) - 1)

            # 构建结果
            result = {
                'timestamp': datetime.now().isoformat(),
                'emotion': main_emotion,
                'emotion_zh': self.emotion_zh.get(main_emotion, main_emotion),
                'confidence': float(confidence),
                'probabilities': probabilities,
                'image_filename': image_filename,
                'image_path': f"images/{image_filename}"
            }

            return result

        except Exception as e:
            print(f"❌ 模拟分析失败: {e}")
            return None

    def _save_result(self, result, timestamp, image_filename):
        """
        保存分析结果

        Args:
            result: 分析结果字典
            timestamp: 时间戳
            image_filename: 图像文件名
        """
        try:
            # 添加图像文件信息
            result['image_filename'] = image_filename
            result['image_path'] = f"images/{image_filename}"

            # 保存结果到JSON文件
            result_filename = f"result_{timestamp}.json"
            result_path = os.path.join(self.save_dir, "results", result_filename)

            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"💾 结果已保存: {result_path}")

        except Exception as e:
            print(f"❌ 保存结果失败: {e}")

    def pause_monitoring(self):
        """暂停监测"""
        if self.is_monitoring and not self.is_paused:
            self.is_paused = True
            print("⏸️ 监测已暂停")
            return True
        return False

    def resume_monitoring(self):
        """继续监测"""
        if self.is_monitoring and self.is_paused:
            self.is_paused = False
            print("▶️ 监测已继续")
            return True
        return False

    def stop_monitoring(self):
        """停止监测"""
        if self.is_monitoring:
            print("🛑 正在停止监测...")
            self.is_monitoring = False
            self.is_paused = False

            # 等待线程结束
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2)
                print("✅ 监测线程已停止")

            # 关闭摄像头
            if self.camera:
                self.camera.release()
                self.camera = None
                print("✅ 摄像头已释放")

            print(f"📊 统计: 共抓拍 {self.total_captures} 张，成功分析 {self.successful_analyses} 张")
            return True
        return False

    def get_status(self):
        """
        获取监测状态

        Returns:
            dict: 状态信息
        """
        return {
            'is_monitoring': self.is_monitoring,
            'is_paused': self.is_paused,
            'total_captures': self.total_captures,
            'successful_analyses': self.successful_analyses,
            'capture_interval': self.capture_interval,
            'save_dir': os.path.abspath(self.save_dir),
            'model_loaded': self.model is not None,
            'camera_available': self.camera_available,
            'camera_opened': self.camera is not None and hasattr(self.camera, 'isOpened') and self.camera.isOpened()
        }

    def analyze_history(self, days=None):
        """
        分析历史数据

        Args:
            days: 分析最近多少天的数据，None表示所有数据

        Returns:
            dict: 综合分析结果
        """
        results_dir = os.path.join(self.save_dir, "results")

        if not os.path.exists(results_dir):
            return {
                'success': False,
                'error': '没有找到历史数据目录',
                'total_results': 0,
                'results_dir': results_dir
            }

        try:
            # 收集所有结果文件
            result_files = []
            for filename in os.listdir(results_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(results_dir, filename)
                    result_files.append(filepath)

            if not result_files:
                return {
                    'success': False,
                    'error': '没有分析结果文件',
                    'total_results': 0
                }

            print(f"📊 分析 {len(result_files)} 个结果文件...")

            # 读取结果
            all_results = []
            for filepath in result_files:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        all_results.append(result)
                except Exception as e:
                    print(f"❌ 读取文件失败 {filepath}: {e}")
                    continue

            if not all_results:
                return {
                    'success': False,
                    'error': '无法读取结果文件',
                    'total_results': 0
                }

            # 按时间筛选
            if days is not None:
                cutoff_time = time.time() - (days * 24 * 3600)
                filtered_results = []
                for result in all_results:
                    try:
                        result_time = datetime.fromisoformat(result['timestamp']).timestamp()
                        if result_time >= cutoff_time:
                            filtered_results.append(result)
                    except:
                        continue
                all_results = filtered_results

            # 进行综合分析
            analysis = self._comprehensive_analysis(all_results)

            return {
                'success': True,
                'total_results': len(all_results),
                'analysis': analysis,
                'summary': self._generate_summary(all_results)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'total_results': 0
            }

    def _comprehensive_analysis(self, results):
        """综合分析"""
        if not results:
            return {}

        # 统计情绪频率
        emotion_counts = {}
        emotion_confidences = {}

        for result in results:
            emotion = result.get('emotion', 'unknown')
            confidence = result.get('confidence', 0)

            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
                emotion_confidences[emotion] = []

            emotion_counts[emotion] += 1
            emotion_confidences[emotion].append(confidence)

        # 计算平均置信度
        avg_confidences = {}
        for emotion, conf_list in emotion_confidences.items():
            if conf_list:
                avg_confidences[emotion] = sum(conf_list) / len(conf_list)
            else:
                avg_confidences[emotion] = 0

        # 找到主要情绪
        if emotion_counts:
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
        else:
            dominant_emotion = ('unknown', 0)

        return {
            'emotion_distribution': emotion_counts,
            'average_confidences': avg_confidences,
            'dominant_emotion': {
                'emotion': dominant_emotion[0],
                'emotion_zh': self.emotion_zh.get(dominant_emotion[0], dominant_emotion[0]),
                'count': dominant_emotion[1],
                'percentage': (dominant_emotion[1] / len(results)) * 100 if results else 0
            },
            'total_samples': len(results)
        }

    def analyze_history_with_advice(self, days=None):
        """分析历史数据并生成健康建议"""
        results_dir = os.path.join(self.save_dir, "results")

        if not os.path.exists(results_dir):
            return {
                'success': False,
                'error': '没有历史数据目录',
                'total_results': 0
            }

        try:
            # 收集所有结果文件
            result_files = []
            for filename in os.listdir(results_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(results_dir, filename)
                    result_files.append(filepath)

            if not result_files:
                return {
                    'success': False,
                    'error': '没有分析结果',
                    'total_results': 0
                }

            print(f"📊 分析 {len(result_files)} 个结果文件...")

            # 读取结果
            all_results = []
            for filepath in result_files:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        all_results.append(result)
                except Exception as e:
                    print(f"❌ 读取文件失败 {filepath}: {e}")
                    continue

            if not all_results:
                return {
                    'success': False,
                    'error': '无法读取结果文件',
                    'total_results': 0
                }

            # 综合分析
            analysis = self._comprehensive_analysis(all_results)

            # 生成健康建议
            health_advice = self._generate_health_advice(analysis)

            # 生成总结报告
            summary = self._generate_summary(all_results, analysis, health_advice)

            return {
                'success': True,
                'total_results': len(all_results),
                'analysis': analysis,
                'health_advice': health_advice,
                'summary': summary,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'total_results': 0
            }

    def _generate_health_advice(self, analysis):
        """生成健康建议"""
        try:
            # 尝试导入健康建议模块
            try:
                from models.health_advisor import HealthAdvisor, EmotionResult, create_advice_from_probabilities
                advisor_available = True
            except ImportError:
                advisor_available = False
                print("⚠️  健康建议模块不可用")

            if not advisor_available:
                # 返回简单建议
                dominant_emotion = analysis.get('dominant_emotion', {})
                emotion = dominant_emotion.get('emotion', 'unknown')
                emotion_zh = self.emotion_zh.get(emotion, emotion)

                return {
                    'description': f'基于历史数据分析，您的主要情绪是{emotion_zh}',
                    'recommendations': [
                        '建议定期进行情绪记录',
                        '注意情绪变化趋势',
                        '保持健康的生活方式'
                    ],
                    'risk_level': 'low' if emotion in ['happy', 'surprised'] else 'medium'
                }

            # 使用健康建议模块
            emotion_distribution = analysis.get('emotion_distribution', {})
            total_samples = analysis.get('total_samples', 1)

            # 计算平均概率
            probabilities = {}
            for emotion, count in emotion_distribution.items():
                probabilities[emotion] = count / total_samples

            # 确保所有情绪都有概率
            for emotion in self.emotions:
                if emotion not in probabilities:
                    probabilities[emotion] = 0.0

            # 生成建议
            report = create_advice_from_probabilities(probabilities)

            # 提取建议信息
            health_advice = {
                'description': report['health_advice']['description'] if 'health_advice' in report else '情绪健康建议',
                'immediate_actions': report['health_advice'].get('immediate_actions', []),
                'daily_tips': report['health_advice'].get('daily_tips', []),
                'long_term_suggestions': report['health_advice'].get('long_term_suggestions', []),
                'risk_level': report['risk_assessment'].get('risk_level',
                                                            'unknown') if 'risk_assessment' in report else 'unknown'
            }

            return health_advice

        except Exception as e:
            print(f"❌ 生成健康建议失败: {e}")
            return {
                'description': '情绪分析报告',
                'recommendations': ['保持积极心态', '注意情绪管理'],
                'risk_level': 'unknown'
            }

    def _generate_summary(self, all_results, analysis, health_advice):
        """生成详细总结报告"""
        summary = f"📊 综合情绪分析报告\n"
        summary += f"📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += f"📈 分析样本: {len(all_results)} 条数据\n\n"

        # 情绪分布
        summary += "🎭 情绪分布:\n"
        emotion_distribution = analysis.get('emotion_distribution', {})
        for emotion, count in sorted(emotion_distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(all_results)) * 100
            emotion_name = self.emotion_zh.get(emotion, emotion)
            summary += f"  {emotion_name}: {count}次 ({percentage:.1f}%)\n"

        # 主要情绪
        dominant = analysis.get('dominant_emotion', {})
        if dominant:
            summary += f"\n👑 主要情绪: {dominant.get('emotion_zh', dominant.get('emotion', '未知'))}\n"
            summary += f"   出现次数: {dominant.get('count', 0)}\n"
            summary += f"   占比: {dominant.get('percentage', 0):.1f}%\n"

        # 稳定性分析
        stability = analysis.get('stability_score', 0)
        summary += f"\n📊 情绪稳定性: {stability:.1f}%\n"
        if stability > 70:
            summary += "   ✅ 情绪较为稳定\n"
        elif stability > 40:
            summary += "   ⚠️  情绪有一定波动\n"
        else:
            summary += "   ⚠️  情绪波动较大\n"

        # 健康建议
        summary += f"\n💡 健康建议:\n"
        summary += f"   {health_advice.get('description', '暂无建议')}\n"

        if 'immediate_actions' in health_advice and health_advice['immediate_actions']:
            summary += "\n   🚨 立即行动:\n"
            for i, action in enumerate(health_advice['immediate_actions'][:3], 1):
                summary += f"     {i}. {action}\n"

        if 'daily_tips' in health_advice and health_advice['daily_tips']:
            summary += "\n   📅 日常贴士:\n"
            for i, tip in enumerate(health_advice['daily_tips'][:3], 1):
                summary += f"     {i}. {tip}\n"

        # 风险评估
        risk_level = health_advice.get('risk_level', 'unknown')
        risk_map = {
            'very_low': '🟢 风险极低',
            'low': '🟢 风险低',
            'medium': '🟡 风险中等',
            'high': '🟠 风险较高',
            'very_high': '🔴 风险很高'
        }
        summary += f"\n⚠️  风险评估: {risk_map.get(risk_level, '未知')}\n"

        return summary

    def get_recent_results(self, limit=10):
        """获取最近的结果"""
        results_dir = os.path.join(self.save_dir, "results")

        if not os.path.exists(results_dir):
            return []

        try:
            # 获取所有JSON文件并按时间排序
            result_files = []
            for filename in os.listdir(results_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(results_dir, filename)
                    mod_time = os.path.getmtime(filepath)
                    result_files.append((mod_time, filepath))

            # 按时间排序
            result_files.sort(reverse=True)

            # 读取最近的结果
            recent_results = []
            for _, filepath in result_files[:limit]:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        recent_results.append(result)
                except:
                    continue

            return recent_results
        except Exception as e:
            print(f"获取最近结果失败: {e}")
            return []


# 全局监测器实例
global_monitor = None


def get_monitor(model_path=None, save_dir=None):
    """获取全局监测器实例"""
    global global_monitor

    if global_monitor is None:
        if save_dir is None:
            # 使用默认保存目录
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            save_dir = os.path.join(base_dir, "data", "monitor_results")

        print(f"📁 初始化摄像头监测器...")
        print(f"📁 保存目录: {save_dir}")

        global_monitor = CameraMonitor(model_path=model_path, save_dir=save_dir)
        print("✅ 摄像头监测器初始化完成")

    return global_monitor


# 测试函数
if __name__ == "__main__":
    print("=" * 60)
    print("摄像头监测模块测试")
    print("=" * 60)

    # 创建监测器
    monitor = get_monitor()

    # 显示状态
    status = monitor.get_status()
    print(f"状态: {status}")

    # 测试功能
    print("\n测试功能:")
    print("1. 启动监测 (5秒)")
    print("2. 暂停/继续")
    print("3. 停止监测")
    print("4. 分析历史数据")

    choice = input("\n选择测试 (1-4): ")

    if choice == '1':
        if monitor.start_monitoring():
            print("监测已启动，等待5秒...")
            time.sleep(5)
            monitor.stop_monitoring()
    elif choice == '2':
        print("暂停/继续功能需要先启动监测")
    elif choice == '3':
        monitor.stop_monitoring()
    elif choice == '4':
        analysis = monitor.analyze_history(days=1)
        if analysis['success']:
            print(analysis['summary'])
        else:
            print(f"分析失败: {analysis['error']}")