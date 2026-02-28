"""
检查USB摄像头连接状态
"""

import cv2
import time
import sys
import os



def list_all_cameras():
    """列出所有可用的摄像头"""
    print("=" * 60)
    print("📷 摄像头检测工具")
    print("=" * 60)

    available_cameras = []

    # 测试多个摄像头索引
    print("🔍 扫描摄像头索引 (0-10)...")
    for i in range(11):  # 检查0-10
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Windows使用DirectShow

        if cap.isOpened():
            # 尝试读取一帧
            ret, frame = cap.read()
            if ret:
                # 获取摄像头信息
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                print(f"✅ 摄像头 {i}:")
                print(f"   分辨率: {width}x{height}")
                print(f"   FPS: {fps:.1f}")
                print(f"   帧读取: 成功")

                # 尝试获取摄像头名称（Windows）
                try:
                    backend_name = cap.getBackendName()
                    print(f"   后端: {backend_name}")
                except:
                    pass

                available_cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'backend': 'DSHOW'
                })

                # 可选：显示预览
                print(f"   按 's' 保存测试图像，'q' 继续扫描")
                cv2.imshow(f'Camera {i}', frame)

                key = cv2.waitKey(2000)  # 显示2秒
                if key == ord('s'):
                    cv2.imwrite(f'camera_test_{i}.jpg', frame)
                    print(f"   测试图像已保存: camera_test_{i}.jpg")
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    break

                cv2.destroyAllWindows()
            else:
                print(f"⚠️  摄像头 {i}: 已打开但无法读取帧")
            cap.release()
        else:
            print(f"❌ 摄像头 {i}: 不可用")

    return available_cameras


def check_usb_camera_specific():
    """专门检查USB摄像头"""
    print("\n" + "=" * 60)
    print("🔌 USB摄像头专用检测")
    print("=" * 60)

    import platform
    system = platform.system()

    if system == 'Windows':
        print("检测到Windows系统")
        print("常用USB摄像头后端:")
        print("1. CAP_DSHOW (DirectShow) - 推荐")
        print("2. CAP_MSMF (Media Foundation)")
        print("3. CAP_VFW (Video for Windows)")

        backends = {
            'DSHOW': cv2.CAP_DSHOW,
            'MSMF': cv2.CAP_MSMF,
            'VFW': cv2.CAP_VFW
        }

        for backend_name, backend_code in backends.items():
            print(f"\n尝试 {backend_name} 后端...")
            for i in [0, 1, 2]:  # 检查前3个索引
                cap = cv2.VideoCapture(i + backend_code)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        print(f"  ✅ {backend_name}: 摄像头 {i} 可用")
                        cap.release()
                        break
                cap.release()

    elif system == 'Linux':
        print("检测到Linux系统")
        print("常用USB摄像头设备路径:")
        print("  /dev/video0")
        print("  /dev/video1")

    elif system == 'Darwin':  # macOS
        print("检测到macOS系统")

    return None


def test_camera_monitor_usage():
    """测试摄像头监测器使用"""
    print("\n" + "=" * 60)
    print("🎯 测试摄像头监测器使用")
    print("=" * 60)

    try:
        # 尝试导入摄像头监测器
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from camera_monitor import CameraMonitor

        # 创建监测器
        monitor = CameraMonitor(model_path=None, capture_interval=2)

        print("✅ 摄像头监测器创建成功")
        print(f"默认保存目录: {monitor.save_dir}")

        # 测试摄像头
        print("\n测试摄像头连接...")
        for i in [0, 1, 2]:
            print(f"尝试摄像头 {i}...")
            success = monitor.start_monitoring(i)

            if success:
                print(f"  ✅ 摄像头 {i} 连接成功")

                # 获取状态
                status = monitor.get_status()
                print(f"  状态: {status}")

                # 等待几秒抓拍
                import time
                print(f"  抓拍测试 (等待5秒)...")
                time.sleep(5)

                # 停止监测
                monitor.stop_monitoring()

                # 检查保存的图像
                save_dir = monitor.save_dir
                images_dir = os.path.join(save_dir, "images")

                if os.path.exists(images_dir):
                    images = os.listdir(images_dir)
                    if images:
                        print(f"  ✅ 成功保存 {len(images)} 张图像")
                        for img in images[:3]:  # 显示前3个
                            print(f"    - {img}")
                    else:
                        print(f"  ⚠️  图像目录为空")
                break
            else:
                print(f"  ❌ 摄像头 {i} 连接失败")

        print("\n摄像头监测器配置:")
        print(f"  抓拍间隔: {monitor.capture_interval}秒")
        print(f"  保存目录: {monitor.save_dir}")
        print(f"  模型加载: {'✅ 已加载' if monitor.model else '❌ 未加载'}")

    except Exception as e:
        print(f"❌ 摄像头监测器测试失败: {e}")
        import traceback
        traceback.print_exc()


def check_opencv_info():
    """检查OpenCV信息"""
    print("\n" + "=" * 60)
    print("📊 OpenCV信息")
    print("=" * 60)

    print(f"OpenCV版本: {cv2.__version__}")
    print(f"构建信息:")

    # 获取构建信息
    build_info = cv2.getBuildInformation()

    # 查找摄像头相关模块
    camera_keywords = [
        'Video I/O',
        'DC1394',
        'FFMPEG',
        'V4L',
        'DSHOW',
        'MSMF',
        'AVFoundation'
    ]

    for keyword in camera_keywords:
        if keyword in build_info:
            lines = [line for line in build_info.split('\n') if keyword in line]
            if lines:
                print(f"  {keyword}:")
                for line in lines[:3]:  # 只显示前3行
                    print(f"    {line.strip()}")


def main():
    """主函数"""
    print("=" * 60)
    print("🔧 USB摄像头连接状态诊断工具")
    print("=" * 60)

    # 检查OpenCV信息
    check_opencv_info()

    # 列出所有摄像头
    cameras = list_all_cameras()

    if cameras:
        print(f"\n🎉 找到 {len(cameras)} 个摄像头:")
        for cam in cameras:
            print(f"  索引 {cam['index']}: {cam['width']}x{cam['height']} @ {cam['fps']:.1f}fps")
    else:
        print("\n❌ 未找到任何摄像头")
        print("可能的原因:")
        print("  1. 摄像头未连接或未通电")
        print("  2. 摄像头驱动未安装")
        print("  3. 摄像头被其他程序占用")

    # 检查USB摄像头特定设置
    check_usb_camera_specific()

    # 测试摄像头监测器
    test_camera_monitor_usage()

    print("\n" + "=" * 60)
    print("✅ 诊断完成")
    print("=" * 60)

    if cameras:
        print("\n🎯 建议:")
        print(f"  程序应使用摄像头索引: {cameras[0]['index']}")
        print("  在API调用中使用:")
        print(f"    camera_index: {cameras[0]['index']}")
        print("  或者前端设置:")
        print(f"    {{'camera_index': {cameras[0]['index']}}}")
    else:
        print("\n⚠️  警告:")
        print("  未检测到摄像头，程序将使用虚拟模式")


if __name__ == "__main__":
    main()