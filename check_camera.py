"""
检查USB摄像头连接状态
"""

import cv2
import sys
import os


def check_camera_connection():
    """检查摄像头连接"""
    print("=" * 60)
    print("📷 摄像头连接检测工具")
    print("=" * 60)

    # 尝试不同的摄像头索引
    print("尝试检测摄像头（0-4）...")

    available_cameras = []

    for i in range(5):
        try:
            # 尝试用DirectShow后端（Windows）
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    print(f"✅ 发现摄像头 {i}: {width}x{height}")

                    # 保存一张测试照片
                    test_dir = "data/camera_test"
                    os.makedirs(test_dir, exist_ok=True)
                    test_path = os.path.join(test_dir, f"camera_test_{i}.jpg")
                    cv2.imwrite(test_path, frame)
                    print(f"   测试图像已保存: {test_path}")

                    available_cameras.append(i)

                    # 显示预览（可选）
                    cv2.imshow(f'Camera {i} - 按任意键继续', frame)
                    cv2.waitKey(1000)
                    cv2.destroyAllWindows()
                else:
                    print(f"⚠️  摄像头 {i}: 已打开但无法读取")
            else:
                print(f"❌ 摄像头 {i}: 不可用")

            cap.release()

        except Exception as e:
            print(f"❌ 检测摄像头 {i} 时出错: {e}")
            continue

    return available_cameras


def main():
    """主函数"""
    print("正在检测USB摄像头...")

    cameras = check_camera_connection()

    print("\n" + "=" * 60)
    if cameras:
        print(f"✅ 检测到 {len(cameras)} 个摄像头:")
        for cam_idx in cameras:
            print(f"   摄像头索引: {cam_idx}")

        print("\n🎯 建议使用摄像头索引 0 或 1")
        print("在Web界面中，可以设置摄像头索引:")
        print("  摄像头索引: 0 (第一个摄像头)")
        print("  抓拍间隔: 5 (秒)")
    else:
        print("❌ 未检测到任何摄像头")
        print("\n⚠️ 可能的原因:")
        print("  1. 摄像头未连接或未通电")
        print("  2. 摄像头驱动程序未安装")
        print("  3. 摄像头被其他程序占用")
        print("  4. 需要管理员权限")

        print("\n💡 解决方案:")
        print("  1. 检查USB连接")
        print("  2. 重启摄像头")
        print("  3. 关闭其他使用摄像头的软件")
        print("  4. 以管理员身份运行此脚本")

    print("\n🔧 系统将使用虚拟模式继续运行")
    print("   虚拟模式下仍可测试其他功能")
    print("=" * 60)

    # 测试OpenCV功能
    print("\n📊 OpenCV信息:")
    print(f"  版本: {cv2.__version__}")

    try:
        # 尝试导入摄像头监测器
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from api.camera_monitor import CameraMonitor

        monitor = CameraMonitor()
        status = monitor.get_status()

        print("\n📊 摄像头监测器状态:")
        print(f"  模型加载: {'✅' if status.get('model_loaded') else '❌'}")
        print(f"  保存目录: {status.get('save_dir', '未知')}")
        print(f"  抓拍间隔: {status.get('capture_interval', 5)}秒")

        print("\n✅ 摄像头监测器初始化成功")

    except Exception as e:
        print(f"\n❌ 摄像头监测器初始化失败: {e}")

    print("\n按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()