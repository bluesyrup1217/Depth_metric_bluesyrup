import cv2
import sys
sys.path.append("./")
# 窗口名称
window_name = "Image Coordinates"

def mouse_callback(event, x, y, flags, param):
    """鼠标点击事件回调函数"""
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"点击坐标: (x={x}, y={y})")
        # 在图像上显示坐标
        temp_img = param.copy()
        cv2.putText(temp_img, f"({x}, {y})", (x + 10, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(temp_img, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow(window_name, temp_img)

def main():
    # 检查命令行参数
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # 输入图像路径
        image_path = "output/20260107_23:49:10/distortioned_orin.png"
    
    # 读取图像
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"无法读取图像: {image_path}")
        print("\n用法:")
        print(f"  python {sys.argv[0]} 图像文件路径")
        print("  或直接运行程序并根据提示输入路径")
        return
    
    # 获取图像尺寸
    height, width = image.shape[:2]
    print(f"图像尺寸: 宽={width}, 高={height}")
    
    # 创建窗口并显示图像
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    
    # 设置鼠标回调函数
    cv2.setMouseCallback(window_name, mouse_callback, image)
    
    print("\n操作说明:")
    print("- 左键点击图像获取坐标")
    print("- 按ESC键退出程序")
    
    # 等待按键
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键
            break
    
    # 关闭窗口
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()