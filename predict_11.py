import argparse
import os
from ultralytics import YOLO

def main(opt):
    # 加载自定义的权重文件路径
    model = YOLO(r'/root/autodl-tmp/YOLO11/re/改进权重和文件/best.pt')

    # 打印模型信息
    model.info()

    # 设置预测图片路径和保存路径
    save_path = r'/root/autodl-tmp/YOLO11/predictions'  # 指定保存预测结果的文件夹
    os.makedirs(save_path, exist_ok=True)  # 如果保存路径不存在，则创建它

    # 使用指定的路径进行预测，保存预测结果
    model.predict(
        r'/root/autodl-tmp/YOLO11/1.jpg',  # 测试图片路径
        save=True,  # 保存预测图像
        imgsz=640,  # 输入图像大小
        conf=0.5,  # 置信度阈值
        save_txt=True,  # 保存检测结果文本
        save_conf=True,  # 保存置信度
        project=save_path,  # 指定保存结果的路径
        name='改进结果'  # 保存的文件夹名
    )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # 删除不必要的--cfg参数，因为不再使用默认权重
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)