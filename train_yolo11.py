import argparse
from pathlib import Path

# https://www.codewithgpu.com/u/iscyy

# sys.path.append('/root/ultralyticsPro/') # Path 以Autodl为例

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def main(opt):
    model = YOLO(opt.cfg)
    if opt.weights:
        model.load(opt.weights)

    model.info()

    model.train(
        data=opt.data,
        epochs=opt.epochs,
        imgsz=opt.imgsz,
        workers=opt.workers,
        batch=opt.batch,
        device=opt.device,
        seed=opt.seed,
        deterministic=opt.deterministic,
    )


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default=str(ROOT / "ultralytics/cfg_yolo11/YOLO11/yolo11-CS-Biformer-AFPN-SA.yaml"),
        help="model YAML path",
    )
    parser.add_argument("--data", type=str, default=str(ROOT / "data.yaml"), help="dataset YAML path")
    parser.add_argument("--weights", type=str, default="", help="optional pretrained checkpoint")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default=None, help="for example: 0, 0,1, cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.set_defaults(deterministic=True)

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
