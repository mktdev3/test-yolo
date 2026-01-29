import argparse
import segmentation

def main():
    # 1. 引数の設定
    parser = argparse.ArgumentParser(description='YOLOv8 Segmentation Test')
    parser.add_argument('--source', type=str, default='https://ultralytics.com/images/bus.jpg',
                        help='Path to image file or URL')
    args = parser.parse_args()

    # 2. 処理の実行
    segmentation.process(args.source)

if __name__ == '__main__':
    main()
