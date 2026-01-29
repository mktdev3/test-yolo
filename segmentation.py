from ultralytics import YOLO
import visualization

def process(source):
    # 2. モデルの読み込み
    model = YOLO('yolov8n-seg.pt')

    # 3. 推論の実行
    print(f"Processing: {source}")
    results = model.predict(source=source, device='mps')

    # 4. 結果の出力 (2種類)
    # 4-1. 枠だけの画像 (白背景)
    visualization.save_segmentation_masks(results, 'path_only_output.png')

    # 4-2. 元画像に色付けした画像
    visualization.save_colored_masks(results, 'colored_output.png')

if __name__ == '__main__':
    # Default for testing if run directly
    process('https://ultralytics.com/images/bus.jpg')
