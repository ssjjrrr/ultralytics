import os
import cv2

def load_yolo_results(file_path):
    results = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            score = float(parts[5])
            results.append((class_id, x_center, y_center, width, height, score))
    return results

def draw_boxes(image, results, color=(0, 255, 0)):
    for result in results:
        class_id, x_center, y_center, width, height, score = result
        x1 = int((x_center - width / 2) * image.shape[1])
        y1 = int((y_center - height / 2) * image.shape[0])
        x2 = int((x_center + width / 2) * image.shape[1])
        y2 = int((y_center + height / 2) * image.shape[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"ID: {class_id} Score: {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def visualize_results(image_dir, results_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_file in os.listdir(image_dir):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):  # 根据需要调整图像文件格式
            image_name, _ = os.path.splitext(image_file)
            image_path = os.path.join(image_dir, image_file)
            results_path = os.path.join(results_dir, f"{image_name}.txt")

            if os.path.exists(results_path):
                image = cv2.imread(image_path)
                results = load_yolo_results(results_path)
                image_with_boxes = draw_boxes(image, results)
                output_image_path = os.path.join(output_dir, image_file)
                cv2.imwrite(output_image_path, image_with_boxes)

if __name__ == "__main__":

    image_dir = "/home/edge/work/datasets/PANDA_dataset/images/val"  # 图像文件的目录
    results_dir = "/home/edge/work/ultralytics/runs/detect/predict21/label_merged_results"  # 检测结果文件的目录
    output_dir = "/home/edge/work/ultralytics/runs/detect/predict21/results_visualized"  # 可视化结果的输出目录
    
    visualize_results(image_dir, results_dir, output_dir)
