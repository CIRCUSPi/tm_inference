import cv2
import numpy as np
import tensorflow as tf
import time
import argparse

def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

def preprocess_image(image, input_shape, is_quantized):
    image = cv2.resize(image, input_shape)
    image = np.expand_dims(image, axis=0)
    if is_quantized:
        image = np.uint8(image)
    else:
        image = image.astype('float32') / 255.0
    return image

def get_top_prediction(predictions, labels):
    return labels[np.argmax(predictions)]

def main(model_path, label_path, camera=0):
    # 載入模型
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    global input_details, output_details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    is_quantized = input_details[0]['dtype'] == np.uint8

    # 載入標籤
    labels = load_labels(label_path)

    input_shape = tuple(input_details[0]['shape'][1:3])  # 轉換為元組形式

    # 初始化Webcam
    cap = cv2.VideoCapture(camera)

    # 初始化FPS計算
    prev_time = 0
    fps_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 預處理影像並進行推論
        input_image = preprocess_image(frame, input_shape, is_quantized)
        interpreter.set_tensor(input_details[0]['index'], input_image)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        # 取得預測結果並顯示在影像上
        prediction_label = get_top_prediction(predictions[0], labels)
        cv2.putText(frame, f"Prediction: {prediction_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 計算FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_list.append(fps)

        # 在影像上顯示FPS
        avg_fps = np.mean(fps_list[-10:])  # 取最後10幀的平均FPS
        cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Classification', frame)

        # 按下"q"結束程式
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time classification using TensorFlow Lite model")
    parser.add_argument("--model", default="model.tflite", help="Path to the TensorFlow Lite model (.tflite)")
    parser.add_argument("--labels", default="labels.txt", help="Path to the labels file (labels.txt)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default is 0)")

    args = parser.parse_args()

    main(args.model, args.labels, camera=args.camera)

