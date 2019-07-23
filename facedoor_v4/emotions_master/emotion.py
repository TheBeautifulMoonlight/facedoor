import cv2
import dlib
import numpy as np
from keras.models import load_model
from statistics import mode


class Emotion_detection(object):
    def __init__(self, path = ''):
        self.x, self.y, self.w, self.h = (0, 0, 0, 0)
        self.color = []
        self.emotion_mode = 'none'
        # 情绪标签
        self.emotion_labels = {
            0: 'angry',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'sad',
            5: 'surprise',
            6: 'neutral'
        }
        self.frame_window = 3  # 调控情绪更新反应
        self.emotion_window = []
        self.emotion_offsets = (20, 40)  # 表情检测块偏差
        self.emotion_classifier = load_model(path+
            'emotion_model.hdf5')  # keemotions_masterras载入情绪分类模型
        self.emotion_target_size = self.emotion_classifier.input_shape[
            1:3]  # 表情分类器表情检测大小 64*64

        self.detector = dlib.get_frontal_face_detector()

    # 应用偏差
    def apply_offsets(self, face_coordinates, offsets):

        x, y, width, height = face_coordinates
        x_off, y_off = offsets
        return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

    def preprocess_input(self, x, v2=True):
        x = x.astype('float32')
        x = x / 255.0
        if v2:
            x = x - 0.5
            x = x * 2.0
        return x

    def detecter(self, frame):
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector(gray_image, 1)
        for face in faces:
            face_coordinates = face.left(), face.top(), face.right()-face.left(), face.bottom()-face.top()
            x1, x2, y1, y2 = self.apply_offsets(
                face_coordinates, self.emotion_offsets)  # 扩大人脸区域偏差
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (self.emotion_target_size))
            except:
                continue

            # 灰度图像预处理
            gray_face = self.preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)

            emotion_prediction = self.emotion_classifier.predict(
                gray_face)  # 对图像进行分类预测
            # print(emotion_prediction)
            # 混淆矩阵的处理
            emotion_probability = np.max(emotion_prediction)
            # print(emotion_probability)
            emotion_label_arg = np.argmax(emotion_prediction)  # 获取索引（情绪）

            emotion_text = self.emotion_labels[emotion_label_arg]
            self.emotion_window.append(emotion_text)  # 添加入emotion_window列表

            if len(self.emotion_window
                   ) > self.frame_window:  # 如果获得的情绪列表大于10 ，就删掉第一个情绪，更新新情绪
                self.emotion_window.pop(0)
            try:
                self.emotion_mode = mode(self.emotion_window)  # 从离散的一组情绪里选取最多出现的一个
            except:
                continue

            # 根据情绪的可能性 取该情绪颜色的明暗，可能性低--》暗，转化为数组
            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)  # 将数组里的值取整数
            self.color = color.tolist()  # array转化为list
            self.x, self.y, self.w, self.h = face_coordinates

    def show(self):
        print(self.emotion_mode)
        cv2.rectangle(self.rgb_image, (self.x, self.y),
                      (self.x + self.w, self.y + self.h), self.color, 2)
        cv2.putText(self.rgb_image, self.emotion_mode, (self.x + 0, self.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.color, 1, cv2.LINE_AA)
        bgr_image = cv2.cvtColor(np.array(self.rgb_image), cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', bgr_image)


if __name__ == "__main__":
    e = Emotion_detection()
    cv2.namedWindow('window_frame')
    cap = cv2.VideoCapture(0)
    while True:
        hasFrame, frame = cap.read()
        e.detecter(frame)
        e.show()
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()