from face import *
from PIL import Image
import matplotlib.pyplot as plt


fs = FaceSystem()
# 预测人脸的例子
image = Image.open("./images/1.jpg")
result = fs.face_detect(image)
fs.show_face_boxes(image, result)

# # 打开摄像头进行识别
# fs.video_face_reg()

# 将人脸切割保存
fs.save_faces(image, result)

# 提取人脸特征
face1 = Image.open('./images/face_0.jpg')
feature1 = fs.get_face_feature(face1)
face2 = Image.open('./images/face_1.jpg')
feature2 = fs.get_face_feature(face2)
print(feature1.shape)
print(feature2.shape)

# 人脸特征对比
dist = fs.feature_compare(feature1, feature2)
dist2 = fs.feature_compare(feature1, feature1)
print(dist)
print(dist2)





