import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tkinter import *
from PIL import Image, ImageTk
# =====================================================
# DANH SÁCH CLASS
# =====================================================

class_name = ['0 VND', '10.000 VND', '100.000 VND', '20.000 VND', '200.000 VND', '50.000 VND', '500.000 VND']

# =====================================================
# TẠO MODEL
# =====================================================

def get_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    # Fully Connected
    x = Flatten(name='flatten')(model_vgg16_conv.output)
    x = Dense(512,activation='relu',name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(256,activation='relu',name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(7,activation='softmax',name='predictions')(x)

    # Compile
    my_model = Model(inputs=model_vgg16_conv.input,outputs=x)
    my_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return my_model

# =====================================================
# LOAD MODEL
# =====================================================

my_model = get_model()
my_model.load_weights("vggmodel_final.keras")

# =====================================================
# GIAO DIỆN
# =====================================================

window = Tk()
window.title('NHẬN DIỆN MỆNH GIÁ TIỀN VIỆT NAM')

# Full màn hình
window.state('zoomed')
# Background
window.configure(bg='#EAF4F4')

# =====================================================
# HEADER
# =====================================================

header = Frame(window,bg='#0B3954',height=130)
header.pack(fill='x')
header.pack_propagate(False)

# LEFT
left_frame = Frame(header,bg='#0B3954')
left_frame.pack(side='left',padx=20)

# CENTER
center_frame = Frame(header,bg='#0B3954')
center_frame.pack(side='left',expand=True)

# RIGHT
right_frame = Frame(header,bg='#0B3954')
right_frame.pack(side='right',padx=20)

# =====================================================
# LOGO TRƯỜNG
# =====================================================

img1 = Image.open('logotruong.png')
img1 = img1.resize((110, 110))
logo = ImageTk.PhotoImage(img1)
logotruong = Label(left_frame,image=logo,bg='#0B3954')
logotruong.pack(pady=10)

# =====================================================
# LOGO KHOA
# =====================================================

img2 = Image.open('logokhoa.png')
img2 = img2.resize((110, 110))
logok = ImageTk.PhotoImage(img2)
logokhoa = Label(right_frame,image=logok,bg='#0B3954')
logokhoa.pack(pady=10)

# =====================================================
# TITLE
# =====================================================

title1 = Label(center_frame,text='HỆ THỐNG NHẬN DIỆN MỆNH GIÁ TIỀN VIỆT NAM',font=('Segoe UI', 28, 'bold'),fg='white',bg='#0B3954')
title1.pack(pady=(15, 5))
title2 = Label(center_frame,text='MẠNG NƠ-RON NHÂN TẠO SỬ DỤNG VGG16',font=('Segoe UI', 16),fg='white',bg='#0B3954')
title2.pack()

# =====================================================
# THÔNG TIN
# =====================================================

info_frame = Frame(window,bg='#EAF4F4')
info_frame.pack(pady=20)

truong = Label(info_frame,text='ĐẠI HỌC CÔNG NGHIỆP HÀ NỘI',fg='#0B3954',bg='#EAF4F4',font=("Segoe UI", 20, 'bold'))
truong.pack(pady=5)

khoa = Label(info_frame,text='TRƯỜNG ĐIỆN - ĐIỆN TỬ',fg='#1D3557',bg='#EAF4F4',font=("Segoe UI", 18, 'bold'))
khoa.pack(pady=5)

bcda = Label(info_frame,text='BÁO CÁO ĐỒ ÁN MÔN HỌC',fg='#D62828',bg='#EAF4F4',font=("Segoe UI", 24, 'bold'))
bcda.pack(pady=5)

detai = Label(info_frame,text='NHẬN DIỆN MỆNH GIÁ TIỀN VIỆT NAM SỬ DỤNG VGG16',fg='#000000',bg='#EAF4F4',font=("Segoe UI", 18, 'bold'),wraplength=1000,justify='center')
detai.pack(pady=10)

gvhd = Label(info_frame,text='GIẢNG VIÊN HƯỚNG DẪN: TS. NGUYỄN THỊ THU',fg='#1D3557',bg='#EAF4F4',font=("Segoe UI", 16))
gvhd.pack(pady=5)

# =====================================================
# KẾT QUẢ NHẬN DIỆN (THÊM MỚI Ở ĐÂY)
# =====================================================

result_frame = Frame(window, bg='#EAF4F4')
result_frame.pack(pady=10)

result_label = Label(result_frame, text="Đang chờ nhận diện...", font=('Segoe UI', 24, 'bold'), fg='#D62828', bg='#EAF4F4')
result_label.pack()

# =====================================================
# CAMERA FRAME
# =====================================================

camera_frame = Frame(window,bg='white',bd=4,relief='ridge')
camera_frame.pack(pady=20)

panel = Label(camera_frame,bg='white')
panel.pack()

# =====================================================
# CAMERA
# =====================================================

cap = cv2.VideoCapture(0)

# =====================================================
# XỬ LÝ CAMERA
# =====================================================

def ctchinh():

    ret, image_org = cap.read()
    if ret:
        # Resize camera
        image_org = cv2.resize(image_org,dsize=None,fx=0.8,fy=0.8)

        # =================================================
        # TIỀN XỬ LÝ
        # =================================================

        image = image_org.copy()

        image = cv2.resize(image,dsize=(128, 128))
        image = image.astype('float') * 1./255
        image = np.expand_dims(image,axis=0)

        # =================================================
        # DỰ ĐOÁN
        # =================================================

        predict = my_model.predict(image)
        max_prob = np.max(predict[0])
        class_idx = np.argmax(predict[0])

        print("This picture is: ",class_name[np.argmax(predict[0])])
        print(np.max(predict[0]))

        # =================================================
        # HIỂN THỊ KẾT QUẢ
        # =================================================

        if (np.max(predict) >= 0.8) and (np.argmax(predict[0]) != 0):

            text_display = f"Mệnh giá: {class_name[class_idx]} - Độ chính xác: {max_prob * 100:.2f}%"
            result_label.config(text=text_display, fg='#2A9D8F')  # Đổi màu xanh lá khi nhận diện thành công
        else:
            # Trạng thái chờ khi không nhận diện được
            result_label.config(text="Đang chờ nhận diện...", fg='#D62828')

        # =================================================
        # HIỂN THỊ CAMERA
        # =================================================

        image = cv2.cvtColor(image_org,cv2.COLOR_BGR2RGB)
        image = ImageTk.PhotoImage(Image.fromarray(image))
        panel.img = image
        panel.config(image=image)
        panel.image = image

        # Lặp camera
        panel.after(10, ctchinh)

# =====================================================
# BẮT ĐẦU CAMERA
# =====================================================

ctchinh()

# =====================================================
# MAIN LOOP
# =====================================================

window.mainloop()
