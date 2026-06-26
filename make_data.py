import numpy as np
import cv2
import time
import os

# Label: 0000 là ko cầm tiền, còn lại là các mệnh giá
label = "200.000 VND"
save_dir = os.path.join('data', str(label))
warmup_frames = 60
max_captures = 2000

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cap = cv2.VideoCapture(0)
frame_count = 0
capture_count = 1000

print(f"Bắt đầu thu thập dữ liệu cho mệnh giá: {label}")

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Lỗi: Không thể đọc được hình ảnh từ camera.")
        break

    frame_count += 1
    frame = cv2.resize(frame, dsize=None,fx=0.3,fy=0.3)

    # Hiển thị
    cv2.imshow('frame',frame)

    # Chờ ổn định camera
    if frame_count <= warmup_frames:
        print(f"Đang chờ camera ổn định... Còn lại: {warmup_frames - frame_count} frames", end='\n')
    # Bắt đầu thu thập ảnh
    elif capture_count < max_captures:
        capture_count += 1
        file_path = os.path.join(save_dir, f"{capture_count}.png")
        cv2.imwrite(file_path, frame)
        print(f"Đang chụp: {capture_count}/{max_captures} ảnh", end='\n')
    # Đã chụp đủ
    else:
        print("Hoàn thành thu thập dữ liệu! Tự động đóng camera.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Đã dừng thủ công !!!")
        break

# Dọn dẹp tài nguyên phần cứng
cap.release()
cv2.destroyAllWindows()