# Nhận diện biển báo giao thông

Tải các file pre-train mode [tại đây](https://drive.google.com/drive/folders/10Ah7g_cmHsk7TqbWVuFZifQ4s4PcRbS2?usp=sharing)
## I. Nhận diện sử dụng Convolution Neural Network
**Yêu cầu:** Sử dụng python3
1. Để train lại model ta dùng lệnh `python3 keras_main.py`.
2. Chạy demo model tay dùng lệnh `python3 keras_cnn_gui.py`
## II. CNN kết hợp SVM 
Mô tả: từ model CNN ta chỉ lấy các vector đặc trưng sau đó cho qua lớp SVM để phân loại
1. Train lại model CNN ta dùng lệnh `python3 keras_cnn_svm/keras_cnn_train.py`
2. Train model svm ta dùng lệnh `python3 keras_cnn_svm/cnn_svm_train.py`
3. Chạy thử model ta dùng lệnh `python3 keras_cnn_svm/cnn_svm_gui.py`