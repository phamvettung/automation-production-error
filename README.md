# Phân loại sản phẩm lỗi trong dây chuyền tự động ứng dụng phương pháp học máy.

### GIỚI THIỆU
Mã nguồn trên là demo cho bài toán phân lớp dựa vào hình ảnh, sử dụng phương pháp học máy k-NN, SVM. Nhằm giải quyết bài toán phát hiện sản phẩm lỗi trong sản xuất tự động.
### CÁC PHƯƠNG PHÁP SỬ DỤNG
***1. Thuật toán k-Nearest Neighbors và phương pháp xác thực chéo kết hợp lấy mẫu phân tầng.*** </br>
Các bước thực hiện:
- 1. Chia tập D mỗi lớp thành k phần bằng nhau và không giao nhau.
- 2. Thực hiện k lần chạy, mỗi lần chạy lấy 1 phần của tập D để test, còn lại dùng để train.
    + Huấn luyện mô hình sử dụng D train với 3 láng giềng gần nhât.
    + Đánh giá hiệu quả mô hình sử dụng D test và độ đo Accuracy
- 4. Lấy trung bình kết quả từ k lần chạy</br>
***2. Support Vector Machines và phương pháp lựa chọn mô hình sử dụng chiến lược Holdout.***
- 1
- 2
- 3
### CÁC CÔNG NGHỆ SỬ DỤNG

### HƯỚNG DẪN

