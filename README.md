# Phân loại sản phẩm lỗi trong dây chuyền tự động ứng dụng học máy.

### GIỚI THIỆU
Mã nguồn trên là demo cho bài toán phân lớp dựa vào hình ảnh, sử dụng phương pháp học máy k-NN, SVM. Nhằm giải quyết bài toán phát hiện sản phẩm lỗi trong dây chuyền sản xuất tự động.
### CÁC PHƯƠNG PHÁP SỬ DỤNG
***1. Thuật toán k-Nearest Neighbors và phương pháp xác thực chéo kết hợp lấy mẫu phân tầng.***
Các bước thực hiện:
- B1. Chia tập D mỗi lớp thành k phần bằng nhau và không giao nhau. (chọn k = 5)
- B2. Thực hiện k lần chạy, mỗi lần chạy lấy 1 phần của tập D để test, còn lại dùng để train.
    + Huấn luyện trên tập D_train với 3 láng giềng gần nhất.
    + Đánh giá hiệu quả trên tập D_test với độ đo accuracy.
- B3. Lấy trung bình kết quả từ k lần chạy. </br>

Thuật toán chạy trên số lượng 1000 ảnh mỗi lớp (1600 ảnh để train, 400 ảnh để test). Kết quả sau 5 lần test cho trung bình ***accuracy = 0.91***</br>
![Automation production error](/assets/k-NN.PNG) </br>


***2. Support Vector Machines với đặc trưng HOG và phương pháp lựa chọn mô hình sử dụng chiến lược Holdout.***
- B1. Chia tập D thành 2 phần: D_train và T_valid
- B2. Chọn ra tập S chứa các giá trị C tiềm năng
- B3. Với mỗi giá trị C thuộc tập S, huấn luyện hệ thống cho tập D_train. Đo hiệu quả trên tập T_valid để lấy kết quả Pc
- B4. Chọn ra giá trị C tốt nhất tương ứng với Pc lớn nhất. <br>

Thuật toán chạy trên số lượng 1000 ảnh mỗi lớp (20 ảnh để train, 1980 ảnh để test). Kết quả test với ***C = 1*** cho ***accuracy = 1***</br>
![Automation production error](/assets/svm.PNG)</br>
Kết quả cho thấy SVM hoạt động tốt hơn so với k-NN.

### CÁC CÔNG NGHỆ SỬ DỤNG
- OpenCV
- Ngôn ngữ: C++
### HƯỚNG DẪN
Để chạy chương trình, cần liên kết tới thư viện OpenCV trong dự án.

Các bước thực hiện trong Visual Studio.

-B1: Click chuột phải vào dự án -> ***Properties*** -> ***VC++ Directories*** -> thêm 2 đường dẫn tới thư viện opencv vào ***Include Directories*** và ***Library Directories***. </br>
![Automation production error](/assets/add_opencv_lib.PNG)

-B2: ***Linker*** -> ***Input*** -> ***Additional Dependencies***. </br>
![Automation production error](/assets/linker_input_tutorial.PNG)
