# Thực nghiệm 1 trên ETH3D

Đánh giá zero-shot relative depth của Depth Anything V1 trên 454 ảnh DSLR thuộc 13 cảnh High-res Multi-view của ETH3D. ETH3D gọi danh sách cảnh có nhãn này là training split trong benchmark multi-view stereo. MiDaS sử dụng đủ 454 ảnh có nhãn khi đánh giá zero-shot. ViT-L trong Bảng 2 của Depth Anything đạt AbsRel **0.127** và δ₁ **0.882**. [ETH3D](https://www.eth3d.net/datasets) · [MiDaS](https://arxiv.org/abs/1907.01341) · [Bài báo Depth Anything](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Depth_Anything_Unleashing_the_Power_of_Large-Scale_Unlabeled_Data_CVPR_2024_paper.pdf)

## Dữ liệu

- `data/eth3d/multi_view_training_dslr_undistorted` có đủ 454 ảnh DSLR đã khử méo và tham số camera. `data/eth3d/multi_view_training_dslr_scan_eval` có point cloud laser của 13 cảnh.
- Gói MoGe `ETH3D.zip` chứa 454 cặp RGB và depth map đã xử lý ở 2048 × 1365. Tất cả khóa `cảnh/tên ảnh` khớp tập DSLR local. Script giải mã PNG depth 16-bit theo công thức MoGe, dùng các giá trị hữu hạn dương và xác nhận shape trùng RGB. [Gói ETH3D của MoGe](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation) · [Cách đọc depth](https://github.com/microsoft/MoGe/blob/main/moge/utils/io.py#L82-L102)
- Các archive chính thức `*_dslr_depth.7z` chứa depth float32 được render từ scan laser. Chúng dùng hệ tọa độ ảnh DSLR gốc có méo ống kính, nên phải ghép với `*_dslr_jpg.7z`. Lượt đối chứng lấy pixel tâm mỗi ô 3 × 3 của depth gốc và dùng RGB giảm kích thước bằng INTER_AREA theo cùng lưới 2016 × 1344. [Định dạng ETH3D](https://www.eth3d.net/documentation#format-of-multi-view-data)

## Giao thức

Cấu hình 518 px theo phụ lục giữ tỉ lệ ảnh, làm tròn mỗi cạnh về bội số 14, không crop và nội suy disparity lên lưới nhãn. Mỗi ảnh được fit scale và shift bằng bình phương tối thiểu trong disparity. AbsRel và δ₁ được tính trên disparity đã fit, với pixel disparity dự đoán không dương được tính là trượt δ₁. Tổng hợp chính lấy trung bình theo ảnh trên cả 454 ảnh.

Những lượt đối chứng đo ảnh hưởng của flip ngang, ngưỡng depth 50/80/200 m và cách lấy trung bình theo pixel hoặc theo cảnh. Các hàng indoor, outdoor và hàng đổi ngược sang depth chỉ dùng để chẩn đoán. Phụ lục và giải đáp của tác giả mô tả phép fit và không gian chấm. Nguồn nhãn ETH3D cụ thể và chính sách mask của Bảng 2 chưa được xác nhận. [Phụ lục](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Yang_Depth_Anything_Unleashing_CVPR_2024_supplemental.pdf) · [Giải đáp của tác giả](https://github.com/LiheYoung/Depth-Anything/issues/174)

## Kết quả trên 454 ảnh

Các hàng dưới dùng ViT-L, disparity và trung bình theo ảnh. Cả hai nguồn được chạy trên cùng danh sách 454 mã ảnh.

| Cấu hình | AbsRel ↓ | δ₁ ↑ |
|---|---:|---:|
| Bảng 2 của Depth Anything | 0,127 | 0,882 |
| MoGe, 518 px, OLS, không flip | 0,047477 | 0,979807 |
| MoGe, 518 px, OLS, trung bình ảnh gốc và flip | 0,047410 | 0,979867 |
| MoGe, 518 px, median–MAD, không flip | 0,051606 | 0,973329 |
| ETH3D chính thức stride 3, 518 px, OLS, không flip | 0,045752 | 0,981929 |
| ETH3D chính thức stride 3, 518 px, median–MAD, không flip | 0,049182 | 0,976639 |

- Với 518 px và OLS, đổi từ MoGe sang cặp RGB/depth chính thức đổi AbsRel khoảng −0,00172 và δ₁ khoảng +0,00212. Mask của MoGe có 835.266.362 pixel hợp lệ trên lưới 2048 × 1365. Mask chính thức stride 3 có 219.873.565 pixel hợp lệ trên lưới 2016 × 1344.
- Ngưỡng 50 m thay đổi AbsRel dưới 0,00009 và δ₁ dưới 0,00011 trong các lượt 518 px, OLS, trung bình theo ảnh. Ngưỡng 80 m và 200 m cho cùng kết quả disparity với cấu hình không giới hạn. Flip và cách lấy trung bình theo pixel hay theo cảnh cũng được lưu trong CSV.
- Riêng với MoGe 518 px OLS, 219 ảnh indoor đạt 0,039846 / 0,987832 và 235 ảnh outdoor đạt 0,054588 / 0,972328.
- Đánh giá độc lập dùng giao thức Marigold cũng báo 0,058 / 0,984 cho Depth Anything trên ETH3D. Các công bố có thể dùng giao thức ETH3D khác nhau. [GenPercept, Bảng 6](https://proceedings.iclr.cc/paper_files/paper/2025/file/13b8d8fb8d05369480c2c344f2ce3f25-Paper-Conference.pdf)

[CSV MoGe OLS](../../results/table2/eth3d/vitl-518-flip-comparison.csv) · [CSV MoGe median–MAD](../../results/table2/eth3d/vitl-518-median-mad-comparison.csv) · [CSV ETH3D chính thức OLS](../../results/table2/eth3d/official-stride3-vitl-518-comparison.csv) · [CSV ETH3D chính thức median–MAD](../../results/table2/eth3d/official-stride3-vitl-518-median-mad-comparison.csv)

### Đối chứng kích thước đầu vào

Trên cùng 26 ảnh (hai ảnh theo thứ tự tên ở mỗi cảnh, lấy ở giữa hai nửa), ETH3D chính thức stride 3, OLS và không flip đạt 0,058876 / 0,962165 ở 518 px và 0,066085 / 0,956424 ở 392 px.

## Chạy lại

Các lượt báo cáo dùng CUDA và checkpoint relative `checkpoints/depth_anything_vitl14.pth`. Lệnh tải dữ liệu chỉ cần chạy khi cache chưa có.

```powershell
New-Item -ItemType Directory -Force .cache/eth3d-relative
curl.exe -fL -o .cache/eth3d-relative/ETH3D.zip "https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation/resolve/main/ETH3D.zip"
.venv\Scripts\python.exe thuc_nghiem_1_eth3d_relative_depth.py --flip --cache-predictions
.venv\Scripts\python.exe prepare_eth3d_official.py
.venv\Scripts\python.exe thuc_nghiem_1_eth3d_relative_depth.py --source official --cache-predictions
```

Mã SHA-256 của gói MoGe đã tải là `a72592c238113e46e74cf29ca7df306a6c5c9d050de396cdd257d0ed2783ab10`. Các file *per-image.csv*, *comparison.csv* và *summary.json* nằm trong `results/table2/eth3d`. File JSON giữ SHA-256 của checkpoint và dữ liệu, cấu hình và số ảnh của từng hàng.

## Kiểm tra nhãn Pipes

Đã so sánh 14 depth map MoGe với depth map ETH3D chính thức. Dùng intrinsics trong gói MoGe và mô hình THIN_PRISM_FISHEYE của ETH3D để xác định tọa độ pixel tương ứng. Có 4.407.618 pixel hợp lệ chung và sai số tương đối trung vị của mỗi ảnh khoảng 0,0008–0,0061%. Với cùng dự đoán và cùng mask giao, AbsRel đổi từ 0,036202 sang 0,036215 khi thay nhãn MoGe bằng nhãn chính thức. Trên lưới đã khử méo, MoGe có 27,28 triệu pixel hợp lệ, còn phép lấy mẫu gần nhất từ nhãn chính thức có 4,41 triệu. [Kết quả theo ảnh](../../results/table2/eth3d/audit-pipes-ground-truth.json).
