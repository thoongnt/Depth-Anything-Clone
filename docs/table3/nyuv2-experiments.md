# Các thực nghiệm Depth Anything V1 có liên quan NYUv2

Bản PDF CVPR 2024 trong `paper/` là mốc đối chiếu. Các số dưới đây là kết quả
bài báo, không mặc định là số đã đo trong project.

## Thực nghiệm tiếp theo: in-domain metric depth (Table 3)

| Chỉ số | Giá trị paper | Ý nghĩa | Chiều tốt hơn |
|---|---:|---|---|
| δ₁ | 0.984 | Tỷ lệ pixel có max(pred/GT, GT/pred) < 1.25 | ↑ |
| δ₂ | 0.998 | Như trên, ngưỡng 1.25² | ↑ |
| δ₃ | 1.000 | Như trên, ngưỡng 1.25³ | ↑ |
| AbsRel | 0.056 | mean(abs(pred − GT) / GT) | ↓ |
| RMSE | 0.206 m | sqrt(mean((pred − GT)²)) | ↓ |
| log10 | 0.024 | mean(abs(log10(pred) − log10(GT))) | ↓ |

Tham chiếu: [metric_depth README chính thức](https://github.com/LiheYoung/Depth-Anything/tree/main/metric_depth),
[mã đánh giá chính thức](https://github.com/LiheYoung/Depth-Anything/blob/main/metric_depth/evaluate.py),
[script BTS chuẩn bị NYUv2](https://github.com/cleinc/bts/blob/master/utils/extract_official_train_test_set_from_mat.py).

### Chạy trên dữ liệu đã tải

```powershell
.\.venv\Scripts\python.exe evaluate_nyuv2_metric.py
```

Notebook: [thuc_nghiem_2_nyuv2_metric_depth.ipynb](../../thuc_nghiem_2_nyuv2_metric_depth.ipynb).

- Đầu vào: `data/nyuv2/nyu_depth_v2_labeled.mat` + `splits.mat`, đủ 654 official test IDs.
- Checkpoint: `checkpoints/depth_anything_metric_depth_indoor.pt`; encoder khởi tạo
  từ `checkpoints/depth_anything_vitl14.pth`, sau đó nạp toàn bộ metric checkpoint.
- Ground truth: **rawDepths**, lượng tử hóa `(rawDepths * 1000).astype(uint16) / 1000`
  đúng export PNG millimet của BTS; không dùng nhãn inpainted `depths`.
- Input RGB được chuẩn bị như BTS: vùng [7:474,7:632] giữ nguyên, viền còn lại
  đặt đen, OpenCV JPEG quality 95 rồi PIL decode RGB [0,1]. Các bước này thực hiện
  trong bộ nhớ, không cần xuất dataset lần nữa. Model core chính thức normalize
  ImageNet và resize 392×518 (keep_aspect_ratio=False theo cấu hình eval công bố).
- Flip augmentation theo mã metric chính thức, trung bình depth dự đoán trước
  khi nội suy bilinear về 480×640, `align_corners=True`.
- Mask **Eigen [45:471,41:601]**, 0.001 < GT < 10 m; prediction clip [0.001,10] m.
- Tính sáu metric bằng hàm chính thức `zoedepth.utils.misc.compute_metrics`.
  **Không căn chỉnh affine, không median scaling, không lấy nghịch đảo prediction.**
- Trung bình metric từng ảnh, không bỏ qua ảnh. Thiếu file, mask rỗng hoặc
  prediction không hữu hạn làm lượt chạy thất bại thay vì báo kết quả thành công.

Tùy chọn `--limit 4` để kiểm tra nhanh (không được coi là full benchmark),
`--device cpu`, `--no-flip`, `--output-dir`, `--cache-dir`, `--data-dir`,
`--checkpoint-dir`. Mặc định dùng CUDA nếu có. Output của subset và no-flip có tên riêng.

Kết quả lưu ở:

- `results/table3/nyuv2/indoor-flip-summary.json`: sáu chỉ số chưa làm tròn,
  mốc Table 3, độ lệch từng chỉ số, protocol, checksum, môi trường và số ảnh.
- `results/table3/nyuv2/indoor-flip-per-image.csv`: từng sample ID và sáu metric.
- `.cache/nyuv2-metric/`: prediction float32 để resume hoặc re-score.

Đây là **đánh giá checkpoint đã fine-tune**, không phải tái huấn luyện Table 3.
Kết quả measured được cập nhật sau khi chạy đủ 654 ảnh; không thay bằng số paper.

Đã kiểm chứng bước chuyển đổi trong bộ nhớ bằng cách đối chiếu bit-for-bit với
OpenCV ghi JPEG/PNG ra đĩa rồi PIL đọc lại, trên các fixture hồi quy.
Lượt chẩn đoán dùng `depths` trực tiếp được lưu riêng dưới tên `filled-mat-diagnostic-*`;
đó không phải benchmark cuối theo BTS. Việc đổi sang rawDepths và chuẩn bị ảnh BTS
có căn cứ từ mã nguồn chuẩn bị dữ liệu, không phải điều chỉnh để khớp số paper.

## Những bảng nào khác có NYUv2?

| Bảng | Thực nghiệm | Phần có thể làm với tài nguyên hiện tại |
|---|---|---|
| Table 2 | Zero-shot relative depth | Đã có evaluator, notebook và kết quả 654 ảnh |
| Table 3 | In-domain metric depth | Đã triển khai evaluator + notebook; kết quả đầy đủ lưu trong JSON/CSV |
| Table 6 | Train riêng từng labeled dataset rồi test NYUv2 | Chưa có checkpoint của từng mô hình BlendedMVS, DIML, HRWSI, IRS, MegaDepth, TartanAir và labeled-only |
| Table 9 | Ablation labeled/unlabeled, strong perturbation, semantic loss | Chỉ có model cuối; thiếu checkpoint các biến thể ablation |
| Table 10 | So sánh encoder MiDaS với Depth Anything sau metric fine-tune | Nhánh Ours dùng lại kết quả Table 3; baseline MiDaS của tác giả chưa có trong checkpoints |
| Table 11 | So sánh DINOv2 gốc với Depth Anything sau metric fine-tune | Nhánh Ours dùng lại kết quả Table 3; thiếu model metric đã fine-tune từ DINOv2 gốc |

Tham chiếu riêng cho NYUv2:

- Table 6: AbsRel/δ₁ lần lượt BlendedMVS 0.068/0.958, DIML 0.055/0.969,
  HRWSI 0.062/0.966, IRS 0.057/0.970, MegaDepth 0.071/0.953,
  TartanAir 0.061/0.964, all-labeled 0.053/0.971.
- Table 9: AbsRel qua bốn cấu hình là 0.053, 0.054, 0.048, 0.043.
- Table 10: MiDaS 0.077/0.951; Ours 0.056/0.984.
- Table 11: DINOv2 0.066/0.973; Ours 0.056/0.984.

Các mốc baseline/ablation này là **paper reference**, không phải số project đã chạy.
Không thể tạo các ablation bằng cách đổi cờ lúc inference trên cùng checkpoint cuối.

Để tự fine-tune lại, repo đã có `metric_depth/train_mono.py`, nhưng danh sách train
đòi **24.231 cặp RGB/depth** trong `nyudepthv2_train_files_with_gt.txt`. Dữ liệu hiện
có chỉ là labeled MAT 1.449 ảnh (795 train + 654 test); nó không thay thế tập train này.
Cần chuẩn bị dữ liệu train đúng danh sách và tài nguyên huấn luyện trước khi tái hiện
các baseline hoặc ablation cần huấn luyện. Lần triển khai này không tải/training các tập đó.

**Table 5 không test trên NYUv2:** NYUv2 là nguồn fine-tune của indoor checkpoint;
target là SUN RGB-D, iBims-1, HyperSim. Semantic segmentation cũng không có benchmark
NYUv2 trong bài này, mà dùng Cityscapes và ADE20K.

## Kết quả metric đã đo trên máy (654/654 ảnh)

| Chỉ số | Measured | Paper Table 3 | Measured − paper |
|---|---:|---:|---:|
| delta1 | 0.983554438 | 0.984 | -0.000445562 |
| delta2 | 0.998000214 | 0.998 | +0.000000214 |
| delta3 | 0.999610778 | 1.000 | -0.000389222 |
| abs_rel | 0.055979702 | 0.056 | -0.000020298 |
| rmse | 0.206142599 | 0.206 | +0.000142599 |
| log10 | 0.024386644 | 0.024 | +0.000386644 |

Đã xác nhận đủ 654 test ID theo đúng thứ tự official split; trung bình CSV khớp JSON.
Đã chạy inference với checkpoint công bố, không thực hiện lại fine-tuning.
