# Tái hiện NYUv2 zero-shot relative depth — Depth Anything V1

Rà soát ngày 2026-11-26. Mục tiêu là **Table 2** của bản CVPR 2024, ViT-L,
654 ảnh official test split. Không dùng checkpoint metric indoor để so với bảng này.

## Kết quả đã kiểm chứng

| Thiết lập | Số ảnh | AbsRel ↓ | δ₁ ↑ |
|---|---:|---:|---:|
| Notebook cũ: filled depths + Eigen mask, score depth | 654 | 0.0527 | 0.9653 |
| Chỉ đổi score sang disparity, giữ filled depths + Eigen mask | 654 | 0.0533684551 | 0.9652895520 |
| Đã sửa: rawDepths + mask MiDaS, score disparity | 654 | 0.0436349930 | 0.9803277444 |
| Paper CVPR 2024, Table 2, ViT-L | — | 0.043 | 0.981 |

Sai khác còn lại: AbsRel +0.0006349930; δ₁ -0.0006722556 (0.06723 điểm phần trăm).
Kết quả rất gần nhưng **chưa trùng số paper ở ba chữ số thập phân**.
Không thay metric, chọn subset hoặc điều chỉnh tham số theo mốc paper.

## Nguyên nhân và nguồn tham chiếu

1. Notebook dùng `depths`, là nhãn đã inpaint. Mã đánh giá NYUv2 được tác giả
   MiDaS công bố dùng `rawDepths`, là điểm đo Kinect đã chiếu sang ảnh RGB nhưng
   chưa lấp lỗ; mask toàn ảnh `0 < depth < 10`, không Eigen crop.
   [Mã MiDaS](https://gist.github.com/ranftlr/a1c7a24ebb24ce0e2f2ace5bce917022),
   được liên kết trong [DPT EVALUATION.md](https://github.com/isl-org/DPT/blob/main/EVALUATION.md).
   [Tài liệu NYUv2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) phân biệt hai trường này.
2. Tác giả Depth Anything xác nhận fit scale/shift giữa raw prediction và 1/GT,
   sau đó **tính metric trong disparity space**, không đổi về depth và không
   min-max prediction về [0,1]. [Issue #174](https://github.com/LiheYoung/Depth-Anything/issues/174).
3. Input giữ nguyên toàn ảnh RGB; resize cạnh ngắn 518, giữ aspect ratio,
   hai chiều là bội của 14. Dự đoán nội suy về 480×640 trước alignment.
   [Phụ lục CVPR, mục 1](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Yang_Depth_Anything_Unleashing_CVPR_2024_supplemental.pdf).

Bộ đánh giá sử dụng float32 inference và float64 centered least squares;
cap disparity dự đoán ở 0.1, tương ứng depth cap 10 m của MiDaS.
AbsRel = mean(abs(pred_disp - gt_disp) / gt_disp).
δ₁ = mean(max(pred_disp/gt_disp, gt_disp/pred_disp) < 1.25).
Tổng hợp bằng trung bình metric từng ảnh theo mã MiDaS; không bỏ qua ảnh.

Paper không cung cấp toàn bộ mã đánh giá relative depth. Đây là bản tái hiện
kết hợp hướng dẫn tác giả với protocol dữ liệu MiDaS, không phải mã chính thức
của tác giả Depth Anything. Phần chênh lệch nhỏ còn lại chưa được xác định nguyên nhân;
không mặc định quy cho GPU hoặc sai số số học.

## Chạy lại

Chạy tại thư mục gốc repository, trong môi trường `.venv` đã cài:

```powershell
.\.venv\Scripts\python.exe thuc_nghiem_1_relative_depth.py --dataset nyuv2 --encoder vitl
```

Hoặc chạy trực tiếp evaluator để chọn vị trí output/cache:

```powershell
.\.venv\Scripts\python.exe evaluate_nyuv2.py --encoder vitl
```

`thuc_nghiem_1_relative_depth.py` mặc định chỉ chạy NYUv2. `--dataset all` giữ
các nhánh cũ cho bộ dữ liệu khác; các nhánh đó chưa được rà soát trong lần sửa này.
Notebook mục 4.1 và script dùng chung `evaluate_nyuv2.evaluate_nyuv2`.
Generator `create_notebook_1.py` cũng đã cập nhật cùng cell.

`--limit N` chỉ phục vụ debug, được ghi `full_test_split: false` và tên output riêng.
`--protocol legacy-filled-eigen` là đối chứng dùng filled depth + Eigen mask,
không phải protocol mặc định. `--flip` là thử nghiệm augmentation tùy chọn,
output riêng; kết quả đã báo cáo ở đây **không dùng flip**.

## Artifacts và kiểm tra

- `results/table2/nyuv2/vitl-midas-summary.json`: metric chưa làm tròn,
  số ảnh, protocol, checkpoint/split SHA-256, môi trường và độ lệch với paper.
- `results/table2/nyuv2/vitl-midas-per-image.csv`: đủ 654 sample ID khác nhau,
  mask pixel count, scale, shift và metric từng ảnh.
- `results/table2/nyuv2/legacy-notebook-nyuv2.json`: lưu output notebook cũ
  để không đánh mất mốc 0.0527 / 0.9653 khi sửa cell.
- `results/table2/nyuv2/vitl-filled-eigen-disparity-diagnostic-summary.json`:
  đối chứng chỉ đổi không gian metric trên cùng 654 dự đoán.
- `results/table2/nyuv2/raw-eigen-diagnostic.json`: đối chứng raw depth + Eigen
  mask cho 654 ảnh (0.0434173740 / 0.9803159290); không chọn làm mặc định.
- `results/table2/nyuv2/checkpoint-comparison.json`: hai bản checkpoint công bố
  có **407/407 tensor giống hệt**, dù hash file khác do serialization.
- `.cache/nyuv2-relative/`: dự đoán float32 khoảng 0.8 GB, có manifest theo
  checkpoint, input, model/transform source và flip. Chạy lại sử dụng cache này.
  Đổi `--cache-dir` sang thư mục mới nếu muốn thực hiện lại toàn bộ inference.

ViT-L inference đủ 654 ảnh mất khoảng 7 phút 41 giây trên RTX 3050 Laptop 4 GB;
re-score từ cache mất khoảng 20 giây. Số liệu thời gian chỉ là lần chạy trên máy này.

Checkpoint đang dùng là bản Hub chính thức, SHA-256
`6fda5199f0043e91e1c6117df00313db57c630d8ed0306070b43da1ecfffbd9d`.
Đã tải thêm bản `.pth` của tác giả vào `checkpoints/original-release/` để so sánh,
giữ nguyên checkpoint hiện có.

Kiểm tra hồi quy:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

6 kiểm tra bao gồm khôi phục affine chính xác, loại pixel invalid, phân biệt mask
MiDaS/Eigen, phát hiện prediction hằng/mask rỗng và bảo đảm AbsRel tính ở disparity.
