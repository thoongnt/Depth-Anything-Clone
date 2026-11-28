# Tái hiện Table 4 — in-domain metric depth trên KITTI (Depth Anything V1)

Chạy ngày 2026-11-28. **Trạng thái: HOÀN THÀNH — KHỚP cả 6 chỉ số.**

Mục tiêu là hàng **Ours** của Table 4 bản CVPR 2024. Đây là phép **đánh giá checkpoint
đã fine-tune** `depth_anything_metric_depth_outdoor.pt` trên split Eigen test của
KITTI. Phép đánh giá này không huấn luyện lại model.

## Kết quả

Ngưỡng giống [Table 3](table3-reproduction.md) và [Table 5](table5-reproduction.md):
kết luận dựa trên AbsRel và δ₁ ở 3 chữ số. KHỚP khi d = 0, ĐẠT khi d ≤ 0.005, còn
lại KHÔNG ĐẠT. Cột "Khớp" cho biết bao nhiêu trong 6 chỉ số trùng paper.

| Lượt | δ₁ ↑ | δ₂ ↑ | δ₃ ↑ | AbsRel ↓ | RMSE ↓ (m) | RMSE log ↓ | Khớp | Kết luận |
|---|---:|---:|---:|---:|---:|---:|---|---|
| **Paper, Ours** | 0.982 | 0.998 | 1.000 | 0.046 | 1.896 | 0.069 | | |
| `evaluate.py -d kitti` (652 ảnh) | 0.982 | 0.998 | 1.000 | 0.046 | 1.896 | 0.069 | **6/6** | **KHỚP** |
| Paper, ZoeDepth\* (tham chiếu) | 0.971 | 0.996 | 0.999 | 0.054 | 2.281 | 0.082 | | |

Khớp ngay ở lần chạy đầu, không phải chỉnh gì so với lệnh chính thức. Log đầy đủ:
`results/table4/kitti.log`.

## Quét cấu hình một yếu tố (giống Table 5)

Lượt quét dùng chung inference với `evaluate.py`; `base` trùng 6/6 với `evaluate.py`.

| Cấu hình | δ₁ | δ₂ | δ₃ | AbsRel | RMSE | RMSE log | Khớp | Kết luận |
|---|---:|---:|---:|---:|---:|---:|---|---|
| base ★: KB crop, garg crop, 0.001–80 m, flip-TTA | 0.982 | 0.998 | 1.000 | 0.046 | 1.896 | 0.069 | 6/6 | KHỚP |
| `min_depth_eval = 1 m` | 0.982 | 0.998 | 1.000 | 0.046 | 1.896 | 0.069 | 6/6 | KHỚP |
| không flip-TTA | 0.982 | 0.998 | 0.999 | 0.047 | 1.922 | 0.070 | 2/6 | ĐẠT |
| `max_depth_eval = 50` | 0.985 | 0.998 | 1.000 | 0.045 | 1.403 | 0.065 | 2/6 | ĐẠT |
| bỏ garg crop | 0.975 | 0.997 | 0.999 | 0.051 | 2.260 | 0.078 | 0/6 | KHÔNG ĐẠT |
| eigen crop thay cho garg crop | 0.974 | 0.997 | 0.999 | 0.053 | 2.368 | 0.080 | 0/6 | KHÔNG ĐẠT |

Yếu tố quyết định là **garg crop**. Flip-TTA giúp một chút, chủ yếu ở RMSE (1.922 → 1.896).
`min_depth_eval` không ảnh hưởng, vì GT KITTI không có điểm nào gần hơn 1 m trong vùng
crop. Cap 50 m cho số "đẹp" hơn nhưng là protocol khác; xem
[improvements.md](improvements.md), mục B.

## Protocol (evaluator chính thức)

```bash
cd metric_depth
../.venv/Scripts/python.exe evaluate.py -m zoedepth \
  --pretrained_resource="local::./checkpoints/depth_anything_metric_depth_outdoor.pt" -d kitti
```

Theo `zoedepth/utils/config.py` và `data_mono.py`:

- **Danh sách ảnh:** `train_test_inputs/kitti_eigen_test_files_with_gt.txt` có 697 dòng.
  652 ảnh có GT được chấm; 45 dòng `None` bị bỏ qua qua cờ `has_valid_depth`.
- **GT:** depth đã annotate của KITTI (`data_depth_annotated`, uint16 / 256), không
  phải LiDAR thô.
- **Input:** KB crop 352×1216, dùng flip-TTA, focal lấy từ file danh sách.
- **Chấm:** garg crop, 0.001 < GT < 80 m, prediction kẹp vào [0.001, 80],
  `compute_metrics`, trung bình theo ảnh.

## Dữ liệu và chuẩn bị

| Kiểm tra | Kết quả |
|---|---|
| Ảnh RGB `data/kitti/raw/<date>/<drive>/image_02/data/` | 697 / 697 |
| GT trong `data/kitti/gt/train` hoặc `data/kitti/gt/val` | 652 / 652 |
| Số drive | 28 |

`data_depth_annotated` chia các drive vào hai thư mục `train/` và `val/`, trong khi
loader chỉ nhận một thư mục gốc. Vì vậy hai thư mục được gộp bằng junction, không chép
dữ liệu và không cần quyền admin:

```powershell
New-Item -ItemType Directory -Force data\kitti\gt_merged | Out-Null
Get-ChildItem data\kitti\gt\train, data\kitti\gt\val -Directory |
  ForEach-Object { New-Item -ItemType Junction -Path "data\kitti\gt_merged\$($_.Name)" -Target $_.FullName | Out-Null }
```

Khối `"kitti"` trong `metric_depth/zoedepth/utils/config.py` được trỏ về dữ liệu local:

- `data_path_eval = <repo>/data/kitti/raw`
- `gt_path_eval = <repo>/data/kitti/gt_merged`

## Chạy lại

```bash
(cd metric_depth && ../.venv/Scripts/python.exe evaluate.py -m zoedepth \
   --pretrained_resource="local::./checkpoints/depth_anything_metric_depth_outdoor.pt" -d kitti) \
   > results/table4/kitti.log 2>&1                                    # khoảng 8 phút
.venv/Scripts/python.exe scripts/sweep_table5.py --dataset kitti --run kitti   # quét cấu hình, khoảng 9 phút
.venv/Scripts/python.exe scripts/collect_table4.py results/table4              # tổng hợp + kết luận
```

## Những điều đã lo ngại nhưng không xảy ra

| Rủi ro dự kiến | Thực tế |
|---|---|
| DataLoader `num_workers=1` lỗi pickle trên Windows | Chạy bình thường: loader KITTI không giữ `lambda`. |
| Lệch do antialias của torchvision, như DIODE | Không có: nhánh KITTI không gọi `transforms.Resize` trên tensor. |
| GT nằm ở hai thư mục | Đã xử lý bằng junction `gt_merged`. |

## Cấu trúc thư mục

```text
docs/table4-reproduction.md      tài liệu này
data/kitti/gt_merged/            junction tới gt/train/<drive> và gt/val/<drive>
results/table4/
  kitti.log                      log của evaluate.py (dòng cuối là dict metric)
  sweep/kitti.json, sweep-kitti.txt   quét cấu hình
  summary.md, summary.json       bảng so sánh và kết luận (collect_table4.py)
scripts/collect_table4.py        tổng hợp Table 4
scripts/sweep_table5.py          dùng chung cho Table 4 (--dataset kitti -> results/table4/sweep/)
```
