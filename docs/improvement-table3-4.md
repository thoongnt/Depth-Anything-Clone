# Thực nghiệm cải tiến — Table 3 (NYUv2) và Table 4 (KITTI)

Bắt đầu ngày 2026-11-28. Mục tiêu: đạt kết quả **tốt hơn số trong paper** cho metric
depth in-domain, bằng cách cải tiến phương pháp. File này ghi lại **mọi** phương pháp
đã thử, kể cả những cách làm kết quả tệ đi, cùng các khó khăn gặp phải.

## Nguyên tắc

- **Protocol chấm giữ nguyên như paper.** Đây chính là protocol đã tái lập Table 3/4
  khớp 6/6:
  - NYUv2: BTS export, `rawDepths`, Eigen crop, 1e-3 – 10 m.
  - KITTI: Eigen test 652 ảnh, KB crop, garg crop, 1e-3 – 80 m.
  - Cả hai đều chấm bằng `compute_metrics` chính thức.
- **Chỉ thay đổi phương pháp inference** (loại A trong [improvements.md](improvements.md)).
  Không dùng GT để căn chỉnh, không đổi mask hay crop.
- **Không huấn luyện lại.** Lý do: GPU chỉ 4 GB, máy không có tập train NYU 24k ảnh,
  và tập train KITTI cũng không đầy đủ.
- **Chống "ăn may" khi chọn cấu hình:**
  - Sàng lọc trên **120 ảnh đầu**.
  - Chọn cấu hình theo kết quả sàng lọc.
  - Chạy **đủ tập** chỉ với các cấu hình đã chọn.
  - Báo cáo cả số sàng lọc lẫn số đủ tập. Cấu hình nào thắng chỉ nhờ được chọn
    trên chính tập test thì ghi rõ.

Script: `scripts/improve_metric.py`. Kết quả: `results/table3/improve/`,
`results/table4/improve/`.

## Mốc so sánh

| | δ₁ | δ₂ | δ₃ | AbsRel | RMSE | log10 / RMSE log |
|---|---:|---:|---:|---:|---:|---:|
| Table 3 paper (NYUv2) | 0.984 | 0.998 | 1.000 | 0.056 | 0.206 | 0.024 |
| Table 3 tái lập | 0.9836 | 0.9980 | 0.9996 | 0.0560 | 0.2061 | 0.0244 |
| Table 4 paper (KITTI) | 0.982 | 0.998 | 1.000 | 0.046 | 1.896 | 0.069 |
| Table 4 tái lập | 0.982 | 0.998 | 1.000 | 0.046 | 1.896 | 0.069 |

## Phân tích trước khi thử

Checkpoint metric resize **mọi** ảnh đầu vào về 392×518, **không giữ tỉ lệ**
(`img_size: [392, 518]`, `keep_aspect_ratio=False` khi eval):

- NYUv2 480×640 → 392×518: tỉ lệ gần như giữ nguyên (0.75 so với 0.76).
- KITTI 352×1216 (sau KB crop) → 392×518: **chiều ngang bị ép 2,35 lần** (tỉ lệ 3,45
  thành 1,32). Model được fine-tune với chính kiểu ép này.

Các hướng thử:

1. Đổi kích thước đầu vào: giữ tỉ lệ ép nhưng phóng to hoặc thu nhỏ; bớt ép; giữ tỉ
   lệ gốc.
2. Ensemble đa tỉ lệ, trung bình cộng hoặc trung bình nhân.
3. (sau sàng lọc) Ghép model relative độ phân giải cao với model metric.

## Nhật ký thực nghiệm

_Đang chạy vòng sàng lọc 120 ảnh; kết quả sẽ được ghi tiếp bên dưới._

## Khó khăn

| Khó khăn | Cách xử lý |
|---|---|
| Không có dữ liệu và GPU để fine-tune lại. | Chỉ cải tiến ở khâu inference. |
| Lúc inference, model resize ảnh bên trong `core.prep`. | Thay `prep.resizer` bằng `FixedResize` (bilinear, `align_corners=True`, giống bản gốc). Đã kiểm tra `default` và `392x518` cho kết quả trùng hệt. |
| Mỗi kích thước thêm vào làm thời gian inference tăng tuyến tính. | Mỗi kích thước chỉ chạy forward một lần; ensemble ghép lại từ các dự đoán đã có. |
