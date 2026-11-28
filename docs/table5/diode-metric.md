# DIODE · metric depth với checkpoint indoor và outdoor

> **Đã giải quyết (2026-11-28):** Khoảng lệch 0.799 / 0.289 của DIODE Outdoor so với paper đến từ `transforms.Resize`. Từ torchvision 0.17, Resize mặc định bật antialias; môi trường gốc dùng torchvision 0.14 nên không có. Đặt `antialias=False` thì ra đúng **0.794 / 0.288**. Xem [Table 5](../table5-reproduction.md).

Đánh giá đủ 771 ảnh native validation theo hai miền. Checkpoint indoor fine-tune trên NYUv2 và checkpoint outdoor fine-tune trên KITTI. Cả hai dự đoán độ sâu theo mét. Ground truth được dùng để tính metric, không dùng để căn chỉnh prediction. Suy luận trung bình ảnh gốc và flip ngang, sau đó nội suy bilinear về kích thước depth GT.

| Miền | Số ảnh | Crop, khoảng depth | Evaluator phát hành · AbsRel / δ₁ | Thêm mask cảm biến · AbsRel / δ₁ |
|---|---:|---|---:|---:|
| Outdoor | 446 | Garg, 0.001–80 m | 0.799441 / 0.289341 | 0.575534 / 0.290726 |
| Indoor | 325 | Eigen, 0.001–10 m | 0.324951 / 0.302634 | 0.312956 / 0.302688 |

Table 5 công bố **0.794 / 0.288** cho DIODE Outdoor. Kết quả outdoor theo evaluator phát hành chưa khớp đồng thời cả hai chỉ số ở ba chữ số thập phân. DIODE Indoor là phép thử bổ sung. [Table 5](https://arxiv.org/html/2401.10891#S4.T5)

Cột evaluator phát hành giữ nguyên cách lấy pixel từ crop và khoảng depth GT. Loader công bố đọc mask cảm biến nhưng trả về RGB và depth, nên scorer không nhận mask đó. Cột thứ hai áp thêm mask cảm biến DIODE vào cùng prediction và protocol crop/range để đo ảnh hưởng của khác biệt này. [Loader công bố](https://github.com/LiheYoung/Depth-Anything/blob/main/metric_depth/zoedepth/data/diode.py)

```powershell
.venv\Scripts\python.exe evaluate_diode_metric.py --domain both
```

- [Outdoor summary](../../results/table5/diode-metric/outdoor-summary.json) · [CSV 446 ảnh](../../results/table5/diode-metric/outdoor-per-image.csv)
- [Indoor summary](../../results/table5/diode-metric/indoor-summary.json) · [CSV 325 ảnh](../../results/table5/diode-metric/indoor-per-image.csv)
- [Đối chiếu số ảnh và CSV](../../results/table5/diode-metric/verification.json)
