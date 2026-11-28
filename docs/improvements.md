# Ghi chép cải tiến — kết quả tốt hơn số của tác giả

Bắt đầu ngày 2026-11-28. File này ghi lại mọi cấu hình cho kết quả **tốt hơn số
trong paper**, gặp được trong quá trình tái lập (Table 2, 3, 4, 5). Mục đích là làm
điểm xuất phát cho giai đoạn thực nghiệm cải tiến, **sau khi** tái lập xong.

**Quy ước quan trọng.** Mỗi mục được xếp vào một trong ba loại:

| Loại | Nghĩa | Có được gọi là "cải tiến" không? |
|---|---|---|
| **A. Cải tiến phương pháp** | Đổi cách *inference*: TTA, độ phân giải, ensemble. Protocol chấm và GT giữ nguyên như paper. | **Có**, nếu kết quả lặp lại được |
| **B. Khác biệt do protocol chấm** | Đổi GT, mask, crop hoặc vùng chấm. Model và dự đoán giữ nguyên. | **Không**. Số "tốt hơn" chỉ do chấm dễ hơn hoặc đúng hơn; dùng để bàn về protocol, không dùng để nói là vượt paper |
| **C. Ý tưởng chưa kiểm chứng** | Hướng gợi ra từ quan sát, chưa chạy | Chưa |

Khi so sánh, luôn giữ cố định mọi yếu tố khác (cùng checkpoint, cùng tập ảnh, cùng
protocol chấm) và chỉ đổi một yếu tố. Mỗi mục ghi kèm file kết quả để truy vết.

## A. Cải tiến phương pháp (inference)

### A1. Flip-TTA cho relative depth (Table 2)

Paper không dùng flip cho relative depth: NYUv2 khớp đúng khi không flip. Thêm
flip-TTA, tức lấy trung bình dự đoán của ảnh gốc và ảnh lật ngang:

| Dataset | Encoder | Không flip (protocol paper) | Flip-TTA | Thay đổi | Nguồn |
|---|---|---:|---:|---|---|
| DIODE (mask MoGe) | ViT-L | 0.0634 / 0.9552 | **0.0624 / 0.9559** | AbsRel −0.0010 | `results/table2/diode/followup/mask-flip-summary.json` |
| ETH3D (gói MoGe) | ViT-L | 0.0475 / 0.9798 | **0.0474 / 0.9799** | gần như không đổi | `results/table2/eth3d/vitl-518-flip-summary.json` |

Kết luận tạm thời: lợi ích nhỏ nhưng ổn định. Chi phí là gấp đôi thời gian inference.

### A2. Flip-TTA cho metric depth (Table 5)

Ở đây flip-TTA **đã là một phần protocol** của paper (`evaluate.py`), nên nó không
phải cải tiến so với paper. Ghi lại để biết TTA đóng góp bao nhiêu:

| Dataset | Có flip (paper) | Không flip | Nhận xét |
|---|---:|---:|---|
| vKITTI2 | 0.085 / 0.911 | 0.087 / 0.907 | flip giúp rõ nhất |
| iBims-1 | 0.150 / 0.714 | 0.150 / 0.713 | gần như không đổi |
| SUN RGB-D | 0.500 / 0.660 | 0.500 / 0.661 | không đổi |
| DIODE Outdoor | 0.794 / 0.288 | 0.794 / 0.289 | không đổi |
| KITTI (Table 4, in-domain) | 0.046 / 0.982, RMSE 1.896 | 0.047 / 0.982, RMSE 1.922 | flip giúp chủ yếu ở RMSE |

Nguồn: `results/table5/sweep/*.json`, `results/table4/sweep/kitti.json`.

### A3. Độ phân giải đầu vào (Table 2)

| Dataset | Cạnh ngắn 392 | Cạnh ngắn 518 (paper) | Ghép 392 + 518 | Nguồn |
|---|---:|---:|---:|---|
| DIODE (mask MoGe), ViT-L | 0.0646 / 0.9529 | **0.0634 / 0.9552** | 0.0635 / 0.9547 | `results/table2/diode/followup/resolution-summary.json` |
| ETH3D, 26 ảnh, ViT-L | 0.0661 / 0.9564 | **0.0589 / 0.9622** | — | `results/table2/eth3d/official-stride3-vitl-*-2per-scene-summary.json` |

Độ phân giải cao hơn tốt hơn rõ rệt, còn ghép đa độ phân giải không có lợi. Vì vậy
nên thử **cạnh ngắn > 518** (xem C1).

## B. "Tốt hơn" do protocol chấm (không phải cải tiến model)

| Bảng / dataset | Protocol paper | Cấu hình khác | Kết quả | Paper | Vì sao tốt hơn |
|---|---|---|---:|---:|---|
| T5 SUN RGB-D | GT ahanda / 10000 (= 0.8 × độ sâu thật) | GT đúng thang (`depth/` mm / 1000) | **0.114 / 0.850** | 0.500 / 0.660 | GT của paper lệch scale; model thực ra tốt hơn số paper rất nhiều |
| T5 DIODE Outdoor | loader bỏ qua `depth_mask.npy` | áp mask cảm biến | **0.572 / 0.290** | 0.794 / 0.288 | loại các pixel không hợp lệ nhưng vẫn mang giá trị depth |
| T2 KITTI | toàn ảnh, không crop | garg crop | S 0.074 / 0.945, B 0.074 / 0.948, L 0.071 / 0.955 | 0.080 / 0.936 · 0.080 / 0.939 · 0.076 / 0.947 | bỏ vùng trời và vùng xa phía trên ảnh |
| T2 DIODE | gói MoGe, không cap | gói MoGe, cap 80 m + kẹp disparity | S 0.068 / 0.946, B 0.061 / 0.952, L 0.058 / 0.956 | 0.076 / 0.939 · 0.069 / 0.946 · 0.066 / 0.952 | loại điểm xa > 80 m |
| T2 ETH3D | chưa xác định | mọi biến thể đã thử (cap, kẹp GT, lưới thu nhỏ, nguồn GT) | S 0.058–0.062, B 0.053, L 0.046–0.052 | 0.127 / 0.126 / 0.127 | protocol paper khắt khe hơn, chưa tìm ra (xem table2-reproduction.md mục 5) |
| T2 Sintel | chưa xác định | cap 72–80 m, loại pixel xa | S 0.340 / 0.731, L 0.300 / 0.781 | 0.464 / 0.739 · 0.458 / 0.760 | bỏ bầu trời và 19 khung `mountain_1` |
| T4 KITTI | cap 80 m | `max_depth_eval = 50` | 0.045 / 0.985, RMSE 1.403 | 0.046 / 0.982, RMSE 1.896 | bỏ vùng xa > 50 m |

Những dòng này hữu ích cho phần **thảo luận về protocol** trong báo cáo, ví dụ:
"số SUN RGB-D của paper bị ảnh hưởng bởi GT lệch thang". Chúng **không** phải cải
tiến phương pháp.

## C. Ý tưởng cần kiểm chứng (gợi ra từ quan sát)

1. **C1. Độ phân giải lớn hơn 518 cho relative depth.** Từ 392 lên 518 cải thiện
   rõ (A3). Thử cạnh ngắn 700 / 840, tốn thêm bộ nhớ trên GPU 4 GB.
2. **C2. Hiệu chỉnh thang đo cho metric zero-shot (Table 5).** Sai số lớn ở DIODE
   Outdoor và SUN RGB-D chủ yếu do sai thang đo. Có thể thử:
   - dùng focal length / intrinsics của camera;
   - median-scaling bằng vài điểm GT thưa (mô phỏng LiDAR) — lưu ý cách này dùng
     thêm thông tin, nên phải báo cáo riêng.
3. **C3. Chọn checkpoint theo loại cảnh.** Hiện SUN và iBims dùng indoor, vKITTI2 và
   DIODE dùng outdoor, theo paper. Thử bộ phân loại indoor/outdoor tự động, hoặc
   trộn dự đoán của hai checkpoint.
4. **C4. Flip-TTA + multi-scale cho metric depth:** kiểm tra lợi ích trên vKITTI2,
   nơi flip đã giúp +0.004 δ₁.

## Nhật ký

| Ngày | Mục | Ghi chú |
|---|---|---|
| 2026-11-28 | A1–A3, B, C1–C4 | Tổng hợp từ các lượt tái lập Table 2 và Table 5 |
| 2026-11-28 | A2, B | Thêm Table 4 (flip, cap 50 m), KITTI garg crop cho S/B/L, DIODE cap 80, ETH3D, Sintel |
