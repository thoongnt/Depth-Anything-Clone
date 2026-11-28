# DIODE V1 · kiểm tra mask, ảnh và độ phân giải

Mốc ViT-L ở Table 2 là **0.066 / 0.952** (AbsRel / δ₁). So sánh ghép cùng 771 ID giữa DIODE gốc và bundle MoGe, gồm 325 indoor và 446 outdoor. Khoảng cách tới paper được tính bằng độ lệch tuyệt đối lớn hơn trong hai chỉ số. Báo cáo theo từng phạm vi và cả tỷ lệ pixel so với mask gốc.

[Dataset card MoGe](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation) ghi rằng bundle DIODE đã lọc lỗi ở biên và hiệu chỉnh FoV. Số lượng theo miền ở đây lấy từ file `.index.txt` trong zip. Báo cáo JSON mỗi lượt lưu SHA256 của archive và số ID đã đối chiếu.

## ViT-L trên đủ 771 ảnh

| Cấu hình | Ảnh | Pixel so với native | AbsRel / δ₁ |
|---|---:|---:|---:|
| Native, cạnh ngắn 518 | 771 | 100.00% | 0.267872 / 0.753645 |
| Native RGB/depth, mask giao MoGe | 771 | 74.75% | 0.063439 / 0.955234 |
| Native RGB, depth MoGe, mask giao | 771 | 74.75% | 0.063439 / 0.955234 |
| Native RGB/depth, mask giao MoGe, flip TTA | 771 | 74.75% | 0.062390 / 0.955876 |
| RGB/depth/mask MoGe, chấm disparity | 771 | 74.75% | 0.064903 / 0.953573 |
| RGB/depth/mask MoGe, chấm depth có clamp | 771 | 74.75% | 0.057057 / 0.956615 |
| Native, cạnh ngắn 392 | 771 | 100.00% | 0.270279 / 0.752031 |
| Native, ghép 392+518 | 771 | 100.00% | 0.268445 / 0.753084 |
| Native RGB/depth, mask giao MoGe, cạnh ngắn 392 | 771 | 74.75% | 0.064582 / 0.952898 |
| Native RGB/depth, mask giao MoGe, ghép 392+518 | 771 | 74.75% | 0.063513 / 0.954668 |

Cấu hình gần mốc ViT-L nhất trong loạt này là `moge-mask-short-side-392` (disparity), đạt **0.064582 / 0.952898**. Cấu hình gần nhất trong disparity đạt **0.064582 / 0.952898**. Số kết quả toàn tập khớp dòng paper tương ứng ở ba chữ số thập phân là **0**.

Cách ghép 392+518 căn chỉnh prediction 392 vào thang của prediction 518 bằng fit trên hai prediction rồi lấy trung bình. Bước này không dùng depth GT. Lượt bundle dùng lưới lưu sẵn 1024×768, không chạy thêm warp hay quy tắc loại điểm xa của loader MoGe. Nhánh depth có clamp dùng công thức trong mã MoGe. Tác giả Depth Anything xác nhận họ chấm trong disparity. [Công thức MoGe](https://github.com/microsoft/MoGe/blob/main/moge/test/metrics.py#L300-L319) · [Giải thích của tác giả Depth Anything](https://github.com/LiheYoung/Depth-Anything/issues/174)

## Hiệu ứng mask lặp lại trên ba checkpoint

Cùng native RGB, native depth, không flip và chấm trong disparity. Cột bên phải lấy dòng Table 2 của đúng encoder.

| Encoder | Mask gốc | Mask giao MoGe | Paper |
|---|---:|---:|---:|
| vits | 0.273667 / 0.745337 | 0.073836 / 0.944392 | 0.076 / 0.939 |
| vitb | 0.270808 / 0.749509 | 0.067442 / 0.950724 | 0.069 / 0.946 |
| vitl | 0.267872 / 0.753645 | 0.063439 / 0.955234 | 0.066 / 0.952 |

Mask MoGe giữ 394.531.136 trên 527.773.509 pixel hợp lệ gốc, tương ứng **74,75% toàn tập**, **90,46% indoor** và **60,29% outdoor**. Trong bộ đã đọc, mọi pixel hợp lệ của MoGe đều nằm trong mask native. Giữ prediction và mask giao rồi chỉ đổi giá trị depth giữa hai bộ làm AbsRel/δ₁ thay đổi dưới 0,000002. Số liệu này xác định mask là tác động chính trong so sánh đã chạy. Chưa có manifest hoặc quy tắc mask từ tác giả để xác nhận kết quả Table 2 dùng cùng vùng pixel.

## Tách tác động của mask khi fit và khi chấm

Cùng RGB, depth và prediction native trên đủ 771 ảnh.

| Mask khi fit | Mask khi chấm | AbsRel / δ₁ |
|---|---|---:|
| Native | Native | 0.267872 / 0.753645 |
| Native | Giao MoGe | 0.340096 / 0.757587 |
| Giao MoGe | Native | 0.126546 / 0.839872 |
| Giao MoGe | Giao MoGe | 0.063439 / 0.955234 |

Kết quả gần paper cần cả fit lại và dùng vùng chấm đã xử lý. Nếu bắt buộc giữ mọi pixel native, cấu hình fit sai số tương đối ở đợt trước đạt **0.119159 / 0.843091**, nên vẫn cách xa paper. [Grid trước đó](diode-variants.md)

## Kết quả và lệnh chạy

Script cho thực nghiệm 1 trên DIODE với mask MoGe: [thuc_nghiem_1_diode_relative_depth_moge.py](../../thuc_nghiem_1_diode_relative_depth_moge.py). Chế độ `resolution` có dòng `moge-mask-short-side-392` trong [resolution-summary.json](../../results/table2/diode/followup/resolution-summary.json).

- [Bảng CSV toàn bộ cấu hình và từng miền](../../results/table2/diode/followup/comparison.csv) và [kết quả xếp hạng](../../results/table2/diode/followup/comparison.json).
- [Đối chứng giữ fit native](../../results/table2/diode/followup/fixed-native-fit-summary.json) và [đối chứng fit MoGe rồi chấm native](../../results/table2/diode/followup/moge-fit-native-score-summary.json).
- Các file `*-per-image.csv` trong thư mục `results/table2/diode/followup/` ghi số pixel và hai chỉ số từng ảnh. Đã kiểm tra lại tổng hợp từ CSV, số lượng ảnh và SHA256 chung của archive.

```powershell
.venv\Scripts\python.exe thuc_nghiem_1_diode_relative_depth_moge.py mask mask-flip bundle resolution
.venv\Scripts\python.exe thuc_nghiem_1_diode_relative_depth_moge.py mask-vits mask-vitb
```

Các lệnh đọc archive ở `.cache/moge-diode/DIODE.zip`. `mask` và `mask-flip` dùng cache ViT-L từ các lượt native đã chạy. `mask-vits` và `mask-vitb` dùng summary và cache native của hai encoder trong báo cáo follow-up. Để tạo lại chúng, chạy `evaluate_diode_relative.py --encoder vits` và `--encoder vitb` với `--output-dir results/table2/diode/followup --cache-dir .cache/diode-followup/native`.
