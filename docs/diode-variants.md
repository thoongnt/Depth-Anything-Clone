# DIODE: so sánh cấu hình với Table 2

Mốc ViT-L: **AbsRel 0.066, δ₁ 0.952**. Đây là thăm dò protocol theo yêu cầu đối chiếu số; gần số paper chưa tự xác nhận đó là protocol tác giả.

Cùng checkpoint relative ViT-L và dữ liệu đã xác thực 2.313 file. Full flip có đủ 771 ảnh: flip ngang, lật prediction về chiều gốc, trung bình với prediction ảnh gốc trước resize về GT. Không dùng lại kết quả probe 16 ảnh làm full TTA.

## Flip và từng miền, giữ nguyên mask/OLS/disparity

| Flip | Phạm vi | Số ảnh | AbsRel ↓ | δ₁ ↑ |
|---|---|---:|---:|---:|
| False | all | 771 | 0.267872421 | 0.753644631 |
| False | indoors | 325 | 0.057393579 | 0.955752289 |
| False | outdoor | 446 | 0.421248258 | 0.606368871 |
| True | all | 771 | 0.266833460 | 0.754083973 |
| True | indoors | 325 | 0.056591391 | 0.956152959 |
| True | outdoor | 446 | 0.420036761 | 0.606836393 |

Ba cảnh outdoor đều lệch so với mốc Table 2. Với flip và cách chấm native, AbsRel từng cảnh nằm trong khoảng 0.266–0.482. [Kết quả từng scene](../results/diode-relative/variants/native-per-scene.csv).

## Gần số paper nhất theo phạm vi

Xếp hạng bằng `max(|AbsRel−0.066|, |δ₁−0.952|)`, yêu cầu đủ ảnh của phạm vi đã chọn. Tỷ lệ pixel là số pixel được chấm chia tổng pixel native hợp lệ cùng phạm vi. Những dòng có ảnh mask rỗng không được xếp là đủ subset.

| Phạm vi | Cấu hình | Flip | Số ảnh | Giữ pixel native | AbsRel ↓ | δ₁ ↑ |
|---|---|---|---:|---:|---:|---:|
| all | relative-ls / image-mean | False | 771 | 100.00% | 0.119159439 | 0.843091329 |
| indoors | range-10m-clamp10 / image-mean | False | 325 | 91.43% | 0.061011122 | 0.947643674 |
| outdoor | relative-ls / image-mean | False | 446 | 100.00% | 0.169474421 | 0.761554105 |

Gần nhất trên đủ **771 ảnh, giữ 100% pixel chấm native, disparity, trung bình từng ảnh**: **relative-ls**, flip=False, **0.119159439 / 0.843091329**.

Số trường hợp đồng thời làm tròn thành **0.066 / 0.952**: **0**.

## Định nghĩa cấu hình

- `native`: official mask, full frame, OLS vào 1/GT, chấm disparity thô.
- `native-clamp-gt`: clamp disparity tối thiểu bằng 1/max GT depth hợp lệ mỗi ảnh.
- `native-depth-metric`: clamp trên rồi nghịch đảo để chấm depth; khác hướng dẫn disparity của tác giả.
- `range-80m`, `range-0.5-80m`, `range-0.6-80m`, `range-10m`, `range-0.5-10m`: chỉ fit/chấm pixel trong khoảng depth ghi ở tên, hai biên không được giữ.
- `range-10m-clamp10`: mask 0 < depth < 10 m; clamp disparity tối thiểu 0.1.
- `nyu-eigen-fraction-crop`: tỷ lệ crop [45/480:471/480, 41/640:601/640] trên DIODE; giả thuyết chuyển crop NYU, chưa có xác nhận cho DIODE.
- `median-mad`: match median và mean absolute deviation, thay OLS.
- `relative-ls`: tối thiểu bình phương relative-disparity error, trọng số 1/(GT disparity)^2.
- `huber-8`: 8 vòng IRLS Huber, ngưỡng 1.345 × scale MAD residual, scale tối thiểu 1e-9; chấm toàn bộ GT mask.

Mọi cấu hình giữ supplied mask DIODE và depth hữu hạn/dương. Ngưỡng/crop/robust fit là các giả thuyết, chưa được xác nhận là protocol paper. Không loại ảnh theo sai số. Ảnh không còn pixel được liệt kê trong JSON, cùng cờ `full_subset=false`.

13 cấu hình × 2 chế độ flip × 3 phạm vi = 78 kết quả; thêm trung bình theo pixel cho native ở 6 nhóm = **84 trường hợp**. Trung bình pixel dùng trọng số số pixel hợp lệ từng ảnh. Các cấu hình subset/crop/range thay đổi phạm vi đánh giá, nên cần đọc cùng số ảnh và tỷ lệ pixel.

## Artefact và cách chạy

- [Toàn bộ CSV](../results/diode-relative/variants/protocol-grid/comparison.csv), [JSON](../results/diode-relative/variants/protocol-grid/comparison.json), [định nghĩa cấu hình](../results/diode-relative/variants/configurations.json).
- [Full flip summary](../results/diode-relative/variants/vitl-native-all-flip-summary.json), [CSV từng ảnh](../results/diode-relative/variants/vitl-native-all-flip-per-image.csv).
- Thư mục `protocol-grid/no-flip-*` và `protocol-grid/flip-*` lưu CSV từng ảnh cho từng cấu hình: metric, pixel, scale/shift, trạng thái.

```powershell
.venv\Scripts\python.exe evaluate_diode_relative.py --flip --data-audit results/diode-relative/audit/official-archive-verification.json --cache-dir .cache/diode-variants/predictions --output-dir results/diode-relative/variants

.venv\Scripts\python.exe research_diode_variants.py --workers 8 --summary results/diode-relative/audit-rerun/fresh-inference-summary.json results/diode-relative/variants/vitl-native-all-flip-summary.json
```

Evaluator mặc định và kết quả NYUv2 không thay đổi. Những kết quả được chọn vì gần số đích là thăm dò sau khi biết số paper, không phải kiểm chứng độc lập protocol.
