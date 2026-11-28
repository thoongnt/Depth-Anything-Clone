> **Rà soát bổ sung:** [Kiểm chứng archive, nhãn, mã nguồn và lượt suy luận mới](diode-audit.md). 2.313 file local trùng SHA256 với bản chính thức; các biến thể scoring đã được kiểm tra đủ 771 ảnh.

# Thực nghiệm 1 — DIODE: nghiên cứu protocol

Ngày rà soát: 2026-11-26. Đã triển khai evaluator zero-shot relative depth và notebook DIODE. Protocol native được ghi rõ; chưa xác nhận tái hiện đúng protocol Table 2 của tác giả.

## Mục tiêu và checkpoint

Table 2 của [bài báo CVPR 2024](../../paper/Yang_Depth_Anything_Unleashing_the_Power_of_Large-Scale_Unlabeled_Data_CVPR_2024_paper.pdf) báo cáo DIODE với Depth Anything V1 ViT-L: **AbsRel 0.066, δ₁ 0.952**. Dùng checkpoint relative `checkpoints/depth_anything_vitl14.pth`, không cần huấn luyện hoặc fine-tune trên DIODE. SHA256 checkpoint đã dùng cho probe: `6fda5199f0043e91e1c6117df00313db57c630d8ed0306070b43da1ecfffbd9d`.

## Dữ liệu hiện có

`data/diode/val/` có 771 ảnh: 325 trong `indoors`, 446 trong `outdoor`. Đã kiểm tra toàn bộ cặp depth/mask: không thiếu file, shape sau squeeze đều 768 × 1024. Đã giải mã RGB toàn bộ 771 ảnh trong full run. Rà soát bổ sung: archive tải lại từ S3 có MD5 khớp mã công bố và toàn bộ 2.313 file RGB/depth/mask local có SHA256 giống bản trong archive; không thiếu hoặc khác file.

Một mẫu gồm ba file cùng stem:

```text
val/indoors/scene_00019/scan_00183/
    00019_00183_indoors_000_010.png
    00019_00183_indoors_000_010_depth.npy
    00019_00183_indoors_000_010_depth_mask.npy
```

Tên miền chính xác là `indoors` và `outdoor`. [Trang chính thức DIODE](https://diode-dataset.org/) xác nhận cách đặt tên, mask hợp lệ bằng 1, ảnh 1024 × 768 và số lượng validation trên. Train/validation được công bố kèm nhãn; bộ validation đang có đủ để bắt đầu đánh giá. Không cần tải train hoặc normals.

| Kiểm tra trên dữ liệu local | Indoors | Outdoor |
|---|---:|---:|
| Số ảnh | 325 | 446 |
| Khoảng depth tại pixel hợp lệ (m) | 0.300–49.958 | 0.300–299.829 |
| Pixel có depth > 0 nhưng mask = 0 | 1,961,181 | 5,452,716 |
| Pixel hợp lệ có depth > 80 m | 0 | 673,048 |

Vì vậy chỉ kiểm tra `depth > 0` sẽ giữ cả điểm cảm biến không hợp lệ. Áp trần 10 m của NYUv2 hoặc 80 m sẽ thay đổi tập pixel được chấm. Chưa có bằng chứng cho phép gọi các ngưỡng đó là protocol DIODE chính thức của bài báo.

## Những lỗi trong nhánh cũ

Các lỗi tìm thấy trong nhánh cũ của `thuc_nghiem_1_relative_depth.py` và notebook (đã thay bằng evaluator dùng chung):

- Glob `**/*_image.png` không khớp tên ảnh thật; cách suy ra đường dẫn depth/mask từ hậu tố này cũng sai.
- Nhánh cũ căn chỉnh đầu ra disparity trực tiếp vào depth GT.
- CLI cũ chưa có lựa chọn riêng `--dataset diode`; lựa chọn `all` không bảo đảm phát hiện danh sách DIODE rỗng.
- Hàm metric chung cần xử lý prediction không dương để δ₁ không nhận nhầm tỉ số âm là đúng.

Loader `metric_depth/zoedepth/data/diode.py` thuộc pipeline metric depth; mask không được giữ trong mẫu đầu ra sau transform. Không dùng nguyên pipeline đó cho relative benchmark. Có thể tái sử dụng `make_transform` và `align_disparity` của `evaluate_nyuv2.py`, nhưng không tái sử dụng mask hoặc hàm chấm chuyên NYUv2 vì có giả định riêng về kích thước/ngưỡng depth.

## Protocol có cơ sở để triển khai

[Tác giả xác nhận trong issue #174](https://github.com/LiheYoung/Depth-Anything/issues/174): chuyển GT depth sang disparity, fit scale và shift vào prediction thô, rồi tính metric trong disparity; không min-max prediction về 0–1.

Pipeline baseline đề xuất cho dữ liệu hiện có:

1. Chốt manifest toàn bộ 771 ảnh, kiểm tra mỗi ảnh có depth và mask; lỗi phải được báo, không âm thầm bỏ mẫu.
2. RGB float32 / 255; dùng transform suy luận V1: cạnh ngắn ít nhất 518, giữ tỉ lệ, kích thước bội 14, OpenCV cubic, ImageNet normalization. Resize prediction về kích thước GT bằng bilinear `align_corners=False`.
3. Mask `M = (depth_mask > 0) & isfinite(depth) & (depth > 0)`. Baseline dùng full frame, không crop hay depth cap; đây là lựa chọn minh bạch khi chưa rõ protocol chi tiết của tác giả.
4. Trên M, đặt `q = 1 / depth`, fit `s, t = argmin Σ(s*p + t - q)²` bằng float64, rồi đặt `a = s*p + t`.
5. `AbsRel = mean(abs(a-q)/q)`. Với δ₁, pixel `a > 0` đạt khi `max(a/q, q/a) < 1.25`; `a <= 0` tính là sai. Giữ nguyên mask GT khi chấm, không loại pixel chỉ vì model dự đoán kém.
6. Tính metric từng ảnh, sau đó lấy trung bình trên ảnh. Xuất cả indoors, outdoor và all. Với full split, all là `(325*mean_indoors + 446*mean_outdoor)/771`, không lấy trung bình đều hai miền.
7. Chạy no-flip baseline trước; flip TTA là biến thể riêng, ghi rõ trong output. Lưu protocol, manifest, checkpoint hash, số pixel, scale/shift và per-image CSV để truy vết.

Các lựa chọn mask chi tiết, clipping prediction, depth cap, split chính xác, TTA và cách tổng hợp chưa được xác nhận đầy đủ cho DIODE trong nguồn tác giả đã kiểm tra. Đặc biệt, chưa có manifest chứng minh số Table 2 được tính đúng trên 771 ảnh validation native này. Không coi việc chưa khớp số là lỗi checkpoint ngay lập tức.

## Kết quả full validation đã chạy

ViT-L, no-flip, native mask, không crop/depth cap. Đã chạy đủ **771/771 ảnh**, không bỏ mẫu; thời gian CLI khoảng **9 phút 58 giây** trên RTX 3050 Laptop 4 GB (1 ảnh đã có cache từ smoke test).

| Phạm vi | Số ảnh | AbsRel disparity ↓ | δ₁ ↑ |
|---|---:|---:|---:|
| Indoors | 325 | 0.057393579 | 0.955752289 |
| Outdoor | 446 | 0.421248258 | 0.606368871 |
| Toàn bộ native validation | 771 | 0.267872421 | 0.753644631 |
| Paper Table 2 — ViT-L, DIODE | Không có manifest đối chiếu | 0.066 | 0.952 |

**Chưa tái hiện được số paper trên protocol native đã chốt.** Chênh lệch lớn hơn làm tròn; outdoor là phần lệch chính. Không lấy riêng kết quả indoor thay cho DIODE aggregate. Chưa xác định nguyên nhân cuối cùng vì split/clipping/TTA chi tiết của paper còn thiếu.

Có **527,773,509 pixel GT hợp lệ**, khớp audit dữ liệu trước đó; **242,811 pixel prediction sau alignment không dương** được giữ trong mẫu số và tính là sai cho δ₁. AbsRel chấm trực tiếp disparity đã alignment, không clip prediction.

Artefact chính:

- [Summary](../../results/table2/diode/vitl-native-all-summary.json), [per-image CSV](../../results/table2/diode/vitl-native-all-per-image.csv), [manifest](../../results/table2/diode/vitl-native-all-manifest.json).
- [Summary lượt suy luận đầu](../../results/table2/diode/initial-full-run-summary.json) được giữ riêng trước khi notebook chấm lại từ cache.
- [Kiểm chứng độc lập](../../results/table2/diode/verification.json): 771 ảnh không trùng, đúng 325/446 theo miền, tổng hợp CSV khớp summary, số pixel khớp audit; prediction của 16 ảnh probe khớp, phép fit `numpy.linalg.lstsq` độc lập cho metric khớp trong sai số số học.

23 unit tests đã qua, gồm 9 test DIODE và 14 test NYUv2 hiện có. Notebook riêng đã thực thi thành công bằng `.venv`: đọc đủ **771 cache hits**, không có cell lỗi; metric từng miền/toàn bộ trùng khớp hoàn toàn lượt CLI đầu tiên.

## Probe đã chạy

Đã chạy ViT-L no-flip trên **16 ảnh**, chọn trước 8 ảnh cách đều theo thứ tự tên file mỗi miền; mask/fit/metric theo baseline ở trên. Đây không phải full benchmark và không đại diện theo tỉ lệ 325/446.

| Miền | Số ảnh probe | AbsRel disparity ↓ | δ₁ ↑ |
|---|---:|---:|---:|
| Indoors | 8 | 0.049719 | 0.961861 |
| Outdoor | 8 | 0.363920 | 0.476991 |

Chênh lệch outdoor cần được điều tra khi chạy toàn bộ, không chọn riêng indoor để đối chiếu số chung của paper. Thử bổ sung clamp disparity theo nghịch đảo depth GT lớn nhất mỗi ảnh trên cùng 16 ảnh không cải thiện outdoor. Đây chỉ là chẩn đoán; không phải protocol được tác giả xác nhận.

Artefact lưu tại [results/table2/diode/research](../../results/table2/diode/research/): `data-audit.json`, `sample-probe.json` (kèm từng ảnh), `clamp-diagnostic.json`, `probe_relative.py`. Có thể chạy lại probe từ repo root bằng `.venv\Scripts\python.exe results/table2/diode/research/probe_relative.py`; script cần CUDA, ghi dữ liệu làm việc vào `.cache/diode-research/`, không tạo kết quả full benchmark.

## Cách chạy evaluator đã triển khai

```powershell
# Chạy đủ 771 ảnh, ViT-L no-flip:
.venv\Scripts\python.exe thuc_nghiem_1_relative_depth.py --dataset diode --encoder vitl

# Chạy từng miền từ cùng cache:
.venv\Scripts\python.exe evaluate_diode_relative.py --subset indoors
.venv\Scripts\python.exe evaluate_diode_relative.py --subset outdoor

# Chẩn đoán 8 ảnh đầu tiên (không phải benchmark đủ split):
.venv\Scripts\python.exe evaluate_diode_relative.py --limit 8

# TTA là biến thể riêng, không ghi đè no-flip:
.venv\Scripts\python.exe evaluate_diode_relative.py --flip
```

Notebook độc lập: [thuc_nghiem_1_diode_relative_depth.ipynb](../../thuc_nghiem_1_diode_relative_depth.ipynb). Notebook tổng thực nghiệm 1 cũng gọi cùng evaluator ở cell DIODE, tái sử dụng model đã nạp để tránh nạp hai model trên GPU 4 GB.

Output mặc định: `results/table2/diode/vitl-native-all-summary.json`, `vitl-native-all-per-image.csv`, `vitl-native-all-manifest.json`. Tên file có subset, flip và limit để phân biệt các lần chạy. Summary ghi counts, metric từng miền/toàn bộ, pixel không dương, checkpoint hash, protocol, môi trường và cache hits. Manifest ghi đường dẫn, size, mtime của từng RGB/depth/mask; đây không phải checksum nội dung dataset.

Prediction cache dùng checkpoint/source/preprocessing/flip và metadata RGB để tách kết quả; fit/metric luôn tính lại từ GT/mask hiện tại. Prediction float32 được ghi qua file tạm trước khi đổi tên; lần chạy bị ngắt có thể chạy lại cùng lệnh. Cache native đủ 771 ảnh cần khoảng 2.43 GB (2.26 GiB). Thiếu cặp, sai số lượng/shape, mask rỗng hay prediction không hợp lệ đều báo lỗi, không bỏ mẫu âm thầm.

`--limit` lấy các ảnh đầu tiên trong thứ tự indoors rồi outdoor; ví dụ `--limit 8` không lấy cân bằng hai miền như probe nghiên cứu 16 ảnh phía trên.


Nếu full baseline lệch paper, đối chiếu manifest và preprocessing trước, sau đó các quy tắc scoring được công bố. Các pipeline như [MoGe](https://github.com/microsoft/MoGe) có bundle/preprocessing và metric riêng, nên chỉ dùng làm đối chứng có mô tả khác biệt. Không thay dataset, bỏ ảnh/pixel lỗi hoặc đổi ngưỡng chỉ để ép khớp 0.066/0.952. Các thảo luận [#172](https://github.com/DepthAnything/Depth-Anything-V2/issues/172) và [#280](https://github.com/DepthAnything/Depth-Anything-V2/issues/280) cũng ghi nhận khó khăn tái hiện; chúng không thay thế protocol chính thức.

Evaluator và kết quả NYUv2 được giữ nguyên; nhánh DIODE của script/notebook thực nghiệm 1 đã thay bằng evaluator riêng. Kiểm thử gồm mask, disparity alignment/scoring, aggregation và xử lý dữ liệu lỗi.
