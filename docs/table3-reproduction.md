# Tái hiện Table 3 — in-domain metric depth trên NYUv2 (Depth Anything V1)

Rà soát ngày 2026-11-28. **Trạng thái: HOÀN THÀNH — KHỚP 6/6 chỉ số, không cần chạy lại.**

Mục tiêu là hàng **Ours** của Table 3 bản CVPR 2024. Đây là phép **đánh giá checkpoint
đã fine-tune** `depth_anything_metric_depth_indoor.pt` trên 654 ảnh test chính thức
của NYUv2. Phép đánh giá này không huấn luyện lại model.

| Chỉ số | δ₁ ↑ | δ₂ ↑ | δ₃ ↑ | AbsRel ↓ | RMSE ↓ (m) | log10 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Paper, Ours | 0.984 | 0.998 | 1.000 | 0.056 | 0.206 | 0.024 |
| Paper, ZoeDepth\* (để tham chiếu) | 0.951 | 0.994 | 0.999 | 0.077 | 0.282 | 0.033 |

## Ngưỡng đánh giá

Kết luận dựa trên **AbsRel và δ₁**, giống [Table 2](table2-reproduction.md) và
[Table 5](table5-reproduction.md). So sánh trên giá trị làm tròn 3 chữ số, với
d = max(|ΔAbsRel|, |Δδ₁|):

| Kết luận | Điều kiện |
|---|---|
| KHỚP | d = 0 |
| ĐẠT | d ≤ 0.005 |
| KHÔNG ĐẠT | d > 0.005 |

Bốn chỉ số còn lại được báo kèm độ lệch. Ghi chú thêm "khớp 6/6" khi cả sáu chỉ số
đều trùng paper.

## Kết quả (nhóm đã chạy)

Kết quả lưu ở `results/table3/nyuv2/` (trước đây là `results/nyuv2-metric/`).

Các lượt này được chạy trên một máy khác, nhưng môi trường giống hệt máy hiện tại:

- torch 2.4.0+cu121, numpy 1.26.4, GPU RTX 3050 Laptop;
- SHA-256 của checkpoint indoor trùng (`e25a9660…`).

Vì vậy chạy lại sẽ không thêm thông tin gì.

| Lượt | Số ảnh | δ₁ | δ₂ | δ₃ | AbsRel | RMSE | log10 | Kết luận |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `indoor-flip`: protocol BTS + rawDepths | 654 | 0.98355 | 0.99800 | 0.99961 | 0.05598 | 0.20614 | 0.02439 | **KHỚP, 6/6** |
| `filled-mat-diagnostic`: nhãn `depths` đã inpaint | 654 | 0.96777 | 0.99243 | 0.99783 | 0.06371 | 0.26753 | 0.02783 | KHÔNG ĐẠT (chẩn đoán) |

Làm tròn 3 chữ số, lượt chính thức ra đúng 0.984 / 0.998 / 1.000 / 0.056 / 0.206 / 0.024.

Con số **thấp** trong thư mục kết quả (δ₁ 0.968, AbsRel 0.064) là **lượt chạy đầu
tiên** của nhóm (14:54 UTC, 2026-11-26). Lượt này đọc thẳng `depths` từ file `.mat`
và đưa toàn bộ ảnh RGB vào model, nên không khớp paper.

Sau đó nhóm lần theo [script chuẩn bị dữ liệu của BTS](https://github.com/cleinc/bts/blob/master/utils/extract_official_train_test_set_from_mat.py)
và đổi sang `rawDepths` cùng cách chuẩn bị RGB của BTS. Lượt chạy lại lúc 15:07 khớp
6/6 chỉ số.

Lượt đầu được giữ lại dưới tên `filled-mat-diagnostic` vì hai lý do:

- **Truy vết:** nó là bằng chứng rằng việc đổi protocol dựa trên mã nguồn của BTS,
  chứ không phải chỉnh tham số cho khớp số paper.
- **Đo mức ảnh hưởng:** nó cho thấy protocol dữ liệu làm lệch đến −0.016 δ₁ và
  +0.008 AbsRel.

Lượt này đổi **hai yếu tố cùng lúc**: nguồn GT (`depths` thay cho `rawDepths`) và khâu
chuẩn bị RGB (toàn ảnh thay cho viền đen + JPEG). Vì vậy nó không tách được từng yếu
tố đóng góp bao nhiêu. Evaluator hiện tại cũng không còn tùy chọn để chạy lại lượt
này. Phần bổ sung B bên dưới (không bắt buộc) có thể tách riêng hai yếu tố.

Protocol của lượt chính thức (chi tiết ở [nyuv2-experiments.md](table3/nyuv2-experiments.md)):

- **GT:** `rawDepths` lượng tử về mm, `(raw*1000).astype(uint16)/1000`, đúng như bản
  PNG BTS xuất.
- **RGB:** chuẩn bị như BTS. Giữ vùng [7:474, 7:632], viền ngoài tô đen, ghi JPEG
  quality 95 rồi decode lại. Toàn bộ làm trong bộ nhớ. Nhóm đã kiểm chứng bit-for-bit
  với việc ghi ra đĩa trên các fixture hồi quy. Kết quả khớp 6/6 cũng xác nhận gián
  tiếp bước chuẩn bị này.
- **Model:** resize 392×518 với `keep_aspect_ratio=False`, dùng flip-TTA. Dự đoán được
  nội suy bilinear về 480×640.
- **Chấm:** Eigen crop [45:471, 41:601], 0.001 < GT < 10 m, dùng `compute_metrics`
  chính thức, trung bình theo ảnh, không bỏ ảnh nào.

## Chạy lại (khi cần)

Đổi môi trường hoặc checkpoint thì chạy lại bằng lệnh dưới, mất khoảng 10 phút:

```powershell
.venv\Scripts\python.exe evaluate_nyuv2_metric.py --limit 4     # smoke test
.venv\Scripts\python.exe evaluate_nyuv2_metric.py               # đủ 654 ảnh
```

## Phần bổ sung (không bắt buộc)

Hai mục dưới không thay đổi kết luận KHỚP. Chỉ làm khi báo cáo cần thêm bằng chứng.

### A. Đối chứng chéo bằng evaluator chính thức `metric_depth/evaluate.py -d nyu`

Mục đích: chạy đúng lệnh trong `metric_depth/README.md` trên dữ liệu PNG theo layout
BTS. Mục này cho thấy evaluator dùng bộ nhớ của nhóm cho kết quả trùng với
evaluator gốc.

1. Xuất official test split ra `data/nyu/` theo layout mà
   `train_test_inputs/nyudepthv2_test_files_with_gt.txt` yêu cầu, dùng
   [script BTS](https://github.com/cleinc/bts/blob/master/utils/extract_official_train_test_set_from_mat.py).
2. Chạy:
   ```bash
   cd metric_depth
   ../.venv/Scripts/python.exe evaluate.py -m zoedepth \
     --pretrained_resource="local::./checkpoints/depth_anything_metric_depth_indoor.pt" -d nyu
   ```
3. So sánh với kết quả `indoor-flip`. Kỳ vọng hai bên trùng ở 3 chữ số.

- **Rủi ro:** nhánh `online_eval` của `DepthDataLoader` dùng `num_workers=1` cố định.
  Trên Windows, worker được tạo bằng `spawn` và phải pickle dataset. Nếu gặp lỗi
  pickle như ở Table 5, chuyển sang `EVAL_WORKERS`.
- **Kết quả:** _chưa có_
- **Kết luận:** _chưa có_

### B. Quét cấu hình một yếu tố, giống Table 5

Mỗi lần chỉ đổi một yếu tố so với protocol gốc, rồi chấm KHỚP / ĐẠT / KHÔNG ĐẠT:

| Cấu hình | Mục đích | Kết quả |
|---|---|---|
| base | protocol gốc | _chưa có_ |
| không flip-TTA (`--no-flip`) | đo ảnh hưởng của TTA | _chưa có_ |
| chỉ đổi GT sang `depths` đã inpaint | tách riêng ảnh hưởng của nguồn GT | _chưa có_ |
| chỉ bỏ bước JPEG + viền đen của BTS | tách riêng ảnh hưởng của khâu chuẩn bị RGB | _chưa có_ |
| đổi cả hai (= lượt đầu `filled-mat-diagnostic`) | đã có: 0.064 / 0.968, KHÔNG ĐẠT | có |
| bỏ Eigen crop | đo ảnh hưởng của crop | _chưa có_ |
| `keep_aspect_ratio=True` | đo ảnh hưởng của cách resize | _chưa có_ |

Cách triển khai: sau mục A đã có PNG theo layout BTS, nên chỉ cần thêm `nyu` vào
`scripts/sweep_table5.py`. Script này chạy inference một lần rồi chấm nhiều cấu hình.
Các biến thể thuộc khâu chuẩn bị RGB hoặc resize cần một lượt inference riêng.

## Ngoài phạm vi

Theo yêu cầu, các bảng khác có NYUv2 (Table 6, 9, 10, 11) **không thực nghiệm lại**.
Chúng cần checkpoint của các biến thể huấn luyện mà tác giả không phát hành (xem
[nyuv2-experiments.md](table3/nyuv2-experiments.md)).

## Cấu trúc thư mục

```text
docs/table3-reproduction.md      tài liệu này
docs/table3/nyuv2-experiments.md        protocol chi tiết của evaluator NYUv2 metric
evaluate_nyuv2_metric.py         evaluator (output mặc định: results/table3/nyuv2/)
results/table3/nyuv2/            indoor-flip-*, filled-mat-diagnostic-*, initial-full-run-*
```
