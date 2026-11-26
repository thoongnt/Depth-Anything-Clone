# Rà soát DIODE: dữ liệu, nhãn và thực nghiệm 1

Ngày: 2026-11-26. Mục tiêu là kiểm tra lỗi có thể sửa để tái hiện Table 2, giữ nguyên bộ validation và các pixel hợp lệ trong baseline.

## Dữ liệu có đúng không?

**Có: toàn bộ file local giống từng byte với bản validation chính thức.** Đã tải lại archive từ [S3 của DIODE](https://diode-dataset.s3.amazonaws.com/val.tar.gz), dung lượng 2,774,625,282 byte; MD5 tính được `5c895d09201b88973c8fe4552a67dd85`, trùng mã [DIODE công bố](https://diode-dataset.org/).

Sau khi xác thực archive, đã tính SHA256 từng file bên trong và đối chiếu với local:

- 771 ảnh RGB, 771 depth map, 771 validity mask: **2.313/2.313 file trùng nội dung**.
- Không thiếu file hoặc có file khác nội dung. Evaluator kiểm tra thêm không có RGB/depth/mask thừa trong local.
- Đúng 325 `indoors` + 446 `outdoor`, phân cấp scene/scan đúng bản phát hành.
- Cặp nhãn lấy theo cùng đường dẫn và stem: `<stem>.png`, `<stem>_depth.npy`, `<stem>_depth_mask.npy`; không ghép theo thứ tự của ba danh sách độc lập.
- Ảnh 1024×768; depth squeeze về 768×1024; mask đúng shape. RGB cả 771 ảnh đã giải mã được trong full inference.

[Report SHA256 từng file](../results/diode-relative/audit/official-archive-verification.json). Việc này chứng minh đúng dữ liệu phát hành và đúng cặp file, không chứng minh mọi đo đạc LiDAR trong nhãn đều chính xác vật lý.

## Nhãn có khớp ảnh không?

Kiểm tra trực quan 6 mẫu chọn trước theo vị trí đầu/giữa/cuối mỗi miền cho thấy hình học depth và mask khớp vị trí vật thể trong RGB; không thấy đổi trục, lật ảnh hay ghép nhầm cảnh. Xem [bảng RGB–depth–mask–prediction](../results/diode-relative/audit/rgb-depth-mask-prediction.png).

Việc kiểm tra bằng checksum áp dụng cho toàn bộ 771 cặp; kiểm tra trực quan áp dụng cho 6 mẫu trên. Một số vùng depth outdoor có điểm rải rác bất thường dù validity mask bằng 1 (quan sát định tính). Chưa đủ bằng chứng xác định nguyên nhân vật lý hoặc đóng góp riêng của chúng vào chênh lệch với paper.

## Có đúng thực nghiệm 1 không?

Đúng loại thực nghiệm **zero-shot relative depth**: dùng checkpoint relative Depth Anything V1 ViT-L đã công bố; không train/fine-tune trên DIODE; căn chỉnh scale/shift từng ảnh vào `1/depth`, rồi chấm disparity theo [giải thích của tác giả](https://github.com/LiheYoung/Depth-Anything/issues/174). Căn chỉnh bằng GT là bước đánh giá relative depth, không phải fine-tune trọng số.

Đối chiếu mã hiện tại với upstream cho thấy AST của cả 3 lớp model `DPTHead`, `DPT_DINOv2`, `DepthAnything` và cả 3 lớp transform `Resize`, `NormalizeImage`, `PrepareForNet` trùng. Preprocessing giữ tỉ lệ, cạnh ngắn 518, bội 14, RGB/ImageNet normalization; output bilinear `align_corners=False`, phù hợp [run.py chính thức](https://github.com/LiheYoung/Depth-Anything/blob/main/run.py). Min-max trong script demo chỉ phục vụ ảnh hiển thị, không áp vào số liệu đánh giá.

[Đối chiếu source](../results/diode-relative/audit/upstream/comparison.json); [cây repo V1 tại commit `1d03336771fe09c5398ffdd211441e33941a97dc`](../results/diode-relative/audit/upstream/v1-tree.json). Cây mã đã kiểm tra có evaluator metric depth, chưa thấy evaluator DIODE relative chính thức. Phản hồi mới nhất trong [V1 #174](../results/diode-relative/audit/upstream/v1-174-comments.json) và [V2 #280](../results/diode-relative/audit/upstream/v2-280-comments.json) cũng chưa cung cấp đầy đủ manifest và protocol cần thiết.

**Chưa xác nhận được toàn bộ protocol tạo ra số Table 2.** Baseline local dùng native validation, supplied mask, không crop/depth cap, fit least squares float64 và chấm disparity thô. Split/manifest, clipping và TTA của paper vẫn chưa rõ. Vì vậy đúng loại thực nghiệm không đồng nghĩa đã tái hiện đúng từng chi tiết đánh giá của tác giả.

## Các cách chấm đã kiểm tra trên cùng đủ 771 ảnh

Các biến thể dưới đây giữ nguyên GT mask và không bỏ ảnh. Chúng dùng lại cùng prediction để tách ảnh hưởng scoring khỏi inference; chỉ baseline native là kết quả chính của evaluator.

| Cách chấm | AbsRel ↓ | δ₁ ↑ | Ý nghĩa |
|---|---:|---:|---|
| Native, affine disparity, không clip | 0.267872421 | 0.753644631 | Baseline hiện tại |
| Clamp disparity tối thiểu bằng 1/max(GT depth hợp lệ) mỗi ảnh | 0.266991524 | 0.754107364 | Chẩn đoán clipping |
| Sau clamp trên, nghịch đảo và chấm depth | 0.214822594 | 0.754107364 | Chẩn đoán metric space, khác hướng dẫn tác giả |
| Paper Table 2 ViT-L | 0.066 | 0.952 | Chưa có manifest/protocol đầy đủ để đối chiếu |

[Chi tiết từng miền và script tái chạy](../results/diode-relative/audit/scoring-diagnostics.json). Clipping thay đổi rất ít, không giải thích được khoảng cách. Kết quả depth-space xấp xỉ báo cáo độc lập trong [V2 #280](https://github.com/DepthAnything/Depth-Anything-V2/issues/280); đây là đối chứng cộng đồng, không phải xác nhận protocol từ tác giả.

## Kết quả suy luận mới sau xác thực dữ liệu

Đã chạy đủ **771/771 ảnh**, buộc forward mới toàn bộ: **cache hits = 0**, `refresh_cache = true`, dữ liệu được xác thực lại đủ 2.313 file trước khi chạy. Thời gian forward/scoring khoảng 9 phút 58 giây.

| Phạm vi | Số ảnh | AbsRel disparity ↓ | δ₁ ↑ |
|---|---:|---:|---:|
| Indoor | 325 | 0.057393579 | 0.955752289 |
| Outdoor | 446 | 0.421248258 | 0.606368871 |
| Toàn bộ | 771 | 0.267872421 | 0.753644631 |
| Paper Table 2 | Chưa có manifest để đối chiếu | 0.066 | 0.952 |

Đối chiếu file prediction mới với cache từ lượt đầu: **771/771 prediction giống từng byte**, metric trùng hoàn toàn. Vì vậy chênh lệch không do cache cũ hoặc một lượt forward không lặp lại được. Chưa tìm được lỗi triển khai có căn cứ để sửa nhằm đạt 0.066/0.952 trên native validation này.

- [Summary lượt suy luận mới](../results/diode-relative/audit-rerun/fresh-inference-summary.json).
- [CSV từng ảnh](../results/diode-relative/audit-rerun/vitl-native-all-per-image.csv), [manifest](../results/diode-relative/audit-rerun/vitl-native-all-manifest.json).
- [Đối chiếu prediction mới/cũ](../results/diode-relative/audit-rerun/fresh-vs-initial-verification.json).

## Flip TTA: kiểm tra trên 16 ảnh đã chọn từ trước

Cùng danh sách probe 8 indoor + 8 outdoor, chọn trước khi biết sai số; giữ nguyên mask và scoring. Đây **không phải full benchmark TTA**.

| Miền | No-flip AbsRel / δ₁ | Flip AbsRel / δ₁ |
|---|---:|---:|
| Indoor (8 ảnh) | 0.049719 / 0.961861 | 0.051488 / 0.962789 |
| Outdoor (8 ảnh) | 0.363920 / 0.476991 | 0.362962 / 0.477322 |

[Chi tiết probe flip](../results/diode-relative/audit/flip-probe.json). Chưa thấy dấu hiệu flip giải thích được khoảng cách lớn trên các mẫu này; không suy rộng thành kết quả TTA đủ 771 ảnh.

## Các sửa đổi

- `download_data.sh`: HTTPS, xác minh MD5 trước khi giải nén DIODE, sửa số ảnh outdoor thành 446 và tên thư mục `indoors`.
- `audit_diode.py`: xác thực archive và so sánh SHA256 từng file; hoặc xác minh local từ report đã tạo, không phải tải lại archive mỗi lần.
- Evaluator và script thực nghiệm 1: thêm `--data-audit` để dừng nếu dữ liệu khác bản chính thức; `--refresh-cache` để buộc forward mới toàn bộ ảnh. Summary ghi provenance và cache hits.
- Notebook DIODE dùng cùng bước xác minh dữ liệu, cache và output của lượt rerun.
- Sửa hai URL issue trước đó trỏ nhầm repository. V2 #172/#280 mới là các thảo luận DIODE đã viện dẫn.

28 unit tests đã qua; kiểm tra cú pháp Bash đã qua. Notebook đã chạy lại thành công, xác thực đủ 2.313 file, đọc 771 cache hits, không có cell lỗi và trả metric trùng hoàn toàn fresh CLI. Evaluator/kết quả NYUv2 không thay đổi. [Kết luận dạng JSON](../results/diode-relative/audit/conclusion.json).

## Cách kiểm tra và chạy lại

Từ repo root, môi trường `.venv` hiện tại:

```powershell
# Kiểm tra lại toàn bộ byte dữ liệu local từ report đã xác thực:
.venv\Scripts\python.exe audit_diode.py --reference-report results/diode-relative/audit/official-archive-verification.json

# Hoặc tạo lại report trực tiếp từ archive đã xác minh:
.venv\Scripts\python.exe audit_diode.py --archive .cache/diode-audit/official-val.tar.gz --output results/diode-relative/audit/official-archive-second-check.json

# Xác minh dữ liệu rồi buộc suy luận mới đủ 771 ảnh:
.venv\Scripts\python.exe evaluate_diode_relative.py --encoder vitl --data-audit results/diode-relative/audit/official-archive-verification.json --refresh-cache --cache-dir .cache/diode-audit/fresh-predictions --output-dir results/diode-relative/audit-rerun
```

Bỏ `--refresh-cache` để chấm lại từ cache hợp lệ. Report SHA256 được tạo từ archive có MD5 chính thức; nếu report chưa có, cần tạo bằng lệnh `--archive` trước. Archive 2.77 GB và prediction cache là artefact làm việc trong `.cache/`, không nằm trong Git.
