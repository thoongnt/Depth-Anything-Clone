# Tái hiện Table 2 — zero-shot relative depth (Depth Anything V1)

Chạy và rà soát ngày 2026-11-28. Mục tiêu: **Table 2** bản CVPR 2024, ba hàng
*Depth Anything* ViT-S / ViT-B / ViT-L. Trong `data.md` bảng này được gọi là "Bảng 1".
Trong PDF, Table 1 chỉ là danh sách dataset huấn luyện, không có thực nghiệm.

Phạm vi gồm KITTI, NYUv2, Sintel, ETH3D và DIODE. **DDAD được bỏ qua**, giống
HyperSim ở Table 5: DDAD chỉ phát hành một file tar gộp train+val khoảng 257 GB.

- Checkpoint: relative `checkpoints/depth_anything_vit{s,b,l}14.pth`, không fine-tune.
- Mỗi lần thử chỉ đổi **một** yếu tố. Mọi cấu hình đều được báo cáo. Cấu hình nào
  được chọn vì gần số paper thì ghi rõ là **giả thuyết** (dấu \*).

## Kết quả

Ngưỡng giống [Table 5](table5-reproduction.md), so sánh trên giá trị làm tròn 3 chữ số,
với d = max(|ΔAbsRel|, |Δδ₁|): **KHỚP** khi d = 0, **ĐẠT** khi d ≤ 0.005, còn lại
**KHÔNG ĐẠT**.

Mỗi dataset dùng **một protocol chung cho cả 3 encoder**. Chọn như vậy khó "ăn may"
hơn so với chọn riêng cấu hình tốt nhất cho từng ô.

| Dataset | Protocol | ViT-S | ViT-B | ViT-L | Kết luận |
|---|---|---|---|---|---|
| **NYUv2** | `rawDepths`, mask 0 < d < 10 m toàn ảnh, không flip | 0.053 / 0.972 **KHỚP** | 0.046 / 0.979 **KHỚP** | 0.044 / 0.980 ĐẠT | **ĐẠT** |
| **KITTI** | Eigen test 652 ảnh, depth annotated, **toàn ảnh (không crop)**, cap 80 m\* | 0.080 / 0.936 **KHỚP** | 0.080 / 0.939 **KHỚP** | 0.076 / 0.946 ĐẠT | **ĐẠT** |
| **DIODE** | gói đánh giá của MoGe (771 ảnh), không cap\* | 0.075 / 0.943 ĐẠT | 0.069 / 0.949 ĐẠT | 0.065 / 0.954 ĐẠT | **ĐẠT** |
| **Sintel** | gần nhất chung: pass final, GT kẹp ở 175 m, kẹp disparity, đủ 1064 ảnh (chỉnh để khớp)\* | 0.473 / 0.718 | 0.453 / 0.749 | 0.427 / 0.766 | **KHÔNG ĐẠT** (B ĐẠT nếu chọn cap riêng) |
| **ETH3D** | gần nhất chung: GT gốc 6048×4032 thu nhỏ bilinear về kích thước mạng, ngưỡng mask 0,642 (chỉnh để khớp)\* | 0.133 / 0.864 | 0.128 / 0.876 | 0.125 / 0.886 ĐẠT | **KHÔNG ĐẠT** (S, L ĐẠT nếu chọn ngưỡng riêng) |
| *Paper* | | KITTI 0.080 / 0.936 · NYU 0.053 / 0.972 · Sintel 0.464 / 0.739 · ETH3D 0.127 / 0.885 · DIODE 0.076 / 0.939 | KITTI 0.080 / 0.939 · NYU 0.046 / 0.979 · Sintel 0.432 / 0.756 · ETH3D 0.126 / 0.884 · DIODE 0.069 / 0.946 | KITTI 0.076 / 0.947 · NYU 0.043 / 0.981 · Sintel 0.458 / 0.760 · ETH3D 0.127 / 0.882 · DIODE 0.066 / 0.952 | |

Tóm lại:

- **3/5 dataset ĐẠT** với một protocol chung cho cả 3 encoder: NYUv2, KITTI và DIODE.
  Trong đó có **6 ô KHỚP** ở 3 chữ số.
- **Sintel và ETH3D chưa tìm được protocol chung.** Khi chỉnh một tham số riêng cho
  từng encoder (cap của Sintel, ngưỡng mask của ETH3D), có thêm 3 ô ĐẠT: Sintel ViT-B,
  ETH3D ViT-S và ETH3D ViT-L. Đây chỉ là bằng chứng yếu; xem các mục 4 và 5.
- **Tổng theo ô:** 12/15 ô ĐẠT nếu tính cả các ô có tham số được chỉnh; 9/15 nếu chỉ
  tính protocol chung.

Ở KITTI, nếu kẹp disparity ở 1/cap thì ViT-L ra 0.076 / 0.947, **KHỚP**, nhưng ViT-S/B
khi đó chỉ ĐẠT (0.079 / 0.937, 0.079 / 0.940). Hai cách đều đạt cả 3 encoder.

Bảng đầy đủ mọi cấu hình nằm ở `results/table2/summary.md`, sinh bởi
`scripts/collect_table2.py`.

## Protocol chung (có nguồn xác nhận)

Nguồn: [issue #174](https://github.com/LiheYoung/Depth-Anything/issues/174),
[phụ lục](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Yang_Depth_Anything_Unleashing_CVPR_2024_supplemental.pdf),
và [mã đánh giá MiDaS](https://gist.github.com/ranftlr/a1c7a24ebb24ce0e2f2ace5bce917022).

1. RGB float32 / 255. Resize cạnh ngắn 518, giữ tỉ lệ, mỗi cạnh làm tròn về bội của 14,
   nội suy cubic, normalize theo ImageNet.
2. Dự đoán disparity, nội suy bilinear (`align_corners=False`) về lưới GT.
3. Mỗi ảnh fit scale + shift bằng least squares float64 vào `1/GT`.
4. AbsRel và δ₁ tính **trong không gian disparity**, lấy trung bình theo ảnh.

Cùng một pipeline này được dùng cho mọi dataset: `evaluate_nyuv2.py` cho NYUv2 và
`scripts/eval_relative.py` cho các bộ còn lại. Phần thay đổi giữa các dataset chỉ là
nguồn GT, mask, crop và depth cap.

## 1. NYUv2 — ĐẠT (ViT-S, ViT-B KHỚP)

| Encoder | Kết quả (4 chữ số) | Paper | Kết luận |
|---|---|---|---|
| ViT-S | 0.0532 / 0.9717 | 0.053 / 0.972 | KHỚP |
| ViT-B | 0.0460 / 0.9792 | 0.046 / 0.979 | KHỚP |
| ViT-L | 0.0436 / 0.9803 | 0.043 / 0.981 | ĐẠT |

Protocol theo mã MiDaS:

- GT là `rawDepths`, không phải `depths` đã inpaint.
- Mask 0 < depth < 10 m trên toàn ảnh, không Eigen crop.
- Disparity sau khi căn chỉnh được kẹp ở 0.1 (tức depth cap 10 m); không flip.

ViT-S/B chạy ngày 2026-11-28 trên máy này; ViT-L là kết quả của nhóm.
Chi tiết: [table2/nyuv2-reproduction.md](table2/nyuv2-reproduction.md).

## 2. KITTI — ĐẠT (ViT-S, ViT-B KHỚP)

Quét 3 nguồn GT × 3 kiểu crop × 2 cap × 2 cách kẹp disparity. Các dòng chính:

| Cấu hình (cap 80, không kẹp disparity) | ViT-S | ViT-B | ViT-L |
|---|---|---|---|
| **annotated, toàn ảnh** ★ | 0.080 / 0.936 | 0.080 / 0.939 | 0.076 / 0.946 |
| annotated, eigen crop | 0.083 / 0.932 | 0.081 / 0.937 | 0.078 / 0.944 |
| annotated, garg crop | 0.074 / 0.945 | 0.074 / 0.948 | 0.071 / 0.955 |
| gói MoGe, toàn ảnh | 0.079 / 0.936 | — | — |
| LiDAR thô chiếu lên ảnh (697 ảnh), toàn ảnh | 0.117 / 0.882 | — | — |
| *Paper* | 0.080 / 0.936 | 0.080 / 0.939 | 0.076 / 0.947 |

Kết luận:

- **Nguồn GT:** dùng depth annotated. LiDAR thô sai số lớn hơn nhiều, KHÔNG ĐẠT.
- **Crop:** dùng toàn ảnh. Garg crop cho kết quả *tốt hơn* paper (KHÔNG ĐẠT).
  Mã KITTI của MiDaS (`validate_kitti.py`) cũng không crop.
- **Cap:** 80 m hay không cap đều cho cùng kết quả, vì GT annotated tối đa khoảng 85 m.

## 3. DIODE — ĐẠT với gói MoGe

| Nguồn (không cap, không kẹp disparity) | ViT-S | ViT-B | ViT-L |
|---|---|---|---|
| mask gốc DIODE | 0.274 / 0.745 | 0.271 / 0.750 | 0.268 / 0.754 |
| giao mask gốc với mask MoGe | 0.074 / 0.944 | 0.067 / 0.951 | 0.063 / 0.955 |
| **toàn bộ gói MoGe** ★ | 0.075 / 0.943 | 0.069 / 0.949 | 0.065 / 0.954 |
| *Paper* | 0.076 / 0.939 | 0.069 / 0.946 | 0.066 / 0.952 |

- Lượt kiểm tra độc lập bằng `eval_relative.py` (ViT-S) **trùng đến chữ số thứ 4** với
  kết quả của nhóm ở hai dòng đầu (0.2737 / 0.7453 và 0.0738 / 0.9444).
- Dùng nguyên gói MoGe (ảnh, depth và mask) cho độ lệch lớn nhất 0.004, tốt hơn mức
  0.005 của cấu hình mask giao.
- Gói MoGe lọc lỗi ở biên và điểm xa. Mask gốc của DIODE giữ cả các pixel đó, nên
  outdoor bị lệch nặng. Chưa có nguồn nào cho thấy tác giả dùng gói này, nên đây vẫn
  là giả thuyết (\*).

Chi tiết lịch sử:

- [table2/diode-relative-depth.md](table2/diode-relative-depth.md)
- [table2/diode-audit.md](table2/diode-audit.md)
- [table2/diode-variants.md](table2/diode-variants.md)
- [table2/diode-followup.md](table2/diode-followup.md)

## 4. Sintel — KHÔNG ĐẠT

Dữ liệu: 1064 khung hình training, RGB lấy từ `MPI-Sintel-training_images.zip`, depth
`.dpt` lấy từ `MPI-Sintel-depth-training-20150305.zip`.

Đặc điểm quan trọng của GT Sintel: bầu trời có depth tới 1e11 m. Khoảng 14% pixel
xa hơn 72 m. Riêng cảnh `mountain_1` có 19 khung hình **không có pixel nào gần hơn 80 m**.

| Cấu hình (pass final) | ViT-S | ViT-B | ViT-L |
|---|---|---|---|
| không cap (giữ cả bầu trời) | AbsRel ~7×10⁵ | ~8×10⁵ | ~7×10⁵ |
| cap 80, **loại** pixel xa (bỏ 19 ảnh) | 0.340 / 0.731 | 0.338 / 0.757 | 0.300 / 0.781 |
| cap 72, **kẹp GT** ở 72 m (đủ 1064 ảnh) | 0.432 / 0.720 | 0.427 / 0.744 | 0.381 / 0.768 |
| cap 80, **kẹp GT** ở 80 m | 0.450 / 0.718 | 0.444 / 0.742 | 0.397 / 0.765 |
| cap 80, kẹp GT, kẹp disparity | 0.371 / 0.735 | 0.359 / 0.764 | 0.332 / 0.782 |
| cap 200, kẹp GT, kẹp disparity | 0.496 / 0.715 | 0.474 / 0.746 | 0.449 / 0.763 |
| pass clean, cap 80, loại pixel xa | 0.364 / 0.724 | — | — |
| gói MoGe, không cap, kẹp disparity | 0.523 / 0.698 | — | — |
| *Paper* | 0.464 / 0.739 | 0.432 / 0.756 | 0.458 / 0.760 |

**Quét cap mịn** (60–300 m × loại / kẹp GT × kẹp disparity, cap được **chỉnh để khớp**;
`scripts/sintel_cap_sweep.py`, `results/table2/sintel/cap-sweep.json`):

| Cấu hình | ViT-S | ViT-B | ViT-L |
|---|---|---|---|
| **chung tốt nhất**: kẹp GT 175 m + kẹp disparity (lệch tối đa 0.031) | 0.473 / 0.718 | 0.453 / 0.749 | 0.427 / 0.766 |
| tốt nhất riêng ViT-S: kẹp GT 150 m + kẹp disparity | 0.450 / 0.723 | — | — |
| tốt nhất riêng ViT-B: kẹp GT 150 m + kẹp disparity | — | **0.432 / 0.752 ĐẠT** | — |
| tốt nhất riêng ViT-L: kẹp GT 120 m | — | — | 0.466 / 0.754 (lệch 0.008) |
| *Paper* | 0.464 / 0.739 | 0.432 / 0.756 | 0.458 / 0.760 |

Nhận xét:

- Cách xử lý điểm xa **quyết định AbsRel**: kẹp GT ở 120–175 m đưa AbsRel vào vùng của
  paper.
- Chỉ ViT-B **ĐẠT** khi chọn cap riêng (150 m). Không có cap chung cho cả 3 encoder.
- ViT-S không bao giờ đạt δ₁ của paper (cao nhất 0.742 trên mọi cấu hình, trong khi
  AbsRel đã vượt xa). Phần thiếu ở ViT-S vì vậy không nằm ở cách xử lý điểm xa.
- Thứ tự AbsRel của chúng ta luôn giảm dần từ S sang L. Thứ tự lạ của paper (ViT-B
  thấp nhất, ViT-L gần bằng ViT-S) chưa tái tạo được.

Những gì còn thiếu có lẽ nằm ở chi tiết GT, mask hoặc tiền xử lý mà paper không công bố.

## 5. ETH3D — KHÔNG ĐẠT

Paper ETH3D gần như không đổi qua S/B/L: 0.127 / 0.126 / 0.127. Điều này gợi ý một
**sàn sai số không phụ thuộc model**. Kết quả của chúng ta tốt hơn khoảng 2,5 lần ở
mọi cấu hình đã thử:

| Giả thuyết đã kiểm tra | Kết quả ViT-S | Kết luận |
|---|---|---|
| gói MoGe, không cap (protocol khớp DIODE) | 0.062 / 0.963 | quá tốt |
| depth cap 20 / 50 / 72 m, loại hoặc kẹp GT | 0.058 – 0.062 / 0.963 – 0.967 | quá tốt |
| chấm trên lưới thu nhỏ 1/2, 1/4, GT nearest | 0.061 / 0.964 | quá tốt |
| chấm trên lưới thu nhỏ, GT bilinear lẫn cả lỗ | 0.38 – 0.40 / 0.45 – 0.50 | quá tệ |
| ghép ảnh đã khử méo với depth của ảnh gốc có méo (cảnh `pipes`, 14 ảnh) | 0.048 so với 0.049 khi ghép đúng | không ảnh hưởng |
| nguồn chính thức stride 3, fit median–MAD, flip (lượt của nhóm, ViT-L) | 0.046 – 0.052 | quá tốt |
| **GT thu nhỏ về kích thước đầu ra mạng (518×784), depth bilinear, giữ pixel có mask nội suy ≥ 0,5** | S 0.098 / 0.903, B 0.091 / 0.910, L 0.087 / 0.917 | **gần nhất**, nhưng vẫn KHÔNG ĐẠT |
| như trên, chỉ giữ mask nội suy = 1 / GT nearest | 0.046 – 0.062 | quá tốt |
| như trên, mask nearest (giữ cả pixel bị trộn với lỗ) | 1.61 / 0.26 | quá tệ |

Kết quả ViT-L trên gói MoGe (0.0475 / 0.9798) **trùng đến chữ số thứ 4** với lượt của
nhóm, nên pipeline nhất quán; khác biệt nằm ở protocol của paper.

Cấu hình gần nhất ("mask ≥ 0,5") tái tạo được **một phần** hiện tượng của paper: sai số
tăng lên và chênh lệch giữa các encoder thu hẹp (0.087 – 0.098). Tuy vậy vẫn lệch
0.03 – 0.04 so với paper. Script: `scripts/eth3d_resize_hypotheses.py`; kết quả:
`results/table2/eth3d/resize-hypotheses-*.json`.

**Thu nhỏ từ GT gốc 6048×4032.** Dùng depth chính thức của cả 13 cảnh, 5,8 GB, đọc
thẳng từ archive. GT gốc chiếu từ laser nên thưa, có nhiều lỗ. Thu nhỏ về kích thước
mạng (518×784):

| Cách thu nhỏ GT gốc | ViT-S | ViT-B | ViT-L |
|---|---|---|---|
| nearest | 0.061 / 0.965 | 0.052 / 0.974 | 0.047 / 0.980 |
| **area (trộn cả lỗ), mask ≥ 0,5** (300 / 454 ảnh còn pixel) | **0.135 / 0.800** | **0.133 / 0.804** | **0.130 / 0.810** |
| bilinear (trộn cả lỗ), mask ≥ 0,5 | 0.203 / 0.620 | 0.199 / 0.624 | 0.196 / 0.628 |
| area có trọng số mask (không trộn lỗ) | 0.038 / 0.983 | 0.032 / 0.988 | 0.029 / 0.991 |
| *Paper* | 0.127 / 0.885 | 0.126 / 0.884 | 0.127 / 0.882 |

Kết luận về ETH3D:

- Thu nhỏ GT gốc thưa *có trộn lỗ* tạo ra đúng **sàn sai số không phụ thuộc encoder**
  như paper. Ba encoder chênh nhau dưới 0.007, trong khi với nearest là 0.014.
- Với cách "area", **AbsRel về đúng mức paper** (0.130 – 0.135), nhưng δ₁ thấp hơn
  khoảng 0.08 và một phần ảnh bị bỏ.
- Vậy **cơ chế** gây ra con số ETH3D của paper rất có thể là ở đây: xử lý GT gốc thưa
  khi thu nhỏ. Nhưng **quy tắc chính xác** (kiểu nội suy, ngưỡng mask, độ phân giải
  chấm) chưa xác định được, nên ETH3D vẫn KHÔNG ĐẠT.

Script: `scripts/eth3d_fullres_gt.py`; kết quả:
`results/table2/eth3d/fullres-gt-hypotheses.{json,txt}`.

**Quét ngưỡng mask (bilinear, GT gốc, đủ 454 ảnh).** Ngưỡng mask được **chỉnh để khớp
paper**, theo đúng cách tìm cấu hình gần nhất đã làm ở Table 5:

| Ngưỡng mask nội suy | ViT-S | ViT-B | ViT-L |
|---|---|---|---|
| ≥ 0,60 | 0.150 / 0.808 | 0.146 / 0.820 | 0.143 / 0.830 |
| ≥ 0,642 | 0.133 / 0.864 | 0.129 / 0.876 | **0.125 / 0.886 ĐẠT** |
| ≥ 0,643 | **0.127 / 0.882 ĐẠT** | 0.122 / 0.894 | 0.119 / 0.904 |
| ≥ 0,70 | 0.110 / 0.922 | 0.105 / 0.933 | 0.102 / 0.944 |
| ≥ 0,90 | 0.063 / 0.967 | 0.056 / 0.974 | 0.051 / 0.981 |
| *Paper* | 0.127 / 0.885 | 0.126 / 0.884 | 0.127 / 0.882 |

- Khi ngưỡng thay đổi, (AbsRel, δ₁) chạy trên một đường cong. Đường cong này **đi qua
  điểm của paper ở cả 3 encoder**. Điều đó củng cố giả thuyết về cơ chế: GT gốc thưa
  bị thu nhỏ bằng bilinear, có trộn lỗ.
- ViT-S và ViT-L **ĐẠT** khi chọn ngưỡng riêng (0,643 và 0,642). ViT-B nằm đúng chỗ
  mask nhảy bậc nên lệch tối thiểu 0.008.
- **Không có ngưỡng chung** cho cả 3 encoder. Ở cùng một ngưỡng, ba encoder chênh
  nhau khoảng 0.008 AbsRel, còn ba số của paper gần như bằng nhau (chênh 0.001).
  Vì vậy quy tắc chính xác của tác giả vẫn khác quy tắc này ở một chi tiết nào đó.
- Ngưỡng 0,64 không phải một lựa chọn "tự nhiên" trong code. Các ô ĐẠT ở đây là kết
  quả chỉnh một tham số, chỉ là bằng chứng yếu.

Kết quả: `results/table2/eth3d/fullres-gt-sweep{,-fine,-fine2}.json`.

Paper Depth Anything V2 (Table 2) cũng báo ETH3D quanh 0.13 cho mọi encoder (0.131 –
0.142), và MiDaS v3.1 là 0.139. Một sàn sai số chung cho nhiều model khác nhau như
vậy càng cho thấy nguyên nhân nằm ở cách xử lý GT của tác giả, không ở model. Paper
V2 không mô tả protocol đánh giá.

**Kết luận chốt (2026-11-28):** nhóm quyết định **dừng ở đây và chấp nhận kết quả**.
Sintel và ETH3D được ghi là **KHÔNG ĐẠT theo protocol chung**. Kèm theo đó là phân
tích cơ chế và 3 ô ĐẠT khi chỉnh tham số riêng cho từng encoder (đánh dấu \*).

**Hướng làm tiếp nếu cần (chưa chạy):** hiệu chỉnh bằng **hàng MiDaS v3.1**
(DPT-BEiT-L-512) của Table 2 (ETH3D 0.139 / 0.867, Sintel 0.587 / 0.699). Quy tắc nào
cho MiDaS đúng hai số này thì đó là protocol của tác giả.

- Đã chuẩn bị sẵn: trọng số `checkpoints/midas/dpt_beit_large_512.pt` và tùy chọn
  `--encoder midas` trong `eval_relative.py`.
- Bước này cần **chạy mã nguồn repo MiDaS tải từ GitHub** (`.cache/midas_repo/`) và
  `timm==0.6.12` (môi trường hiện có `timm` 1.0.29), nên phải được cho phép riêng.

Chi tiết lượt của nhóm: [table2/eth3d-relative-depth.md](table2/eth3d-relative-depth.md).
Thử nghiệm lệch méo: `scripts/eth3d_distortion_pilot.py` và
`results/table2/eth3d/pilot-distortion-pipes-vits.txt`.

## Khó khăn đã gặp

| Khó khăn | Dấu hiệu | Cách xử lý |
|---|---|---|
| Không có evaluator relative depth chính thức. | Mỗi dataset phải tự đoán mask, crop, cap. | Một pipeline chung, quét lưới cấu hình; chọn protocol khớp cả 3 encoder. |
| Chạy thử 3 ảnh KITTI cho kết quả lệch so với khi chạy đủ. | 3 ảnh đầu ra khoảng 0.06; chạy đủ 652 ảnh ra 0.080. | Luôn kết luận trên toàn bộ tập. |
| Sintel: bầu trời depth 1e11 m. | Không cap thì AbsRel lên tới 10⁵. | Thử cap, kẹp GT, kẹp disparity. |
| Sintel: 19 khung `mountain_1` không còn pixel nào dưới cap. | Cấu hình loại pixel xa chỉ chấm được 1045 ảnh. | Ghi rõ số ảnh bị bỏ; thử thêm chế độ kẹp GT (đủ 1064 ảnh). |
| Script tổng hợp từng loại bỏ cấu hình có ảnh bị bỏ. | Bảng chọn nhầm cấu hình không cap (AbsRel 10⁵). | Giữ lại và đánh dấu số ảnh bị bỏ. |
| Link depth Sintel chính thức bị 404. | `files.is.tue.mpg.de/sintel/...` | Dùng `files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip`. |
| Depth ETH3D chính thức chỉ phát hành theo từng cảnh, mỗi ảnh 97 MB khi giải nén. | Cả bộ khoảng 44 GB, vượt dung lượng đĩa. | Đọc thẳng từ `.7z` bằng `C:\Windows\System32\tar.exe`; chỉ thử nghiệm trên 1 cảnh. |
| Trong Git Bash, `tar.exe` là GNU tar. | Không đọc được `.7z`. | Gọi đúng đường dẫn bsdtar của Windows. |
| DINOv2 bản trong repo dùng xformers, không chạy được trên CPU. | `NotImplementedError: memory_efficient_attention`. | Chặn import: `sys.modules["xformers"] = None`. |
| Chạy song song nhiều job trên GPU 4 GB. | Nguy cơ hết bộ nhớ. | Xếp hàng đợi tuần tự; phần chấm lại từ cache chạy trên CPU. |
| Checkpoint ViT-S/B chưa có trên máy. | Chỉ có ViT-L trong `checkpoints/`. | Tải từ Hugging Face (99 MB và 390 MB). |

## Dữ liệu

| Dataset | Nguồn | Vị trí |
|---|---|---|
| NYUv2 | `nyu_depth_v2_labeled.mat`, `splits.mat` | `data/nyuv2/` |
| KITTI | raw drives + `data_depth_annotated` (Eigen test) | `data/kitti/raw`, `data/kitti/gt/{train,val}` |
| Sintel | [`MPI-Sintel-training_images.zip`](http://sintel.is.tue.mpg.de/), [`MPI-Sintel-depth-training-20150305.zip`](http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip) | `data/Sintel/`, `data/sintel_depth/` |
| ETH3D, KITTI, Sintel, DIODE (gói MoGe) | [Ruicheng/monocular-geometry-evaluation](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation) | `data/moge/{ETH3D,KITTI,Sintel,DIODE}.zip` |
| DIODE gốc | `val.tar.gz` | `data/diode/val/` |
| ETH3D chính thức (thử nghiệm cảnh `pipes`) | `pipes_dslr_depth.7z`, `pipes_dslr_jpg.7z` từ eth3d.net | `data/eth3d_official/` |

Các file zip được đọc trực tiếp, không cần giải nén.

## Cấu trúc thư mục

```text
docs/table2-reproduction.md      tài liệu này
docs/table2/                     tài liệu chi tiết các lượt trước của nhóm (NYUv2, DIODE, ETH3D)
results/table2/
  summary.md, summary.json       mọi cấu hình, kết luận, protocol chung (collect_table2.py)
  nyuv2/                         vit{s,b,l}-midas-summary.json, ...
  kitti/  sintel/  eth3d/  diode/
    <source>-<encoder>[-ext]-grid.json        metric của mọi cấu hình
    <source>-<encoder>[-ext]-per-image.npz    metric từng ảnh
    <source>-<encoder>[-ext].txt              log
scripts/eval_relative.py         evaluator chung (KITTI, Sintel, ETH3D, DIODE)
scripts/collect_table2.py        tổng hợp và chấm kết luận
scripts/eth3d_distortion_pilot.py   thử nghiệm lệch méo (cảnh pipes)
scripts/eth3d_resize_hypotheses.py  giả thuyết thu nhỏ GT (gói MoGe) về kích thước mạng (dùng cache)
scripts/eth3d_fullres_gt.py         giả thuyết thu nhỏ GT gốc 6048×4032 (data/eth3d_official/*_dslr_depth.7z), --sweep
scripts/sintel_cap_sweep.py         quét cap mịn cho Sintel (dùng cache)
.cache/table2/                   dự đoán float32 (--cache), không commit
```

## Chạy lại

```powershell
# NYUv2 (654 ảnh)
.venv\Scripts\python.exe evaluate_nyuv2.py --encoder vits      # vitb, vitl
# KITTI / DIODE / ETH3D / Sintel: một lượt inference, chấm cả lưới cấu hình
.venv\Scripts\python.exe scripts\eval_relative.py --dataset kitti  --source annotated --encoder vitl
.venv\Scripts\python.exe scripts\eval_relative.py --dataset diode  --source moge      --encoder vitl
.venv\Scripts\python.exe scripts\eval_relative.py --dataset eth3d  --source moge      --encoder vitl
.venv\Scripts\python.exe scripts\eval_relative.py --dataset sintel --source final     --encoder vitl --cache
# Lưới mở rộng (cap, kẹp GT, lưới thu nhỏ), chấm lại từ cache, không cần GPU
.venv\Scripts\python.exe scripts\eval_relative.py --dataset sintel --source final --encoder vitl --grid extended --cache
# Tổng hợp
.venv\Scripts\python.exe scripts\collect_table2.py results\table2 --tol 0.005
```

Thời gian trên RTX 3050 Laptop 4 GB:

| Dataset | ViT-S | ViT-L |
|---|---|---|
| KITTI | ~6 phút | ~29 phút (ảnh rộng, đầu vào 518×1715) |
| Sintel | ~8 phút | ~25 phút |
| DIODE | ~5 phút | ~12 phút |
| ETH3D | ~8 phút | ~16 phút |
