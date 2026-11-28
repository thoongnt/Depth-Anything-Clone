> **Cập nhật Table 5 — zero-shot metric depth (2026-11-28):** Xem [protocol và kết quả](table5-reproduction.md). SUN RGB-D, iBims-1 và DIODE Outdoor trùng paper ở 3 chữ số; vKITTI2 đạt 0.085 / 0.911 (seed 0), số paper nằm trong dao động theo split. SUN RGB-D của paper dùng GT bằng 0.8 × độ sâu thật.

> **Cập nhật Table 2 — zero-shot relative depth (2026-11-28):** Xem [kết quả và protocol](table2-reproduction.md): NYUv2, KITTI, DIODE ĐẠT với một protocol chung cho 3 encoder; Sintel và ETH3D chưa tìm ra protocol của paper. Table 3 và Table 4 (metric in-domain) khớp 6/6 chỉ số: [Table 3](table3-reproduction.md), [Table 4](table4-reproduction.md). "Bảng 1" ở mục 2 bên dưới chính là **Table 2** trong PDF CVPR (Table 1 của PDF là danh sách dataset huấn luyện). Kết quả và tài liệu chi tiết đã chuyển vào `results/table2/` và `docs/table2/`. Đã sửa ViT-S Sintel AbsRel thành 0.464 theo PDF (trước đây ghi nhầm 0.432).

> **Rà soát DIODE (2026-11-26):** Xem [protocol và probe DIODE](table2/diode-relative-depth.md). Validation local có 325 indoor + 446 outdoor; split/protocol chính xác của số DIODE trong Table 2 chưa được xác nhận đầy đủ.

> **Cập nhật NYUv2 (2026-11-26):** Xem [protocol và kết quả đã chạy](table2/nyuv2-reproduction.md). Benchmark relative depth là Table 2 trong PDF CVPR; dùng rawDepths, mask MiDaS và metric disparity. Một số mô tả bên dưới là ghi chú cũ, không thay thế protocol đã kiểm chứng.

# Tìm hiểu về dataset, các bảng đánh giá hiện tại và đưa ra các phương án đề xuất để chạy thực nghiệm tối ưu trên bộ dữ liệu nhỏ

> Nguồn tham khảo: Yang et al., *Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data*, CVPR 2024.

## 1. Bộ dữ liệu dùng để huấn luyện (Training Datasets)

*(Phần tham khảo)*

Mô hình kết hợp huấn luyện trên 1,5 triệu ảnh có nhãn và hơn 62 triệu ảnh không nhãn.

### 6 bộ dữ liệu có nhãn (1,5M ảnh)

| Dataset | Số ảnh | Phương pháp gán nhãn |
|---|---|---|
| BlendedMVS | 115K | Stereo |
| DIML | 927K | Stereo |
| HRWSI | 20K | Stereo |
| IRS | 103K | Stereo |
| MegaDepth | 128K | Structure-from-Motion (SfM) |
| TartanAir | 306K | Stereo |

### 8 bộ dữ liệu không nhãn quy mô lớn (62M+ ảnh)

| Dataset | Số ảnh | Mô tả miền dữ liệu |
|---|---|---|
| BDD100K | 8,2M | Cảnh lái xe/giao thông |
| Google Landmarks | 4,1M | Địa danh/công trình |
| ImageNet-21K | 13,1M | Vật thể/hình ảnh tổng hợp |
| LSUN | 9,8M | Cảnh trong nhà và ngoài trời |
| Objects365 | 1,7M | Phát hiện vật thể |
| Open Images V7 | 7,8M | Hình ảnh tự nhiên đa dạng |
| Places365 | 6,5M | Nhận dạng khung cảnh |
| SA-1B | 11,1M | Dữ liệu phân đoạn quy mô lớn |

---

## 2. Bộ dữ liệu đánh giá về độ sâu tương đối

Kiểm tra khả năng tổng quát hóa trên môi trường hoàn toàn mới.

### Đánh giá trên những bộ dữ liệu nào

Sử dụng 6 bộ dữ liệu:

1. **KITTI**: Cảnh giao thông đô thị ngoài trời. Cảnh lái xe đô thị được trích xuất từ các đoạn video do xe tự lái ghi lại.
2. **NYUv2**: Cảnh môi trường trong nhà. Các khung hình RGB-D trích ra từ video ngắn quay cảnh không gian trong nhà bằng cảm biến Microsoft Kinect.
3. **Sintel**: Cảnh hoạt hình 3D phức tạp. Khung hình trích từ phim hoạt hình 3D đồ họa vi tính.
4. **DDAD**: Cảnh tự lái đô thị đa góc nhìn. Khung hình trích từ chuỗi video đa camera của hệ thống tự lái.
5. **ETH3D**: Khung cảnh trong nhà và ngoài trời có độ phân giải cao. Ảnh chụp tĩnh và khung hình video độ phân giải cao thu bằng máy ảnh DSLR và máy quét laser.
6. **DIODE**: Tập dữ liệu độ sâu dày đặc cho cả indoor và outdoor. Ảnh tĩnh indoor/outdoor quét bằng cảm biến LiDAR chuyên dụng.

### Độ sâu tương đối là gì? Đánh giá trên độ sâu tương đối cụ thể là đánh giá như nào?

Ước lượng độ sâu tương đối là việc thuật toán dự đoán được thứ tự gần xa của một đối tượng trong một bức ảnh. Mô hình AI sẽ xuất ra một bản đồ độ sâu chỉ ra vật thể nào gần máy ảnh hơn, vật thể nào xa máy ảnh hơn, và không cần cung cấp khoảng cách vật lý chính xác.

- Màu càng sáng (gần trắng): Vật thể càng gần máy ảnh.
- Màu càng tối (gần đen): Vật thể càng xa máy ảnh.

Cách đánh giá đang được sử dụng trong bài: Vì mỗi tập dữ liệu thử nghiệm có thang đo và độ lệch khoảng cách khác nhau, bài báo áp dụng phương pháp **Căn chỉnh Affine** (affine-invariant alignment). Giá trị dự đoán d̂ sẽ được căn chỉnh theo trung vị (median) và độ lệch chuẩn của nhãn thực d trước khi so sánh. (Các tập dữ liệu khác nhau thì sẽ có cách đo lường khác nhau — cái dùng mét, cái dùng đơn vị vô hướng quy đổi từ LiDAR, v.v. — khi qua mô hình dự đoán thì kết quả là d̂, cần dùng một phép biến đổi tuyến tính để đưa nó về cùng một hệ quy chiếu của d để đánh giá cho chính xác.)

### Kết quả

**Bảng 1. Đánh giá zero-shot độ sâu tương đối (Zero-shot relative depth estimation).** AbsRel càng thấp càng tốt (↓), δ₁ càng cao càng tốt (↑).

| Method | Encoder | KITTI AbsRel↓ | KITTI δ₁↑ | NYUv2 AbsRel↓ | NYUv2 δ₁↑ | Sintel AbsRel↓ | Sintel δ₁↑ | DDAD AbsRel↓ | DDAD δ₁↑ | ETH3D AbsRel↓ | ETH3D δ₁↑ | DIODE AbsRel↓ | DIODE δ₁↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MiDaS v3.1 | ViT-L | 0.127 | 0.850 | 0.048 | 0.980 | 0.587 | 0.699 | 0.251 | 0.766 | 0.139 | 0.867 | 0.075 | 0.942 |
| Depth Anything | ViT-S | 0.080 | 0.936 | 0.053 | 0.972 | 0.464 | 0.739 | 0.247 | 0.768 | 0.127 | 0.885 | 0.076 | 0.939 |
| Depth Anything | ViT-B | 0.080 | 0.939 | 0.046 | 0.979 | 0.432 | 0.756 | 0.232 | 0.786 | 0.126 | 0.884 | 0.069 | 0.946 |
| Depth Anything | ViT-L | **0.076** | **0.947** | **0.043** | **0.981** | 0.458 | 0.760 | **0.230** | **0.789** | 0.127 | 0.882 | **0.066** | **0.952** |

*So sánh với mô hình DPT-BEiT_L-512 tốt nhất của MiDaS v3.1 (mô hình này dùng nhiều dữ liệu có nhãn hơn Depth Anything). Ba quy mô mô hình được cung cấp: ViT-S (24,8M tham số), ViT-B (97,5M), ViT-L (335,3M).*

### Nhận xét

Thứ tự dưới đây không phải để so sánh mô hình nào giỏi hơn, mà dùng để xem **môi trường nào dễ "nuốt" nhất** và **môi trường nào làm khó mô hình nhất** (dựa vào việc sai số AbsRel thấp nhất và độ chính xác δ₁ cao nhất, tính trên kết quả Depth Anything ViT-L):

1. **NYUv2 (Dễ nhất — Sai số cực thấp 4.3%, độ chính xác 98.1%):**
   - *Lý do:* Đây là không gian trong nhà tĩnh (như phòng khách, phòng ngủ), các vật thể có cấu trúc rõ ràng và ít bị nhiễu. Mô hình dự đoán độ sâu ở đây là chính xác nhất.
2. **DIODE (Sai số 6.6%, độ chính xác 95.2%):**
   - *Lý do:* Dữ liệu quét bằng laser có độ chính xác cao bao gồm cả trong nhà lẫn ngoài trời, giúp mô hình dễ dàng nắm bắt khoảng cách thực tế.
3. **KITTI (Sai số 7.6%, độ chính xác 94.7%):**
   - *Lý do:* Cảnh đường phố ngoài trời quen thuộc khi lái xe. Các đối tượng như xe cộ, đường sá có chiều sâu rõ ràng nên mô hình xử lý khá tốt.
4. **ETH3D (Sai số 12.7%, độ chính xác 88.2%):**
   - *Lý do:* Các khung cảnh tự nhiên và công trình kiến trúc có độ phân giải cao, bắt đầu xuất hiện nhiều chi tiết phức tạp hơn khiến sai số tăng lên một chút.
5. **DDAD (Sai số 23.0%, độ chính xác 78.9%):**
   - *Lý do:* Cảnh lái xe nhưng ở góc nhìn rộng hơn, đa dạng hơn và có rất nhiều vật thể nhỏ nằm ở khoảng cách rất xa, làm mô hình bắt đầu bối rối.
6. **Sintel (Khó nhất — Sai số lên tới 45.8%, độ chính xác tụt còn 76.0%):**
   - *Lý do:* Đây là ảnh cắt ra từ phim hoạt hình 3D. Do không phải là ảnh thực tế ngoài đời và chứa nhiều hiệu ứng chuyển động, ánh sáng ảo phi thực tế, mô hình bị đánh lừa nhiều nhất và sai số lớn nhất.

### Mức độ dữ liệu tối thiểu để chạy kiểm chứng lại kết quả

Để kiểm chứng lại kết quả, ta chỉ cần tải trọng số mô hình pre-trained (checkpoint) đã công bố của Depth Anything và chạy suy luận (inference) trên tập test/validation chuẩn của các bộ dữ liệu.

Kiểm chứng toàn bộ benchmark (Full Test Split):

- NYUv2: Official test split = 654 ảnh.
- KITTI: Eigen test split = 652 ảnh.
- Sintel: Test/validation split = 1.064 ảnh.
- ETH3D: Test split = 454 ảnh.
- DIODE: Validation split = 771 ảnh.
- DDAD: Validation split = 3.950 ảnh.

***(Tổng cộng xấp xỉ ~7.500 ảnh cho cả 6 tập dữ liệu.)***

---

## 3. Bộ dữ liệu đánh giá về độ sâu tuyệt đối

### Khái niệm độ sâu tuyệt đối

- Là khoảng cách cụ thể được đo bằng đơn vị chiều dài cụ thể (mét hoặc centimet) từ vị trí ống kính đến bề mặt từng điểm ảnh trong ảnh 2D.
- Độ sâu tương đối chỉ cung cấp thứ tự gần xa, trên thang đo chuẩn hóa [0, 1] mà không biết khoảng cách thực.
- Bài toán này bắt buộc mô hình phải suy luận được tiêu cự camera, góc nhìn và quy mô vật lý thực tế của khung cảnh.

### Các chỉ số đánh giá

| Chỉ số | Công thức | Ý nghĩa | Chiều tốt hơn |
|---|---|---|---|
| AbsRel | mean(\|d̂ − d*\| / d*) | Sai số tương đối trung bình so với độ sâu thật | Thấp hơn (↓) |
| RMSE | sqrt(mean((d̂ − d*)²)) | Sai số căn quân phương, tính theo mét | Thấp hơn (↓) |
| RMSE log | sqrt(mean((log d̂ − log d*)²)) | RMSE tính trên thang log, giảm ảnh hưởng của vùng quá xa | Thấp hơn (↓) |
| log10 | mean(\|log₁₀ d̂ − log₁₀ d*\|) | Sai số log cơ số 10 | Thấp hơn (↓) |
| δ₁ | % điểm ảnh có max(d̂/d*, d*/d̂) < 1.25 | Tỷ lệ điểm ảnh dự đoán "đúng" trong ngưỡng 1.25 lần | Cao hơn (↑) |
| δ₂ | % điểm ảnh có max(d̂/d*, d*/d̂) < 1.25² | Ngưỡng nới lỏng hơn δ₁ | Cao hơn (↑) |
| δ₃ | % điểm ảnh có max(d̂/d*, d*/d̂) < 1.25³ | Ngưỡng nới lỏng nhất | Cao hơn (↑) |

### Phương pháp In-domain và Zero-shot

Tác giả sử dụng trọng số encoder pre-trained từ Depth Anything (ViT-L), kết hợp với ZoeDepth để fine-tune với thông tin độ sâu tuyệt đối.

#### A. In-domain

- Mô hình được fine-tune (huấn luyện bổ sung) và đánh giá trên cùng một tập dữ liệu/miền không gian.
- Bộ dữ liệu áp dụng: NYUv2 (không gian trong nhà) và KITTI (không gian giao thông ngoài trời).

**Bảng 2. Fine-tune và đánh giá trên NYUv2.**

| Method | δ₁↑ | δ₂↑ | δ₃↑ | AbsRel↓ | RMSE↓ | log10↓ |
|---|---|---|---|---|---|---|
| AdaBins | 0.903 | 0.984 | 0.997 | 0.103 | 0.364 | 0.044 |
| DPT | 0.904 | 0.988 | 0.998 | 0.110 | 0.357 | 0.045 |
| P3Depth | 0.898 | 0.981 | 0.996 | 0.104 | 0.356 | 0.043 |
| SwinV2-L | 0.949 | 0.994 | 0.999 | 0.083 | 0.287 | 0.035 |
| AiT | 0.954 | 0.994 | 0.999 | 0.076 | 0.275 | 0.033 |
| VPD | 0.964 | 0.995 | 0.999 | 0.069 | 0.254 | 0.030 |
| ZoeDepth* | 0.951 | 0.994 | 0.999 | 0.077 | 0.282 | 0.033 |
| **Ours (Depth Anything)** | **0.984** | **0.998** | **1.000** | **0.056** | **0.206** | **0.024** |

**Bảng 3. Fine-tune và đánh giá trên KITTI.**

| Method | δ₁↑ | δ₂↑ | δ₃↑ | AbsRel↓ | RMSE↓ | RMSE log↓ |
|---|---|---|---|---|---|---|
| AdaBins | 0.964 | 0.995 | 0.999 | 0.058 | 2.360 | 0.088 |
| DPT | 0.959 | 0.995 | 0.999 | 0.062 | 2.573 | 0.092 |
| P3Depth | 0.953 | 0.993 | 0.998 | 0.071 | 2.842 | 0.103 |
| NeWCRFs | 0.974 | 0.997 | 0.999 | 0.052 | 2.129 | 0.079 |
| SwinV2-L | 0.977 | 0.998 | 1.000 | 0.050 | 1.966 | 0.075 |
| NDDepth | 0.978 | 0.998 | 0.999 | 0.050 | 2.025 | 0.075 |
| GEDepth | 0.976 | 0.997 | 0.999 | 0.048 | 2.044 | 0.076 |
| ZoeDepth* | 0.971 | 0.996 | 0.999 | 0.054 | 2.281 | 0.082 |
| **Ours (Depth Anything)** | **0.982** | **0.998** | **1.000** | **0.046** | **1.896** | **0.069** |

*(*): Kết quả do nhóm tác giả Depth Anything tái hiện lại (reproduced).*

Đối với Bảng In-domain:

- **Thiết lập SOTA mới:** Mô hình tạo ra bước tiến lớn trên cả 2 benchmark tiêu chuẩn kinh điển của cộng đồng MDE.
- **Mức giảm sai số ấn tượng:** Trên tập NYUv2, chỉ số AbsRel giảm từ 0.069 (VPD) xuống 0.056 (giảm tới 18.8% sai số), RMSE giảm từ 0.254 m xuống 0.206 m. Trên KITTI, chỉ số δ₁ tăng từ 0.978 (NDDepth) lên 0.982, RMSE giảm xuống dưới 1.9 m (1.896 m).
- **Ý nghĩa:** Kết quả này chứng minh rằng trọng số khởi tạo (pre-trained encoder) của Depth Anything cung cấp biểu diễn đặc trưng không gian vượt trội so với các backbone phổ biến như SwinV2 hay ConvNeXt, giúp mô hình hội tụ về kết quả đo đạc chính xác hơn rất nhiều.

#### B. Zero-shot

- Mô hình chỉ được fine-tune trên một tập dữ liệu nguồn (NYUv2 cho indoor hoặc KITTI cho outdoor), sau đó đem đi đánh giá trực tiếp trên các tập dữ liệu hoàn toàn mới (unseen domains) mà không hề fine-tune thêm.

Bộ dữ liệu áp dụng:

- Mô hình train trên NYUv2 → Đánh giá zero-shot trên SUN RGB-D, iBims-1, HyperSim (Indoor).
- Mô hình train trên KITTI → Đánh giá zero-shot trên Virtual KITTI 2, DIODE Outdoor (Outdoor).

**Bảng 4. Đánh giá zero-shot độ sâu tuyệt đối (Zero-shot metric depth estimation).** Ba tập đầu là cảnh indoor (dùng model train trên NYUv2), hai tập sau là cảnh outdoor (dùng model train trên KITTI).

| Method | SUN RGB-D AbsRel↓ | SUN RGB-D δ₁↑ | iBims-1 AbsRel↓ | iBims-1 δ₁↑ | HyperSim AbsRel↓ | HyperSim δ₁↑ | Virtual KITTI 2 AbsRel↓ | Virtual KITTI 2 δ₁↑ | DIODE Outdoor AbsRel↓ | DIODE Outdoor δ₁↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| ZoeDepth | 0.520 | 0.545 | 0.169 | 0.656 | 0.407 | 0.302 | 0.106 | 0.844 | 0.814 | 0.237 |
| **Depth Anything** | **0.500** | **0.660** | **0.150** | **0.714** | **0.363** | **0.361** | **0.085** | **0.913** | **0.794** | **0.288** |

*Kết quả ZoeDepth được nhóm tác giả Depth Anything tái hiện lại trong cùng môi trường để so sánh công bằng.*

Đối với Bảng Zero-shot Metric Depth:

- **Thống trị toàn diện:** Depth Anything đánh bại ZoeDepth trên toàn bộ 5/5 tập dữ liệu kiểm thử ở cả 2 chỉ số AbsRel và δ₁.
- **Khả năng tổng quát hóa quy mô mét (Scale Generalization):** Sự tăng vọt đáng chú ý ở chỉ số δ₁ trên SUN RGB-D (tăng từ 54.5% lên 66.0%) và Virtual KITTI 2 (tăng từ 84.4% lên 91.3%).
- **Ý nghĩa:** Điều này khẳng định việc đào tạo trên 62 triệu+ ảnh không nhãn giúp mô hình học được mối quan hệ về mặt hình học và ngữ nghĩa mở (open-world semantic priors), tránh được hiện tượng overfitting vào góc nhìn hay thông số camera của tập dữ liệu huấn luyện ban đầu.

### Mức độ dữ liệu tối thiểu để chạy kiểm chứng lại kết quả

Tải tập test chuẩn, sử dụng trực tiếp trọng số checkpoint Metric Depth đã được tác giả công bố:

- In-domain NYUv2 Test: 654 ảnh.
- In-domain KITTI Test (Eigen Split): 652 ảnh.
- **Zero-shot Metric Sets:** SUN RGB-D (~5.050 ảnh test), iBims-1 (100 ảnh), DIODE Outdoor (446 ảnh validation).

---

## 4. Bộ dữ liệu đánh giá chuyển giao nhiệm vụ

- Đánh giá **khả năng chuyển giao nhiệm vụ** kiểm tra xem **biểu diễn đặc trưng** được học bởi encoder của Depth Anything từ quá trình huấn luyện ước lượng độ sâu quy mô lớn (1,5M ảnh có nhãn + 62M+ ảnh không nhãn) **có thể tái sử dụng làm một backbone/feature extractor** đa năng cho các nhiệm vụ thị giác máy tính cấp cao hay không.
- Tác giả lựa chọn nhiệm vụ Phân đoạn ngữ nghĩa (Semantic Segmentation) — bài toán phân loại ngữ nghĩa chi tiết đến từng điểm ảnh.
- Tác giả giữ nguyên trọng số của pre-trained encoder Depth Anything (ViT-L), gắn thêm một head/decoder phân đoạn ngữ nghĩa (như Mask2Former), sau đó fine-tune trên các tập dữ liệu Semantic Segmentation tiêu chuẩn.
- Nhờ thiết kế Semantic-Assisted Perception với hàm mất mát căn chỉnh đặc trưng (feature alignment loss L_feat) đi kèm ngưỡng dung sai α = 0.85, encoder vừa học được khả năng phân biệt độ sâu chi tiết ở mức điểm ảnh, vừa bảo tồn được tri thức ngữ nghĩa phong phú từ DINOv2 (kiến trúc pipeline minh họa ở Hình 2 của bài báo gốc: một encoder dùng chung cho cả ảnh có nhãn/không nhãn, ràng buộc căn chỉnh đặc trưng với một encoder DINOv2 đóng băng).

### Bộ dữ liệu áp dụng

2 bộ dữ liệu phân đoạn ngữ nghĩa tiêu chuẩn nhất trong cộng đồng Computer Vision để thử nghiệm chuyển giao:

- **Cityscapes:** Bối cảnh đường phố đô thị ngoài trời (urban street scenes) chụp ở 50 thành phố Châu Âu, chứa nhãn phân đoạn ngữ nghĩa chi tiết cho 19 lớp đối tượng (xe hơi, đường, người đi bộ, biển báo...) ở độ phân giải cao 1024 × 2048.
- **ADE20K:** Bối cảnh tự nhiên phức tạp đa dạng cả trong nhà (indoor) và ngoài trời (outdoor), chứa nhãn phân đoạn ngữ nghĩa cho 150 lớp vật thể và nền cảnh (mặt đất, đồ nội thất, tòa nhà, bầu trời...).

### Kết quả so sánh

**Bảng 5. Chuyển giao encoder MDE sang Cityscapes cho semantic segmentation.** s.s./m.s.: đánh giá single-scale / multi-scale. Không dùng Mapillary để pre-train.

| Method | Encoder | mIoU (s.s.) | mIoU (m.s.) |
|---|---|---|---|
| Segmenter | ViT-L | – | 82.2 |
| SegFormer | MiT-B5 | 82.4 | 84.0 |
| Mask2Former | Swin-L | 83.3 | 84.3 |
| OneFormer | Swin-L | 83.0 | 84.4 |
| OneFormer | ConvNeXt-XL | 83.6 | 84.6 |
| DDP | ConvNeXt-L | 83.2 | 83.9 |
| **Ours** | **ViT-L** | **84.8** | **86.2** |

**Bảng 6. Chuyển giao encoder MDE sang ADE20K cho semantic segmentation.** Dùng Mask2Former làm decoder phân đoạn.

| Method | Encoder | mIoU |
|---|---|---|
| Segmenter | ViT-L | 51.8 |
| SegFormer | MiT-B5 | 51.0 |
| Mask2Former | Swin-L | 56.4 |
| UperNet | BEiT-L | 56.3 |
| ViT-Adapter | BEiT-L | 58.3 |
| OneFormer | Swin-L | 57.4 |
| OneFormer | ConvNeXt-XL | 57.4 |
| **Ours** | **ViT-L** | **59.4** |

**Bảng 7. So sánh encoder Depth Anything với encoder MiDaS v3.1 khi fine-tune cho downstream task.**

| Encoder | NYUv2 AbsRel↓ | NYUv2 δ₁↑ | KITTI AbsRel↓ | KITTI δ₁↑ | Cityscapes mIoU↑ | ADE20K mIoU↑ |
|---|---|---|---|---|---|---|
| MiDaS | 0.077 | 0.951 | 0.054 | 0.971 | 82.1 | 52.4 |
| **Ours (Depth Anything)** | **0.056** | **0.984** | **0.046** | **0.982** | **84.8** | **59.4** |

**Bảng 8. So sánh encoder Depth Anything với encoder DINOv2 gốc khi fine-tune cho downstream task.**

| Encoder | NYUv2 AbsRel↓ | NYUv2 δ₁↑ | KITTI AbsRel↓ | KITTI δ₁↑ | ADE20K mIoU↑ |
|---|---|---|---|---|---|
| DINOv2 | 0.066 | 0.973 | 0.058 | 0.971 | 58.8 |
| **Ours (Depth Anything)** | **0.056** | **0.984** | **0.046** | **0.982** | **59.4** |

### Đánh giá

**Vượt qua các backbone đại trà mạnh nhất:**

- Trên tập **Cityscapes**, encoder Depth Anything (ViT-L) đạt **86.2 mIoU** (m.s.), vượt trội hơn hẳn các backbone nổi tiếng được pre-train trên ImageNet-21K như Swin-L (**84.3 mIoU**) hay ConvNeXt-XL (**84.6 mIoU**).
- Trên tập **ADE20K**, mô hình đạt **59.4 mIoU**, thiết lập kỷ lục SOTA mới, vượt qua ViT-Adapter (**58.3 mIoU**) và OneFormer (**57.4 mIoU**).

**So sánh với encoder của MiDaS v3.1 (Bảng 7):**

- MiDaS v3.1 chỉ đạt **82.1 mIoU** trên Cityscapes và **52.4 mIoU** trên ADE20K.
- *Nguyên nhân:* MiDaS thuần túy học độ sâu tương đối nên làm mất mát rất nhiều thông tin ngữ nghĩa cấp cao. Ngược lại, Depth Anything nhờ cơ chế căn chỉnh đặc trưng L_feat giữ lại được các đặc trưng ngữ nghĩa liên tục, vượt trội hơn MiDaS tới **+7.0 mIoU** trên ADE20K.

**Vượt qua chính DINOv2 gốc (Bảng 8):**

- Encoder DINOv2 nguyên bản đạt **58.8 mIoU** trên ADE20K. Khi trải qua quá trình huấn luyện MDE quy mô lớn (62M+ ảnh), encoder Depth Anything đạt **59.4 mIoU**.
- *Ý nghĩa:* Quá trình huấn luyện MDE quy mô lớn không làm suy giảm biểu diễn ngữ nghĩa của DINOv2 mà còn bổ sung thêm khả năng phân biệt chi tiết theo từng điểm ảnh, giúp encoder trở thành một **multitask encoder đa năng** cho cả bài toán độ sâu (middle-level) lẫn phân đoạn ngữ nghĩa (high-level).

### Mức độ dữ liệu tối thiểu để chạy kiểm chứng lại kết quả

Chỉ chạy suy luận đánh giá checkpoint đã fine-tune:

- **Cityscapes Validation Set:** 500 ảnh (1024 × 2048).
- **ADE20K Validation Set:** 2.000 ảnh.

---

## Danh sách tổng hợp bộ dữ liệu đánh giá

**1. Đánh giá Độ sâu Tương đối Zero-shot**

1. KITTI: *Eigen Test Split* (chuẩn 652 ảnh).
2. NYUv2: *Official Test Split* (chuẩn 654 ảnh).
3. Sintel: *Validation/Test Split* (1.064 ảnh).
4. DDAD: *Validation Split* (3.950 ảnh).
5. ETH3D: *Benchmark Test Split* (454 ảnh).
6. DIODE: *Validation Split* (771 ảnh indoor & outdoor).

**2. Đánh giá Độ sâu Tuyệt đối**

*In-domain Metric Depth (Fine-tune và đánh giá trên cùng tập):*

1. NYUv2: *Official Test Split* (654 ảnh).
2. KITTI: *Eigen Test Split* (652 ảnh).

*Zero-shot Metric Depth (Huấn luyện trên NYUv2/KITTI, thử nghiệm sang tập mới):*

1. SUN RGB-D: *Indoor Test Set* (~5.050 ảnh).
2. iBims-1: *Indoor Benchmark Set* (100 ảnh).
3. HyperSim: *Indoor Synthetic Test Set*.
4. Virtual KITTI 2: *Outdoor Synthetic Test Set*.
5. DIODE Outdoor: *Outdoor Validation Subset* (446 ảnh).

**3. Đánh giá Chuyển giao Nhiệm vụ Phân đoạn Ngữ nghĩa**

*Dùng để kiểm tra chất lượng biểu diễn đặc trưng của encoder:*

1. **Cityscapes**: *Validation Split* (500 ảnh, độ phân giải 1024 × 2048).
2. **ADE20K**: *Validation Split* (2.000 ảnh).
