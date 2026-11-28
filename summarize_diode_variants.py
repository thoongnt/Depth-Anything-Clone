"""Merge completed DIODE protocol grids and verify native metrics against inference."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def merge(grid_dir, baseline, flip_summary):
    reports = [json.loads(p.read_text(encoding='utf-8'))
               for p in Path(grid_dir).glob('*/summary.json')]
    rows = [r for report in reports for r in report['results']]
    keys = [(r['flip'], r['configuration'], r['subset'], r['aggregation']) for r in rows]
    if len(rows) != 84 or len(set(keys)) != 84:
        raise ValueError(f'Expected 84 unique comparisons, found {len(rows)}')
    sources = {False: json.loads(Path(baseline).read_text()),
               True: json.loads(Path(flip_summary).read_text())}
    for flip, summary in sources.items():
        assert summary['full_validation_split'] and summary['num_images'] == 771
        for subset in ['all', 'indoors', 'outdoor']:
            row = next(r for r in rows if r['flip'] == flip and r['configuration'] == 'native'
                       and r['subset'] == subset and r['aggregation'] == 'image-mean')
            expected = summary if subset == 'all' else summary['by_domain'][subset]
            assert row['num_images'] == expected['num_images']
            assert row['valid_pixels'] == expected['valid_pixels']
            for metric in ['abs_rel', 'delta1']:
                assert abs(row[metric]-expected['metrics'][metric]) < 5e-8, (flip, subset, metric)
        identity = json.loads((Path(summary['prediction_cache'])/'manifest.json').read_text())
        assert identity['flip'] == flip
        assert identity['checkpoint_sha256'] == summary['protocol']['checkpoint_sha256']
    full = [r for r in rows if r['full_subset']]
    best = {subset: min([r for r in full if r['subset'] == subset],
                        key=lambda r:r['max_absolute_metric_gap'])
            for subset in ['all', 'indoors', 'outdoor']}
    same_pixels = [r for r in full if r['subset'] == 'all' and r['aggregation'] == 'image-mean'
                   and r['native_valid_pixel_retention'] == 1.
                   and r['configuration'] != 'native-depth-metric']
    result = {'scope': 'Exploratory comparison, not author-confirmed paper protocol',
              'paper_reference': {'abs_rel': .066, 'delta1': .952}, 'num_comparisons': len(rows),
              'ranking': 'minimize max(abs(AbsRel-.066), abs(delta1-.952)); full subsets only',
              'native_results_verified_against_inference_summaries': True,
              'best_by_subset': best,
              'best_all_with_original_scoring_pixels': min(same_pixels, key=lambda r:r['max_absolute_metric_gap']),
              'rounding_matches': [r for r in rows if r['matches_both_paper_rounding']], 'results': rows}
    grid_dir = Path(grid_dir)
    (grid_dir/'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    csv_rows = [{**{k:v for k,v in r.items() if k != 'unscorable_images'},
                 'unscorable_images_count': len(r['unscorable_images'])} for r in rows]
    with (grid_dir/'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    return result


def write_report(result):
    rows = result['results']
    text = '''# DIODE: so sánh cấu hình với Table 2

Mốc ViT-L: **AbsRel 0.066, δ₁ 0.952**. Đây là thăm dò protocol theo yêu cầu đối chiếu số; gần số paper chưa tự xác nhận đó là protocol tác giả.

Cùng checkpoint relative ViT-L và dữ liệu đã xác thực 2.313 file. Full flip có đủ 771 ảnh: flip ngang, lật prediction về chiều gốc, trung bình với prediction ảnh gốc trước resize về GT. Không dùng lại kết quả probe 16 ảnh làm full TTA.

## Flip và từng miền, giữ nguyên mask/OLS/disparity

| Flip | Phạm vi | Số ảnh | AbsRel ↓ | δ₁ ↑ |
|---|---|---:|---:|---:|
'''
    for flip in [False, True]:
        for subset in ['all', 'indoors', 'outdoor']:
            row = next(r for r in rows if r['flip'] == flip and r['subset'] == subset
                       and r['configuration'] == 'native' and r['aggregation'] == 'image-mean')
            text += f"| {flip} | {subset} | {row['num_images']} | {row['abs_rel']:.9f} | {row['delta1']:.9f} |\n"
    text += '''
Ba cảnh outdoor đều lệch so với mốc Table 2. Với flip và cách chấm native, AbsRel từng cảnh nằm trong khoảng 0.266–0.482. [Kết quả từng scene](../results/table2/diode/variants/native-per-scene.csv).

## Gần số paper nhất theo phạm vi

Xếp hạng bằng `max(|AbsRel−0.066|, |δ₁−0.952|)`, yêu cầu đủ ảnh của phạm vi đã chọn. Tỷ lệ pixel là số pixel được chấm chia tổng pixel native hợp lệ cùng phạm vi. Những dòng có ảnh mask rỗng không được xếp là đủ subset.

| Phạm vi | Cấu hình | Flip | Số ảnh | Giữ pixel native | AbsRel ↓ | δ₁ ↑ |
|---|---|---|---:|---:|---:|---:|
'''
    for subset, row in result['best_by_subset'].items():
        text += f"| {subset} | {row['configuration']} / {row['aggregation']} | {row['flip']} | {row['num_images']} | {100*row['native_valid_pixel_retention']:.2f}% | {row['abs_rel']:.9f} | {row['delta1']:.9f} |\n"
    best = result['best_all_with_original_scoring_pixels']
    text += f"\nGần nhất trên đủ **771 ảnh, giữ 100% pixel chấm native, disparity, trung bình từng ảnh**: **{best['configuration']}**, flip={best['flip']}, **{best['abs_rel']:.9f} / {best['delta1']:.9f}**.\n"
    text += f"\nSố trường hợp đồng thời làm tròn thành **0.066 / 0.952**: **{len(result['rounding_matches'])}**.\n"
    for row in result['rounding_matches']:
        text += f"\n- {row['configuration']}, flip={row['flip']}, subset={row['subset']}, aggregation={row['aggregation']}, giữ {100*row['native_valid_pixel_retention']:.2f}% pixel native.\n"
    text += '''
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

- [Toàn bộ CSV](../results/table2/diode/variants/protocol-grid/comparison.csv), [JSON](../results/table2/diode/variants/protocol-grid/comparison.json), [định nghĩa cấu hình](../results/table2/diode/variants/configurations.json).
- [Full flip summary](../results/table2/diode/variants/vitl-native-all-flip-summary.json), [CSV từng ảnh](../results/table2/diode/variants/vitl-native-all-flip-per-image.csv).
- Thư mục `protocol-grid/no-flip-*` và `protocol-grid/flip-*` lưu CSV từng ảnh cho từng cấu hình: metric, pixel, scale/shift, trạng thái.

```powershell
.venv\\Scripts\\python.exe evaluate_diode_relative.py --flip --data-audit results/table2/diode/audit/official-archive-verification.json --cache-dir .cache/diode-variants/predictions --output-dir results/table2/diode/variants

.venv\\Scripts\\python.exe research_diode_variants.py --workers 8 --summary results/table2/diode/audit-rerun/fresh-inference-summary.json results/table2/diode/variants/vitl-native-all-flip-summary.json
```

Evaluator mặc định và kết quả NYUv2 không thay đổi. Những kết quả được chọn vì gần số đích là thăm dò sau khi biết số paper, không phải kiểm chứng độc lập protocol.
'''
    (ROOT/'docs/table2/diode-variants.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid-dir', type=Path, default=ROOT/'results/table2/diode/variants/protocol-grid')
    parser.add_argument('--baseline', type=Path, default=ROOT/'results/table2/diode/audit-rerun/fresh-inference-summary.json')
    parser.add_argument('--flip-summary', type=Path, default=ROOT/'results/table2/diode/variants/vitl-native-all-flip-summary.json')
    args = parser.parse_args()
    result = merge(args.grid_dir, args.baseline, args.flip_summary)
    write_report(result)
    print(json.dumps({k:v for k,v in result.items() if k != 'results'}, indent=2))
