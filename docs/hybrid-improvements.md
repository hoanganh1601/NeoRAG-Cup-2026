# Pipeline hybrid nâng cấp

Pipeline mới nằm trong `src/neorag/pipelines/hybrid.py`, giữ nguyên baseline hiện tại để đối chiếu.

Các thay đổi:

- BM25 thay cho phép đếm từ khóa giao nhau đơn giản; có tính tần suất từ và độ dài tài liệu.
- Chuẩn hóa dấu tiếng Việt riêng cho tìm kiếm từ khóa, giữ nguyên nội dung gốc và đầu vào embedding.
- Thống nhất tên viết tắt ProPTIT, MPC, GCC, CLB LT PTIT và CLB MM khi khớp từ khóa.
- Bổ sung cặp từ liên tiếp để nhận diện cụm như “đối ngoại”, “ban chủ nhiệm”.
- Cộng điểm mềm cho CLB được nhắc rõ trong câu hỏi; không loại các CLB khác khỏi ứng viên.
- Context có nhãn ID, CLB và chủ đề để phân biệt các đoạn. Giữ prompt và top 7 đã được thử nghiệm.

Tham số hiện tại: 40% điểm dense + 60% điểm BM25 đã chuẩn hóa, cộng 0.2 điểm mềm cho CLB phù hợp.
Điểm cộng này không phải xác suất và không phải bộ lọc loại tài liệu.

Kết quả retrieval trên train, dùng cùng embedding đã cache:

| Metric | Hybrid cũ | Hybrid BM25 |
|---|---:|---:|
| Recall@5 | 0.7814 | 0.9010 |
| MRR@5 | 0.7277 | 0.8613 |
| NDCG@5 | 0.7136 | 0.8384 |
| Retrieval score | 0.7409 | 0.8669 |

Số câu train có đáp án nhưng không lấy được passage đúng nào trong top 5 giảm từ 19 xuống 4.
Điểm generation, Benchmark và Public Test cần được đo riêng; không suy ra từ bảng retrieval.

Sau khi sinh đủ 126 câu train và chấm bằng bộ chấm chính thức với `--no-judge`:

| Metric generation | Hybrid cũ, alias, top 7 | Hybrid BM25 |
|---|---:|---:|
| String Presence | 0.7572 | 0.8543 |
| ROUGE-L | 0.5188 | 0.5731 |
| BLEU-4 (tham khảo) | 0.3608 | 0.3992 |

Chưa có Benchmark mới vì chưa chạy judge cho bản BM25. ZIP Public Test đã được tạo và validate đủ 42 câu;
điểm trên website phải đợi chấm sau khi nộp. Các bài kiểm tra tự động hiện tại: 45 bài đều đạt.

Không dùng đáp án hoặc nhãn của Public Test/Private Test. Các tham số thử nghiệm được so sánh trên train;
đây là điểm phát triển, không phải điểm đánh giá độc lập trên test.

## Kiểm tra retrieval không tốn API

```powershell
python scripts/benchmark_retrieval.py
```

Script chỉ đọc embedding từ `.cache/pipeline.sqlite`. Thiếu cache sẽ báo lỗi thay vì gọi API.
Kết quả được lưu ở `outputs/retrieval_experiments.jsonl` và `outputs/retrieval_candidate_per_query.jsonl`.
Ba nhóm train trong báo cáo chỉ dùng kiểm tra mức dao động, không phải cross-validation độc lập.

## Chạy pipeline mới

```powershell
neorag run train --pipeline neorag.pipelines.hybrid:HybridPipeline --no-judge --out outputs/train_hybrid_bm25_pred.jsonl
neorag evaluate train --predictions outputs/train_hybrid_bm25_pred.jsonl
neorag run public_test --pipeline neorag.pipelines.hybrid:HybridPipeline --out outputs/public_test_hybrid_bm25_pred.jsonl
```

File nộp: `outputs/public_test_hybrid_bm25_submission.zip`.
Chọn đúng `--pipeline`; nếu bỏ tùy chọn này CLI vẫn dùng baseline cũ.

Tạo câu trả lời và chấm bằng judge dùng ngân sách gateway. Các file cũ không bị ghi đè bởi các lệnh trên.
Nếu điểm mới kém hơn, dùng lại pipeline baseline và ZIP `public_test_hybrid_alias_top7_submission.zip`.

## Bản thử nghiệm tiếp theo: Claude reranking

Pipeline riêng: `neorag.pipelines.rerank:RerankPipeline`.
Hybrid lấy 20 ứng viên; Claude xếp lại tối đa 10 ID; 7 đoạn đầu được đưa vào prompt trả lời hiện tại.
Mỗi câu thường có thêm một request Claude để rerank, dùng chung cache của pipeline.
`--no-judge` chỉ tắt bộ chấm LLM; vẫn có request reranking và sinh câu trả lời.

Reranker chỉ nhận câu hỏi và nội dung corpus. Nếu JSON trả về không hợp lệ, pipeline cảnh báo và
dùng lại thứ tự hybrid. ID ngoài danh sách ứng viên, ID trùng hoặc sai kiểu bị bỏ qua;
các vị trí còn thiếu được điền theo thứ tự hybrid. Lỗi kết nối/key/budget vẫn được client và CLI xử lý.
Không có request sửa JSON bổ sung. Không thay đổi prompt sinh câu trả lời, model, hoặc bộ chấm.

Chạy train trước (ghi file mới):

```powershell
neorag run train --pipeline neorag.pipelines.rerank:RerankPipeline --no-judge --out outputs/train_hybrid_rerank_pred.jsonl
```

So với bản BM25: Recall@5 = 0.9010, MRR@5 = 0.8613, NDCG@5 = 0.8384,
String Presence = 0.8543, ROUGE-L = 0.5731. Chưa chạy API cho bản rerank và chưa có kết quả mới.
Chỉ triển khai các kiểm tra bằng LLM giả lập khi chuẩn bị bản này.

Nếu muốn chấm đầy đủ file train vừa tạo:

```powershell
neorag evaluate train --predictions outputs/train_hybrid_rerank_pred.jsonl
```

Sau khi so sánh kết quả train, tạo bài Public Test mới:

```powershell
neorag run public_test --pipeline neorag.pipelines.rerank:RerankPipeline --out outputs/public_test_hybrid_rerank_pred.jsonl
```

File nộp: `outputs/public_test_hybrid_rerank_submission.zip`.
Bản BM25 đạt raw score Public Test 72.30 theo kết quả người dùng cung cấp; giữ ZIP
`public_test_hybrid_bm25_submission.zip` để đối chiếu. Không thay pipeline mặc định của CLI;
phải truyền đúng `--pipeline` để kích hoạt reranking.

## Thử nghiệm prompt trả lời tập trung và đủ dữ kiện

Pipeline: `neorag.pipelines.focused:FocusedRerankPipeline`.
Giữ nguyên cách xếp hạng BM25, prompt rerank, 20 ứng viên, 10 ID đầu ra và 7 đoạn context.
Chỉ thay chỉ dẫn sinh câu trả lời: nêu rõ chủ thể, bảo toàn dữ kiện cần thiết, gộp ý trùng,
tránh giải thích ngoài phạm vi, xử lý nguồn mâu thuẫn và từ chối có nêu rõ phần thiếu.
Không ép số câu cố định và không thêm bước gọi LLM. Các request rerank đã cache được dùng lại.
Prompt không chứa đáp án train, câu hỏi test, nội dung nhằm tác động judge hoặc chỉ dẫn chấm điểm.

Mốc reranker trước khi sửa prompt: Retrieval 0.9703, String Presence 0.8975, ROUGE-L 0.5788,
Groundedness 0.9852, Response Relevancy 0.6878, Abstention 0.9091, Benchmark train 0.9008.
Chưa chạy API hoặc khẳng định cải thiện metric cho prompt mới.

```powershell
neorag run train --pipeline neorag.pipelines.focused:FocusedRerankPipeline --no-judge --out outputs/train_hybrid_rerank_focused_pred.jsonl
neorag evaluate train --predictions outputs/train_hybrid_rerank_focused_pred.jsonl
```

Chấm lexical trước để xem có làm mất dữ kiện không; chấm đầy đủ để xác định Response Relevancy,
Groundedness, Abstention và Benchmark có cải thiện không. Không kết luận từ riêng độ dài câu trả lời.
Khi so sánh, dùng các file predictions riêng, không sửa bộ chấm hoặc `configs/eval.toml`.

ROUGE-L phụ thuộc cách diễn đạt so với đáp án; Response Relevancy dùng câu hỏi sinh ngược và cosine
embedding, nên một câu trả lời đúng vẫn có thể nhận điểm thấp, đặc biệt với câu hỏi viết tắt/không dấu.
Không thêm câu hỏi lặp lại hoặc dữ kiện ngoài phạm vi chỉ để tăng độ trùng khớp.

## Biến thể cô đọng: ConciseRerankPipeline

`neorag.pipelines.concise:ConciseRerankPipeline` giữ nguyên reranking và context,
thử prompt ưu tiên một đoạn văn cô đọng nhưng vẫn trả lời đủ từng ý. Không thay đổi bộ chấm,
không thêm bước gọi model, không dùng nhãn test. Kết quả phải được đo độc lập so với bản focused.

```powershell
neorag run train --pipeline neorag.pipelines.concise:ConciseRerankPipeline --no-judge --out outputs/train_hybrid_rerank_concise_pred.jsonl
neorag evaluate train --predictions outputs/train_hybrid_rerank_concise_pred.jsonl
neorag run public_test --pipeline neorag.pipelines.concise:ConciseRerankPipeline --out outputs/public_test_hybrid_rerank_concise_pred.jsonl
```

ZIP mới: `outputs/public_test_hybrid_rerank_concise_submission.zip`.
Bản reranker với prompt ban đầu đã đạt Public raw score 84.51 theo kết quả người dùng cung cấp;
giữ `outputs/public_test_hybrid_rerank_submission.zip` làm mốc đối chiếu.

### Kết quả đo ngày 07/11/2026

Đã tạo đủ train và Public Test cho cả focused và concise. Focused có String Presence 0.9232,
nhưng ROUGE-L giảm xuống 0.5192; chưa chấm judge cho focused và không chọn làm bản khuyến nghị thử tiếp.

Concise đã được chấm đầy đủ bằng bộ chấm chính thức, không đổi cấu hình:

| Metric train | Reranker ban đầu | Concise |
|---|---:|---:|
| Retrieval | 0.9703 | 0.9703 |
| String Presence | 0.8975 | 0.9203 |
| ROUGE-L | 0.5788 | 0.5795 |
| Groundedness | 0.9852 | 0.9848 |
| Response Relevancy | 0.6878 | 0.6735 |
| Noise Sensitivity (thấp tốt hơn) | 0.0704 | 0.1224 |
| Abstention | 0.9091 | 1.0000 |
| Generation | 0.8313 | 0.8393 |
| Benchmark | 0.9008 | 0.9048 |

Đây là đánh đổi: tăng dữ kiện và khả năng từ chối, nhưng Response Relevancy và Noise Sensitivity kém hơn.
ROUGE-L gần như không đổi. Mức tăng Benchmark nhỏ; không đảm bảo Public Test cao hơn raw 84.51.
Chỉ xem concise là một bản thử nghiệm. Giữ nguyên ZIP bản reranker ban đầu để so sánh.
Đã xác nhận các ID/thứ tự retrieval của concise giống bản reranker ban đầu trên cả train và Public Test.
ZIP `public_test_hybrid_rerank_concise_submission.zip` chứa đúng một JSONL, đủ 42 predictions hợp lệ.
Các kiểm tra tự động: 60 bài đều đạt. Không sửa corpus, bộ chấm hoặc cấu hình chấm.
