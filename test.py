from pipeline import run_full_pipeline
import json
from datetime import datetime

video_path = "kotuadaykonusma.mp4"

result = run_full_pipeline(video_path)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_name = video_path.rsplit(".", 1)[0]  # "cidd"
output_path = f"analiz_{video_name}_{timestamp}.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Kaydedildi: {output_path}")





