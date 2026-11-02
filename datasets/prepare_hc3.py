
import json
import csv
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "raw" / "hc3_wiki_csai.jsonl"
OUTPUT_FILE = BASE_DIR / "processed" / "hc3_wiki_processed.csv"


def prepare_hc3():
    """
    Convert HC3 records into K-Detect's standardized format.

    Labels:
        0 = Human-written
        1 = AI-generated
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    samples = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            record = json.loads(line)

            source_id = f"hc3_wiki_{index}"

            for answer in record["human_answers"]:
                if answer.strip():
                    samples.append({
                        "text": answer.strip(),
                        "label": 0,
                        "source_id": source_id,
                        "domain": "general_knowledge",
                        "generation_model": "human",
                        "editing_status": "original",
                        "license": "CC-BY-SA-4.0",
                        "split_group": source_id,
                    })

            for answer in record["chatgpt_answers"]:
                if answer.strip():
                    samples.append({
                        "text": answer.strip(),
                        "label": 1,
                        "source_id": source_id,
                        "domain": "general_knowledge",
                        "generation_model": "ChatGPT (exact version unknown)",
                        "editing_status": "AI-generated",
                        "license": "HC3 source terms; verify before reuse",
                        "split_group": source_id,
                    })

    fieldnames = [
        "text",
        "label",
        "source_id",
        "domain",
        "generation_model",
        "editing_status",
        "license",
        "split_group",
    ]

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)

    print("HC3 preprocessing completed!")
    print("Total samples:", len(samples))
    print("Human samples:", sum(s["label"] == 0 for s in samples))
    print("AI samples:", sum(s["label"] == 1 for s in samples))
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    prepare_hc3()
