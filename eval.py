"""השוואת ארבע תצורות אחזור.

Hit@K בודק אם המחרוזת המצופה מופיעה באחד הקטעים שהוחזרו.
אין יצירת תשובות; בתצורות עם הרחבה יש קריאת LLM להרחבת השאלה.
"""

import time

import pandas as pd

from config import EVAL_QUESTIONS, TOP_K
from query import retrieve


CONFIGURATIONS = [
    ("וקטורי בלבד", False, False),
    ("וקטורי + הרחבה", False, True),
    ("היברידי", True, False),
    ("היברידי + הרחבה", True, True),
]


def is_hit(nodes, expected) -> bool:
    """בודק אם אחד הקטעים מכיל את המחרוזת המצופה."""
    if expected is None:
        return False
    return any(expected in node.get_text() for node in nodes)


def run(limit: int = None, top_k: int = TOP_K) -> pd.DataFrame:
    questions = EVAL_QUESTIONS[:limit] if limit else EVAL_QUESTIONS
    rows = []

    for label, use_hybrid, use_expansion in CONFIGURATIONS:
        hits, times, misses = 0, [], []

        for item in questions:
            start = time.perf_counter()

            try:
                nodes, _ = retrieve(
                    item["question"],
                    top_k=top_k,
                    use_hybrid=use_hybrid,
                    use_expansion=use_expansion,
                )
            except Exception as exc:
                print("[שגיאה] " + item["question"] + " -> " + str(exc))
                nodes = []

            times.append(time.perf_counter() - start)

            if item["expected"] is None:
                # שאלת השלילה אינה נכללת בציון.
                # הימנעות מהמצאה נבדקת ידנית בממשק.
                continue

            if is_hit(nodes, item["expected"]):
                hits += 1
            else:
                misses.append(item["note"])

        scored = [q for q in questions if q["expected"] is not None]

        rows.append({
            "קונפיגורציה": label,
            "Hit@" + str(top_k): round(
                hits / max(len(scored), 1) * 100, 1
            ),
            "פספוסים": len(misses),
            "זמן ממוצע (שנ')": round(sum(times) / len(times), 2),
            "השאלות שנכשלו": ", ".join(sorted(set(misses))) or "-",
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    df = run(limit=limit)

    print()
    print(df.to_string(index=False))
    df.to_csv("eval_results.csv", index=False, encoding="utf-8-sig")
    print("\nנשמר לקובץ eval_results.csv")