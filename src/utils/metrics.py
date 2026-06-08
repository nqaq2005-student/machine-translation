from typing import List
import sacrebleu


def compute_bleu(references: List[str], predictions: List[str]) -> float:
    if len(predictions) == 0:
        return 0.0
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return float(bleu.score)
