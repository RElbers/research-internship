from pathlib import Path

import torch

from eval.evaluation import Evaluator

with torch.no_grad():
    tests = []
    tests.extend(Path(r'../output').glob('*'))

    for path in tests:
        if path.name == 'eval':
            continue

        name = path.name.split('--')[-1]
        eval = Evaluator(Path(path), name=name)

        # checkpoint = eval.find_best_model('test')
        checkpoint = eval.checkpoints[-1]  # Last model

        eval.evaluate(checkpoint)
        eval.attention_map(checkpoint, 'per_abnormality')
        eval.attention_map(checkpoint, 'per_mammogram')
