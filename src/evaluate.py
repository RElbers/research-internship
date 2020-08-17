from pathlib import Path

import torch

from eval.evaluation import Evaluator

with torch.no_grad():
    tests = [
        # Path(r'../data\1024\2020-06-30--23-36-59--[1024_guided_bce_no_mult]'),
        # Path(r'../data\1024\2020-07-02--01-00-57--[1024_guided_bce_weight_1]'),
    ]
    # tests.extend(Path(r'D:\Libraries\Documents\projects\research-internship\data\1024').glob('*'))
    # tests.extend(Path(r'D:\Libraries\Documents\projects\research-internship\data\512').glob('*'))
    tests.extend(Path(r'D:\Libraries\Documents\projects\research-internship\output\test').glob('*'))

    # confusion_stuff()

    for path in tests:
        name = path.name.split('--')[-1]

        eval = Evaluator(Path(path), name=name)
        # checkpoint = eval.find_best_model('test')
        checkpoint = eval.checkpoints[-1]

        eval.evaluate(checkpoint)
        eval.attention_mask(checkpoint, 'per_abnormality')
        eval.attention_mask(checkpoint, 'per_mammogram')
