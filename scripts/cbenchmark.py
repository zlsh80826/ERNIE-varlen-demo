import subprocess
import itertools
import argparse
import datetime
import time
import sys
import os

BROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
import pandas as pd

from utils import read_label, get_metric, get_num_heads
from utils import get_valid_data, get_num_labels, get_valid_labels


class Benchmarker:
    def __init__(self, dataset, model, mode, batch_size, seq_len, args):
        self.dataset = dataset
        self.model = model
        self.mode = mode
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.args = args

    def benchmark(self):
        model_path = os.path.join(BROOT, f'models/{self.dataset}-{self.model}-2.0')
        valid_data = os.path.join(BROOT, get_valid_data(self.dataset, self.model))
        inference_bin = os.path.join(BROOT, 'build/inference')
        cmd = [inference_bin, '--logtostderr',
                        '--model', model_path,
                        '--data', valid_data,
                        '--mode', mode,
                        '--batch_size', str(batch_size),
                        '--num_labels', get_num_labels(self.dataset),
                        '--seq_lens', str(self.seq_len),
                        '--min_graph', str(self.args.min_graph),
                        '--ignore_copy', str(self.args.ignore_copy),
                        ]
        ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ret.returncode != 0:
            print(ret.stderr.decode('ascii'))
            assert False, 'Prediction failed.'
        prediction = list()
        for line in ret.stdout.decode('ascii').splitlines():
            if line.startswith('Sents/s'):
                _, qps = line.split()
            else:
                prediction.append(int(line))
        prediction = np.asarray(prediction)
        testcase = os.path.join(BROOT, get_valid_labels(self.dataset))
        labels = read_label(testcase)
        metric = get_metric(self.dataset)
        ret = metric(prediction, labels)
        stat = {'Sents/s': float(qps)}
        stat['metric_value'] = ret
        stat['metric'] = metric.__name__
        stat['batch_size'] = batch_size
        stat['dataset'] = self.dataset
        stat['model'] = self.model + '-2.0'
        stat['mode'] = self.mode
        if self.seq_len == 0:
            stat['seq_len'] = 'dynamic'
        else:
            stat['seq_len'] = self.seq_len
        return stat


def parse_args():
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    parser = argparse.ArgumentParser(description='Benchmarker for PaddlePaddle ERNIE')
    parser.add_argument('--dataset', '-d', nargs='+', default=['CoLA'],
                        choices=['QNLI'])
    parser.add_argument('--model', '-m', nargs='+', default=['base'], choices=['base'])
    parser.add_argument('--inference', '-i', nargs='+', default=['fp32'], choices=['fp32', 'trt-fp32', 'trt-fp16'])
    parser.add_argument('--batch_size', '-b', type=int, default=[1], nargs='+')
    parser.add_argument('--seq_len', type=int, default=[0], nargs='+', help='whether use fix length, default is dynamic')
    parser.add_argument('--stats_csv', type=str, default=os.path.join(BROOT, f'logs/benchmark.{timestamp}.csv'))
    parser.add_argument('--cool_down', '-c', type=int, default=0)
    parser.add_argument('--ignore_copy', type=int, default=1, help='whether to ignore copy cost')
    parser.add_argument('--min_graph', type=int, default=3, help='min graph size in trt options')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    results = pd.DataFrame(columns=['dataset', 'model', 'mode', 'seq_len', 'batch_size', 'metric', 'metric_value', 'Sents/s'])

    combinations = [args.dataset, args.model, args.inference, args.batch_size, args.seq_len]

    for dataset, model, mode, batch_size, seq_len in itertools.product(*combinations):
        benchmarker = Benchmarker(dataset, model, mode, batch_size, seq_len, args)
        print(f'Start benchmark {dataset}-{model}, {mode}, bs: {batch_size}, seq_len: {seq_len}', end='... ')
        stat = benchmarker.benchmark()
        print(f"Sents/s: {stat['Sents/s']}")
        results = results.append(pd.Series(stat), ignore_index=True)
        time.sleep(args.cool_down)
    results.style.hide_index()
    results.to_csv(args.stats_csv, index=False)
    print(f'Saving statistics to {args.stats_csv}')
    print(results)
