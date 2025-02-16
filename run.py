import argparse
import os
import sys
import warnings
from time import perf_counter

import gpt_2_simple as gpt2

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf

"""
From src/sample.py:17: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
src/accumulate.py:14: Variable.initialized_value (from tensorflow.python.ops.variables) is deprecated and will be removed in a future version.
Instructions for updating:
Use Variable.read_value. Variables in 2.X are initialized automatically both in eager and graph (inside tf.defun) contexts.

python ./run.py -t --num_epochs 10000 --target_loss 0.8 -tm jens_345M -fp ../training_text/jens.txt --save_epochs 100 -gm 345M --gpu_frac 0.75
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Runs gpt-2-simple.')
    parser.add_argument('-tm', '--tune_model', required=True, type=str,
                        help='tune model name')
    # Operation
    parser.add_argument('-t', '--train', required=False, action='store_true',
                        help='train model')
    parser.add_argument('-g', '--generate', required=False, action='store_true',
                        help='generate text')
    # Training
    parser.add_argument('-gm', '--gpt2_model', required=False, type=str,
                        help='GPT2 base model name, 117M, 124M or 345M', default='117M')
    parser.add_argument('-fp', '--file_path', required=False, type=str,
                        help='Optional path to text file.', default='./')
    parser.add_argument('--batch_size', required=False, type=int,
                        help='Specify batch size', default=1)
    parser.add_argument('--gpu_frac', required=False, type=float,
                        help='Fraction of GPU memory to use.', default=None)
    parser.add_argument('--num_epochs', required=False, type=int,
                        help='Max epochs', default=-1)
    parser.add_argument('--target_loss', required=False, type=float,
                        help='Max epochs', default=1.5)
    parser.add_argument('--save_epochs', required=False, type=int,
                        help='How often to save.', default=100)
    parser.add_argument('--verbose', required=False, type=int,
                        help='Verbosity level, 0 silent, 1 progress bar, 2 epoch only.', default=1)
    # Generation
    parser.add_argument('-n', '--num_words', required=False, type=int,
                        help='number of words to generate', default=100)
    parser.add_argument('--temp', required=False, type=float,
                        help='Temperature to use when generating.', default=0.5)
    parser.add_argument('--cpu', required=False, action='store_true',
                        help='Force CPU only.')
    parser.add_argument('--save', required=False, action='store_true',
                        help='Save output to text file')
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    # Hide debug info
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    args = parse_args()
    textpath = args['file_path']
    if not os.path.exists(textpath) and args['train']:
        print(F"No file {textpath} found.")
        sys.exit(0)

    if args['cpu']:
        sess = gpt2.start_tf_sess(force_cpu=True)
    else:
        sess = gpt2.start_tf_sess(gpu_frac=args['gpu_frac'])

    if args['train']:
        if not os.path.exists(f"models/{args['gpt2_model']}"):
            gpt2.download_gpt2(args['gpt2_model'])
        gpt2.finetune(sess,
            textpath,
            model_name=args['gpt2_model'],
            run_name=args["tune_model"],
            save_every=args['save_epochs'],
            steps=args['num_epochs'],
            target_loss=args['target_loss'],
            print_every=20,
            batch_size=args['batch_size'])
    # Generate
    elif args['generate']:
        start = perf_counter()
        gpt2.load_gpt2(sess, run_name=args["tune_model"])
        print(f'Model loaded in {(perf_counter()-start):.2f}s')
        start = perf_counter()
        gen = gpt2.generate(sess, run_name=args["tune_model"], temperature=args["temp"],
                            length=args["num_words"], return_as_list=True)
        if args['save']:
            with open(f'samples/generated_{args["tune_model"]}_{args["gpt2_model"]}.txt', 'w', encoding='utf-8') as fw:
                fw.write(gen[0])
        else:
            print(gen[0])
        print(f'Generated {len(gen[0])} words in {(perf_counter()-start):.2f}s')
