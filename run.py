import argparse
import os
import sys
import warnings
from time import perf_counter

import gpt_2_simple as gpt2

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf


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
                        help='GPT2 base model name', default='117M')
    parser.add_argument('-fp', '--file_path', required=False, type=str,
                        help='Optional path to text file.', default='./')
    parser.add_argument('--batch_size', required=False, type=int,
                        help='Specify batch size', default=1)
    parser.add_argument('--gpu_frac', required=False, type=float,
                        help='Fraction of GPU memory to use.', default=0.8)
    parser.add_argument('--num_epochs', required=False, type=int,
                        help='Max epochs', default=1000)
    parser.add_argument('--save_epochs', required=False, type=int,
                        help='How often to save.', default=20)
    parser.add_argument('--verbose', required=False, type=int,
                        help='Verbosity level, 0 silent, 1 progress bar, 2 epoch only.', default=1)
    # Generation
    parser.add_argument('-n', '--num_words', required=False, type=int,
                        help='number of words to generate', default=20)
    parser.add_argument('--temp', required=False, type=float,
                        help='Temperature to use when generating.', default=0.5)
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
    sess = gpt2.start_tf_sess(gpu_frac=args['gpu_frac'])
    if args['train']:
        gpt2.finetune(sess,
            textpath,
            model_name=args['model_name'],
            run_name=args["tune_model"],
            sample_every=1000000,
            save_every=args['save_epochs'],
            steps=args['num_epochs'],
            batch_size=args['batch_size'])
    # Generate
    elif args['generate']:
        start = perf_counter()
        gpt2.load_gpt2(sess, run_name=args["tune_model"])
        print(f'Model loaded in {(perf_counter()-start):.2f}s')
        start = perf_counter()
        gen = gpt2.generate(sess, run_name=args["tune_model"], temperature=args["temp"],
                            length=args["num_words"], return_as_list=True)
        print(gen[0])
        print(f'Generated in {(perf_counter()-start):.2f}s')
