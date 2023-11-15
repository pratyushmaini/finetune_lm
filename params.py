import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='commonsense_qa')
    parser.add_argument('--model_path', type=str, default='gpt2-medium')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--total_batch_size', type=int, default=16)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--num_save_steps', type=int, default=5)
    parser.add_argument('--warmup_ratio', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_strategy', type=str, default='steps')
    parser.add_argument('--report_to', type=str, default='none')
    parser.add_argument('--do_train_eval', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--n_shot', type=int, default=None)

    return parser

