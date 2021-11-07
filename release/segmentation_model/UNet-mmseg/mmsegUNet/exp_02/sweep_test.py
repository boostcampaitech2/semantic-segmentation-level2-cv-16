
import argparse

parser = argparse.ArgumentParser(description='sweep_params')
parser.add_argument('--w_1', type=float,
                    help='an integer for the accumulator')
args = parser.parse_args()
print("w_1 ::: ", args.w_1)
# print("sum ::: ", sum(args.w_1))
print("="*50)
