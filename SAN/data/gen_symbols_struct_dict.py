import os
import glob
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Spatial channel attention')
parser.add_argument('--train_labels_path', default='/train', type=str, help='path to train labels')
parser.add_argument('--test_labels_path', type=str, help='path to test labels')
args = parser.parse_args()

train_label_path = args.train_labels_path

if args.test_labels_path:
    test_label_path = args.test_labels_path
    test_labels = glob.glob(os.path.join(test_label_path, '*.txt'))

train_labels = glob.glob(os.path.join(train_label_path, '*.txt'))

words_dict = set(['<eos>', '<sos>', 'struct'])

with open('word.txt', 'w', encoding="UTF8") as writer:
    writer.write('<eos>\n<sos>\nstruct\n')
    i = 3
    for item in tqdm(train_labels):
        # print('item:', item)
        with open(item, encoding="utf8") as f:
            lines = f.readlines()
        # print('lines:', lines)
        for line in lines:
            # print('line:', line)
            # cid, c, pid, p, *r = line.strip().split()
            cid, c, *r = line.strip().split()
            # print(cid, c)
            # input()
            if c not in words_dict:
                # print('adding', c)
                words_dict.add(c)
                writer.write(f'{c}\n')
                i += 1
            # else:
            #     print('already added', c)

    if args.test_labels_path:
        for item in tqdm(test_labels):
            with open(item, encoding="utf8") as f:
                lines = f.readlines()
            for line in lines:
                # cid, c, pid, p, *r = line.strip().split()
                cid, c, *r = line.strip().split()
                if c not in words_dict:
                    # print('adding', c)
                    words_dict.add(c)
                    writer.write(f'{c}\n')
                    i += 1
                # else:
                # print('already added', c)

    writer.write('above\nbelow\nsub\nsup\nL-sup\ninside\nright')
print(i)


