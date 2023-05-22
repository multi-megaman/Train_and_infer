import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Spatial channel attention')
parser.add_argument('--labels_path', default='./data/train/train_labels.txt', type=str,
                    help='path to train or test labels')
parser.add_argument('--train_test', default='train', type=str, help='train or test')
args = parser.parse_args()


class Tree:
    def __init__(self, label, parent_label='None', id=0, parent_id=0, op='none'):
        self.children = []
        self.label = label
        self.id = id
        self.parent_id = parent_id
        self.parent_label = parent_label
        self.op = op


def convert(root: Tree, f):
    if root.tag == 'N-T':
        f.write(f'{root.id}\t{root.label}\t{root.parent_id}\t{root.parent_label}\t{root.tag}\n')
        for child in root.children:
            convert(child, f)
    else:
        f.write(f'{root.id}\t{root.label}\t{root.parent_id}\t{root.parent_label}\t{root.tag}\n')


# label = '../train_latex.txt'
label = args.labels_path
print()
print('label: ', label)
print()

out = args.train_test + '_hyb'

position = set(['^', '_'])
math = set(['\\frac', '\sqrt'])

with open(label, encoding="utf8") as f:
    lines = f.readlines()
num = 0
for line in tqdm(lines):
    # line = 'train_476.jpg	\\textcircled { 1 } + \\textcircled { 2 } = 2 x + 5 y + 3 x - 5 y = 1 0'
    name, *words = line.split()
    name = name.split('.')[0]
    print(name)

    parents = []
    root = Tree('root', parent_label='root', parent_id=-1)

    struct_list = ['\\frac', '\sqrt']

    labels = []
    id = 1
    parents = [Tree('<sos>', id=0)]
    parent = Tree('<sos>', id=0)

    print('line:', line)
    print('name:', name)
    print('words:', words)

    '''
    \\overrightarrow
    \\text circled
    \\xrightarrow
    '''

    for i in range(len(words)):
        a = words[i]
        if a == '\\limits':
            continue
        if i == 0 and words[i] in ['_', '^', '{', '}']:
            print(name)
            break

        elif words[i] == '{':
            if words[i - 1] == '\\frac':
                labels.append([id, 'struct', parent.id, parent.label])
                parents.append(Tree('\\frac', id=parent.id, op='above'))
                id += 1
                parent = Tree('above', id=parents[-1].id + 1)
            elif words[i - 1] == '}' and parents[-1].label == '\\frac' and parents[-1].op == 'above':
                parent = Tree('below', id=parents[-1].id + 1)
                parents[-1].op = 'below'

            # -------------------------------------------------------------------------------------------
            # SÓ PASSANDO DIRETO
            # elif words[i-1] == '\\overset':
            #     # labels.append([id, words[i], parent.id, parent.label])
            #     # parent = Tree(words[i],id=id)
            #     # id += 1
            #     continue
            # elif words[i-1] == '}' and words[i-4] == '\\overset':
            #     continue
            # USANDO A ESTRUTURA DO \\FRAC
            elif words[i - 1] == '\\overset':
                labels.append([id, 'struct', parent.id, '\\overset'])
                parents.append(Tree('\\overset', id=parent.id, op='sup'))
                id += 1
                parent = Tree('sup', id=parents[-1].id + 1)
            elif words[i - 1] == '}' and parents[-1].label == '\\overset' and parents[-1].op == 'sup':
                parent = Tree('below', id=parents[-1].id + 1)
                parents[-1].op = 'below'
                

                # USANDO A ESTRUTURA DO ^
            # elif words[i-1] == '\\overset':
            #     labels.append([id, 'struct', parent.id, parent.label])
            #     parents.append(Tree(words[i-2], id=parent.id))
            #     parent = Tree('sup', id=id)
            #     id += 1
            # -------------------------------------------------------------------------------------------

            elif words[i - 1] == '\sqrt':
                labels.append([id, 'struct', parent.id, '\sqrt'])
                parents.append(Tree('\sqrt', id=parent.id))
                parent = Tree('inside', id=id)
                id += 1
            elif words[i - 1] == '\\textcircled':
                labels.append([id, 'struct', parent.id, '\\textcircled'])
                parents.append(Tree('\\textcircled', id=parent.id))
                parent = Tree('inside', id=id)
                id += 1
            elif words[i - 1] == '\\overrightarrow':
                labels.append([id, 'struct', parent.id, '\\overrightarrow'])
                parents.append(Tree('\\overrightarrow', id=parent.id))
                parent = Tree('inside', id=id)
                id += 1
            elif words[i - 1] == '\\xrightarrow':
                labels.append([id, 'struct', parent.id, '\\xrightarrow'])
                parents.append(Tree('\\xrightarrow', id=parent.id))
                parent = Tree('inside', id=id)
                id += 1
            elif words[i - 1] == '\\xlongequal':
                labels.append([id, 'struct', parent.id, '\\xlongequal'])
                parents.append(Tree('\\xlongequal', id=parent.id))
                parent = Tree('inside', id=id)
                id += 1
            elif words[i - 1] == '\\dot':
                labels.append([id, 'struct', parent.id, '\\dot'])
                parents.append(Tree('\\dot', id=parent.id))
                parent = Tree('inside', id=id)
                id += 1
            elif words[i - 1] == '\\widehat':
                labels.append([id, 'struct', parent.id, '\\widehat'])
                parents.append(Tree('\\widehat', id=parent.id))
                parent = Tree('inside', id=id)
                id += 1
            elif words[i - 1] == '\\overline':
                labels.append([id, 'struct', parent.id, '\\overline'])
                parents.append(Tree('\\overline', id=parent.id))
                parent = Tree('inside', id=id)
                id += 1
            elif words[i - 1] == '\\mathop':
                labels.append([id, 'struct', parent.id, '\\mathop'])
                parents.append(Tree('\\mathop', id=parent.id))
                parent = Tree('inside', id=id)
                id += 1
            elif words[i - 1] == '\\boxed':
                labels.append([id, 'struct', parent.id, '\\boxed'])
                parents.append(Tree('\\boxed', id=parent.id))
                parent = Tree('inside', id=id)
                id += 1
            elif words[i - 1] == '\\Delta':
                labels.append([id, 'struct', parent.id, '\\Delta'])
                parents.append(Tree('\\Delta', id=parent.id))
                parent = Tree('inside', id=id)
                id += 1
            elif words[i - 1] == '\\ddot':
                labels.append([id, 'struct', parent.id, '\\ddot'])
                parents.append(Tree('\\ddot', id=parent.id))
                parent = Tree('inside', id=id)
                id += 1
            elif words[i - 1] == ']' and parents[-1].label == '\sqrt':
                parent = Tree('inside', id=parents[-1].id + 1)

            elif words[i - 1] == '^':
                if words[i - 2] != '}':
                    if words[i - 2] == '\sum':
                        labels.append([id, 'struct', parent.id, parent.label])
                        parents.append(Tree('\sum', id=parent.id))
                        parent = Tree('above', id=id)
                        id += 1

                    else:
                        labels.append([id, 'struct', parent.id, parent.label])
                        parents.append(Tree(words[i - 2], id=parent.id))
                        parent = Tree('sup', id=id)
                        id += 1

                else:
                    # labels.append([id, 'struct', parents[-1].id, parents[-1].label])
                    if parents[-1].label == '\sum':
                        parent = Tree('above', id=parents[-1].id + 1)
                    else:
                        parent = Tree('sup', id=parents[-1].id + 1)
                    # id += 1

            elif words[i - 1] == '_':
                if words[i - 2] != '}':
                    if words[i - 2] == '\sum':
                        labels.append([id, 'struct', parent.id, parent.label])
                        parents.append(Tree('\sum', id=parent.id))
                        parent = Tree('below', id=id)
                        id += 1

                    else:
                        labels.append([id, 'struct', parent.id, parent.label])
                        parents.append(Tree(words[i - 2], id=parent.id))
                        parent = Tree('sub', id=id)
                        id += 1

                else:
                    # labels.append([id, 'struct', parents[-1].id, parents[-1].label])
                    if parents[-1].label == '\sum':
                        parent = Tree('below', id=parents[-1].id + 1)
                    else:
                        parent = Tree('above', id=parents[-1].id + 1)
                    # id += 1
            else:
                print('unknown word before {', name, i, words[i - 1])
                print(parents[-1].label)
                input()
                # continue


        elif words[i] == '[' and words[i - 1] == '\sqrt':
            labels.append([id, 'struct', parent.id, '\sqrt'])
            parents.append(Tree('\sqrt', id=parent.id))
            parent = Tree('L-sup', id=id)
            id += 1
        elif words[i] == ']' and parents[-1].label == '\sqrt':
            labels.append([id, '<eos>', parent.id, parent.label])
            id += 1

        elif words[i] == '}':
            # print(i)
            # print(words)
            # print(words[i])
            # print(words[i - 1])
            # print(words[i + 1])

            if words[i - 1] != '}':
                labels.append([id, '<eos>', parent.id, parent.label])
                id += 1

            if i + 1 < len(words) and words[i + 1] == '{' and parents[-1].label == '\\frac' and parents[
                -1].op == 'above':
                continue
            # TESTE--------------------------------------------
            if i + 1 < len(words) and words[i + 1] == '{' and parents[-1].label == '\\overset' and parents[
                -1].op == 'sup':
                continue
            # -------------------------------------------------
            if i + 1 < len(words) and words[i + 1] in ['_', '^']:
                continue
            elif i + 1 < len(words) and words[i + 1] != '}':
                parent = Tree('right', id=parents[-1].id + 1)

            parents.pop()


        else:
            if words[i] in ['^', '_']:
                continue
            labels.append([id, words[i], parent.id, parent.label])
            parent = Tree(words[i], id=id)
            id += 1

    parent_dict = {0: []}
    for i in range(len(labels)):
        parent_dict[i + 1] = []
        parent_dict[labels[i][2]].append(labels[i][3])

    if not os.path.exists(out):
        os.makedirs(out)
    with open(f'{out}/{name}.txt', 'w', encoding="utf8") as f:
        for line in labels:
            id, label, parent_id, parent_label = line
            if label != 'struct':
                f.write(f'{id}\t{label}\t{parent_id}\t{parent_label}\tNone\tNone\tNone\tNone\tNone\tNone\tNone\n')
            else:
                tem = f'{id}\t{label}\t{parent_id}\t{parent_label}'
                tem = tem + '\tabove' if 'above' in parent_dict[id] else tem + '\tNone'
                tem = tem + '\tbelow' if 'below' in parent_dict[id] else tem + '\tNone'
                tem = tem + '\tsub' if 'sub' in parent_dict[id] else tem + '\tNone'
                tem = tem + '\tsup' if 'sup' in parent_dict[id] else tem + '\tNone'
                tem = tem + '\tL-sup' if 'L-sup' in parent_dict[id] else tem + '\tNone'
                tem = tem + '\tinside' if 'inside' in parent_dict[id] else tem + '\tNone'
                tem = tem + '\tright' if 'right' in parent_dict[id] else tem + '\tNone'
                f.write(tem + '\n')
        if label != '<eos>':
            f.write(f'{id + 1}\t<eos>\t{id}\t{label}\tNone\tNone\tNone\tNone\tNone\tNone\tNone\n')







