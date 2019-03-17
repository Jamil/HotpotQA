import sys
import os

log_file = sys.argv[1]
log = open(log_file).read()

losses = []
for line in log.split('\n'):
    if 'train loss' in line:
        loss_val = float(line.split('|')[-1].split(' ')[-1])
        losses.append([loss_val,''])
    elif 'dev loss' in line:
        loss_val = float(line.split('|')[3].split(' ')[-2])
        losses[-1][1] = loss_val

out_str = ''
for i, (t, v) in enumerate(losses):
    out_str += f'{i+1}\t{t}\t{v}\n'


os.system(f'echo "{out_str}" | pbcopy')
