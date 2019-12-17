import cPickle
import os

# Convert token ID to token
def convert(input_file, vocab_file, log_dir):
    f_vocab = open(vocab_file, 'r')
    id2word = {}
    cnt = 0
    for line in f_vocab.readlines():
        id2word[cnt] = line.strip()
        cnt += 1
    wordlen = cnt

    input = input_file
    output_file = os.path.join(log_dir, 'cotra_' + input_file.split('le')[-1])
    with open(output_file, 'w')as fout:
        with open(input) as fin:
            for line in fin:
                line = line.split()
                line = [int(x) for x in line]
                if all(i < wordlen for i in line) is False:
                    continue
                line = [id2word[x] for x in line]
                line = ' '.join(line) + '\n'
                fout.write(line)
