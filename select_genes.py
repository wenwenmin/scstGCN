import argparse
from utils import load_tsv, write_lines

def main(prefix, n_top):

    cnts = load_tsv(f'{prefix}cnts.csv')
    order = cnts.var().to_numpy().argsort()[::-1]
    names = cnts.columns.to_list()
    names_all = [names[i] for i in order]

    write_lines(names_all, f'{prefix}gene-names.txt')

