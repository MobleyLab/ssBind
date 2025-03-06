#!/usr/bin/env python

import sys
import argparse

class MyParser(argparse.ArgumentParser):
    """display help message for every error"""
    def error(self, message):
        self.print_help()
        sys.stderr.write('\nERROR:\n  %s\n' % message)
        sys.exit(2)

def get_args():
    parser = MyParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('fld',   help='filename of FLD file (ends in .maps.fld)')
    parser.add_argument('-t', '--newtype',  help='new type to add', required=True)
    args = parser.parse_args()
    return args

def parse_fld(filename):
    labels = []
    variables = []
    is_label_line = []
    is_variable_line = []
    lines = []
    map_prefixes = set()
    with open(filename) as f:
        index = 0
        for line in f:
            if line.startswith('label='):
                string = line.split()[0].split('=')[1]
                labels.append(string)
                is_label_line.append(True)
                is_variable_line.append(False)
                lines.append(None)
                last_label_index = index
            elif line.startswith('variable '):
                string = line.split()[2].split('=')[1]
                variables.append(string)
                is_label_line.append(False)
                is_variable_line.append(True)
                lines.append(None)
                map_prefixes.add(string.split('.')[0])
                last_variable_index = index
            else:
                is_label_line.append(False)
                is_variable_line.append(False)
                lines.append(line)
            index += 1
    assert(len(map_prefixes) == 1)
    map_prefix = list(map_prefixes)[0]
    return labels, variables, is_label_line, is_variable_line, lines, map_prefix, last_label_index, last_variable_index

def write_fld(labels, variables, is_label_line, is_variable_line, lines):
    n_lines = len(lines)
    text = ""
    label_index = 0
    variable_index = 0
    for i in range(n_lines):
        if is_label_line[i]:
            text += 'label=%s\n' % labels[label_index]
            label_index += 1
        elif is_variable_line[i]:
            text += 'variable %d file=%s filetype=ascii skip=6\n' % (variable_index+1, variables[variable_index])
            variable_index += 1
        else:
            text += lines[i]
    return text

def add_to_labels(newtype, labels, is_label_line, is_variable_line, last_label_index, lines):
    new_label = "%s-affinity" % newtype
    labels.insert(-2, new_label)
    lines.insert(last_label_index-1, None)
    is_label_line.insert(last_label_index-1, True)
    is_variable_line.insert(last_label_index-1, False)
    return labels, is_label_line, lines

def add_to_variables(newtype, map_prefix, variables, is_variable_line, is_label_line, last_variable_index, lines):
    new_map = "%s.%s.map" % (map_prefix, newtype)
    variables.insert(-2, new_map)
    lines.insert(last_variable_index-1, None)
    is_variable_line.insert(last_variable_index-1, True)
    is_label_line.insert(last_variable_index-1, False)
    return variables, is_variable_line, lines
    
if __name__ == '__main__':
    args = get_args()
    labels, variables, is_label_line, is_variable_line, lines, map_prefix, last_label_index, last_variable_index = parse_fld(args.fld) 
    if "%s-affinity" % args.newtype in labels:
        print("Not modifying %s because type %s is already there." % (args.fld, args.newtype))
        sys.exit()
    variables, is_variable_line, lines = add_to_variables(args.newtype, map_prefix, variables,
                                                          is_variable_line, is_label_line, last_variable_index, lines)
    labels, is_label_line, lines = add_to_labels(args.newtype, labels, is_label_line, is_variable_line, last_label_index, lines)
    new_fld_text = write_fld(labels, variables, is_label_line, is_variable_line, lines)

print(new_fld_text, file=open(args.fld, 'w'), end='')



