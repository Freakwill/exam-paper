#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import codecs
import json
import pathlib

import yaml

# path = pathlib.Path('or_fill.json')

def json2yaml(path):
    # .json  -> .yaml
    with path.open(encoding='utf-8') as fo:
        s = json.load(fo)
        y = yaml.dump(s, encoding='utf-8')
        y = y.decode("unicode-escape")

    yaml_path = path.with_suffix('.yaml')
    yaml_path.write_text(y, encoding='utf-8')

def yaml2json(path):
    # .yaml  -> .json
    y = path.read_text(encoding='utf-8')
    s = yaml.load(y)

    json_path = path.with_suffix('.json')
    with json_path.open('w', encoding='utf-8') as fo:
        j = json.dump(s, fo, ensure_ascii=False, indent=2)


path = pathlib.Path('or_truefalse.yaml')
yaml2json(path)