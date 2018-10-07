#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import jinja2

def tostr(x):
    if isinstance(x, str):
        return x
    elif isinstance(x, (tuple, list)):
        return tuple(map(tostr, x))
    elif hasattr(x, 'dumps'):
        return x.dumps()
    else:
        return str(x)

        
class BaseTemplate:
    r'''a wrapper of jinja.Template
    template for math problems or the proof

    __template: str, template of a problem
    parameter: dict, parameters of a problem

    __str__: str
    totex: str, translate it to tex grammar

    Examples:
    --------
    >>> bt = BaseTemplate('I love {{name}}', {'name':'Lily'})
    >>> print(bt)
    # I love Lily

    import mymat
    bt = BaseTemplate('Matrix A = {{matrix}}', {'matrix':mymat.MyMat('1,2;3,4')})
    print(bt.totex())  # never use print(bt)
    # Matrix A = \begin{bmatrix}
    # 1.0 & 2.0\\
    # 3.0 & 4.0
    # \end{bmatrix}
    '''

    def __init__(self, template='', parameter={}):
        self.template = template
        self.parameter = parameter

    # basic methods:
    @property
    def template(self):
        return self.__template

    @template.setter
    def template(self, val):
        if isinstance(val, str):
            self.__template = jinja2.Template(val)
        else:
            self.__template = val

    def __len__(self):
        return len(self.parameter)

    def __getitem__(self, key):
        return self.parameter[key]

    def __setitem__(self, key, value):
        self.parameter[key] = value

    def update(self, *args, **kwargs):
        self.parameter.update(*args, **kwargs)

    @classmethod
    def fromDict(cls, d):
        return cls(d['template'], d['parameter'])

    # format control:
    def format(self, parameter=None):
        # the core of the class
        # call render method
        if parameter is None:
            parameter = self.parameter
        return self.template.render(parameter)

    def __str__(self):
        # convert to string
        parameter = {key: tostr(val) for key, val in self.parameter.items()}
        return self.format(parameter)

    def totex(self):
        # convert to tex form
        parameter = {}
        for key, val in self.parameter.items():
            if hasattr(val, 'totex'):
                parameter[key] = val.totex()
            elif hasattr(val, 'dumps'):
                parameter[key] = val.dumps()
            else:
                parameter[key] = tostr(val)
        return self.format(parameter)

    def convert(self, func):
        # convert the values in parameter dictionary
        if callable(func):
            for key, val in self.parameter.items():
                self.parameter[key] = func(val)
        elif isinstance(func, dict):
            for key, val in self.parameter.items():
                self.parameter[key] = func[val]
        else:
            # func is a constant
            for key, val in self.parameter.items():
                self.parameter[key] = func

    def mask_with(self, keys={'answer'}, mask='***'):
        '''mask the some parameters
        In examination, you have to mask the answer
        In login interface, you have to mask the password
        '''
        for key in keys:
            self[key] = mask
