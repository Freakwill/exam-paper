#!/usr/bin/env python3

r"""myfile.exam

To edit examination paper

-------------------------------
Path: examsystem/exam.py
Author: William/2016-01-02
"""

import collections
import pathlib
import datetime
import copy

import numpy as np

from pylatex import *
from pylatex.base_classes import *
from pylatex.utils import *

import pylatex_ext

from base import *

 
class Solve(Environment):
    """solve environment
    Solve(data='the solution')
    """
    escape = False
    content_separator = "\\\\\n"


class ExamPaper(pylatex_ext.XeDocument):
    """ExamPaper < Document
    """
    def __init__(self, subject='', title=None, *args, **kwargs):
        """
        Argument:
            subject: str, the name of the subject of the examination;
            title: str, will be given automaticly
        """
        super(ExamPaper, self).__init__(documentclass='ctexart', document_options='12pt,a4paper', *args, **kwargs)
        self.latex_name = 'document'
        self.escape = False
        self.subject = subject
        if title is None:
            import semester
            s = semester.Semester()
            title = '''浙江工业大学之江学院%s试卷'''%s.totex()
        self.title = title

        self.usepackage(('mathrsfs, amsfonts, amsmath, amssymb', 'enumerate', 'analysis, algebra', 'exampaper',
            'fancyhdr', 'geometry'))

        self.preamble.append(Command('geometry', 'left=3.3cm,right=3.3cm,top=2.3cm,foot=1.5cm'))
        self.preamble.append(Command('pagestyle', 'fancy'))
        self.preamble.append(Command('chead', NoEscape(Command('textbf', '浙 江 工 业 大 学 之 江 学 院 考 试 命 题 纸').dumps())))
        self.preamble.append(Command('cfoot', NoEscape(r'\footnotesize{第~\thepage~页~(共~\pageref{LastPage}~页)}')))
        self.preamble.append(Command('renewcommand', arguments=Arguments(NoEscape(r'\headrulewidth'), '0pt')))

        # header = PageStyle("header")      
        # with header.create(Foot("C")):
        #     ft = Command('footnotesize', arguments=NoEscape('第~\\thepage~页~(共~\pageref{LastPage}~页)'))
        #     header.append(ft)
        # self.preamble.append(header)

    def build(self):
        
        # the head of the paper
        self.make_head()
        
        # make problems
        if hasattr(self, 'fill'):
            self.make_fill()
        self.append('\n\n')
        if hasattr(self, 'truefalse'):
            self.make_truefalse()
        self.append('\n\n')
        if hasattr(self, 'choice'):
            self.make_choice()
        self.append('\n\n')
        if hasattr(self, 'calculation'):
            self.make_calculation()

    def make_head(self):
        self.append(Center(data=pylatex_ext.large(bold(NoEscape(self.title)))))
        
        table = Tabular('lclc')
        table.escape = False
        table.add_row(r'二级学院：&理学院&专业名称：&信计专业'.split('&'))
        table.add_row(r'课程名称：&Python程序设计&课程代码：&10800570'.split('&'))
        table.add_row(r'主讲教师：&宋丛威&&'.split('&'))

        table = Tabular('|c|c|c|c|c|c|')
        table.escape = False
        table.add_hline()
        table.add_row(r'\sws{题号} \sws{一} \sws{二} \sws{三} \sws{四} \sws{总评}'.split())
        table.add_hline()
        table.add_row((MultiRow(2, data='计分'), '', '', '', '', ''))
        table.add_empty_row()
        table.add_hline()
        self.append(Center(data=table))
        self.append(Command('thispagestyle', 'plain'))

    def make_fill(self):
        # make filling problems
        self.append('\\noindent 一、填空题 (每空 2 分, 共 20 分):')
        with self.create(Enumerate(options='1)')) as enum:
            enum.escape = False
            for p in self.fill:
                enum.add_item(NoEscape(p.totex()))

    def make_truefalse(self):
        # make true-false problem
        self.append('\\noindent 二、判断题 (每空 2 分, 共 10 分):')
        with self.create(Enumerate(options='1)')) as enum:
            enum.escape = False
            for p in self.truefalse:
                enum.add_item(NoEscape(p.totex()))

    def make_choice(self):
        # make choice problems
        self.append('\\noindent 三、选择题 (每空 2 分, 共 10 分):')
        with self.create(Enumerate(options='1)')) as enum:
            enum.escape = False
            for p in self.choice:
                enum.add_item(NoEscape(p.totex()))


    def make_calculation(self):
        # make calculation problems
        self.append('\\noindent 四、计算题 (每题 10 分, 共 60 分):')
        with self.create(Enumerate(options='1)')) as enum:
            enum.escape = False
            for p in self.calculation:
                if p.solution is None:
                    enum.add_item(NoEscape(p.totex() + '\n\n' + Command('vspace', '10cm').dumps()))
                else:
                    enum.add_item(NoEscape(p.totex()))

    def write(self, filename=None):
        if filename is None:
            filename = self.subject + 'exam'
        super(ExamPaper, self).write(filename)

    def topdf(self, filename=None):
        if filename is None:
            filename = self.subject + 'exam'
        super(ExamPaper, self).topdf(filename)


class Problem(BaseTemplate):
    # Problem class
    def __init__(self, template='', parameter={}, realm=None):
        super(Problem, self).__init__(template, parameter)
        self.realm = realm
        self.point = 10
        self.solution = None # :Solution

    def totex(self):
        solution = self.solution
        if solution:   # with solution
            if issubclass(solution, Solution):
                # solution is a class
                solution = solution.fromProblem(self)
            return super(Problem, self).totex() + '\n\n' + Solve(data=solution.totex()).dumps()
        else:  # without solution
            return super(Problem, self).totex()


class Solution(BaseTemplate):
    """Solution class
    
    solution of a problem
    
    Extends:
        BaseTemplate
    """

    @classmethod
    def fromProblem(cls, problem):
        obj = cls(parameter=problem.parameter)
        obj.genTemplate(problem)
        return obj

    def genTemplate(self, problem=None):
        # self.template = ''
        pass



# select problems from banks
USER_FOLDER = pathlib.Path('~').expanduser()
BANK_FOLDER = USER_FOLDER / 'Teaching/examsystem/bank'
PAPER_FOLDER = USER_FOLDER / 'Teaching/考试'


import json, yaml

class OtherSolution(Solution):

    def genTemplate(self, problem):
        self.template = problem.template

class OtherProblem(Problem):
    solution = OtherSolution
    mask = Command('mypar', '')
    mask_flag = False
    masked = {'answer'}

    def totex(self):
        if self.mask_flag:
            for k in self.masked:
                self[k] = self.mask
        return super(OtherProblem, self).totex()

    def __setstate__(self, state):
        self.template, self.parameter, answer = state['template'] + '~~{{answer}}', state.get('parameter', {}), state['answer']
        self.solution = None

    @staticmethod
    def random(filename='', n=1, encoding='utf-8', *args, **kwargs):
        # read n problems from yaml files (randomly)
        filename = (BANK_FOLDER / filename).with_suffix('.yaml')
        problems = yaml.load(filename.read_text(encoding=encoding))
        ret = []
        for _ in range(n):
            p = np.random.choice(problems)
            problems.remove(p)
            ret.append(p)
        return ret

    @classmethod
    def read_yaml(cls, filename, encoding='utf-8', *args, **kwargs):
        filename = (BANK_FOLDER / filename).with_suffix('.yaml')
        return yaml.load(filename.read_text(encoding=encoding))


class TrueFalseProblem(OtherProblem):

    def __setstate__(self, state):
        super(TrueFalseProblem, self).__setstate__(state)
        if 'answer' in state:
            if isinstance(state['answer'], bool):
                answer = 'true' if state['answer'] else 'false'
            else:
                answer = state['answer']
        else:
            answer = 'true'
        self.parameter.update({'answer': Command(answer)})


class ChoiceProblem(OtherProblem):

    def __setstate__(self, state):
        choices = '~~'.join(['(%s) %s'%(k, v) for k, v in state['options'].items()])
        self.template, self.parameter, answer = state['template'] + '~~{{answer}}\\\\\n' + choices, state.get('parameter', {}), state['answer']
        self.solution = None
        self.parameter.update({'answer':Command('mypar', answer)})


class FillProblem(OtherProblem):
    mask = Command('autolenunderline', '')

    def __setstate__(self, state):
        self.template, self.parameter, answer = state['template'], state.get('parameter', {}), state.get('answer', {})
        self.solution = None
        self.masked = set(answer.keys()) 
        self.parameter.update({k:Command('autolenunderline', NoEscape(v)) for k, v in answer.items()})


# with open('bank/python_choice.yaml', encoding='utf-8') as fo:
#     problem = yaml.load(fo)[0]
