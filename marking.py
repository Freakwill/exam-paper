# -*- coding: utf-8 -*-

import re
import pathlib
import copy

import numpy as np
import pandas as pd
# import pyparsing

import keytable
# import pypx
import classes


Path = pathlib.Path('~/Teaching/examsystem/').expanduser()

STUDENTKEYS = ['name', 'no', 'class', 'gender']

DAILY_RATIO = .2
EXAM_RATIO = .8


def adjust(r2, R=60):
    # rectification
    r1 = np.ceil((R-DAILY_RATIO*r2)/DAILY_RATIO)
    return r1

def count(nums, key=lambda x: x>=60):
    if isinstance(key, int):
        return len([a for a in nums if key<=a])
    elif isinstance(key, (tuple, list)):
        return len([a for a in nums if key[0]<=a<=key[1]])
    else:
        return len([a for a in nums if key(a)])

def ratio(nums, key=lambda x: x>=60):
    return count(nums, key) / len(nums)


class Marking:
    def __init__(self, class_, scores=None):
        self.class_=class_
        self.scores = scores
        self.diff = diff = 5
        self.dict ={'A+':100, 'A':100-diff, 'A-':100-2*diff, 'B+':100-3*diff, 'B':100-4*diff, 'B-':100-5*diff, '':0, np.NaN:0}
        # self.dict ={'A+':100, 'A':100-diff, 'A-':100-2*diff, 'B+':100-3*diff, 'B':100-4*diff, 'B-':100-5*diff, '':0}


    @staticmethod
    def fromClass(c):
        index = []
        body = []
        for s in c.members:
            index.append(s.no)
            body.append([v for k, v in s.score.items()])
        df = pd.DataFrame(body, index=pd.Index(index, name='no'), columns=s.score.keys())
        m = Marking(c, df)
        m.keys = s.score.keys()
        return m

    def toClass(self):
        c = self.class_
        for s in c:
            s.score.update({k:self.scores.loc[s.no, k] for k in ('daily', 'exam', 'total')})
        return c

    # basic methods
    def convert(self):
        def toint(s):
            if s in self.dict:
                return self.dict[s]
            elif isinstance(s, str) and 1<=len(s)<=3 and s.isdigit():
                return int(s)
            else:
                return s
        self.apply(toint)

    def apply(self, func):
        self.scores = pd.DataFrame([[func(a) for a in r] for k, r in self.scores.iterrows()], index=self.scores.index, columns=self.scores.columns)


    # def del_zero(self):
    #     self.scores = self.scores.loc[lambda df: df.exam>0]

    def __len__(self):
        if self.scores is None:
            return 0
        else:
            return len(self.scores)

    def copy(self):
        cpy = Marking(self.class_)
        if self.scores:
            cpy.scores = copy.deepcopy(self.scores)
        return cpy

    
    # advanced methods
    def get_exam(self):
        return self.scores['exam']

    def get_daily(self):
        cpy = self % (['daily','total','exam','extra'])
        return cpy.values

    def get_extra(self):
        cpy = self % (['daily','total','exam'])
        return cpy.values

    def __mod__(self, lst):
        sc = self.scores.copy()
        for item in lst:
            if item in sc.keys():
                sc.pop(item)
        return sc

    def calc_daily(self, lst):
        d = 0
        N = 0
        for a, b in zip(self.keys, lst):
            if a in {'class', 'name', 'no', 'gender', '班级', '姓名', '学号', '性别'}:
                continue
            if a.startswith('present') or a.startswith('签到'):
                d += b * 50
                N += 0.5
            elif a.startswith('c'):
                if b == 1:
                    d += 60
                else:
                    d += 100
                N += 1
            elif a != 'extra':
                w = self.weight.get(a, 1)
                d += b * w
                N += w
        return d / N


    def calc_diff(self):
        # self.ab ={'A+':100, 'A':100-diff, 'A-':100-2*diff, 'b+':100-3*diff, 'B':100-4*diff, 'B-':100-5*diff, '0':0} 
        exam = self.get_exam()
        Sm = 100
        for diff in np.linspace(3, 6, 50):
            ab = {'A+':100, 'A':100-diff, 'A-':100-2*diff, 'B+':100-3*diff, 'B':100-4*diff, 'B-':100-5*diff, '0':0}
            scores = [self.calc_daily([ab.get(a, a) for a in w]) for w in self.get_daily()]
            S = np.mean(tuple(map(lambda x,y:abs(x-y), scores, exam)))
            if S < Sm:
                Sm = S
                self.diff = diff
        self.dict = {'A+':100, 'A':100-self.diff, 'A-':100-2*self.diff, 'B+':100-3*self.diff, 'B':100-4*self.diff, 'B-':100-5*self.diff, '0':0}


    def stat(self, key='exam', show=True):
        # statistics
        scores = self.scores[key]
        mean = np.mean(scores)
        var = np.var(scores)
        rate = ratio(scores, 60)
        freq = count(scores, (90,100)), count(scores, (80,89)), count(scores, (70,79)), count(scores, (60,69)), count(scores, (0,59)), count(scores, (50,59)), count(scores, (0,49))
        dist = ratio(scores, (90,100)), ratio(scores, (80,89)), ratio(scores, (70,79)), ratio(scores, (60,69)), ratio(scores, (0,59)), ratio(scores, (50,59)), ratio(scores, (0,49))
        if show:
            print('class: %s(%d)'%(self.class_, self.class_.examSize))
            print('%s result:'%key)
            print('mean: %.2f\nvariance: %.2f\nrate: %.2f%%'%(mean, var, rate*100))
            print('frequency: %d, %d, %d, %d, %d(=%d+%d)'%freq)
            print('distribution: %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%(=%.2f%%+%.2f%%)'%tuple(d * 100 for d in dist))

        return mean, var, rate, freq, dist


    def report(self):
        # self.del_zero()
        print('------------REPORT-------------')
        exam_result = self.stat()
        print('--------------------------------')
        exam_result = self.stat('daily')
        print('--------------------------------')
        if 'total' in self.scores:
            final_result = self.stat('total')
        else:
            print('no total result')
        print('--------------------------------')


    def bar(self, label=None, folder=Path):
        if label is None:
            label = self.class_.name
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
        font = FontProperties(fname = "/usr/share/fonts/truetype/arphic/ukai.ttc", size=14)
        X = np.array([1,2,3,4,5])
        width = 0.75
        Y = self.stat(key='total', show=False)[4][:5]
        p = plt.bar(X, Y, width, facecolor='blue')

        plt.xticks(X + width/2, ('Excellent','Good','Average','Fair','Poor'))
        plt.legend((p[0],), (label,), loc='upper left')
        # plt.show()
        plt.savefig(folder + self.class_.name)

    def marking(self):
        # get total grade and correct daily grade
        total=[]
        daily=[]
        print(self.class_.name+':')
        print('no: homework(raw)    examination    final result')
        N=0
        # calculate diff, then convert
        self.calc_diff()
        self.convert()
        d = self.get_daily()

        x = self.scores['exam']
        for k, xk in enumerate(x):
            r1, r2 = self.calc_daily(d[k]), xk    # score1, score2
            if 'extra' in self.keys and self.scores.iloc[k]['extra']>0:
                r1 += self.scores.iloc[k]['extra']
            rr = r1 = np.ceil(np.mean(r1))  # calculate score1 and save it
            R=round(EXAM_RATIO*r1+DAILY_RATIO*r2)     # total score
       
            # case of low score
            if 50 <= r2 < 55:                 # 50-54
                R = 60
                r1 = adjust(r2, R)
            elif 55 <= r2 < 60:
                if R >= 61:    # 55-59
                    R = 61
                    r1 = adjust(r2, R)
                else:
                    R = 60
                    r1 = adjust(r2, R)
            elif 65 > r2 >= 60 and R < 60:    # 60-65
                R = 60
                r1 = adjust(r2, R)
            elif r2>=65 and R<=60:             # 65+
                R = 61
                r1 = adjust(r2, R)
            
            if r2 < 50:
                if r1-r2 > 15:    # too low
                    r1 = r2+15

            if 60 <= r2 <= 89:    # high score
                if r1-r2 > 10:
                    r1 = r2+10
                elif r2-20>r1:
                    r1 = r2-20
            elif 90 <= r2 < 95:
                r1 = max(r2,90)
            elif r2 >= 95:
                r1 = max(r2, 95)
                #elif r2-r1>5:
                   # r1=r2-5
            
            #     S=S+abs(r1-r2)
            R = round(EXAM_RATIO*r1+DAILY_RATIO*r2)        # re-calculate
            print('%d: %d(%d)    %d    %d'%(k+1,r1,rr,r2,R))
            if r2!=0:
                total.append(R)
                daily.append(r1)
            else:
                total.append(0)
                daily.append(0)
            if r1-r2>=20:
                N+=1
        print('attention: %d'%N)
        self.scores['daily'] = daily
        self.scores['total'] = total

    def write(self, fname=None, sheetname=None):
        if sheetname is None:
            sheetname = self.class_
        if fname is None:
            fname = self.class_
        self.scores.write(fname, sheetname)

namelists = pathlib.Path('~/Teaching/student lists').expanduser()
c = classes.Class.read_excel(namelists / 'web', sheetname='xinji16', skiprows=0)
c.filter()
m = Marking.fromClass(c)
m.weight ={}
m.marking()

m.report()
c = m.toClass()

c.to_excel(pathlib.Path('test'))

