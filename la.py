# -*- coding: utf-8 -*-
'''exercise of linear algebra

To edit examination paper
'''

import re
import codecs

import numpy as np

from pylatex import *
from pylatex.base_classes import *
from pylatex.utils import *

import exam
import mymat

import scipy.linalg as LA


def var2tex(x, dollar=False):
    # x is like abc123
    r = re.compile('(\D+)(\d*)')
    m = r.match(x)
    d = m.group(2)
    if d == '':
        sub = ''
    elif len(d) == 1:
        sub = '_%s'%m.group(2)
    else:
        sub = '_{%s}'%m.group(2)
    if dollar:
        return '$%s%s$'%(m.group(1), sub)
    else:
        return '%s%s'%(m.group(1), sub)

class LAExamPaper(exam.ExamPaper):
    def __init__(self, *args, **kwargs):
        super(LAExamPaper, self).__init__(subject='线性代数', *args, **kwargs)


class ExercisePaper(LAExamPaper):

    def build(self):

        self.usepackage('mathrsfs, amsfonts, amsmath, amssymb')
        self.usepackage(('enumerate', 'analysis, algebra', 'exampaper'))
        header = PageStyle("header")
        with header.create(Foot("C")):
            ft = Command('footnotesize', arguments=NoEscape('第~\\thepage~页~(共~\pageref{LastPage}~页)'))
            header.append(ft)
        self.preamble.append(header)

        title = '''浙江工业大学之江学院%s课堂测试'''%self.subject
        self.append(Center(data=bold(title)))
        self.append(Center(data=[NoEscape('姓名\myline[4cm]~学号\myline[4cm]')]))
        
        if hasattr(self, 'fill'):
            self.make_fill()
        self.append('\n\n')
        if hasattr(self, 'truefalse'):
            self.make_truefalse()
        self.append('\n\n')
        if hasattr(self, 'calculation'):
            self.make_calculation()
    


class LAProblem(exam.Problem):
    def __init__(self, template='', parameter={}):
        super(LAProblem, self).__init__(template, parameter)
        self.point = 10

    def totex(self):
        # tex code of solution
        return super(LAProblem, self).totex()


class LASolution(exam.Solution):
    pass


class DeterminantProblem(LAProblem):

    @staticmethod
    def fromMatrix(matrix):
        template = '计算行列式${{determinant}}$.'
        determinant = Matrix(matrix, mtype='v')
        parameter = {'matrix':matrix, 'determinant':determinant}
        return DeterminantProblem(template, parameter)

    @staticmethod
    def randint(order=3, lb=-7, ub=8):
        m = mymat.MyMat.randint(order, order, lb, ub)
        return DeterminantProblem.fromMatrix(m)

    @staticmethod
    def random(order=3, elements=None):
        m = np.random.choice(elements, (order, order))
        return DeterminantProblem.fromMatrix(matrix.dumps())

    @staticmethod
    def example1():
        m = np.array([['a1', '0', 'a2', '0'], ['0', 'b1', '0', 'b2'], ['c1', '0', 'c2', '0'], ['0', 'd1', '0', 'd2']])
        return DeterminantProblem.fromMatrix(matrix.dumps())

    @staticmethod
    def example2():
        m = np.array([['a+x', 'a', 'a', 'a'], ['a', 'a+x', 'a', 'a'], ['a', 'a', 'a+x', 'a'], ['a', 'a', 'a', 'a+x']])
        return DeterminantProblem.fromMatrix(matrix.dumps())

class DeterminantSolution(LASolution):
    def totex(self):
        A = self['matrix']
        return '计算如下:\\\\\n$%d$'%(LA.det(A))


class LinearOperationProblem(LAProblem):

    @staticmethod
    def randint(m=3, n=3, lb=-4, ub=5):
        A = mymat.MyMat.randint(m, n, lb, ub)
        B = mymat.MyMat.randint(m, n, lb, ub)
        a = np.random.randint(-3,3)
        b = np.random.randint(-3,3)
        if b < 0:
            template = '计算${{a}}{{matrixA}} {{b}}{{matrixB}}$.'
        else:
            template = '计算${{a}}{{matrixA}} + {{b}}{{matrixB}}$.'
        parameter = {'matrixA':A, 'matrixB':B, 'a':a, 'b':b}
        return LinearOperationProblem(template, parameter)

class LinearOperationSolution(LASolution):
    def totex(self):
        aA = self['a'] * self['matrixA']
        bB = self['b'] * self['matrixB']
        C = aA +bB
        return '计算如下:\\\\\n$%s+%s=%s.$'%(aA.totex(),bB.totex(),C.totex())


class MatrixMultiplicationProblem(LAProblem):

    @staticmethod
    def randint(m=3, l=3, n=3, lb=-4, ub=5):
        A = mymat.MyMat.randint(m, l, lb, ub)
        B = mymat.MyMat.randint(l, n, lb, ub)
        template = '计算${{matrixA}}{{matrixB}}$.'
        parameter = {'matrixA':A, 'matrixB':B}
        return MatrixMultiplicationProblem(template, parameter)


class MatrixMultiplicationSolution(LASolution):
    def totex(self):
        A = self['matrixA']
        B = self['matrixB']
        AB = A * B
        return '计算如下:\\\\\n$%s%s=%s$'%(A.totex(), B.totex(), AB.totex())


class BlockMatrixProblem(LAProblem):
    @staticmethod
    def example():
        A1 = mymat.MyMat.randint(1, 3, -5, 6)
        A2 = mymat.MyMat.randint(2, 1, -5, 6)
        A = mymat.MyMat(LA.block_diag(A1, A2))
        B1 = mymat.MyMat.randint(3, 2, -5, 6)
        B2 = mymat.MyMat.zeros(1, 2)
        B = np.vstack((B1, B2))
        template = '用分块矩阵的方法计算${{matrixA}}{{matrixB}}$. (用线条注明分块方案)'
        parameter = {'matrixA':A.tofrac(), 'matrixB':B.tofrac()}
        return MatrixMultiplicationProblem(template, parameter)



class ElementaryTransformProblem(LAProblem):
    @staticmethod
    def randint(m=3, n=3, lb=-4, ub=5):
        A = mymat.MyMat.randint(r, n, lb, ub)
        template = '对${{matrixA}}$做初等变换${{transform1}}$,${{transform2}}$,${{transform3}}$.'
        parameter = {'matrixA':A, 'transform1':t1, 'transform2':t2}
        return ElementaryTransformProblem(template, parameter)

    @staticmethod
    def example1(m=3, n=4, lb=-4, ub=5):
        A = mymat.MyMat.randint(m, n, lb, ub)
        B = A.copy()
        B.row_transform1(1, 3).row_transform3(1, 2, 2)
        template = '${{matrixA}}$经初等变换${{transform}}$变成${{matrixB}}$.'
        parameter = {'matrixA':A, 'transform':'r_1\leftrightarrow r_3, r_1 + 2r_2', 'matrixB':B}
        return ElementaryTransformProblem(template, parameter)

    @staticmethod
    def example2(m=3, n=3, lb=-4, ub=5):
        A = mymat.MyMat.randint(m, n, lb, ub)
        B = A.copy()
        B.row_transform1(1, 2).row_transform3(1, 3, 2)
        template = '${{matrixA}}$经初等变换${{transform}}$变成${{matrixB}}$.'
        parameter = {'matrixA':A, 'transform':'[?,?]', 'matrixB':B}
        return ElementaryTransformProblem(template, parameter)

    @staticmethod
    def example3(m=3, n=4, lb=-4, ub=5):
        A = mymat.MyMat.randint(m, n, lb, ub)
        A[:, 1] = mymat.MyMat([[1],[0],[0]])
        B = A.copy()
        B.row_transform2(1, 2).row_transform3(2, 1, 2).row_transform3(3, 1, -1)
        template = '${{matrixB}}$经初等变换${{transform}}$变成${{matrixA}}$.'
        parameter = {'matrixA':A, 'transform':'?,?', 'matrixB':B}
        return ElementaryTransformProblem(template, parameter)


class ElementaryTransformSolution(LASolution):
    pass
    

class GaussEliminationProblem(LAProblem):

    @staticmethod
    def fromMatrix(A):
        template = '用初等变换将矩阵${{matrix}}$变成标准型.'
        parameter = {'matrix':A}
        p = GaussEliminationProblem(template, parameter)
        return p

    @staticmethod
    def example():
        A = mymat.MyMat('3, 0, -12, 3; -3, -4, 16, 17; -5, -4, 21, 30').tofrac()
        return GaussEliminationProblem.fromMatrix(A)

    @staticmethod
    def makeup(m=3, n=4, lb=-5, ub=5):
        A = mymat.MyMat.randint(m, n, lb, ub)
        L = A.tril()[:,:m]
        U = A.itriu()
        for k in range(m):
            if L[k,k] == 0:
                L[k,k] = np.random.randint(lb, ub)
        A = mymat.MyMat((L * U).tolist(), dtype=np.float64).tofrac()
        return GaussEliminationProblem.fromMatrix(A)


class GaussEliminationSolution(LASolution):

    def totex(self):
        A = self['matrix'].copy()
        tol = 0.0001
        r, c = A.shape
        m = min(r, c)
        tex = ''
        k = l = 1
        cols = []
        while k <= r and l < c:
            if all(A[k:r, l] == 0):
                l += 1
                continue
            #p = max(abs(A[k:r,k]))   # get the pivot
            #ind = abs(A[k:r,k]).tolist().index(p) + 1
            for index, p in enumerate(A[k:r, l].T.tolist()):
                # find the first nonzero element or the element with maximum abstract value
                if p != 0:
                    ind = index
                    break
            if abs(p) < tol:
                raise Exception('the pivot is too small!')
            else:
                if ind != 0:
                    t = ind + k
                    A[k, l:c], A[t, l:c] = A[t, l:c].copy(), A[k, l:c].copy()        # swap row r and row k
                    tex += '\\xrightarrow{{r_%d \\leftrightarrow r_%d}}'%(k, t)      # print rule (elementary transform)
                    tex += A.totex()                                                 # print matrix

                if A(k,l) != 1:
                    if l < c:
                        tex += '\\xrightarrow{{1/{1} * r_{0}}}'.format(k, A[k,k])   # rule_print
                        A[k, (l+1):c] /= A[k, l]; A[k, l] = 1
                    tex += A.totex() 

                if k < r:                                                            # [1  beta] __\ [1  beta         ]       
                    for s in range(k+1, r+1):                                        # [alpha A] --/ [0 A-alpha*beta]
                        if A[s,k] != 0:
                            tex += '\\xrightarrow{{r_{1} -{2}*r_{0}}}'.format(k, s, A[s,k])  # print rule (elementary transform)
                    A[k+1:r, l+1:c] -= A[k+1:r, l] * A[k, l+1:c]
                    A[k+1:r, l] = 0
                    tex += A.totex()

                cols.append(l)
                k += 1; l += 1
        
        # Back Substitution Process
        tex1 = ''
        R = len(cols)  # rank(A)
        for l in cols[-1:0:-1]:
            for k, a in enumerate(A[1:R-1, l].T.tolist(), 1):
                if a != 0:
                    tex1 += '\\xrightarrow{{r_{1} - {3}*r_{0}}}'.format(R, k, l, a)
                    if l<c:
                        A[k, l+1:c] -= a * A[R, l+1:c]
            A[1:R-1, l] = 0
            tex1 += A.totex()
            R -= 1
        return '初等变换过程如下:\n消元过程:$%s$\n\n回代过程:$%s$'%(tex, tex1)


class LinearEquationProblem(LAProblem):
        # solution = LinearEquationSolution

    @staticmethod
    def fromEquation(lineq):
        parameter = {'equation':lineq}
        p = LinearEquationProblem('求解线性方程组${{equation}}$.', parameter)
        # p.solution = LinearEquationSolution(A)
        return p

    @staticmethod
    def example():
        A = mymat.MyMat([[3, -3, 3], [-3, 4, 0], [3, -4, -4]])
        b = mymat.MyMat([1, -1, 1])
        return LinearEquationProblem.fromEquation(mymat.LinearEquation(A, b))

    @staticmethod
    def makeup(m=3, n=3, lb=-5, ub=5):
        A = mymat.MyMat.randint(m, n, lb, ub)
        L = A.tril()[:,:m]
        U = A.itriu()
        for k in range(m):
            if L[k,k] == 0:
                L[k,k] = np.random.randint(lb, ub)
        A = mymat.MyMat((L * U).tolist(), dtype=np.float64).tofrac()
        b = mymat.MyMat.randint(3, 1, lb, ub).tofrac()
        return LinearEquationProblem.fromEquation(mymat.LinearEquation(A, b))

class LinearEquationSolution(LASolution):
    def totex(self):
        le = self['equation']
        Ab = le.A_b
        ges = GaussEliminationSolution(parameter={'matrix':Ab.tofrac()})
        return ges.totex() + '答：方程组的解为$x = {{answer}}$.'

    def answer(self):
        le = self['equation']
        return le.solve()

class InverseMatrixProblem(LAProblem):

    @staticmethod
    def fromMatrix(A):
        parameter = {'matrix':A}
        p = InverseMatrixProblem("求${{matrix}}$的逆矩阵.", parameter)
        # p.solution = LinearEquationSolution(A)
        return p

    @staticmethod
    def example():
        A = mymat.MyMat('-1 & 2 & -1\\-2 & 2 & 2\\-2 & 5 & -3').tofrac()
        return InverseMatrixProblem.fromMatrix(A)

    @staticmethod
    def makeup(m=3, lb=-3, ub=3):
        A = mymat.MyMat.randint(m, m, lb, ub)
        L = A.tril()[:,:m]
        U = A.itriu()
        for k in range(m):
            if L[k,k] == 0:
                L[k,k] = np.random.randint(lb, ub)
        A = mymat.MyMat((L * U).tolist(), dtype=np.float64)
        return InverseMatrixProblem.fromMatrix(A)


class InverseMatrixSolution(LASolution):
    def totex(self):
        A = self['matrix']
        m, n = A.shape
        AI = np.hstack((A, mymat.MyMat.eye(n))).tofrac()
        ges = GaussEliminationSolution(parameter={'matrix':AI})
        return ges.totex()

