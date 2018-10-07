# -*- coding: utf-8 -*-
'''myfile.exam

To edit examination paper

-------------------------------
Path: mywork\exam.py
Author: William/2016-01-02
'''

import numpy as np

from pylatex import Tabular

import exam
import mymat
import mymat.linalg as la
import mymath.equation
import mymath.numerical
import mymath.eigenvalue


class NAProblem(exam.Problem):
    def __init__(self, template='', parameter={}):
        super(NAProblem, self).__init__(template, parameter)
        self.realm = '数值分析'
        self.point = 10

class NAExamPaper(exam.ExamPaper):
    def __init__(self, problems):
        realm = '数值分析 \\& 计算方法'
        year = 16
        super(NAExamPaper, self).__init__(problems, realm, year)


class InterpolationProblem(NAProblem):


    template = '给定插值点${{table}}$, 求拉格朗日插值多项式和牛顿插值多项式.'

    def __len__(self):
        return len(self['xdata'])

    @staticmethod
    def get_table(xdata, ydata):
        N = len(xdata)
        table = Tabular('|c|' + 'c|'*N)
        table.add_hline()
        table.add_row(('x_i',) + xdata)
        table.add_hline()
        table.add_row(('y_i',) + ydata)
        table.add_hline()
        return table.dumps()


    @staticmethod
    def random():
        x0 = np.random.randint(-9, 7)
        x1 = np.random.randint(x0+1, 9)
        x2 = np.random.randint(x1+1, 10)
        ip = InterpolationProblem.example(xdata=(x0, x1, x2), ydata=np.random.randint(-20, 20, 3))
        ip.solve()
        return ip

    @staticmethod
    def example(xdata=(-2, 2, 8), ydata=(6, 22, 166)):
        parameter = {'xdata':xdata, 'ydata':ydata, 'table':InterpolationProblem.get_table(xdata, ydata)}
        ip = InterpolationProblem(InterpolationProblem.template, parameter)
        ip.solve(strategy='newton')
        return ip


    @staticmethod
    def makeup(c=None):
        x0 = np.random.randint(-8, 7)
        x1 = np.random.randint(x0+1, 8)
        x2 = np.random.randint(x1+1, 9)
        if c:
            poly = np.poly1d(c)
        else:
            poly = np.poly1d([np.random.randint(-3,4), np.random.randint(-6,7), np.random.randint(-7,8)])
        ip = InterpolationProblem.example(xdata=(x0, x1, x2), ydata=poly((x0, x1, x2)))
        ip.solve(strategy='newton')
        return ip

    def solve(self, strategy='lagrange'):
        N = len(self)
        if strategy == 'lagrange':
            ind = [k for k in range(N)]
            t = ''
            for k in range(N):
                t += '{{ydata[%d]}}\\frac{'%k
                tempind = ind.copy()
                tempind.pop(k)
                for l in tempind:
                    t += '(x-{{xdata[%d]}})'%l
                t += '}{'+ str(np.prod([self['xdata'][k] - self['xdata'][l] for l in tempind])) + '}'
            template = '$L_n(x)=' + t + '$'
            self.solution = exam.Solution(template, self.parameter)
        else:
            import mymath.interpolate
            interp = mymath.interpolate.Interpolator1D(self['xdata'], self['ydata'], basis='newton')
            c = interp.getNewtonCoef()
            t = str(c[0]) + ' + '
            t += ' + '.join(str(c[k+1]) + ''.join('(x-{{xdata[%d]}})'%l for l in range(k+1)) for k in range(N-1))
            template = '牛顿多项式为$N_n(x)=' + t + '$'
            print(template)
            self.solution = exam.Solution(template, self.parameter)


class ApproximationProblem(NAProblem):

    def __init__(self, poly, deg=2):
        template='给定多项式{{poly}}的最佳{{deg}}次逼近.'
        parameter={'poly':poly, 'deg':deg}
        super(ApproximationProblem, self).__init__(template, parameter)


    @staticmethod
    def example(c=np.array([1,0,0,0,0]), a=-1, b=1):
        poly = np.poly1d(c)
        G = np.array([[b-a, (b**2-a**2)/2, (b**3-a**3)/3],[(b**2-a**2)/2, (b**3-a**3)/3,(b**4-a**4)/4],[(b**3-a**3)/3,(b**4-a**4)/4,(b**5-a**5)/5]])
        ap = ApproximationProblem(poly)
        eq = mymat.LinearEquation(G, np.array([(b**5-a**5)/5,(b**6-a**6)/6,(b**7-a**7)/7]))
        d = eq.solve()

        template = '法方程为\[{{normeq}}\]系数为{{coef}}, 最佳逼近多项式为${{poly}}$.'
        parameter = {'normeq':eq, 'coef':d, 'poly':np.poly1d(d)}
        ap.solution = exam.Solution(template, parameter)
        return ap


class LeastSquareProblem(NAProblem):

    def __init__(self, xdata, ydata, basis='1,x,x^2'):
        N = xdata.shape[0]

        table = Tabular('|c|' + 'c|'*N)
        table.add_hline()
        table.add_row(('$x_i$',) + xdata)
        table.add_hline()
        table.add_row(('$y_i$',) + ydata)
        table.add_hline()

        template = '用一组基$\\{{{basis}}\\}$拟合数据 %s.'%table.dumps()
        parameter = {'basis':basis, 'xdata':xdata, 'ydata':ydata}
        super(LeastSquareProblem, self).__init__(template, parameter)

    @staticmethod
    def example(xdata = np.array([-2,-1,0,1,2]), ydata = np.array([0,0.2,0.5,0.8,1])):
        basis = [1, lambda x:x]
        A = np.array([[b(x) if hasattr(b, '__call__') else b for b in basis] for x in xdata])
        lsp = LeastSquareProblem(xdata, ydata, basis='1,x,x^3')
        G = A.T @ A
        b = A.T @ ydata
        a = np.linalg.solve(G, b)
        neq = mymat.MyMat(G) | (mymat.MyMat(b)).T
        neq = neq.tolineq()
        template = '法方程为\n\[{{normeq}}\]\n系数为{{coef}}, 拟合函数为${{function}}$.'
        parameter = {'normeq':neq, 'coef':a, 'function':np.poly1d(a)}
        lsp.solution = exam.Solution(template, parameter)
        return lsp


class LUDecompsitionProblem(NAProblem):

    def __init__(self, matrix):
        template = '用三角分解法对矩阵${{matrix}}$进行分解'
        parameter = {'matrix':matrix}
        return super(LUDecompsitionProblem, self).__init__(template, parameter)

    @staticmethod
    def example():
        A = mymat.MyMat('1, 2, 3;2, 5, 8;3, 8, 14').tofrac()
        p = LUDecompsitionProblem(A)
        template = '分解过程为${{process}}$.\n\n答案为$L={{L}},U={{U}}$.'
        lu = la.LUDecompsition(A)
        p.solution = exam.Solution(template, parameter={'L':'L', 'U':'U'}, solver=lu)
        return p

    @staticmethod
    def makeup():
        matrix = mymat.MyMat.randint(3,3,lb=-10,ub=10)
        L = matrix.itril()
        U = matrix.triu()
        for k in range(3):
            if U[k,k] == 0:
                U[k,k] = np.random.randint(1, 10)
        A = (L * U).tofrac()
        p = LUDecompsitionProblem(A)
        template = '分解过程为${{process}}$.\n\n答案为$L={{L}},U={{U}}$.'
        lu = la.LUDecompsition(A)
        p.solution = exam.Solution(template, parameter={'L':L, 'U':U}, solver=lu)
        return p


class CholeskyProblem(NAProblem):

    def __init__(self, matrix):
        template = '用平方根法对矩阵${{matrix}}$进行分解.'
        parameter = {'matrix':matrix}
        return super(CholeskyProblem, self).__init__(template, parameter)

    @staticmethod
    def example():
        A = mymat.MyMat('1,6,5;6,37,23;5,23,75')
        p = CholeskyProblem(A)
        template = '分解过程为${{process}$.\n\n答案为$L={{L}}$.'
        cd = la.CholeskyDecompsition(A)
        p.solution = exam.Solution(template, parameter={'L':'L'}, solver=cd)
        return p

    @staticmethod
    def makeup():
        matrix = mymat.MyMat.randint(3,3,lb=-10,ub=10)
        L = matrix.itril()
        A = L * L.T
        p = CholeskyProblem(A)
        template = '分解过程为${{process}}$.\n\n答案为$L={{L}}$.'
        cd = la.CholeskyDecompsition(A)
        p.solution = exam.Solution(template, parameter={'L':L}, solver=cd)
        return p


class ChasingProblem(NAProblem):

    def __init__(self, matrix):
        template = '用追赶法对矩阵${{matrix}}$进行分解.'
        parameter = {'matrix':matrix}
        return super(ChasingProblem, self).__init__(template, parameter)

    @staticmethod
    def example():
        matrix = mymat.MyMat('1,2;3,3')
        p = ChasingProblem(matrix)

        template = '分解过程为${{process}}$.'
        cd = la.ChasingDecompsition(A)
        p.solution = exam.Solution(template, parameter={}, solver=cd)
        return p


class EquationProblem(NAProblem):

    @staticmethod
    def example(strategy='牛顿迭代法'):
        if strategy == '二分法':
            function = 'sin x -\\frac{1}{2}'
            func = lambda x: np.sin(x) -1/2
            lb, ub, tol=0, 1, 0.01
            template = '用{{strategy}}计算方程${{function}}=0$在$[{{lb}}, {{ub}}]$上的解. (至少迭代5次或误差小于{{tol}})'
            parameter = {'function':function,'strategy':'二分法','lb':lb, 'ub':ub, 'tol':tol}
            ep = EquationProblem(template, parameter)
            template = '过程如下:\n{{process}}\n方程${function}=0$的解约为${{answer}}$.'
            parameter = {'function':function}
            b = mymath.equation.Bisection(func, lb, ub, tol)
        else:
            func = lambda x: x * np.tan(x) - 1
            dfunc = lambda x: np.tan(x) + x / np.cos(x) ** 2
            tol=0.0001
            x0=1
            equation = 'x\\tan x=1'
            template = '用{{strategy}}计算方程${{equation}}$在${{x0}}$附近的解. (至少迭代5次或误差小于{{tol}})'
            parameter = {'equation':equation,'strategy':strategy,'x0':x0, 'tol':tol}
            ep = EquationProblem(template, parameter)
            template = '不动点迭代格式是${{iteration}}$, 过程如下:\n{{process}}\n方程${{equation}}$的解约为${{answer}}$.'
            parameter = {'iteration':'x-f(x)/f\'(x)','equation':equation}
            b = mymath.equation.NewtonIter(func, dfunc, x0, tol)
        ep.solution = exam.Solution(template, parameter, solver=b)
        return ep


class IntegralProblem(NAProblem):
    def __init__(self, parameter):
        template = '用复合{{n}}次{{strategy}}公式计算函数${{function}}$在$[{{lb}}, {{ub}}]$上的积分.'
        return super(IntegralProblem, self).__init__(template, parameter)


    @staticmethod
    def example(strategy='trapezoid'):
        function = '\sqrt{1-x^2}'
        func = lambda x: np.sqrt(1 - x**2)
        lb, ub = 0, 1
        if strategy == 'simpson':
            n = 4
            parameter = {'function':function,'strategy':'simpson','lb':lb, 'ub':ub, 'n':n}
            ip = IntegralProblem(parameter=parameter)
            template = '计算如下:$n={{n}},h={h}$,\n{{process}}\n函数${{function}}$的积分约为${{answer}}$.'
            parameter = {'function':function, 'n':n, 'h':(ub-lb)/n}
        else:
            n = 8
            parameter = {'function':function,'strategy':'trapezoid','lb':lb, 'ub':ub, 'n':n}
            ip = IntegralProblem(parameter=parameter)
            template = '计算如下:$n={{n}},h={h}$,\n{{process}}\n函数${{function}}$的积分约为${{answer}}$.'
            parameter = {'function':function, 'n':n, 'h':(ub-lb)/n}
        integral = mymath.numerical.Integration(func, lb, ub, n, strategy)
        ip.solution = exam.Solution(template, parameter, solver=integral)
        return ip

class PowerMethodProblem(NAProblem):
    def __init__(self, parameter):
        template = '用幂法计算矩阵$A={{matrix}}$的主特征值.'
        return super(PowerMethodProblem, self).__init__(template, parameter)

    @staticmethod
    def example():
        matrix = np.array([[1,1,0.5],[1,1,0.25],[0.5,0.25,2]])
        parameter = {'matrix':mymat.MyMat(matrix)}
        pmp = PowerMethodProblem(parameter=parameter)
        template = '计算过程如下:\n{{process}}\n矩阵${{matrix}}$的主特征值约为${{answer}}$.'
        parameter = {'matrix':'A','answer':''}
        pm = mymath.eigenvalue.PowerMethod(matrix)
        pmp.solution = exam.Solution(template, parameter, solver=pm)
        return pmp

    @staticmethod
    def random():
        matrix = mymat.MyMat.randint(3,3,lb=-6,ub=6)
        parameter = {'matrix':mymat.MyMat(matrix)}
        pmp = PowerMethodProblem(parameter=parameter)
        template = '计算过程如下:\n{{process}}\n矩阵${{matrix}}$的主特征值约为${{answer}}$.'
        parameter = {'answer':'l', 'matrix':'A'}
        pm = mymath.eigenvalue.PowerMethod(matrix)
        pmp.solution = exam.Solution(template, parameter, solver=pm)
        return pmp


class LinEqIterProblem(NAProblem):
    def __init__(self, parameter):
        template = '写出方程${{equation}}$的雅克比迭代格式和高斯-赛德尔迭代格式, 并判断收敛性.'
        return super(LinEqIterProblem, self).__init__(template, parameter)

    @staticmethod
    def random():
        A = mymat.MyMat.randint(3, 3, lb=-3, ub=4)
        for k in range(1, 4):
            A[k, k]=np.random.randint(10, 20)
        b = mymat.MyMat.randint(3, 1, lb=-10, ub=10)
        eq = mymat.LinearEquation(A, b)
        parameter = {'equation':eq}
        lp = LinEqIterProblem(parameter=parameter)
        template = '雅克比迭代公式为:\n\[{{jacobi}}\]\n高斯-赛德尔迭代公式为:\n\[{{gauss}}\]\n系数矩阵是对角占优矩阵, 故迭代收敛.'
        parameter = {'jacobi':eq.jacobiIter(), 'gauss':eq.jacobiIter()}    
        lp.solution = exam.Solution(template, parameter)
        return lp


if __name__=='__main__':

    # Problems=[InterpolationProblem,ApproximationProblem,IntegralProblem,CholeskyProblem,EquationProblem,PowerMethodProblem, LinEqIterProblem]
    # problems=[]
    # for P in Problems:
    #     if hasattr(P, 'makeup'):
    #         problems.append(P.makeup())
    #     elif hasattr(P, 'random'):
    #         problems.append(P.random())
    #     elif hasattr(P, 'example'):
    #         problems.append(P.example())

    # paper = NAExamPaper(problems=problems)
    # print(paper.totex())

    p = ApproximationProblem.example()
    print(p.totex())




