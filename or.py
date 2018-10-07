# -*- coding: utf-8 -*-
'''
To edit examination paper for operation research

'''

import fractions
import os.path

import numpy as np

import exam
import pyopr
import mymath.numerical

from pylatex import *
from pylatex.base_classes import *
from pylatex.utils import *

FOLDER = os.path.expanduser('~/Teaching/考试/运筹学考试')


class ORProblem(exam.Problem):
    def __init__(self, template='', parameter={}):
        super(ORProblem, self).__init__(template, parameter)
        self.realm = '运筹与优化'
        self.point = 10

class ORSolution(exam.Solution):
    def __init__(self, template=None, parameter={}):
        if template is None:
            template = '{{process}}\n\n答：最优解是{{optimum}}, 目标函数值{{value}}.'        
        super(ORSolution, self).__init__(template, parameter)


class LinearProgrammingProblem(ORProblem):

    @staticmethod
    def fromMatrix(c, A, b):
        lp = pyopr.LinearProgramming(c, A, b)
        lp.fraction = True
        template = '用单纯形法解线性规划.（只需给出一个最优解, 规范绘制单纯形表）\n{{lp}}'
        parameter = {'lp':lp}
        return LinearProgrammingProblem(template, parameter)

    @staticmethod
    def example():
        # c, A, b
        c = np.array([10,15,12])
        A = np.array([[5, 3, 1], [-5, 6, 15]])
        b = np.array([9, 15])
        return LinearProgrammingProblem.fromMatrix(c, A, b)


class LinearProgrammingSolution(ORSolution):

    @classmethod
    def fromProblem(cls, problem):
        lp = problem['lp']
        lp.standard()

        if lp.abase == []:
            st = lp.toTableau()
            sequence = st.sequence()
            process = '\\\\\n'.join([s.totex() for s in sequence])
        else:
            st = lp.toTableau()
            sequence = st.sequence()
            process = '\\\\\n'.join([s.totex() for s in sequence])
            A = np.delete(st.A, lp.abase, 1)
            if set(st.base).isdisjoint(lp.abase):
                st = pyopr.SimplexTableau(lp.cs, A, st.get_x()[st.base], st.base)
                st = pyopr.SimplexTableau(lp.cs, lp.As, lp.bs, lp.base)
                process += '\\\\\n'.join([s.totex() for s in st.sequence()])
            else:
                raise Exception('no feasible solution!')
        st = sequence[-1]
        base = st.base
        xx = st.get_x()
        x = xx[:lp.nvar]
        z = lp.get_value(x)
        if lp.max_min == 'min':
            z = -z
            y = st.get_dualb()
        else:
            y = -st.get_dualb()
        # solution information: optimal solution, solution with slack, optimal base, the value of goal function, dual variable
        parameter={'optimum':x, 'optimum_slack':xx, 'optimal_base':base, 'value':z, 'dual':y, 'process':process}
        return cls(parameter=parameter)
        
class DualSimplexProblem(ORProblem):

    @staticmethod
    def fromMatrix(c, A, b):
        lp = pyopr.LinearProgramming(c, A, b, max_min='min')
        lp.fraction = True
        template = '用对偶单纯形法解线性规划.（只需给出一个最优解, 规范绘制单纯形表）\n{{lp}}'
        parameter = {'lp':lp}
        return DualSimplexProblem(template, parameter)


    @staticmethod
    def example():
        # c, A, b
        c = np.array([15, 24, 5])
        A = np.array([[0, -6, -1], [-5, -2, -1]])
        b = np.array([-2, -1])
        return DualSimplexProblem.fromMatrix(c, A, b)

class DualSimplexSolution(ORSolution):

    @classmethod
    def fromProblem(cls, problem):
        lp = problem['lp']

        st = lp.toDualTableau()
        sequence = st.sequence()
        process = '\\\\\n'.join([s.totex() for s in sequence])

        st = sequence[-1]
        base = st.base
        xx = st.get_x()
        x = xx[:lp.nvar]
        z = lp.get_value(x)
        # z = -z
        # y = st.get_dualb()
        # solution information: optimal solution, solution with slack, optimal base, the value of goal function, dual variable
        parameter={'optimum':x, 'optimum_slack':xx, 'optimal_base':base,'process':process}
        return cls(parameter=parameter)

class TransportationIBFSProblem(ORProblem):
 
    default_template = '根据运输问题的运价表, 用{{strategy}}给出初始解. \n {{tp}}'

    @staticmethod
    def fromMatrix(c, a, b, strategy='minimum'):
        parameter = {'tp':pyopr.TransportationIBFSProblem(c, a, b), 'strategy':strategy}
        return TransportationIBFSProblem(template, parameter)


    @staticmethod
    def random(strategy='minimum'):
        parameter = {'tp':pyopr.TransportationProblem.random(), 'strategy':strategy}
        return TransportationIBFSProblem(TransportationIBFSProblem.default_template, parameter)

class TransportationIBFSSolution(ORSolution):

    @classmethod
    def fromProblem(cls, problem):
        tp = problem['tp']
        base, x = tp.get_ibfs()
        tt = tp.toTableau()
        z = tp.get_value(x)
        return cls(template='{{tt}}运价为{{z}}', parameter={'z':z, 'tt':tt})

class TransportationUVProblem(ORProblem):
 
    default_template = '根据运输问题的运价表和可行解, 用位势法计算检验值, 并改进解. \n {{tt}}'

    @staticmethod
    def random(strategy='minimum'):
        tp = pyopr.TransportationProblem.random()
        base, x= tp.get_ibfs()
        tt = tp.toTableau()
        parameter = {'tt':tt, 'strategy':strategy}
        return TransportationUVProblem(TransportationUVProblem.default_template, parameter)

class TransportationUVSolution(ORSolution):

    @classmethod
    def fromProblem(cls, problem):
        tt = problem['tt']
        u, v = tt.get_uv()
        return cls(template='位势为u={{u}}, v={{v}}', parameter={'u':u, 'v':v})


class GoalProgrammingProblem(ORProblem):

    @staticmethod
    def example():
        W = np.array(np.matrix('1 0 0 0 0 0;0 0 1 0 0 0;0 0 0 0 1 0;0 0 0 0 0 1'), dtype=np.int)
        GA = np.array([[1,2],[1,-2],[0,1]], dtype=np.float64)
        Gb = np.array([6,4,2], dtype=np.float64)
        gp = pyopr.GoalProgramming(W, GA, Gb)
        gp.fraction = True
        template = '某目标规划问题有如下要求（从上到下优先性递减）, 请写出合理的目标规划模型, 和初始单纯形表.{{gp}}'
        parameter = {'gp':gp}
        return GoalProgrammingProblem(template, parameter)

class GoalProgrammingSolution(ORSolution):
    @classmethod
    def fromProblem(cls, problem):
        gp = problem['gp']
        t = gp.toTableau()
        return cls(template='初始单纯形表为: \n%s'%t.totex(), parameter=problem.parameter)

class BuildGoalProgrammingProblem(ORProblem):

    @staticmethod
    def example():
        W = np.array(np.matrix('1 0 0 0 0 0;0 0 1 0 0 0;0 0 0 0 1 0;0 0 0 0 0 1'), dtype=np.int)
        GA = np.array([[1,2],[1,-2],[0,1]], dtype=np.float64)
        Gb = np.array([6,4,2], dtype=np.float64)
        gp = pyopr.GoalProgramming(W, GA, Gb)
        gp.fraction = True
        template = '某目标规划问题有如下要求（从上到下优先性递减）, 请写出合理的目标规划模型, 和初始单纯形表.{{gp}}'
        parameter = {'gp':gp}
        return BuildGoalProgrammingProblem(template, parameter)

class BuildGoalProgrammingSolution(ORSolution):
    @classmethod
    def fromProblem(cls, problem):
        gp = problem['gp']
        t = gp.toTableau()
        return cls(template='初始单纯形表为: \n%s'%t.totex(), parameter=problem.parameter)

class Search1DProblem(ORProblem):

    @staticmethod
    def example(strategy='f'):
        function = '|\ln t-1|'
        func = lambda x: np.abs(np.log(x)-1)
        n = 6
        lb, ub = 1, 3
        template = '用{{strategy}}一维搜索定义在$[{{lb}},{{ub}}]$上的单峰函数$f(t)={{function}}$的极小值.（要求计算$n={{n}}$次函数值, 提示：利用对称性）'
        parameter = {'lb':lb,'ub':ub, 'function':function, 'strategy':strategy, 'func':func, 'n':n}
        return Search1DProblem(template, parameter)
        
    @staticmethod
    def example2(strategy='f'):
        # function = '|\ln x-1|'
        function = '|\sqrt{t}-2|'
        func = lambda x: np.abs(np.sqrt(x)-2)
        n = 6
        lb, ub = 3, 5
        template = '用{{strategy}}一维搜索定义在$[{{lb}},{{ub}}]$上的单峰函数$f(t)={{function}}$的极小值.（要求计算$n={{n}}$次函数值, 提示：利用对称性）'
        parameter = {'lb':lb,'ub':ub, 'function':function, 'strategy':strategy, 'func':func, 'n':n}
        return Search1DProblem(template, parameter)

    @staticmethod
    def example3(strategy='f'):
        # function = '|\ln x-1|'
        function = '\max\{t^2, t+1\}'
        func = lambda x: np.max((x**2, x+1))
        n = 6
        lb, ub = -1, 0
        template = '用{{strategy}}一维搜索定义在$[{{lb}},{{ub}}]$上的单峰函数$f(t)={{function}}$的极小值.（要求计算$n={{n}}$次函数值, 提示：利用对称性）'
        parameter = {'lb':lb,'ub':ub, 'function':function, 'strategy':strategy, 'func':func, 'n':n}
        return Search1DProblem(template, parameter)

class Search1DSolution(ORSolution):

    @classmethod
    def fromProblem(cls, problem):
        lb = problem['lb']
        ub = problem['ub']
        func = problem['func']
        strategy = problem['strategy']
        n = problem['n']
        s1d = mymath.numerical.Searcher1D(func, lb, ub, iterations=n, strategy=strategy)
        f = func
        a, b = lb, ub
        d = b - a
        mu = s1d.get_mu()
        t1 = b - mu * (b-a)
        t1p = a + b - t1
        f1 = f(t1)
        f1p = f(t1p)
        epsilon = 0.001
        table = Tabular('c|c|c|c|c|c|c')
        table.escape = False
        table.add_hline()
        table.add_row(('$k$', '$a$', '$t_k$', '$t_k\'$', '$b$', '$f(t_k)$', '$f(t_k\')$'))
        table.add_hline()
        table.add_row((0, *('%.4f'%x for x in (a, t1, t1p, b, f1, f1p))))
        for k in range(1, n-2):
            if f1 < f1p:
                b = t1p
                t1p = t1
                t1 = a + b - t1p
                f1p = f1
                f1 = f(t1)
            else:
                a = t1
                t1 = t1p
                t1p = a + b - t1
                f1 = f1p
                f1p = f(t1p)
            table.add_row((k, *('%.4f'%x for x in (a, t1, t1p, b, f1, f1p))))
        if f1 < f1p:
            b = t1p
        else:
            a = t1
        t1 = (a + b)/2
        t1p = t1 + epsilon*(b-a)
        f1 = f(t1)
        f1p = f(t1p)
        table.add_row((n-2, *('%.4f'%x for x in (a, t1, t1p, b, f1, f1p))))
        if f1 < f1p:
            ans = Math(data=NoEscape('x^*\in [%0.4f,  %0.4f), f(%0.4f) = %0.4f'%(a, t1p, t1, f1)))
        elif f1 > f1p:
            ans = Math(data=NoEscape('x^*\in (%0.4f,  %0.4f], f(%0.4f) = %0.4f'%(t1, b, t1p, f1p)))
        else:
            s = (t1 + t1p) / 2
            fs = f(s)
            ans = Math(data=NoEscape('x^*\in (%0.4f,  %0.4f), f(%0.4f) = %0.4f'%(t1, t1p, s, fs)))
        table.add_hline()
        return cls(table.dumps() + '\n\n' + ans.dumps())


class GameProblem(ORProblem):
    pass

class MatrixGameProblem(GameProblem):
    @staticmethod
    def fromMatrix(A):
        game = pyopr.MatrixGame(A)
        template = '用图解法计算矩阵对策{{game}}的最优混合策略.'
        parameter = {'game':game}
        return MatrixGameProblem(template, parameter)

    @staticmethod
    def example():
        A = np.array([[2, 3, 11], [7, 5, 2]])
        return MatrixGameProblem.fromMatrix(A)

    @staticmethod
    def example1():
        player1 = pyopr.Player('下属', strategies=['欺瞒', '坦诚'])
        player2 = pyopr.Player('上司', strategies=['检查', '不检查', '恐吓'])
        players = [player1, player2]
        A = np.array([[-5, 2, 3], [2, -1, -2]])
        game = pyopr.MatrixGame(A, players)
        game.fraction = True
        template = '用图解法计算矩阵对策 (“上有政策下有对策”博弈) {{game}}的最优混合策略.'
        parameter = {'game':game}
        return MatrixGameProblem(template, parameter)

    @staticmethod
    def example2():
        player1 = pyopr.Player('受方', strategies=['躲开', '不躲'])
        player2 = pyopr.Player('攻方', strategies=['不进攻', '进攻', '佯攻'])
        players = [player1, player2]
        A = np.array([[-1, 3, -2], [2, -4, 3]])
        game = pyopr.MatrixGame(A, players)
        template = '用图解法计算矩阵对策 (“你攻我受”博弈) {{game}}的最优混合策略.'
        parameter = {'game':game}
        return MatrixGameProblem(template, parameter)

    @staticmethod
    def example3():
        player1 = pyopr.Player('足球前锋', strategies=['朝左踢', '右'])
        player2 = pyopr.Player('守门员', strategies=['朝左扑', '右'])
        players = [player1, player2]
        A = np.array([[-1, 1], [1, -1]])
        game = pyopr.MatrixGame(A, players)
        template = '用图解法计算矩阵对策{{game}}的最优混合策略.'
        parameter = {'game':game}
        return MatrixGameProblem(template, parameter)

class MatrixGameSolution(ORSolution):
    @classmethod
    def fromProblem(cls, problem):
        game = problem['game']
        res = game.solve()
        template = '图解法计算如下{{figure}}, 最优混合策略{{optimum}}, 赢得值为{{value}}.'
        parameter = res
        return cls(template, parameter)

class DecisionProblem(ORProblem):
    @staticmethod
    def example(criterion=pyopr.Criterion('maximin')):
        matrix = np.array([[-100, -50, 0, 100, 200], [-50, -20, 10, 50, 90], [0, 0, -10, -20, -40], [100, 80, 40, 40, 20]])
        template = '之江老师在学校附近购买住房后准备装修. 给定决策表 (数值代表收益估计) {{table}}, 根据{{criterion}}, 给出最优方案.'
        decisionMaker = pyopr.Player('之江老师', strategies=['精装', '简装', '不装', '卖房'])
        states = ['只工作一年', '两年', '三年', '四年', '五年']
        model = pyopr.DecisionModel(matrix, decisionMaker, states)
        parameter = {'table':model, 'criterion':criterion, 'matrix':matrix}
        return DecisionProblem(template, parameter)

    @staticmethod
    def example2(criterion=pyopr.Criterion('maximin')):
        template = '给定炒股决策表 (数值代表股票收益) {{table}}, 根据{{criterion}}, 给出最优方案.'
        decisionMaker = pyopr.Player('巴菲特', strategies=['重仓', '加仓', '减仓', '全抛'])
        states = ['大涨', '小涨', '横盘', '小跌', '大跌']
        matrix = np.array([[100, 10, 0, -10, -100], [20, 5, 0, -5, 20], [10, 1, 0, -1, -10], [-2, -1, 0, 0, 0]])
        model = pyopr.DecisionModel(matrix, decisionMaker, states)
        parameter = {'table':model, 'criterion':criterion, 'matrix':matrix}
        return DecisionProblem(template, parameter)

    @staticmethod
    def example3(criterion=pyopr.Criterion('maximin')):
        template = '给定买保险决策表 (数值代表去掉保费后的保险收益) {{table}}, 根据{{criterion}}, 给出最优方案.'
        decisionMaker = pyopr.Player('旅客', strategies=['不买保险', '买医疗保险', '买死亡保险', '双保险'])
        states = ['平安', '伤残', '死亡']
        matrix = np.array([[0, -100, -1000], [-10, -1, -1010], [-100, -100, -50], [-110, -1, -60]])
        model = pyopr.DecisionModel(matrix, decisionMaker, states)
        parameter = {'table':model, 'criterion':criterion, 'matrix':matrix}
        return DecisionProblem(template, parameter)


class DecisionSolution(ORSolution):
    @classmethod
    def fromProblem(cls, problem):
        model = problem['table']
        criterion = problem['criterion']
        res = model.solve()
        template = '根据{{criterion}}, 最优方案为{{choice}}, 效用值{{value}}'
        parameter = problem.parameter
        parameter.update(res)
        return cls(template, parameter)



class DynamicProgrammingProblem(ORProblem):
    pass


class ShortestPathProblem(DynamicProgrammingProblem):
    default_template = "用动态规划逆序法（或顺序法）求出初始状态{{start}}到目标状态{{target}}的最短路 (其中一条) 和最短路长. 将指标在下图的顶点旁边. \n {{figure}}"
        
    @staticmethod
    def fromStages(start, target, stages, policies, imname):
        import graphx
        import matplotlib.pyplot as plt
        f = graphx.StageGraph.draw(start, target, stages, policies, label_pos=0.7, condition=lambda x:x>=0)
        g, pos, labels = graphx.StageGraph.fromStages(start, target, stages, policies, condition=lambda x:x>=0)
        f.savefig(imname)
        plt.close('all')
        fig = Figure(position='h')
        fig.add_image(imname)
        fig.add_caption('动态规划模型图')
        parameter = {'start': '$A_1$', 'terminal':'$F_1$', 'figure':fig.dumps(), 'graph':g}
        return ShortestPathProblem(ShortestPathProblem.default_template, parameter)

    @staticmethod
    def example():
        import graphx
        import matplotlib.pyplot as plt
        stages = ['A','B','C','D','E','F']
        start = 'A1'
        target = 'F1'
        policies = [np.array([[4,5]]), np.array([[2,3,6,-1], [-1,8,7,7]]), np.array([[5,8,-1], [4,5,-1], [-1,3,4],[-1,8,4]]), np.array([[3,5],[6,2],[1,3]]),np.array([[4],[3]])]
        return ShortestPathProblem.fromStages(start, target, stages, policies, imname='dp171.eps')

    @staticmethod
    def example2():
        import graphx
        import matplotlib.pyplot as plt
        stages = ['A','B','C','D','E','F']
        start = 'A1'
        target = 'F1'
        policies = [np.array([[1,5]]), np.array([[11,10,6,-1], [-1,8,3,7]]), np.array([[5,8,-1], [4,5,-1], [-1,3,4],[-1,-1,4]]), np.array([[4,5],[7,2],[-1,3]]),np.array([[4],[5]])]
        return ShortestPathProblem.fromStages(start, target, stages, policies, imname='dp172.eps')


class ShortestPathSolution(ORSolution):
    @classmethod
    def fromProblem(cls, problem):
        G = problem['graph']
        template = '最短路：{{path}}, 最短路长: {{length}}'
        length, path = G.get_bellman_path()
        parameter = {'path':'-'.join(path), 'length':length}
        return ShortestPathSolution(template, parameter)


class ORExamPaper(exam.ExamPaper):
    def __init__(self, *args, **kwargs):
        super(ORExamPaper, self).__init__(subject='运筹与优化', *args, **kwargs)
        self.selector = exam.Selector()
        
        self.fill = self.selector.random('or_fill', 5)
        self.get_truefalse = self.selector.random('or_truefalse', 5)

        problems = []
        p = DualSimplexProblem.example()
        p.solution = DualSimplexSolution
        problems.append(p)
        
        p = TransportationIBFSProblem.random()
        p.solution = TransportationIBFSSolution
        problems.append(p)

        p = GoalProgrammingProblem.example()
        p.solution = GoalProgrammingSolution
        problems.append(p)

        p = Search1DProblem.example3()
        p.solution = Search1DSolution
        problems.append(p)

        p = ShortestPathProblem.example2()
        p.solution = ShortestPathSolution
        problems.append(p)
        
        p = MatrixGameProblem.example3()
        p.solution = MatrixGameSolution
        problems.append(p)

        p = DecisionProblem.example3()
        p.solution = DecisionSolution
        problems.append(p)
        self.calculation = problems



if __name__=='__main__':

    # import os.path
    # paper.write(os.path.join(exam.PAPER_FOLDER, '17-18运筹与优化试卷I'))
    # print(paper.dumps())

    p = MatrixGameProblem.example1()
    p.solution = MatrixGameSolution
    print(p.totex())

