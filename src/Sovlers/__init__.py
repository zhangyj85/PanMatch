# from .solver_train_flow import TrainSolver as FlowTrainSolver
# from .solver_test_flow import SubmitSolver as FlowTestSolver
from .solver_demo_flow import DemoSolver
# from cross_task_eval.depth.solver import BenchmarkSolver as DepthSolver
# from cross_task_eval.match.solver import BenchmarkSolver as MatchTestSolver


def get_solver(config):

    # mode_cfg = config['mode'].lower()
    # task_cfg = config['task'].lower()

    # if mode_cfg == 'train':
    #     solver = FlowTrainSolver(config)
    # elif mode_cfg == 'test' and task_cfg in ['flow', 'stereo']:
    #     solver = FlowTestSolver(config)
    # elif mode_cfg == 'test' and task_cfg in ['match']:
    #     solver = MatchTestSolver(config)
    # elif mode_cfg == 'demo':
    #     solver = DemoSolver(config)
    # elif mode_cfg == 'test' and task_cfg in ['depth']:
    #     solver = DepthSolver(config)
    # else:
    #     raise NotImplementedError('Solver [{:s}] is not supported.'.format(mode_cfg))
    
    # return solver
    return DemoSolver(config)
