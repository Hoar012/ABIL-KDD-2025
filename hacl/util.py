import vision_evaluation.evaluators as v_eval

def accuracy(y_label, y_pred):
    """ Compute Top1 accuracy
    Args:
        y_label: the ground truth labels. Shape (N,)
        y_pred: the prediction of a model. Shape (N,)
    """
    evaluator = v_eval.TopKAccuracyEvaluator(1)
    evaluator.add_predictions(predictions=y_pred, targets=y_label)
    return evaluator.get_report()['accuracy_top1']

from hacl.pdsketch.interface.v2.state import State
from hacl.pdsketch.interface.v2.expr import ValueOutputExpression
from hacl.pdsketch.interface.v2.domain import Domain

def goal_test(
    domain: Domain, state: State, goal_expr: ValueOutputExpression,
):
    score = domain.eval(goal_expr, state).item()
    return score