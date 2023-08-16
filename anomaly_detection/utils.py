#Helper functions for my views.py funcitons

#Libraries to help with my computations.
import scipy.stats as stats


def apply_normality_test(current_feature):
    stat, p = stats.normaltest(current_feature)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
      return True
    else:
      return False

def normality_test_values(current_feature):
    pass
    