import numpy as np
import scipy.stats as st


def ReportAccuracies(avg_acc, avg_recall, avg_f1):
    ic_acc = st.t.interval(
        0.9, len(avg_acc) - 1, loc=np.mean(avg_acc), scale=st.sem(avg_acc)
    )
    ic_recall = st.t.interval(
        0.9, len(avg_recall) - 1, loc=np.mean(avg_recall), scale=st.sem(avg_recall)
    )
    ic_f1 = st.t.interval(
        0.9, len(avg_f1) - 1, loc=np.mean(avg_f1), scale=st.sem(avg_f1)
    )
    print(
        "Mean Accuracy[{:.4f}] IC [{:.4f}, {:.4f}]".format(
            np.mean(avg_acc), ic_acc[0], ic_acc[1]
        )
    )
    print(
        "Mean Recall[{:.4f}] IC [{:.4f}, {:.4f}]".format(
            np.mean(avg_recall), ic_recall[0], ic_recall[1]
        )
    )
    print(
        "Mean F1[{:.4f}] IC [{:.4f}, {:.4f}]".format(
            np.mean(avg_f1), ic_f1[0], ic_f1[1]
        )
    )
