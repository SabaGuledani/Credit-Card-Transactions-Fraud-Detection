from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, average_precision_score, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt


def get_probs(model, X_test):
    """
    Get the probabilities of the model for the test set
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)

    raise AttributeError(
        "Model does not implement predict_proba or decision_function."
    )


def get_best_recall(precision, recall, target_precision=0.9):
    """
    Get the best recall for a given precision and recall
    """
    # find indices where precision >= target
    valid_idx = np.where(precision >= target_precision)[0]

    if len(valid_idx) > 0:
        best_recall = recall[valid_idx].max()
    else:
        best_recall = 0.0
    return best_recall

def get_roc_auc(y_test, y_pred_prob):
    """
    Get the ROC AUC score for a given test set and predicted probabilities
    """
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    return roc_auc

def get_average_precision(y_test, y_pred_prob):
    """
    Get the average precision score for a given test set and predicted probabilities
    """
    pr_auc = average_precision_score(y_test, y_pred_prob)
    return pr_auc

def get_precision_recall_curve(y_test, y_pred_prob):
    """
    Get the precision recall curve for a given test set and predicted probabilities
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    return precision, recall, thresholds

def draw_precision_recall_curve(y_test, y_pred_prob):
    precision, recall, thresholds = get_precision_recall_curve(y_test, y_pred_prob)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()

def draw_confusion_matrix(y_test,y_pred_prob, threshold=0.9):
    y_pred = (y_pred_prob >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()

def print_scores(y_test, y_pred_prob, target_precision=0.9):
    roc_auc = get_roc_auc(y_test, y_pred_prob)
    pr_auc = get_average_precision(y_test, y_pred_prob)
    precision, recall, thresholds = get_precision_recall_curve(y_test, y_pred_prob)
    best_recall = get_best_recall(precision, recall, target_precision=target_precision)
    print(f"roc_auc: {roc_auc}, precision recall score: {pr_auc}, best recall at {target_precision} precision: {best_recall}")
    return roc_auc, pr_auc, best_recall


