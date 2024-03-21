from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import GridSearchCV

def calc_metrics(y, y_pred):
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    return accuracy, precision, recall, f1

def get_scores(model, X_train,y_train, X_test, y_test):
    # Make predictions on the training set
    sns.set(style="darkgrid")

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))  # 1 row, 2 columns

    y_pred = model.predict(X_train)

    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)

    print("- Train Accuracy:", accuracy)
    print("- Train Precision:", precision)
    print("- Train Recall:", recall)
    print("- Train F1 Score:", f1)
    
    #print(classification_report(y, y_pred))

    # Calculate confusion matrix
    cm1 = confusion_matrix(y_train, y_pred)

    # Plot the confusion matrix
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax = axes[0])
    axes[0].set_title("Confusion Matrix - Train")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n- Test Accuracy:", accuracy)
    print("- Test Precision:", precision)
    print("- Test Recall:", recall)
    print("- Test F1 Score:", f1)
    
    #print(classification_report(y, y_pred))

    # Calculate confusion matrix
    cm2 = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues", ax = axes[1])
    axes[1].set_title("Confusion Matrix - Test")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate the false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # Calculate the AUC score
    roc_auc = auc(fpr, tpr)


    axes[2].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[2].set_xlim([0.0, 1.0])
    axes[2].set_ylim([0.0, 1.05])
    axes[2].set_xlabel('False Positive Rate')
    axes[2].set_ylabel('True Positive Rate')
    axes[2].set_title('Receiver Operating Characteristic (ROC) Curve')
    axes[2].legend(loc="lower right")
    
    plt.show()
    
    return [accuracy, precision, recall, f1, roc_auc]
    
def get_roc_auc(y_test, y_pred_proba):

    # Calculate the false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # Calculate the AUC score
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    return roc_auc