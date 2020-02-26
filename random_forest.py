
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main(X_train, y_train, X_test):
    #Create a Gaussian Classifier
    model=RandomForestClassifier(n_estimators=100) #currently using default hyperparmeters
    #grid search in future to figure out what parameters to use here
    #
    #Train the model
    model.fit(X_train,y_train)
    #
    #Predict for test set
    #y_pred=clf.predict(X_test)
    y_prob=model.predict_proba(X_test)
    #
    #Features by importance
    feature_imp = pd.Series(model.feature_importances_,index=list(X_train.columns)).sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=feature_imp, y=feature_imp.index)
    ax.set_xlabel('Feature Importance Score')
    ax.set_ylabel('Features')
    ax.set_title('Features by Importance')
    fig.subplots_adjust(left = 0.25)
    fig.savefig("feature_importance.png")
    return([model, y_prob])
