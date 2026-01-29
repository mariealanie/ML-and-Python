import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd


def train_svm_and_predict(train_features, train_target, test_features):
    """
    train_features: np.array, (num_elements_train x num_features) - train data description, the same features and the same order as in train data
    train_target: np.array, (num_elements_train) - train data target
    test_features: np.array, (num_elements_test x num_features) -- some test data, features are in the same order as train features

    return: np.array, (num_elements_test) - test data predicted target, 1d array
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            C=7,
            kernel='poly',
            degree=2,
            gamma='scale',
            coef0=0.7,
            probability=True,
            random_state=42,
            verbose=False
        ))
    ])

    pipeline.fit(train_features, train_target)

    test_predictions = pipeline.predict(test_features)

    return test_predictions


def svm_cross_validation(X_train, y_train, cv_folds=5, n_jobs=-1):

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(random_state=42))
    ])

    param_grid = [
        {
            'svm__C': [0.5, 1, 5, 10, 20, 50],
            'svm__kernel': ['rbf'],
            'svm__gamma': [0.005, 0.01, 0.05, 0.1, 0.5, 1, 'scale', 'auto']
        },
        {
            'svm__C': [5, 7, 10, 15, 20],
            'svm__kernel': ['poly'],
            'svm__degree': [2, 3],
            'svm__gamma': [0.01, 0.05, 0.1, 0.5, 'scale', 'auto'],
            'svm__coef0': [0.0, 0.3, 0.5, 0.7, 1.0]
        },
    ]

    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print("\n" + "="*50)
    print("ЛУЧШАЯ МОДЕЛЬ:")
    print("="*50)
    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Лучшая точность на кросс-валидации: {grid_search.best_score_:.5f}")

    cv_results = pd.DataFrame(grid_search.cv_results_)

    cv_results = cv_results.sort_values('mean_test_score', ascending=False)

    print(f"\nТоп-5 моделей:")
    print("-"*50)
    for i in range(min(5, len(cv_results))):
        params = cv_results.iloc[i]['params']
        score = cv_results.iloc[i]['mean_test_score']
        print(f"{i+1}. Точность: {score:.4f}, Параметры: {params}")


if __name__ == '__main__':
    X_train = np.load('public_tests/00_test_data_input/train/cX_train.npy')
    y_train = np.load('public_tests/00_test_data_input/train/cy_train.npy')
    X_test = np.load('public_tests/00_test_data_input/test/cX_test.npy')

    svm_cross_validation(X_train, y_train)
