# RandomForest Classifier Hyperparameter Tuning 
## 프로젝트 개요 
* 이 프로젝트는 RandomForestClassifier를 활용하여 주어진 데이터셋에 대한 하이퍼파라미터 튜닝과 모델 평가를 수행합니다. 최적의 하이퍼파라미터를 찾기 위해 GridSearchCV를 사용하며, 최적 모델을 통해 예측 및 평가를 진행합니다.
## 주요 내용 
### 1. 환경 설정 
* Python 3.11.5
* 주요 라이브러리: pandas, numpy : 데이터 처리 sklearn : 머신러닝 모델 및 평가 도구 제공

### 2. 데이터셋 분리 * 주어진 데이터셋을 훈련용과 테스트용으로 나누어 모델 학습과 평가를 진행합니다. 
### 3. RandomForestClassifier 모델 및 하이퍼파라미터 튜닝 
* GridSearchCV를 사용하여 다음과 같은 하이퍼파라미터에 대한 탐색을 수행합니다:
```n_estimators: [50, 100, 200] max_depth: [None, 10, 20, 30] min_samples_split: [2, 5, 10] param_grid = { 'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10] } grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2) grid_search.fit(X_train, y_train) ```
### 4. 최적 하이퍼파라미터 및 모델 선택 
* GridSearchCV를 통해 최적의 하이퍼파라미터를 도출하고 이를 사용한 모델을 선택합니다.
```print("Best parameters found: ", grid_search.best_params_) Best Parameters: max_depth=20, min_samples_split=5, n_estimators=200 ```
### 5. 모델 예측 및 평가 * 최적 모델을 사용해 테스트 데이터를 예측하고 혼동 행렬 및 정확도, 정밀도, 재현율, F1-score 등을 평가합니다. 
```print(confusion_matrix(y_test, y_pred_best)) print(classification_report(y_test, y_pred_best)) ``` 
### 6. 교차 검증 * 5-Fold 교차 검증을 통해 모델의 성능을 평가합니다. 
```cv_scores = cross_val_score(best_rf_model, X, y, cv=5) print("Cross-validation scores: ", cv_scores) print("Mean cross-validation score: ", cv_scores.mean()) ``` 
* Mean Cross-Validation Score: 약 0.929
#### 실행 방법 * 필수 라이브러리 설치: 
bash ```pip install pandas numpy scikit-learn``` Jupyter Notebook 실행: 
bash ```jupyter notebook proect2.ipynb``` 모든 셀을 실행하여 하이퍼파라미터 튜닝 및 모델 평가 수행. 
### 결과 요약 
* 최적의 하이퍼파라미터: max_depth=20, min_samples_split=5, n_estimators=200
*  모델 평가 결과: * Accuracy: 93% * Precision, Recall, F1-Score 제공됨.
*   교차 검증 평균 성능: 약 92.9%
