import os
import pickle

from lightgbm import LGBMClassifier, log_evaluation
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.base import clone as sklearn_clone
import numpy as np
import pandas as pd
import shap


class ModelApp:
    def __init__(self, data):
        self.data = data
        self.feats = [f for f in self.data.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
        self.threshold = 0.5
        self.n_folds = 5
        self.model = None
        self.data_graph1 = None
        self.feature_importances = None
        self.shap_explainer = None
        self.shap_data_desc = [None, None]

    def fit_model(self):
        train_data = self.data[self.data['TARGET'].notnull()]
        lgbm_filename = 'lgbm_model.sav'
        if os.path.exists(lgbm_filename):
            self.model = pickle.load(open(lgbm_filename, 'rb'))
        else:
            self.model = LGBMClassifier(
                objective='binary',
                n_jobs=2,
                verbose=0,
                force_row_wise=True,
                is_unbalance=True,
                num_leaves=16,
                max_depth=8,
                min_child_samples=1000,
                learning_rate=0.1,
                colsample_bytree=0.5,
            )
            self.model.fit(
                train_data[self.feats],
                train_data['TARGET'],
                eval_set=[(train_data[self.feats], train_data['TARGET'])],
                eval_metric='auc',
                callbacks=[log_evaluation(period=10, show_stdv=False)]
            )
            pickle.dump(self.model, open(lgbm_filename, 'wb'))
        self.feature_importances = pd.Series(self.model.feature_importances_, index=self.feats).sort_values(
            ascending=False).to_dict()

    def refine_threshold(self):
        threshold_filename = 'threshold.sav'
        if os.path.exists(threshold_filename):
            self.threshold = pickle.load(open(threshold_filename, 'rb'))
        else:
            model = sklearn_clone(self.model)
            train_data = self.data[self.data['TARGET'].notnull()]

            folds = StratifiedKFold(n_splits=self.n_folds, random_state=None, shuffle=False)
            best_thresh = []
            for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_data[self.feats], train_data['TARGET'])):
                train_x, train_y = train_data[self.feats].iloc[train_idx], train_data['TARGET'].iloc[train_idx]
                valid_x, valid_y = train_data[self.feats].iloc[valid_idx], train_data['TARGET'].iloc[valid_idx]

                model.fit(
                        train_x, train_y,
                        eval_set=[(train_x, train_y), (valid_x, valid_y)],
                        eval_metric='auc',
                        callbacks=[log_evaluation(period=10, show_stdv=False)]
                )

                yhat = model.predict_proba(valid_x)
                yhat = yhat[:, 1]
                fpr, tpr, thresholds = roc_curve(valid_y, yhat)
                J = tpr - fpr
                ix = np.argmax(J)
                best_thresh.append(thresholds[ix])
                print('Best Threshold=%f' % (thresholds[ix]))
                mean_thres = np.mean(best_thresh)
                print('Mean Threshold=%f' % (mean_thres))
                self.threshold = mean_thres
            pickle.dump(self.threshold, open(threshold_filename, 'wb'))

    def make_data_for_graph1(self):
        # Reduce data for graph 1
        columns = ['AMT_ANNUITY', 'AMT_CREDIT']
        n_clusters = 1000
        df = self.data[columns].copy()
        df = df.fillna(df.mean())
        kmeans_filename = 'kmean_model.sav'
        if os.path.exists(kmeans_filename):
            kmeans = pickle.load(open(kmeans_filename, 'rb'))
        else:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=3072)
            kmeans.fit(df.values)
            pickle.dump(kmeans, open(kmeans_filename, 'wb'))
        df['CLUSTER'] = kmeans.predict(df.values)
        self.data_graph1 = pd.DataFrame(kmeans.cluster_centers_, columns=['CENTROID_X', 'CENTROID_Y'], index=range(1000))
        self.data_graph1['COUNT'] = df.groupby(['CLUSTER'])['CLUSTER'].count()
        self.data_graph1['COUNT'].fillna(0, inplace=True)
        self.data_graph1['SQRT_COUNT'] = np.sqrt(self.data_graph1['COUNT'])
        self.data_graph1['SQRT2_COUNT'] = np.sqrt(self.data_graph1['SQRT_COUNT'])

    def make_explanation(self):
        shap_filename = 'shap_explainer.sav'
        shap_desc_filename = 'shap_desc.sav'
        if os.path.exists(shap_filename) and os.path.exists(shap_desc_filename):
            self.shap_explainer = pickle.load(open(shap_filename, 'rb'))
            self.shap_data_desc = pickle.load(open(shap_desc_filename, 'rb'))
        else:
            self.shap_explainer = shap.TreeExplainer(self.model)
            shap_values = self.shap_explainer.shap_values(self.data[self.feats])
            pickle.dump(self.shap_explainer, open(shap_filename, 'wb'))

            shap_data = [None, None]
            shap_data[0] = pd.DataFrame(shap_values[0], columns=self.feats, index=self.data.index)
            shap_data[1] = pd.DataFrame(shap_values[1], columns=self.feats, index=self.data.index)

            self.shap_data_desc[1] = self.shap_data[1].describe(percentiles=[0.25, 0.5, 0.75]).transpose()
            self.shap_data_desc[1]['dist'] = np.abs(self.shap_data_desc[1]['max'] - self.shap_data_desc[1]['min'])
            self.shap_data_desc[1]['feature_importance'] = self.model.feature_importances_
            self.shap_data_desc[1].sort_values('feature_importance', ascending=False, inplace=True)

            self.shap_data_desc[0] = self.shap_data[0].describe(percentiles=[0.25, 0.5, 0.75]).transpose()
            self.shap_data_desc[0]['dist'] = np.abs(self.shap_data_desc[0]['max'] - self.shap_data_desc[0]['min'])
            self.shap_data_desc[0]['feature_importance'] = self.model.feature_importances_
            self.shap_data_desc[0].sort_values('feature_importance', ascending=False, inplace=True)

            pickle.dump(self.shap_data_desc, open(shap_desc_filename, 'wb'))
