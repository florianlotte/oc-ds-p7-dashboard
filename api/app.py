import logging
import os
from collections import Counter

import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.logger import logger

from model.app import ModelApp

logging.basicConfig(level=logging.DEBUG)

# Init app
app = FastAPI(title="Home Credit API", version="0.0.1")

path = './data'
# Load data
logger.info("Load Data...")
filename = 'Home_Credit_Default_Risk.csv'
data = pd.read_csv(os.path.join(path, filename))
data.set_index('SK_ID_CURR', inplace=True)
logger.info(f"Data: {data.shape}")
# Load data description
logger.info("Load description Data...")
filename = 'HomeCredit_columns_description.csv'
data_desc = pd.read_csv(os.path.join(path, filename))
data_desc = data_desc[['Row', 'Description', 'Special']]
data_desc.set_index('Row', inplace=True)
logger.info(f"Data descriptions: {data_desc.shape}")


# Make model
model_app = ModelApp(data)
logger.info("Fit lightgbm model...")
model_app.fit_model()
logger.info("Refine threshold...")
model_app.refine_threshold()
logger.info(f"Best(mean) threshold: {model_app.threshold}")
logger.info("Make data for graph...")
model_app.make_data_for_graph1()
logger.info("Make explanation...")
model_app.make_explanation()


@app.get("/", summary="get app info", description="get app info")
def get_info():
    return {
        'name': app.title,
        'version': app.version
    }


@app.get("/credit", tags=['credit'], summary="get credits stats", description="get credits stats")
def get_credits_stats():
    target_counter = Counter(data.loc[data['TARGET'].notnull(), 'TARGET'].astype(int))

    return {
        'total': data.shape[0],
        'features_agg': data.aggregate(['mean', 'std']).to_dict(),
        'filled_rate': round(data.notnull().sum(axis=0).sum() / len(data) / len(data.columns) * 100),
        'target_counter': target_counter,
        'threshold': model_app.threshold,
        'feature_importances': model_app.feature_importances
    }


@app.get("/credit/graph/1", tags=['credit'], summary="get graph 1 data", description="get graph 1 data")
def get_graph_1():
    return {'data': model_app.data_graph1.to_dict(orient='list')}


@app.get("/credit/graph/2", tags=['credit'], summary="get graph 2 data", description="get graph 2 data")
def get_graph_2(credit_id: int):
    res = data.loc[credit_id]
    pred_proba = model_app.model.predict_proba(res[model_app.feats].to_frame().transpose())
    pred = 1 if pred_proba[0][1] > model_app.threshold else 0
    explainer = model_app.shap_explainer
    shap_values = explainer.shap_values(model_app.data.loc[[credit_id], model_app.feats])
    credit = pd.Series(shap_values[pred][0], index=model_app.feats)
    return {'data': model_app.shap_data_desc[pred].to_dict(),
            'credit': credit.to_dict()}


@app.get("/credit/id/{credit_id}", tags=['credit'], summary="get credit by id", description="get credit by id")
def get_credit_by_id(credit_id: int):
    try:
        res = data.loc[credit_id].copy()
        pred = model_app.model.predict_proba(res[model_app.feats].to_frame().transpose())
        if pred[0][1] > model_app.threshold:
            score = (1 - pred[0][1]) * 0.5 / (1 - model_app.threshold)
        else:
            score = 1 - (pred[0][1] * 0.5 / model_app.threshold)
        res["PROBA"] = pred[0][1]
        res["SCORE"] = round(score * 100)
        res["SK_ID_CURR"] = res.name
        res.dropna(inplace=True)
        return res.to_dict()
    except:
        return {}


@app.get("/feature", tags=['feature'], summary="get features", description="get features")
def get_features(features: str, limit: float = 1000):
    feature_safe = [f.strip() for f in features.split(',') if f.strip() in data.columns]
    if feature_safe:
        if int(limit) > 1:
            sample = data.sample(n=int(limit))
        elif 0 < limit < 1:
            sample = data.sample(frac=limit)
        else:
            sample = data

        feature_sample = sample[feature_safe]
        proba = model_app.model.predict_proba(sample[model_app.feats])
        feature_sample['PROBA'] = proba[:, 1]
        feature_sample['SCORE'] = (1 - (feature_sample['PROBA'] * 0.5 / model_app.threshold)) * 100
        response = {'features': feature_sample.replace({np.nan:None}).to_dict()}
        return response
    return {}
