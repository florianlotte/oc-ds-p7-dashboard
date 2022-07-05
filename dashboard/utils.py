import os

import requests

api_host = os.getenv("URL_BACKEND", "http://127.0.0.1:8889")


def get_credit_stats():
    try:
        response = requests.get(f'{api_host}/credit')
        response = response.json()
        return response
    except:
        return {}


def get_credit_feature(feature, limit=100):
    try:
        response = requests.get(f'{api_host}/credit/', data={'feature': feature, 'limit': limit})
        response = response.json()
        return response.get(feature, {})
    except:
        return {}


def get_graph_1():
    try:
        response = requests.get(f'{api_host}/credit/graph/1')
        response = response.json()
        return response
    except:
        return {}


def get_graph_2(credit_id):
    try:
        query = {'credit_id': credit_id}
        response = requests.get(f'{api_host}/credit/graph/2', params=query)
        response = response.json()
        return response
    except:
        return {}


def get_id_list():
    query = {'features': 'SK_ID_CURR', 'limit': 0}
    try:
        response = requests.get(f'{api_host}/feature', params=query)
        response = response.json()
        return response
    except:
        return {}


def get_credit_by_id(credit_id):
    try:
        response = requests.get(f'{api_host}/credit/id/{credit_id}')
        response = response.json()
        return response
    except:
        return {}


def get_features(features="", limit=1000):
    query = {'features': features, 'limit': limit}
    try:
        response = requests.get(f'{api_host}/feature', params=query)
        response = response.json()
        return response
    except:
        return {}
