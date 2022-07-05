from flask import Flask
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, dcc, html, State

from .utils import get_credit_stats, get_graph_1, get_credit_by_id, get_graph_2, get_features

app_name = "Home Credit Dashboard"
server = Flask(app_name)
app = dash.Dash(
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
logger = app.server.logger

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("Credit", href="/credit")),
    ],
    brand="Home Credit Dashboard",
    brand_href="/",
    color="dark",
    dark=True,
    fluid=True,
)

content = html.Div(id="page-content")

app.layout = html.Div([dcc.Location(id="url"), navbar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    stats = get_credit_stats()
    filled = stats.get('filled_rate', 0)
    if 'target_counter' in stats:
        target = stats['target_counter']
        target_rate = round((target.get('0', 0) / sum(list(target.values()))) * 100)
    else:
        target_rate = 0
    nb = stats.get('total', 0)
    mean_amount = round(stats.get('features_agg', {}).get('AMT_CREDIT', {}).get('mean', 0))
    mean_duration = round(stats.get('features_agg', {}).get('AMT_ANNUITY', {}).get('mean', 0))
    if pathname == "/":
        return dbc.Container(
            children=[
                html.H1("Home"),
                html.Hr(),
                html.Div(
                    dbc.Row(
                        [
                            dbc.Col(get_progress_card(
                                tag="DATA",
                                title="Taux de renseignements",
                                desciption="Pourcentage de remplissage des données",
                                value=filled,
                                color="success"),
                            ),
                            dbc.Col(get_progress_card(
                                tag="TARGET",
                                title="Taux de succès",
                                desciption="Pourcentage sans défaut de paiement",
                                value=target_rate,
                                color="success")
                            ),
                            dbc.Col(get_value_card(
                                tag="CREDIT",
                                title="Nombre total de crédits",
                                value=f'{nb:,}'.replace(',', ' '),
                                color="primary")
                            ),
                            dbc.Col(get_value_card(
                                tag="CREDIT",
                                title="Montant moyen des crédits",
                                value=f'{mean_amount:,}'.replace(',', ' '),
                                color="primary")
                            ),
                            dbc.Col(get_value_card(
                                tag="CREDIT",
                                title="Durée moyenne des crédits",
                                value=f'{mean_duration:,}'.replace(',', ' '),
                                color="primary")
                            ),
                        ],
                    )
                ),
                html.Hr(),
                html.Div(dbc.Row(get_credit_scatter()))
            ],
            fluid=True,
            className="p-4"
        )
    elif pathname == "/credit":
        return dbc.Container([
            dcc.Store(id='memory'),
            dbc.Row([
                dbc.Col([html.H1("Credit")]),
                dbc.Col([html.Div(id="credit-info")])
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dbc.Input(id="input", placeholder="Entrer un ID de crédit...", type="number")
                        ])
                    ]),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col(id='score-card'),
                        dbc.Col(id='note-card')
                    ]),
                    dbc.Row([dbc.Col(
                        get_card_graph(tag="SCATTER", title="Analyse de 2 paramètres",
                                       body=get_scatter_selector(), color="primary"
                                       ), id='scatter-card')]),
                ], xl=6),
                dbc.Col([
                    dbc.Row(dbc.Col(id='other-card')),
                    dbc.Row(dbc.Col(id='feature-card'))
                ], xl=6)
            ])
        ], fluid=True, className="p-4")
    return dbc.Container(
        children=[
            html.H1("404: Not found", class_name="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        fluid=True,
        className="p-4"
    )


def get_progress_card(tag="", title="", desciption="", value=0, color=None, outline=True):
    card = dbc.Card(
        [
            dbc.CardHeader(tag),
            dbc.CardBody(
                [
                    html.H5(title, className="card-title"),
                    html.P(desciption, className="card-text"),

                ]
            ),
            dbc.Progress(label=f"{value}%", color=color, value=value, className="m-2"),
        ],
        color=color,
        outline=outline,
        className="mb-4",
    )
    return card


def get_value_card(tag="", title="", desciption="", value="0", color=None, outline=True):
    if desciption:
        p_description = [html.P(desciption, className="card-text")]
    else:
        p_description = []

    card = dbc.Card(
        [
            dbc.CardHeader(tag),
            *p_description,
            dbc.CardBody(
                [
                    html.H5(title, className="card-title"),
                    html.H1(value, className="display-6"),
                ]
            ),
        ],
        color=color,
        outline=outline,
        className="mb-4",
    )
    return card


def get_credit_scatter():
    res = get_graph_1()
    if 'data' not in res:
        return []
    data = pd.DataFrame.from_dict(get_graph_1()['data'])
    data.rename(columns={'CENTROID_X': 'AMT_ANNUITY', 'CENTROID_Y': 'AMT_CREDIT'}, inplace=True)
    fig = px.scatter(data, x='AMT_ANNUITY', y='AMT_CREDIT', color='SQRT2_COUNT',
                     hover_data=['AMT_ANNUITY', 'AMT_CREDIT', 'COUNT'], color_continuous_scale='viridis')
    fig.update_layout(margin={'l': 0, 'r': 10, 't': 10, 'b': 0})
    return [
        html.H5("Repartition du montant du crédit en fonction de la durée"),
        dbc.Col(html.Div([dcc.Graph(id="credit-scatter", figure=fig)], id='div-sell-dot'), width=4)
    ]


@app.callback([
    Output("credit-info", "children"),
    Output("scatter-card", "children"),
    Output("feature-card", "children"),
    Output("note-card", "children"),
    Output("score-card", "children"),
    Output("other-card", "children"),
    Output('memory', 'credit')],
    [Input("input", "value")]
)
def output_text(credit_id):
    if not credit_id:
        return (
            dbc.Alert("Je n'ai pas trouvé de crédit correspondant à cette ID", color="primary", className="m-0"),
            get_card_graph(tag="SCATTER", title="Analyse de 2 paramètres", color="primary"),
            get_card_graph(tag="DETAILS", title="Classement des paramètres par importance", color="primary"),
            get_badge_num(tag="NOTE", title="Note du client (/100)", value="--", color="secondary"),
            get_badge_text(tag="TARGET", title="Incident de paiement", value="--", color="secondary"),
            get_card_graph(tag="DETAILS", title="Explication des valeurs", color="primary"),
            {}
        )

    credit_data = get_credit_by_id(credit_id)
    if 'SCORE' in credit_data:
        score_value = credit_data['SCORE']
        score_str_value = str(round(credit_data['SCORE']))
        score_color = 'success' if score_value >= 50 else 'danger'
    else:
        score_value = None
        score_str_value, score_color = ('PAS DE DONNEES', 'warning')

    if 'TARGET' in credit_data:
        target_value, target_color = ('OK', 'success') if credit_data['TARGET'] == 0 else ('INCIDENT', 'danger')
    elif score_value:
        target_value, target_color = ('PREDICT OK', 'success') if score_value >= 50 else (
            'PREDICT INCIDENT', 'danger')
    else:
        target_value, target_color = ('PAS DE DONNEES', 'warning')

    return (
        [],
        get_card_graph(tag="SCATTER", title="Analyse de 2 paramètres", body=get_scatter_selector(credit_data),
                       color="primary"),
        get_card_graph(tag="DETAILS", title="Classement des paramètres par importance",
                       body=get_feature_importances(credit_data), color="primary"),
        get_badge_num(tag="NOTE", title="Note du client (/100)", value=score_str_value, color=score_color),
        get_badge_text(tag="TARGET", title="Incident de paiement", value=target_value, color=target_color),
        get_card_graph(tag="DETAILS", title="Explication des valeurs", body=get_explainer_graph(credit_data),
                       color="primary"),
        credit_data
    )


def get_badge_num(tag="", title="", desciption="", value="0", color=None, outline=True):
    if desciption:
        p_description = [html.P(desciption, className="card-text")]
    else:
        p_description = []

    card = dbc.Card(
        [
            dbc.CardHeader(tag),
            *p_description,
            dbc.CardBody(
                [
                    html.H5(title, className="card-title"),
                    html.H1(dbc.Badge(value, color=color, className="me-1")),
                ]
            ),
        ],
        color=color,
        outline=outline,
        className="mb-4",
    )

    return card


def get_badge_text(tag="", title="", desciption="", value="NO VALUE", color="primary", outline=True):
    if desciption:
        p_description = [html.P(desciption, className="card-text")]
    else:
        p_description = []

    card = dbc.Card(
        [
            dbc.CardHeader(tag),
            *p_description,
            dbc.CardBody(
                [
                    html.H5(title, className="card-title"),
                    html.H1(dbc.Badge(value, color=color, className="me-1")),
                ]
            ),
        ],
        color=color,
        outline=outline,
        className="mb-4",
    )

    return card


def get_card_graph(tag="", title="", desciption="", body=(), color="primary", outline=True):
    if desciption:
        p_description = [html.P(desciption, className="card-text")]
    else:
        p_description = []

    card = dbc.Card(
        [
            dbc.CardHeader(tag),
            *p_description,
            dbc.CardBody([html.H5(title, className="card-title"), *body], id="card-body-details"),
        ],
        color=color,
        outline=outline,
        className="mb-4",
    )

    return card


def get_explainer_graph(credit):
    if 'SK_ID_CURR' not in credit:
        return []

    res = get_graph_2(int(credit['SK_ID_CURR']))
    if 'data' in res:
        data = pd.DataFrame.from_dict(res['data'])
        main_features = data.index[:10].tolist()
        main_features.reverse()

        if 'credit' in res:
            credit = res['credit']
        else:
            credit = {c: 0 for c in main_features}

        fig = go.Figure()
        for i, c in enumerate(main_features):
            marker_color = 'green' if credit[c] >= 0 else 'red'
            fig.add_trace(go.Box(
                y0=c,
                x=[[credit[c]]],
                pointpos=0,
                q1=data.loc[[c], '25%'],
                median=data.loc[[c], '50%'],
                q3=data.loc[[c], '75%'],
                lowerfence=data.loc[[c], 'min'],
                upperfence=data.loc[[c], 'max'],
                mean=data.loc[[c], 'mean'],
                marker={'color': marker_color},
                line={'color': f'hsl(224, 50, 40)'},
                name=c,
                hoverinfo='x',
                showlegend=False)
            )
            fig.update_layout(
                margin=dict(l=10, r=0, t=10, b=0),
            )
        return [html.Div([dcc.Graph(id="credit-details", figure=fig)])]
    else:
        return []


def get_feature_importances(credit):
    if not credit:
        return []
    feature_importances = pd.Series(get_credit_stats()['feature_importances']).to_frame(name='feature_importances')
    credit_data = pd.Series(credit).to_frame(name='credit')
    data = pd.concat([feature_importances, credit_data.loc[feature_importances.index]], axis=1)
    fig = px.bar(
        data.sort_values('feature_importances')[-20:],
        x='feature_importances',
        hover_data=['credit'],
        color_discrete_sequence=['hsl(224, 50, 40)'] * 20
    )
    fig.update_layout(margin=dict(l=10, r=0, t=10, b=0))
    fig.layout.showlegend = False
    return [html.Div([dcc.Graph(id="credit-details", figure=fig)])]


def get_scatter_selector(credit=None):
    if not credit:
        return [dcc.Dropdown([], multi=True, id="scatter-multi"), html.Div(id='scatter-graph')]

    stats = get_credit_stats()
    return [
        dcc.Dropdown(list(stats['feature_importances'].keys()), multi=True, id="scatter-multi"),
        html.Div(id='scatter-graph')
    ]


@app.callback([Output('scatter-graph', 'children')], Input("scatter-multi", "value"), State("memory", "credit"))
def get_scatter_graph(features, credit_data):
    if features and len(features) == 2:
        res = get_features(features=f"{features[0]},{features[1]}", limit=1000)
        if 'features' not in res:
            return [html.P("")]
        data = pd.DataFrame(res['features'])
        fig = px.scatter(data, x=features[0], y=features[1], color='SCORE',
                         hover_data=[*features, 'SCORE'], color_continuous_scale='viridis')

        # credit_data = get_credit_by_id(credit)
        if features[0] in credit_data and features[1] in credit_data:
            fig.add_trace(go.Scatter(x=[credit_data[features[0]]], y=[credit_data[features[1]]], mode='markers',
                                     showlegend=False, marker={'size': 10}))
        fig.update_layout(margin={'l': 0, 'r': 10, 't': 10, 'b': 0})
        return [dcc.Graph(figure=fig)]
    else:
        return [html.P("")]
