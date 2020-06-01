import networkx as nx
import plotly
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pandas as pd


def gui_graph_run(matrix=None):
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    app.title = "ECHR Clustering Example"

    if matrix is not None:
        G = nx.from_numpy_array(A=matrix, create_using=nx.Graph)
    else:
        # TODO error handling istedet for det her
        print('No matrix found, creating default one')
        G = nx.Graph()
        G.add_node(0)
        G.add_node(1)
        G.add_node(2)
        G.add_node(3)
        G.add_node(4)
        G.add_edge(1, 2)
        G.add_edge(0, 2)
        G.add_edge(3, 4)
        G.add_edge(4, 2)
        G.add_edge(1, 4)
        G.add_edge(4, 1)

    # TODO gør dynamisk her under
    web_dict = {0: 'https://hudoc.echr.coe.int/eng#{%22tabview%22:[%22document%22],%22itemid%22:[%22001-202429%22]}',
                1: 'https://hudoc.echr.coe.int/eng#{%22tabview%22:[%22document%22],%22itemid%22:[%22001-202433%22]}',
                2: 'https://hudoc.echr.coe.int/eng#{%22tabview%22:[%22document%22],%22itemid%22:[%22001-202212%22]}',
                3: 'https://hudoc.echr.coe.int/eng#{%22tabview%22:[%22document%22],%22itemid%22:[%22001-202392%22]}',
                4: 'https://hudoc.echr.coe.int/eng#{%22tabview%22:[%22document%22],%22itemid%22:[%22001-202428%22]}'}
    name_dict = {0: 'CASE OF KOROSTELEV v. RUSSIA',
                 1: 'CASE OF SUDITA KEITA v. HUNGARY',
                 2: 'CASE OF ANAHIT MKRTCHYAN v. ARMENIA',
                 3: 'CASE OF VARDOSANIDZE v. GEORGIA',
                 4: 'CASE OF CANLI v. TURKEY'}

    pos = nx.layout.shell_layout(G)
    for node in G.nodes:
        G.nodes[node]['pos'] = list(pos[node])

    traceRecode = []
    node_trace = go.Scatter(x=[],
                            y=[],
                            hovertext=[],
                            text=[],
                            mode='markers+text',
                            textposition="middle center",
                            textfont=dict(family="sans serif", size=18, color="Black"),
                            hoverinfo="text",
                            marker={'symbol': 'circle', 'size': 50, 'color': 'steelblue'})

    index = 0
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        hovertext = name_dict[node] #str(node)
        text = str(node)
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['hovertext'] += tuple([hovertext])
        node_trace['text'] += tuple([text])
        index = index + 1

    traceRecode.append(node_trace)

    w_height = 600
    w_width = w_height * 21/9

    my_figure = {
        "data": traceRecode,
        "layout": go.Layout(title='ECHR Clustering', showlegend=False, hovermode='closest',
                            margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                            xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            height= w_height,
                            width= w_width,
                            clickmode='event+select',
                            annotations=[
                                dict(
                                    #ax=(G.nodes[edge[0]]['pos'][0] + G.nodes[edge[1]]['pos'][0]) / 2,
                                    #ay=(G.nodes[edge[0]]['pos'][1] + G.nodes[edge[1]]['pos'][1]) / 2, axref='x', ayref='y',
                                    #x=(G.nodes[edge[1]]['pos'][0] * 3 + G.nodes[edge[0]]['pos'][0]) / 4,
                                    #y=(G.nodes[edge[1]]['pos'][1] * 3 + G.nodes[edge[0]]['pos'][1]) / 4, xref='x', yref='y',
                                    ax= G.nodes[edge[0]]['pos'][0],
                                    ay= G.nodes[edge[0]]['pos'][1], axref='x', ayref='y',
                                    x= G.nodes[edge[1]]['pos'][0],
                                    y = G.nodes[edge[1]]['pos'][1], xref='x', yref='y',
                                    showarrow=True,
                                    arrowhead=0,
                                    arrowsize=4,
                                    arrowwidth=1,
                                    opacity=1
                                ) for edge in G.edges]
                            )}

    app.layout = html.Div([
        html.Div(className="main_graph",
                 children=[dcc.Graph(id="my-graph", figure=my_figure),]),
        html.Div(
            className="two columns",
            children=[
                html.Div(
                    className='twelve columns',
                    children=[
                        dcc.Markdown("""
                                    **Selected document:**
                                    """),
                        html.Pre(id='click-data', children=['No document selected, click on a note to select a document'])
                    ],
                    style={'height': '400px'})
            ]
        )
    ])

    @app.callback(
        dash.dependencies.Output('click-data', 'children'),
        [dash.dependencies.Input('my-graph', 'clickData')])
    def display_click_data(clickData):
        return html.Div(html.A(href=web_dict[clickData['points'][0]['pointIndex']], target='_blank', title='Click to go to article', children=[name_dict[clickData['points'][0]['pointIndex']],]), 'test')

    app.run_server(debug=True)
