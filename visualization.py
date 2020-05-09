import networkx as nx
import plotly
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pandas as pd


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "ECHR Clustering Example"


#temp graph
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

web_dict = {0: 'https://www.google.dk', 1: 'https://www.twitter.com', 2: 'https://www.reddit.com', 3: 'https://www.imgur.com', 4: 'https://www.twitch.tv'}
name_dict = {0: 'Google', 1: 'Twitter', 2: 'Reddit', 3: 'Imgur', 4: 'Twitch'}




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
    hovertext = name_dict[node]
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
                                ax=(G.nodes[edge[0]]['pos'][0] + G.nodes[edge[1]]['pos'][0]) / 2,
                                ay=(G.nodes[edge[0]]['pos'][1] + G.nodes[edge[1]]['pos'][1]) / 2, axref='x', ayref='y',
                                x=(G.nodes[edge[1]]['pos'][0] * 3 + G.nodes[edge[0]]['pos'][0]) / 4,
                                y=(G.nodes[edge[1]]['pos'][1] * 3 + G.nodes[edge[0]]['pos'][1]) / 4, xref='x', yref='y',
                                showarrow=True,
                                arrowhead=3,
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
                                **Selected article:**
                                """),
                    html.Pre(id='click-data', children=['No article selected, click on a note to select and article'])
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






if __name__ == '__main__':
    app.run_server(debug=True)



'''

document.getElementById('myDiv').on('plotly_click', function(data){
    var pts = '';
    for(var i=0; i < data.points.length; i++){//find nearest point
        pts = 'x = '+data.points[i].x +'\ny = '+
            data.points[i].y.toPrecision(4) + '\n\n';
    }
    window.open("https://www.google.de/"+pts,'_blank');//open a url in a new tab
});

'''