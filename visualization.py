import networkx as nx
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import sys
import numpy as np


def create_adj_matrix(documents, cosine_similarities):
    m = len(documents)
    matrix = np.full((m, m), 0.0)
    for x in range(m):
        if x % 1000 == 0:
            print('Progress: ' + str(x) + ' / ' + str(m))
        for y in range(m):
            if cosine_similarities[x][y] > 0.0:  # Change 0.0 if there needs to be a cutoff based on the similarity
                matrix[x][y] = cosine_similarities[x][y]

    return matrix


def create_name_and_web_dicts(k_clusters):
    name_dict = {}
    web_dict = {}
    for i in range(len(k_clusters)):
        cluster_name = {}
        cluster_web = {}
        for j in range(len(k_clusters[i])):
            cluster_name[j] = k_clusters[i][j].title
            cluster_web[j] = 'http://hudoc.echr.coe.int/eng?i=' + k_clusters[i][j].document_id
        name_dict[i] = cluster_name
        web_dict[i] = cluster_web
    return name_dict, web_dict


def create_clusters(documents, clusters, k):
    masks = []
    for i in range(k):
        mask = clusters == i
        masks.append(mask)

    k_clusters = []
    for mask in masks:
        cluster = documents[mask]
        k_clusters.append(cluster)
    return k_clusters


def doc_to_vec_visualize(documents=None, adj_matrix=None, labels=None):


    documents = np.array(documents)
    clusters = create_clusters(clusters=labels, documents=documents, k=adj_matrix.shape[0])

    name_dict, web_dict = create_name_and_web_dicts(clusters)
    sorted_clusters = []
    for clust in clusters:
        sorted_clusters.append(sorted(clust, key=lambda x: x.pagerank, reverse=True))

    gui_graph_run_doc_to_vec(matrix=adj_matrix, name_dict=name_dict, web_dict=web_dict, docs=sorted_clusters)


def gui_graph_run_doc_to_vec(matrix=None, name_dict=None, web_dict=None, docs=None):
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    app.title = "ECHR Clustering Example"

    if matrix is not None:
        G = nx.from_numpy_array(A=matrix, create_using=nx.Graph)
    else:
        sys.exit('matrix is None, stopping the program')
    if name_dict is None:
        sys.exit('name_dict is None, stopping the program')
    if web_dict is None:
        sys.exit('web_dict is None, stopping the program')
    if docs is None:
        sys.exit('docs is None, stopping the program')

    pos = nx.layout.spring_layout(G)
    for node in G.nodes:
        G.nodes[node]['pos'] = list(pos[node])

    sizes = []
    for doc in docs:
        sizes.append(np.log(len(doc))*10)

    traceRecode = []
    node_trace = go.Scatter(x=[],
                            y=[],
                            hovertext=[],
                            text=[],
                            mode='markers+text',
                            textposition="middle center",
                            textfont=dict(family="sans serif", size=25, color="Black"),
                            hoverinfo="text",
                            marker={'symbol': 'circle', 'size': sizes, 'color': 'steelblue'})

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        hovertext = 'Cluster ' + str(node) #name_dict[node] #str(node)
        text = str(node)
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['hovertext'] += tuple([hovertext])
        node_trace['text'] += tuple([text])

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
                                    ax= G.nodes[edge[0]]['pos'][0],
                                    ay= G.nodes[edge[0]]['pos'][1], axref='x', ayref='y',
                                    x= G.nodes[edge[1]]['pos'][0],
                                    y = G.nodes[edge[1]]['pos'][1], xref='x', yref='y',
                                    showarrow=True,
                                    arrowhead=0,
                                    arrowsize=4,
                                    arrowwidth=1+(matrix[edge[0]][edge[1]]*10),
                                    opacity=1
                                ) for edge in G.edges
                            ]
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
                        html.H2('Selected cluster:'),
                        html.Pre(id='click-data', children=['No cluster selected, click on a cluster to select a document'])
                    ],
                    style={'height': '400px'})
            ]
        )
    ])

    @app.callback(
        dash.dependencies.Output('click-data', 'children'),
        [dash.dependencies.Input('my-graph', 'clickData')])
    def display_click_data(clickData):
        doc_conclusion_header = html.H3('Highest pageranked document\n')
        doc_conclusion_link = html.A(href=web_dict[clickData['points'][0]['pointIndex']][0], target='_blank', title='Click to go to article', children=[name_dict[clickData['points'][0]['pointIndex']][0],])
        doc_conclusion_articles = html.H6('\nArticles\n')
        doc_conclusion = html.H6('\nConclusion\n')

        documents_in_cluster = html.H3('\nOther documents in cluster(sorted by page rank): \n')
        documents_with_links = []

        for i in range(len(web_dict[clickData['points'][0]['pointIndex']])):
            documents_with_links.append(html.Div(children=[html.A(href=web_dict[clickData['points'][0]['pointIndex']][i], target='_blank', title='Click to go to article', children=['{:_<60}'.format(name_dict[clickData['points'][0]['pointIndex']][i]),]),
                                                 '\tArticles: ',
                                                 docs[clickData['points'][0]['pointIndex']][i].articles]))
            documents_with_links.append('\n')

        return html.Div(children=[doc_conclusion_header,
                                  doc_conclusion_link,
                                  doc_conclusion_articles,
                                  docs[clickData['points'][0]['pointIndex']][0].articles,
                                  doc_conclusion,
                                  docs[clickData['points'][0]['pointIndex']][0].conclusion.replace(';', '\n'),
                                  '\n',
                                  documents_in_cluster,
                                  html.Div(children=documents_with_links)])

    app.run_server(debug=False)