import os
import pandas as pd

from graph_tool.all import *

def print_hi(name):

    data_dir="/Users/tdunn/Data/SoS"

    data_in=os.path.join(data_dir,'evolutionary_biology_top_1000_with_network_data.parquet')
    col_headers=['id','doi','title','publication_date','publication_year','cited_by_count',
                 'authors_count','countries_distinct_count','institutions_distinct_count','fwci',
                 'citation_normalized_percentile','is_retracted','has_abstract','has_fulltext',
                 'language','type','open_access','field','bibliography','citations_by_year_post_pub']

    df = pd.read_parquet(data_in)
    #column_names = df.columns.tolist()
    #print(column_names)
    #df2 = pd.read_parquet('data.parquet', columns=['user_id', 'transaction_date'])

    head=df.head()
    data_head = os.path.join(data_dir, 'head.txt')
    with open(data_head, "w", encoding="utf-8") as file:
        file.write(head.to_string())
    # Initialize a new directed graph
    g = Graph()

    # Add vertices (returns vertex descriptors)
    v1 = g.add_vertex()
    v2 = g.add_vertex()
    v3 = g.add_vertex()

    # Add directed edges between vertices
    e1 = g.add_edge(v1, v2)
    e2 = g.add_edge(v2, v3)
    e3 = g.add_edge(v3, v1)

    outpng=os.path.join(data_dir,"graph.png")
    outgt=os.path.join(data_dir,"my_graph.gt")
    graph_draw(g, vertex_text='v_name', output=outpng)

    # Save the entire graph structure and its properties to a file
    g.save(outgt)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
