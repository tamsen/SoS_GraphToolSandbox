import os

from graph_tool.all import *

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

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

    # Draw the graph and save it as an image
    output_dir="/Users/tdunn/Data/SoS"
    outpng=os.path.join(output_dir,"graph.png")
    outgt=os.path.join(output_dir,"my_graph.gt")
    graph_draw(g, vertex_text='v_name', output=outpng)

    # Save the entire graph structure and its properties to a file
    g.save(outgt)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
