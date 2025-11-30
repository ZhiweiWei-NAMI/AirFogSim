import networkx as nx
import random
import matplotlib.pyplot as plt

def generate_random_dag(num_nodes, edge_probability):
    """
    Generates a random DAG using NetworkX.

    Args:
        num_nodes: The number of nodes in the DAG.
        edge_probability: The probability of an edge existing between two nodes.

    Returns:
        A NetworkX DiGraph representing the DAG.
    """
    dag = nx.DiGraph()
    dag.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_probability:
                dag.add_edge(i, j)

    # Ensure the graph is acyclic
    if not nx.is_directed_acyclic_graph(dag):
        return generate_random_dag(num_nodes, edge_probability) # Regenerate if cyclic

    return dag

# Example usage:
num_nodes = 5
edge_probability = 0.3
dag = generate_random_dag(num_nodes, edge_probability)

# Print the edges:
print("Edges:", dag.edges())
print("Nodes:", dag.nodes())

# Draw the DAG (optional):
nx.draw(dag, with_labels=True, node_color="lightblue", node_size=500, font_size=10, font_weight="bold")
plt.title("Generated DAG")
plt.savefig("dag.png")
plt.show()
plt.close()