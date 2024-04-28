#include "Graph.h"


// i think this is wrong i need to fix before we attempt bfs 


// basic constructor :D 
Graph::Graph() {
    totalVertices = 0;
    totalEdges = 0;
 
}

// constructor with total vertices, if the user of API wants to specify a graph! 
Graph::Graph(int totalVertices) {
    this->totalVertices = totalVertices;
    totalEdges = 0;
    adjacencyList.resize(totalVertices);
}

Graph::~Graph() {
    adjacencyList.clear();
}

// add an edge to the graph by adding the edge to the adjacency list of both vertices
void Graph::addEdge(int u, int v) {
    adjacencyList[u].push_back(v);
    adjacencyList[v].push_back(u);
    totalEdges++;
}

// remove an edge from the graph by removing the edge from the adjacency list of both vertices
void Graph::removeEdge(int u, int v) {
    adjacencyList[u].erase(std::remove(adjacencyList[u].begin(), adjacencyList[u].end(), v), adjacencyList[u].end());
    adjacencyList[v].erase(std::remove(adjacencyList[v].begin(), adjacencyList[v].end(), u), adjacencyList[v].end());
    totalEdges--;
}

int Graph::getTotalVertices() {
    return totalVertices;
}

int Graph::getTotalEdges() {
    return totalEdges;
}

// access that adjeancecy list from above. ? - check this for correctness @Mihir
const std::vector<int>& Graph::getNeighbors(int u) {
    return adjacencyList[u];
}
