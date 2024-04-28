#include <vector>

class Graph {
    public: 
        Graph();
        ~Graph();
        void addEdge(int u, int v);
        void removeEdge(int u, int v);
        int getTotalVertices();
        int getTotalEdges();
        const std::vector<int>& getNeighbors(int u);
    
    private:
        std::vector<std::vector<int>> adjacencyList;
        int totalVertices;
        int totalEdges;
};
