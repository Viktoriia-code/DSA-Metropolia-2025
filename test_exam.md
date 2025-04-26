## Question 14

Write a class that implements a sorted Doubly linked list. The list is sorted always. When inserting a new element, the new node will be inserted at the right place given its value.

For example, after inserting values 9, 5, 7, 1, 3, the list will contain (in this order): [1, 3, 5, 7, 9]

A skeleton of the class is provided. Implement at least the methods that are defined. You can implement new helper methods if you consider it necessary.

```
class Node:
    def __init__(self, data):
        self._data = data
        self._next = None
        self._previous = None
    def next(self):
        return self._next
    def previous(self):
        return self._previous
    def link_next(self, node):
        self._next = node
    def link_previous(self, node):
        self._previous = node
    def value(self):
        return self._data


class Sorted_Doubly_Linked_List:
    def __init__(self):
        self._head_node = None

    def print_list(self):
        current = self._head_node
        print('[', end='')
        while current is not None:
            print(current.value(), end='')
            current = current.next()
            if current is not None:
                print(', ', end='')
        print(']')

    def append(self, data):

        # Implement this method
        new_node = Node(data)

        # Case 1: List is empty
        if self._head_node is None:
            self._head_node = new_node
            return
    
        current = self._head_node
    
        # Case 2: New node should be inserted at the beginning
        if data < current.value():
            new_node.link_next(current)
            current.link_previous(new_node)
            self._head_node = new_node
            return
    
        # Traverse to find correct position
        while current.next() is not None and current.next().value() < data:
            current = current.next()
    
        # Insert after 'current'
        next_node = current.next()
        current.link_next(new_node)
        new_node.link_previous(current)
    
        if next_node:
            new_node.link_next(next_node)
            next_node.link_previous(new_node)
```

## Question 15
Given the Sorted_Doubly_Linked_List that you implemented in the previous exercise. Implement a function that merges two of them. The function is actually a method in the class, that accepts another object of the same class as parameter and reorganize the links to merge both list over the current object. No new node is created, they are only reorganized. After the merging, both objects point to the same sorted nodes.

For example, given list [1, 3, 5, 7, 9] and [0, 2, 4, 6, 8,]. After merging them, both objects contain the same list [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

```
    # Respect the indentation, so the method can be added to the class
    def merge(self, other):
        # Pointers to the start of both lists
        node1 = self._head_node
        node2 = other._head_node

        # Reset both heads to None (will be updated during merge)
        self._head_node = None
        other._head_node = None

        # Dummy node to simplify link logic
        dummy = Node(0)
        tail = dummy

        # Merge process like in merge sort
        while node1 and node2:
            if node1.value() <= node2.value():
                next_node = node1.next()
                tail.link_next(node1)
                node1.link_previous(tail)
                tail = node1
                node1 = next_node
            else:
                next_node = node2.next()
                tail.link_next(node2)
                node2.link_previous(tail)
                tail = node2
                node2 = next_node

        # Attach the remaining nodes
        remaining = node1 if node1 else node2
        while remaining:
            next_node = remaining.next()
            tail.link_next(remaining)
            remaining.link_previous(tail)
            tail = remaining
            remaining = next_node

        # Update heads of both lists
        self._head_node = dummy.next()
        if self._head_node:
            self._head_node.link_previous(None)
        other._head_node = self._head_node
```

## Question 21
Write a function that recursively solves the Tower of Hanoi game (follow the link to see the rules if you don't know the game)

A skeleton of the function is provided. The function should be callable with the number of pieces to move as only parameter

The general idea is:

Move n-1 from source to auxiliary
Move 1 from source to destination
Move n-1 from auxiliary to destination
And the base case being when the number of pieces to move is 1.

The function uses a list of 3 lists that work as stacks. When moving a piece, it is popped from one stack to be pushed into the other. The function should print the main stack (the list of lists) at every step. That is, first the original position and then after each movement. Finally the function should return the number of moves done to fulfill the task.

Even if this instructions can seem complicated, a solution can be accomplished writing less than 10 lines more than already are.

```
def tower_of_hanoi(count, stacks=None, source=0, auxiliary=1, destination=2, moves=0):
    if not stacks:
        stacks = [['ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i] for i in range(count-1, -1, -1)], [], []]
        moves = 1
        print(stacks)
    if count == 1:
        stacks[destination].append(stacks[source].pop())
        print(stacks)
        moves += 1
        return moves
    
    moves += tower_of_hanoi(count-1, stacks, source, destination, auxiliary)
    stacks[destination].append(stacks[source].pop())
    print(stacks)
    moves += 1
    moves += tower_of_hanoi(count-1, stacks, auxiliary, source, destination)
    return moves
```

## Question 26
Write the second entry of a hash table that results from using the hash function, f(x) = (5*n + 3) mod 8, to hash the keys 50, 27, 59, 1, 43, 52, 40, 63, 9 and 56, assuming collisions are handled by chaining. Write the result as a Python list.

Answer: [27, 49 ,53]

## Question 27

Write the hash table that results from using the hash function, f(x) = (5*n + 3) mod 8, to hash the keys 50, 27, 59, 1, 43, 52, 40 and 63, assuming collisions are handled by linear addressing. Write the result as a Python list.

The correct answer is: [1, 63, 27, 59, 43, 50, 40, 52]

## Question 30

Given the following adjacency map of a graph:

graph = {
    A: {B: (A, B), D: (A, D), G: (A, G)},
    B: {A: (A, B), C: (B, C), D: (B, D), E: (B, E)},
    C: {B: (B, C), F: (C, F)},
    D: {A: (A, D), B: (B, D), E: (D, E), H: (D, H)},
    E: {B: (B, E), D: (D, E), F: (E, F), H: (E, H)},
    F: {C: (C, F), E: (E, F), J: (F, J)},
    G: {A: (A, G), H: (G, H)},
    H: {D: (D, H), E: (E, H), G: (G, H), J: (H, J)},
    J: {F: (F, J), H: (H, J)}
}


Are  vertices C and D adjacents? no

What is the minimum number of edges to go from vertex C to vertex G? 3

## Question 31

Given a matrix map (implemented in Python as a list of lists) that represents a series of nodes and their connections (two nodes are connected if they are neighbours, including diagonally). Write a function that finds the number of independent groups of nodes. 1 or more nodes are independent if they are not connected with other nodes.

An example matrix can be:

map = [
    [1, 1, 1, 0, 1],
    [1, 1, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 0, 1]
]

In this matrix there are 4 different groups of nodes. Notice that 2 nodes can be neighbours also in diagonal, and hence, they belong to the same group.

The idea is to maintain a list of visited nodes and to traverse the positions of the matrix, checking if the position contains a node (value is 1) and is not visited, increment a group counter and apply a DFS function on it, that visits all connected nodes to this one. The function will finally return the counter giving the number of independent groups.

You can create helper functions you consider necessary. For example, the DFS function is a probably a good idea to be implemented on its own recursive function.

The DFS function should check the 8 possible neighbours positions that contain a node (value is 1) and are not visited, and apply DFS recursively on them. DFS should update the list of visited nodes. The function should take into account that possible neighbour's possition is out the matrix.

Note: DFS = Deep-First Search

```
def DFS(coord, visited, map):
    x, y = coord
    rows = len(map)
    cols = len(map[0])

    # Mark the current node as visited
    visited[x][y] = True

    # All 8 possible directions (including diagonals)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1), (1, 0), (1, 1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        # Check boundaries
        if 0 <= nx < rows and 0 <= ny < cols:
            if map[nx][ny] == 1 and not visited[nx][ny]:
                DFS((nx, ny), visited, map)

def get_groups(map):
    rows = len(map)
    cols = len(map[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    group_count = 0

    for i in range(rows):
        for j in range(cols):
            if map[i][j] == 1 and not visited[i][j]:
                DFS((i, j), visited, map)
                group_count += 1

    return group_count
```

## Question 32

You are considering traveling to a certain country for your holidays. You already have a list of cities you want to visit and your plan is to stay in one of the cities and visit the rest of them one city a day.

As you want your trip to be as cheap as possible and you don't have any city preference to stay in, you plan to write a function that given a graph with the cities, the connection between them and the cost associated to travel between them, gives you what is the best city to stay in. That is, the city that has a minimum cost to travel to the rest of the cities.

Given the graph object and the dijkstra algorithm you studied in the course, write a modified dijkstra function and main function that gives you the answer. Your function should accept the graph as a parameter and call your modified dijkstra function on each of the vertices (cities). The modified dijkstra function should return the table with the costs, so you can collect the costs for all cities and calculate which one of them has a smaller value for for traveling to the rest of cities (the minimum sum of all costs for each city). Your function should return a tuple containing the name of the city (the value of the vertex) and the total cost of traveling to the rest of the cities from that one.

```
import heapq

class Vertex:
    def __init__(self, value):
        self._value = value
    def __repr__(self):
        return f'<Vertex: {self._value}>'
    def __hash__(self):
        return hash(id(self))
        
class Edge:
    def __init__(self, u, v, x):
        self._first = u
        self._second = v
        self._value = x
    
    def __repr__(self):
        return f'<Edge ({self._value}): {self._first} --> {self._second}>'
    def endpoints(self):
        return (self._first, self._second)
    def opposite(self, v):
        return self._second if v is self._first else self._first
    def value(self):
        return self._value
    def __hash__(self):
        return hash( (self._first, self._second))

class Graph:
    def __init__(self, adj_map = None):
        if adj_map:
            self._adj_map = adj_map
        else:
            self._adj_map = {}
    def get_vertices(self):
        return self._adj_map.keys()
    def get_edges(self):
        """Return a set of all edges of the graph."""
        result = set()        
        for secondary_map in self._adj_map.values():
            result.update(secondary_map.values())       
        return result
    def get_edge(self, u, v):
        """
        Returns the edge from u to v, or None if not adjacents.
        """
        return self._adj_map[u].get(v)
    def degree(self, u):
        """
        Returns the number of edges incident to vertex u
        """
        return len(self._adj_map[u])
    def get_adjacent_vertices(self, u):
        """
        Return a list of the adjacent vertices of a given vertex
        """
        return list(self._adj_map[u].keys())
    def get_incident_edges(self, u):
        """
        Returns edges incident to vertex u
        """
        return list(self._adj_map[u].values())
    def add_vertex(self, value):
        vertex = Vertex(value)
        self._adj_map[vertex] = {}
        return vertex
    
    def add_edge(self, u, v, x=None):
        edge = Edge(u, v, x)
        self._adj_map[u][v] = edge
        self._adj_map[v][u] = edge
    def get_adj_map(self):
        return self._adj_map
    def get_adj_matrix(self):
        all_vertices = self._adj_map.keys()
        return [[int(bool(self._adj_map[u].get(v))) for v in all_vertices] for u in all_vertices]
        
def dijkstra(graph, start):
    distances = {v: float('inf') for v in graph.get_vertices()}
    distances[start] = 0
    # Use (distance, vertex_value, vertex_object) for heap
    pq = [(0, start._value, start)]

    while pq:
        current_distance, _, current_vertex = heapq.heappop(pq)

        if current_distance > distances[current_vertex]:
            continue

        for neighbor in graph.get_adjacent_vertices(current_vertex):
            edge = graph.get_edge(current_vertex, neighbor)
            cost = edge.value()

            if cost is None:
                continue

            new_distance = current_distance + cost

            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(pq, (new_distance, neighbor._value, neighbor))

    return distances


def get_best_city(graph):
    best_city = None
    min_total_cost = float('inf')

    # Use get_vertices to iterate over the graph's cities
    for city in graph.get_vertices():
        distances = dijkstra(graph, city)
        total_cost = sum(distances.values())

        if total_cost < min_total_cost:
            min_total_cost = total_cost
            best_city = city

    return (best_city._value, min_total_cost)
```

## Question 35
What does each pop() call return within the following sequence of priority queue operations?

add(5)
add(4)
add(7)
add(1)
pop( )
add(3)
add(6)
pop( )
pop( )
add(8)
pop( )
add(2)
pop( )
pop( )
Write the answer as a comma separated list of numbers

Answer: 1,3,4,5,2,6

