class Node:
    def __init__(self, value):
        self.value = value
        self.neighbors = []

def dfs(start_nodes):
    stack = [start_nodes] 
    visited = set()       
    print("The DFS Journey:")
    
    while stack: 
        current_node = stack.pop() 
        if current_node not in visited: 
            print(current_node.value )
            visited.add(current_node)         
            for neighbor in current_node.neighbors:
                if neighbor not in visited:
                    stack.append(neighbor)                
print()

node_A = Node('A')
node_B = Node('B')
node_C = Node('C')
node_D = Node('D')
node_E = Node('E')
node_F = Node('F')

node_A.neighbors = [node_B, node_C] 
node_B.neighbors = [node_D, node_E]
dfs(node_A)






        
