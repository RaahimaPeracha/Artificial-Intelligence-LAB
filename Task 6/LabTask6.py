graph = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F", "G"],
    "D": [],
    "E": [],
    "F": [],
    "G": []
}

def bfs(start):
    visited = []
    to_visit = [start]
    
    while to_visit:
        current = to_visit[0]
        to_visit = to_visit[1:]
        
        if current not in visited:
            print(current)
            visited.append(current)
            to_visit += graph[current]

bfs("A")