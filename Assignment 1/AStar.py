

def astar(puzzle):
    def h(node):
        # Implement heuristic function
        # For example, Manhattan distance heuristic
        return sum(abs(a // 3 - b // 3) + abs(a % 3 - b % 3) for a, b in ((node.index(i), goal.index(i)) for i in range(1, 9)))

    def f(node, g):
        return g + h(node)

    def neighbors(node):
        empty_index = node.index(0)
        row, col = divmod(empty_index, 3)

        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        valid_neighbors = []

        for move in moves:
            new_row, new_col = row + move[0], col + move[1]
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_index = new_row * 3 + new_col
                new_node = node[:]
                new_node[empty_index], new_node[new_index] = new_node[new_index], new_node[empty_index]
                valid_neighbors.append(new_node)

        return valid_neighbors

    goal = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    open_set = [(h(puzzle), puzzle, [puzzle])]
    closed_set = set()

    while open_set:
        open_set.sort()
        current = open_set.pop(0)
        h_value, node, path = current

        if node == goal:
            return path

        closed_set.add(tuple(node))

        for neighbor in neighbors(node):
            if tuple(neighbor) not in closed_set:
                g_value = len(path)
                f_value = f(neighbor, g_value)
                open_set.append((f_value, neighbor, path + [neighbor]))

    return []

a=astar([0, 4, 1, 3, 8, 2, 6, 7, 5])
print(a)
