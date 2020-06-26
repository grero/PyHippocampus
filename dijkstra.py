import numpy as np


def setup_graph(graph, b, s):
    if s == 1:
        for i in range(0, graph.shape[0]):
            for j in range(0, graph.shape[0]):
                if graph[i, j] == 0:
                    graph[i, j] = b
    if s == 2:
        for i in range(0, graph.shape[0]):
            for j in range(0, graph.shape[0]):
                if graph[i, j] == b:
                    graph[i, j] = 0
    return graph


def exchange_node(graph, a, b):
    buffer = graph[:, a]
    graph[:, a] = graph[:, b]
    graph[:, b] = buffer

    buffer = graph[a, :]
    graph[a, :] = graph[b, :]
    graph[b, :] = buffer
    return graph


def list_dijkstra(l, w, s, d):
    index = w.shape[0]
    while index > 0:
        if w[1, d] == w[w.shape[0]-1, d]:
            l = np.array([l, s])
            index = 0
        else:
            index2 = w.shape[0]
            while index2 > 0:
                if w[index2-1,d] < w[index2-2,d]:
                    l = np.array([l, w[index2,1]])
                    l = list_dijkstra(l,w,s,w[index2,0])
                    index2 = 0
                else:
                    index2 = index2 -1
                index = 0


def dijkstra(graph, source, des):
    if source == des:
        cost = 0
        route = [source]
    else:
        graph = setup_graph(graph, float("inf"), 1)
        if des == 0: #careful
            des = source
        graph = exchange_node(graph, 1, source)
        length = graph.shape[0]
        w = np.zeros((length, length))
        for i in range(1, length):
            w[0, i] = i  # careful
            w[1, i] = graph[0, i]

        d = np.zeros((length, 2))
        for i in range(0, length):
            d[i, 0] = graph[0, i]
            d[i, 1] = i
        d2 = d[1:length, :]
        loop = 2
        while loop <= w.shape[0]-1:
            loop += 1
            d2 = np.sort(d2.view('i8,i8,i8'), order=['f0'], axis=0).view(np.int)
            k = d2[0, 1]
            w[loop-1, 0] = k
            d2 = np.delete(d2, 0, 0)

            for i in range(0, d2.shape[0]):
                if d[d2[i, 1], 0] > (d[k, 0] + graph[k, d2[i, 1]]):
                    d[d2[i, 1], 0] = d[k, 0] + graph[k, d2[i, 1]]
                    d2[i, 0] = d[d2[i, 1], 0]
            for i in range(1, length):
                w[loop-1, i] = d[i, 0]
        if des == source:
            loop = 1
        else:
            loop = des
        cost = w[w.shape[0]-1, des]
        list_dijkstra(loop,w,source,des)
    return cost, route


