from collections import defaultdict
import dgl
import pandas as pd
import numpy as np
import torch


def get_top_corr_events(folder_path):
    quadruple_idx_path = folder_path + '/quadruple_idx.txt'
    data = pd.read_csv(quadruple_idx_path, sep = '\t', names = ["src","rel","tgt","date"])
    data = data.groupby(by = ["date"])[["src", "tgt", "rel"]].apply(lambda x:x).droplevel(1)
    data = data.reset_index()
    merged = pd.merge(data, data, left_on=["src", "tgt", "date"], right_on=["tgt", "src", "date"])

    inverse_rule_mappings = defaultdict(dict)
    most_related_events = defaultdict(list)
    
    for i in range(len(merged)):
        if merged.rel_x[i] != merged.rel_y[i]:
            inverse_rule_mappings[merged.rel_x[i]][merged.rel_y[i]] = inverse_rule_mappings[merged.rel_x[i]].get(merged.rel_y[i],0) + 1
    for relation, mappings in inverse_rule_mappings.items():
        mappings = sorted(mappings.items(), key= lambda x: x[1], reverse=True)
        for events in mappings[:3]:
            most_related_events[relation].append(events[0])
    return most_related_events

def get_sg_by_entity_corr(graph, left, relation, right, date, most_related_events):
    edges = graph.edges(form='all')
    edata = graph.edata

    src_nodes = list(map(int, edges[0]))
    dst_nodes = list(map(int, edges[1]))
    rel_types = list(map(int, edata['type']))
    dates = list(map(int, edata['date']))

    main_events = []
    left_events = []
    right_events = []
    final_events_src = []
    final_events_dst = []
    relations = []
    time = []
    
    for i in range(len(src_nodes)):
        if src_nodes[i] == left and dst_nodes[i] == right and rel_types[i] == relation and dates[i] == date:
            main_events.append((src_nodes[i], dst_nodes[i], rel_types[i], dates[i]))

    for i in range(len(src_nodes)):
        if src_nodes[i]!= right and dst_nodes[i] == left and rel_types[i] in most_related_events[relation] and (dates[i]>=date-5 and dates[i]<=date+5):
            left_events.append((src_nodes[i], dst_nodes[i], rel_types[i], dates[i]))

    for i in range(len(src_nodes)):
        if src_nodes[i]== right and dst_nodes[i] != left and rel_types[i] in most_related_events[relation] and (dates[i]>=date-5 and dates[i]<=date-5):
            right_events.append((src_nodes[i], dst_nodes[i], rel_types[i], dates[i]))

    for i in set(main_events):
        final_events_src.append(i[0])
        final_events_dst.append(i[1])
        relations.append(i[2])
        time.append(i[3])

    for i in set(left_events):
        final_events_src.append(i[0])
        final_events_dst.append(i[1])
        relations.append(i[2])
        time.append(i[3])

    for i in set(right_events):
        final_events_src.append(i[0])
        final_events_dst.append(i[1])
        relations.append(i[2])
        time.append(i[3])
        
    final_events_src = torch.LongTensor(final_events_src)
    final_events_dst = torch.LongTensor(final_events_dst)
    
    # print(final_events_src, final_events_dst, relations, time)
    unique_nodes, edges = torch.unique(torch.cat((final_events_src, final_events_dst)), return_inverse=True)
    # print(len(unique_nodes))
    final_src, final_dst = torch.reshape(edges,(2,-1))
    
    sg = dgl.DGLGraph(num_nodes = len(unique_nodes))
    sg.add_edges(final_src, final_dst)
    sg.ndata.update({'id': unique_nodes.view(-1, 1)})
    sg.edata["type"] = torch.LongTensor(relations)
    sg.edata["date"] = torch.LongTensor(time)

    return set(main_events), set(left_events), set(right_events), sg

def construct_graph(folder_path):
    src, rel, dst, date = [], [], [], []
    quadruple_idx_path = folder_path + '/quadruple_idx.txt'
    with open (quadruple_idx_path, 'r') as qdrple:
        for line in qdrple:
            row = line.split()
            src.append(row[0])
            rel.append(row[1])
            dst.append(row[2])
            date.append(row[3])
    src = np.asarray(src, dtype="int64")
    dst = np.asarray(dst, dtype="int64")
    rel = np.asarray(rel, dtype="int64")
    date = np.asarray(date, dtype="int64")
    uniq_v = np.unique((src, dst))  
    g = dgl.graph((src,dst))
    # g.add_nodes(len(uniq_v))

    # g.add_edges(src, dst) # array list
    g.edata['type'] = torch.LongTensor(rel)
    g.edata['date'] = torch.LongTensor(date)
    g.ids = {}
    idx = 0
    for id in uniq_v:
        g.ids[id] = idx
        idx += 1
    
    return g

print("Generating graph...")
graph = construct_graph("D:\\personal-Shreyas\AIRS\\data\\raw_data\\rawdat\\IND")
print("Graph Created!")
print("getting most correlted events...")
m_r_e = get_top_corr_events("D:\\personal-Shreyas\AIRS\\data\\raw_data\\rawdat\\IND")
print("getting left and right events")
main_events, left_events, right_events = get_sg_by_entity_corr(graph, 1,58,31,0, m_r_e)


print("Main Events:\n")
for i in main_events:
    print(i)

print("Left Events:\n")
for i in left_events:
    print(i)

print("Right Events:\n")
for i in right_events:
    print(i)