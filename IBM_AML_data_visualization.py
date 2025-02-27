import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

# Create figure with 4 subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
axs = axs.flatten()

# Helper function to format node labels
def format_node(account_id):
    return account_id[-4:]  # Show just last 4 chars

# 1. CYCLE Pattern (3-hop)
cycle_data = [
    ["2022/09/03 15:36", "0119", "811C597B0", "0222", "811B83280", 2692.50],
    ["2022/09/05 06:54", "0222", "811B83280", "0048309", "811C599A0", 2638.36],
    ["2022/09/06 13:44", "0048309", "811C599A0", "0119", "811C597B0", 2795.93]
]

G_cycle = nx.DiGraph()
for trans in cycle_data:
    G_cycle.add_edge(format_node(trans[2]), format_node(trans[4]), 
                    weight=trans[5]/500, 
                    timestamp=datetime.strptime(trans[0], "%Y/%m/%d %H:%M"))

pos_cycle = nx.circular_layout(G_cycle)
nx.draw_networkx(G_cycle, pos=pos_cycle, ax=axs[0], node_color='skyblue', 
                node_size=1500, arrowsize=20, font_weight='bold', 
                width=[G_cycle[u][v]['weight'] for u, v in G_cycle.edges()])
axs[0].set_title("3-hop CYCLE Pattern", fontsize=16)
axs[0].axis('off')

# 2. FAN-IN Pattern (5-degree)
fan_in_data = [
    ["2022/09/03 17:21", "015", "803A52250", "019535", "80B18E250", 1161.00],
    ["2022/09/05 19:45", "0137888", "80DFCA010", "019535", "80B18E250", 1316.39],
    ["2022/09/06 08:40", "025866", "802C6C120", "019535", "80B18E250", 3405.82],
    ["2022/09/06 12:52", "011974", "801998AE0", "019535", "80B18E250", 4109.34],
    ["2022/09/07 06:05", "0028237", "80AD1C820", "019535", "80B18E250", 10003.75]
]

G_fan_in = nx.DiGraph()
target = format_node("80B18E250")
for trans in fan_in_data:
    G_fan_in.add_edge(format_node(trans[2]), target, 
                      weight=trans[5]/1000, 
                      timestamp=datetime.strptime(trans[0], "%Y/%m/%d %H:%M"))

pos_fan_in = nx.spring_layout(G_fan_in)
nx.draw_networkx(G_fan_in, pos=pos_fan_in, ax=axs[1], node_color='lightgreen',
                node_size=1500, arrowsize=20, font_weight='bold',
                width=[G_fan_in[u][v]['weight'] for u, v in G_fan_in.edges()])
axs[1].set_title("5-degree FAN-IN Pattern", fontsize=16)
axs[1].axis('off')

# 3. SCATTER-GATHER Pattern
scatter_data = [
    ["2022/09/03 19:30", "0012979", "80D9A9270", "027", "80C771F80", 337462.77],
    ["2022/09/05 12:41", "027", "80C771F80", "0140257", "80F0169B0", 126134.94],
    ["2022/09/03 21:27", "0012979", "80D9A9270", "0024941", "809DC9020", 22406.35],
    ["2022/09/05 18:27", "0024941", "809DC9020", "0140257", "80F0169B0", 113748.79],
    ["2022/09/04 17:35", "0012979", "80D9A9270", "0117143", "806ED9F10", 855353.98],
    ["2022/09/06 08:34", "0117143", "806ED9F10", "0140257", "80F0169B0", 73631.42]
]

G_scatter = nx.DiGraph()
for trans in scatter_data:
    G_scatter.add_edge(format_node(trans[2]), format_node(trans[4]), 
                      weight=np.log10(trans[5])/3, 
                      timestamp=datetime.strptime(trans[0], "%Y/%m/%d %H:%M"))

pos_scatter = nx.shell_layout(G_scatter)
nx.draw_networkx(G_scatter, pos=pos_scatter, ax=axs[2], node_color='salmon',
                node_size=1500, arrowsize=20, font_weight='bold',
                width=[G_scatter[u][v]['weight'] for u, v in G_scatter.edges()])
axs[2].set_title("SCATTER-GATHER Pattern", fontsize=16)
axs[2].axis('off')

# 4. FAN-OUT Pattern (sample of 8 from the 16-degree pattern)
fan_out_data = [
    ["2022/09/01 00:06", "021174", "800737690", "012", "80011F990", 2848.96],
    ["2022/09/01 04:33", "021174", "800737690", "020", "80020C5B0", 8630.40],
    ["2022/09/01 09:14", "021174", "800737690", "020", "80006A5E0", 35642.49],
    ["2022/09/01 09:56", "021174", "800737690", "00220", "8007A5B70", 5738987.96],
    ["2022/09/01 11:28", "021174", "800737690", "001244", "80093C0D0", 7254.53],
    ["2022/09/01 13:13", "021174", "800737690", "00513", "80078E200", 6990.87],
    ["2022/09/01 14:11", "021174", "800737690", "020", "80066B990", 12536.92],
    ["2022/09/02 15:40", "021174", "800737690", "00410", "8002CC310", 3511.82]
]

G_fan_out = nx.DiGraph()
source = format_node("800737690")
for trans in fan_out_data:
    G_fan_out.add_edge(source, format_node(trans[4]), 
                      weight=np.log10(trans[5])/2, 
                      timestamp=datetime.strptime(trans[0], "%Y/%m/%d %H:%M"))

pos_fan_out = nx.spring_layout(G_fan_out)
nx.draw_networkx(G_fan_out, pos=pos_fan_out, ax=axs[3], node_color='lightyellow',
                node_size=1500, arrowsize=20, font_weight='bold',
                width=[G_fan_out[u][v]['weight'] for u, v in G_fan_out.edges()])
axs[3].set_title("FAN-OUT Pattern (partial, 8 of 16)", fontsize=16)
axs[3].axis('off')

plt.tight_layout()
plt.savefig("aml_patterns_visualization.png", dpi=300, bbox_inches='tight')
plt.show()