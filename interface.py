from tkinter import ttk, Tk, Text
from pyvis.network import Network
from typing import List
from explain import Connection
import networkx as nx
import platform
import os
import webbrowser

default_text = "Click a node (operator) to get all the relevant info!"

css = """            #info-box {
                position: absolute;
                top: 20px;
                right: 20px;
                background-color: white;
                border: 1px solid black;
                padding: 10px;
            }
"""

div = f'            <div id="info-box">{default_text}</div>'

js = f"""                  function updateInfoBox(content) {{
                    var infoBox = document.getElementById('info-box');
                    infoBox.innerHTML = content.replace(/\\n/g, "<br>");
                  }}

                  network.on('click', function(properties) {{
                    if(properties.nodes.length > 0) {{
                        var nodeId = properties.nodes[0];
                        var clickedNode = nodes.get(nodeId);
                        updateInfoBox(clickedNode.explanation);
                    }} else {{
                        updateInfoBox("{default_text}");
                    }}
                  }});
"""

ImageMapper = {
    'Aggregate': {'image': 'ex_aggregate.svg', 'image_text': 'Aggregate'},
    'Append': {'image': 'ex_append.svg', 'image_text': 'Append'},
    'Bitmap Index Scan': lambda data: {'image': 'ex_bmp_index.svg', 'image_text': data['Index Name']},
    'Bitmap Heap Scan': lambda data: {'image': 'ex_bmp_heap.svg', 'image_text': data['Relation Name']},
    'BitmapAnd': {'image': 'ex_bmp_and.svg', 'image_text': 'Bitmap AND'},
    'BitmapOr': {'image': 'ex_bmp_or.svg', 'image_text': 'Bitmap OR'},
    'CTE Scan': {'image': 'ex_cte_scan.svg', 'image_text': 'CTE Scan'},
    'Function Scan': {'image': 'ex_result.svg', 'image_text': 'Function Scan'},
    'Foreign Scan': {'image': 'ex_foreign_scan.svg', 'image_text': 'Foreign Scan'},
    'Gather': {'image': 'ex_gather_motion.svg', 'image_text': 'Gather'},
    'Gather Merge': {'image': 'ex_gather_merge.svg', 'image_text': 'Gather Merge'},
    'Group': {'image': 'ex_group.svg', 'image_text': 'Group'},
    'GroupAggregate': {'image': 'ex_aggregate.svg', 'image_text': 'Group Aggregate'},
    'Hash': {'image': 'ex_hash.svg', 'image_text': 'Hash'},
    'Hash Join': lambda data: {
        'image': 'ex_join.svg' if not data['Join Type'] else
                 ('ex_hash_anti_join.svg' if data['Join Type'] == 'Anti' else
                  'ex_hash_semi_join.svg' if data['Join Type'] == 'Semi' else
                  'ex_hash.svg'),
        'image_text': 'Join' if not data['Join Type'] else
                      ('Hash Anti Join' if data['Join Type'] == 'Anti' else
                       'Hash Semi Join' if data['Join Type'] == 'Semi' else
                       'Hash ' + data['Join Type'] + ' Join'),
    },
    'HashAggregate': { 'image': 'ex_aggregate.svg','image_text': 'Hash Aggregate'},
    'Index Only Scan': lambda data: {'image': 'ex_index_only_scan.svg', 'image_text': data['Index Name']},
    'Index Scan': lambda data: {'image': 'ex_index_scan.svg', 'image_text': data['Index Name']},
    'Index Scan Backword': {'image': 'ex_index_scan.svg', 'image_text': 'Index Backward Scan'},
    'Limit': {'image': 'ex_limit.svg', 'image_text': 'Limit'},
    'LockRows': {'image': 'ex_lock_rows.svg', 'image_text': 'Lock Rows'},
    'Materialize': {'image': 'ex_materialize.svg', 'image_text': 'Materialize'},
    'Merge Append': {'image': 'ex_merge_append.svg', 'image_text': 'Merge Append'},
    'Merge Join': lambda data: {
        'image': 'ex_merge_anti_join.svg' if data['Join Type'] == 'Anti' else
                 'ex_merge_semi_join.svg' if data['Join Type'] == 'Semi' else
                 'ex_merge.svg',
        'image_text': 'Merge Anti Join' if data['Join Type'] == 'Anti' else
                      'Merge Semi Join' if data['Join Type'] == 'Semi' else
                      'Merge ' + data['Join Type'] + ' Join',
    },
    'ModifyTable': lambda data: {
        'image': 'ex_insert.svg' if data['Operation'] == 'Insert' else
                 'ex_update.svg' if data['Operation'] == 'Update' else
                 'ex_delete.svg' if data['Operation'] == 'Delete' else
                 'ex_merge.svg' if data['Operation'] == 'Merge' else None,
        'image_text': 'Insert' if data['Operation'] == 'Insert' else
                      'Update' if data['Operation'] == 'Update' else
                      'Delete' if data['Operation'] == 'Delete' else
                      'Merge' if data['Operation'] == 'Merge' else None,
    },
    'Named Tuplestore Scan': {'image': 'ex_named_tuplestore_scan.svg','image_text': 'Named Tuplestore Scan',},
    'Nested Loop': lambda data: {
        'image': 'ex_nested_loop_anti_join.svg' if data['Join Type'] == 'Anti' else
                 'ex_nested_loop_semi_join.svg' if data['Join Type'] == 'Semi' else
                 'ex_nested.svg',
        'image_text': 'Nested Loop Anti Join' if data['Join Type'] == 'Anti' else
                      'Nested Loop Semi Join' if data['Join Type'] == 'Semi' else
                      'Nested Loop ' + data['Join Type'] + ' Join',
    },
    'ProjectSet': {'image': 'ex_projectset.svg','image_text': 'ProjectSet'},
    'Recursive Union': {'image': 'ex_recursive_union.svg', 'image_text': 'Recursive Union'},
    'Result': {'image': 'ex_result.svg', 'image_text': 'Result'},
    'Sample Scan': {'image': 'ex_scan.svg', 'image_text': 'Sample Scan'},
    'Scan': {'image': 'ex_scan.svg', 'image_text': 'Scan'},
    'Seek': {'image': 'ex_seek.svg', 'image_text': 'Seek'},
    'SetOp': lambda data: {
    'image': 'ex_hash_setop_intersect_all.svg' if data['Command'] == 'Intersect All' else
             'ex_hash_setop_intersect.svg' if data['Command'].startswith('Intersect') else
             'ex_hash_setop_except_all.svg' if data['Command'] == 'Except All' else
             'ex_hash_setop_except.svg' if data['Command'].startswith('Except') else
             'ex_hash_setop_unknown.svg' if data['Strategy'] == 'Hashed' else
             'ex_setop.svg',
    'image_text': 'Hashed Intersect All' if data['Command'] == 'Intersect All' else
                  'Hashed Intersect' if data['Command'].startswith('Intersect') else
                  'Hashed Except All' if data['Command'] == 'Except All' else
                  'Hash Except' if data['Command'].startswith('Except') else
                  'Hashed SetOp Unknown' if data['Strategy'] == 'Hashed' else
                  'SetOp',
    },
    'Seq Scan': lambda data: {'image': 'ex_scan.svg', 'image_text': data['Relation Name']},
    'Subquery Scan': {'image': 'ex_subplan.svg', 'image_text': 'SubQuery Scan'},
    'Sort': {'image': 'ex_sort.svg', 'image_text': 'Sort'},
    'Tid Scan': {'image': 'ex_tid_scan.svg', 'image_text': 'Tid Scan'},
    'Table Function Scan': {'image': 'ex_table_func_scan.svg', 'image_text': 'Table Function Scan'},
    'Unique': {'image': 'ex_unique.svg', 'image_text': 'Unique'},
    'Values Scan': {'image': 'ex_values_scan.svg', 'image_text': 'Values Scan'},
    'WindowAgg': {'image': 'ex_window_aggregate.svg', 'image_text': 'Window Aggregate'},
    'WorkTable Scan': { 'image': 'ex_worktable_scan.svg', 'image_text': 'WorkTable Scan'},
}

class Visualizer():
    def new_viz(self, plan: dict) -> None:
        graph = nx.DiGraph()

        self.node_id = 0
        self.imagemap = {}
        self.add_nodes_and_edges(graph, plan, 0)

        pos = nx.multipartite_layout(graph)

        net = Network('1080px', '1920px', directed=True)
        net.toggle_drag_nodes(False)
        net.toggle_physics(False)

        for node in list(graph.nodes(data="label")):
            color = {
                "border": "#000000",
                "background": "#FFFFFF",
                "highlight": {
                    "border": "#000000",
                    "background": "#E6ECF5"
                }
            }
            img = f"img/{self.imagemap[node[0]]}"
            net.add_node(node[0], label=node[1], explanation=f"Cost Explanation for {node[1]}\nPostgres says 1000. This is because cost_cpu_tuple = 0.01. 100000 rows * 0.01 = 1000", x = pos[node[0]][0] * 1000, y = pos[node[0]][1] * 1000, shape="circularImage", image=img, borderWidth = 1.5, borderWidthSelected = 2, color=color, size=25)
        
        for edge in graph.edges:
            net.add_edge(edge[0], edge[1], color="black", width = 2, chosen=False, arrowStrikethrough=False)

        output_file = "QEP.html"
        html_content = self.modify_html(net.generate_html().splitlines())
        with open(output_file, "w") as f:
            f.write(html_content)
        absolute_path = os.path.join(os.getcwd(), output_file)
        webbrowser.open(f"file://{absolute_path}")
        
    
    def add_nodes_and_edges(self, graph: nx.DiGraph, plan: dict, subset: int, parent=None) -> None:
        self.node_id += 1

        label = plan["Node Type"]
        img = "ex_unknown.svg"
        if label in ImageMapper:
            if callable(ImageMapper[label]):
                img = (ImageMapper[label](plan))["image"]
                label = (ImageMapper[label](plan))["image_text"]
            else:
                img = ImageMapper[label]["image"]
                label = ImageMapper[label]["image_text"]
        self.imagemap[self.node_id] = img

        graph.add_node(self.node_id, subset=subset, label=label)
        if parent:
            graph.add_edge(self.node_id, parent)
        if "Plans" in plan:
            parent_id = self.node_id
            for child_plan in plan["Plans"]:
                self.add_nodes_and_edges(graph, child_plan, subset-1, parent_id)

    def modify_html(self, html: List[str]) -> str:
        iter = enumerate(html)
        index = 0
        line = ""

        while "</style>" not in line:
            index, line = next(iter)
        
        html.insert(index, css)

        while '<div id="mynetwork"' not in line:
            index, line = next(iter)
        
        html.insert(index+1, div)

        while 'return network;' not in line:
            index, line = next(iter)
        
        html.insert(index, js)

        return '\n'.join(html)

def set_window_size(frame: ttk.Frame, size: tuple) -> None:
    root:Tk = frame.master
    width, height = size
    screenwidth = root.winfo_screenwidth()
    screenheight = root.winfo_screenheight()
    alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
    root.geometry(alignstr)
    dark_title_bar(root)

def dark_title_bar(window: Tk) -> None:
    if platform.system() == "Windows":
        from ctypes import windll, c_int, byref, sizeof
        window.update()
        set_window_attribute = windll.dwmapi.DwmSetWindowAttribute
        get_parent = windll.user32.GetParent
        hwnd = get_parent(window.winfo_id())
        set_window_attribute(hwnd, 20, byref(c_int(2)), sizeof(c_int))

        # TODO: check if needed once or every time on Lab Win10 (2019 version)
        if platform.release() == "10":
            window.withdraw()
            window.deiconify()
 
LOGIN_SIZE = (400, 375)
APP_SIZE = (600, 500)
VIZ = Visualizer()
CONN = Connection()

class Login:
    def __init__(self, root:ttk.Frame, app_frame:ttk.Frame) -> None:
        self.app_frame = app_frame
        
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)

        self.header_label=ttk.Label(root, justify="center", text="Connect to PostgreSQL")
        self.header_label.grid(row=0,column=0,padx=10,pady=10)

        db_label=ttk.Label(root, justify="center", text="Name:")
        db_label.grid(row=1,column=0)

        self.db_input=ttk.Entry(root, justify="center")
        self.db_input.insert("end", "TPC-H")
        self.db_input.grid(row=1,column=1,pady=5,padx=10)

        user_label=ttk.Label(root, justify="center", text="User:")
        user_label.grid(row=2,column=0)

        self.user_input=ttk.Entry(root, justify="center")
        self.user_input.insert("end", "postgres")
        self.user_input.grid(row=2,column=1,pady=5,padx=10)

        pw_label=ttk.Label(root, justify="center", text="Pass:")
        pw_label.grid(row=4,column=0)

        self.pw_input=ttk.Entry(root, justify="center", show="*")
        self.pw_input.grid(row=4,column=1,pady=5,padx=10)

        host_label=ttk.Label(root, justify="center", text="Host")
        host_label.grid(row=3,column=0)

        self.host_input=ttk.Entry(root, justify="center")
        self.host_input.insert("end", "localhost")
        self.host_input.grid(row=3,column=1,pady=5,padx=10)

        port_label=ttk.Label(root, justify="center", text="Port:")
        port_label.grid(row=5,column=0)

        self.port_input=ttk.Entry(root, justify="center")
        self.port_input.insert("end", "5432")
        self.port_input.grid(row=5,column=1,pady=5,padx=10)

        connect_btn=ttk.Button(root, text="Connect", command=self.connect_btn_command)
        connect_btn.grid(row=6,column=0,pady=10)
        
        self.error_label = Text(root, height=4, state="disabled")
        self.error_label.grid(row=7,column=0,columnspan=2,pady=5,sticky="nsew")

        self.root = root
    
    def set_error(self, text:str) -> None:
        self.error_label["state"] = "normal"
        self.error_label.delete(1.0,"end")
        self.error_label.insert("end",text)
        self.error_label["state"] = "disabled"

    def connect_btn_command(self) -> None:
        print(f"Pass: {self.pw_input.get()}")
        connect_res = CONN.connect(dbname=self.db_input.get(), user=self.user_input.get(), password=self.pw_input.get(), host=self.host_input.get(), port=self.port_input.get())
        self.set_error(connect_res)
        if len(connect_res) == 0:
            set_window_size(self.app_frame, APP_SIZE)
            self.app_frame.tkraise()
            
class App():
    def __init__(self, root:ttk.Frame, login_frame:ttk.Frame) -> None:
        self.login_frame = login_frame

        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)

        header_label = ttk.Label(root, justify="left", text="Enter one query at a time and click 'Explain' to generate an explanation. This will launch an interactable graph in your browser", wraplength=450)
        disconnect_btn = ttk.Button(root, text="Disconnect", command=self.disconnect_btn_command)

        header_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        disconnect_btn.grid(row=0, column=1, padx=10, pady=5, sticky="e")

        self.explain_input = Text(root, height=25, highlightthickness=1, highlightbackground = "white", highlightcolor= "white")
        self.explain_input.grid(row=1,column=0,columnspan=2,pady=5,sticky="nsew")

        explain_btn = ttk.Button(root, text="Explain", command=self.explain_btn_command)
        explain_btn.grid(row=2, column=0, columnspan=2, pady=5, padx=10)

        self.explain_status = Text(root, height=5, state="disabled")
        self.explain_status.grid(row=3,column=0,columnspan=2,pady=5,sticky="nsew")
    
    def set_status(self, status) -> None:
        self.explain_status["state"] = "normal"
        self.explain_status.delete(1.0,"end")
        self.explain_status.insert("end", status)
        self.explain_status["state"] = "disabled"

    def disconnect_btn_command(self) -> None:
        self.set_status("")
        self.explain_input.delete(1.0,"end")

        CONN.disconnect()
        set_window_size(self.login_frame, LOGIN_SIZE)
        self.login_frame.tkraise()
    
    def explain_btn_command(self) -> None:
        explain_res = CONN.explain(query=self.explain_input.get("1.0",'end-1c'))
        if type(explain_res) == str:
            self.set_status(status=explain_res)
        else:
            self.set_status("Plan generated successfully! Adding explanations now.")
            VIZ.new_viz(plan=explain_res)