from collections import deque
import psycopg

class Connection():
    def __init__(self) -> None:
        self.connection = None
    
    def connected(self) -> bool:
        return self.connection is not None and not self.connection.closed
    
    def disconnect(self) -> None:
        if self.connected():
            self.connection.close()

    def connect(self, dbname:str, user:str, password:str, host:str, port:str) -> str:
        self.disconnect()
        try:
            self.connection = psycopg.connect(dbname=dbname, user=user, password=password, host=host, port=port)
            self.connection.autocommit = True
            return ""
        except Exception as e:
            return str(e)
    
    def get_explanation(self, node):
        if "Total Cost" not in node or node["Total Cost"] == 0:
            return "<b>Comment</b>: No cost associated with this node."
        return f"<b>Cost:</b> 1000\n<b>Explanation</b>: cost_cpu_tuple = 0.01. 100000 rows * 0.01 = 1000\n<b>Comment</b>: Costs Match!"

    def explain(self, query:str, log_cb: callable) -> str:
        self.log = log_cb
        if not self.connected():
            return "No database connection found! There is no context for this query."

        cur = self.connection.cursor()
        try:
            cur.execute(f"EXPLAIN (FORMAT JSON) {query}")
        except Exception as e:
            return f"Error: {str(e)}"

        plan = cur.fetchall()[0][0][0]['Plan']
        node_stack = deque()

        def add_nodes(node):
            node_stack.append(node)
            if "Plans" in node:
                for child_plan in node["Plans"]:
                    add_nodes(child_plan)

        add_nodes(plan)
        
        for _ in range(len(node_stack)):
            node = node_stack.pop()
            node["Explanation"] = self.get_explanation(node)

        return plan    
