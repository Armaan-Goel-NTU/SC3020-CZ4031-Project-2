from collections import deque
import psycopg

cache = None
class Cache():
    def __init__(self, cur: psycopg.Cursor) -> None:
        self.dict = {}
        self.cur = cur
    
    def query_setting(self, setting: str):
        self.cur.execute(f"SELECT setting FROM pg_settings WHERE name = '{setting}'")
        return self.cur.fetchall()[0][0]

    def query_pagecount(self, relation: str):
        self.cur.execute(f"SELECT pg_relation_size('{relation}'::regclass) / current_setting('block_size')::BIGINT;")
        return self.cur.fetchall()[0][0]
    
    def get_setting(self, setting: str):
        if setting not in self.dict:
            self.dict[setting] = self.query_setting(setting)
        return self.dict[setting]
            
    def get_page_count(self, relation: str):
        key = f"relpages/{relation}"
        if key not in self.dict:
            self.dict[key] = self.query_pagecount(relation)
        return self.dict[key]            

def explain_seqscan(node: dict) -> str:
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    seq_page_cost = float(cache.get_setting("seq_page_cost"))
    print(node["Relation Name"])
    page_count = cache.get_page_count(node["Relation Name"])
    row_count = node["Plan Rows"]

    cost = cpu_tuple_cost * row_count + seq_page_cost * page_count
    explanation = f"Sequential Scan has a cpu cost of cpu_tuple_cost * T(R) and a disk cost of seq_page_cost * B(R).\nB(R) = {page_count}, T(R) = {row_count}, cpu_tuple_cost={cpu_tuple_cost}, seq_page_cost={seq_page_cost}. Plugging in these values, we get {cost}"

    return (cost, explanation)

def explain_append(node: dict) -> str:
    # gather child costs
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    child_costs = [child["Total Cost"] for child in node.get("Plans", [])]
    child_startup_costs = [child["Startup Cost"] for child in node.get("Plans", [])]
    row_count = node["Plan Rows"]

    if not child_costs: 
        return (0, "No child plans for this Append node.")

    min_startup_cost = min(child_startup_costs)
    total_cost = sum(child_costs) + cpu_tuple_cost * row_count * 0.5  # 0.5 is APPEND_CPU_COST_MULTIPLIER
    total_cost=round(total_cost, 1)   #check issue with decimal without round

    #explanation:
    explanation = f"Append node combines several plans. The startup cost is the minimum of the startup costs of its children, and the total cost includes CPU costs associated with processing rows.\n"
    explanation += f"Child startup costs: {child_startup_costs}\n"
    explanation += f"Child total costs: {child_costs}\n"
    explanation += f"Estimated row count: {row_count}\n"
    explanation += f"Minimum startup cost: {min_startup_cost}, Adjusted total cost: {total_cost} (including CPU cost for handling rows)"

    return (total_cost, explanation)
    
fn_dict = {
    "Result": None,
    "ProjectSet": None,
    "ModifyTable": None,
    "Append": explain_append,
    "Merge Append": None,
    "Recursive Union": None,
    "BitmapAnd": None,
    "BitmapOr": None,
    "Nested Loop": None,
    "Merge Join": None,
    "Hash Join": None,
    "Seq Scan": explain_seqscan,
    "Sample Scan": None,
    "Gather": None,
    "Gather Merge": None,
    "Index Scan": None,
    "Index Only Scan": None,
    "Bitmap Index Scan": None,
    "Bitmap Heap Scan": None,
    "Tid Scan": None,
    "Tid Range Scan": None,
    "Subquery Scan": None,
    "Function Scan": None,
    "Table Function Scan": None,
    "Values Scan": None,
    "CTE Scan": None,
    "Named Tuplestore Scan": None,
    "WorkTable Scan": None,
    "Foreign Scan": None,
    "Custom Scan": None,
    "Materialize": None,
    "Memoize": None,
    "Sort": None,
    "Incremental Sort": None,
    "Group": None,
    "Aggregate": None,
    "WindowAgg": None,
    "Unique": None,
    "SetOp": None,
    "LockRows": None,
    "Limit": None,
    "Hash": None
}

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
            global cache
            cache = Cache(self.connection.cursor())
            return ""
        except Exception as e:
            return str(e)
    
    def get_explanation(self, node: dict) -> str:
        if "Total Cost" not in node or node["Total Cost"] == 0:
            return "<b>Comment</b>: No cost associated with this node."

        node_type = node["Node Type"] 
        if node_type in fn_dict and fn_dict[node_type] is not None:
            try:
                cost, explanation = fn_dict[node_type](node)
                comment = "Costs Match!" if cost == node["Total Cost"] else "Mismatch!"
                return f"<b>Cost</b>: {cost}\n<b>Explanation</b>: {explanation}\n<b>Postgres Cost</b>: {node['Total Cost']}\n<b>Comment</b>: {comment}"
            except Exception as e:
                return f"<b>Comment</b>: Encountered an error when generating an explanation. {str(e)}"
        
        return f"<b>Comment</b>: I really don't know how to explain the cost for this operator!"

    def explain(self, query:str, log_cb: callable) -> str:
        self.log = log_cb
        if not self.connected():
            return "No database connection found! There is no context for this query."

        cur = self.connection.cursor()
        try:
            cur.execute(f"EXPLAIN (COSTS, FORMAT JSON) {query}")
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
