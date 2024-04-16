from collections import deque
import psycopg
import json
import re
import decimal

def trunc(a : float) -> float:
    return float(decimal.Decimal(str(a)).quantize(decimal.Decimal('.01'), rounding=decimal.ROUND_DOWN))

cache = None
class Cache():
    def __init__(self, cur: psycopg.Cursor) -> None:
        self.dict = {}
        self.cur = cur
    
    def query_setting(self, setting: str):
        self.cur.execute(f"SHOW {setting}")
        return self.cur.fetchall()[0][0]

    def query_pagecount(self, relation: str):
        self.cur.execute(f"SELECT relpages FROM pg_class WHERE relname = '{relation}'")
        return self.cur.fetchall()[0][0]

    def query_tuplecount(self, relation: str):
        self.cur.execute(f"SELECT reltuples FROM pg_class where relname = '{relation}'")
        return self.cur.fetchall()[0][0]
    
    def get_setting(self, setting: str):
        key = f"setting/{setting}"
        if key not in self.dict:
            self.dict[key] = self.query_setting(setting)
        return self.dict[key]
            
    def get_page_count(self, relation: str):
        key = f"relpages/{relation}"
        if key not in self.dict:
            self.dict[key] = self.query_pagecount(relation)
        return self.dict[key]        

    def get_tuple_count(self, relation: str):
        key = f"reltuples/{relation}"    
        if key not in self.dict:
            self.dict[key] = self.query_tuplecount(relation)
        return self.dict[key]
    
    def set_tuple_count(self, relation: str, count:int):
        key = f"reltuples/{relation}"
        self.dict[key] = count

def explain_seqscan(node: dict) -> str:
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    seq_page_cost = float(cache.get_setting("seq_page_cost"))
    rel = node["Relation Name"]
    page_count = cache.get_page_count(rel)
    row_count = cache.get_tuple_count(rel)
    comment = ""
    workers = 1
    filter_cost = 0

    explanation = f"Sequential Scan has a cpu cost of cpu_tuple_cost * T(R) and a disk cost of seq_page_cost * B(R).\n"
    explanation += f"B(R) = {page_count}, T(R) = {row_count}, cpu_tuple_cost={cpu_tuple_cost}, seq_page_cost={seq_page_cost}.\n"
    if "Filter" in node:
        pattern = r"\([^()]*\)"
        filters = len(re.findall(pattern, node["Filter"]))
        cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
        filter_cost = filters * cpu_operator_cost
        explanation += f"CPU cost per tuple is increased by {filter_cost:.4f} for {filters} filters * cpu_operator_cost ({cpu_operator_cost})\n"

    if node["Parallel Aware"] and "Workers Planned" in node:
        workers = node["Workers Planned"]
        if cache.get_setting("parallel_leader_participation") == "on" and workers < 4:
            workers += 1 - (workers * 0.3)
        explanation += f"The total CPU cost is reduced by a parallelization factor of {workers:.1f}\n" 

    disk_cost = seq_page_cost * page_count
    cost = trunc((cpu_tuple_cost + filter_cost)/workers * row_count + disk_cost)
    expected_cost = node["Total Cost"]
    if cost != expected_cost and "Filter" in node:
        expected_cost -= disk_cost
        expected_cost /= row_count
        expected_cost *= workers
        expected_cost -= cpu_tuple_cost

        
        if expected_cost != filter_cost:
            comment = f"The difference in costs is likely due to the way filtering is handled. The expected filtering cost is {expected_cost:.4f}, but we have used {filter_cost}"

    explanation += f"Plugging in these values, we get {cost}"

    return (cost, explanation, comment)

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

def explain_gather(node: dict) -> str:

    parallel_setup_cost = float(cache.get_setting("parallel_setup_cost"))
    parallel_tuple_cost = float(cache.get_setting("parallel_tuple_cost"))

    if 'Plans' not in node or not node['Plans']:
        return (0, "No child plans for this Gather node.")

    subpath = node['Plans'][0]  # Assuming the first plan is the subpath

    rows = node["Plan Rows"]

    # Extract costs from the subpath
    subpath_startup_cost = subpath['Startup Cost']
    subpath_total_cost = subpath['Total Cost']
    subpath_run_cost = subpath_total_cost - subpath_startup_cost

    # incorporating parallel costs
    startup_cost = subpath_startup_cost + parallel_setup_cost
    run_cost = subpath_run_cost + parallel_tuple_cost * rows

    total_cost = startup_cost + run_cost

    # explanation:
    explanation = f"Gather node coordinates parallel execution. It has startup and run phases:\n"
    explanation += f"Subpath startup cost: {subpath_startup_cost}\n"
    explanation += f"Subpath run cost (excluding startup): {subpath_run_cost}\n"
    explanation += f"Parallel setup cost: {parallel_setup_cost}\n"
    explanation += f"Parallel tuple cost per row: {parallel_tuple_cost}, for {rows} rows\n"
    explanation += f"Total Gather node cost: {total_cost} (including parallel overhead)"

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
    "Gather": explain_gather,
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
                params = fn_dict[node_type](node)
                cost = trunc(params[0])
                explanation = params[1]
                color = "c0ebcc" if cost == node["Total Cost"] else "ebc6c0"
                explanation = f"<b>Calculated Cost</b>: <span style=\"background-color:#{color};\">{cost}</span>\n<b>Explanation</b>: {explanation}"
                if len(params) > 2 and len(params[2]) > 0:
                    explanation += f"\n<b>Comments</b>: {params[2]}"
                return explanation
            except Exception as e:
                return f"<b>Comment</b>: Encountered an error when generating an explanation. {str(e)}"
        
        return f"<b>Comment</b>: I really don't know how to explain the cost for this operator!"

    def explain(self, query:str, force_analysis:bool, log_cb: callable) -> str:
        self.log = log_cb
        if not self.connected():
            return "No database connection found! There is no context for this query."

        cur = self.connection.cursor()
        try:
            cur.execute(f"EXPLAIN (COSTS, FORMAT JSON) {query}")
        except Exception as e:
            return f"Error: {str(e)}"

        plan = cur.fetchall()[0][0][0]['Plan']
        with open("plan.json","w") as f:
            f.write(json.dumps(plan, indent=2))
        node_stack = deque()

        def add_nodes(node, workers):
            if "Workers Planned" in node:
                workers = node["Workers Planned"]
            elif node["Parallel Aware"]:
                node["Workers Planned"] = workers
            node_stack.append(node)
            if "Plans" in node:
                for child_plan in node["Plans"]:
                    add_nodes(child_plan, workers)

        add_nodes(plan, 1)

        if force_analysis:
            stack_copy = node_stack.copy()
            reexplain = False
            for _ in range(len(stack_copy)):
                node = stack_copy.pop()
                if "Relation Name" in node:
                    rel = node["Relation Name"]
                    tuple_count = cache.query_tuplecount(rel)
                    if tuple_count == -1:
                        reexplain = True
                        try:
                            cur.execute(f"ANALYZE {rel}")
                        except Exception as e:
                            return f"Error: {str(e)}"
                    else:
                        cache.set_tuple_count(rel, tuple_count)
                        
            if reexplain:
                return self.explain(query, False, log_cb)
                
        for _ in range(len(node_stack)):
            node = node_stack.pop()
            node["Explanation"] = self.get_explanation(node)

        return plan    
