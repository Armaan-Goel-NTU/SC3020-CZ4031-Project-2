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
    
def explain_limit(node: dict) -> str:
    if 'Plans' not in node or len(node['Plans']) != 1:
        return (0, "Limit node must have exactly one child plan.")

    # Extract child plan
    child_plan = node['Plans'][0]
    child_total_cost = child_plan['Total Cost']
    child_startup_cost = child_plan['Startup Cost']
    child_plan_rows = child_plan['Plan Rows']

    # Get the limit value
    limit_count = node.get('Plan Rows', child_plan_rows)  # Default to all rows if no limit specified

    
    if limit_count < child_plan_rows:
        cost_per_row = (child_total_cost - child_startup_cost) / child_plan_rows if child_plan_rows else 0
        total_cost = child_startup_cost + cost_per_row * limit_count
        total_cost = round(total_cost, 2)
        explanation = f"Limit operation processes only the first {limit_count} rows of its child node. "
        explanation += f"Total cost is reduced proportionally based on the reduced number of rows processed. "
        explanation += f"Estimated reduced cost: {total_cost}, based on original total cost: {child_total_cost} and row count: {child_plan_rows}."
    else:
        total_cost = child_total_cost
        total_cost = round(total_cost, 2)
        explanation = f"Limit operation does not reduce the number of rows processed as the limit ({limit_count}) is higher than or equal to the child's row count ({child_plan_rows}). Total cost remains unchanged."
    
    
    return (total_cost, explanation)

def explain_indexscan(node: dict) -> str:
    cpu_index_tuple_cost = float(cache.get_setting("cpu_index_tuple_cost"))
    index_page_cost = float(cache.get_setting("effective_cache_size"))  # This adjusts the page cost based on caching

    # Extract necessary data from the node
    index_name = node.get("Index Name", "unknown index")
    page_count = cache.get_page_count(index_name)  # Assuming this retrieves index-related pages
    row_count = node["Plan Rows"]

    # Calculate costs
    cost = cpu_index_tuple_cost * row_count + index_page_cost * page_count
    explanation = f"Index Scan on {index_name} involves a CPU cost per tuple and an I/O cost per page. " \
                  f"Total tuples (rows) fetched: {row_count}, Index pages read: {page_count}, " \
                  f"cpu_index_tuple_cost={cpu_index_tuple_cost}, index_page_cost={index_page_cost}. " \
                  f"Total cost calculated: {cost}"
    
    return (cost, explanation)


import math

def explain_sort(node: dict) -> str:
    tuples = node["Plan Rows"]
    width = node["Plan Width"]
    num_workers = node.get("Workers", 1) if "Workers" in node else 1  # Safe access to 'Workers' key
    sort_mem_kb = float(cache.get_setting("work_mem"))
    sort_mem_bytes = sort_mem_kb * 1024
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    comparison_cost = cpu_operator_cost * 3  # Increase the weight to account for additional overhead
    seq_page_cost = float(cache.get_setting("seq_page_cost"))
    random_page_cost = float(cache.get_setting("random_page_cost"))
    block_size = float(cache.get_setting("block_size"))

    input_bytes = tuples * width
    num_pages = math.ceil(input_bytes / block_size)

    if input_bytes > sort_mem_bytes:
        # Disk-based sort calculations
        nruns = max(1, input_bytes / sort_mem_bytes)
        merge_order = max(2, math.floor(math.log2(sort_mem_bytes / block_size)))
        log_runs = max(1, math.ceil(math.log(nruns) / math.log(merge_order)))
        npageaccesses = num_pages * log_runs
        disk_io_cost = npageaccesses * (seq_page_cost * 0.75 + random_page_cost * 0.25)
        startup_cost_per_worker = comparison_cost * tuples * math.log(max(2, tuples)) / math.log(2) + disk_io_cost
    else:
        # In-memory sort
        startup_cost_per_worker = comparison_cost * tuples * math.log(max(2, tuples)) / math.log(2)

    # Parallel overhead recalculated more accurately
    total_startup_cost = startup_cost_per_worker * num_workers
    parallel_overhead = total_startup_cost * (0.1)  # Assuming a 10% overhead for coordination in parallel setup
    total_startup_cost += parallel_overhead

    run_cost_per_worker = cpu_operator_cost * tuples
    total_run_cost = run_cost_per_worker * num_workers

    total_cost = total_startup_cost + total_run_cost

    explanation = f"Sort operation on column(s) {node['Sort Key']}, executed in parallel across {num_workers} workers. " \
                  f"Input tuples per worker: {tuples / num_workers}, Total tuples: {tuples}, Tuple width: {width} bytes, " \
                  f"Memory per worker: {sort_mem_bytes / 1024 / 1024} MB, Pages per worker: {num_pages}, " \
                  f"Startup cost per worker: {startup_cost_per_worker:.2f}, Total startup cost: {total_startup_cost:.2f}, " \
                  f"Run cost per worker: {run_cost_per_worker:.2f}, Total run cost: {total_run_cost:.2f}, " \
                  f"Total cost: {total_cost:.2f}."
    return total_cost, explanation

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
    "Sort": explain_sort,
    "Incremental Sort": None,
    "Group": None,
    "Aggregate": None,
    "WindowAgg": None,
    "Unique": None,
    "SetOp": None,
    "LockRows": None,
    "Limit": explain_limit,
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
