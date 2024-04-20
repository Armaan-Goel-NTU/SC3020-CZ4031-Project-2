from collections import deque
import psycopg
import json
import re
import math
import decimal

def truncate_cost(a : float) -> float:
    return round(a, 2)

def truncate(a : float) -> float:
    return float(decimal.Decimal(str(a)).quantize(decimal.Decimal('.0001'), rounding=decimal.ROUND_DOWN))

cache = None
class Cache():
    def __init__(self, cur: psycopg.Cursor) -> None:
        self.dict = {}
        self.cur = cur
    
    def query_setting(self, setting: str):
        self.cur.execute(f"SELECT setting FROM pg_settings WHERE name = '{setting}'")
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
    cost = truncate_cost((cpu_tuple_cost + filter_cost)/workers * row_count + disk_cost)
    expected_cost = node["Total Cost"]
    if cost != expected_cost:
        expected_cost -= disk_cost
        expected_cost /= row_count
        expected_cost *= workers
        expected_cost -= cpu_tuple_cost

        if expected_cost != filter_cost:
            comment = f"The difference in costs is likely due to the way filtering/functions are handled. The expected filtering/function cost is {expected_cost:.4f}, but we have used {filter_cost}"

    explanation += f"Plugging in these values, we get {cost}"

    return (cost, explanation, comment)

def align(val : int, len : int) -> int:
    return val + (len - (val % len))

def explain_materialize(node: dict) -> str:
    tuples = node["Plan Rows"]
    width = node["Plan Width"]
    child = node["Plans"][0]
    startup_cost = child["Startup Cost"]

    explanation = f"Materialize has the same startup cost as its child. ({startup_cost})\n"
    work_mem_bytes = float(cache.get_setting("work_mem")) * 1024
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    seq_page_cost = float(cache.get_setting("seq_page_cost"))
    nbytes = tuples * (align(width, 8) + align(23, 8))
    explanation += f"Materialize charges 2 * cpu_operator_cost ({cpu_operator_cost}) per tuple as overhead. There are {tuples} tuple(s)\n"
    cost = 2 * cpu_operator_cost * tuples
    block_size = float(cache.get_setting("block_size"))

    if nbytes > work_mem_bytes:
        explanation += f"The relation to materialize is larger that working memory space of {work_mem_bytes / 1024}KB\n"
        npages = math.ceil(nbytes/block_size)
        explanation += f"Disk costs will be incurred. The projected amount to materialize is {nbytes}, which will take {npages} to fit with a page size of {block_size}\n"
        f"seq_page_cost ({seq_page_cost}) will be incurred for each page.\n"
        cost += npages
    
    explanation += f"Additional cost incurred is {cost:.2f}"
    return (cost + child['Total Cost'], explanation)

def explain_merge_append(node: dict) -> str: 
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    tuples = sum(child["Plan Rows"] for child in node.get("Plans", []))

    N = max(2,len(node["Plans"]))
    logN = math.log2(N)
    comparison_cost = 2.0 * cpu_operator_cost

    startup_cost = comparison_cost * N * logN
    explanation = f"In the startup phase, a Heap is built for each child node (N={N} in this case)\n"
    explanation += f"This will cost N * log2(N) * comparison cost. Comparison cost is 2 * cpu_operator_cost ({cpu_operator_cost})\n"
    explanation += f"Additional startup cost comes out to be {startup_cost:.2f}\n"

    run_cost = tuples * comparison_cost * logN
    run_cost += cpu_tuple_cost * 0.5 * tuples
    explanation += f"A per-tuple heap maintaince cost of comparison_cost * log2(N) applied.\n"
    explanation += f"cpu_tuple_cost ({cpu_tuple_cost}) * the append multiplier (0.5) is also applied per tuple for overhead\n"
    explanation += f"For {tuples} tuples, the run cost is an additional {run_cost:.2f}"

    return (startup_cost + run_cost + sum(child["Total Cost"] for child in node.get("Plans", [])), explanation)


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

def explain_gather_merge(node: dict) -> str:
    parallel_setup_cost = float(cache.get_setting("parallel_setup_cost"))
    parallel_tuple_cost = float(cache.get_setting("parallel_tuple_cost"))
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    rows = node["Plan Rows"]
    comparison_cost = 2.0 * cpu_operator_cost

    subpath = node['Plans'][0]  # Assuming the first plan is the subpath
    subpath_total_cost = subpath['Total Cost']

    N = node["Workers Planned"] + 1
    logN = math.log2(N) 

    startup_cost = comparison_cost * N * logN
    startup_cost += parallel_setup_cost
    explanation = f"In the startup phase, a Heap is built for each parallel worker ({N} in this case)\n"
    explanation += f"This will cost N * log2(N) * comparison cost. Comparison cost is 2 * cpu_operator_cost ({cpu_operator_cost})\n"
    explanation += f"Startup also includes the parallel setup cost of {parallel_setup_cost}\n"
    explanation += f"Additional startup cost comes out to be {startup_cost:.2f}\n"

    run_cost = rows * comparison_cost * logN
    run_cost += cpu_operator_cost * rows
    run_cost += parallel_tuple_cost * rows * 1.05
    explanation += f"A per-tuple heap maintaince cost of comparison_cost * log2(N) applied.\n"
    explanation += f"cpu_operator_cost ({cpu_operator_cost}) is also applied per tuple for heap management overhead\n"
    explanation += f"Lastly, parallel_tuple_cost ({parallel_tuple_cost}) and a 5% penalty to wait for every work is also incurred.\n"
    explanation += f"For {rows} tuples, the run cost is an additional {run_cost:.2f}"

    return (startup_cost + run_cost + subpath_total_cost, explanation)

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

def explain_result(node: dict) -> str:
    expected_cost = node["Total Cost"]
    tuples = node["Plan Rows"]
    comment = ""
    if "Plans" in node:
        cost = node["Plans"][0]["Total Cost"]
        explanation = "Result usually has additional no cost associated with it."
        if expected_cost > cost:
            comment = f"Perhaps there is a filtering cost of {(expected_cost - cost) / tuples} being applied per tuple. There are {tuples} tuple(s)"
        return (cost, explanation, comment)
    else:
        if expected_cost == 0:
            return (0, "Result usually has additional no cost associated with it.")
        cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
        explanation = f"Result incurs cpu_tuple_cost ({cpu_tuple_cost}) per tuple. There are {tuples} tuple(s)"
        cost = truncate_cost(cpu_tuple_cost * tuples)
        if expected_cost != cost:
            comment = f"Perhaps the cost per tuple here is {(expected_cost - cost) / tuples} instead."
        return (cost, explanation, comment)
import math

def explain_sort(node: dict) -> str:
    tuples = node["Plan Rows"]
    width = node["Plan Width"]
    input_startup_cost = node['Startup Cost']  # Startup cost from PostgreSQL
    
    # Handle the 'Workers' key safely
    num_workers = node["Workers"] if "Workers" in node else 1

    sort_mem_kb = float(cache.get_setting("work_mem"))
    sort_mem_bytes = sort_mem_kb * 1024
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    block_size = float(cache.get_setting("block_size"))

    input_bytes = tuples * width
    num_pages_per_worker = math.ceil(input_bytes / block_size / num_workers)

    # Calculate run cost
    run_cost_per_worker = cpu_operator_cost * tuples
    total_run_cost = run_cost_per_worker * num_workers

    # Calculate total cost using the input startup cost and the computed run cost
    total_cost = input_startup_cost + total_run_cost

    explanation = f"Sort operation on column(s) {node['Sort Key']}, executed in parallel across {num_workers} workers. " \
                  f"Input tuples per worker: {tuples / num_workers}, Total tuples: {tuples}, Tuple width: {width} bytes, " \
                  f"Memory per worker: {sort_mem_bytes / 1024 / 1024} MB, Pages per worker: {num_pages_per_worker}, " \
                  f"Startup Cost: {input_startup_cost:.2f}, " \
                  f"Run cost per worker: {run_cost_per_worker:.2f}, Total run cost: {total_run_cost:.2f}, " \
                  f"Total cost: {total_cost:.2f}."
    return total_cost, explanation


#TODO: can make sum of children more precise (check if bitmap and/or/index for these functions)
def explain_bitmap_or(node: dict) -> str:
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    cost = sum([child["Total Cost"] for child in node.get("Plans", [])])
    tuples = node["Plan Rows"]

    num_index = 0
    num_cost = 0
    first = True
    for child in node["Plans"]:
        if child["Node Type"] == "Bitmap Index Scan":
            cost += tuples * cpu_operator_cost * 0.1
            num_index += 1
        elif not first:
            cost += 100 * cpu_operator_cost
            num_cost += 1
        if first:
            first = False

    explanation = f"The startup and total costs for the BitmapOr operator are the same\n"
    explanation += f"The total cost consists of the sum of all the child operators\n"
    explanation += f"For each child index scan ({num_index} in this case), an overhead of 0.1 * cpu_operator_cost ({cpu_operator_cost}) * tuples ({tuples}) is added.\n"
    explanation += f"The bitwise operation cost is 100 * cpu_operator_cost ({cpu_operator_cost}). It is applied for all non-index children except the first one ({num_cost})\n"
    explanation += f"The total comes out to be {cost:.2f}"
    
    return (cost, explanation)

def explain_bitmap_and(node: dict) -> str:
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    cost = sum([child["Total Cost"] for child in node.get("Plans", [])])
    tuples = node["Plan Rows"]

    num_index = 0
    for child in node["Plans"]:
        if child["Node Type"] == "Bitmap Index Scan":
            cost += tuples * cpu_operator_cost * 0.1
            num_index += 1
    
    cost += (len(node["Plans"]) - 1) * 100 * cpu_operator_cost
    explanation = f"The startup and total costs for the BitmapOr operator are the same\n"
    explanation += f"The total cost consists of the sum of all the child operators\n"
    explanation += f"For each child index scan ({num_index} in this case), an overhead of 0.1 * cpu_operator_cost ({cpu_operator_cost}) * tuples ({tuples}) is added.\n"
    explanation += f"The bitwise operation cost is 100 * cpu_operator_cost ({cpu_operator_cost}). It is applied for all children except the first ({len(node['Plans']) - 1})\n"
    explanation += f"The total comes out to be {cost:.2f}"
    
    return (cost, explanation)

def explain_group(node: dict) -> str:
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    numGroupCols = len(node["Group Key"])
    input_tuples = node["Plans"][0]["Plan Rows"]
    total_cost = cpu_operator_cost * input_tuples * numGroupCols

    explanation = f"There is no additional startup cost for the Group operator\n"
    explanation = f"cpu_operator_cost ({cpu_operator_cost}) is incurred for every input tuple ({input_tuples}) and grouping clause ({numGroupCols}) combination\n"

    if "Filter" in node:
        pattern = r"\([^()]*\)"
        filters = len(re.findall(pattern, node["Filter"]))
        cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
        filter_cost = filters * cpu_operator_cost
        explanation += f"CPU cost per output tuple is increased by {filter_cost:.4f} for {filters} filters * cpu_operator_cost ({cpu_operator_cost})\n"
        total_cost += filter_cost * node["Plan Rows"]

    explanation += f"The additional cost comes out to be {total_cost:.2f}"

    return (total_cost + node["Plans"][0]["Total Cost"], explanation)

#TODO find a query that uses lockrows
def explain_lockrows(node: dict) -> str:
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    tuples = node["Plans"][0]["Plan Rows"]
    total_cost = cpu_tuple_cost * tuples
    explanation = f"There is no additional startup cost for the LockRows operator\n"
    explanation += f"Lock rows incurs cpu_tuple_cost ({cpu_tuple_cost}) for each input row ({tuples})\n"
    explanation += f"This adds {total_cost}"
    return (total_cost + node["Plans"][0]["Total Cost"], explanation)

def clean_output(output: list) -> list:
    pattern = '^\(?-?\d+(\.\d+)?\)?$'
    for x in reversed(range(0,len(output))):
        if re.match(pattern, output[x]) is not None:
            del output[x]
    return output

def explain_setop(node: dict) -> str:
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    subpath_cost = node["Plans"][0]["Total Cost"]
    explanation = f"There is no additional startup cost for the SetOp operator\n"
    tuples = node["Plans"][0]["Plan Rows"]
    total_cost = cpu_operator_cost * tuples
    len_distinct = len(clean_output(node["Output"]))
    total_cost *= len_distinct
    explanation += f"cpu_operator_cost ({cpu_operator_cost}) is incurred for every row ({tuples}) and distinct column ({len_distinct}) combination\n"
    explanation += f"This adds a cost of {total_cost}"
    return (total_cost + subpath_cost, explanation)

#TODO: check if filtering needed
def explain_subqueryscan(node: dict) -> str:
    explanation = f"There is no additional startup cost for the Subquery Scan operator\n"
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    subpath_cost = node["Plans"][0]["Total Cost"]
    tuples = node["Plans"][0]["Plan Rows"]
    total_cost = cpu_tuple_cost * tuples
    explanation += f"cpu_tuple_cost ({cpu_tuple_cost}) is incurred for every input row ({tuples})\n"
    explanation += f"This adds a cost of {total_cost}"
    return (total_cost + subpath_cost, explanation)

#TODO: check if filtering needed
def explain_valuescan(node: dict) -> str:
    explanation = f"There is no startup cost for the Subquery Scan operator\n"
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    tuples = node["Plan Rows"]
    total_cost = (cpu_operator_cost + cpu_tuple_cost) * tuples
    explanation += f"cpu_tuple_cost ({cpu_tuple_cost}) and cpu_operator_cost ({cpu_operator_cost}) is incurred for every input row ({tuples})\n"
    explanation += f"The total cost is {total_cost}"
    return (total_cost, explanation)

def explain_modify_table(node: dict) -> str:
    explanation = f"There is no cost associated with this node."
    return (node["Plans"][0]["Total Cost"], explanation)

def explain_project_set(node: dict) -> str:
    explanation = f"There is no startup cost for the ProjectSet operator\n"
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    subpath_cost = node["Plans"][0]["Total Cost"]
    subpath_tuples = node["Plans"][0]["Plan Rows"]
    tuples = node["Plan Rows"]

    run_cost = cpu_tuple_cost * subpath_tuples
    run_cost += (tuples - subpath_tuples) * cpu_tuple_cost / 2

    explanation += f"cpu_tuple_cost ({cpu_tuple_cost}) is incurred for every input row {subpath_tuples}\n"
    explanation += f"half of cpu_tuple_cost ({cpu_tuple_cost/2}) is incurred for every added output row {tuples - subpath_tuples}\n"
    explanation += f"This adds {run_cost} to the total cost"
    return (run_cost + subpath_cost, explanation)

def explain_recursive_union(node: dict) -> str:
    explanation = f"There is no startup cost for the Recursive Union operator\n"
    nterm = node["Plans"][0]
    rterm = node["Plans"][1]

    total_cost = nterm["Total Cost"]
    explanation += f"The initial cost is the same as the non-recursive term ({total_cost})\n"

    total_cost += 10 * rterm["Total Cost"]
    explanation += f"Assuming 10 recursions, the cost of the recursive term ({rterm['Total Cost']}) is added 10 times\n"

    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    total_rows = node["Plan Rows"]
    total_cost += cpu_tuple_cost * total_rows
    explanation += f"Finally, cpu_tuple_cost ({cpu_tuple_cost}) is incurred for every output row ({total_rows})"

    return (total_cost, explanation)

def explain_hash(node: dict) -> str:
    explanation = f"The hash operator incurs no additional cost as it builds a hash table in memory.\n"
    return (node["Plans"][0]["Total Cost"], explanation)

def explain_aggregate(node: dict) -> str: #not done, no of workers not accounted
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    child = node["Plans"][0]
    child_cost = child["Total Cost"]
    child_tuples = child["Plan Rows"]
    strategy = node["Strategy"]
    total_cost = child_cost
    if strategy == "Plain":
        explanation = f"In Plain Aggregation, there is no grouping involved.\n"
        explanation += f"The startup cost is the total cost of the child node.\n"
        explanation += f"cpu_tuple_cost ({cpu_tuple_cost} is added to the the total cost.\n"
        return (total_cost + cpu_tuple_cost, explanation)
    elif strategy == "Sorted" or strategy == "Mixed":
        numGroupCols = len(clean_output(node["Output"]))
        total_cost += cpu_operator_cost * numGroupCols * child_tuples
        total_cost += cpu_tuple_cost * node["Plan Rows"]
        explanation = f"When sorting, no additional startup cost is incurred.\n"
        explanation += f"cpu_operator_cost ({cpu_operator_cost}) * length of grouping keys ({numGroupCols}) is charged for each input tuple\n"
        explanation += f"cpu_tuple_cost ({cpu_tuple_cost}) is charged for each output tuple"
        
        if strategy == "Sorted":
            return (total_cost, explanation)
    else:
        total_cost += cpu_operator_cost * numGroupCols * child_tuples
        total_cost += cpu_tuple_cost * node["Plan Rows"]
        explanation = f"When hashing, cpu_operator_cost ({cpu_operator_cost}) * length of grouping keys ({numGroupCols}) is charged for each input tuple during startup. This represents the cost of the hash computation.\n"
        explanation += f"cpu_tuple_cost ({cpu_tuple_cost}) is charged for each output tuple"

    child = node["Plans"][0]
    tuples = node["Plan Rows"]
    startup_cost = child["Total Cost"] + (cpu_operator_cost * child["Plan Rows"])
    explanation = f"The Aggregate's startup cost consists of cost of child operator and \
    the cpu_operator_cost ({cpu_operator_cost}) multipled by number of input rows\n"
    explanation += f"The Aggregate's total cost is then increased by cpu_tuple_cost ({cpu_tuple_cost}), \
    for processing every resulting output row"
    total_cost = startup_cost + cpu_tuple_cost * tuples
    return (total_cost, explanation)

def explain_nestedloop(node: dict) -> str: #total cost is wrong
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    outer_path = node["Plans"][0]
    inner_path = node["Plans"][1]
    outer_path_rows = outer_path["Plan Rows"]
    inner_path_rows = inner_path["Plan Rows"]

    # Calculate the startup and run costs
    startup_cost = 0
    run_cost = 0

    # Startup cost includes the startup cost of both outer and inner paths
    startup_cost += outer_path["Startup Cost"] + inner_path["Startup Cost"]

    # Run cost includes the total cost of the outer path excluding startup cost
    run_cost += outer_path["Total Cost"] - outer_path["Startup Cost"]

    # Calculate the run cost of the inner relation excluding startup cost
    inner_run_cost = inner_path["Total Cost"] - inner_path["Startup Cost"]
    run_cost += inner_run_cost

    run_cost += cpu_tuple_cost * outer_path_rows * inner_path_rows
    run_cost += cpu_tuple_cost * node["Plan Rows"]

    # Prepare explanation
    explanation = f"The startup and total costs for the Nested Loop join are as follows:\n"
    explanation += f"Startup cost: {startup_cost}\n"
    explanation += f"Total cost: {startup_cost + run_cost}\n"
    explanation += f"- Startup cost includes the startup cost of both outer and inner paths\n"
    explanation += f"- Total cost includes the total cost of the outer path excluding startup cost\n"
    explanation += f"- Total cost also includes the cpu_tuple_cost ({cpu_tuple_cost}) of matching each tuples between outer and inner tuples\n"
    explanation += f"- Total cost also includes the cpu_tuple_cost ({cpu_tuple_cost}) of processing every resulting output row"
    return startup_cost + run_cost, explanation


def explain_hash_join(node: dict) -> str:
    # Configuration settings
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    seq_page_cost = float(cache.get_setting("seq_page_cost"))
    work_mem = float(cache.get_setting("work_mem")) * 1024 * 1024  # Convert MB to bytes
    block_size = float(cache.get_setting("block_size"))

    # Extracting plan information
    outer_path = node["Plans"][0]
    inner_path = node["Plans"][1]
    outer_path_rows = outer_path["Plan Rows"]
    inner_path_rows = inner_path["Plan Rows"]

    # Default tuple sizes (or retrieve from settings)
    inner_tuple_size = inner_path.get("Plan Width", 50)
    outer_tuple_size = outer_path.get("Plan Width", 50)

    # Memory and batch calculations
    total_inner_size = inner_path_rows * inner_tuple_size
    inner_pages = total_inner_size / block_size
    numbatches = max(1, total_inner_size / work_mem)

    # Hash function computation costs
    hash_cost_per_tuple = cpu_operator_cost * 2
    total_hash_cost = hash_cost_per_tuple * (inner_path_rows + outer_path_rows)

    # I/O cost for batches
    if numbatches > 1:
        io_cost = (seq_page_cost * inner_pages * 2) * numbatches  # Read/write each batch
        total_hash_cost += io_cost

    # Run cost adjusted for tuple comparisons and hash evaluations
    run_cost = total_hash_cost + (cpu_tuple_cost * outer_path_rows * (inner_path_rows / numbatches))

    # Startup cost directly from the node
    startup_cost = node.get('Startup Cost', 1.56)

    # Total cost calculation
    total_cost = startup_cost + run_cost

    # Constructing the explanation
    explanation = f"Hash Join Cost Estimation:\n" \
                  f"Join Type: Inner\n" \
                  f"Outer Rows: {outer_path_rows}, Inner Rows: {inner_path_rows}\n" \
                  f"Startup Cost: {startup_cost:.2f}\n" \
                  f"Run Cost: {run_cost:.2f}, Number of Batches: {numbatches}\n" \
                  f"Total Cost: {total_cost:.2f}\n"

    return total_cost, explanation



# def explain_hash_join(node: dict) -> str:
#     cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
#     cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
#     seq_page_cost = float(cache.get_setting("seq_page_cost"))
#     block_size = float(cache.get_setting("block_size"))
    
#     outer_path = node["Plans"][0]
#     inner_path = node["Plans"][1]
#     outer_path_rows = outer_path["Plan Rows"]
#     inner_path_rows = inner_path["Plan Rows"]

#     # Assuming startup cost is given:
#     input_startup_cost = node['Startup Cost']

#     # Calculate run cost:
#     run_cost = 0

#     # Compute cost of hashing all tuples in the inner relation
#     # Assuming one cpu_operator_cost per tuple for hashing
#     run_cost += (cpu_operator_cost + cpu_tuple_cost) * inner_path_rows

#     # Compute cost of processing each outer tuple
#     # Assuming a match needs to check each outer tuple against hash table entries
#     run_cost += cpu_operator_cost * outer_path_rows

#     # If batching is necessary due to memory constraints
#     estimated_inner_size = inner_path_rows * inner_path.get("Plan Width", 100)  # Default width
#     inner_pages = estimated_inner_size / block_size

#     work_mem = float(cache.get_setting("work_mem")) * 1024 * 1024  # Convert MB to bytes
#     if estimated_inner_size > work_mem:
#         # If more than one batch is needed, add cost of writing/reading batches to/from disk
#         numbatches = max(1, estimated_inner_size / work_mem)
#         run_cost += seq_page_cost * inner_pages * numbatches

#     total_cost = input_startup_cost + run_cost

#     # Prepare the explanation
#     explanation = f"Hash Join Cost Estimation:\n"
#     explanation += f"Outer Rows: {outer_path_rows}, Inner Rows: {inner_path_rows}\n"
#     explanation += f"Startup Cost: {input_startup_cost:.2f}\n"
#     explanation += f"Run Cost: {run_cost:.2f}\n"
#     explanation += f"Total Cost: {total_cost:.2f}\n"
#     explanation += f"- Run cost includes the cost of hashing inner tuples and checking outer tuples against hash table\n"
#     explanation += f"- Additional cost for handling batches if memory constraints require disk-based operations\n"

#     return total_cost, explanation

def explain_unique(node: dict):
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    subpath = node["Plans"][0]
    sub_cost = subpath["Total Cost"]
    input_tuples = subpath["Plan Rows"]
    numCols = len(clean_output(node["Output"]))
    total_cost = input_tuples * numCols * cpu_operator_cost
    explanation = f"There is no additional startup for the Unique operator.\n"
    explanation += f"We charge cpu_operator_cost ({cpu_operator_cost}) * number of columns ({numCols}) for each input tuple ({input_tuples}).\n"
    explanation += f"This adds {total_cost} to the total cost."

    return (sub_cost + total_cost, explanation)

def explain_cte(node : dict):
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost")) 
    cost_per_tuple = cpu_tuple_cost * 2
    explanation = f"WorkTable/CTE Scan charges 2 * cpu_tuple_cost ({cpu_tuple_cost}) per tuple.\n"
    base_cost = 0
    rows = node["Plan Rows"]
    comment = ""
    if "Plans" in node:
        rows = node["Plans"][0]["Plan Rows"]
        base_cost = node["Plans"][0]["Total Cost"]
    if "Filter" in node:
        pattern = r"\([^()]*\)"
        filters = len(re.findall(pattern, node["Filter"]))
        cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
        filter_cost = filters * cpu_operator_cost
        cost_per_tuple += filter_cost
        explanation += f"CPU cost per tuple is increased by {filter_cost:.4f} for {filters} filters * cpu_operator_cost ({cpu_operator_cost})\n"

    total_cost = rows * cost_per_tuple
    explanation += f"The cost per tuple is {cost_per_tuple}. For {rows} row(s), the total cost is {total_cost}"

    expected_cost = node["Total Cost"]
    if truncate_cost(base_cost + total_cost) != expected_cost:
        comment = "The given plan row count by PostgresSQL is not the same as the input tuples.\n"
        comment += "We cannot get the actual tuple count for a WorkTable/CTE\n"
        comment += f"If the estimated cost per tuple is right, there should have been {int(expected_cost/cost_per_tuple)} input tuples."
    
    return (base_cost + total_cost, explanation, comment)

def explain_xyz_scan(node: dict, fn_name:str):
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost")) 
    tuples = node["Plan Rows"]
    total_cost = tuples * cpu_tuple_cost
    startup_cost = node["Startup Cost"]
    expected_cost = node["Total Cost"] - startup_cost
    comment = ""

    explanation = f"{fn_name} charges cpu_tuple_cost ({cpu_tuple_cost}) per tuple ({tuples})."

    if truncate_cost(total_cost) != expected_cost:
        comment = f"{fn_name} scan may involve other costs to evaluate expressesions.\n"
        comment += f"The cost per tuple should have been {expected_cost/tuples}"
    
    return (startup_cost + total_cost, explanation, comment)

def explain_functionscan(node: dict):
    return explain_xyz_scan(node, "Function Scan")

def explain_tablefunctionscan(node: dict):
    return explain_xyz_scan(node, "Table Function Scan")

def explain_namedtuplestorescan(node: dict):
    return explain_xyz_scan(node, "Named Tuplestore Scan")

fn_dict = {
    "Result": explain_result,
    "ProjectSet": explain_project_set,
    "ModifyTable": explain_modify_table,
    "Append": explain_append,
    "Merge Append": explain_merge_append,
    "Recursive Union": explain_recursive_union,
    "BitmapAnd": explain_bitmap_and,
    "BitmapOr": explain_bitmap_or,
    "Nested Loop": explain_nestedloop,
    "Merge Join": None,
    "Hash Join": explain_hash_join,
    "Seq Scan": explain_seqscan,
    "Sample Scan": None,
    "Gather": explain_gather,
    "Gather Merge": explain_gather_merge,
    "Index Scan": None,
    "Index Only Scan": None,
    "Bitmap Index Scan": None,
    "Bitmap Heap Scan": None,
    "Tid Scan": None,
    "Tid Range Scan": None,
    "Subquery Scan": explain_subqueryscan,
    "Function Scan": explain_functionscan,
    "Table Function Scan": explain_tablefunctionscan,
    "Values Scan": explain_valuescan,
    "CTE Scan": explain_cte,
    "Named Tuplestore Scan": explain_namedtuplestorescan,
    "WorkTable Scan": explain_cte,
    "Foreign Scan": None,
    "Custom Scan": None,
    "Materialize": explain_materialize,
    "Memoize": None,
    "Sort": explain_sort,
    "Incremental Sort": None,
    "Group": explain_group,
    "Aggregate": explain_aggregate,
    "WindowAgg": None,
    "Unique": explain_unique,
    "SetOp": explain_setop,
    "LockRows": explain_lockrows,
    "Limit": explain_limit,
    "Hash": explain_hash,
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
        if "Total Cost" not in node:
            return "<b>Comment</b>: No cost associated with this node."

        node_type = node["Node Type"] 
        if node_type in fn_dict and fn_dict[node_type] is not None:
            try:
                expected_cost = node["Total Cost"]
                params = fn_dict[node_type](node)
                full_cost = params[0]
                cost = truncate_cost(full_cost)
                explanation = params[1]
                color = "c0ebcc" if cost == expected_cost else "ebc6c0"
                explanation = f"<b>Calculated Cost</b>: <span style=\"background-color:#{color};\">{cost}</span>\n<b>Explanation</b>: {explanation}"
                if len(params) > 2 and len(params[2]) > 0:
                    explanation += f"\n<b>Comments</b>: {params[2]}"
                elif cost != expected_cost:
                    diff = truncate(abs(full_cost - expected_cost))
                    if diff <= 0.05:
                        explanation += f"\n<b>Comments</b>: Calculated cost is off by {diff:.4f}, which is most likely a rounding error."
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
            cur.execute(f"EXPLAIN (COSTS, VERBOSE, FORMAT JSON) {query}")
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
