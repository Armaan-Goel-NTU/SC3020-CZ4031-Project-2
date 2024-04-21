from collections import deque
import psycopg
import re
import math
import decimal
import json

# used to round calculations. same as Postgres - 2dp
def truncate_cost(a : float) -> float:
    return round(a, 2)

# truncates to 4dp, used to show precise differences in calculations
def truncate(a : float) -> float:
    return float(decimal.Decimal(str(a)).quantize(decimal.Decimal('.0001'), rounding=decimal.ROUND_DOWN))

# we cache all the variables we get from Postgres
# this cache lasts for each login session (cleared when disconnected)
cache = None
class Cache():
    # use a simple dictionary to keep track
    # cursor is provided to execute queries
    def __init__(self, cur: psycopg.Cursor) -> None:
        self.dict = {}
        self.cur = cur
    
    # for settings such as cpu_tuple_cost, seq_page_cost etc.
    def query_setting(self, setting: str) -> str:
        self.cur.execute(f"SELECT setting FROM pg_settings WHERE name = '{setting}'")
        return self.cur.fetchall()[0][0]

    # gets the number of pages for a base relation
    def query_pagecount(self, relation: str) -> int:
        self.cur.execute(f"SELECT relpages FROM pg_class WHERE relname = '{relation}'")
        return self.cur.fetchall()[0][0]

    # gets the number of tuples for a base relation
    def query_tuplecount(self, relation: str) -> int:
        self.cur.execute(f"SELECT reltuples FROM pg_class where relname = '{relation}'")
        return self.cur.fetchall()[0][0]
    
    # all explanation functions will call this
    def get_setting(self, setting: str) -> str:
        key = f"setting/{setting}"
        # query only if not present currently and save output
        if key not in self.dict:
            self.log_cb(f"Querying {key}")
            self.dict[key] = self.query_setting(setting)
        return self.dict[key]

    # all explanation functions will call this
    def get_page_count(self, relation: str) -> int:
        key = f"relpages/{relation}"
        # query only if not present currently and save output
        if key not in self.dict:
            self.log_cb(f"Querying {key}")
            self.dict[key] = self.query_pagecount(relation)
        return self.dict[key]        

    # all explanation functions will call this
    def get_tuple_count(self, relation: str) -> int:
        key = f"reltuples/{relation}"    
        # query only if not present currently and save output
        if key not in self.dict:
            self.log_cb(f"Querying {key}")
            self.dict[key] = self.query_tuplecount(relation)
        return self.dict[key]
    
    # for many relations we may check to tuple count before generating explanations
    # this is to check if auto analysis has been done
    # more details where this is called
    def set_tuple_count(self, relation: str, count:int):
        key = f"reltuples/{relation}"
        self.dict[key] = count
    
    # log_cb passed from interface to here. Will inform whatever it is querying
    def set_log_cb(self, log_cb: callable):
        self.log_cb = log_cb

# count the number of clauses in a filter or similar condition
# we assume clauses can only be connected by OR and AND (should be okay?)
def count_clauses(condition: str) -> int:
    or_count = condition.count(') OR (')
    and_count = condition.count(') AND (')
    total_count = or_count + and_count + 1 
    
    return total_count

# explanation function for Sequential Scan
# adapted from cost_seqscan in src/backend/optimizer/path/costsize.c
def explain_seqscan(node: dict) -> tuple[float, str, str]:
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
        filters = count_clauses(node["Filter"])
        cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
        filter_cost = filters * cpu_operator_cost
        explanation += f"CPU cost per tuple is increased by {filter_cost:.4f} for {filters} filters * cpu_operator_cost ({cpu_operator_cost})\n"

    cpu_cost = (cpu_tuple_cost + filter_cost) * row_count
    
    # account for parallelisation
    # code adapted from get_parallel_divisor in src/backend/optimizer/path/costsize.c
    if node["Parallel Aware"] and "Workers Planned" in node:
        workers = node["Workers Planned"]
        if cache.get_setting("parallel_leader_participation") == "on" and workers < 4:
            workers += 1 - (workers * 0.3)
        explanation += f"The total CPU cost is reduced by a parallelization factor of {workers:.1f}\n" 

    disk_cost = seq_page_cost * page_count
    cost = truncate_cost(cpu_cost / workers + disk_cost)

    # reverse all the calculations to get the expected filtering cost
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

# taken from src/include/c.h
def align(val : int, len : int) -> int:
    return val + (len - (val % len))

# explanation function for Materialize
# adapted from cost_material in src/backend/optimizer/path/costsize.c
def explain_materialize(node: dict) -> tuple[float, str]:
    tuples = node["Plan Rows"]
    width = node["Plan Width"]
    child = node["Plans"][0]
    startup_cost = child["Startup Cost"]

    explanation = f"Materialize has the same startup cost as its child. ({startup_cost})\n"
    work_mem_bytes = float(cache.get_setting("work_mem")) * 1024
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    seq_page_cost = float(cache.get_setting("seq_page_cost"))

    # 23 is the size of the HeapTupleHeader
    nbytes = tuples * (align(width, 8) + align(23, 8))
    explanation += f"Materialize charges 2 * cpu_operator_cost ({cpu_operator_cost}) per tuple as overhead. There are {tuples} tuple(s)\n"
    cost = 2 * cpu_operator_cost * tuples
    block_size = float(cache.get_setting("block_size"))
    
    # spill cost
    if nbytes > work_mem_bytes:
        explanation += f"The relation to materialize is larger that working memory space of {work_mem_bytes / 1024}KB\n"
        npages = math.ceil(nbytes/block_size)
        explanation += f"Disk costs will be incurred. The projected amount to materialize is {nbytes}, which will take {npages} to fit with a page size of {block_size}\n"
        f"seq_page_cost ({seq_page_cost}) will be incurred for each page.\n"
        cost += npages
    
    explanation += f"Additional cost incurred is {cost:.2f}"
    return (cost + child['Total Cost'], explanation)

# explanation function for Merge Append
# adapted from cost_merge_append in src/backend/optimizer/path/costsize.c
def explain_merge_append(node: dict) -> tuple[float, str]:
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

# explanation function for Append
# adapted form cost_append in src/backend/optimizer/path/costsize.c
def explain_append(node: dict) -> tuple[float, str]:
    # gather child costs
    comment = ""
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    child_costs = [child["Total Cost"] for child in node.get("Plans", [])]
    child_startup_costs = [child["Startup Cost"] for child in node.get("Plans", [])]
    row_count = node["Plan Rows"]

    # just in case
    if not child_costs: 
        return (0, "No child plans for this Append node.")

    min_startup_cost = min(child_startup_costs)
    total_cost = sum(child_costs) + cpu_tuple_cost * row_count * 0.5  # 0.5 is APPEND_CPU_COST_MULTIPLIER

    #explanation:
    explanation = f"Append node combines several plans. The startup cost is the minimum of the startup costs of its children, and the total cost includes CPU costs associated with processing rows.\n"
    explanation += f"Child startup costs: {child_startup_costs}\n"
    explanation += f"Child total costs: {child_costs}\n"
    explanation += f"Estimated row count: {row_count}\n"
    explanation += f"Minimum startup cost: {min_startup_cost}, Adjusted total cost: {total_cost:.2f} (including CPU cost for handling rows)"

    # append does not account for parallelisation as of now
    expected_cost = node["Total Cost"]
    if truncate_cost(total_cost) != expected_cost and node["Parallel Aware"]:
        comment = "Our calculation for the append does not consider the effect of parallelisation.\n"
        comment += "This involves the knowledge of partial and non-partial paths.\n"
        comment += f"The cost is likely the sum of any of the {len(node['Plans'])} child nodes + the cpu processing cost of {cpu_tuple_cost * row_count * 0.5}."

    return (total_cost, explanation, comment)

# explanation function for Gather
# adapted from cost_gather in src/backend/optimizer/path/costsize.c
def explain_gather(node: dict) -> tuple[float, str]:
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

# explanation function for Merge
# adapted from cost_gather_merge in src/backend/optimizer/path/costsize.c
def explain_gather_merge(node: dict) -> tuple[float, str]:
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
    
    # 1.05 is the penalty cost directly taken from the source code
    run_cost += parallel_tuple_cost * rows * 1.05
    explanation += f"A per-tuple heap maintaince cost of comparison_cost * log2(N) applied.\n"
    explanation += f"cpu_operator_cost ({cpu_operator_cost}) is also applied per tuple for heap management overhead\n"
    explanation += f"Lastly, parallel_tuple_cost ({parallel_tuple_cost}) and a 5% penalty to wait for every work is also incurred.\n"
    explanation += f"For {rows} tuples, the run cost is an additional {run_cost:.2f}"

    return (startup_cost + run_cost + subpath_total_cost, explanation)

# explanation function for Limit
# adapted from adjust_limit_rows_costs in src/backend/optimizer/util/pathnode.c
def explain_limit(node: dict) -> tuple[float, str]:
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

# explanation function for Index Scan
# attempted to adapt from cost_index in rc/backend/optimizer/path/costsize.c
def explain_indexscan(node: dict) -> tuple[float, str]:
    index_name = node.get("Index Name", "unknown index")
    index_page_count = cache.get_page_count(index_name)  
    relation_name = node["Relation Name"]
    table_page_count = cache.get_page_count(relation_name)  
    table_tuple_count = cache.get_tuple_count(relation_name)  
    seq_page_cost = float(cache.get_setting("seq_page_cost"))
    random_page_cost = float(cache.get_setting("random_page_cost"))
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    effective_cache_size = float(cache.get_setting("effective_cache_size"))

    # Selectivity estimation:
    index_selectivity = node["Plan Rows"] / table_tuple_count

    # Fetch the number of tuples expected to be fetched by the index
    tuples_fetched = index_selectivity * table_tuple_count

    # Applying Mackert and Lohman formula as given in source code
    pages_fetched = index_pages_fetched(tuples_fetched, table_page_count, index_page_count, effective_cache_size)

    # Calculate I/O cost
    if index_selectivity == 1:  
        if pages_fetched > 1:
            io_cost = random_page_cost + (pages_fetched - 1) * seq_page_cost
        else:
            io_cost = pages_fetched * seq_page_cost
    else:  
        io_cost = pages_fetched * random_page_cost

    
    total_cost = io_cost + tuples_fetched * cpu_tuple_cost

    explanation = (
        f"Index Scan on {index_name} over the table '{relation_name}' with an index selectivity of {index_selectivity:.4f} implies that "
        f"approximately {tuples_fetched:.0f} tuples ({100 * index_selectivity:.2f}%) out of {table_tuple_count} are fetched. "
        f"This results in fetching about {pages_fetched:.0f} pages due to the correlation and caching effects. "
        f"The I/O cost calculated as {io_cost:.2f} and CPU cost for processing these tuples is {tuples_fetched * cpu_tuple_cost:.2f}. "
        f"Thus, the total estimated cost of this index scan is {total_cost:.2f}."
    )

    return (total_cost, explanation)

# adapted from index_pages_fetched in rc/backend/optimizer/path/costsize.c
def index_pages_fetched(tuples_fetched, table_pages, index_pages, effective_cache_size) -> float:
    T = max(table_pages, 1)
    b = effective_cache_size  
    Ns = tuples_fetched
    if T <= b:   
        pages_fetched = min(2 * T * Ns / (2 * T + Ns), T)
    else:
        lim = 2 * T * b / (2 * T - b)
        if Ns <= lim:
            pages_fetched = 2 * T * Ns / (2 * T + Ns)
        else:
            pages_fetched = b + (Ns - lim) * (T - b) / T
    return math.ceil(pages_fetched)

# explanation function for Result
# adapted from cost_resultscan in src/backend/optimizer/path/costsize.c
def explain_result(node: dict) -> tuple[float, str, str]:
    expected_cost = node["Total Cost"]
    tuples = node["Plan Rows"]
    comment = ""
    # cost is usually the same as the child
    if "Plans" in node:
        cost = node["Plans"][0]["Total Cost"]
        explanation = "Result usually has additional no cost associated with it."
        
        # in case there are filters or functions
        if expected_cost > cost:
            comment = f"Perhaps there is a filtering cost of {(expected_cost - cost) / tuples} being applied per tuple. There are {tuples} tuple(s)"
        return (cost, explanation, comment)
    else:
        # dummy case (plain SELECT)
        if expected_cost == 0:
            return (0, "Result usually has additional no cost associated with it.")
        cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
        explanation = f"Result incurs cpu_tuple_cost ({cpu_tuple_cost}) per tuple. There are {tuples} tuple(s)"
        cost = truncate_cost(cpu_tuple_cost * tuples)

        # in case there are filters or functions
        if expected_cost != cost:
            comment = f"Perhaps the cost per tuple here is {(expected_cost - cost) / tuples} instead."
        return (cost, explanation, comment)

# explanation function for Sort
# adapted from cost_sort in src/backend/optimizer/path/costsize.c
def explain_sort(node: dict) -> tuple[float, str]:
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


# for BitmapOr and BitmapAnd, the child bitmap costs are adapted from cost_bitmap_tree_node in src/backend/optimizer/path/costsize.c

# explanation function for BitmapOr
# adapted from cost_bitmap_or_node in src/backend/optimizer/path/costsize.c
def explain_bitmap_or(node: dict) -> tuple[float, str]:
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    cost = sum([child["Total Cost"] for child in node.get("Plans", [])])
    tuples = node["Plan Rows"]

    num_index = 0
    num_cost = 0
    first = True

    # children can be bitmap index scan, bitmapAnd, bitmapOr
    # they are treated with slight differences
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

# explanation function for BitmapAnd
# adapted from cost_bitmap_and_node in src/backend/optimizer/path/costsize.c
def explain_bitmap_and(node: dict) -> tuple[float, str]:
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    cost = sum([child["Total Cost"] for child in node.get("Plans", [])])
    tuples = node["Plan Rows"]

    num_index = 0
    # more cost is added for bitmap index children
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

# explanation function for Group
# adapted from cost_group in src/backend/optimizer/path/costsize.c
def explain_group(node: dict) -> tuple[float, str]:
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    numGroupCols = len(node["Group Key"])
    input_tuples = node["Plans"][0]["Plan Rows"]
    total_cost = cpu_operator_cost * input_tuples * numGroupCols

    explanation = f"There is no additional startup cost for the Group operator\n"
    explanation = f"cpu_operator_cost ({cpu_operator_cost}) is incurred for every input tuple ({input_tuples}) and grouping clause ({numGroupCols}) combination\n"

    # HAVING clauses. paid per output row
    if "Filter" in node:
        filters = count_clauses(node["Filter"])
        cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
        filter_cost = filters * cpu_operator_cost
        explanation += f"CPU cost per output tuple is increased by {filter_cost:.4f} for {filters} filters * cpu_operator_cost ({cpu_operator_cost})\n"
        total_cost += filter_cost * node["Plan Rows"]

    explanation += f"The additional cost comes out to be {total_cost:.2f}"

    child_cost = node["Plans"][0]["Total Cost"]

        # Best we can do given both the filtering cost could be wrong or the number of output tuples is unreliable
    expected_cost = node["Total Cost"] - child_cost
    comment = ""
    if truncate_cost(total_cost) != expected_cost:
        comment = "The Group operator may involve HAVING clauses with differnet cost functions.\n"
        comment += "The output rows are reduced with each clause, costs may have been applied to a larger number of rows than returned."

    return (total_cost + child_cost, explanation, comment)

# explanation function for Lockrows
# adapted from create_lockrows_path in src/backend/optimizer/util/pathnode.c
def explain_lockrows(node: dict) -> tuple[float, str]:
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    tuples = node["Plans"][0]["Plan Rows"]
    total_cost = cpu_tuple_cost * tuples
    explanation = f"There is no additional startup cost for the LockRows operator\n"
    explanation += f"Lock rows incurs cpu_tuple_cost ({cpu_tuple_cost}) for each input row ({tuples})\n"
    explanation += f"This adds {total_cost}"
    return (total_cost + node["Plans"][0]["Total Cost"], explanation)

# removes numbers or (numbers) from the output list of a node
def clean_output(output: list) -> list:
    pattern = '^\(?-?\d+(\.\d+)?\)?$'
    for x in reversed(range(0,len(output))):
        if re.match(pattern, output[x]) is not None:
            del output[x]
    return output

# explanation function for SetOp
# adapted from create_setop_path in src/backend/optimizer/util/pathnode.c
def explain_setop(node: dict) -> tuple[float, str]:
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    subpath_cost = node["Plans"][0]["Total Cost"]
    explanation = f"There is no additional startup cost for the SetOp operator\n"
    tuples = node["Plans"][0]["Plan Rows"]
    total_cost = cpu_operator_cost * tuples
    len_distinct = len(clean_output(node["Output"]))
    total_cost *= len_distinct
    explanation += f"cpu_operator_cost ({cpu_operator_cost}) is incurred for every row ({tuples}) and distinct column ({len_distinct}) combination\n"
    explanation += f"This adds a cost of {total_cost}"

    # this is likely the issue given no filtering/functions are involved.
    comment = ""
    expected_cost = node["Total Cost"] - subpath_cost
    if truncate_cost(total_cost) != expected_cost:
        comment = "The number of distinct columns may have been calculated incorrectly.\n"
        comment += f"It should have been {round(expected_cost/(cpu_operator_cost * tuples))}"

    return (total_cost + subpath_cost, explanation, comment)

# explanation function for Subquery Scan
# adapted from cost_subqueryscan in src/backend/optimizer/path/costsize.c
def explain_subqueryscan(node: dict) -> tuple[float, str]:
    explanation = f"There is no additional startup cost for the Subquery Scan operator\n"
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    subpath_cost = node["Plans"][0]["Total Cost"]
    tuples = node["Plans"][0]["Plan Rows"]
    total_cost = cpu_tuple_cost * tuples
    explanation += f"cpu_tuple_cost ({cpu_tuple_cost}) is incurred for every input row ({tuples})\n"
    explanation += f"This adds a cost of {total_cost}"

    expected_cost = node["Total Cost"] - subpath_cost

    # filtering 
    comment = ""
    if truncate_cost(total_cost) != expected_cost:
        comment = "There may have been additional costs (filtering/functions) that may not have been accounted for.\n"
        comment += f"An additional cost of {truncate((expected_cost/tuples) - cpu_tuple_cost)} should have been applied per tuple."
    return (total_cost + subpath_cost, explanation, comment)

# explanation function for Value Scan
# adapted from cost_valuesscan in src/backend/optimizer/path/costsize.c
def explain_valuescan(node: dict) -> tuple[float, str]:
    explanation = f"There is no startup cost for the Subquery Scan operator\n"
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    tuples = node["Plan Rows"]
    total_cost = (cpu_operator_cost + cpu_tuple_cost) * tuples
    explanation += f"cpu_tuple_cost ({cpu_tuple_cost}) and cpu_operator_cost ({cpu_operator_cost}) is incurred for every input row ({tuples})\n"
    explanation += f"The total cost is {total_cost}"

    expected_cost = node["Total Cost"]

    # filtering again
    comment = ""
    if truncate_cost(total_cost) != expected_cost:
        comment = "There may have been additional costs (filtering/functions) that may not have been accounted for.\n"
        comment += f"An additional cost of {truncate((expected_cost/tuples) - (cpu_tuple_cost + cpu_operator_cost))} should have been applied per tuple."
    return (total_cost, explanation)

# explanation function for Modify Table
# adapted from create_modifytable_path in src/backend/optimizer/util/pathnode.c
def explain_modify_table(node: dict) -> tuple[float, str]:
    explanation = f"There is no cost associated with this node."
    return (node["Plans"][0]["Total Cost"], explanation)

# explanation function for ProjectSet
# adapted from create_set_projection_path in src/backend/optimizer/util/pathnode.c
def explain_project_set(node: dict) -> tuple[float, str]:
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

# explanation function for Recursive Union
# adapted from cost_recursive_union in src/backend/optimizer/path/costsize.c
def explain_recursive_union(node: dict) -> tuple[float, str]:
    explanation = f"There is no startup cost for the Recursive Union operator\n"
    nterm = node["Plans"][0]
    rterm = node["Plans"][1]

    total_cost = nterm["Total Cost"]
    explanation += f"The initial cost is the same as the non-recursive term ({total_cost})\n"

    # 10 is the assumed max recursion depth (or the number of loops to be executed)
    total_cost += 10 * rterm["Total Cost"]
    explanation += f"Assuming 10 recursions, the cost of the recursive term ({rterm['Total Cost']}) is added 10 times\n"

    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    total_rows = node["Plan Rows"]
    total_cost += cpu_tuple_cost * total_rows
    explanation += f"Finally, cpu_tuple_cost ({cpu_tuple_cost}) is incurred for every output row ({total_rows})"

    return (total_cost, explanation)

# explanation function for Hash
def explain_hash(node: dict) -> tuple[float, str]:
    explanation = f"The hash operator incurs no additional cost as it builds a hash table in memory.\n"
    return (node["Plans"][0]["Total Cost"], explanation)

# explanation function for Aggregate
# attempted to adapt from cost_agg in src/backend/optimizer/path/costsize.c
# it is missing the disk spill costs for hash/mixed strategies
def explain_aggregate(node: dict) -> tuple[float, str, str]: 
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    child = node["Plans"][0]
    child_cost = child["Total Cost"]
    child_tuples = child["Plan Rows"]
    strategy = node["Strategy"]

    # plain agg is simple if startup cost is ignored
    if strategy == "Plain":
        total_cost = node["Startup Cost"]
        explanation = f"In Plain Aggregation, there is no grouping involved.\n"
        explanation += f"The startup cost is the total cost of the child node and aggregation costs on input tuples.\n"
        explanation += f"cpu_tuple_cost ({cpu_tuple_cost}) is added to the the total cost."
        return (total_cost + cpu_tuple_cost, explanation)

    total_cost = child_cost
    # just in case
    numGroupCols = len(node["Group Key"]) if "Group Key" in node else len(clean_output(node["Output"]))
    total_cost += cpu_operator_cost * numGroupCols * child_tuples
    total_cost += cpu_tuple_cost * node["Plan Rows"]

    total_cost += cpu_operator_cost * child_tuples
    total_cost += cpu_operator_cost * node["Plan Rows"]
    
    comment = ""

    # sorted, mixed, and hashed have the same base total cost (hash has computation cost charged during startup instead)
    if strategy == "Sorted" or strategy == "Mixed":
        explanation = f"When sorting, no additional startup cost is incurred.\n"
        explanation += f"cpu_operator_cost ({cpu_operator_cost}) * length of grouping keys ({numGroupCols}) is charged for each input tuple\n"
        explanation += f"cpu_tuple_cost ({cpu_tuple_cost}) is charged for each output tuple.\n"
        explanation += f"cpu_operator_cost also charged to each input and output tuple as aggregation cost."
    else:
        explanation = f"When hashing, cpu_operator_cost ({cpu_operator_cost}) * length of grouping keys ({numGroupCols}) is charged for each input tuple during startup. This represents the cost of the hash computation.\n"
        explanation += f"cpu_tuple_cost ({cpu_tuple_cost}) is charged for each output tuple\n"
        explanation += f"cpu_operator_cost also charged to each input and output tuple as aggregation cost."

    # disk spill costs not accounted for given the estimate is likely to be incorrect anyway

    # every aggregation has its own trans and final costs. we don't know what they are
    # modest assumption is one aggregation charging cpu_operator_cost for trans and final each
    expected_cost = node["Total Cost"]
    if truncate_cost(total_cost) != expected_cost:
        comment = "The assumption of 1 cpu_operator_cost to each input & output tuple is likely to be an underestimate.\n"
        comment += "We do not know the amount and costs of the aggregation functions used."

    return (total_cost, explanation, comment)

# estimate the M from the lectures by using work_mem and block_size
def get_m() -> float:
    return float(cache.get_setting('work_mem')) * 1024 / float(cache.get_setting('block_size'))

# Try to find a base relation for a node
def find_relation_name(plan):
    # Check if the current node directly has a 'Relation Name'
    if 'Relation Name' in plan:
        return plan['Relation Name']
      
    # If there are sub-plans, recursively search them
    if 'Plans' in plan:
        for subplan in plan['Plans']:
            relation_name = find_relation_name(subplan)
            if relation_name:
                return relation_name

    return None 

# explanation function for Nested Loop
# attempted to adapt from initial_cost_nestloop and final_cost_nestloop in src/backend/optimizer/path/costsize.c
def explain_nestedloop(node: dict) -> str:
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    # Extracting costs and rows from the node dictionary
    outer_plan = node['Plans'][0]
    inner_plan = node['Plans'][1]

    outer_startup_cost = outer_plan['Startup Cost']
    outer_total_cost = outer_plan['Total Cost']
    outer_rows = outer_plan['Plan Rows']
    
    inner_startup_cost = inner_plan['Startup Cost'] #inner_rescan_startup_cost
    inner_total_cost = inner_plan['Total Cost'] #inner_rescan_total_cost
    inner_rows = inner_plan['Plan Rows']
    
    outer_relation = find_relation_name(outer_plan)
    inner_relation = find_relation_name(inner_plan)
    
    actual_outer_rows = cache.get_tuple_count(outer_relation)
    actual_inner_rows = cache.get_tuple_count(inner_relation)
    #Protect some assumptions below that rowcounts aren't zero 
    if (outer_rows <= 0):
        outer_rows = 1

    if (inner_rows <= 0):
        inner_rows = 1
    startup_cost =0
    run_cost=0
    # Basic calculations for startup and run costs
    startup_cost += outer_startup_cost + inner_startup_cost
    #run_cost = (outer_total_cost - outer_startup_cost) + (inner_total_cost - inner_startup_cost) * outer_rows
    run_cost += (outer_total_cost - outer_startup_cost)

    if (outer_rows>1):
        run_cost+=(outer_rows-1)*inner_startup_cost   #inner_rescan_start_cost

    inner_run_cost = inner_total_cost-inner_startup_cost
    inner_rescan_run_cost= inner_total_cost - inner_startup_cost

    relation_name = find_relation_name(node)  
    
    actual_rows = cache.get_tuple_count(relation_name)

    # Adjustments based on join type
    if node['Join Type'] in ['Inner']:
            cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))

            # Extracting necessary components from the node
            outer_plan = node['Plans'][0]
            inner_plan = node['Plans'][1]
            
            outer_startup_cost = outer_plan['Startup Cost']
            outer_total_cost = outer_plan['Total Cost']
            outer_rows = outer_plan['Plan Rows']
            
            inner_startup_cost = inner_plan['Startup Cost']
            inner_total_cost = inner_plan['Total Cost']
            inner_rows = inner_plan['Plan Rows']

            startup_cost = outer_startup_cost + inner_startup_cost
            run_cost = outer_total_cost - outer_startup_cost 

            if outer_rows > 1:
                run_cost += (outer_rows - 1) * inner_startup_cost

            inner_run_cost = inner_total_cost - inner_startup_cost

            run_cost += outer_rows * inner_run_cost

            total_cost = startup_cost + run_cost

            explanation = (
                f"Nested Loop Join (Inner Join) Explanation:\n"
                f"- Outer Plan: {outer_plan['Node Type']} with {outer_rows} rows\n"
                f"- Inner Plan: {inner_plan['Node Type']} with {inner_rows} rows\n"
                f"- Startup Cost: Outer = {outer_startup_cost:.2f}, Inner = {inner_startup_cost:.2f}, Total = {startup_cost:.2f}\n"
                f"- Run Cost: Outer Total = {outer_total_cost:.2f}, Inner per Row = {inner_run_cost:.2f}, Total Run = {run_cost:.2f}\n"
                f"- Total Cost: {total_cost:.2f}\n"
            )

            return total_cost, explanation


    if node['Join Type'] in ['Semi', 'Anti']:
        # Assuming early exit after the first match
        outer_match_frac = min(1, actual_rows / actual_outer_rows if actual_outer_rows > 0 else 1)
        match_count = actual_rows / outer_rows if outer_rows > 0 else 0

        outer_matched_rows=outer_rows*outer_match_frac
        outer_unmatched_rows= outer_rows - outer_matched_rows
        inner_scan_frac = 2.0 / (match_count + 1.0)

        ntuples = outer_matched_rows * inner_rows * inner_scan_frac

    # Calculate the number of tuples processed (not necessarily resulting in output)
        ntuples = outer_matched_rows * inner_rows * inner_scan_frac
        #run_cost+= inner_run_cost * inner_scan_frac
        #run_cost = (outer_total_cost - outer_startup_cost) + inner_startup_cost * outer_rows #final loop

        #if no index join quals
        ntuples += outer_unmatched_rows * inner_rows
        run_cost += inner_run_cost
        if (outer_unmatched_rows >= 1):
            outer_unmatched_rows -= 1
        else:
            outer_matched_rows -= 1
        if (outer_matched_rows > 0):
            run_cost += outer_matched_rows * inner_rescan_run_cost * inner_scan_frac
        if(outer_unmatched_rows > 0):
            run_cost += outer_unmatched_rows * inner_rescan_run_cost

    #normal case:
    else:
        run_cost+=inner_run_cost
        if (outer_rows >1):
            run_cost+=(outer_rows-1)*inner_rescan_run_cost
        ntuples = outer_rows * inner_rows
        run_cost +=  cpu_tuple_cost* ntuples

    total_cost = startup_cost + run_cost

    # Explanation assembly
    explanation = f"Nested Loop Join Explanation:\n" \
                  f"- Outer Rows: {outer_rows}, Inner Rows: {inner_rows}\n" \
                  f"- Startup Costs: Outer = {outer_startup_cost}, Inner = {inner_startup_cost}, Total = {startup_cost}\n" \
                  f"- Run Costs: Outer Run Cost = {outer_total_cost - outer_startup_cost}\n " \
                  f"-Inner Run Cost per Outer Row = {inner_total_cost - inner_startup_cost}\n " \
                  f"-Total Run Cost = {run_cost}\n" \
                  f"- Total Cost: {total_cost}\n"

    return total_cost, explanation

# explanation function for Hash Join
# attempted to adapt from initial_cost_hashjoin and final_cost_hashjoin in src/backend/optimizer/path/costsize.c
def explain_hash_join(node: dict) -> tuple[float, str]:
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

# explanation function for Hash Join based on lecture slides
def explain_hashjoinlect(node: dict) -> str: #grace hash join
    seq_page_cost = float(cache.get_setting("seq_page_cost"))
    outer_path = [child for child in node["Plans"] if child["Parent Relationship"] == "Outer"][0]
    inner_path = [child for child in node["Plans"] if child["Parent Relationship"] == "Inner"][0]
    outer_path_rows = outer_path["Plan Rows"]
    inner_path_rows = inner_path["Plan Rows"]
    inputbuffers = get_m()
    cpu_block_size = float(cache.get_setting('block_size'))
          
    # Calculate the startup and run costs
    startup_cost = 0
    run_cost = 0

    # Startup cost
    startup_cost = node.get('Startup Cost', 1.56)

    # Calculate the run cost of the inner relation excluding startup cost
    outer_blocks = math.ceil(outer_path_rows * outer_path["Plan Width"] / cpu_block_size)
    inner_blocks = math.ceil(inner_path_rows * inner_path["Plan Width"] / cpu_block_size)
    blocks_accessed = 3 * (outer_blocks + inner_blocks)
    run_cost += blocks_accessed * seq_page_cost


    # Prepare explanation
    explanation = f"The startup and total costs for the hash join are as follows:\n"
    explanation += f"Startup cost: {startup_cost}\n"
    explanation += f"Total cost is the startup cost and run cost: {startup_cost + run_cost}\n"
    explanation += f"- Following the formula of 3(B(R) + B(S))\n"
    explanation += f"- B(outerloop) = {outer_blocks}, B(innerloop) = {inner_blocks}, and M = {inputbuffers}\n"
    explanation += f"- The run cost will be the seq_page_cost({seq_page_cost}) of accessing all the blocks"
    return startup_cost + run_cost, explanation

# explanation function for Merge Join based on lecture slides
def explain_mergejoinlect(node: dict) -> str: #refined sort-merge join
    seq_page_cost = float(cache.get_setting("seq_page_cost"))
    outer_path = [child for child in node["Plans"] if child["Parent Relationship"] == "Outer"][0]
    inner_path = [child for child in node["Plans"] if child["Parent Relationship"] == "Inner"][0]
    outer_path_rows = outer_path["Plan Rows"]
    inner_path_rows = inner_path["Plan Rows"]
    inputbuffers = get_m()
    cpu_block_size = float(cache.get_setting('block_size'))

    # Calculate the startup and run costs
    startup_cost = 0
    run_cost = 0

    # Startup cost
    startup_cost = node.get('Startup Cost', 1.56)

    # Calculate the run cost of the inner relation excluding startup cost
    outer_blocks = math.ceil(outer_path_rows * outer_path["Plan Width"] / cpu_block_size)
    inner_blocks = math.ceil(inner_path_rows * inner_path["Plan Width"] / cpu_block_size)
    blocks_accessed = 3 * (outer_blocks + inner_blocks)
    run_cost += blocks_accessed * seq_page_cost

    # Prepare explanation
    explanation = f"The startup and total costs for merge join are as follows:\n"
    explanation += f"Startup cost: {startup_cost}\n"
    explanation += f"Total cost is the startup cost and run cost: {startup_cost + run_cost}\n"
    explanation += f"- Following the formula of 3(B(R) + B(S))\n"
    explanation += f"- B(outerloop) = {outer_blocks}, B(innerloop) = {inner_blocks}, and M = {inputbuffers}\n"
    explanation += f"- The run cost will be the seq_page_cost({seq_page_cost}) of accessing all the blocks"
    return startup_cost + run_cost, explanation

# explanation function for Unique
# adapted from create_upper_unique_path and create_unique_path in src/backend/optimizer/util/pathnode.c
# assumption here is if hashing is used then an aggregate will show up instead of unique
def explain_unique(node: dict) -> tuple[float, str]:
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    subpath = node["Plans"][0]
    sub_cost = subpath["Total Cost"]
    input_tuples = subpath["Plan Rows"]
    numCols = len(clean_output(node["Output"]))
    total_cost = input_tuples * numCols * cpu_operator_cost
    explanation = f"There is no additional startup for the Unique operator.\n"
    explanation += f"We charge cpu_operator_cost ({cpu_operator_cost}) * number of columns ({numCols}) for each input tuple ({input_tuples}).\n"
    explanation += f"This adds {total_cost} to the total cost."

    comment = ""
    expected_cost = node["Total Cost"] - sub_cost
    if truncate_cost(total_cost) != expected_cost:
        comment = "The difference most likely arises from a hashing strategy.\n"
        comment += "The strategy used applies for plain and sort-based unqiue.\n"
        comment += "We assumed a hashed unique would show up as an aggregated instead."

    return (sub_cost + total_cost, explanation)

# explanation function for CTE Scan
# adapted from cost_ctescan in src/backend/optimizer/path/costsize.c
# applies for both CTE Scan and WorkTable Scan
def explain_cte(node : dict) -> tuple[float, str, str]:
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
        filters = count_clauses(node["Filter"])
        cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
        filter_cost = filters * cpu_operator_cost
        cost_per_tuple += filter_cost
        explanation += f"CPU cost per tuple is increased by {filter_cost:.4f} for {filters} filters * cpu_operator_cost ({cpu_operator_cost})\n"

    total_cost = rows * cost_per_tuple
    explanation += f"The cost per tuple is {cost_per_tuple}. For {rows} row(s), the total cost is {total_cost}"

    # likely to happen for scans with no children, the input tuples are omitted.
    expected_cost = node["Total Cost"]
    if truncate_cost(base_cost + total_cost) != expected_cost:
        comment = "The given plan row count by PostgresSQL is not the same as the input tuples.\n"
        comment += "We cannot get the actual tuple count for a WorkTable/CTE\n"
        comment += f"If the estimated cost per tuple is right, there should have been {round(expected_cost/cost_per_tuple)} input tuples."
    
    return (base_cost + total_cost, explanation, comment)

# explanation function for Function, Table Function and Named Tuplestore Scans
# these scans follow the same basic principles
# adapted from cost_functionscan, cost_tablefuncscan and cost_namedtuplestorescan in src/backend/optimizer/path/costsize.c
def explain_xyz_scan(node: dict, fn_name:str) -> tuple[float, str, str]:
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost")) 
    tuples = node["Plan Rows"]
    total_cost = tuples * cpu_tuple_cost
    startup_cost = node["Startup Cost"]
    expected_cost = node["Total Cost"] - startup_cost
    comment = ""

    explanation = f"{fn_name} charges cpu_tuple_cost ({cpu_tuple_cost}) per tuple ({tuples})."

    # likely to happen since we are not attempting to get the function tuple costs
    if truncate_cost(total_cost) != expected_cost:
        comment = f"{fn_name} scan may involve other costs to evaluate expressesions.\n"
        comment += f"The cost per tuple should have been {expected_cost/tuples}"
    
    return (startup_cost + total_cost, explanation, comment)

# explanation function for Function Scan
def explain_functionscan(node: dict) -> tuple[float, str, str]:
    return explain_xyz_scan(node, "Function Scan")

# explanation function for Table Function Scan
def explain_tablefunctionscan(node: dict) -> tuple[float, str, str]:
    return explain_xyz_scan(node, "Table Function Scan")

# explanation function for Named Tuplestore Scan
def explain_namedtuplestorescan(node: dict) -> tuple[float, str, str]:
    return explain_xyz_scan(node, "Named Tuplestore Scan")

# explanation function for Tid Scan
# adapted from cost_tidscan in src/backend/optimizer/path/costsize.c
def explain_tidscan(node: dict) -> tuple[float, str, str]:
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost")) 
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost")) 
    random_page_cost = float(cache.get_setting("random_page_cost")) 
    comment = ""

    # both conditions apply
    tid_filters = count_clauses(node["TID Cond"])
    table_filters = count_clauses(node["Filter"]) if "Filter" in node else 0

    explanation = f"TID Scan charges cpu_operator_cost ({cpu_operator_cost}) for every TID filter at startup ({tid_filters}).\n"
    startup_cost = cpu_operator_cost * tid_filters
    tuples = node["Plan Rows"]

    cpu_per_tuple = cpu_tuple_cost + random_page_cost + (table_filters * cpu_operator_cost)
    total_cost = cpu_per_tuple * tuples

    explanation += f"TID Scan assumes each tuple is on a random page, so we charge random_page_cost ({random_page_cost}) per tuple."
    explanation += f"cpu_tuple_cost ({cpu_tuple_cost}) is also incurred per tuple." 
    explanation += f"TID scan also charges cpu_operator_cost ({cpu_operator_cost}) for {table_filters} filters per tuple." 
    explanation += f"For {tuples} tuples, this cost comes out to be {total_cost:.4f}."

    expected_cost = node["Total Cost"]

    # tries its best to explain the difference
    # if both are wrong then this output isn't very useful
    if truncate_cost(total_cost + startup_cost) != expected_cost:
        expected_cost -= node["Startup Cost"]
        comment = f"The difference can arise from the actual number of tuples scanned (since we only get the output rows) or the filtering cost.\n"
        comment += f"If the filtering cost is correct, the actual number of rows is {round(expected_cost/cpu_per_tuple)}\n"
        comment += f"If the number of rows is correct, the actual cost per tuple is {(expected_cost/tuples):.4f}\n"

    return (startup_cost + total_cost, explanation, comment)

# explanation function for Tid Range Scan
def explain_tidrangescan(node: dict) -> tuple[float, str, str]:
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost")) 
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost")) 
    random_page_cost = float(cache.get_setting("random_page_cost")) 
    seq_page_cost = float(cache.get_setting("seq_page_cost"))
    reltuples = cache.get_tuple_count(node["Relation Name"])
    relpages = cache.get_page_count(node["Relation Name"])
    comment = ""

    # both conditions apply
    tid_filters = count_clauses(node["TID Cond"])
    table_filters = count_clauses(node["Filter"]) if "Filter" in node else 0

    explanation = f"TID Scan charges cpu_operator_cost ({cpu_operator_cost}) for every TID filter at startup ({tid_filters}).\n"
    startup_cost = cpu_operator_cost * tid_filters
    tuples = node["Plan Rows"]
    # overall selectivity estimation
    selectivity = tuples/reltuples
    pages = math.ceil(selectivity * relpages)
    seq_pages = pages - 1

    disk_cost = random_page_cost + seq_pages * seq_page_cost
    total_cost = disk_cost
    cpu_per_tuple = cpu_tuple_cost + (table_filters * cpu_operator_cost)
    total_cost += cpu_per_tuple * tuples

    explanation += f"TID Range Scan charges random_page_cost ({random_page_cost}) for the first page and then seq_page_cost ({seq_page_cost}) for the remaining. We have assumed {pages} pages.\n"
    explanation += f"cpu_tuple_cost ({cpu_tuple_cost}) is also incurred per tuple.\n" 
    explanation += f"TID Range scan also charges cpu_operator_cost ({cpu_operator_cost}) for {table_filters} filters per tuple.\n" 
    explanation += f"For {tuples} tuples, this cost comes out to be {total_cost:.4f}."

    expected_cost = node["Total Cost"]

    # tries its best to explain the difference
    # if both are wrong then this output isn't very useful
    if truncate_cost(total_cost + startup_cost) != expected_cost:
        expected_cost -= node["Startup Cost"]
        comment = f"The difference can arise from the actual number of tuples scanned (since we only get the output rows) or the filtering cost.\n"
        
        # not sure how to reverse a function with math.ceil() in it, so we try to get as close of an estimate as possible instead
        e_cost = 0
        x = 1
        while e_cost < expected_cost:
            e_cost = random_page_cost + (math.ceil(x/reltuples * relpages) - 1) * seq_page_cost + cpu_per_tuple * x
            x += 1

        comment += f"If the filtering cost is correct, the actual number of rows is {x-1}\n"
        comment += f"If the number of rows is correct, the actual cost per tuple is {((expected_cost - disk_cost)/tuples):.4f}\n"

    return (startup_cost + total_cost, explanation, comment)

# explanation function for Sample Scan
# adapted from cost_samplescan in src/backend/optimizer/path/costsize.c
def explain_samplescan(node: dict) -> tuple[float, str, str]:
    relpages = cache.get_page_count(node["Relation Name"])
    reltuples = cache.get_tuple_count(node["Relation Name"])
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    seq_page_cost = float(cache.get_setting("seq_page_cost"))
    random_page_cost = float(cache.get_setting("random_page_cost"))

    # just in case
    method = node["Sampling Method"]
    if method != "bernoulli" and method != "system":
        return (0, "", "I am unaware of this sampling method. (bernoulli & system only)")
    
    rate = float(node["Sampling Parameters"][0].split("'")[1])
    tuples = round(reltuples * rate/100)
    explanation = f"N = {relpages}. S = {rate}.\n"
    comment = ""
    
    # bernoulli returns pages (but scans all of them first)
    if method == "bernoulli":
        explanation = f"For Bernoulli Sampling, all pages are sequentially scanned ({relpages}). seq_scan_cost is {seq_page_cost} per page.\n"
        page_cost = relpages * seq_page_cost
    # system returns tuples (assume random & unclustered)
    else:
        explanation = f"For System Sampling, (N * S/100) pages are scanned ({round(relpages * rate/100)}). random_page_cost is {random_page_cost} per page.\n"
        page_cost = round(relpages * rate/100) * random_page_cost

    # both charge the same amount of processing cost
    cpu_per_tuple = cpu_tuple_cost
    explanation += f"(N * S/100) = {tuples} tuples will be scanned.\n"
    explanation += f"cpu_tuple_cost {cpu_tuple_cost} is charged for each scanned tuple.\n"

    if "Filter" in node:
        num_filters = count_clauses(node["Filter"])
        cpu_per_tuple += cpu_operator_cost * num_filters
        explanation += f"For filtering, cpu_operator_cost ({cpu_operator_cost}) is applied for each filter ({num_filters} per tuple.\n"
    
    total_cost = page_cost + cpu_per_tuple * tuples
    explanation += f"This gives the total cost as {total_cost}."

    # page_cost should most likely be correct
    expected_cost = node["Total Cost"]
    if truncate_cost(total_cost) != expected_cost:
        comment = "The cost per tuple is most likely incorrect. This could be due to incorrect functions/filtering cost.\n"
        comment += f"It should be {((expected_cost - page_cost) / tuples):.4f} instead of {cpu_per_tuple}"

    return (total_cost, explanation, comment)    

# explanation function for Memoize
# adapted from create_memoize_path in src/backend/optimizer/util/pathnode.c
def explain_memoize(node: dict) -> tuple[float, str]:
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    explanation = f"Postgres charges cpu_tuple_cost ({cpu_tuple_cost}) at startup (and subsequently total cost) as the cost to cache the first entry.\n"
    explanation += f"The actual cost of rescanning and caching during the many loops of a Nested Join is calculated by the Nested Loop operator.\n"

    return (node["Plans"][0]["Total Cost"] + 0.01, explanation)

# explanation function for Window Aggregation
# attempted to adapt from cost_windowagg in src/backend/optimizer/path/costsize.c
def explain_windowagg(node: dict) -> tuple[float, str, str]:
    cpu_operator_cost = float(cache.get_setting("cpu_operator_cost"))
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))

    child = node["Plans"][0]
    child_cost = child["Total Cost"]
    input_tuples = child["Plan Rows"]

    explanation = f"There is no additional startup cost for the WindowAgg operator.\n"
    explanation = f"All cost is applied to the input tuples. We are assuming the minimum cost which is cpu_tuple_cost ({cpu_tuple_cost}) + 2 * cpu_operator_cost {cpu_operator_cost}."

    # same issue as agg.  
    # modest assumption is one aggregation charging cpu_operator_cost for trans and final each
    cpu_per_tuple = cpu_tuple_cost + cpu_operator_cost * 2
    total_cost = cpu_per_tuple * input_tuples

    comment = ""
    expected_cost = node["Total Cost"] - child_cost
    if truncate_cost(total_cost) != expected_cost:
        comment = f"WindowAgg charges cpu_operator_cost for every Parition and Order column. This information is not visible to us.\n"
        comment += f"There may be additional costs for filtering.\n"
        comment += f"The cost per tuple should have been {(expected_cost/input_tuples):.4f} instead of the assumed {cpu_per_tuple}."

    return (total_cost + child_cost, explanation, comment)


def explain_bitmap_heap_scan(node: dict):
    # Assume we have fetched settings for costs and page sizes
    cpu_tuple_cost = float(cache.get_setting("cpu_tuple_cost"))
    seq_page_cost = float(cache.get_setting("seq_page_cost"))  # Example: 1.0
    random_page_cost = float(cache.get_setting("random_page_cost"))  # Example: 4.0

    # Assuming the node dictionary includes necessary details
    pages = node.get("Total Pages", 1000)  # Total pages in the relation
    pages_fetched = node.get("Pages Fetched", 100)  # Estimated pages fetched by bitmap
    tuples_fetched = node.get("Tuples Fetched", 10000)  # Estimated tuples fetched
    
    # Calculate the interpolated cost per page fetched
    if pages_fetched >= 2.0:
        cost_per_page = random_page_cost - (random_page_cost - seq_page_cost) * (pages_fetched / pages) ** 0.5
    else:
        cost_per_page = random_page_cost

    # Costs initialization
    startup_cost = 0
    run_cost = pages_fetched * cost_per_page

    # CPU cost per tuple fetched
    cpu_per_tuple = cpu_tuple_cost  # Adjust as needed
    cpu_run_cost = cpu_per_tuple * tuples_fetched

    # Total cost calculations
    total_startup_cost = startup_cost  # Add specific startup operations cost if needed
    total_run_cost = run_cost + cpu_run_cost
    total_cost = total_startup_cost + total_run_cost

    explanation = f"Bitmap Heap Scan Cost Estimation:\n" \
                  f"Total Pages in Relation: {pages}, Pages Fetched: {pages_fetched}\n" \
                  f"Tuples Fetched: {tuples_fetched}\n" \
                  f"Startup Cost: {total_startup_cost}\n" \
                  f"Run Cost: {total_run_cost}\n" \
                  f"Total Cost: {total_cost}\n"

    return total_cost, explanation

fn_dict = {
    "Nested Loop": explain_nestedloop,
    "Merge Join": explain_mergejoinlect,
    "Hash Join": explain_hashjoinlect,

    "Index Scan": explain_indexscan,
    "Index Only Scan": None,
    "Bitmap Index Scan": None,
    "Bitmap Heap Scan": explain_bitmap_heap_scan,
    "Incremental Sort": None,
    "Aggregate": explain_aggregate,
    "WindowAgg": explain_windowagg,
    "Result": explain_result,
    "ProjectSet": explain_project_set,
    "ModifyTable": explain_modify_table,
    "Append": explain_append,
    "Merge Append": explain_merge_append,
    "Recursive Union": explain_recursive_union,
    "BitmapAnd": explain_bitmap_and,
    "BitmapOr": explain_bitmap_or,
    "Seq Scan": explain_seqscan,
    "Sample Scan": explain_samplescan,
    "Gather": explain_gather,
    "Gather Merge": explain_gather_merge,
    "Tid Scan": explain_tidscan,
    "Tid Range Scan": explain_tidrangescan,
    "Subquery Scan": explain_subqueryscan,
    "Function Scan": explain_functionscan,
    "Table Function Scan": explain_tablefunctionscan,
    "Values Scan": explain_valuescan,
    "CTE Scan": explain_cte,
    "Named Tuplestore Scan": explain_namedtuplestorescan,
    "WorkTable Scan": explain_cte,
    "Materialize": explain_materialize,
    "Memoize": explain_memoize,
    "Sort": explain_sort,
    "Group": explain_group,
    "Unique": explain_unique,
    "SetOp": explain_setop,
    "LockRows": explain_lockrows,
    "Limit": explain_limit,
    "Hash": explain_hash,

    # not worth it
    # ommited due to complexity and irrelevance to lecture content
    "Foreign Scan": None,
    "Custom Scan": None,
    "Incremental Sort": None
}

# class responsible for maintaining Postgres connection and communicating with the interface
class Connection():
    def __init__(self) -> None:
        self.connection = None
    
    # check if connected already
    def connected(self) -> bool:
        return self.connection is not None and not self.connection.closed
    
    # attempt to disconnect
    def disconnect(self) -> None:
        if self.connected():
            self.connection.close()

    # attempt to connect
    def connect(self, dbname:str, user:str, password:str, host:str, port:str) -> str:
        # disconenct first 
        self.disconnect()
        try:
            self.connection = psycopg.connect(dbname=dbname, user=user, password=password, host=host, port=port)
            # autocommit to get results instantly
            self.connection.autocommit = True

            # initialize cache for this session
            global cache
            cache = Cache(self.connection.cursor())
            return ""
        except Exception as e:
            return str(e)
    
    # responsible for calling the explanation function and adding comments as needed
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
                if cost != expected_cost:
                    # assume rounding error by default
                    diff = truncate(abs(full_cost - expected_cost))
                    if diff <= 0.05:
                        explanation += f"\n<b>Comments</b>: Calculated cost is off by {diff:.4f}, which is most likely a rounding error."
                    # otherwise use comments if provided
                    elif len(params) > 2 and len(params[2]) > 0:
                        explanation += f"\n<b>Comments</b>: {params[2]}"
                return explanation
            except Exception as e:
                # we do this instead of crashing the app in case other nodes made it
                return f"<b>Comment</b>: Encountered an error when generating an explanation. {str(e)}"
        
        # default
        return f"<b>Comment</b>: I really don't know how to explain the cost for this operator!"

    # this is the function called by the interface
    # force_analysis is true whenever called by the interface
    def explain(self, query:str, force_analysis:bool, log_cb: callable) -> str:
        # save the log callback function
        self.log = log_cb
        cache.set_log_cb(log_cb)

        # just in case
        if not self.connected():    
            return "No database connection found! There is no context for this query."

        # get the cursor and execute query
        cur = self.connection.cursor()
        log_cb("Asking Postgres for QEP")
        try:
            # add explain automatically
            # same behaviour as pgAdmin whereby if the user also specifies EXPLAIN, an error will be thrown
            cur.execute(f"EXPLAIN (COSTS, VERBOSE, FORMAT JSON) {query}")
        except Exception as e:
            return f"Error: {str(e)}"
        
        # get the plan and pretty print it to the disk (useful to understand)
        plan = cur.fetchall()[0][0][0]['Plan']
        with open("plan.json","w") as f:
            f.write(json.dumps(plan, indent=2))

        # we explain all nodes bottom up
        # no real reason to do this tbh
        node_stack = deque()

        # Workers Planned may be needed by different nodes, but it may only be specificed for upper nodes
        log_cb("Pushing Workers Planned Down")
        def add_nodes(node, workers):
            if "Workers Planned" in node:
                workers = node["Workers Planned"]
            elif node["Parallel Aware"]:
                # no point if not parallely aware
                node["Workers Planned"] = workers
            node_stack.append(node)
            if "Plans" in node:
                for child_plan in node["Plans"]:
                    add_nodes(child_plan, workers)

        # call the recursive function
        add_nodes(plan, 1)

        # for tables with no autoanalysis, we will force them to analyse the table
        # this usually happens for small table with < 50 values (region and nation)
        # this check is only done when called by the interface
        log_cb("Checking for missing analysis")
        if force_analysis:
            # make a copy of the stack for traversal
            stack_copy = node_stack.copy()
            reexplain = False
            for _ in range(len(stack_copy)):
                node = stack_copy.pop()
                if "Relation Name" in node:
                    rel = node["Relation Name"]
                    # we use tuplecount == -1 as a way to determine if analysis has been done
                    tuple_count = cache.query_tuplecount(rel)
                    if tuple_count == -1:
                        # QEP needs to be re-explained
                        reexplain = True
                        log_cb(f"Analysis missing for {rel}\n")
                        try:
                            # this should be enough for analysis to take place
                            cur.execute(f"ANALYZE {rel}")
                        except Exception as e:
                            return f"Error: {str(e)}"
                    else:
                        cache.set_tuple_count(rel, tuple_count)
            
            # need to regen the QEP even if one node has to be re-explained                
            if reexplain:
                log_cb(f"Asking Postgres to generate QEP again\n")
                return self.explain(query, False, log_cb)

        # add explanation for each node         
        for _ in range(len(node_stack)):
            node = node_stack.pop()
            node["Explanation"] = self.get_explanation(node)

        # return plan with explanation
        return plan    
