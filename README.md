# SC3020-CZ4031-Project-2
Project 2 Query Execution Plan Cost Explainer

## Installation Prerequesites
- All cost mappings are for `PostgreSQL 16.2`.
- Tested with `Python 3.11.4` or higher on `Windows 10/11`

## Installation
```
cd SC3020-CZ4031-Project-2
pip install -r requirements.txt
python project.py
```

### Libraries Used
1. `networkx` - graph layouts
2. `pyvis` - for the interactive graph visualiser and interface
3. `sv_ttk` - a dark theme for tkinter
4. `psycopg[binary]` - to establish a connection with a postgres server in python
5. `numpy` - used by `networkx` but it might not get installed as a dependency, which is why it has been included in `requirements.txt `

## Usage
The login screen has 5 inputs that are used to establish connection with a PostgreSQL Server (Tested on 16.2 only)

1. **Name** - Database Name [*TPC-H*]
2. **Username** - PostgreSQL Server Username [*postgres*]
3. **Password** - PostgreSQL Server Password []
4. **Host** - PostgreSQL Server Host [*localhost*]
4. **Port** - PostgreSQL Server Port [*5432*]

Click the connect button to move to the next frame.

In the explanation page, there is a large input field for the input query. Do not include `EXPLAIN` in the query as the program will add it automatically. Click the 'Explain' button to generate an input. The text field below the button will populate with status updates as the program executes. When the explained plan is ready, a webpage should be launched. This webpage should have an interactable graph where you can zoom in and out, move the screen and click on each node to get its cost explanation. The webpage is saved as `QEP.html` in the current working directory. Please open it from there if auto launch fails. 