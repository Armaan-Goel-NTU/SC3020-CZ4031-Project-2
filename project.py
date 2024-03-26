import interface
import sv_ttk
from tkinter import ttk, Tk

root = Tk()
root.resizable(width=False, height=False)
root.title("QEP Explainer")

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

login_frame = ttk.Frame(root)
app_frame = ttk.Frame(root)

login_frame.grid(row=0, column=0, sticky="nsew")
app_frame.grid(row=0, column=0, sticky="nsew")

login = interface.Login(login_frame, app_frame)
app = interface.App(app_frame, login_frame)

interface.set_window_size(login_frame, interface.LOGIN_SIZE)
login_frame.tkraise()

sv_ttk.set_theme("dark")
root.mainloop()