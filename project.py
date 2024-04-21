import interface
import sv_ttk
from tkinter import ttk, Tk

# Main tkinter window
root = Tk()
root.resizable(width=False, height=False)
root.title("QEP Explainer")

# root size is (1,1). 1 partition that takes up all the space
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# define login and main app frames
login_frame = ttk.Frame(root)
app_frame = ttk.Frame(root)

# assign both frames to the root partition and take up all available space
login_frame.grid(row=0, column=0, sticky="nsew")
app_frame.grid(row=0, column=0, sticky="nsew")

# create both frames
login = interface.Login(login_frame, app_frame)
app = interface.App(app_frame, login_frame)

# show login_frame
interface.set_window_size(login_frame, interface.LOGIN_SIZE, True)
login_frame.tkraise()

# tkinter theme to make it look different from the default UI
sv_ttk.set_theme("dark")

# starts the UI thread
root.mainloop()