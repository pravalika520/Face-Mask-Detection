import tkinter as tk
root = tk.Tk()
root.title("Tkinter Test")
label = tk.Label(root, text="If you see this, Tkinter is working!")
label.pack(pady=20)
root.lift()
root.attributes('-topmost', True)
root.mainloop()
