import tkinter as tk
from tkinter import filedialog


class DrawingPad:
    def __init__(self, master):
        self.master = master
        self.file =""
        master.title("Drawing Pad")

        # Create canvas for drawing
        self.canvas = tk.Canvas(master, width=300, height=300, bg="white")
        self.canvas.pack()

        # Initialize grid data (all white)
        self.grid = [[-1 for _ in range(10)] for _ in range(10)]

        # Bind mouse click event
        self.canvas.bind("<Button-1>", self.on_click)

    def on_click(self, event):
        # Get coordinates of click
        x = event.x // 30
        y = event.y // 30

        # Check if click is within grid bounds
        if 0 <= x < 10 and 0 <= y < 10:
            # Toggle grid value and redraw square
            self.grid[y][x] *= -1
            self.draw_square(x, y, self.grid[y][x])

    def draw_square(self, x, y, color):
        # Define square color based on grid value
        fill_color = "black" if color == 1 else "white"

        # Draw the square on canvas
        self.canvas.create_rectangle(30 * x, 30 * y, 30 * (x + 1), 30 * (y + 1), fill=fill_color)

    def get_grid_matrix(self):
        # Return the grid data as a matrix
        return self.grid

    def save(self):
        # Get grid matrix and print it
        self.file = filedialog.asksaveasfile(initialfile = 'Untitled.txt', defaultextension=".txt",filetypes=[("All Files","*.*"),("Text Documents","*.txt")])
        with open(self.file.name, "w") as f:
            grid_matrix = self.get_grid_matrix()
            matrix = ""
            for row in grid_matrix:
                matrix = matrix + str(row) + ","
            matrix = matrix.rstrip(",")
            f.write(matrix)
        print("Created ", self.file)


        
       
# Create the main window
root = tk.Tk()

# Create the drawing pad instance
pad = DrawingPad(root)

# Add a submit button

save_button = tk.Button(root, text="Save", command=pad.save)
save_button.pack()

# Run the main event loop
root.mainloop()