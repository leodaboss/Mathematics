import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys

matplotlib.use('TkAgg')  # Use Tkinter for interactive plots

# Sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Enable interactive mode
plt.ion()

# Create a figure and axis
fig, ax = plt.subplots()
ax.plot(x, y, label="y = sin(x)")
ax.set_title("Click on a point to see its coordinates")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()

# Function to handle clicks
def onclick(event):
    # Check if the click was within the axes
    if event.inaxes == ax:
        # Get the x, y coordinates of the click
        x_click, y_click = event.xdata, event.ydata
        # Display the coordinates
        print(f"Clicked at: x = {x_click:.2f}, y = {y_click:.2f}")
        sys.stdout.flush()  # Ensure immediate output
        # Optionally annotate the point on the graph
        ax.plot(x_click, y_click, 'ro')  # Red dot
        ax.annotate(f"({x_click:.2f}, {y_click:.2f})",
                    (x_click, y_click),
                    textcoords="offset points",
                    xytext=(10, 10),
                    ha='center',
                    fontsize=8,
                    color='red')
        fig.canvas.draw()  # Update the plot

# Connect the click event to the handler
fig.canvas.mpl_connect('button_press_event', onclick)

# Keep the plot open
plt.show(block=True)
