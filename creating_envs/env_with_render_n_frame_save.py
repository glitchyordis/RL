import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from gymnasium.spaces import Box
from gymnasium.wrappers import RecordVideo
import imageio
import cv2

"""
An example code on howw to render framee with imageio

main point: -> just look at render function
"""

class MyColormapEnv(gym.Env):
    def __init__(self, size=20, render_mode=None):
        super().__init__()
        self.size = size
        self.render_mode = render_mode
        self.x = np.linspace(-5, 5, size)
        self.y = np.linspace(-5, 5, size)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.data = np.random.rand(size, size)  # Example data for colormap
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(size, size), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.fig = None
        self.ax = None
        self.agent_position = np.array([0.0, 0.0], dtype=np.float64)  # Initialize as float

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.data = np.random.rand(self.size, self.size)
        self.agent_position = np.array([0.0, 0.0], dtype=np.float64)  # Reset as float
        observation = self.data
        info = {}
        return observation, info

    def step(self, action):
        # Update the agent's position based on the action
        self.agent_position += action
        self.agent_position = np.clip(self.agent_position, [-5, -5], [5, 5])  # Keep within bounds

        # Calculate reward (example: negative distance to goal)
        goal_position = np.array([self.x.max(), self.y.max()])
        reward = -np.linalg.norm(self.agent_position - goal_position)

        # Check termination condition (example: agent reaches goal)
        terminated = np.linalg.norm(self.agent_position - goal_position) < 0.5
        truncated = False

        observation = self.data
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self, video_writer=None):
        if self.render_mode is None:
            gym.logger.warn(
                "render() was called without render_mode specified."
                "Set render_mode='human' if you want to see the rendering visually."
            )
            return None

        if self.fig is None or self.ax is None:
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=(8, 8))

        self.ax.clear()

        # Plot the heatmap
        im = self.ax.imshow(
            self.data,
            cmap=cm.cividis,
            extent=[self.x.min(), self.x.max(), self.y.min(), self.y.max()],
            origin='lower'
        )

        # Add the agent's position
        self.ax.scatter(self.agent_position[0], self.agent_position[1], color='red', s=100, marker='s', label="Agent")

        # Add the goal position (example: top-right corner)
        goal_x, goal_y = self.x.max(), self.y.max()
        self.ax.scatter(goal_x, goal_y, color='green', s=100, marker='^', label="Goal")

        # Add the target position (example: bottom-left corner)
        target_x, target_y = self.x.min(), self.y.min()
        self.ax.scatter(target_x, target_y, color='blue', s=100, marker='o', label="Target")

        # Add labels and legend
        self.ax.set_title("Agent Movement with Goal and Target")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

        # Add colorbar
        if not hasattr(self, 'cbar') or self.cbar is None:
            self.cbar = self.fig.colorbar(im)

        # # Convert the Matplotlib figure to a NumPy array
        canvas = self.fig.canvas
        canvas.draw()
        width, height = canvas.get_width_height()
        buffer = canvas.buffer_rgba()
        image_array = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
        image_array_rgb = image_array[:, :, :3]
        
        if video_writer:
            # either auto resize or resize to 16x16 multiple to avoid IMAGEIO FFMPEG_WRITER. 
            height, width, _ = image_array_rgb.shape
            new_width = (width + 15) // 16 * 16  # Round up to the nearest multiple of 16
            new_height = (height + 15) // 16 * 16
            resized_frame = cv2.resize(image_array_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
            video_writer.append_data(resized_frame)
            
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        return image_array

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            
video_path = "output_video.mp4"
video_writer = imageio.get_writer(video_path, fps=30)
env = MyColormapEnv(size=100, render_mode="human")
observation, info = env.reset()
terminated = False
truncated = False

for _ in range(5):
    print(_)
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    env.render(video_writer)  # Render the environment
    if terminated or truncated:
        observation, info = env.reset()

env.close()
