import numpy as np
import heapq

class KalmanTracker:
    def __init__(self, dt=1.0):
        # A simple linear 2D Kalman filter for bounding boxes (cx, cy)
        # States: [x, y, vx, vy]
        self.dt = dt
        self.state = np.zeros((4, 1))
        
        # State Transition Matrix A
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0,  dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])
        
        # Measurement matrix H
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Covariance Matrices
        self.P = np.eye(4) * 1000
        self.R = np.eye(2) * 10  # Measurement noise
        self.Q = np.eye(4) * 0.1 # Process noise

    def predict(self):
        self.state = np.dot(self.A, self.state)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.state[:2].flatten()

    def update(self, measurement):
        # measurement is [cx, cy]
        z = np.array(measurement).reshape((2, 1))
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        y = z - np.dot(self.H, self.state)
        self.state = self.state + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        return self.state[:2].flatten()


def astar_path(grid, start, goal):
    """
    Finds a simple A* path on a 2D occupancy grid where 0 is free, 1 is obstacle.
    """
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    directions = [(0,1),(1,0),(0,-1),(-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
            
        for d in directions:
            neighbor = (current[0] + d[0], current[1] + d[1])
            
            # Bounds check
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                # Obstacle check
                if grid[neighbor[0], neighbor[1]] == 1:
                    continue
                    
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
    return [] # Path not found

def heuristic(a, b):
    # Euclidean distance
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
