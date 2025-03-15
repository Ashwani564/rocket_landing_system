import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D


class Rocket:
    def __init__(self):
        # Rocket properties
        self.height = 70.0  # meters
        self.width = 3.7    # meters
        
        # Initial state - starting from different position
        self.position = np.array([200.0, 1000.0])  # x, y in meters
        self.velocity = np.array([-10.0, -30.0])   # Initial velocity with horizontal component
        self.orientation = -np.pi/12  # slight initial tilt
        
        # Landing parameters
        self.max_thrust = 845.0  # kN
        self.mass = 25000.0      # kg (mass during landing)
        self.fuel_mass = 6000.0  # kg of remaining fuel (increased)
        self.fuel_burn_rate = 150.0  # kg/second
        self.gravity = 9.81  # m/s^2
        self.landing_legs_deployed = False
        
        # Control parameters
        self.target_landing_x = 0.0  # target landing position
        self.pid_p = 0.3    # stronger proportional gain
        self.pid_d = 0.6    # stronger derivative gain
        self.pid_i = 0.025  # integral gain
        self.integral_error_x = 0.0
        self.integral_error_o = 0.0
        
        # Simulation parameters
        self.time = 0.0
        self.dt = 0.1  # simulation time step in seconds
        
        # Flight history
        self.history = {
            'time': [],
            'position': [],
            'velocity': [],
            'orientation': [],
            'thrust': [],
            'mass': [],
            'fuel': [],
            'legs_deployed': []
        }
        
    def calculate_control(self):
        # Landing controller
        x, y = self.position
        vx, vy = self.velocity
        
        # Target velocity based on distance to landing pad (stronger correction)
        x_distance = x - self.target_landing_x
        target_vx = -x_distance * 0.2
        
        # Adjust vertical speed based on altitude
        if y > 500:
            target_vy = -30.0  # faster descent at high altitude
        elif y > 200:
            target_vy = -15.0  # moderate descent
        elif y > 50:
            target_vy = -5.0   # slow descent
        else:
            target_vy = -1.5   # very slow final approach
        
        # Calculate error
        error_vx = target_vx - vx
        error_vy = target_vy - vy
        
        # Update integral error (with anti-windup)
        self.integral_error_x += error_vx * self.dt
        self.integral_error_x = np.clip(self.integral_error_x, -30, 30)
        
        # PID controller for horizontal correction
        thrust_angle = (self.pid_p * error_vx + 
                       self.pid_d * (-vx) + 
                       self.pid_i * self.integral_error_x)
        
        # Angle limits that progressively narrow as we approach ground
        max_angle = np.pi/6 if y > 500 else np.pi/8 if y > 200 else np.pi/12 if y > 50 else np.pi/24
        thrust_angle = np.clip(thrust_angle, -max_angle, max_angle)
        
        # Orientation control - actively force vertical as we approach landing
        if y < 300:
            # Progressive orientation correction
            correction_strength = min(1.0, (300 - y) / 250) * 1.5
            self.integral_error_o += self.orientation * self.dt * correction_strength
            self.integral_error_o = np.clip(self.integral_error_o, -np.pi/3, np.pi/3)
            
            # Add orientation correction to thrust angle
            orientation_correction = -self.orientation * 0.8 * correction_strength - self.integral_error_o * 0.2
            thrust_angle = thrust_angle * (1.0 - correction_strength) + orientation_correction
            
            # Target orientation moves toward vertical
            target_orientation = thrust_angle
        else:
            # At higher altitudes, orient toward thrust direction
            target_orientation = thrust_angle
        
        # Smoother orientation changes, more aggressive near ground
        if y < 100:
            self.orientation = 0.4 * self.orientation + 0.6 * target_orientation
        elif y < 300:
            self.orientation = 0.7 * self.orientation + 0.3 * target_orientation
        else:
            self.orientation = 0.8 * self.orientation + 0.2 * target_orientation
        
        # Calculate thrust magnitude based on altitude and velocity
        if y > 600:
            thrust_percentage = 0.4 + 0.2 * (error_vy / 20.0)
        elif y > 300:
            thrust_percentage = 0.6 + 0.2 * (error_vy / 15.0)
        elif y > 100:
            thrust_percentage = 0.8 + 0.2 * (error_vy / 10.0)
        elif y > 20:
            thrust_percentage = 0.9 + 0.1 * (error_vy / 5.0)
        else:
            # Very precise control near ground
            hover_thrust = (self.mass * self.gravity) / self.max_thrust
            thrust_percentage = hover_thrust + 0.2 * (error_vy / 2.0)
        
        # Ensure thrust is within limits
        thrust_percentage = np.clip(thrust_percentage, 0.0, 1.0)
        
        # Deploy landing legs when close to ground
        if y < 100 and not self.landing_legs_deployed:
            self.landing_legs_deployed = True
        
        return thrust_percentage, self.orientation
    
    def update(self):
        # Store current state
        self.history['time'].append(self.time)
        self.history['position'].append(self.position.copy())
        self.history['velocity'].append(self.velocity.copy())
        self.history['orientation'].append(self.orientation)
        self.history['mass'].append(self.mass)
        self.history['fuel'].append(self.fuel_mass)
        self.history['legs_deployed'].append(self.landing_legs_deployed)
        
        # Calculate control inputs
        thrust_percentage, orientation = self.calculate_control()
        
        # Calculate thrust
        if self.fuel_mass > 0:
            thrust_magnitude = self.max_thrust * thrust_percentage
            fuel_used = self.fuel_burn_rate * thrust_percentage * self.dt
            self.fuel_mass = max(0, self.fuel_mass - fuel_used)
            self.mass = self.mass - fuel_used
        else:
            thrust_magnitude = 0
        
        self.history['thrust'].append(thrust_magnitude)
        
        # Calculate forces
        thrust_direction = np.array([np.sin(orientation), np.cos(orientation)])
        thrust_force = thrust_direction * thrust_magnitude
        gravity_force = np.array([0, -self.gravity * self.mass])
        
        # Enhanced air resistance for better stability
        air_resistance = -0.03 * self.velocity * np.linalg.norm(self.velocity)
        
        # Total force
        total_force = thrust_force + gravity_force + air_resistance
        
        # Calculate acceleration
        acceleration = total_force / self.mass
        
        # Update velocity and position (Euler integration)
        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt
        
        # Near-ground forced vertical correction
        if self.position[1] < 50:
            # Force near-vertical orientation for final landing
            correction_rate = 0.2 if self.position[1] > 20 else 0.5
            self.orientation = self.orientation * (1 - correction_rate)
        
        # Check for landing/crash
        if self.position[1] <= 0:
            self.position[1] = 0
            
            # Landing conditions
            orientation_degrees = abs(self.orientation * 180/np.pi)
            if abs(self.velocity[1]) < 3.0 and abs(self.velocity[0]) < 2.0 and orientation_degrees < 5.0:
                # Successful landing
                self.velocity = np.array([0.0, 0.0])
                self.orientation = 0.0  # Perfectly vertical
                print("Successful landing!")
                print(f"Final position: ({self.position[0]:.2f}, 0.00)")
                print(f"Final orientation: {self.orientation * 180/np.pi:.2f} degrees (vertical)")
            else:
                # Crash
                self.velocity = np.array([0.0, 0.0])
                print(f"Crash landing! Velocity at impact: {np.linalg.norm(self.velocity):.2f} m/s")
                print(f"Orientation at impact: {self.orientation * 180/np.pi:.2f} degrees")
                print(f"Crash position: ({self.position[0]:.2f}, 0.00)")
        
        # Update time
        self.time += self.dt
        
        # Return True if landed
        return self.position[1] <= 0

def run_simulation():
    # Create rocket and run simulation
    rocket = Rocket()
    landed = False
    
    while not landed and rocket.time < 120:  # 2-minute simulation max
        landed = rocket.update()
    
    return rocket

def plot_simulation_steps(rocket, num_steps=8):
    """Plot the rocket at several points during its descent instead of animating"""
    positions = np.array(rocket.history['position'])
    orientations = np.array(rocket.history['orientation'])
    legs_deployed = np.array(rocket.history['legs_deployed'])
    thrusts = np.array(rocket.history['thrust'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate proper limits to show full trajectory
    x_min = min(positions[:, 0]) - 50
    x_max = max(positions[:, 0]) + 50
    if abs(x_min) > abs(x_max):
        x_max = abs(x_min)
    else:
        x_min = -abs(x_max)
    y_max = max(positions[:, 1]) + 50
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-10, y_max)
    
    # Draw landing pad
    landing_pad = patches.Rectangle((-20, -2), 40, 2, color='grey')
    ax.add_patch(landing_pad)
    
    # Draw full trajectory
    ax.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.5, label='Trajectory')
    
    # Calculate indices to plot
    total_frames = len(positions)
    indices = [int(i * total_frames / num_steps) for i in range(num_steps)]
    indices[-1] = min(indices[-1], total_frames - 1)  # Ensure we don't go out of bounds
    
    # Draw rocket at each selected point
    for idx in indices:
        pos = positions[idx]
        orient = orientations[idx]
        legs = legs_deployed[idx]
        thrust_val = thrusts[idx]
        
        # Skip positions below ground
        if pos[1] <= 0:
            continue
        
        # Create transform
        t = Affine2D()
        t.rotate(orient)
        t.translate(pos[0], pos[1])
        
        # Draw rocket body
        rocket_width = rocket.width
        rocket_height = rocket.height
        rocket_body = patches.Rectangle(
            (-rocket_width/2, -rocket_height/2), 
            rocket_width, rocket_height, 
            fill=True, color='gray', alpha=0.7
        )
        rocket_body.set_transform(t + ax.transData)
        ax.add_patch(rocket_body)
        
        # Draw legs if deployed
        if legs:
            leg_length = 10
            # Left leg
            left_leg = patches.Polygon([
                [-rocket_width/2, -rocket_height/2],
                [-rocket_width*2, -rocket_height/2-5],
                [-rocket_width/2, -rocket_height/2-5]
            ], closed=True, fill=True, color='black', alpha=0.7)
            left_leg.set_transform(t + ax.transData)
            ax.add_patch(left_leg)
            
            # Right leg
            right_leg = patches.Polygon([
                [rocket_width/2, -rocket_height/2],
                [rocket_width*2, -rocket_height/2-5],
                [rocket_width/2, -rocket_height/2-5]
            ], closed=True, fill=True, color='black', alpha=0.7)
            right_leg.set_transform(t + ax.transData)
            ax.add_patch(right_leg)
        
        # Draw thrust if present
        if thrust_val > 0:
            flame_length = (thrust_val / rocket.max_thrust) * 15
            exhaust = patches.Polygon([
                [-rocket_width/4, -rocket_height/2],
                [0, -rocket_height/2 - flame_length],
                [rocket_width/4, -rocket_height/2]
            ], closed=True, fill=True, color='orange', alpha=0.7)
            exhaust.set_transform(t + ax.transData)
            ax.add_patch(exhaust)
            
    # Add grid, labels, etc.
    ax.grid(True)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Rocket Landing Simulation')
    
    return fig

# Run simulation and plot
if __name__ == "__main__":
    print("Starting rocket landing simulation...")
    rocket = run_simulation()
    print(f"Simulation completed in {rocket.time:.2f} seconds")
    
    # Plot static representation at different time points
    fig = plot_simulation_steps(rocket, num_steps=12)
    plt.show()
