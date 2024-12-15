import numpy as np
from simulator import Simulator
from pathlib import Path
from typing import Dict
import pinocchio as pin
import os
import scipy
import matplotlib.pyplot as plt

# Load the robot model from scene XML
current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
model = pin.buildModelFromMJCF(xml_path)
data = model.createData()

traj = False


def plot_results(times: np.ndarray, positions: np.ndarray, velocities: np.ndarray, control: np.ndarray):
    """Plot and save simulation results."""
    # Joint positions plot
    plt.figure(figsize=(10, 6))

    for i in range(positions.shape[1]):
        plt.plot(times, positions[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Positions [rad]')
    plt.title('Joint Positions over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/hw2_positions_trajectory.png')
    plt.close()
    
    # Joint velocities plot
    plt.figure(figsize=(10, 6))
    for i in range(velocities.shape[1]):
        plt.plot(times, velocities[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Velocities [rad/s]')
    plt.title('Joint Velocities over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/hw2_velocities_trajectory.png')
    plt.close()

    # Joint controls plot
    plt.figure(figsize=(10, 6))
    for i in range(control.shape[1]):
        plt.plot(times, control[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint control signals')
    plt.title('Joint control signals over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/hw2_controls_trajectory.png')
    plt.close()


def task_space_controller(q: np.ndarray, dq: np.ndarray, t: float, desired: Dict) -> np.ndarray:
    """Example task space controller."""

    def so3_error(Rd, R):
        """Compute orientation error using matrix logarithm"""
        error_matrix = Rd @ R.T
        error_log = scipy.linalg.logm(error_matrix)
        error_vector = skew_to_vector(error_log)
        return error_vector

    def skew_to_vector(skew_matrix):
        """Extract the vector from a skew-symmetric matrix"""
        return np.array([skew_matrix[2, 1],
                         skew_matrix[0, 2],
                         skew_matrix[1, 0]])
    
    def circular_trajectory(t, center, radius=0.2, w = 2 * np.pi):
        x_c, y_c, z_c = center

        position = np.array([x_c + radius * np.cos(w * t),
                             y_c,
                             z_c + radius * np.sin(w * t)])

        velocity = np.array([-radius * w * np.sin(w * t),
                             0,
                             radius * w * np.cos(w * t),
                             0,
                             0,
                             0])

        acceleration = np.array([-radius * w**2 * np.cos(w * t),
                                 0,
                                 -radius * w**2 * np.sin(w * t),
                                 0,
                                 0,
                                 0])

        return position, velocity, acceleration


    # Gains
    kp = np.diag([100, 100, 100, 100, 100, 100])
    kd = 2 * np.sqrt(kp)

    if traj:
        desired_position, desired_vel, desired_acc = circular_trajectory(t, [0.4, 0.3, 0.5])
        desired_rotation = pin.utils.rpyToMatrix(0, 0, 0)

    else:
        # Convert desired pose to SE3
        desired_position = desired['pos']
        desired_quaternion = desired['quat'] # [w, x, y, z] in MuJoCo format
        desired_quaternion_pin = np.array([*desired_quaternion[1:], desired_quaternion[0]]) # Convert to [x,y,z,w] for Pinocchio

        # Convert to pose and SE3
        desired_pose = np.concatenate([desired_position, desired_quaternion_pin])
        desired_rotation = pin.XYZQUATToSE3(desired_pose).rotation

        desired_vel = np.zeros(6)
        desired_acc = np.zeros(6)
    
    # Compute all dynamics quantities at once
    pin.computeAllTerms(model, data, q, dq)

    # Get end-effector frame ID
    ee_frame_id = model.getFrameId("end_effector")
    frame = pin.LOCAL

    # Calculate kinematics of frames
    pin.updateFramePlacement(model, data, ee_frame_id)

    ee_pose = data.oMf[ee_frame_id]
    current_position = ee_pose.translation
    #print('Translation\n', current_position)
    current_rotation = ee_pose.rotation
    #print('Orientation\n', current_rotation)

    # Get velocities and accelerations
    twist = pin.getFrameVelocity(model, data, ee_frame_id, frame)
    current_vel = np.hstack((twist.linear, twist.angular))

    # Transform desired velocity to end-effector frame
    desired_vel = ee_pose.actInv(pin.Motion(desired_vel))
    desired_vel = np.hstack([desired_vel.linear, desired_vel.angular])

    # Mass Matrix
    M = data.M

    # Nonlinear effects (Coriolis + gravity)
    nle = data.nle

    # Jacobian
    J = pin.getFrameJacobian(model, data, ee_frame_id, frame)

    # Derriative Jacobian
    pin.computeJointJacobiansTimeVariation(model, data, q, dq)
    dJ = pin.getFrameJacobianTimeVariation(model, data, ee_frame_id, frame)

    # Errors
    error_position = desired_position - current_position
    error_rotation = so3_error(desired_rotation, current_rotation)

    error_general = np.hstack((error_position, error_rotation))
    print('General error\n', error_general)

    error_vel = desired_vel - current_vel

    # Control
    acc = kp @ error_general + kd @ error_vel + desired_acc

    ddq_des = np.linalg.pinv(J) @ (acc - dJ @ dq)

    tau = M @ ddq_des + nle
    return tau

def main():
    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    
    print("\nRunning task space controller...")
    sim = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        enable_task_space=True,
        show_viewer=True,
        record_video=True,
        video_path="logs/videos/hw2_trajectory.mp4",
        fps=30,
        width=1920,
        height=1080
    )
    sim.set_controller(task_space_controller)

    # # Simulation parameters
    # t = 0
    # dt = sim.dt
    # time_limit = 10.0
    
    # # Data collection
    # times = []
    # positions = []
    # velocities = []
    # control = []
    
    # while t < time_limit:
    #     state = sim.get_state()
    #     times.append(t)
    #     positions.append(state['q'])
    #     velocities.append(state['dq'])
        
    #     tau = task_space_controller(q=state['q'], dq=state['dq'], t=t, desired=state['desired'])
    #     control.append(tau)
    #     sim.step(tau)
        
    #     if sim.record_video and len(sim.frames) < sim.fps * t:
    #         sim.frames.append(sim._capture_frame())
    #     t += dt
    
    # # Process and save results
    # times = np.array(times)
    # positions = np.array(positions)
    # velocities = np.array(velocities)
    # control = np.array(control)
    # plot_results(times, positions, velocities, control)

    sim.run(time_limit=10.0)

if __name__ == "__main__":
    main() 