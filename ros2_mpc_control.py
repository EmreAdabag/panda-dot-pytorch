#!/usr/bin/env python3

import torch
import numpy as np
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# Import MPC components and Panda dynamics from local mpc.py
from mpcpanda import (
    PandaEETrackingMPCLayer,
    GradMethods,
)


class MPCControlNode(Node):
    def __init__(self):
        super().__init__('mpc_control_node')
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self.get_logger().info(f'Using device: {self.device}')
        
        # Timing parameters
        self.solve_timestep = 0.05
        
        # Robot parameters
        self.urdf_path = "/home/emrea/panda_pytorch/robot_description/panda_with_gripper.urdf"
        
        # Initialize MPC layer
        self.layer = PandaEETrackingMPCLayer(
            urdf_path=self.urdf_path,
            T=20,
            dt=self.solve_timestep,
            device=self.device,
            with_gravity=True,
            lqr_iter=2,
            eps=1e-1,
            verbose=0,
        ).to(self.device)
        
        self.n = self.layer.n_ctrl  # number of actuated DoF
        
        # Base initial configuration for the first 7 joints
        base_initial_q7 = [0.0, -0.8, 0.0, -np.pi / 2, 0.0, 0.5, np.pi / 4]
        
        # Build initial_q of length n
        self.initial_q = torch.zeros(self.n, device=self.device)
        for i in range(min(7, self.n)):
            self.initial_q[i] = base_initial_q7[i]
        
        # Sine wave parameters for cyclic motion
        self.sine_amplitudes = torch.tensor([0.3, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0], dtype=torch.get_default_dtype(), device=self.device)
        self.sine_frequencies = torch.tensor([0.3, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0], dtype=torch.get_default_dtype(), device=self.device)  # Hz
        self.sine_offsets = torch.tensor([0.0, 0.0, 0.0, -0.2, 0.0, 0.0, 0.0], dtype=torch.get_default_dtype(), device=self.device)
        
        # Weights for cost terms
        self.v_weight = torch.tensor(1e-2, dtype=torch.get_default_dtype(), device=self.device)
        self.u_weight = torch.tensor(1e-9, dtype=torch.get_default_dtype(), device=self.device)
        self.q_weight = torch.tensor(4.0, dtype=torch.get_default_dtype(), device=self.device)
        
        # Current state
        self.current_q = self.initial_q.clone().detach()
        self.current_qdot = torch.zeros(self.n, device=self.device)
        self.joint_state_received = False
        
        # Step counter for time tracking
        self.step = 0
        
        # ROS2 setup
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.joint_command_pub = self.create_publisher(
            JointState,
            '/joint_commands',
            10
        )
        
        # Control timer
        self.control_timer = self.create_timer(self.solve_timestep, self.control_callback)
        
        self.get_logger().info('MPC Control Node initialized')
    
    def joint_state_callback(self, msg):
        """Callback for joint state messages"""
        if len(msg.position) >= self.n:
            # Update current joint positions and velocities
            for i in range(self.n):
                self.current_q[i] = msg.position[i]
                if len(msg.velocity) > i:
                    self.current_qdot[i] = msg.velocity[i]
                else:
                    self.current_qdot[i] = 0.0
            
            self.joint_state_received = True
    
    def control_callback(self):
        """Main control loop callback"""
        if not self.joint_state_received:
            self.get_logger().warn('No joint state received yet')
            return
        
        s = time.monotonic()
        
        # Generate sine wave targets for current time
        current_time = self.step * self.solve_timestep
        q_target = self.initial_q.clone()
        for i in range(min(7, self.n)):
            if self.sine_frequencies[i] > 0:  # Only apply sine wave if frequency > 0
                q_target[i] = self.initial_q[i] + self.sine_amplitudes[i] * torch.sin(2 * np.pi * self.sine_frequencies[i] * current_time) + self.sine_offsets[i]
        
        # Create single goal (K=1) for sine wave tracking
        joint_goals = q_target.unsqueeze(0)  # [K=1, n]
        jg_BKn = joint_goals.unsqueeze(0)  # [B=1, K=1, n]
        goal_timesteps = torch.tensor([10], dtype=torch.long, device=self.device)  # Fixed horizon
        
        # Current state
        x_init = torch.cat([self.current_q, self.current_qdot]).unsqueeze(0)  # [B=1, n_state]
        
        # Solve MPC via differentiable layer
        x_mpc, u_mpc, obj = self.layer(
            x_init,
            jg_BKn,
            goal_timesteps,
            self.q_weight,
            self.v_weight,
            self.u_weight,
        )
        
        elapsed = time.monotonic() - s
        self.get_logger().info(f"MPC solve time: {elapsed:.4f}s")
        
        # Extract position command (using position at timestep 4 as in original)
        x_cmd = x_mpc[0, 4].detach().cpu().numpy()
        
        # Publish joint command
        cmd_msg = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.position = x_cmd[:self.n].tolist()
        self.joint_command_pub.publish(cmd_msg)
        
        self.step += 1


def main(args=None):
    rclpy.init(args=args)
    
    node = MPCControlNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
