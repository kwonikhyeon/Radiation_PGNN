#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────
# test/modules/rrt_path_planner.py - RRT-based Path Planning Module
# ──────────────────────────────────────────────────────────────
"""
RRT (Rapidly-exploring Random Tree) based path planning module for
radiation field exploration. Provides n-step path planning with
global exploration capability and obstacle avoidance.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from scipy.spatial.distance import euclidean
import random
import math

__all__ = [
    "RRTPathPlanner", "RRTNode", "RRTParameters", "PathPlanningResult"
]


@dataclass
class RRTParameters:
    """Parameters for RRT path planning."""

    # Basic RRT parameters
    max_iterations: int = 1000          # Maximum RRT iterations
    step_size_meters: float = 1.0       # Step size in meters
    min_step_size_meters: float = 0.5   # Minimum step size in meters
    goal_bias: float = 0.05             # Reduced probability of sampling toward goal for more exploration
    start_bias_reduction: float = 0.7   # Reduce sampling near start (0.0=no reduction, 1.0=full reduction)
    directional_spread: float = 2.0     # Encourage directional diversity in sampling

    # Path planning parameters
    n_steps: int = 5                    # Number of steps in planned path
    exploration_radius_meters: float = 4.0  # Increased exploration radius for more diverse paths
    branch_factor: int = 7              # Number of branches to explore from each node

    # Safety and constraints
    min_radiation_threshold: float = 0.8    # Avoid high radiation areas
    obstacle_avoidance: bool = True          # Enable obstacle avoidance
    source_avoidance: bool = False           # Avoid predicted source locations during planning
    post_process_source_avoidance: bool = True  # Add detour points after planning to avoid sources
    source_avoidance_radius_meters: float = 1.0  # Base avoidance radius around sources
    dynamic_avoidance: bool = True               # Use dynamic avoidance based on source intensity
    intensity_scale_factor: float = 2.0          # Scale factor for intensity-based radius calculation
    min_avoidance_radius_meters: float = 0.5     # Minimum avoidance radius
    max_avoidance_radius_meters: float = 3.0     # Maximum avoidance radius

    # Weighted map integration
    use_weighted_map: bool = True            # Use weighted map for path scoring
    reward_weight: float = 1.0               # Weight for path reward calculation
    safety_weight: float = 0.5               # Weight for safety in path evaluation

    # RRT tree optimization
    max_tree_nodes: int = 500                # Maximum nodes in RRT tree
    rewire_radius_meters: float = 2.0        # Radius for RRT* rewiring
    use_rrt_star: bool = True                # Use RRT* optimization

    # Termination criteria
    max_planning_time_seconds: float = 3.0   # Maximum planning time
    convergence_threshold: float = 0.01      # Minimum improvement to continue
    convergence_window: int = 50             # Window size for convergence check
    quality_threshold: float = None          # Auto-set based on weighted map
    min_candidate_paths: int = 10            # Minimum paths to evaluate before early termination


class RRTNode:
    """Node in the RRT tree."""

    def __init__(self, position: Tuple[int, int], parent: Optional['RRTNode'] = None):
        self.position = position  # (y, x) in pixels
        self.parent = parent
        self.children: List['RRTNode'] = []
        self.cost = 0.0  # Cost from start
        self.reward = 0.0  # Reward for reaching this node

        if parent is not None:
            parent.children.append(self)
            self.cost = parent.cost + self._calculate_distance_cost(parent.position, position)

    def _calculate_distance_cost(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance cost between two positions."""
        return euclidean(pos1, pos2)

    def get_path_to_root(self) -> List[Tuple[int, int]]:
        """Get path from this node back to root."""
        path = []
        current = self
        while current is not None:
            path.append(current.position)
            current = current.parent
        return path[::-1]  # Reverse to get root-to-node path


@dataclass
class PathPlanningResult:
    """Result of RRT path planning."""

    path: List[Tuple[int, int]]         # Planned path as list of (y, x) positions
    tree_nodes: List[RRTNode]           # All nodes in the RRT tree
    total_reward: float                 # Total reward of the planned path
    total_cost: float                   # Total cost of the planned path
    planning_time: float                # Time taken for planning (seconds)
    success: bool                       # Whether planning succeeded
    metadata: Dict[str, Any]            # Additional planning metadata


class RRTPathPlanner:
    """RRT-based path planner for radiation field exploration."""

    def __init__(self, parameters: Optional[RRTParameters] = None):
        self.params = parameters or RRTParameters()
        self.field_size_pixels = 256
        self.field_size_meters = 10.0
        self.pixels_per_meter = self.field_size_pixels / self.field_size_meters

        # RRT tree
        self.tree_nodes: List[RRTNode] = []
        self.root_node: Optional[RRTNode] = None

        # Environment data
        self.radiation_field: Optional[np.ndarray] = None
        self.weighted_map: Optional[np.ndarray] = None
        self.obstacle_map: Optional[np.ndarray] = None
        self.predicted_sources: Optional[List[Tuple[int, int, float]]] = None  # List of (y, x, intensity) source info

    def set_environment(self, radiation_field: np.ndarray,
                       weighted_map: Optional[np.ndarray] = None,
                       obstacle_map: Optional[np.ndarray] = None,
                       predicted_sources: Optional[List[Tuple[int, int, float]]] = None):
        """Set the environment for path planning.

        Args:
            radiation_field: 2D radiation field array
            weighted_map: Optional weighted map for path scoring
            obstacle_map: Optional obstacle map for avoidance
            predicted_sources: Optional list of (y, x, intensity) tuples for sources
        """
        self.radiation_field = radiation_field
        self.weighted_map = weighted_map
        self.obstacle_map = obstacle_map
        self.predicted_sources = predicted_sources

    def plan_path(self, start_position: Tuple[int, int],
                  measurement_positions: Optional[List[Tuple[int, int]]] = None) -> PathPlanningResult:
        """
        Plan an n-step path using RRT.

        Args:
            start_position: Starting position (y, x) in pixels
            measurement_positions: Previously measured positions to avoid

        Returns:
            PathPlanningResult containing the planned path and metadata
        """
        import time
        start_time = time.time()

        # Initialize tree
        self.tree_nodes = []
        self.root_node = RRTNode(start_position)
        self.tree_nodes.append(self.root_node)

        # Set up constraints
        measured_positions = set(measurement_positions or [])

        # RRT exploration
        best_path = None
        best_score = -float('inf')

        # Initialize direction tracking
        if not hasattr(self, '_explored_directions'):
            self._explored_directions = []

        # Initialize termination tracking
        score_history = []
        candidate_paths_found = 0
        last_improvement_iteration = 0
        quality_threshold = self._calculate_quality_threshold()

        for iteration in range(self.params.max_iterations):
            # Check time-based termination
            current_time = time.time()
            if current_time - start_time > self.params.max_planning_time_seconds:
                print(f"⏰ Terminating due to time limit: {current_time - start_time:.2f}s")
                break
            # Generate multiple samples per iteration to increase branching
            samples_this_iteration = min(self.params.branch_factor,
                                       max(1, self.params.max_iterations // 100))  # Adaptive sampling

            for sample_idx in range(samples_this_iteration):
                # Sample random point or goal-biased point with reduced goal bias
                if random.random() < self.params.goal_bias:
                    sample_point = self._sample_goal_biased(start_position)
                else:
                    sample_point = self._sample_random(start_position)

                # Find nearest node in tree
                nearest_node = self._find_nearest_node(sample_point)

                # Extend tree toward sample
                new_node = self._extend_tree(nearest_node, sample_point, measured_positions)

                if new_node is not None:
                    self.tree_nodes.append(new_node)

                    # RRT* rewiring
                    if self.params.use_rrt_star:
                        self._rewire_tree(new_node)

                    # Check if we have a promising path
                    if len(self.tree_nodes) >= self.params.n_steps:
                        candidate_path = self._extract_best_n_step_path()
                        if candidate_path is not None:
                            score = self._evaluate_path(candidate_path)
                            candidate_paths_found += 1

                            if score > best_score:
                                improvement = score - best_score
                                best_score = score
                                best_path = candidate_path
                                last_improvement_iteration = iteration

                                # Log significant improvements
                                if improvement > self.params.convergence_threshold:
                                    print(f"📈 Iteration {iteration}: Score improved by {improvement:.3f} to {score:.3f}")

                            # Track score history for convergence analysis
                            score_history.append(score)
                            if len(score_history) > self.params.convergence_window:
                                score_history.pop(0)

            # Apply intelligent termination criteria
            should_terminate, termination_reason = self._should_terminate(
                iteration, best_score, score_history, candidate_paths_found,
                last_improvement_iteration, quality_threshold
            )

            if should_terminate:
                print(f"🏁 Terminating: {termination_reason}")
                break

        planning_time = time.time() - start_time

        # Create result
        if best_path is not None:
            total_cost = self._calculate_path_cost(best_path)
            total_reward = self._calculate_path_reward(best_path)

            result = PathPlanningResult(
                path=best_path,
                tree_nodes=self.tree_nodes.copy(),
                total_reward=total_reward,
                total_cost=total_cost,
                planning_time=planning_time,
                success=True,
                metadata={
                    'iterations': iteration + 1,
                    'tree_size': len(self.tree_nodes),
                    'path_score': best_score,
                    'parameters': self.params
                }
            )
        else:
            # Fallback: use greedy path from current approach
            result = self._fallback_planning(start_position, measured_positions, planning_time)

        return result

    def _sample_random(self, start_position: Tuple[int, int]) -> Tuple[int, int]:
        """Sample a random point with reduced start bias and directional diversity."""
        start_y, start_x = start_position

        # Convert exploration radius to pixels
        radius_pixels = self.params.exploration_radius_meters * self.pixels_per_meter

        # Apply start bias reduction: prefer samples farther from start
        min_radius_pixels = radius_pixels * self.params.start_bias_reduction

        # Sample point in annular region (ring) around start for better exploration
        while True:
            angle = random.uniform(0, 2 * math.pi)
            # Use bias toward outer ring to explore farther from start
            radius = random.uniform(min_radius_pixels, radius_pixels)

            # Add directional spreading: bias toward less explored directions
            if hasattr(self, '_explored_directions'):
                # Adjust angle to favor unexplored directions
                angle = self._adjust_angle_for_diversity(angle)
            else:
                self._explored_directions = []

            sample_x = int(start_x + radius * math.cos(angle))
            sample_y = int(start_y + radius * math.sin(angle))

            # Track explored directions for diversity
            self._explored_directions.append(angle)

            # Check bounds
            if (0 <= sample_y < self.field_size_pixels and
                0 <= sample_x < self.field_size_pixels):
                return (sample_y, sample_x)

    def _adjust_angle_for_diversity(self, angle: float) -> float:
        """Adjust sampling angle to promote directional diversity."""
        if not self._explored_directions:
            return angle

        # Find the least explored direction
        angle_bins = 16  # Divide circle into bins
        bin_size = (2 * math.pi) / angle_bins
        bin_counts = [0] * angle_bins

        # Count samples in each directional bin
        for explored_angle in self._explored_directions[-50:]:  # Use recent history
            bin_idx = int((explored_angle % (2 * math.pi)) / bin_size)
            bin_counts[bin_idx] += 1

        # Find least explored bin
        min_count = min(bin_counts)
        least_explored_bins = [i for i, count in enumerate(bin_counts) if count == min_count]

        # Bias toward least explored direction with some randomness
        if random.random() < self.params.directional_spread / 3.0:  # Apply diversity bias
            target_bin = random.choice(least_explored_bins)
            target_angle = (target_bin + 0.5) * bin_size
            # Blend with original angle
            blend_factor = 0.7
            angle = (1 - blend_factor) * angle + blend_factor * target_angle

        return angle

    def _sample_goal_biased(self, start_position: Tuple[int, int]) -> Tuple[int, int]:
        """Sample a point biased toward high-value areas."""
        if self.weighted_map is not None:
            # Sample from weighted map distribution
            flat_weights = self.weighted_map.flatten()
            flat_weights = np.maximum(flat_weights, 0)  # Ensure non-negative

            if flat_weights.sum() > 0:
                # Normalize to probability distribution
                probabilities = flat_weights / flat_weights.sum()

                # Sample index
                idx = np.random.choice(len(probabilities), p=probabilities)
                y, x = np.unravel_index(idx, self.weighted_map.shape)

                return (int(y), int(x))

        # Fallback to random sampling
        return self._sample_random(start_position)

    def _find_nearest_node(self, sample_point: Tuple[int, int]) -> RRTNode:
        """Find nearest node in tree to sample point."""
        min_distance = float('inf')
        nearest_node = self.root_node

        for node in self.tree_nodes:
            distance = euclidean(node.position, sample_point)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node

        return nearest_node

    def _extend_tree(self, nearest_node: RRTNode, sample_point: Tuple[int, int],
                    measured_positions: set) -> Optional[RRTNode]:
        """Extend tree from nearest node toward sample point."""
        # Calculate direction and distance
        dx = sample_point[1] - nearest_node.position[1]
        dy = sample_point[0] - nearest_node.position[0]
        distance = math.sqrt(dx*dx + dy*dy)

        if distance == 0:
            return None

        # Step size in pixels (ensure minimum step size)
        min_step_pixels = self.params.min_step_size_meters * self.pixels_per_meter
        step_pixels = self.params.step_size_meters * self.pixels_per_meter

        # Ensure minimum step size is met
        if distance < min_step_pixels:
            # If target is too close, extend to minimum distance
            step_ratio = min_step_pixels / distance
        else:
            # Normal step size
            step_ratio = min(step_pixels / distance, 1.0)

        new_x = int(nearest_node.position[1] + dx * step_ratio)
        new_y = int(nearest_node.position[0] + dy * step_ratio)
        new_position = (new_y, new_x)

        # Verify minimum distance constraint
        actual_distance = math.sqrt((new_x - nearest_node.position[1])**2 +
                                  (new_y - nearest_node.position[0])**2)
        actual_distance_meters = actual_distance / self.pixels_per_meter

        if actual_distance_meters < self.params.min_step_size_meters * 0.9:  # Allow 10% tolerance
            # print(f"⚠️  Skipping step: {actual_distance_meters:.2f}m < {self.params.min_step_size_meters:.2f}m")
            return None  # Skip if step is too small

        # Check constraints
        if not self._is_valid_position(new_position, measured_positions):
            return None

        # Create new node
        new_node = RRTNode(new_position, nearest_node)

        # Calculate reward for this position
        if self.weighted_map is not None:
            new_node.reward = self.weighted_map[new_y, new_x]

        return new_node

    def _is_valid_position(self, position: Tuple[int, int],
                          measured_positions: set) -> bool:
        """Check if position is valid (within bounds, not measured, safe)."""
        y, x = position

        # Check bounds
        if not (0 <= y < self.field_size_pixels and 0 <= x < self.field_size_pixels):
            return False

        # Check if already measured
        if position in measured_positions:
            return False

        # Check obstacle map
        if self.obstacle_map is not None and self.obstacle_map[y, x] > 0:
            return False

        # Check radiation safety
        if (self.radiation_field is not None and
            self.radiation_field[y, x] > self.params.min_radiation_threshold):
            return False

        # Check source avoidance
        if (self.params.source_avoidance and
            self.predicted_sources is not None and
            self._is_too_close_to_sources(position)):
            return False

        return True

    def _calculate_dynamic_avoidance_radius(self, intensity: float) -> float:
        """Calculate dynamic avoidance radius based on source intensity.

        Args:
            intensity: Source intensity (0.0 to 1.0)

        Returns:
            avoidance_radius_meters: Dynamic avoidance radius in meters
        """
        if not self.params.dynamic_avoidance:
            return self.params.source_avoidance_radius_meters

        # Calculate radius based on intensity: stronger sources need larger avoidance
        # Formula: base_radius + (intensity * scale_factor)
        dynamic_radius = (self.params.source_avoidance_radius_meters +
                         intensity * self.params.intensity_scale_factor)

        # Clamp to min/max bounds
        return max(self.params.min_avoidance_radius_meters,
                  min(dynamic_radius, self.params.max_avoidance_radius_meters))

    def _is_too_close_to_sources(self, position: Tuple[int, int]) -> bool:
        """Check if position is too close to any predicted source with dynamic radius."""
        if not self.predicted_sources:
            return False

        y, x = position

        for source_data in self.predicted_sources:
            # Handle both old format (y, x) and new format (y, x, intensity)
            if len(source_data) == 2:
                source_y, source_x = source_data
                intensity = 1.0  # Default intensity for backward compatibility
            else:
                source_y, source_x, intensity = source_data

            # Calculate dynamic avoidance radius
            avoidance_radius_meters = self._calculate_dynamic_avoidance_radius(intensity)
            avoidance_radius_pixels = avoidance_radius_meters * self.pixels_per_meter

            distance_pixels = math.sqrt((y - source_y)**2 + (x - source_x)**2)
            if distance_pixels < avoidance_radius_pixels:
                return True

        return False

    def _line_intersects_source(self, start: Tuple[int, int], end: Tuple[int, int],
                               source_pos: Tuple[int, int], source_intensity: float) -> bool:
        """Check if line segment intersects with source avoidance area.

        Args:
            start: Start point of line segment
            end: End point of line segment
            source_pos: Source position (y, x)
            source_intensity: Source intensity for dynamic radius calculation

        Returns:
            True if line intersects source avoidance area
        """
        if len(source_pos) == 2:
            source_y, source_x = source_pos
            intensity = source_intensity
        else:
            source_y, source_x, intensity = source_pos

        # Calculate dynamic avoidance radius
        avoidance_radius_meters = self._calculate_dynamic_avoidance_radius(intensity)
        avoidance_radius_pixels = avoidance_radius_meters * self.pixels_per_meter

        # Calculate distance from source to line segment
        x1, y1 = start[1], start[0]  # Convert to x,y coordinates
        x2, y2 = end[1], end[0]
        x0, y0 = source_x, source_y

        # Line segment vector
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            # Start and end are the same point
            distance = math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
            return distance < avoidance_radius_pixels

        # Parameter t for closest point on line segment
        t = max(0, min(1, ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)))

        # Closest point on line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        # Distance from source to closest point
        distance = math.sqrt((x0 - closest_x)**2 + (y0 - closest_y)**2)

        return distance < avoidance_radius_pixels

    def _find_detour_point(self, start: Tuple[int, int], end: Tuple[int, int],
                          source_pos: Tuple[int, int], source_intensity: float) -> Tuple[int, int]:
        """Find a detour point that goes around the source.

        Args:
            start: Start point of original segment
            end: End point of original segment
            source_pos: Source position
            source_intensity: Source intensity

        Returns:
            Detour point coordinates (y, x)
        """
        if len(source_pos) == 2:
            source_y, source_x = source_pos
            intensity = source_intensity
        else:
            source_y, source_x, intensity = source_pos

        # Calculate dynamic avoidance radius with safety margin
        avoidance_radius_meters = self._calculate_dynamic_avoidance_radius(intensity)
        safety_margin_meters = 0.5  # Additional safety margin
        detour_radius_meters = avoidance_radius_meters + safety_margin_meters
        detour_radius_pixels = detour_radius_meters * self.pixels_per_meter

        # Vector from start to end
        start_x, start_y = start[1], start[0]
        end_x, end_y = end[1], end[0]

        # Vector perpendicular to start-end line
        line_dx = end_x - start_x
        line_dy = end_y - start_y
        line_length = math.sqrt(line_dx**2 + line_dy**2)

        if line_length == 0:
            return start

        # Normalize line vector
        line_dx /= line_length
        line_dy /= line_length

        # Perpendicular vector (rotate 90 degrees)
        perp_dx = -line_dy
        perp_dy = line_dx

        # Try both sides of the source to find the better detour
        detour_candidates = []

        for direction in [1, -1]:
            # Calculate detour point
            detour_x = source_x + direction * perp_dx * detour_radius_pixels
            detour_y = source_y + direction * perp_dy * detour_radius_pixels

            # Ensure detour point is within bounds
            H, W = self.radiation_field.shape
            detour_x = max(0, min(W-1, detour_x))
            detour_y = max(0, min(H-1, detour_y))

            # Calculate total detour distance
            dist_start_detour = math.sqrt((start_x - detour_x)**2 + (start_y - detour_y)**2)
            dist_detour_end = math.sqrt((detour_x - end_x)**2 + (detour_y - end_y)**2)
            total_distance = dist_start_detour + dist_detour_end

            detour_candidates.append(((int(detour_y), int(detour_x)), total_distance))

        # Choose the detour with shorter total distance
        best_detour = min(detour_candidates, key=lambda x: x[1])
        return best_detour[0]

    def _post_process_path_for_sources(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Post-process path to add detour points around sources.

        Args:
            path: Original path points

        Returns:
            Modified path with detour points inserted
        """
        if not self.params.post_process_source_avoidance or not self.predicted_sources:
            return path

        if len(path) < 2:
            return path

        processed_path = [path[0]]  # Start with first point

        for i in range(len(path) - 1):
            current_point = path[i]
            next_point = path[i + 1]

            # Check if this segment intersects any source
            intersecting_sources = []

            for source_data in self.predicted_sources:
                if len(source_data) == 2:
                    source_pos = source_data
                    intensity = 1.0  # Default intensity
                else:
                    source_pos = source_data[:2]
                    intensity = source_data[2]

                if self._line_intersects_source(current_point, next_point, source_pos, intensity):
                    intersecting_sources.append((source_pos, intensity))

            # If segment intersects sources, add detour points
            if intersecting_sources:
                # For multiple intersecting sources, handle the strongest one first
                intersecting_sources.sort(key=lambda x: x[1], reverse=True)

                for source_pos, intensity in intersecting_sources:
                    detour_point = self._find_detour_point(current_point, next_point, source_pos, intensity)
                    processed_path.append(detour_point)
                    print(f"🔄 Added detour point at ({detour_point[1]/self.pixels_per_meter:.1f}m, {detour_point[0]/self.pixels_per_meter:.1f}m) to avoid source at ({source_pos[1]/self.pixels_per_meter:.1f}m, {source_pos[0]/self.pixels_per_meter:.1f}m)")

            processed_path.append(next_point)

        return processed_path

    def _calculate_quality_threshold(self) -> float:
        """Calculate adaptive quality threshold based on weighted map statistics."""
        if self.weighted_map is None:
            return 0.0

        # Analyze weighted map to set reasonable quality expectations
        map_mean = float(np.mean(self.weighted_map))
        map_max = float(np.max(self.weighted_map))
        map_std = float(np.std(self.weighted_map))

        # Set threshold as mean + 1.5 * std for n-step path
        # This represents a "good" path that's significantly above average
        expected_good_score = (map_mean + 1.5 * map_std) * self.params.n_steps

        print(f"🎯 Quality threshold set to {expected_good_score:.3f} "
              f"(map stats: mean={map_mean:.3f}, max={map_max:.3f}, std={map_std:.3f})")

        return expected_good_score

    def _should_terminate(self, iteration: int, best_score: float, score_history: list,
                        candidate_paths_found: int, last_improvement_iteration: int,
                        quality_threshold: float) -> tuple[bool, str]:
        """Determine if RRT should terminate and provide reason."""

        # 1. Tree size limit (safety)
        if len(self.tree_nodes) > self.params.max_tree_nodes:
            return True, f"Max tree size reached ({len(self.tree_nodes)} nodes)"

        # 2. Quality threshold reached
        if quality_threshold > 0 and best_score >= quality_threshold:
            return True, f"Quality threshold reached (score: {best_score:.3f} >= {quality_threshold:.3f})"

        # 3. Convergence check (need sufficient history)
        if len(score_history) >= self.params.convergence_window:
            recent_scores = score_history[-self.params.convergence_window//2:]
            older_scores = score_history[-self.params.convergence_window:-self.params.convergence_window//2]

            if recent_scores and older_scores:
                recent_avg = sum(recent_scores) / len(recent_scores)
                older_avg = sum(older_scores) / len(older_scores)
                improvement_rate = recent_avg - older_avg

                if improvement_rate < self.params.convergence_threshold:
                    return True, f"Convergence detected (improvement: {improvement_rate:.4f} < {self.params.convergence_threshold})"

        # 4. No improvement for too long (stagnation)
        iterations_without_improvement = iteration - last_improvement_iteration
        max_stagnation = max(100, self.params.convergence_window * 2)

        if iterations_without_improvement > max_stagnation and candidate_paths_found >= self.params.min_candidate_paths:
            return True, f"Stagnation detected ({iterations_without_improvement} iterations without improvement)"

        # 5. Sufficient exploration with good result
        if (candidate_paths_found >= self.params.min_candidate_paths * 3 and
            best_score > 0 and
            iteration > 200):  # Minimum exploration
            return True, f"Sufficient exploration completed ({candidate_paths_found} paths evaluated)"

        return False, ""

    def _rewire_tree(self, new_node: RRTNode):
        """RRT* rewiring to optimize tree structure."""
        rewire_radius_pixels = self.params.rewire_radius_meters * self.pixels_per_meter

        # Find nodes within rewiring radius
        nearby_nodes = []
        for node in self.tree_nodes:
            if node != new_node:
                distance = euclidean(node.position, new_node.position)
                if distance <= rewire_radius_pixels:
                    nearby_nodes.append(node)

        # Try to improve path to new_node
        best_parent = new_node.parent
        best_cost = new_node.cost

        for node in nearby_nodes:
            potential_cost = node.cost + euclidean(node.position, new_node.position)
            if potential_cost < best_cost:
                best_parent = node
                best_cost = potential_cost

        # Rewire if better parent found
        if best_parent != new_node.parent:
            new_node.parent.children.remove(new_node)
            new_node.parent = best_parent
            new_node.cost = best_cost
            best_parent.children.append(new_node)

        # Try to improve paths through new_node
        for node in nearby_nodes:
            potential_cost = new_node.cost + euclidean(new_node.position, node.position)
            if potential_cost < node.cost:
                node.parent.children.remove(node)
                node.parent = new_node
                node.cost = potential_cost
                new_node.children.append(node)

    def _extract_best_n_step_path(self) -> Optional[List[Tuple[int, int]]]:
        """Extract best n-step path from current tree."""
        best_path = None
        best_score = -float('inf')

        # Try all possible n-step paths
        for leaf_node in self.tree_nodes:
            path = leaf_node.get_path_to_root()

            # Check if path has desired length
            if len(path) >= self.params.n_steps + 1:  # +1 for start position
                # Take first n steps plus start
                n_step_path = path[:self.params.n_steps + 1]

                # Remove duplicate consecutive positions
                filtered_path = [n_step_path[0]]
                for i in range(1, len(n_step_path)):
                    if n_step_path[i] != filtered_path[-1]:
                        filtered_path.append(n_step_path[i])

                # Ensure we have enough unique steps
                if len(filtered_path) >= 2:  # At least start + 1 step
                    score = self._evaluate_path(filtered_path)
                    if score > best_score:
                        best_score = score
                        best_path = filtered_path

        return best_path

    def _evaluate_path(self, path: List[Tuple[int, int]]) -> float:
        """Evaluate path quality using pure weighted map sum for n-step paths."""
        if not path:
            return -float('inf')

        # For n-step path evaluation, use pure weighted map sum
        # This ensures we select paths with maximum information gain
        total_reward = self._calculate_path_reward(path)

        # Optional: Add small penalty for extreme distances to avoid unrealistic paths
        # but keep it minimal to prioritize weighted map values
        total_cost = self._calculate_path_cost(path)
        distance_penalty = total_cost * 0.001  # Very small penalty

        # Pure reward-based scoring with minimal distance consideration
        score = total_reward - distance_penalty

        return score

    def _calculate_path_reward(self, path: List[Tuple[int, int]]) -> float:
        """Calculate total reward for path."""
        if self.weighted_map is None:
            return 0.0

        total_reward = 0.0
        for y, x in path:
            if 0 <= y < self.weighted_map.shape[0] and 0 <= x < self.weighted_map.shape[1]:
                total_reward += self.weighted_map[y, x]

        return total_reward

    def _calculate_path_cost(self, path: List[Tuple[int, int]]) -> float:
        """Calculate total Euclidean distance cost for path."""
        if len(path) < 2:
            return 0.0

        total_cost = 0.0
        for i in range(len(path) - 1):
            total_cost += euclidean(path[i], path[i + 1])

        return total_cost

    def _calculate_safety_penalty(self, path: List[Tuple[int, int]]) -> float:
        """Calculate safety penalty for path (higher radiation = higher penalty)."""
        if self.radiation_field is None:
            return 0.0

        total_penalty = 0.0
        for y, x in path:
            if 0 <= y < self.radiation_field.shape[0] and 0 <= x < self.radiation_field.shape[1]:
                radiation_level = self.radiation_field[y, x]
                if radiation_level > self.params.min_radiation_threshold:
                    # Exponential penalty for high radiation
                    total_penalty += np.exp(radiation_level * 5) - 1

        return total_penalty

    def _fallback_planning(self, start_position: Tuple[int, int],
                          measured_positions: set, planning_time: float) -> PathPlanningResult:
        """Fallback to greedy planning if RRT fails."""
        # Simple greedy path planning
        path = [start_position]
        current_pos = start_position

        for _ in range(self.params.n_steps):
            best_next = None
            best_score = -float('inf')

            # Generate candidates around current position
            for dy in range(-20, 21, 5):
                for dx in range(-20, 21, 5):
                    candidate = (current_pos[0] + dy, current_pos[1] + dx)

                    if self._is_valid_position(candidate, measured_positions):
                        score = 0.0
                        if self.weighted_map is not None:
                            score = self.weighted_map[candidate[0], candidate[1]]

                        if score > best_score:
                            best_score = score
                            best_next = candidate

            if best_next is not None:
                path.append(best_next)
                current_pos = best_next
                measured_positions.add(best_next)
            else:
                break

        # Post-process path to avoid sources if enabled
        if self.params.post_process_source_avoidance:
            original_path_length = len(path)
            path = self._post_process_path_for_sources(path)
            if len(path) > original_path_length:
                print(f"📍 Path post-processing: {original_path_length} → {len(path)} points (added {len(path) - original_path_length} detour points)")

        return PathPlanningResult(
            path=path,
            tree_nodes=self.tree_nodes.copy(),
            total_reward=self._calculate_path_reward(path),
            total_cost=self._calculate_path_cost(path),
            planning_time=planning_time,
            success=len(path) > 1,
            metadata={
                'method': 'fallback_greedy',
                'parameters': self.params
            }
        )

    def visualize_tree(self, ax: plt.Axes, path: Optional[List[Tuple[int, int]]] = None):
        """Visualize RRT tree and planned path."""
        # Draw tree edges
        for node in self.tree_nodes:
            if node.parent is not None:
                x1, y1 = node.parent.position[1], node.parent.position[0]
                x2, y2 = node.position[1], node.position[0]
                ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linewidth=0.5)

        # Draw tree nodes
        positions = [node.position for node in self.tree_nodes]
        if positions:
            positions_array = np.array(positions)
            ax.scatter(positions_array[:, 1], positions_array[:, 0],
                      c='lightblue', s=10, alpha=0.6, marker='.')

        # Draw planned path
        if path is not None and len(path) > 1:
            path_array = np.array(path)
            ax.plot(path_array[:, 1], path_array[:, 0], 'red', linewidth=3, alpha=0.8)
            ax.scatter(path_array[:, 1], path_array[:, 0],
                      c='red', s=30, marker='o', edgecolor='black', linewidth=1)

        # Mark start position
        if self.root_node is not None:
            start_x, start_y = self.root_node.position[1], self.root_node.position[0]
            ax.scatter(start_x, start_y, c='green', s=100, marker='s',
                      edgecolor='black', linewidth=2, label='Start')


# Convenience function for standalone use
def plan_rrt_path(start_position: Tuple[int, int],
                  radiation_field: np.ndarray,
                  weighted_map: Optional[np.ndarray] = None,
                  measurement_positions: Optional[List[Tuple[int, int]]] = None,
                  parameters: Optional[RRTParameters] = None) -> PathPlanningResult:
    """
    Standalone function for RRT path planning.

    Args:
        start_position: Starting position (y, x) in pixels
        radiation_field: Radiation field array
        weighted_map: Optional weighted map for path evaluation
        measurement_positions: Previously measured positions to avoid
        parameters: Optional RRT parameters

    Returns:
        PathPlanningResult containing planned path and metadata
    """
    planner = RRTPathPlanner(parameters)
    planner.set_environment(radiation_field, weighted_map)
    return planner.plan_path(start_position, measurement_positions)


if __name__ == "__main__":
    # Test the RRT path planner
    print("Testing RRT Path Planner...")

    # Create test environment
    field_size = 256
    radiation_field = np.random.random((field_size, field_size)) * 0.5

    # Add some high radiation areas (obstacles)
    radiation_field[100:120, 100:120] = 0.9  # High radiation zone

    # Create weighted map (inverse of radiation for safety)
    weighted_map = 1.0 - radiation_field

    # Test parameters
    params = RRTParameters(
        max_iterations=500,
        n_steps=5,
        exploration_radius_meters=3.0,
        use_rrt_star=True
    )

    # Plan path
    start_pos = (50, 50)
    result = plan_rrt_path(
        start_position=start_pos,
        radiation_field=radiation_field,
        weighted_map=weighted_map,
        parameters=params
    )

    # Print results
    print(f"Planning success: {result.success}")
    print(f"Path length: {len(result.path)}")
    print(f"Total reward: {result.total_reward:.3f}")
    print(f"Total cost: {result.total_cost:.3f}")
    print(f"Planning time: {result.planning_time:.3f}s")
    print(f"Tree size: {len(result.tree_nodes)}")

    if result.path:
        print("Planned path:")
        for i, pos in enumerate(result.path):
            print(f"  Step {i}: {pos}")

    # Visualize if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot radiation field
        im1 = ax1.imshow(radiation_field, cmap='hot', origin='lower')
        ax1.set_title('Radiation Field')
        plt.colorbar(im1, ax=ax1)

        # Plot RRT tree and path
        im2 = ax2.imshow(weighted_map, cmap='RdYlGn', origin='lower')
        ax2.set_title('RRT Tree and Planned Path')

        # Visualize tree
        planner = RRTPathPlanner(params)
        planner.set_environment(radiation_field, weighted_map)
        planner.tree_nodes = result.tree_nodes
        if result.tree_nodes:
            planner.root_node = result.tree_nodes[0]
        planner.visualize_tree(ax2, result.path)

        plt.colorbar(im2, ax=ax2)
        plt.tight_layout()
        plt.savefig('rrt_path_planning_test.png', dpi=150, bbox_inches='tight')
        print("Saved visualization to 'rrt_path_planning_test.png'")

    except ImportError:
        print("Matplotlib not available for visualization")

    print("RRT path planner test complete!")