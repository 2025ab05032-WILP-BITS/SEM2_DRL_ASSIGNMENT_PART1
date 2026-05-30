
#  Project: Autonomous Drone Rescue (DP Edition)

### *A mathematically optimized rescue mission in a disaster-hit city.*

This repository contains a full-stack Reinforcement Learning solution using **Dynamic Programming** to solve a complex navigation and resource-management problem. Based on the group configuration for **Group ID ending in 5**, the environment is a high-stakes 6x6 grid.

---

##  Requirement Traceability Matrix

This table maps the assignment requirements directly to the implementation in `DRL_part_2_dp_drone_rescue_Group225.ipynb`.

| Requirement Section | Specific Criteria | Implementation Detail in Code |
| --- | --- | --- |
| **1. Env Configuration** | 6x6 Grid, 3 Targets, 2 Charging Stations | `self.grid_size = 6`, `self.rescue_targets`, and `self.charging_stations` are initialized using Group 225 logic. |
| **2. Battery System** | Max Battery: 15 units (Odd Group ID) | `self.max_battery = 15`. Every `step()` subtracts `1` unit; battery refill logic is in the `C` cell check. |
| **3. Wind Dynamics** | 30% Stochastic Probability | The `get_transitions()` function applies a `0.30` probability to move the drone in a random direction if currently on a `W` cell. |
| **4. Hover Action** | +2 Battery on 'C', -1 elsewhere | In `step('hover')`, logic checks `if cell == 'C'`: `battery + 2`, else `battery - 1`. |
| **5. DP Solver** | Value Iteration ($10^{-3}$ threshold) | The `value_iteration()` function runs a `while delta > theta` loop, updating the Bellman equation across all reachable states. |
| **6. State Space** | `(x, y, battery, target_mask)` | States are mapped as a tuple of coordinates, current fuel, and a bitmask representing which civilians are saved. |

---

##  Technical Code Walkthrough

### Phase 1: Modeling the "Disaster World"

The environment is not just a static grid; it is a **Finite Markov Decision Process (MDP)**.

* **The State Space:** Unlike simple pathfinders, our state space is 4-dimensional. It doesn't just know *where* the drone is, but also *how much fuel* it has and *who* is left to save.
* **The Transition Function:** This is the "Physics" of our world. We implemented a transition matrix $P(s'|s,a)$ that accounts for the 30% wind interference, ensuring the drone learns to "hedge its bets" when flying near hazards.

### Phase 2: The "Brain" (Value Iteration)

We use the **Bellman Optimality Equation** to solve the grid:


$$V^*(s) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma V^*(s')]$$

**Why this works:** The drone doesn't just look for the closest rescue target. Because of the `-20` penalty for battery depletion, the Value Iteration forces the drone to "visualize" its future. If a rescue target is 5 steps away but the battery is at 4, the "Value" of those target-facing cells will drop, and the "Value" of the path to the Charging Station will rise.

### Phase 3: Strategic Rewards

We engineered a reward economy to guide behavior without "hard-coding" rules:

* **Rescue (+20):** High incentive to complete the mission.
* **Danger (-10):** Teaches the drone that fire/radiation is "expensive" but not necessarily mission-ending if a life can be saved.
* **Battery Death (-20):** A catastrophic penalty that forces the drone to respect the charging station.

---

##  Visualizing Success

The notebook generates a **Policy Map** and **Value Heatmap**.

* **The Heatmap:** Darker areas represent "Low Value" (Danger zones or low-battery death traps). Brighter areas represent the optimal path toward rescue.
* **The Policy:** Represented by arrows ($\rightarrow, \uparrow, \dots$). You will notice the drone "hugs" the walls to stay away from Wind zones ($W$) and curves its path toward the Charging Station ($C$) when the battery counter drops below a critical threshold.

---

##  Scalability

In the final section of the code, we discuss the **Curse of Dimensionality**.

* **The Math:** In our 6x6 grid, we have $36 \times 15 \times 2^3 = 4,320$ possible states.
* **The Problem:** If we move to a 100x100 city with 10 targets, the states jump to over $10^7$.
* **The Future:** This DP approach provides the *ground truth* optimal solution. For larger scales, we would transition to **Deep Q-Networks (DQN)**, using the logic established here as the training foundation.

---

##  How to Use

1. **Requirement:** Python 3.x, NumPy, Matplotlib.
2. **Run:** Open `DRL_part_2_dp_drone_rescue_Group225.ipynb`.
3. **Observe:** The Value Iteration loop will print the "Delta" error. Once it hits `< 0.001`, the optimal policy is locked in and ready for the rescue demo.

---

**Submitted by:** Vaibhav Bhandeo (Group 225)

**Course:** Deep Reinforcement Learning

**Date:** June 2026
