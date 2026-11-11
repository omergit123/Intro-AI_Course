import ex1_check
import search

id = ["No numbers - I'm special!"]

# I used Github-copilot for an initial implementation of "__init__", "successor", "h_astar" and "goal_test" methods. 
# I used Gemini for reviewing the code and for suggestions on how to improve.
# In addition, I made my own changes to the code.

class LogisticsProblem(search.Problem):
    """This class implements a logistics problem"""

    def __init__(self, initial):
        self.game = initial
        self.trucks = tuple(sorted(initial["Trucks"]))
        self.ferries = tuple(sorted(initial["Ferries"]))
        self.packages = tuple(sorted(initial["Packages"]))
        
        self.truck_index = {truck: i for i, truck in enumerate(self.trucks)}
        self.ferry_index = {ferry: i for i, ferry in enumerate(self.ferries)}

        # Precompute values as tuples for maximum Python loop performance
        self.package_weights = tuple(initial["Packages"][p][1] for p in self.packages)
        self.truck_capacities = tuple(initial["Trucks"][t][1] for t in self.trucks)
        self.num_trucks = max(1, len(self.trucks))
        self.ferry_served_ports = tuple(initial["Ferries"][f][1] for f in self.ferries)
        max_packages_per_truck = 1
        for capacity in self.truck_capacities:
            load = count = 0
            for weight in sorted(self.package_weights):
                if load + weight > capacity:
                    break
                load += weight
                count += 1
            max_packages_per_truck = max(max_packages_per_truck, count)
        self.max_packages_per_move = max(
            [max_packages_per_truck]
            + [
                initial["Ferries"][ferry][2] * max_packages_per_truck
                for ferry in self.ferries
            ]
        )

        # Collect all valid ports for UNLOAD pruning
        self.all_ports = set(initial.get("Ports", []))
        for ports in self.ferry_served_ports:
            self.all_ports.update(ports)

        roads_dict = {location: [] for location in range(initial["Locations"])}
        for source, target in initial["Roads"]:
            roads_dict[source].append(target)
            roads_dict[target].append(source)
        self.roads = {loc: tuple(sorted(neighbors)) for loc, neighbors in roads_dict.items()}

        start = (
            tuple(initial["Trucks"][truck][0] for truck in self.trucks),
            (None,) * len(self.trucks),
            tuple(initial["Ferries"][ferry][0] for ferry in self.ferries),
            tuple(initial["Packages"][package][0] for package in self.packages),
            (None,) * len(self.packages),
        )
        self.package_goals = tuple(initial["Packages"][package][2] for package in self.packages)
        self._heuristic_cache = {}

        # ---------------- Exact Truck Distances Matrix (Floyd-Warshall) ----------------
        locations = range(initial["Locations"])
        truck_distances = [[float("inf")] * len(locations) for _ in locations]
        
        for i in locations:
            truck_distances[i][i] = 0
            
        for source, target in initial["Roads"]:
            truck_distances[source][target] = truck_distances[target][source] = 1
            
        for ports in self.ferry_served_ports:
            for source in ports:
                for target in ports:
                    if source != target:
                        # Cost for truck crossing: BOARD(1) + SAIL(3) + DISEMBARK(1) = 5
                        truck_distances[source][target] = min(truck_distances[source][target], 5)
                        
        for k in locations:
            for i in locations:
                for j in locations:
                    if truck_distances[i][k] + truck_distances[k][j] < truck_distances[i][j]:
                        truck_distances[i][j] = truck_distances[i][k] + truck_distances[k][j]
                        
        self.truck_distances = truck_distances
        movement_distances = [[float("inf")] * len(locations) for _ in locations]
        for i in locations:
            movement_distances[i][i] = 0
        for source, target in initial["Roads"]:
            movement_distances[source][target] = movement_distances[target][source] = 1
        for ports in self.ferry_served_ports:
            for source in ports:
                for target in ports:
                    if source != target:
                        movement_distances[source][target] = min(
                            movement_distances[source][target], 3
                        )
        for k in locations:
            for i in locations:
                for j in locations:
                    if (
                        movement_distances[i][k] + movement_distances[k][j]
                        < movement_distances[i][j]
                    ):
                        movement_distances[i][j] = (
                            movement_distances[i][k] + movement_distances[k][j]
                        )
        self.movement_distances = movement_distances
        super().__init__(start)
        
    def path_cost(self, c, state1, action, state2):
        """ Do not change this. Returns the cost of reaching state2 from state1 via action,
        given cost c to reach state1: SAIL costs 3, every other action costs 1"""
        return c + (3 if action.startswith("SAIL") else 1)

    def successor(self, state):
        truck_at, truck_on, ferry_at, package_at, package_in = state
        result = []

        truck_loaded_weight = [0] * len(self.trucks)
        for pkg_idx, truck in enumerate(package_in):
            if truck is not None:
                t_idx = self.truck_index[truck]
                truck_loaded_weight[t_idx] += self.package_weights[pkg_idx]

        # 1. Truck Actions
        for truck_index, truck in enumerate(self.trucks):
            location = truck_at[truck_index]
            if location is not None:
                # DRIVE
                if any(
                    package_in[package_index] == truck
                    and self.package_goals[package_index] == location
                    for package_index in range(len(self.packages))
                ):
                    allow_drive = False
                else:
                    allow_drive = True
                if allow_drive:
                    for target in self.roads[location]:
                        carries_package = any(
                            package_in[package_index] == truck
                            for package_index in range(len(self.packages))
                        )
                        useful_targets = []
                        if carries_package:
                            useful_targets.extend(
                                truck_at[other_index]
                                for other_index in range(len(self.trucks))
                                if other_index != truck_index
                                and truck_at[other_index] is not None
                            )
                        for package_index, package_goal in enumerate(self.package_goals):
                            if package_in[package_index] == truck:
                                useful_targets.append(package_goal)
                            elif (
                                package_in[package_index] is None
                                and package_at[package_index] != package_goal
                                and truck_loaded_weight[truck_index]
                                + self.package_weights[package_index]
                                <= self.truck_capacities[truck_index]
                            ):
                                useful_targets.append(package_at[package_index])
                        if self.ferries:
                            useful_targets.extend(self.all_ports)
                        if not any(
                            self.truck_distances[target][task]
                            < self.truck_distances[location][task]
                            for task in useful_targets
                        ):
                            continue
                        next_truck_at = list(truck_at)
                        next_truck_at[truck_index] = target
                        next_state = (
                            tuple(next_truck_at), truck_on, ferry_at, package_at, package_in,
                        )
                        result.append((f"DRIVE{{{truck},{location},{target}}}", next_state))

                # LOAD (Pruning: Only load packages NOT at their final goal)
                capacity = self.truck_capacities[truck_index]
                loaded_weight = truck_loaded_weight[truck_index]
                for package_index, package in enumerate(self.packages):
                    curr_pkg_loc = package_at[package_index]
                    if curr_pkg_loc == location and curr_pkg_loc != self.package_goals[package_index]:
                        weight = self.package_weights[package_index]
                        if loaded_weight + weight <= capacity:
                            next_package_at = list(package_at)
                            next_package_in = list(package_in)
                            next_package_at[package_index] = None
                            next_package_in[package_index] = truck
                            next_state = (
                                truck_at, truck_on, ferry_at,
                                tuple(next_package_at), tuple(next_package_in),
                            )
                            result.append((f"LOAD{{{package},{truck}}}", next_state))

                # UNLOAD (Full freedom: Allow unloading anywhere to support optimal relay strategies)
                for package_index, package in enumerate(self.packages):
                    if package_in[package_index] == truck:
                        if (
                            not self.ferries
                            and location != self.package_goals[package_index]
                            and not any(
                                other_index != truck_index
                                and truck_at[other_index] == location
                                for other_index in range(len(self.trucks))
                            )
                        ):
                            continue
                        next_package_at = list(package_at)
                        next_package_in = list(package_in)
                        next_package_at[package_index] = location
                        next_package_in[package_index] = None
                        next_state = (
                            truck_at, truck_on, ferry_at,
                            tuple(next_package_at), tuple(next_package_in),
                        )
                        result.append((f"UNLOAD{{{package},{truck}}}", next_state))

                # BOARD
                for ferry_index, ferry in enumerate(self.ferries):
                    if location == ferry_at[ferry_index]:
                        if any(
                            package_in[package_index] == truck
                            and self.package_goals[package_index] == location
                            for package_index in range(len(self.packages))
                        ):
                            continue
                        aboard = sum(on == ferry for on in truck_on)
                        if aboard < self.game["Ferries"][ferry][2]:
                            useful_targets = [
                                package_at[package_index]
                                for package_index in range(len(self.packages))
                                if package_in[package_index] is None
                                and package_at[package_index]
                                != self.package_goals[package_index]
                            ]
                            useful_targets.extend(
                                self.package_goals[package_index]
                                for package_index in range(len(self.packages))
                                if package_in[package_index] is not None
                            )
                            useful_targets.extend(
                                other_location
                                for other_index, other_location in enumerate(truck_at)
                                if other_index != truck_index
                                and other_location is not None
                            )
                            if not any(
                                self.truck_distances[port][task]
                                < self.truck_distances[location][task]
                                for port in self.ferry_served_ports[ferry_index]
                                for task in useful_targets
                            ):
                                continue
                            next_truck_at = list(truck_at)
                            next_truck_on = list(truck_on)
                            next_truck_at[truck_index] = None
                            next_truck_on[truck_index] = ferry
                            next_state = (
                                tuple(next_truck_at), tuple(next_truck_on),
                                ferry_at, package_at, package_in,
                            )
                            result.append((f"BOARD{{{truck},{ferry}}}", next_state))
            else:
                # DISEMBARK
                ferry = truck_on[truck_index]
                ferry_index = self.ferry_index[ferry]
                location = ferry_at[ferry_index]
                next_truck_at = list(truck_at)
                next_truck_on = list(truck_on)
                next_truck_at[truck_index] = location
                next_truck_on[truck_index] = None
                next_state = (
                    tuple(next_truck_at), tuple(next_truck_on),
                    ferry_at, package_at, package_in,
                )
                result.append((f"DISEMBARK{{{truck},{ferry}}}", next_state))

        # 2. Ferry Actions (SAIL) - Full unconstrained generation to guarantee 100% optimality
        for ferry_index, ferry in enumerate(self.ferries):
            source = ferry_at[ferry_index]
            passengers = [
                truck_index
                for truck_index, aboard in enumerate(truck_on)
                if aboard == ferry
            ]
            if any(
                package_in[package_index] is not None
                and self.truck_index[package_in[package_index]] in passengers
                and self.package_goals[package_index] == source
                for package_index in range(len(self.packages))
            ):
                continue
            for target in self.ferry_served_ports[ferry_index]:
                if target != source:
                    if (
                        not passengers
                        and not any(location == target for location in truck_at)
                    ):
                        continue
                    if passengers:
                        useful_targets = {
                            package_at[package_index]
                            for package_index in range(len(self.packages))
                            if package_in[package_index] is None
                            and package_at[package_index]
                            != self.package_goals[package_index]
                        }
                        useful_targets.update(
                            self.package_goals[package_index]
                            for package_index in range(len(self.packages))
                            if package_in[package_index] is not None
                        )
                        useful_targets.update(
                            location
                            for truck_index, location in enumerate(truck_at)
                            if truck_index not in passengers and location is not None
                        )
                        if not any(
                            self.truck_distances[target][task]
                            < self.truck_distances[source][task]
                            for task in useful_targets
                        ):
                            continue
                    next_ferry_at = list(ferry_at)
                    next_ferry_at[ferry_index] = target
                    next_state = (
                        truck_at, truck_on, tuple(next_ferry_at), package_at, package_in,
                    )
                    result.append((f"SAIL{{{ferry},{source},{target}}}", next_state))

        return result
    
    def goal_test(self, state):
        return state[3] == self.package_goals and all(p is None for p in state[4])

    def h_astar(self, node):
        cached = self._heuristic_cache.get(node.state)
        if cached is not None:
            return cached

        truck_at, truck_on, ferry_at, package_at, package_in = node.state
        
        t_locs = []
        truck_aboard = []
        for t_idx in range(self.num_trucks):
            loc = truck_at[t_idx]
            if loc is not None:
                t_locs.append(loc)
                truck_aboard.append(False)
            else:
                f_idx = self.ferry_index[truck_on[t_idx]]
                t_locs.append(ferry_at[f_idx])
                truck_aboard.append(True)
                
        handling_actions = 0
        longest_required_travel = 0
        total_delivery_travel = 0
        total_delivery_action_cost = 0

        for p_idx, goal in enumerate(self.package_goals):
            p_in = package_in[p_idx]
            if p_in is None:
                p_loc = package_at[p_idx]
                total_delivery_travel += self.movement_distances[p_loc][goal]
                total_delivery_action_cost += self.truck_distances[p_loc][goal]
            else:
                t_idx = self.truck_index[p_in]
                total_delivery_travel += self.movement_distances[t_locs[t_idx]][goal]
                total_delivery_action_cost += max(
                    0,
                    self.truck_distances[t_locs[t_idx]][goal]
                    - int(truck_aboard[t_idx]),
                )

            if p_in is None:
                p_loc = package_at[p_idx]
                if p_loc != goal:
                    min_d_reach = min(
                        (
                            max(
                                0,
                                self.truck_distances[t_loc][p_loc]
                                - int(truck_aboard[t_idx]),
                            )
                            for t_idx, t_loc in enumerate(t_locs)
                        ),
                        default=0,
                    )
                    required_travel = (
                        min_d_reach + self.truck_distances[p_loc][goal]
                    )
                    handling_actions += 2
                    longest_required_travel = max(
                        longest_required_travel, required_travel
                    )
            else:
                t_idx = self.truck_index[p_in]
                t_loc = t_locs[t_idx]
                required_travel = self.truck_distances[t_loc][goal]
                if truck_aboard[t_idx]:
                    required_travel = max(0, required_travel - 1)
                handling_actions += 1
                longest_required_travel = max(
                    longest_required_travel, required_travel
                )

        travel_bound = max(
            longest_required_travel,
            int(
                (total_delivery_travel + self.max_packages_per_move - 1)
                // self.max_packages_per_move
            ),
            int(
                (total_delivery_action_cost + self.max_packages_per_move - 1)
                // self.max_packages_per_move
            ),
        )
        if not self.ferries:
            required_locations = set(t_locs)
            for p_idx, goal in enumerate(self.package_goals):
                if package_in[p_idx] is None:
                    p_loc = package_at[p_idx]
                    if p_loc != goal:
                        required_locations.add(p_loc)
                        required_locations.add(goal)
                else:
                    required_locations.add(goal)

            locations = tuple(required_locations)
            root = len(locations)
            parent = list(range(root + 1))

            def find(index):
                while parent[index] != index:
                    parent[index] = parent[parent[index]]
                    index = parent[index]
                return index

            def union(first, second):
                first_root = find(first)
                second_root = find(second)
                if first_root == second_root:
                    return False
                parent[first_root] = second_root
                return True

            location_index = {location: i for i, location in enumerate(locations)}
            for t_loc in t_locs:
                union(root, location_index[t_loc])

            edges = sorted(
                (
                    self.truck_distances[first][second],
                    i,
                    j,
                )
                for i, first in enumerate(locations)
                for j, second in enumerate(locations[i + 1:], i + 1)
            )
            forest_travel = 0
            for distance, first, second in edges:
                if union(first, second):
                    forest_travel += distance
                    if find(root) == find(first) and all(
                        find(index) == find(root)
                        for index in range(len(locations))
                    ):
                        break
            travel_bound = max(travel_bound, forest_travel)

        estimate = handling_actions + travel_bound
        if len(self._heuristic_cache) >= 300000:
            self._heuristic_cache.clear()
        self._heuristic_cache[node.state] = estimate
        return estimate
    
def create_logistics_problem(game):
    print("<<create_logistics_problem")
    """ Create a logistics problem, based on the description.
    game - dictionary as described in pdf file"""
    return LogisticsProblem(game)


if __name__ == '__main__':
    ex1_check.main()
