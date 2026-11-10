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
                for target in self.roads[location]:
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
                        aboard = sum(on == ferry for on in truck_on)
                        if aboard < self.game["Ferries"][ferry][2]:
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
            for target in self.ferry_served_ports[ferry_index]:
                if target != source:
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
        truck_at, truck_on, ferry_at, package_at, package_in = node.state
        
        t_locs = []
        for t_idx in range(self.num_trucks):
            loc = truck_at[t_idx]
            if loc is not None:
                t_locs.append(loc)
            else:
                f_idx = self.ferry_index[truck_on[t_idx]]
                t_locs.append(ferry_at[f_idx])
                
        max_h = 0
        sum_delivery_h = 0

        for p_idx, goal in enumerate(self.package_goals):
            p_in = package_in[p_idx]
            if p_in is None:
                p_loc = package_at[p_idx]
                if p_loc != goal:
                    min_d_reach = float('inf')
                    for t_loc in t_locs:
                        d = self.truck_distances[t_loc][p_loc]
                        if d < min_d_reach: min_d_reach = d
                    if not t_locs: min_d_reach = 0
                        
                    delivery_cost = 2 + self.truck_distances[p_loc][goal]
                    h_pkg = min_d_reach + delivery_cost
                    
                    sum_delivery_h += delivery_cost
                    if h_pkg > max_h: max_h = h_pkg
            else:
                t_idx = self.truck_index[p_in]
                t_loc = t_locs[t_idx]
                
                delivery_cost = 1 + self.truck_distances[t_loc][goal]
                sum_delivery_h += delivery_cost
                if delivery_cost > max_h: max_h = delivery_cost
                    
        parallel_bound = (sum_delivery_h + self.num_trucks - 1) // self.num_trucks
        return max(max_h, parallel_bound)
    
def create_logistics_problem(game):
    print("<<create_logistics_problem")
    """ Create a logistics problem, based on the description.
    game - dictionary as described in pdf file"""
    return LogisticsProblem(game)


if __name__ == '__main__':
    ex1_check.main()
