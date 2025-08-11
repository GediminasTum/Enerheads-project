import pandas as pd
import pyomo.environ as pe

market_df = pd.read_csv(
    "market_data.csv",
    parse_dates=["Unnamed: 0"],
    index_col="Unnamed: 0")

market_df.index = pd.to_datetime(market_df.index, utc=True)
df = market_df[[
    'LT_mfrr_SA_up_activ',
    'LT_mfrr_SA_down_activ',
    'LT_up_sa_cbmp',
    'LT_down_sa_cbmp'
]]

df = df.ffill().bfill()


if df.isna().any().any():
    print("DataFrame contains NaN values.")
else:
    print("No NaN values found.")



class SingleMarketSolver:

    battery_power_mw = 1
    battery_capacity_mwh = 2
    max_cycles = 2
    interval_duration_hours = 0.25
    intervals_per_day = int(24 / interval_duration_hours)

    def __init__(self, battery_power_mw, battery_capacity_mwh, max_cycles_per_day, initial_soc, **kwargs):

        self.battery_power_mw = battery_power_mw
        self.battery_capacity_mwh = battery_capacity_mwh
        self.max_cycles_per_day = max_cycles_per_day
        self.energy_per_interval = battery_power_mw * 0.25
        self.initial_soc = initial_soc if initial_soc is not None else battery_capacity_mwh / 2

        self.intervals_per_day = int(24/0.25)
        self.max_energy_per_day = max_cycles_per_day * battery_capacity_mwh

        self.output = []
        self.model = None
        self.data = None

    def define(self, data, **kwargs):

        self.data = data
        n_intervals = len(data.index)

        model = pe.ConcreteModel()
        model.T = pe.RangeSet(0, len(data.index) - 1)

        model.ca = pe.Param(model.T, initialize=data['LT_mfrr_SA_down_activ'].to_dict())
        model.da = pe.Param(model.T, initialize=data['LT_mfrr_SA_up_activ'].to_dict())
        model.cp = pe.Param(model.T, initialize=data['LT_down_sa_cbmp'].to_dict())
        model.dp = pe.Param(model.T, initialize=data['LT_up_sa_cbmp'].to_dict())

        model.charge = pe.Var(model.T, within=pe.Binary)
        model.discharge = pe.Var(model.T, within=pe.Binary)
        model.soc = pe.Var(model.T, bounds=(0, self.battery_capacity_mwh))

        model.profit = pe.Objective(rule=self._maximize_profit, sense=pe.maximize)

        model.charge_activation = pe.Constraint(model.T, rule=self._charge_activation_con)
        model.discharge_activation = pe.Constraint(model.T, rule=self._discharge_activation_con)
        model.cycle_constraint = pe.Constraint(rule=self._cycle_con)
        model.soc_evolution = pe.Constraint(model.T, rule=self._soc_evolution_con)

        self.model = model
        return self
    
    def solve(self):

        if self.model is not None:

            solver = pe.SolverFactory('glpk', executable=r"C:\Users\Vartotojas\Downloads\winglpk-4.65\glpk-4.65\w64\glpsol.exe")
            solver.options['mipgap'] = 0.01 
            results = solver.solve(self.model, tee=True)
            print(results.solver.status, results.solver.termination_condition)
            
            res = []
            total_profit = pe.value(self.model.profit)

            for t in self.model.T:
                rec = {
                    'datetime': self.data.index[t],
                    'sys_charge_activ': pe.value(self.model.ca[t]),
                    'sys_discharge_activ': pe.value(self.model.da[t]),
                    'charge_price': pe.value(self.model.cp[t]),
                    'discharge_price': pe.value(self.model.dp[t]),
                    'charge': pe.value(self.model.charge[t]),
                    'discharge': pe.value(self.model.discharge[t]),
                    'soc': pe.value(self.model.soc[t]),
                    'profit_contribution': (
                        pe.value(self.model.discharge[t]) * self.energy_per_interval * pe.value(self.model.dp[t]) -
                        pe.value(self.model.charge[t]) * self.energy_per_interval * pe.value(self.model.cp[t])
                    )
                }
                res.append(rec)
            self.output = res
            self.total_profit = total_profit

            print(f"Total profit: {total_profit:.2f} EUR")
        return self
    
    def _maximise_profit(self, m):

        return sum(
            m.discharge[t] * self.energy_per_interval * m.dp[t] - 
            m.charge[t] * self.energy_per_interval * m.cp[t]       
            for t in m.T
        )

    def _example_constr(self, m, t):
        '''
        This is an example constraint, pyomo passes model object (m) and model timestep index (t)
        '''
        # Allow charging only if the system has a down activation.
        return m.charge[t] <= m.ca[t]
        
    def _some_constraint(self, m, t):
        '''
        Some other constraint
        '''    

