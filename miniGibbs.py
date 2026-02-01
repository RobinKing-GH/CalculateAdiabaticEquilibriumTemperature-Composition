import numpy as np
from scipy.optimize import minimize

# gas constant (J/mol·K)
R = 8.314462618

class ThermodynamicProperties:
    """thermodynamic properties calculation class"""
    
    def __init__(self):
        # NASA polynomial coefficient (300-1000K, 1000-5000K)
        self.nasa_coeffs = {
            # original species
            'H2': {
                'low': [2.34433112, 0.00798052075, -1.9478151e-05, 2.01572094e-08, -7.37611761e-12, -917.935173, 0.683010238],
                'high': [3.3372792, -4.94024731e-05, 4.99456778e-07, -1.79566394e-10, 2.00255376e-14, -950.158922, -3.20502331]
            },
            'CH4': {
                'low': [5.14987613, -0.0136709788, 4.91800599e-05, -4.84743026e-08, 1.66693956e-11, -10246.1816, -4.64130376],
                'high': [7.4851495, -0.0129533942, 2.99063862e-05, -3.09308455e-08, 1.06181488e-11, -10024.758, 2.03484332]
            },
            'CO': {
                'low': [3.57953347, -0.00061035369, 1.01681433e-06, 9.07005884e-10, -9.04424499e-13, -14344.086, 3.50840928],
                'high': [2.71518561, 0.00206252743, -9.98825771e-07, 2.30053008e-10, -2.03647716e-14, -14198.417, 7.81868772]
            },
            'CO2': {
                'low': [3.85746029, 0.00441437026, -2.21481404e-05, 5.23490188e-08, -4.72084164e-11, -48759.166, 2.27163806],
                'high': [2.35677352, 0.00898459677, -7.12356269e-06, 2.45919022e-09, -1.43699548e-13, -48374.242, 9.90105222]
            },
            'O2': {
                'low': [3.78245636, -0.00299673416, 9.84730201e-06, -9.68129509e-09, 3.24372837e-12, -1063.94356, 3.65767573],
                'high': [3.28253784, 0.00148308754, -7.57966669e-07, 2.09470555e-10, -2.16717794e-14, -1088.45772, 5.45323129]
            },
            'H2O': {
                'low': [4.19864056, -0.0020364341, 6.52040211e-06, -5.48797062e-09, 1.77197817e-12, -30293.7267, -0.849032208],
                'high': [3.03399249, 0.00217691804, -1.64072518e-07, -9.7041987e-11, 1.68200992e-14, -30004.2971, 4.9667701]
            },
            'N2': {
                'low': [3.29867700, 0.00140824000, -3.96322200e-06, 5.64151500e-09, -2.44485400e-12, -1020.89990, 3.95037200],
                'high': [2.92664000, 0.00148797680, -5.68476000e-07, 1.00970380e-10, -6.75335100e-15, -922.79770, 5.98052800]
            },
            'C2H2': {
                'low': [8.08681094, -0.00162172988, 2.83908087e-05, -2.92520052e-08, 1.13785738e-11, 25512.4833, -1.23028121],
                'high': [4.43677086, 0.00796259424, -2.98103070e-06, 4.44359479e-10, -2.49801535e-14, 25790.7715, 2.28372251]
            },
            'C2H4': {
                'low': [3.95920148, -0.00757052247, 5.70990292e-05, -6.91588753e-08, 2.69884373e-11, 494.896033, 4.09733096],
                'high': [2.03611116, 0.01464501590, -6.09985088e-06, 1.09145050e-09, -7.39432734e-14, 1098.34838, 11.2907437]
            },
            'C2H6': {
                'low': [4.29142492, -0.00550154270, 5.99438288e-05, -7.08466285e-08, 2.68685771e-11, -1154.70431, 2.66682316],
                'high': [1.07188150, 0.02168526590, -1.00256067e-05, 2.21412001e-09, -1.90002890e-13, -465.543292, 17.5845692]
            },
            'C3H8': {
                'low': [0.93355380, 0.02621699070, -6.63550035e-05, 9.20536440e-08, -4.12758840e-11, -13924.5601, 18.0241365],
                'high': [7.52521776, 0.01050502130, -3.02359489e-06, 4.33493744e-10, -2.28732756e-14, -13998.0497, 17.0943675]
            },
            'C6H6': {
                'low': [-0.20616650, 0.03927729120, -1.18228602e-04, 1.76519602e-07, -1.04940938e-10, 8294.65750, 22.2278962],
                'high': [4.12040217, 0.03582314540, -1.47635789e-05, 2.97580268e-09, -2.33303995e-13, 7833.79100, -5.90651628]
            },
            'C10H8': {
                'low': [-0.48271420, 0.06841564720, -2.10656730e-04, 3.17977586e-07, -1.89516684e-10, 18218.0829, 32.5129796],
                'high': [15.5509141, 0.03607102800, -1.48546316e-05, 2.99944712e-09, -2.34982555e-13, 17954.5515, -37.6156884]
            },
            'soot': {
                'low': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'high': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            }
        }
        # soot thermodynamic parameter calculation
        self.soot_properties = {
            'Cp': lambda T: 8.5 + 0.0048 * T - 8.54e5 / T**2,  # J/mol·K (approximation of graphite)
            'H_ref': 0.0,
            'S_ref': 5.7,
        }
    
    def get_coefficients(self, species, T):
        """get the NASA coefficient of the specified temperature and species"""
        if species == 'soot':
            return None
        if T <= 1000:
            return self.nasa_coeffs[species]['low']
        else:
            return self.nasa_coeffs[species]['high']
    
    def calc_Cp(self, species, T):
        """calculation of the costant pressure heat capacity (J/mol·K)"""
        if species == 'soot':
            return self.soot_properties['Cp'](T)
        
        coeffs = self.get_coefficients(species, T)
        if coeffs is None:
            return 0.0
        Cp_R = (coeffs[0] + coeffs[1]*T + coeffs[2]*T**2 + 
                coeffs[3]*T**3 + coeffs[4]*T**4)
        return Cp_R * R
    
    def calc_H(self, species, T, T_ref=298.15):
        """calculation of enthalpy (J/mol)"""
        if species == 'soot':
            Cp_avg = self.soot_properties['Cp']((T + T_ref) / 2)
            return self.soot_properties['H_ref'] + Cp_avg * (T - T_ref)
        coeffs = self.get_coefficients(species, T)
        if coeffs is None:
            return 0.0
        H_RT = (coeffs[0] + coeffs[1]*T/2 + coeffs[2]*T**2/3 + 
                coeffs[3]*T**3/4 + coeffs[4]*T**4/5 + coeffs[5]/T)
        return H_RT * R * T
    
    def calc_S(self, species, T, P=1.0, T_ref=298.15, P_ref=1.0):
        """calculation of entropy (J/mol·K)"""
        if species == 'soot':
            Cp_avg = self.soot_properties['Cp']((T + T_ref) / 2)
            S_T = self.soot_properties['S_ref'] + Cp_avg * np.log(T / T_ref)
            return S_T - R * np.log(P / P_ref)
        coeffs = self.get_coefficients(species, T)
        if coeffs is None:
            return 0.0
        S_R = (coeffs[0]*np.log(T) + coeffs[1]*T + coeffs[2]*T**2/2 + 
               coeffs[3]*T**3/3 + coeffs[4]*T**4/4 + coeffs[6])
        return S_R * R - R * np.log(P)
    
    def calc_G(self, species, T, P=1.0):
        """calculation of Gibbs free energy (J/mol)"""
        H = self.calc_H(species, T)
        S = self.calc_S(species, T, P)
        return H - T * S

def calc_gibbs_from_coeffs(T, coeffs):
    """calculation of standard molar Gibbs free energy from coefficient array G0 (J/mol)"""
    if coeffs is None:
        return 0.0
    a = coeffs
    H_RT = (a[0] + 
            a[1]*T/2.0 + 
            a[2]*T**2/3.0 + 
            a[3]*T**3/4.0 + 
            a[4]*T**4/5.0 + 
            a[5]/T)
    S_R = (a[0]*np.log(T) + 
           a[1]*T + 
           a[2]*T**2/2.0 + 
           a[3]*T**3/3.0 + 
           a[4]*T**4/4.0 + 
           a[6])
    G0 = R * T * (H_RT - S_R)
    return G0

def initial_enthalpy(initial_comp, T_initial, thermo):
    """calculate the enthalpy of the initial mixutre"""
    total_moles = sum(initial_comp.values())
    H_initial = 0
    for species, moles in initial_comp.items():
        if species in thermo.nasa_coeffs or species == 'soot':
            H_initial += moles * thermo.calc_H(species, T_initial)
    return H_initial / total_moles
# ==================== optimization objective function and constraints ====================
def total_gibbs_scaled(log_n, T, P, P0, species_list, G0_dict):
    n = np.exp(log_n)
    n = np.maximum(n, 1e-20)
    n_total = np.sum(n)
    if n_total < 1e-30:
        return 1e30
    y = n / n_total
    total_G = 0.0
    for i, species in enumerate(species_list):
        if species == 'soot':
            mu = G0_dict[species]
        else:
            mu = G0_dict[species] + R * T * np.log(y[i] * P / P0)
        total_G += n[i] * mu
    return total_G / (R * T)

def element_constraints_scaled(log_n, element_matrix, init_atoms):
    """atom conservation constraint"""
    n = np.exp(log_n)
    n = np.maximum(n, 1e-20)
    current_atoms = np.dot(element_matrix, n)
    return (current_atoms - init_atoms) / np.maximum(init_atoms, 1e-10)

def bounds_constraint(log_n):
    """ensure that all species have a certain probability of existence"""
    return np.sum(np.exp(log_n)) - 1e-10

def gibbs_minimization(T, P, thermo, species_list, element_matrix, init_atoms, initial_guess=None):
    """calculation of equilibrium composition by minimum Gibbs principle"""
#    print(f"calculate the equilibrium composition at temperature {T} K ...")
    G0_dict = {}
    for species in species_list:
        if species == 'soot':
            G0_dict[species] = thermo.calc_G(species, T, P)
        else:
            coeffs = thermo.get_coefficients(species, T)
            G0_dict[species] = calc_gibbs_from_coeffs(T, coeffs)
    if initial_guess is None:
        n0_linear = np.ones(len(species_list)) * 0.1
        hydrocarbon_indices = [i for i, s in enumerate(species_list) 
                             if s in ['CH4', 'C2H2', 'C2H4', 'C2H6', 'C3H8', 'C6H6', 'C10H8']]
        for idx in hydrocarbon_indices:
            n0_linear[idx] = 0.01
    else:
        n0_linear = initial_guess.copy()
    current_atoms = np.dot(element_matrix, n0_linear)
    scale_factors = init_atoms / np.maximum(current_atoms, 1e-10)
    scale = np.min(scale_factors[scale_factors > 0])
    if np.isfinite(scale):
        n0_linear = n0_linear * scale
    n0 = np.log(np.maximum(n0_linear, 1e-10))
    constraints = [
        {
            'type': 'eq',
            'fun': lambda x: element_constraints_scaled(x, element_matrix, init_atoms),
            'tol': 1e-6
        },
        {
            'type': 'ineq',
            'fun': bounds_constraint
        }
    ]
    bounds = []
    for i, species in enumerate(species_list):
        if species == 'soot':
            bounds.append((-30, 10))
        elif species in ['C6H6', 'C10H8']:
            bounds.append((-30, 5))
        else:
            bounds.append((-20, 20))
    try:
        result = minimize(
            lambda x: total_gibbs_scaled(x, T, P, 1.0, species_list, G0_dict),
            n0,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={
                'maxiter': 3000,
                'ftol': 1e-10,
                'eps': 1e-8,
                'disp': False
            }
        )
        
        if result.success:
            n_eq = np.exp(result.x)
            n_total = np.sum(n_eq)
            if n_total > 1e-30:
                y_eq = n_eq / n_total
#                print(f"Gibbs free energy minimization succeed! Total Gibbs free energy: {result.fun * R * T:.2f} J")
                return n_eq, y_eq, True
            else:
                print("The total number of moles is too small, and the calculation failed")
                return None, None, False
        else:
            print(f"Optimization is not convergent: {result.message}")
            return None, None, False
    except Exception as e:
        print(f"Gibbs minimization failed: {e}")
        return None, None, False

def calculate_mixture_enthalpy(composition, T, thermo, species_list):
    """Calculate the enthalpy of the mixture"""
    total_enthalpy = 0
    for i, species in enumerate(species_list):
        if composition[i] > 0:
            total_enthalpy += composition[i] * thermo.calc_H(species, T)
    return total_enthalpy

def adiabatic_flame_temperature(initial_comp, T_initial, P, thermo, species_list, element_matrix, init_atoms, tol=1e-6, max_iter=100):
    """calculate adiabatic equilibrium temperature"""
    init_n = np.zeros(len(species_list))
    for i, species in enumerate(species_list):
        if species in initial_comp:
            init_n[i] = initial_comp[species]
    total_moles_initial = np.sum(init_n)
    H_initial = 0
    for i, species in enumerate(species_list):
        if init_n[i] > 0:
            H_initial += (init_n[i] / total_moles_initial) * thermo.calc_H(species, T_initial)
    print(f"Initial conditions:")
    print(f"Initial temperature: {T_initial} K")
    print(f"Initial pressure: {P} atm")
#    print(f"Initial enthalpy: {H_initial:.2f} J/mol (归一化)")
#    print(f"Elemental abundance (H,C,O,N): {init_atoms}")
    print("="*50)
    
    # Iterative calculation of adiabatic equilibrium temperature
    T_low = 300.0
    T_high = 5000.0
    T_guess = 1500.0
    last_successful_comp = None
    for iteration in range(max_iter):
        print(f"\nIterate {iteration+1}: iterate temperature = {T_guess:.2f} K")
        if last_successful_comp is not None and iteration > 0:
            initial_guess = last_successful_comp.copy()
        else:
            initial_guess = None
        n_eq, y_eq, success = gibbs_minimization(T_guess, P, thermo, species_list, element_matrix, init_atoms, initial_guess)
        if not success:
            print(f"Calculation of equilibrium composition at temperature {T_guess} K failed.")
            if iteration % 2 == 0:
                T_guess = (T_guess + T_high) / 2
            else:
                T_guess = (T_low + T_guess) / 2
            continue
        last_successful_comp = n_eq.copy()
        H_current = calculate_mixture_enthalpy(y_eq, T_guess, thermo, species_list)
        delta_H = H_current - H_initial
#        print(f"Current enthalpy: {H_current:.2f} J/mol, enthalpy difference: {delta_H:.2f} J/mol")
        if abs(delta_H) < tol:
            print(f"\nConvergence to {iteration+1} iterations")
            return T_guess, y_eq, species_list
        if delta_H > 0:
            T_high = T_guess
            T_new = (T_low + T_guess) / 2
        else:
            T_low = T_guess
            T_new = (T_guess + T_high) / 2
        if abs(T_new - T_guess) < 0.1:
            print(f"The temperature changes very little ({abs(T_new - T_guess):.4f} K)，calculation succeed.") #可以删除1
            return T_guess, y_eq, species_list
        T_guess = max(300.0, min(5000.0, T_new))
    print(f"\nWarning: Not fully converged after {max_iter} iterations.")
    return T_guess, last_successful_comp / np.sum(last_successful_comp) if last_successful_comp is not None else None, species_list

def main():
    # list of species
    species_list = ['H2', 'CH4', 'CO', 'CO2', 'O2', 'H2O', 'N2', 
                   'C2H2', 'C2H4', 'C2H6', 'C3H8', 'C6H6', 'C10H8', 'soot']
    
    # element matrix (H, C, O, N)
    # squence: H2, CH4, CO, CO2, O2, H2O, N2, C2H2, C2H4, C2H6, C3H8, C6H6, C10H8, soot
    element_matrix = np.array([
        [2, 4, 0, 0, 0, 2, 0, 2, 4, 6, 8, 6, 8, 0],# atom number of H
        [0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 3, 6, 10, 1],# atom number of C
        [0, 0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],# atom number of O
        [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]# atom number of N
    ], dtype=float)
    
    # initial matetial composition
    initial_comp = {
        'CH4': 21.0,
        'O2': 16.5,
        'N2': 9.4,
        'C2H6': 0.1,
        'C3H8': 0.05,
        'H2':6,
        'H2O':10,
        'soot':9,
    }
    T_initial = 298.15  # initial temperature，K
    P = 1  # initial pressure，atm
    thermo = ThermodynamicProperties()
    init_n = np.zeros(len(species_list))
    for i, species in enumerate(species_list):
        if species in initial_comp:
            init_n[i] = initial_comp[species]
    total_moles_initial = np.sum(init_n)
    init_atoms = np.dot(element_matrix, init_n) / total_moles_initial
    print("Initial mixture composition:")
    for species, moles in initial_comp.items():
        print(f"  {species}: {moles/total_moles_initial*100:.2f}%")
#    print(f"Total number of moles: {total_moles_initial:.3f} mol")
#    print(f"elemental abundance (H,C,O,N): {init_atoms}")
    # calculate adiabatic equilibrium temperature
    try:
        T_ad, comp, species_list = adiabatic_flame_temperature(initial_comp, T_initial, P, thermo, species_list,element_matrix, init_atoms, tol=1e-3, max_iter=50)
        print("\n" + "="*60)
        print("Adiabatic equilibrium calculation results:")
        print("="*60)
        print(f"Adiabatic equilibrium temperature: {T_ad:.2f} K, {T_ad - 273.15:.2f} °C")
        print("\nEquilibrium composition (mole fraction):")
        print("-"*40)
        # sorting output by mole fraction
        indices_sorted = np.argsort(-comp)
        total_moles = np.sum(comp)
        for idx in indices_sorted:
            mole_fraction = comp[idx]
            if mole_fraction > 1e-6:
                species = species_list[idx]
                print(f"{species:6s}: {mole_fraction:.6f} ({mole_fraction*100:.4f}%)")
        # computational phase distribution
#        print("\nPhase distribution:")
#        print("-"*40)
        gas_total = sum(comp[i] for i, s in enumerate(species_list) if s != 'soot')
        soot_fraction = comp[species_list.index('soot')] if 'soot' in species_list else 0.0
#        print(f"Gas phase: {gas_total:.6f} ({gas_total*100:.4f}%)")
#        print(f"Solid phase(soot): {soot_fraction:.6f} ({soot_fraction*100:.4f}%)")
        print("\nVerification of conservation of elements:")
        print("-"*45)
        element_abundance_final = np.dot(element_matrix, comp)
        elements = ['H', 'C', 'O', 'N']
        print(f"{'Elements':<5} {'Initial':<12} {'Final':<12} {'Relative error':<12}")
        print("-"*45)
        for j, elem in enumerate(elements):
            error = abs(element_abundance_final[j] - init_atoms[j]) / init_atoms[j] * 100
            print(f"{elem:<5} {init_atoms[j]:<12.6f} {element_abundance_final[j]:<12.6f} {error:<12.4f}%")
        H_final = calculate_mixture_enthalpy(comp, T_ad, thermo, species_list)
        H_initial = initial_enthalpy(initial_comp, T_initial, thermo)
        print(f"Enthalpy verification:")
#        print(f"Initial enthalpy: {H_initial:.2f} J/mol")
#        print(f"Final enthalpy: {H_final:.2f} J/mol")
#        print(f"Enthalpy difference: {H_final - H_initial:.2f} J/mol")
        print(f"Relative error of enthalpy change: {abs(H_final - H_initial)/abs(H_initial)*100:.6f}%")
    except Exception as e:
        print(f"The error occurred during the calculation: {e}")
        import traceback
        traceback.print_exc()
#        print("\nAnalysis of main combustion products:")
#        print("-"*40)
"""
        combustion_products = {
            'CO2': comp[species_list.index('CO2')] if 'CO2' in species_list else 0.0,
            'H2O': comp[species_list.index('H2O')] if 'H2O' in species_list else 0.0,
            'CO': comp[species_list.index('CO')] if 'CO' in species_list else 0.0,
        }
        total_combustion = sum(combustion_products.values())
        if total_combustion > 0:
            for product, fraction in combustion_products.items():
                if fraction > 0:
                    print(f"{product}: {fraction/total_combustion*100:.2f}%") #可以删除
    except Exception as e:
        print(f"The error occurred during the calculation: {e}")
        import traceback
        traceback.print_exc()
"""
if __name__ == "__main__":
    print("Extended thermodynamic equilibrium calculation - multi-component combustion analysis")
    print("="*60)
    main()