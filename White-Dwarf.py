import numpy as np
import scipy.integrate as scint
import matplotlib.pyplot as plt

# =========
# Define variables
# =========
mu_e = 2
rho0 = (9.74e5*mu_e)
M0 = 5.67e33/(mu_e**2)
R0 = 7.72e8/mu_e

# =========
# ODEs
# =========

# State vector: y=[rho,m]
def odes(r, y):
    """
    Defines coupled set of ODEs that govern the mass and density of white dwarf stars (simplified with dimensionless quantities)
    :param r: (float) radius
    :param y: (array) state vector
    :return: (array) differentiated state vector
    """
    x = y[0]**(1./3)
    gamma = (x**2)/(3*np.sqrt((1+x**2)))

    # ODEs
    drho_dr = -(y[1]*y[0])/(gamma*r**2)
    dm_dr = (r**2)*y[0]

    return [drho_dr, dm_dr]


# =========
# IVP Solution
# =========
def whiteDwarf(rho_c, method='RK45'):
    """
    Uses odes function to solve coupled ODEs governing mass and density of white dwarf stars using scipy.integrate
    :param rho_c: (array) parameters governing family of solutions
    :param method: (string) ODE solving method
    :return: (arrays) radii - radii integrated over; masses - mass at each iteration, densities - densities at each iteration
    """
    masses = []
    radii = []
    densities = []



    for i in range(len(rho_c)):
        rho_temp = rho_c[i]
        rSpan = [3e-14, 10]  # Integration range
        initCond = [rho_temp, 0]
        # Set terminal condition
        rho_f = lambda r, y: y[0] - 5.13e-17  # Density very close to zero
        rho_f.terminal = True
        # Solve IVP
        sol = scint.solve_ivp(odes, t_span=rSpan, y0=initCond, method=method, events=rho_f)

        # Unit Conversion before adding to lists to be returned
        rhoValues = sol.y[0] * rho0
        mValues = sol.y[1] * M0
        rValues = sol.t * R0
        masses.append(mValues[-1])
        densities.append(rhoValues[-1])
        radii.append(rValues[-1])

    # Return final radius, mass, and density values for each rho_c
    return radii, masses, densities


# =========
# Part 1
# =========
ran = np.logspace(-1, 6, num=10)
solutions1 = whiteDwarf(ran)


# =========
# Part 2
# =========
plt.scatter(solutions1[1], solutions1[0])
plt.ylabel("Radius (cm)")
plt.xlabel("Mass (g)")
plt.title("Radius vs mass of white dwarf stars of various \u03C1_c")
plt.show()


# =========
# Part 3
# =========
solutions1 = whiteDwarf(ran[3:6])
solutions2 = whiteDwarf(ran[3:6], 'RK23')  # Use RK23 instead of RK45
plt.scatter(solutions1[1], solutions1[0], label='RK45')
plt.scatter(solutions2[1], solutions2[0], label='RK23')
# Print numerical difference
print("Differences in masses are: ", abs(np.array(solutions1[1])-np.array(solutions2[1])))
print("Differences in radii are: ", abs(np.array(solutions1[0]-np.array(solutions2[0]))))
plt.legend()
plt.ylabel("Radius (cm)")
plt.xlabel("Mass (g)")
plt.title("Radius vs mass of white dwarf stars of various\n \u03C1_c using RK45 and RK23 ODE solutions")
plt.show()


# =========
# Part 4
# =========
Msun = 1.9891e33  # Mass of Sun in grams
Rsun = 6.957e10  # Radius of Sun in cm

# Read and organize data
sunData = np.zeros((26, 4))
inFile = open("wd_mass_radius.csv", "r")
inFile.readline()  # Skip first line
i = 0  # Indexing
for line in inFile:
    data = line.split(',')
    sunData[i, 0] = float(data[0]) * Msun
    sunData[i, 1] = float(data[1]) * Msun
    sunData[i, 2] = float(data[2]) * Rsun
    sunData[i, 3] = float(data[3]) * Rsun
    i += 1
inFile.close()

# Plotting
solutions1 = whiteDwarf(ran)
plt.scatter(solutions1[1], solutions1[0], label='Theoretical')
plt.scatter(sunData[:, 0], sunData[:, 2], label='Tremblay et al. (2017)')
plt.errorbar(sunData[:, 0], sunData[:, 2], sunData[:, 3], sunData[:, 1], c='g', fmt='.', linestyle=None)
plt.legend()
plt.ylabel("Radius (cm)")
plt.xlabel("Mass (g)")
plt.title("Theoretical and observational radius mass relation of dwarf stars\n ")
plt.show()
