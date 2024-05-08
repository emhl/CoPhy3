# Computational Physics - Course Summer 2024

## Exercise 1. Ising model

Goal: We start by simulating the 3D Ising model using the Metropolis-based single-spin flip Monte Carlo method.

Write a program for a Monte Carlo simulation to solve the three-dimensional Ising model with periodic boundary conditions. 

Implement the single-spin flip Metropolis algorithm for sampling.

As you will have to reuse this code for upcoming exercise sheets, it might be worth to make sure that it is well-structured!

1. Measure and plot the energy $E$, the magnetization $M$ , the magnetic susceptibility $\chi$ and the heat capacity $C_V$ at different temperatures $T$.

2. Determine the critical temperature $T_c$.
Hint: You should obtain $T_c \approx 4.51$.

3. Study how your results depend on the system size.
Hint: Start with small systems to reduce the computation time.

4. (OPTIONAL): Save computation time by avoiding unnecessary reevaluations of the exponential function. To achieve this, use an array to store the possible spin-flip acceptance
probabilities.

5. (OPTIONAL): Plot the time dependence of $M$ for a temperature $T < T_c$. Hint: For small systems you should be able to observe sign-flips in $M$.

## Exercise 2. Finite-size scaling

Goal: In numerical simulations we are only able to tackle relatively small system sizes whereas real physical systems are usually much larger. Finite size scaling analysis is a technique which allows us to get good approximations for the thermodynamic limit.

1. Use your program of the first exercise sheet to perform simulations of the 3D Ising system for different system sizes to determine the critical exponents γ and ν.

    Hint: Use the finite size scaling relation of the magnetic susceptibility and the fact that the critical temperature is given by Tc ≈ 4.51.

    You might find the following points useful:
   - You can get a first estimate for the ratio γ/ν by plotting χmax as a function of the system size.
   - Vary γ/ν and 1/ν until you get the best possible data collapse. Judge the quality of the data collapse ”by eye”.
2. (OPTIONAL): Repeat the same process for the specific heat.


## Exercise 3.1: Binder cumulant

Goal: We have done extensive analyses of the phase transition of the 3D Ising model. However, we still need a way to determine the critical temperature more precisely than with a mere observation of the initial growth of the spontaneous magnetization. For this we will use the higher order cumulant, i.e., Binder cumulant.

The Binder cumulant is defined as
$$U = 1 - \frac{\langle M^4 \rangle}{3\langle M^2 \rangle^2} = \begin{cases} 2/3 & \text{for } T < T_c \\ \text{const.} & \text{for } T = T_c  \\ 0 & \text{for } T > T_c \end{cases}$$
where $M$ is the magnetization.

The Binder cumulant is a dimensionless quantity which is expected to be very sensitive at the critical point (phase transition). The higher order cumulant makes it possible to determine the critical temperature at higher precision.

1. Compute the Binder cumulant for different system sizes and temperatures. You can
use previous simulated data.
2. Determine the critical temperature $T_c$ by taking the derivative of the curve
    
## Exercise 3.2 Microcanonical Monte Carlo

_Goal: So far, we treated the Ising model in the canonical ensemble (fixed temperature) where the samples were drawn according to the Boltzmann distribution. In this week’s exercise we are going to perform a microcanonical Monte Carlo simulation of the 3D Ising model according to the Creutz algorithm (M. Creutz, Phys. Rev. Lett., 50, 1411, (1983))._

The Creutz algorithm is defined in the following way:

1. Start with an initial spin configuration $x$ of a given energy $E$ and choose an initial container energy $E_d$ (demon energy) such that $E_{max} \geq E_d \geq 0$.
2. Choose a spin at random and flip it to obtain the configuration $y$.
3. Calculate the energy difference $\Delta E$ between the configurations $x$ and $y$.
4. if $E_{max} \geq E_d - \Delta E \geq 0$ accept the new configuration $x'=y$. And set the new daemon energy $E_d' = E_d - \Delta E$. Otherwise reject the new configuration
5. repeat steps 2-4 

**Tasks:**
1. Modify your program of the first exercise to simulate a microcanonical Ising system
using the Creutz algorithm.
1. Determine the corresponding temperature $T$ using $P(E_d) \propto \exp(-\beta E_d)$.
2. Compute $T$ for different $E_{max}$. Plot energy and magnetization as a function of temperature and compare your results to the results obtained with the Metropolis algorithm.
4. Repeat the above tasks for different system sizes and compare your results.
5. (OPTIONAL) What happens in the case $E_{max} = 0$ (Q2R algorithm)? Discuss the issue of ergodicity.
