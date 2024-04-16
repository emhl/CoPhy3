# Exercise 1. Ising model

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