#!/usr/bin/env python3
#
# Please look for "TODO" in the comments, which indicate where you
# need to write your code.
#
# Part 4: Solve the Coupled Simple Harmonic Oscillator Problem (1 point)
#
# * Objective:
#   Take the coupled harmonic oscillator problem we solved in class
#   and rewrite it using a well-structured Python class.
# * Details:
#   The description of the problem and the solution template can be
#   found in `hw1/p4.py`.
#
# From lecture `02w`, we solve systems of coupled harmonic oscillators
# semi-analytically by numerically solving eigenvalue problems.
# However, the code structure was not very clean, making the code hard
# to reuse.
# Although numerical analysis in general does not require
# object-oriented programming, it is sometime useful to package
# stateful caluation into classes.
# For this assignment, we will provide a template class.
# Your responsibility to implement the methods in the class.


import numpy as np


class CoupledOscillators:
    """A class to model a system of coupled harmonic oscillators.

    Attributes:
        Omega (np.ndarray): array of angular frequencies of the normal modes.
        V     (np.ndarray): matrix of eigenvectors representing normal modes.
        M0    (np.ndarray): initial amplitudes of the normal modes.

    """

    def __init__(self, X0=[-0.5, 0, 0.5], m=1.0, k=1.0):
        """Initialize the coupled harmonic oscillator system.

        Args:
            X0 (list or np.ndarray): initial displacements of the oscillators.
            m  (float):              mass of each oscillator (assumed identical for all oscillators).
            k  (float):              spring constant (assumed identical for all springs).

        """
        # TODO: Construct the stiffness matrix K
        X0 = np.asarray(X0) # ensuring we have an array for x
        n = X0.size # getting dimensions
        K = np.zeros((n, n)) # empty array of ones 

        for i in range(n):
            K[i,i] = 2 * k # setting diagonals to sum of k_i + k_(i+1)
            for j in range(n):
                if abs(i - j)== 1: # seeting off-diagonals to -k_i
                    K[i,j] = -k 
                    K[j,i] = -k

        # TODO: Solve the eigenvalue problem for K to find normal modes
        eigs, vecs = np.linalg.eig(K)

        # TODO: Store angular frequencies and eigenvectors
        self.Omega = np.sqrt(eigs)
        self.V = vecs
        # TODO: Compute initial modal amplitudes M0 (normal mode decomposition)
        self.M0 = vecs.T @ X0

        # print(self.Omega)


    def __call__(self, t):
        """Calculate the displacements of the oscillators at time t.

        Args:
            t (float): time at which to compute the displacements.

        Returns:
            np.ndarray: displacements of the oscillators at time t.

        """
        # TODO: Reconstruct the displacements from normal modes

        disp = self.M0 * np.cos(self.Omega * t)
        return self.V @ disp

if __name__ == "__main__":

    # Initialize the coupled oscillator system with default parameters
    co = CoupledOscillators()

    # Print displacements of the oscillators at each time step
    print("Time(s)  Displacements")
    print("----------------------")
    for t in np.linspace(0, 10, num=101):
        X = co(t)             # compute displacements at time t
        print(f"{t:.2f}", X)  # print values for reference
