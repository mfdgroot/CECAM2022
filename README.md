# CECAM2022

These are two notebooks to showcase how one can extract forces from quantum chemistry calculations on quantum computers. The notebooks are geared toward understanding the physics more than coding. Interested people can go into the utility files to see what happens under the hood. All answers to the questions can be found by the data generated in the notebook. There are extra questions that require modifying or extending the code but which require some familiarity with the packages used and the theory of quantum chemistry on quantum computers.

The first notebook is about finite differences and how both the finite difference increment and noise on the energy affect the quality of the forces. The calculations in this notebook are very time consuming. You can bypass them by setting the variable `iampatient=False`. This loads the data from disk that would otherwise be generated. 

The second notebook handles force operators. You can verify for yourself that the force operators give the right result. You will also understand more about the size and nature of the Pulay forces.
