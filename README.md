# Interpolating-Orbits

This repo provides an algorithm to interpolate the orbital elements for the major bodies in the solar system for up to 18 Myr after J2000 and return that value. It is based on data from two simulations of the solar system in Brutus. The first was run by Dr. Tjarda Boekholt with Brutus standalone, while the second was run by me in Fall 2018 with Brutus in AMUSE.
All you need to do is to clone this repo to your drive, open the file "interpolating-orbits.ipynb" to adjust the directory variables in the header and finally provide the time and body for which the orbital elements are desired before running the script. At the moment, the algorithm is not yet optimized as a function that returns the array of orbital elements, but has to be run alone.

For the algorithm, I parametrized the progression of the orbital elements of a new simulation by fitting them with one to three sinusoids. Please take a look into my thesis that can be found in this repo for a documentation of the algorithm itself as well as the two simulations that this algorithm is based on. The document also gives an estimation for the precision of the algorithm.

This repo does not contain the data from the two simulations since they are too big (a few 100 MB), but a collection of plots that show the fits and the progression of the orbital elements from both simulations.

This algorithm made use of Numpy, Scipy, as well as AMUSE and Brutus: http://amusecode.org/
