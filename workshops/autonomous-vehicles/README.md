# SC5 Hackathon: Autonomous Vehicles

## Aim of the hackathon

The aim of this hackathon is to use learn about convolutional neural networks and apply that knowledge to solve a practical problem related to autonomous vehicles: can we use machine learning to train a model that, given a picture of a road, can accurately predict the direction the steering wheel should be turned?

In the interest of time, we'll use a small training set and only try to guess the general direction in which one should steer (straight forward, to the left, or to the right). In reality, getting autonomous vehicles to steer themselves is a very hard problem, and only one part of the equation: a proper vehicle also needs to now when to apply the brakes/throttle, avoid obstacles, read traffic signs...the list goes on and on. But the basic principles used to solve these issues aren't that much different from our example!

## Prerequisites
* Basic Python programming skills
* Nothing else, really. The aim is to learn together!

## Technical setup
We're going to use [Keras](https://keras.io) to build and train our neural network. It's an efficient library that provides a very elegant Python API for designing network architectures. Assuming you already have Python and pip installed on you machine, installing Keras is as simple as:

`pip install keras`

If you prefer using Python 3:

`pip3 install keras`

<br />
<br />
<p align="center"><em>Brought to you, with love, by</em></p>
<p align="center"><a href="https://sc5.io"><img src="https://github.com/SC5/sc5-machine-learning/blob/master/images/sc5logo-small.png" /></a></p>
