# 6.843 Final Project

Repository forked from [here](https://github.com/pangtao22/pick_and_place_benchmarking_framework)

## Installation Tips 

Works with Drake downloaded for Ubuntu and also installed through pip with python3.7. To add required libraries, download the [robotics_utilities](https://github.com/pangtao22/robotics_utilities) library and put it in the `pick_and_place` directory. Also clone the [manipulation](https://github.com/RussTedrake/manipulation) repo and move the `manipulation/manipulation` directory into the `pick_and_place` directory. Pip install all other required packages (most are in the `manipulation/*_requirements.txt` file and `robotics_utilities/requirements.txt` file, the rest come up as errors when trying to run any python file in `pick_and_place`. Install as needed. `Graphviz` should also be pip installed as well as installed to the machine as a package (ie `apt install graphviz` or `pacman -S graphviz`). With Drake installed through pip, right now the `package.xml` is missing through path- see [here](https://github.com/RobotLocomotion/drake/issues/16069) to fix. 


## Objectives

We would like to pick, sort, and place different produce into bins using the KUKA iiwa arm. Currently we have models for cucumbers, mangoes, and limes.


## Current TODO

- [x] Add multiple produce items
- [ ] Edit camera pose to try and avoid collisions?
- [x] Add extra box(es) for sorted produce
- [ ] Distinguish between what produce the robot is currently grasping- should do in grasp sampling, detect bounding boxes between produce to figure out how to grasp and also what to grasp
- [ ] Change trajectory based on what the robot is grasping
- [ ] Make animation go faster?
