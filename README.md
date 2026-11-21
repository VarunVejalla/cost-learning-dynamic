# Setup

# we want to, given a dataset of trajectories, learn how to predict the trajectories, given the trajectories up to a certain point in time

# overall:
# input: state. Output: action taken by each agent at this state


# input: state, action taken, new states(?). Output: cost vector
# input: state, cost vector --> cost
# input: states, cost --> action

# input: current states and action taken by each agent. Output: new states (dynamics)

# input: cost vector
# input: states 


# MA_IRL



collaboration with Negar on Multi Agent-IRL problem


### Setup:
(this worked for me - John)
Execute the following sequentially:
```console
python3.12 -m venv venv
source venv/bin/activate # run every time to activate environment
pip install -r requirements.txt
sudo wget https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.4-linux-x86_64.tar.gz -P /opt/
tar zxvf /opt/julia-1.11.4-linux-x86_64.tar.gz -C /opt
echo "export PATH='$PATH:/opt/julia-1.11.4/bin'" >> ~/.bashrc
source ~/.bashrc
python -c "import julia; julia.install()"
julia 
import Pkg
Pkg.add("PyCall")
Pkg.build("PyCall")
ENV["PYTHON"] = "<full-path-to-venv>/bin/python"
Pkg.build("PyCall")
using Pkg
Pkg.instantiate()
exit()
# now you can run julia and it will use your python environment for PyCall
```
