# ship_spotter

Learning to use ML for computer vision.  I am following the instructions in
[Deep Learning with python, by Francois Chollet](https://tanthiamhuat.files.wordpress.com/2018/03/deeplearningwithpython.pdf)


## Installation

Dependencies are listed in requirements.txt.  Note the OSX-specific versions
of tensorflow in `requirements.txt`, you will need to change these for other
machines. Since the ML modules can be hefty, it's recommended to create a
virtual environment.

```shell
python3 -m venv env
source env/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data

I chose the [Ships in Satellite Imagery](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery)
dataset available on Kaggle, for my practice because I'm specifically
interested in the use of Earth Observation data.
