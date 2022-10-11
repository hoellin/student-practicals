

This project is a small introduction to Data Assimilation Networks (DAN).

## Installation

### Dependencies

- Python 3
- PyTorch

### Usage

To run the code, check you have all the dependencies installed and run the following command:

```bash
python main.py -save lin2d_exp.py -run
```
to run the "Linear 2D experiment" (data assimilation of a 2D ODS) defined in `lin2d_exp.py` and save the results in the `results` folder.

or
```bash
python main.py -save lorenz_exp.py -run
```
to run the "Lorenz experiment" (data assimilation of a 40D Lorenz system) defined in `lorenz_exp.py` and save the results in the `results` folder.

## Credits

This project is based on a practical I had during my Master 2 under the supervision of [Zhang Sixin](https://www.irit.fr/~Sixin.Zhang/).