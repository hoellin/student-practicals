

The notebook is a very small practical in sparse reconstruction with Julia.

## Dependencies

### Install Julia

You can download Julia from [julialang.org](julialang.org) or install it with your favorite package manager.

### Install IJulia and dependencies

To run tbis notebook, you need to install `Julia`and Julia kernel:
    
    ```bash
    julia -e 'using Pkg; Pkg.add("IJulia")'
    ```
        
        You also need to install the following packages if they are not already installed:
        
        ```julia
        using Pkg
        Pkg.add("Images")
        Pkg.add("ImageMagick")
        Pkg.add("FFTW")
        Pkg.add("Colors")
        Pkg.add("Plots")
        Pkg.add("ORCA")
        ````

## Run the notebook

Then, you can simply run the notebook as usual:

```bash
jupyter notebook non_smooth_optim.ipynb
```
