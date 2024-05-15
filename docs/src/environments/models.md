# Models
Submodule for generating discretised evolution operators from inputted pulses:
```math
    U(t) = {%
        \mathcal{T}\exp{\left(-i\int H(\epsilon(t'))\text{d}t'\right)}
    }\approx\prod_{i=1}^{N}\exp{\left(-i\Delta tH(\epsilon(t_i))\right)}
```
```math
    \mathcal{E}(t) = {%
        \mathcal{T}\exp{\left(\int\mathcal{L}(\epsilon(t'))\text{d}t'\right)}
    }\approx{%
        \prod_{i=1}^{N}\exp{\left(\Delta t\mathcal{L}(\epsilon(t_i))\right)}
    }
```

```@docs
ModelFunction
```

## Custom Models Implemented
```@docs
QuantumDot2
```
