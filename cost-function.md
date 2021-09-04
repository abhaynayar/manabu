# Cost function

```
J(θ) = -1/m Σ [y*log(h) + (1-y)*log(1-h)]
```

Both terms adjust depending upon the labeled value.

log(0) = ∞
log(1) = 0

The first term's loss approaches infinity as prediction approaches 0.
The second term's loss approaches infinity as prediction approaches 1.

