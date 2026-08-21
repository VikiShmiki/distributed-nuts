# T51 preferred statistical protocol

- Seeds: 5
- Blocked warm repetitions per seed/configuration: 30
- Bucketed configurations: 36
- Faster than monolithic: 6/36
- Median/best/worst speedup: 0.896x / 1.059x / 0.529x
- Leapfrog/divergence/max-depth parity: True

Intervals use a deterministic hierarchical bootstrap: sampler seeds and the retained monolithic/method timing samples are independently resampled; the reported statistic is the median speedup across seeds. Repeat indices are not treated as paired because methods were timed sequentially.
