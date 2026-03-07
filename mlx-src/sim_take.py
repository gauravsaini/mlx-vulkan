import mlx.core as mx

# Monkey patch mx to see the Gather primitive directly? 
# We can't access `Gather` directly from python, 
# but we can look at `mx.gather`. Is `gather` exposed?
# No, `take` calls `gather` in C++.
# Let's write a quick C++ script to test it, or we can just 
# trust the source code in `ops.cpp`.
