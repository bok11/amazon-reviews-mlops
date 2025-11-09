# First steps
I open my task, and orient myself.
I am asked to create a end-to-end machine learning system, to do predictions on the included dataset.
My immidiate thoughts are that i need to go through some high level steps:
- Explore the dataset, and create a validation set from the start.
- Based on my exploration, decide on a simple model to use as baseline.
- Build a runtime component to host the inference model

I also identify the non-functional requirement:
- The runtime must respond to queries in less than 300ms with a p99

After getting an overview i open a jupyter notebook for my exploration phase.

(see exploration.ipynb)
