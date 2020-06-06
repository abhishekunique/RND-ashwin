softlearning run_example_local examples.distance_learning `# Notice, the variants will be taken from the examples/distance_learning directory!!` \
        --exp-name=ddl_maze_embed `# This is the name your experiment will be saved under ~/ray_results` \
        --algorithm=DDL `# Algorithm of choice (ex: DDL, etc.) ` \
        --num-samples=2 `# Number of seeds PER choice of hyperparameters (will be multiplied by number of param combinations if you tune over a bunch)` \
        --trial-gpus=0.5 `# Number of GPUs that will be utilized PER seed` \
        --trial-cpus=2 `# Number of CPUs that will be utilized PER seed (doesn't really affect much)` \
        --universe=gym `# Environment universe (usually gym)` \
        --domain=Point2D `# Environment domain, your environment name should be something like <Domain><Task> concatenated together` \
        --task=Maze-v0 `# Training environment task` \
        --task-evaluation=Maze-v0 `# Evaluation environment task` \
        --video-save-frequency=25 `# Evaluation video save frequency (every _ iterations)` \
        --save-training-video-frequency=0`# Training video save frequency (every _ rollouts); 0 = disabled` \
        --checkpoint-frequency=25 `# Checkpoint frequency (every _ iterations)` \
        --checkpoint-replay-pool=True `# Whether or not to save the replay pool on checkpointing` \
        --vision=False `# Running from state vs. vision` \
        --preprocessor-type="None" `# Usually used for running from pixels (what type of preprocessor do you want to use, ConvnetPreprocessor, VAEPreprocessor, etc.)`