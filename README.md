# CPP_MNIST

First, source the bash profile to set environment variables and aliases:

```bash
source job_bashrc
```

To do a Full Build, run from project root:

```bash
bld -B
```

To Build only changed files, run from project root:

```bash
bld
```

To train MLP: First build, then run: 

```bash
./bin/trainMLP
```