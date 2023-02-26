# Rockfish Tutorial 
This is the repo for the tutorial of runing tasks on Rockfish.
Rockfish is a community-shared cluster, i.e., everyone can use any nodes within the limits of their allocated utilization. Every quarter you request a certain amount of GPU hours (i.e., number of cores x wall time) and provide justification for them. These assigned limits will be reset on a quarterly basis (use it or lose it). 

To start, create an account [here](https://coldfront.rockfish.jhu.edu/) with your JHED. 

**IMPORTANT:** 
 - It is important for you to use JHED otherwise the system will have difficulty authenticating you. 
 - If you're a student, you need to get access through your PI (usually you advisor). 

# Repository Content 
 - [`test_gpus`](test_gpus) shows how to run a simple script on the GPU processor. You can run `srun test_gpus.sh` to submit the job and make sure that you have access to GPU nodes. 
 - [`classification-example`](classification-example) shows how to run a simple classification task on the GPU processor.

# FAQ

- **How do I find what the queue of a partition?** 
  - Type the command `sinfo -s` to get a list of the partitions. 
  - `sinfo -p partition-name` will display the utilization for this partition.
- **How do I know what nodes are available?** 
  - Type the command `sinfo -N` to get a list of the nodes. 
  - `sinfo -N -p partition-name` will display the utilization for this partition.
- **How do I interpret "states"?**
  - `idle`: The node is available for use. 
  - `alloc`: The node is currently being used by a job. 
  - `mix`: Some of the node's processors are currently being used by a job.
  - Further details here: https://slurm.schedmd.com/sinfo.html 
- **How do I see how many GPUs are available on each node?**
  - Type the command `sinfo -N -p a100` to get a list of the nodes and the number of GPUs available on each node.
- **How do I submit multiple jobs?** 
  - You can submit multiple jobs by using the `--array` flag. 
  - For example, `sbatch --array=1-10 job.sh` will submit 10 jobs. 
  - You can access the job number using the environment variable `$SLURM_ARRAY_TASK_ID`. 
  - For example, `echo $SLURM_ARRAY_TASK_ID` will print the job number.
- **How do I submit a job to a specific node?** 
  - You can submit a job to a specific node by using the `--nodelist` flag. 
  - For example, `sbatch --nodelist=node1 job.sh` will submit the job to node1. 
  - You can also use the `--exclude` flag to exclude a node. 
  - For example, `sbatch --exclude=node1 job.sh` will submit the job to any node except node1.
- **How do I create interactive session?** 
  - You can create an interactive session by using the `salloc` command.
  - For example, `salloc -p a100 --gres=gpu:1 --time=00:30:00` will create an interactive session with 1 GPU for 30 minutes.
  - You can also `interact` command which internally makes call to `salloc` command.
- **How do I see the queue of my jobs?** 
  - You can see the queue of your jobs by using the `squeue` command.
  - You can specialize it for a partition: `squeue -p a100`
  - For example, `squeue -u userid` will show the queue of your jobs.
- **What do status labels mean in the output of `squeue`?**
  - `PD`: Pending. The job is awaiting resource allocation.
  - `R`: Running. The job currently has an allocation.
  - `CG`: Completing. The job is in the process of completing. Some processes on some nodes may still be active.
  - `CD`: Completed. The job has terminated all processes on all nodes.
  - `F`: Failed. The job terminated with non-zero exit code or other failure condition.
  - `TO`: Timeout. The job terminated upon reaching its time limit.
  - `NF`: Node Failure. The job terminated due to failure of one or more allocated nodes.
  - `CA`: Canceled. The job was explicitly canceled by the user or system administrator. The job may or may not have been initiated.
  - `SE`: Special Exit. The job was requeued in a special state by the scheduler; it may or may not have been initiated.
  - `ST`: Suspended. The job has an allocation, but execution has been suspended and CPUs have been released for other jobs.
  - `S`: Suspended by user. The job has an allocation, but execution has been suspended and CPUs have been released for other jobs at the request of the user.
  - `PR`: Preempted. The job was preempted.
- **How do I cancel a job?** 
  - You can cancel a job by using the `scancel` command.
  - For example, `scancel jobid` will cancel the job with the given jobid.
- **How do I check the statistics of a finished job?** 
  - You can check the statistics of a finished job by using the `sacct` command.
  - For example, `sacct -j jobid` will show the statistics of the job with the given jobid.


# Additional Resources
 - Tracking the accounts/usage: https://coldfront.rockfish.jhu.edu/ 
 - Login node: `ssh userid@login.rockfish.jhu.edu` 
 - Help desk:  help@rockfish.jhu.edu if you face any issues, email these folks! :) 
 - User Guide: https://www.arch.jhu.edu/access/user-guide/
 - System configuration: https://www.arch.jhu.edu/about-rockfish/system-configuration/ 
 - Tutorials: https://marcc.readthedocs.io/
 - FAQs: https://www.arch.jhu.edu/access/faq/
 - A [useful collection of slides](https://livejohnshopkins-my.sharepoint.com/:p:/g/personal/bzheng12_jh_edu/EQOyArR6h0lEtRxJxrMqefIBPs_aFuYLr6hA8qLlBUEiqw?e=i51Ifu) on Rockfish. 
 - Debugging with [TotalView](https://www.youtube.com/watch?v=Zn1xKY7Jxrk) 


If you struggle with SLURM, these are useful cheatsheets: 
 - [Slurm basics](https://hpc.nmsu.edu/discovery/slurm/slurm-commands/) 
 - [Tutorial: Using Slurm Workload Manager](https://www.cs.sfu.ca/~ashriram/Courses/CS431/slurm.html)
 - [Job submit commands with examples](https://uwaterloo.ca/math-faculty-computing-facility/services/service-catalogue-teaching-linux/job-submit-commands-examples)
