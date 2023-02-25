# Tutorial on Rockfish
This is the repo for the tutorial of runing tasks on Rockfish.
Rockfish is a community-shared cluster, i.e., everyone can use any nodes within the limits of their allocated utilization. Every quarter you request a certain amount of GPU hours (i.e., number of cores x wall time) and provide justification for them. These assigned limits will be reset on a quarterly basis (use it or lose it). 

To start, create an account [here](https://coldfront.rockfish.jhu.edu/) with your JHED. 

**IMPORTANT:** 
 - It is important for you to use JHED otherwise the system will have difficulty authenticating you. 
 - If you're a student, you need to get access through your PI (usually you advisor). 

# Repository Content 
 - [`test_gpus`](test_gpus) shows how to run a simple script on the GPU processor.
 - [`classification-example`](classification-example) shows how to run a simple classification task on the GPU processor.

# FAQ

- **How do I find what queues are available?** 
  - Type the command `sinfo -s` to get a list of the partition/queues. 
  - `sinfo -p partition-name` will display the utilization for this partition.
- **How do I know what nodes are available?** 
  - Type the command `sinfo -N` to get a list of the nodes. 
  - `sinfo -N -p partition-name` will display the utilization for this partition.
- **How do I interpret "states"?**
  - `idle`: The node is available for use. 
  - `alloc`: The node is currently being used by a job. 
  - `mix`: The node is currently unavailable.
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
 - [Tutorial: Using Slurm Workload Manager](https://www.cs.sfu.ca/~ashriram/Courses/CS431/slurm.html)
 - [Job submit commands with examples](https://uwaterloo.ca/math-faculty-computing-facility/services/service-catalogue-teaching-linux/job-submit-commands-examples)
