"""
``odak.manager``

Provides necessary definition for submitting, running and gathering compute.

"""

import threading
try:
    import dispy
except:
    print('odak.manager relies on dispy. Install it using: pip install dispy')
from tqdm import tqdm
from ..tools import shell_command


def start_server():
    """
    Definition to start a dispynode.py.
    """
    cmd = [
        'dispynode.py',
        '--clean',
        '--daemon'
    ]
    shell_command(cmd)
    return True


class agent():
    def __init__(self, compute, cluster=False, depends=[], server=False):
        """
        Class to submit and run task(s), same class gathers results at the end of a compute.

        Parameters
        ----------
        compute        : function
                         Compute function to be run.
        cluster        : bool
                         Set it to True to distribute your job across a cluster or multi CPU cores ofa computer. When set to True, make sure that dispynode.py is running, see https://pgiri.github.io/dispy/dispynode.html .Set it to False to run your tasks locally with a single CPU core of a computer.
        depends        : list
                         List of modules, default is an empty list. Use this only when cluster flag is set to True.

        """
        self.cluster = cluster
        self.compute = compute
        self.depends = depends
        self.server = server
        self.results = []
        self.jobs = []
        if self.cluster == True:
            self.job_cluster = dispy.JobCluster(
                self.compute, depends=self.depends)
        if self.server == True:
            self.server_thread = threading.Thread(target=start_server)
            self.server_thread.start()

    def submit(self, args):
        """
        Definition to submitting jobs.

        Parameters
        ----------
        args           : list
                         Variable arguments to pass to the compute.
        """
        if self.cluster == True:
            job = self.job_cluster.submit(*args)
            self.jobs.append(job)
        elif self.cluster == False:
            job = args
            self.jobs.append(job)

    def run(self):
        """
        Definition to run submitted jobs.

        Returns
        -------

        results       : list
                        Returns results from compute with a list.
        """
        print('Progress of the submitted jobs:')
        pbar = tqdm(total=len(self.jobs))
        self.results = []
        if self.cluster == False:
            for job in self.jobs:
                arguments = job
                result = self.compute(*arguments)
                self.results.append(result)
                pbar.update(1)
        if self.cluster == True:
            while len(self.jobs) > 0:
                for job in self.jobs:
                    if job.status == dispy.DispyJob.Finished:
                        result = job.result
                        self.results.append(result)
                        self.jobs.remove(job)
                        pbar.update(1)
            pbar.close()
            self.job_cluster.wait()
            self.job_cluster.print_status()
        return self.results

    def close(self):
        """
        Definition to close the cluster.
        """
        if self.cluster == True:
            self.job_cluster.close()
        return True
