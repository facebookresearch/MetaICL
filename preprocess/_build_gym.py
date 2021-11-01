import argparse
import hashlib
import os
import subprocess

import multiprocessing as mp
from multiprocessing import Process, Manager

from _md5sum import MD5SUM
from _all_tasks import ALL_TASKS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="../data", type=str)
    parser.add_argument("--task_dir", default="./", type=str)
    parser.add_argument("--n_proc", default=1, type=int)
    parser.add_argument('--build', action='store_true',
                        help="Construct data from hg datasets.")
    parser.add_argument('--verify', action='store_true',
                        help="Verify the datafiles with pre-computed MD5")
    parser.add_argument('--debug', action='store_true',
                        help="Run 2 tasks per process to test the code")

    parser.add_argument('--inst', action='store_true',
                        help="Construct data from hg datasets.")
    parser.add_argument('--do_train', action='store_true',
                        help="Verify the datafiles with pre-computed MD5")
    parser.add_argument('--do_test', action='store_true',
                        help="Run 2 tasks per process to test the code")

    args = parser.parse_args()

    if args.do_train and args.do_test:
        raise NotImplementedError("You should specify one of `--do_train` and `--do_test`, not both")
    if not args.do_train and not args.do_test:
        raise NotImplementedError("You should specify one of `--do_train` and `--do_test`")

    return args

def process_tasks(idx, task_list, args, fail_dict):

    # debug mode, process 2 tasks in each process
    if args.debug:
        task_list = task_list[:2]

    print("Process {} is handling the following tasks: {}".format(idx, task_list))

    failed_tasks = []
    for task in task_list:
        print("Process {}: Processing {} ...".format(idx, task))
        command = "python %s%s%s%s" % (task, " --inst" if args.inst else ""," --do_train" if args.do_train else "", " --do_test" if args.do_test else "")
        ret_code = subprocess.run([command], shell=True) # stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
        if ret_code.returncode != 0:
            print("Process {}: Processing {} ... [Failed]".format(idx, task))
            failed_tasks.append(command)
        else:
            print("Process {}: Processing {} ... [Success]".format(idx, task))
    fail_dict[idx] = failed_tasks

def build_gym(args):
    successful = []
    failed = []
    all_tasks = []
    for filename in sorted(os.listdir(args.task_dir)):
        if filename.endswith(".py") and not filename.startswith("0") and not filename.startswith("_") and \
                filename!="utils.py" and "unifiedqa" not in filename:
            all_tasks.append(filename)

    assert all_tasks == ALL_TASKS
    print("Passing file checks ...")

    manager = Manager()
    fail_dict = manager.dict()

    if args.n_proc > 1:
        tasks_per_proc = int(len(all_tasks) / args.n_proc)
        tasks_split = [all_tasks[i * tasks_per_proc: (i+1) * tasks_per_proc] for i in range(args.n_proc - 1)]
        tasks_split.append(all_tasks[(args.n_proc-1) * tasks_per_proc:])

        processes = []
        for i in range(args.n_proc):
            p = mp.Process(target=process_tasks, args=(i+1, tasks_split[i], args, fail_dict))
            p.start()
            processes.append(p)

        for proc in processes:
            proc.join()

    else:
        process_tasks(0, all_tasks, args, fail_dict)

    all_failed_tasks = []
    for item in fail_dict.values():
        all_failed_tasks += item
    if len(all_failed_tasks) > 0:
        print("Please try the following tasks later by running {}".format(all_failed_tasks))
    else:
        print("Processing finished successfully.")

def get_md5(filename):
    # code from https://www.tecmint.com/generate-verify-check-files-md5-checksum-linux/
    md5_hash = hashlib.md5()
    a_file = open(filename, "rb")
    content = a_file.read()
    md5_hash.update(content)
    digest = md5_hash.hexdigest()
    return digest

def md5_verify(args):
    failed = []
    flags = {k: False for k in MD5SUM.keys()}

    subdirectories = sorted([x[0] for x in os.walk(args.output_dir)])
    for subdirectory in subdirectories[1:]:

        print("Verifying {}".format(subdirectory))

        files = os.listdir(subdirectory)
        for filename in files:
            if not filename.endswith(".tsv"):
                continue

            if filename not in MD5SUM:
                print("Unexpected file ``{}``".format(filename))
                continue

            md5sum = get_md5(os.path.join(subdirectory, filename))
            if md5sum != MD5SUM[filename]:
                print("{} is not consistent ...".format(filename))
                failed.append(filename)
            else:
                flags[filename] = True

    print("\n===== Verification Finished =====")
    if len(failed) == 0 and all(flags.values()):
        print("[Success] All files are consistent.")
    elif len(failed) != 0:
        print("[Failed] Some files are not consistent. \nPlease try re-running individual scripts for these tasks:\n{}".format(failed))
    else:
        print("[Failed] Some files are missing. \nPlease try re-running individual scripts for these tasks:")
        missing_files = [k for k, v in flags.items() if not v]
        print(missing_files)

def main():
    args = parse_args()
    if args.build:
        build_gym(args)
    if args.verify:
        md5_verify(args)

if __name__ == "__main__":
    main()
