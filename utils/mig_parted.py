"""
refernce: https://github.com/nvidia/mig-parted
"""
import subprocess


class MIGController:
    """
    In charge of mig partition and recovery
    """
    def __init__(self):
        pass

    def partition(self):
        """
        takes in an assigned partition(gpiID with mig partition),
        if success, returns a data structure contains the deviceIds of all created mig device.
        Return example:
            mig-devices:
                1g.5gb: 'MIG-29f07255-51c0-5691-82c4-cbc57760ff63'
                2g.10gb: 'MIG-ad654d5e-db91-50e1-bcf4-1e9c24f825ad'
                3g.20gb: 'MIG-ea08ed7b-a485-5967-9c78-2fa6e548c43a'
        """
        pass

    def recover(self):
        """
        stop all cuda processes on img devices and then delete all mig devices.
        """
        pass


def get_mig_devices(gpuID):
    mig_devices = []
    try:
        process = subprocess.Popen(['nvidia-smi', '-L'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        output, unused_err = process.communicate(timeout=10)
        output = output.decode("utf-8")
        find_gpu = False
        for line in output.splitlines():
            if line.strip().split(':')[0].split(' ')[0] == 'GPU':
                if line.strip().split(':')[0].split(' ')[1] == str(gpuID):
                    find_gpu = True
                else:
                    find_gpu = False
                    continue
            if find_gpu and line.strip().split(' ')[0] == 'MIG':
                mig_devices.append(
                    {
                        'mig_name': line.strip().split(' ')[1],
                        'uuid': line.strip().split(':')[-1].strip().split(')')[0],
                        'instance_id': line.strip().split(':')[0].split(' ')[-1]
                    }
                )
        return mig_devices

    except Exception as e:
      process.kill()
      print(e, e.__traceback__.tb_lineno, 'nvidia-smi failed')
